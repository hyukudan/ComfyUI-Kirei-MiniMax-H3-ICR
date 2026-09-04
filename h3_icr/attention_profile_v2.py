from __future__ import annotations

from typing import Any

import torch

from .attention_profile import (
    PROFILE_RUNTIME_KEY,
    PROFILE_WRAPPER_KEY,
    AttentionProfileConfig,
    AttentionProfileRuntime,
    AttentionProfilerOverride,
    _architecture,
    _canonical_digest,
    _locate_native_h3,
    _modality_indices,
    _target_video_grid_coords,
    attention_profile_diffusion_wrapper,
    propose_attention_policy,
)

PAIR_METRICS = (
    "diagonal_score",
    "spatial_neighbor_score",
    "temporal_neighbor_score",
    "far_video_score",
    "spatial_minus_far",
    "temporal_minus_far",
)


def _target_video_neighbor_rows(
    rows: torch.Tensor,
    *,
    video_start: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    patch_h: int,
    patch_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = _target_video_grid_coords(
        rows,
        video_start=video_start,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        patch_h=patch_h,
        patch_w=patch_w,
    )
    gh, gw = latent_h // patch_h, latent_w // patch_w
    frame_rows = gh * gw

    spatial: list[int] = []
    temporal: list[int] = []
    far: list[int] = []
    for t_value, y_value, x_value in coords.detach().cpu().tolist():
        t = int(t_value)
        y = int(y_value)
        x = int(x_value)

        if gw > 1:
            sx = x + 1 if x + 1 < gw else x - 1
            sy = y
        elif gh > 1:
            sx = x
            sy = y + 1 if y + 1 < gh else y - 1
        else:
            sx, sy = x, y

        if latent_t > 1:
            tt = t + 1 if t + 1 < latent_t else t - 1
        else:
            tt = t

        ft = (t + max(1, latent_t // 2)) % latent_t if latent_t > 1 else t
        fy = (y + max(1, gh // 2)) % gh if gh > 1 else y
        fx = (x + max(1, gw // 2)) % gw if gw > 1 else x

        spatial.append(video_start + t * frame_rows + sy * gw + sx)
        temporal.append(video_start + tt * frame_rows + y * gw + x)
        far.append(video_start + ft * frame_rows + fy * gw + fx)

    device = rows.device
    return (
        torch.tensor(spatial, dtype=torch.long, device=device),
        torch.tensor(temporal, dtype=torch.long, device=device),
        torch.tensor(far, dtype=torch.long, device=device),
    )


def _paired_score(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
) -> torch.Tensor:
    if q_rows.numel() == 0:
        return torch.zeros(q.shape[1], device=q.device, dtype=torch.float32)
    if q_rows.numel() != k_rows.numel():
        raise ValueError("paired Q/K calibration rows must have identical length")
    qs = q[0].index_select(1, q_rows).to(torch.float32)
    ks = k[0].index_select(1, k_rows).to(torch.float32)
    scale = q.shape[-1] ** -0.5
    return (qs * ks).sum(dim=-1).mul_(scale).mean(dim=-1)


def exact_video_pair_affinity(
    q: torch.Tensor,
    k: torch.Tensor,
    runtime: AttentionProfileRuntime,
) -> dict[str, list[float]]:
    active = runtime._active
    if active is None:
        return {}
    layout = active.layout
    q_rows, _ = _modality_indices(
        layout,
        "target_video",
        runtime.config.query_samples,
        device=q.device,
    )
    if q_rows.numel() == 0:
        return {}
    video_segments = [(int(a), int(b)) for a, b, kind in layout.segments if kind == "video"]
    if len(video_segments) != 1:
        return {}
    video_start, _ = video_segments[0]
    spatial_rows, temporal_rows, far_rows = _target_video_neighbor_rows(
        q_rows,
        video_start=video_start,
        latent_t=active.latent_t,
        latent_h=active.latent_h,
        latent_w=active.latent_w,
        patch_h=active.patch_h,
        patch_w=active.patch_w,
    )
    diagonal = _paired_score(q, k, q_rows, q_rows)
    spatial = _paired_score(q, k, q_rows, spatial_rows)
    temporal = _paired_score(q, k, q_rows, temporal_rows)
    far = _paired_score(q, k, q_rows, far_rows)
    return {
        "diagonal_score": diagonal.detach().cpu().tolist(),
        "spatial_neighbor_score": spatial.detach().cpu().tolist(),
        "temporal_neighbor_score": temporal.detach().cpu().tolist(),
        "far_video_score": far.detach().cpu().tolist(),
        "spatial_minus_far": (spatial - far).detach().cpu().tolist(),
        "temporal_minus_far": (temporal - far).detach().cpu().tolist(),
    }


class AttentionProfileRuntimeV2(AttentionProfileRuntime):
    def __init__(self, config: AttentionProfileConfig, architecture: dict[str, Any]):
        super().__init__(config, architecture)
        self._pair_sums: dict[tuple[str, float, int], dict[str, list[float]]] = {}
        self._pair_counts: dict[tuple[str, float, int], int] = {}

    @staticmethod
    def _sum(current: list[float] | None, values: list[float]) -> list[float]:
        if current is None:
            return list(values)
        if len(current) != len(values):
            raise RuntimeError("exact-pair head count changed inside one attention bucket")
        return [left + right for left, right in zip(current, values)]

    def observe(self, q: torch.Tensor, k: torch.Tensor, heads: int) -> None:
        active = self._active
        if active is None:
            return
        self.attention_calls_seen += 1
        if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
            self.skipped_non_h3_attention += 1
            return
        seq_len = int(getattr(active.layout, "seq_len", -1))
        if int(q.shape[-2]) != seq_len or int(k.shape[-2]) != seq_len:
            self.skipped_non_h3_attention += 1
            return

        layer = active.attention_index
        active.attention_index += 1
        if layer % self.config.layer_stride:
            return
        key = (active.branch, active.sigma, layer)
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.config.max_buckets:
                self.dropped_new_buckets += 1
                return
            from .attention_profile import ProfileBucket

            bucket = self._buckets[key] = ProfileBucket()

        bucket.add(self._sample_attention(q, k, heads))
        pair = exact_video_pair_affinity(q, k, self)
        if pair:
            target = self._pair_sums.setdefault(key, {})
            for name, values in pair.items():
                target[name] = self._sum(target.get(name), values)
            self._pair_counts[key] = self._pair_counts.get(key, 0) + 1
        self.attention_calls_profiled += 1

    def report(self) -> dict[str, Any]:
        report = super().report()
        for bucket in report.get("buckets", []):
            key = (str(bucket["branch"]), float(bucket["sigma"]), int(bucket["layer"]))
            count = self._pair_counts.get(key, 0)
            sums = self._pair_sums.get(key, {})
            bucket["video_pair_affinity"] = {
                name: [value / max(1, count) for value in values]
                for name, values in sums.items()
            }
            bucket["video_pair_samples"] = count
        report["exact_pair_metrics"] = list(PAIR_METRICS)
        report["profile_digest"] = _canonical_digest({k: v for k, v in report.items() if k != "profile_digest"})
        return report


class AttentionProfilerOverrideV2(AttentionProfilerOverride):
    pass


def patch_attention_profiler_v2(
    model: Any,
    *,
    layer_stride: int = 5,
    query_samples: int = 24,
    key_samples_per_modality: int = 48,
    sigma_decimals: int = 3,
    max_buckets: int = 2048,
    model_id: str = "",
) -> tuple[Any, AttentionProfileRuntimeV2]:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("H3 attention profiler expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    inner = _locate_native_h3(patched)
    config = AttentionProfileConfig(
        layer_stride=int(layer_stride),
        query_samples=int(query_samples),
        key_samples_per_modality=int(key_samples_per_modality),
        sigma_decimals=int(sigma_decimals),
        max_buckets=int(max_buckets),
        model_id=str(model_id).strip(),
    )
    runtime = AttentionProfileRuntimeV2(config, _architecture(inner, config.model_id))

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    previous = transformer.get("optimized_attention_override")
    if previous is not None and hasattr(previous, "container_function"):
        raise RuntimeError("cannot safely chain an existing container-style optimized_attention_override")
    transformer["optimized_attention_override"] = AttentionProfilerOverrideV2(runtime, previous)
    transformer[PROFILE_RUNTIME_KEY] = runtime

    wrappers = dict(transformer.get("wrappers", {}))
    diffusion = dict(wrappers.get("diffusion_model", {}))
    diffusion.pop(PROFILE_WRAPPER_KEY, None)
    diffusion[PROFILE_WRAPPER_KEY] = [attention_profile_diffusion_wrapper]
    wrappers["diffusion_model"] = diffusion
    transformer["wrappers"] = wrappers
    options["transformer_options"] = transformer
    patched.model_options = options
    return patched, runtime


def propose_attention_policy_v2(report: dict[str, Any]) -> dict[str, Any]:
    proposal = propose_attention_policy(report)
    buckets_by_layer: dict[int, list[dict[str, Any]]] = {}
    for bucket in report.get("buckets", []):
        buckets_by_layer.setdefault(int(bucket["layer"]), []).append(bucket)

    for layer_key, layer_data in proposal.get("layers", {}).items():
        layer = int(layer_key)
        buckets = buckets_by_layer.get(layer, [])
        for head_row in layer_data.get("heads", []):
            head = int(head_row["head"])
            spatial_values = []
            temporal_values = []
            for bucket in buckets:
                pair = bucket.get("video_pair_affinity", {})
                spatial = pair.get("spatial_minus_far", [])
                temporal = pair.get("temporal_minus_far", [])
                if head < len(spatial):
                    spatial_values.append(float(spatial[head]))
                if head < len(temporal):
                    temporal_values.append(float(temporal[head]))
            if not spatial_values and not temporal_values:
                continue
            spatial_margin = sum(spatial_values) / max(1, len(spatial_values))
            temporal_margin = sum(temporal_values) / max(1, len(temporal_values))
            head_row["exact_spatial_minus_far"] = spatial_margin
            head_row["exact_temporal_minus_far"] = temporal_margin
            head_row["classification_basis"] = "sampled_modality_mass_plus_exact_qk_pair_margin"

            self_mass = float(head_row.get("target_video_mass", 0.0))
            threshold = 0.05
            if self_mass < 0.55:
                classification = "global_or_cross_modal"
            elif spatial_margin > threshold and temporal_margin > threshold:
                classification = "local_3d_pair_candidate"
            elif spatial_margin > threshold:
                classification = "spatial_pair_candidate"
            elif temporal_margin > threshold:
                classification = "temporal_pair_candidate"
            else:
                classification = "mixed_dense"
            head_row["classification"] = classification

    proposal["exact_pair_evidence"] = True
    proposal["pair_margin_threshold"] = 0.05
    proposal["warning"] = (
        "Pair-margin thresholds are research heuristics used only to prioritize sparse-kernel experiments. "
        "They do not enable sparse execution and require decoded-media validation."
    )
    proposal["proposal_digest"] = _canonical_digest(
        {key: value for key, value in proposal.items() if key != "proposal_digest"}
    )
    return proposal
