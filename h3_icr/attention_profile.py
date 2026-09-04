from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch

PROFILE_RUNTIME_KEY = "h3_icr_attention_profile_runtime"
PROFILE_WRAPPER_KEY = "h3_icr_attention_profiler"

_MODALITY_MAP = {
    "text": "text",
    "cond": "visual_cond",
    "ref_img": "visual_cond",
    "cond_audio": "audio_cond",
    "ref_audio": "audio_cond",
    "audio": "target_audio",
    "video": "target_video",
}
_MODALITY_ORDER = ("text", "visual_cond", "audio_cond", "target_audio", "target_video")


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _even_indices(start: int, stop: int, count: int, *, device: torch.device) -> torch.Tensor:
    size = stop - start
    if size <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    n = min(size, count)
    if n == size:
        return torch.arange(start, stop, dtype=torch.long, device=device)
    return torch.linspace(float(start), float(stop - 1), steps=n, device=device).round().long().unique(sorted=True)


def _modality_indices(layout: Any, modality: str, count: int, *, device: torch.device) -> tuple[torch.Tensor, int]:
    spans = [
        (int(a), int(b))
        for a, b, kind in getattr(layout, "segments", ())
        if _MODALITY_MAP.get(kind) == modality and int(b) > int(a)
    ]
    population = sum(b - a for a, b in spans)
    if population == 0:
        return torch.empty(0, dtype=torch.long, device=device), 0
    target = min(population, count)
    if len(spans) == 1:
        return _even_indices(spans[0][0], spans[0][1], target, device=device), population

    all_rows = torch.cat([torch.arange(a, b, dtype=torch.long, device=device) for a, b in spans])
    selector = _even_indices(0, int(all_rows.numel()), target, device=device)
    return all_rows.index_select(0, selector), population


def _target_video_grid_coords(
    rows: torch.Tensor,
    *,
    video_start: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    gh, gw = latent_h // patch_h, latent_w // patch_w
    rows_per_frame = gh * gw
    local = rows - int(video_start)
    t = torch.div(local, rows_per_frame, rounding_mode="floor")
    spatial = local.remainder(rows_per_frame)
    y = torch.div(spatial, gw, rounding_mode="floor")
    x = spatial.remainder(gw)
    coords = torch.stack((t, y, x), dim=-1)
    if coords.numel() and (coords[:, 0].min() < 0 or coords[:, 0].max() >= latent_t):
        raise RuntimeError("profiled target-video row escaped the active H3 video grid")
    return coords


def _sum_lists(current: list[float] | None, values: list[float]) -> list[float]:
    if current is None:
        return list(values)
    if len(current) != len(values):
        raise RuntimeError("attention-profile head count changed inside one bucket")
    return [a + b for a, b in zip(current, values)]


@dataclass(slots=True)
class ProfileBucket:
    count: int = 0
    seq_len: int = 0
    heads: int = 0
    modality_rows: dict[str, int] = field(default_factory=dict)
    q_norm: dict[str, list[float]] = field(default_factory=dict)
    k_norm: dict[str, list[float]] = field(default_factory=dict)
    q_to_k_mass: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    video_structure: dict[str, list[float]] = field(default_factory=dict)

    def add(self, sample: dict[str, Any]) -> None:
        self.count += 1
        self.seq_len = int(sample["seq_len"])
        self.heads = int(sample["heads"])
        self.modality_rows = dict(sample["modality_rows"])
        for key, values in sample["q_norm"].items():
            self.q_norm[key] = _sum_lists(self.q_norm.get(key), values)
        for key, values in sample["k_norm"].items():
            self.k_norm[key] = _sum_lists(self.k_norm.get(key), values)
        for q_mod, masses in sample["q_to_k_mass"].items():
            target = self.q_to_k_mass.setdefault(q_mod, {})
            for k_mod, values in masses.items():
                target[k_mod] = _sum_lists(target.get(k_mod), values)
        for key, values in sample["video_structure"].items():
            self.video_structure[key] = _sum_lists(self.video_structure.get(key), values)

    def to_dict(self) -> dict[str, Any]:
        denom = max(1, self.count)
        avg = lambda values: [value / denom for value in values]
        return {
            "count": self.count,
            "seq_len": self.seq_len,
            "heads": self.heads,
            "modality_rows": self.modality_rows,
            "q_norm": {key: avg(values) for key, values in self.q_norm.items()},
            "k_norm": {key: avg(values) for key, values in self.k_norm.items()},
            "q_to_k_mass": {
                q: {k: avg(values) for k, values in masses.items()} for q, masses in self.q_to_k_mass.items()
            },
            "video_structure": {key: avg(values) for key, values in self.video_structure.items()},
        }


@dataclass(frozen=True, slots=True)
class AttentionProfileConfig:
    layer_stride: int = 5
    query_samples: int = 24
    key_samples_per_modality: int = 48
    sigma_decimals: int = 3
    max_buckets: int = 2048
    model_id: str = ""

    def __post_init__(self) -> None:
        if self.layer_stride < 1:
            raise ValueError("layer_stride must be >= 1")
        if self.query_samples < 1 or self.key_samples_per_modality < 1:
            raise ValueError("attention sampling counts must be positive")
        if not 0 <= self.sigma_decimals <= 6:
            raise ValueError("sigma_decimals must be between 0 and 6")
        if self.max_buckets < 1:
            raise ValueError("max_buckets must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_stride": self.layer_stride,
            "query_samples": self.query_samples,
            "key_samples_per_modality": self.key_samples_per_modality,
            "sigma_decimals": self.sigma_decimals,
            "max_buckets": self.max_buckets,
            "model_id": self.model_id,
        }


@dataclass(slots=True)
class _ActiveCall:
    layout: Any
    sigma: float
    branch: str
    latent_t: int
    latent_h: int
    latent_w: int
    patch_h: int
    patch_w: int
    attention_index: int = 0


class AttentionProfileRuntime:
    def __init__(self, config: AttentionProfileConfig, architecture: dict[str, Any]):
        self.config = config
        self.architecture = architecture
        self.architecture_digest = _canonical_digest(architecture)
        self._active: _ActiveCall | None = None
        self._buckets: dict[tuple[str, float, int], ProfileBucket] = {}
        self.model_calls = 0
        self.attention_calls_seen = 0
        self.attention_calls_profiled = 0
        self.skipped_non_h3_attention = 0
        self.dropped_new_buckets = 0

    def begin_call(self, *, layout: Any, sigma: float, branch: str, latent_t: int, latent_h: int, latent_w: int, patch_h: int, patch_w: int) -> None:
        if self._active is not None:
            raise RuntimeError("nested H3 attention-profile calls are not supported")
        self.model_calls += 1
        self._active = _ActiveCall(
            layout=layout,
            sigma=round(float(sigma), self.config.sigma_decimals),
            branch=branch,
            latent_t=int(latent_t),
            latent_h=int(latent_h),
            latent_w=int(latent_w),
            patch_h=int(patch_h),
            patch_w=int(patch_w),
        )

    def end_call(self) -> None:
        self._active = None

    def _sample_attention(self, q: torch.Tensor, k: torch.Tensor, heads: int) -> dict[str, Any]:
        active = self._active
        if active is None:
            raise RuntimeError("attention profile has no active H3 call")
        layout, device = active.layout, q.device

        q_norm: dict[str, list[float]] = {}
        k_norm: dict[str, list[float]] = {}
        modality_rows: dict[str, int] = {}
        key_groups: list[tuple[str, int, int, int, torch.Tensor]] = []
        key_parts: list[torch.Tensor] = []
        cursor = 0

        for modality in _MODALITY_ORDER:
            q_idx, population = _modality_indices(layout, modality, self.config.query_samples, device=device)
            k_idx, population_k = _modality_indices(layout, modality, self.config.key_samples_per_modality, device=device)
            population = max(population, population_k)
            if not population:
                continue
            modality_rows[modality] = population
            if q_idx.numel():
                q_norm[modality] = q[0].index_select(1, q_idx).float().norm(dim=-1).mean(dim=-1).cpu().tolist()
            if k_idx.numel():
                k_norm[modality] = k[0].index_select(1, k_idx).float().norm(dim=-1).mean(dim=-1).cpu().tolist()
                key_parts.append(k_idx)
                count = int(k_idx.numel())
                key_groups.append((modality, cursor, cursor + count, population, k_idx))
                cursor += count

        if not key_parts:
            raise RuntimeError("H3 attention profile could not sample any key rows")
        key_idx = torch.cat(key_parts)
        k_selected = k[0].index_select(1, key_idx).float()
        correction = torch.empty(key_idx.numel(), device=device, dtype=torch.float32)
        for _, start, stop, population, _ in key_groups:
            correction[start:stop] = math.log(max(1.0, population / max(1, stop - start)))

        q_to_k_mass: dict[str, dict[str, list[float]]] = {}
        video_structure: dict[str, list[float]] = {}
        scale = q.shape[-1] ** -0.5

        for q_modality in _MODALITY_ORDER:
            q_idx, _ = _modality_indices(layout, q_modality, self.config.query_samples, device=device)
            if not q_idx.numel():
                continue
            q_selected = q[0].index_select(1, q_idx).float()
            logits = torch.einsum("hqd,hkd->hqk", q_selected, k_selected) * scale + correction[None, None]
            probs = torch.softmax(logits, dim=-1)
            q_to_k_mass[q_modality] = {
                k_modality: probs[..., start:stop].sum(dim=-1).mean(dim=-1).cpu().tolist()
                for k_modality, start, stop, _, _ in key_groups
            }

            if q_modality != "target_video":
                continue
            video_group = next((group for group in key_groups if group[0] == "target_video"), None)
            if video_group is None:
                continue
            _, start, stop, _, video_k_idx = video_group
            va, _ = next((int(a), int(b)) for a, b, kind in layout.segments if kind == "video")
            q_coords = _target_video_grid_coords(
                q_idx,
                video_start=va,
                latent_t=active.latent_t,
                latent_h=active.latent_h,
                latent_w=active.latent_w,
                patch_h=active.patch_h,
                patch_w=active.patch_w,
            )
            k_coords = _target_video_grid_coords(
                video_k_idx,
                video_start=va,
                latent_t=active.latent_t,
                latent_h=active.latent_h,
                latent_w=active.latent_w,
                patch_h=active.patch_h,
                patch_w=active.patch_w,
            )
            delta = (q_coords[:, None] - k_coords[None]).abs()
            video_probs = probs[..., start:stop]
            masks = {
                "same_frame_mass": delta[..., 0] == 0,
                "spatial_r2_same_frame_mass": (delta[..., 0] == 0) & (delta[..., 1] <= 2) & (delta[..., 2] <= 2),
                "temporal_r1_same_spatial_mass": (delta[..., 0] <= 1) & (delta[..., 1] == 0) & (delta[..., 2] == 0),
                "local_3d_r1_mass": (delta[..., 0] <= 1) & (delta[..., 1] <= 1) & (delta[..., 2] <= 1),
            }
            for name, mask in masks.items():
                video_structure[name] = (
                    (video_probs * mask[None].to(video_probs.dtype)).sum(dim=-1).mean(dim=-1).cpu().tolist()
                )

        return {
            "seq_len": int(q.shape[-2]),
            "heads": int(heads),
            "modality_rows": modality_rows,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "q_to_k_mass": q_to_k_mass,
            "video_structure": video_structure,
        }

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
            bucket = self._buckets[key] = ProfileBucket()
        bucket.add(self._sample_attention(q, k, heads))
        self.attention_calls_profiled += 1

    def report(self) -> dict[str, Any]:
        buckets = []
        for (branch, sigma, layer), bucket in sorted(self._buckets.items(), key=lambda item: (item[0][0], -item[0][1], item[0][2])):
            row = bucket.to_dict()
            row.update({"branch": branch, "sigma": sigma, "layer": layer})
            buckets.append(row)
        result = {
            "api": 1,
            "config": self.config.to_dict(),
            "architecture": self.architecture,
            "architecture_digest": self.architecture_digest,
            "model_calls": self.model_calls,
            "attention_calls_seen": self.attention_calls_seen,
            "attention_calls_profiled": self.attention_calls_profiled,
            "skipped_non_h3_attention": self.skipped_non_h3_attention,
            "dropped_new_buckets": self.dropped_new_buckets,
            "buckets": buckets,
        }
        result["profile_digest"] = _canonical_digest(result)
        return result


class AttentionProfilerOverride:
    def __init__(self, runtime: AttentionProfileRuntime, previous_override: Any = None):
        self.runtime = runtime
        self.previous_override = previous_override

    def __call__(self, func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, transformer_options=None, **kwargs):
        if skip_reshape:
            self.runtime.observe(q, k, int(heads))
        call_kwargs = dict(kwargs)
        call_kwargs.update(
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            transformer_options=transformer_options,
        )
        if self.previous_override is not None:
            return self.previous_override(func, q, k, v, heads, **call_kwargs)
        return func(q, k, v, heads, **call_kwargs)


def _locate_native_h3(model: Any) -> Any:
    outer = getattr(model, "model", None)
    inner = getattr(outer, "diffusion_model", None) or getattr(model, "diffusion_model", None)
    if inner is None or type(inner).__module__ != "comfy.ldm.minimax.model" or type(inner).__name__ != "MiniMaxH3Model":
        actual = "missing" if inner is None else f"{type(inner).__module__}.{type(inner).__name__}"
        raise TypeError(f"attention profiler requires native MiniMaxH3Model; discovered {actual}")
    return inner


def _architecture(inner: Any, model_id: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "module": type(inner).__module__,
        "class": type(inner).__name__,
        "layers": len(getattr(inner, "blocks", ())),
        "hidden_size": int(getattr(inner, "hidden_size", 0)),
        "patch_size": tuple(int(v) for v in getattr(inner, "patch_size", ())),
        "video_channels": int(getattr(inner, "latents_dim", 0)),
        "audio_channels": int(getattr(inner, "audio_latents_dim", 0)),
        "adaln_curves": bool(getattr(inner, "use_adaln_curves", False)),
    }


def _branch_name(options: dict[str, Any], video_x: torch.Tensor) -> str:
    tiled = options.get("h3_icr_tiled_renderer")
    if tiled is None:
        return "dense"
    h, w = int(video_x.shape[-2]), int(video_x.shape[-1])
    if (h, w) == (int(getattr(tiled, "prior_h", -1)), int(getattr(tiled, "prior_w", -1))):
        return "m4_global_prior"
    return "m4_hr_tile"


def attention_profile_diffusion_wrapper(executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
    options = transformer_options or {}
    runtime = options.get(PROFILE_RUNTIME_KEY)
    if not isinstance(runtime, AttentionProfileRuntime) or not isinstance(x, (list, tuple)) or len(x) != 2:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    video_x, _audio_x = x
    payload = minimax_payload or {}
    layout = payload.get("layout")
    if layout is None:
        raise RuntimeError("H3 attention profiler requires minimax_payload.layout from native ComfyUI")
    inner = executor.class_obj
    if type(inner).__module__ != "comfy.ldm.minimax.model" or type(inner).__name__ != "MiniMaxH3Model":
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    patch = tuple(int(v) for v in getattr(inner, "patch_size", (1, 2, 2)))
    sigma = float(timestep.flatten()[0].detach().float().cpu().item()) / 1000.0
    runtime.begin_call(
        layout=layout,
        sigma=sigma,
        branch=_branch_name(options, video_x),
        latent_t=int(video_x.shape[-3]),
        latent_h=int(video_x.shape[-2]),
        latent_w=int(video_x.shape[-1]),
        patch_h=patch[-2],
        patch_w=patch[-1],
    )
    try:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    finally:
        runtime.end_call()


def patch_attention_profiler(model: Any, *, layer_stride: int = 5, query_samples: int = 24, key_samples_per_modality: int = 48, sigma_decimals: int = 3, max_buckets: int = 2048, model_id: str = "") -> tuple[Any, AttentionProfileRuntime]:
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
    runtime = AttentionProfileRuntime(config, _architecture(inner, config.model_id))

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    previous = transformer.get("optimized_attention_override")
    if previous is not None and hasattr(previous, "container_function"):
        raise RuntimeError("cannot safely chain an existing container-style optimized_attention_override")
    transformer["optimized_attention_override"] = AttentionProfilerOverride(runtime, previous)
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


def propose_attention_policy(report: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative measurement-derived proposal; never enable sparse execution."""
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for bucket in report.get("buckets", []):
        by_layer.setdefault(int(bucket["layer"]), []).append(bucket)

    layers: dict[str, Any] = {}
    for layer, buckets in sorted(by_layer.items()):
        heads = max((int(bucket.get("heads", 0)) for bucket in buckets), default=0)
        if heads <= 0:
            continue
        accum = {name: [0.0] * heads for name in ("local_3d_r1_mass", "spatial_r2_same_frame_mass", "temporal_r1_same_spatial_mass")}
        video_mass = [0.0] * heads
        samples = 0
        for bucket in buckets:
            structure = bucket.get("video_structure", {})
            masses = bucket.get("q_to_k_mass", {}).get("target_video", {})
            if not structure or "target_video" not in masses:
                continue
            samples += 1
            for name in accum:
                values = structure.get(name, [0.0] * heads)
                accum[name] = [a + float(b) for a, b in zip(accum[name], values)]
            video_mass = [a + float(b) for a, b in zip(video_mass, masses["target_video"])]
        if samples == 0:
            continue
        head_rows = []
        for head in range(heads):
            local = accum["local_3d_r1_mass"][head] / samples
            spatial = accum["spatial_r2_same_frame_mass"][head] / samples
            temporal = accum["temporal_r1_same_spatial_mass"][head] / samples
            self_mass = video_mass[head] / samples
            if self_mass < 0.55:
                kind = "global_or_cross_modal"
            elif local >= 0.55:
                kind = "local_3d_candidate"
            elif spatial >= 0.55:
                kind = "spatial_window_candidate"
            elif temporal >= 0.30:
                kind = "temporal_stripe_candidate"
            else:
                kind = "mixed_dense"
            head_rows.append(
                {
                    "head": head,
                    "classification": kind,
                    "target_video_mass": self_mass,
                    "local_3d_r1_mass": local,
                    "spatial_r2_same_frame_mass": spatial,
                    "temporal_r1_same_spatial_mass": temporal,
                }
            )
        layers[str(layer)] = {"samples": samples, "heads": head_rows}

    proposal = {
        "api": 1,
        "status": "proposal_only_no_sparse_kernel_enabled",
        "source_profile_digest": report.get("profile_digest", ""),
        "architecture_digest": report.get("architecture_digest", ""),
        "layers": layers,
        "late_dense_required": True,
        "dense_fallback_required": True,
    }
    proposal["proposal_digest"] = _canonical_digest(proposal)
    return proposal
