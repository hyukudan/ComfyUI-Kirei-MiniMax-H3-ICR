from __future__ import annotations

from typing import Any

import torch

from .attention_profile_v2 import topology_digest
from .sparse_attention import (
    SPARSE_RUNTIME_KEY,
    SPARSE_WRAPPER_KEY,
    FlexSparseConfig,
    FlexSparseRuntime,
    _HEAD_DENSE,
    _HEAD_LOCAL_3D,
    _HEAD_SPATIAL,
    _HEAD_TEMPORAL,
    _architecture,
    _branch_name,
    _digest,
    _get_flex_api,
    _locate_native_h3,
    _make_mask_mod,
    _proposal_digest,
    parse_and_validate_policy,
)

_CLASS_TO_CODE_V2 = {
    "global_or_cross_modal": _HEAD_DENSE,
    "mixed_dense": _HEAD_DENSE,
    "local_3d_candidate": _HEAD_LOCAL_3D,
    "spatial_window_candidate": _HEAD_SPATIAL,
    "temporal_stripe_candidate": _HEAD_TEMPORAL,
    "local_3d_pair_candidate": _HEAD_LOCAL_3D,
    "spatial_pair_candidate": _HEAD_SPATIAL,
    "temporal_pair_candidate": _HEAD_TEMPORAL,
}


def policy_head_codes_v2(
    policy: dict[str, Any],
    layer: int,
    heads: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    layer_row = policy.get("layers", {}).get(str(layer))
    if not isinstance(layer_row, dict):
        return None
    rows = layer_row.get("heads")
    if not isinstance(rows, list) or len(rows) != heads:
        return None
    codes = [_HEAD_DENSE] * heads
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            return None
        head = int(row.get("head", -1))
        kind = row.get("classification")
        if head < 0 or head >= heads or head in seen or kind not in _CLASS_TO_CODE_V2:
            return None
        seen.add(head)
        codes[head] = _CLASS_TO_CODE_V2[kind]
    if len(seen) != heads:
        return None
    return torch.tensor(codes, dtype=torch.long, device=device)


def topology_matches_policy(policy: dict[str, Any], branch: str, layout: Any) -> bool:
    rows = policy.get("calibrated_topologies")
    if not isinstance(rows, dict):
        return False
    calibrated = rows.get(str(branch))
    if not isinstance(calibrated, dict):
        return False
    expected = calibrated.get("digest")
    if not isinstance(expected, str) or not expected:
        return False
    return topology_digest(layout) == expected


class FlexSparseRuntimeV2(FlexSparseRuntime):
    def __init__(self, policy: dict[str, Any], config: FlexSparseConfig, architecture_digest: str):
        super().__init__(policy, config, architecture_digest)
        self.topology_fallback_calls = 0

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["api"] = 2
        report["topology_bound"] = True
        report["calibrated_topologies"] = self.policy.get("calibrated_topologies", {})
        report["stats"]["dense_topology_fallback_calls"] = self.topology_fallback_calls
        return report


def _block_mask_v2(
    runtime: FlexSparseRuntimeV2,
    active,
    layer: int,
    heads: int,
    seq_len: int,
    device: torch.device,
):
    codes = policy_head_codes_v2(runtime.policy, layer, heads, device=device)
    if codes is None or bool((codes == _HEAD_DENSE).all().item()):
        return None
    cache_key = (
        active.branch,
        topology_digest(active.layout),
        layer,
        heads,
        seq_len,
        active.latent_t,
        active.latent_h,
        active.latent_w,
        str(device),
        runtime.config.block_size,
        tuple(int(v) for v in codes.detach().cpu().tolist()),
    )
    if cache_key in runtime._mask_cache:
        runtime.stats.mask_cache_hits += 1
        return runtime._mask_cache[cache_key]

    create_block_mask, _ = _get_flex_api()
    mask = create_block_mask(
        _make_mask_mod(active, codes, runtime.config),
        B=1,
        H=heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=runtime.config.block_size,
        _compile=False,
        separate_full_blocks=True,
    )
    runtime.stats.mask_builds += 1
    runtime.stats.last_block_sparsity = float(mask.sparsity())
    runtime.stats.max_block_sparsity = max(
        runtime.stats.max_block_sparsity,
        runtime.stats.last_block_sparsity,
    )
    runtime._mask_cache[cache_key] = mask
    return mask


def _flex_dispatch_v2(
    runtime: FlexSparseRuntimeV2,
    original,
    q,
    k,
    v,
    heads,
    *,
    active,
    layer: int,
    mask,
    skip_output_reshape: bool,
    kwargs: dict[str, Any],
):
    if q.device.type != "cuda":
        runtime.stats.dense_runtime_fallback_calls += 1
        return original(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
    if mask is not None:
        runtime.stats.dense_runtime_fallback_calls += 1
        return original(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
    if active.sigma <= runtime.config.dense_tail_sigma:
        runtime.stats.dense_tail_calls += 1
        return original(
            q,
            k,
            v,
            heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    block_mask = _block_mask_v2(runtime, active, layer, heads, int(q.shape[-2]), q.device)
    if block_mask is None:
        runtime.stats.dense_policy_fallback_calls += 1
        return original(
            q,
            k,
            v,
            heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
    if float(block_mask.sparsity()) < runtime.config.min_block_sparsity:
        runtime.stats.dense_policy_fallback_calls += 1
        return original(
            q,
            k,
            v,
            heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    _, flex_attention = _get_flex_api()
    kernel_options = {"ROWS_GUARANTEED_SAFE": True}
    if runtime.config.force_flex_kernel:
        kernel_options["FORCE_USE_FLEX_ATTENTION"] = True
    out = flex_attention(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    runtime.stats.sparse_calls += 1
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(q.shape[0], q.shape[-2], heads * q.shape[-1])
    return out


class FlexSparseOverrideV2:
    def __init__(self, runtime: FlexSparseRuntimeV2, previous_override: Any = None):
        self.runtime = runtime
        self.previous_override = previous_override

    def __call__(
        self,
        func,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        transformer_options=None,
        **kwargs,
    ):
        del attn_precision, transformer_options
        self.runtime.stats.attention_calls += 1
        layer_info = self.runtime.next_layer(int(q.shape[-2])) if skip_reshape and q.ndim == 4 else None
        if layer_info is None:
            if self.previous_override is not None:
                return self.previous_override(
                    func,
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            return func(
                q,
                k,
                v,
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        active, layer = layer_info

        def dispatch(q2, k2, v2, heads2, **inner_kwargs):
            inner_mask = inner_kwargs.pop("mask", mask)
            inner_skip_output = inner_kwargs.pop("skip_output_reshape", skip_output_reshape)
            inner_kwargs.pop("skip_reshape", None)
            inner_kwargs.pop("attn_precision", None)
            inner_kwargs.pop("transformer_options", None)
            return _flex_dispatch_v2(
                self.runtime,
                func,
                q2,
                k2,
                v2,
                heads2,
                active=active,
                layer=layer,
                mask=inner_mask,
                skip_output_reshape=inner_skip_output,
                kwargs=inner_kwargs,
            )

        if self.previous_override is not None:
            return self.previous_override(
                dispatch,
                q,
                k,
                v,
                heads,
                mask=mask,
                skip_reshape=True,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        return dispatch(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )


def sparse_diffusion_wrapper_v2(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = transformer_options or {}
    runtime = options.get(SPARSE_RUNTIME_KEY)
    if not isinstance(runtime, FlexSparseRuntimeV2) or not isinstance(x, (list, tuple)) or len(x) != 2:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    video_x, _audio_x = x
    payload = minimax_payload or {}
    layout = payload.get("layout")
    if layout is None:
        raise RuntimeError("Flex sparse backend requires native minimax_payload.layout")
    inner = executor.class_obj
    if type(inner).__module__ != "comfy.ldm.minimax.model" or type(inner).__name__ != "MiniMaxH3Model":
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    branch = _branch_name(options, video_x)
    if not topology_matches_policy(runtime.policy, branch, layout):
        runtime.topology_fallback_calls += 1
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    patch = tuple(int(v) for v in getattr(inner, "patch_size", (1, 2, 2)))
    sigma = float(timestep.flatten()[0].detach().float().cpu().item()) / 1000.0
    runtime.begin_call(
        layout=layout,
        sigma=sigma,
        latent_t=int(video_x.shape[-3]),
        latent_h=int(video_x.shape[-2]),
        latent_w=int(video_x.shape[-1]),
        patch_h=patch[-2],
        patch_w=patch[-1],
        branch=branch,
    )
    try:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    finally:
        runtime.end_call()


def parse_and_validate_policy_v2(
    policy_json: str,
    *,
    inner: Any,
    model_id: str,
    profile_json: str = "",
) -> tuple[dict[str, Any], str]:
    policy, architecture_digest = parse_and_validate_policy(
        policy_json,
        inner=inner,
        model_id=model_id,
        profile_json=profile_json,
    )
    topologies = policy.get("calibrated_topologies")
    if not isinstance(topologies, dict) or not topologies:
        raise ValueError(
            "Flex sparse v2 requires an M5 v2 policy with calibrated_topologies; "
            "run the current passive attention profiler first"
        )
    for branch, row in topologies.items():
        if not isinstance(branch, str) or not isinstance(row, dict):
            raise ValueError("calibrated_topologies contains an invalid branch entry")
        if not isinstance(row.get("digest"), str) or not row["digest"]:
            raise ValueError("calibrated topology is missing its digest")
    return policy, architecture_digest


def patch_flex_sparse_attention_v2(
    model: Any,
    *,
    policy_json: str,
    profile_json: str = "",
    model_id: str = "",
    dense_tail_sigma: float = 0.12,
    local_t_radius: int = 1,
    local_y_radius: int = 2,
    local_x_radius: int = 2,
    temporal_radius: int = 2,
    block_size: int = 128,
    min_block_sparsity: float = 5.0,
    force_flex_kernel: bool = True,
) -> tuple[Any, FlexSparseRuntimeV2]:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("Flex sparse backend expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    inner = _locate_native_h3(patched)
    policy, architecture_digest = parse_and_validate_policy_v2(
        policy_json,
        inner=inner,
        model_id=str(model_id).strip(),
        profile_json=profile_json,
    )
    config = FlexSparseConfig(
        dense_tail_sigma=float(dense_tail_sigma),
        local_t_radius=int(local_t_radius),
        local_y_radius=int(local_y_radius),
        local_x_radius=int(local_x_radius),
        temporal_radius=int(temporal_radius),
        block_size=int(block_size),
        min_block_sparsity=float(min_block_sparsity),
        force_flex_kernel=bool(force_flex_kernel),
        model_id=str(model_id).strip(),
    )
    runtime = FlexSparseRuntimeV2(policy, config, architecture_digest)

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    previous = transformer.get("optimized_attention_override")
    if previous is not None and hasattr(previous, "container_function"):
        raise RuntimeError("Flex sparse backend cannot chain a container-style optimized_attention_override")
    transformer["optimized_attention_override"] = FlexSparseOverrideV2(runtime, previous)
    transformer[SPARSE_RUNTIME_KEY] = runtime

    wrappers = dict(transformer.get("wrappers", {}))
    diffusion = dict(wrappers.get("diffusion_model", {}))
    diffusion.pop(SPARSE_WRAPPER_KEY, None)
    diffusion[SPARSE_WRAPPER_KEY] = [sparse_diffusion_wrapper_v2]
    wrappers["diffusion_model"] = diffusion
    transformer["wrappers"] = wrappers
    options["transformer_options"] = transformer
    patched.model_options = options
    return patched, runtime


def policy_fingerprint_v2(policy: dict[str, Any]) -> str:
    return _digest(
        {
            "proposal_digest": policy.get("proposal_digest", ""),
            "architecture_digest": policy.get("architecture_digest", ""),
            "calibrated_topologies": policy.get("calibrated_topologies", {}),
        }
    )
