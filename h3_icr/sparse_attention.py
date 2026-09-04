from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch

SPARSE_RUNTIME_KEY = "h3_icr_sparse_runtime"
SPARSE_WRAPPER_KEY = "h3_icr_flex_sparse"

_HEAD_DENSE = 0
_HEAD_LOCAL_3D = 1
_HEAD_SPATIAL = 2
_HEAD_TEMPORAL = 3

_CLASS_TO_CODE = {
    "global_or_cross_modal": _HEAD_DENSE,
    "mixed_dense": _HEAD_DENSE,
    "local_3d_candidate": _HEAD_LOCAL_3D,
    "spatial_window_candidate": _HEAD_SPATIAL,
    "temporal_stripe_candidate": _HEAD_TEMPORAL,
}


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _proposal_digest(policy: dict[str, Any]) -> str:
    core = dict(policy)
    core.pop("proposal_digest", None)
    return _digest(core)


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


def _locate_native_h3(model: Any) -> Any:
    outer = getattr(model, "model", None)
    inner = getattr(outer, "diffusion_model", None) or getattr(model, "diffusion_model", None)
    if inner is None or type(inner).__module__ != "comfy.ldm.minimax.model" or type(inner).__name__ != "MiniMaxH3Model":
        actual = "missing" if inner is None else f"{type(inner).__module__}.{type(inner).__name__}"
        raise TypeError(f"Flex sparse backend requires native MiniMaxH3Model; discovered {actual}")
    return inner


def _video_segment(layout: Any) -> tuple[int, int]:
    segments = [(int(a), int(b)) for a, b, kind in getattr(layout, "segments", ()) if kind == "video"]
    if len(segments) != 1:
        raise RuntimeError("native H3 sparse backend expected exactly one target-video segment")
    return segments[0]


def _policy_head_codes(policy: dict[str, Any], layer: int, heads: int, *, device: torch.device) -> torch.Tensor | None:
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
        if head < 0 or head >= heads or head in seen or kind not in _CLASS_TO_CODE:
            return None
        seen.add(head)
        codes[head] = _CLASS_TO_CODE[kind]
    if len(seen) != heads:
        return None
    return torch.tensor(codes, dtype=torch.long, device=device)


@dataclass(frozen=True, slots=True)
class FlexSparseConfig:
    dense_tail_sigma: float = 0.12
    local_t_radius: int = 1
    local_y_radius: int = 2
    local_x_radius: int = 2
    temporal_radius: int = 2
    block_size: int = 128
    min_block_sparsity: float = 5.0
    force_flex_kernel: bool = True
    model_id: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.dense_tail_sigma < 1.0:
            raise ValueError("dense_tail_sigma must be in [0, 1)")
        if min(self.local_t_radius, self.local_y_radius, self.local_x_radius, self.temporal_radius) < 0:
            raise ValueError("sparse radii must be non-negative")
        if self.block_size not in {16, 32, 64, 128, 256}:
            raise ValueError("block_size must be one of 16, 32, 64, 128, 256")
        if not 0.0 <= self.min_block_sparsity <= 100.0:
            raise ValueError("min_block_sparsity must be a percentage in [0, 100]")


@dataclass(slots=True)
class SparseStats:
    attention_calls: int = 0
    sparse_calls: int = 0
    dense_tail_calls: int = 0
    dense_policy_fallback_calls: int = 0
    dense_runtime_fallback_calls: int = 0
    mask_builds: int = 0
    mask_cache_hits: int = 0
    last_block_sparsity: float = 0.0
    max_block_sparsity: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "attention_calls": self.attention_calls,
            "sparse_calls": self.sparse_calls,
            "dense_tail_calls": self.dense_tail_calls,
            "dense_policy_fallback_calls": self.dense_policy_fallback_calls,
            "dense_runtime_fallback_calls": self.dense_runtime_fallback_calls,
            "mask_builds": self.mask_builds,
            "mask_cache_hits": self.mask_cache_hits,
            "last_block_sparsity": self.last_block_sparsity,
            "max_block_sparsity": self.max_block_sparsity,
        }


@dataclass(slots=True)
class _ActiveCall:
    layout: Any
    sigma: float
    latent_t: int
    latent_h: int
    latent_w: int
    patch_h: int
    patch_w: int
    branch: str
    attention_index: int = 0


class FlexSparseRuntime:
    def __init__(self, policy: dict[str, Any], config: FlexSparseConfig, architecture_digest: str):
        self.policy = policy
        self.config = config
        self.architecture_digest = architecture_digest
        self.stats = SparseStats()
        self._active: _ActiveCall | None = None
        self._mask_cache: dict[tuple[Any, ...], Any] = {}

    def begin_call(
        self,
        *,
        layout: Any,
        sigma: float,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        patch_h: int,
        patch_w: int,
        branch: str,
    ) -> None:
        if self._active is not None:
            raise RuntimeError("nested H3 sparse-attention calls are not supported")
        self._active = _ActiveCall(
            layout=layout,
            sigma=float(sigma),
            latent_t=int(latent_t),
            latent_h=int(latent_h),
            latent_w=int(latent_w),
            patch_h=int(patch_h),
            patch_w=int(patch_w),
            branch=branch,
        )

    def end_call(self) -> None:
        self._active = None

    def next_layer(self, seq_len: int) -> tuple[_ActiveCall, int] | None:
        active = self._active
        if active is None or int(getattr(active.layout, "seq_len", -1)) != seq_len:
            return None
        layer = active.attention_index
        active.attention_index += 1
        return active, layer

    def report(self) -> dict[str, Any]:
        return {
            "api": 1,
            "backend": "torch_flex_attention_blockmask",
            "architecture_digest": self.architecture_digest,
            "source_profile_digest": self.policy.get("source_profile_digest", ""),
            "proposal_digest": self.policy.get("proposal_digest", ""),
            "config": {
                "dense_tail_sigma": self.config.dense_tail_sigma,
                "local_t_radius": self.config.local_t_radius,
                "local_y_radius": self.config.local_y_radius,
                "local_x_radius": self.config.local_x_radius,
                "temporal_radius": self.config.temporal_radius,
                "block_size": self.config.block_size,
                "min_block_sparsity": self.config.min_block_sparsity,
                "force_flex_kernel": self.config.force_flex_kernel,
                "model_id": self.config.model_id,
            },
            "stats": self.stats.to_dict(),
        }


def _branch_name(options: dict[str, Any], video_x: torch.Tensor) -> str:
    tiled = options.get("h3_icr_tiled_renderer")
    if tiled is None:
        return "dense"
    h, w = int(video_x.shape[-2]), int(video_x.shape[-1])
    if (h, w) == (int(getattr(tiled, "prior_h", -1)), int(getattr(tiled, "prior_w", -1))):
        return "m4_global_prior"
    return "m4_hr_tile"


def _mask_allowed_scalar(
    *,
    head_code: int,
    q_idx: int,
    kv_idx: int,
    video_start: int,
    video_stop: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    patch_h: int,
    patch_w: int,
    config: FlexSparseConfig,
) -> bool:
    if head_code == _HEAD_DENSE:
        return True
    q_video = video_start <= q_idx < video_stop
    k_video = video_start <= kv_idx < video_stop
    if not q_video or not k_video:
        return True
    gh, gw = latent_h // patch_h, latent_w // patch_w
    rows_per_frame = gh * gw
    q_local, k_local = q_idx - video_start, kv_idx - video_start
    qt, kt = q_local // rows_per_frame, k_local // rows_per_frame
    qs, ks = q_local % rows_per_frame, k_local % rows_per_frame
    qy, ky = qs // gw, ks // gw
    qx, kx = qs % gw, ks % gw
    if not (0 <= qt < latent_t and 0 <= kt < latent_t):
        return False
    dt, dy, dx = abs(qt - kt), abs(qy - ky), abs(qx - kx)
    if head_code == _HEAD_LOCAL_3D:
        return dt <= config.local_t_radius and dy <= config.local_y_radius and dx <= config.local_x_radius
    if head_code == _HEAD_SPATIAL:
        return dt == 0 and dy <= config.local_y_radius and dx <= config.local_x_radius
    if head_code == _HEAD_TEMPORAL:
        return dt <= config.temporal_radius and dy == 0 and dx == 0
    return True


def _make_mask_mod(active: _ActiveCall, head_codes: torch.Tensor, config: FlexSparseConfig):
    video_start, video_stop = _video_segment(active.layout)
    gh = active.latent_h // active.patch_h
    gw = active.latent_w // active.patch_w
    rows_per_frame = gh * gw
    local_t, local_y, local_x = config.local_t_radius, config.local_y_radius, config.local_x_radius
    temporal = config.temporal_radius

    def mask_mod(b, h, q_idx, kv_idx):
        del b
        code = head_codes[h]
        q_video = (q_idx >= video_start) & (q_idx < video_stop)
        k_video = (kv_idx >= video_start) & (kv_idx < video_stop)
        dense = code == _HEAD_DENSE
        non_video = (~q_video) | (~k_video)

        q_local = q_idx - video_start
        k_local = kv_idx - video_start
        qt = torch.div(q_local, rows_per_frame, rounding_mode="floor")
        kt = torch.div(k_local, rows_per_frame, rounding_mode="floor")
        qs = q_local.remainder(rows_per_frame)
        ks = k_local.remainder(rows_per_frame)
        qy = torch.div(qs, gw, rounding_mode="floor")
        ky = torch.div(ks, gw, rounding_mode="floor")
        qx = qs.remainder(gw)
        kx = ks.remainder(gw)
        dt, dy, dx = (qt - kt).abs(), (qy - ky).abs(), (qx - kx).abs()

        local3d = (dt <= local_t) & (dy <= local_y) & (dx <= local_x)
        spatial = (dt == 0) & (dy <= local_y) & (dx <= local_x)
        temporal_ok = (dt <= temporal) & (dy == 0) & (dx == 0)
        sparse_video = (
            ((code == _HEAD_LOCAL_3D) & local3d)
            | ((code == _HEAD_SPATIAL) & spatial)
            | ((code == _HEAD_TEMPORAL) & temporal_ok)
        )
        return dense | non_video | sparse_video

    return mask_mod


def _get_flex_api():
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("PyTorch FlexAttention/BlockMask is unavailable") from exc
    return create_block_mask, flex_attention


def _block_mask(runtime: FlexSparseRuntime, active: _ActiveCall, layer: int, heads: int, seq_len: int, device: torch.device):
    codes = _policy_head_codes(runtime.policy, layer, heads, device=device)
    if codes is None or bool((codes == _HEAD_DENSE).all().item()):
        return None
    cache_key = (
        active.branch,
        round(active.sigma, 4),
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
    runtime.stats.max_block_sparsity = max(runtime.stats.max_block_sparsity, runtime.stats.last_block_sparsity)
    runtime._mask_cache[cache_key] = mask
    return mask


def _flex_dispatch(
    runtime: FlexSparseRuntime,
    original,
    q,
    k,
    v,
    heads,
    *,
    active: _ActiveCall,
    layer: int,
    mask,
    skip_output_reshape: bool,
    kwargs: dict[str, Any],
):
    if q.device.type != "cuda":
        runtime.stats.dense_runtime_fallback_calls += 1
        return original(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)
    if mask is not None:
        runtime.stats.dense_runtime_fallback_calls += 1
        return original(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)
    if active.sigma <= runtime.config.dense_tail_sigma:
        runtime.stats.dense_tail_calls += 1
        return original(q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)

    block_mask = _block_mask(runtime, active, layer, heads, int(q.shape[-2]), q.device)
    if block_mask is None:
        runtime.stats.dense_policy_fallback_calls += 1
        return original(q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)
    if float(block_mask.sparsity()) < runtime.config.min_block_sparsity:
        runtime.stats.dense_policy_fallback_calls += 1
        return original(q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)

    _, flex_attention = _get_flex_api()
    kernel_options = {"ROWS_GUARANTEED_SAFE": True}
    if runtime.config.force_flex_kernel:
        kernel_options["FORCE_USE_FLEX_ATTENTION"] = True
    out = flex_attention(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    runtime.stats.sparse_calls += 1
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(q.shape[0], q.shape[-2], heads * q.shape[-1])
    return out


class FlexSparseOverride:
    def __init__(self, runtime: FlexSparseRuntime, previous_override: Any = None):
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
            return _flex_dispatch(
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
        return dispatch(q, k, v, heads, mask=mask, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs)


def sparse_diffusion_wrapper(executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
    options = transformer_options or {}
    runtime = options.get(SPARSE_RUNTIME_KEY)
    if not isinstance(runtime, FlexSparseRuntime) or not isinstance(x, (list, tuple)) or len(x) != 2:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    video_x, _audio_x = x
    payload = minimax_payload or {}
    layout = payload.get("layout")
    if layout is None:
        raise RuntimeError("Flex sparse backend requires native minimax_payload.layout")
    inner = executor.class_obj
    if type(inner).__module__ != "comfy.ldm.minimax.model" or type(inner).__name__ != "MiniMaxH3Model":
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
        branch=_branch_name(options, video_x),
    )
    try:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    finally:
        runtime.end_call()


def parse_and_validate_policy(
    policy_json: str,
    *,
    inner: Any,
    model_id: str,
    profile_json: str = "",
) -> tuple[dict[str, Any], str]:
    try:
        policy = json.loads(policy_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sparse policy JSON: {exc}") from exc
    if not isinstance(policy, dict) or policy.get("status") != "proposal_only_no_sparse_kernel_enabled":
        raise ValueError("sparse backend currently accepts only an M5 proposal JSON as its calibration source")
    if policy.get("proposal_digest") != _proposal_digest(policy):
        raise ValueError("sparse policy proposal_digest does not match its content")
    architecture_digest = _digest(_architecture(inner, model_id))
    if policy.get("architecture_digest") != architecture_digest:
        raise ValueError("sparse policy architecture fingerprint does not match this H3 MODEL/model_id")
    if profile_json.strip():
        try:
            profile = json.loads(profile_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid attention profile JSON: {exc}") from exc
        if not isinstance(profile, dict):
            raise ValueError("attention profile JSON must contain an object")
        expected = profile.get("profile_digest")
        core = dict(profile)
        core.pop("profile_digest", None)
        if expected != _digest(core):
            raise ValueError("attention profile digest does not match its content")
        if policy.get("source_profile_digest") != expected:
            raise ValueError("sparse policy was not derived from the supplied attention profile")
    return policy, architecture_digest


def patch_flex_sparse_attention(
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
) -> tuple[Any, FlexSparseRuntime]:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("Flex sparse backend expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    inner = _locate_native_h3(patched)
    policy, architecture_digest = parse_and_validate_policy(
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
    runtime = FlexSparseRuntime(policy, config, architecture_digest)

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    previous = transformer.get("optimized_attention_override")
    if previous is not None and hasattr(previous, "container_function"):
        raise RuntimeError("Flex sparse backend cannot chain a container-style optimized_attention_override")
    transformer["optimized_attention_override"] = FlexSparseOverride(runtime, previous)
    transformer[SPARSE_RUNTIME_KEY] = runtime
    wrappers = dict(transformer.get("wrappers", {}))
    diffusion = dict(wrappers.get("diffusion_model", {}))
    diffusion.pop(SPARSE_WRAPPER_KEY, None)
    diffusion[SPARSE_WRAPPER_KEY] = [sparse_diffusion_wrapper]
    wrappers["diffusion_model"] = diffusion
    transformer["wrappers"] = wrappers
    options["transformer_options"] = transformer
    patched.model_options = options
    return patched, runtime
