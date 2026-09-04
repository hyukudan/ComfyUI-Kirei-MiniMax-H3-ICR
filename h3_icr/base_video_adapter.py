from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_API = 1
ADAPTER_RUNTIME_KEY = "h3_icr_base_video_adapter_runtime"
ADAPTER_WRAPPER_KEY = "h3_icr_base_video_adapter"


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_injection_blocks(value: str | tuple[int, ...] | list[int], *, layer_count: int) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        try:
            blocks = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError("adapter injection_blocks must be comma-separated integers") from exc
    else:
        blocks = [int(part) for part in value]
    if not blocks:
        raise ValueError("adapter injection_blocks cannot be empty")
    blocks = sorted(set(blocks))
    if blocks[0] < 0 or blocks[-1] >= layer_count:
        raise ValueError(f"adapter injection block outside native H3 layer range 0..{layer_count - 1}")
    return tuple(blocks)


def locate_native_h3(model: Any) -> Any:
    outer = getattr(model, "model", None)
    inner = getattr(outer, "diffusion_model", None) or getattr(model, "diffusion_model", None)
    if inner is None or type(inner).__module__ != "comfy.ldm.minimax.model" or type(inner).__name__ != "MiniMaxH3Model":
        actual = "missing" if inner is None else f"{type(inner).__module__}.{type(inner).__name__}"
        raise TypeError(f"M6 BaseVideo Adapter requires native MiniMaxH3Model; discovered {actual}")
    return inner


def h3_architecture_descriptor(inner: Any, model_id: str = "") -> dict[str, Any]:
    return {
        "model_id": str(model_id).strip(),
        "module": type(inner).__module__,
        "class": type(inner).__name__,
        "layers": len(getattr(inner, "blocks", ())),
        "hidden_size": int(getattr(inner, "hidden_size", 0)),
        "patch_size": tuple(int(v) for v in getattr(inner, "patch_size", ())),
        "video_channels": int(getattr(inner, "latents_dim", 0)),
        "audio_channels": int(getattr(inner, "audio_latents_dim", 0)),
        "adaln_curves": bool(getattr(inner, "use_adaln_curves", False)),
    }


def h3_architecture_digest(inner: Any, model_id: str = "") -> str:
    return _digest(h3_architecture_descriptor(inner, model_id))


@dataclass(frozen=True, slots=True)
class BaseVideoAdapterConfig:
    injection_blocks: tuple[int, ...]
    adapter_dim: int = 256
    gate_floor: float = 0.15
    gate_power: float = 1.0
    temporal_kernel: int = 3
    spatial_kernel: int = 3

    def __post_init__(self) -> None:
        if not self.injection_blocks:
            raise ValueError("adapter injection_blocks cannot be empty")
        if self.adapter_dim < 32 or self.adapter_dim > 2048:
            raise ValueError("adapter_dim must be in [32, 2048]")
        if not 0.0 <= self.gate_floor <= 1.0:
            raise ValueError("gate_floor must be in [0, 1]")
        if self.gate_power < 0.0:
            raise ValueError("gate_power must be non-negative")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if self.spatial_kernel < 1 or self.spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be a positive odd integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_blocks": list(self.injection_blocks),
            "adapter_dim": self.adapter_dim,
            "gate_floor": self.gate_floor,
            "gate_power": self.gate_power,
            "temporal_kernel": self.temporal_kernel,
            "spatial_kernel": self.spatial_kernel,
        }


class StateAwareBaseVideoAdapter(nn.Module):
    """Linear-cost local adapter over aligned Base and current H3 target rows.

    The final projection is zero-initialized. Therefore a newly-created module
    is an exact no-op even though the internal static/dynamic paths have normal
    initialization. A trained checkpoint is expected to learn the residual
    projection and the local feature mixer.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_channels: int,
        patch_size: tuple[int, int, int],
        config: BaseVideoAdapterConfig,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_channels = int(latent_channels)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.config = config
        patch_dim = self.latent_channels * math.prod(self.patch_size)
        dim = config.adapter_dim

        self.dynamic_norm = nn.LayerNorm(self.hidden_size)
        self.dynamic_proj = nn.Linear(self.hidden_size, dim)
        self.static_proj = nn.Linear(patch_dim, dim)
        self.fuse_proj = nn.Linear(dim * 3, dim)
        self.depthwise = nn.Conv3d(
            dim,
            dim,
            kernel_size=(config.temporal_kernel, config.spatial_kernel, config.spatial_kernel),
            padding=(config.temporal_kernel // 2, config.spatial_kernel // 2, config.spatial_kernel // 2),
            groups=dim,
        )
        self.pointwise = nn.Conv3d(dim, dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, self.hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def output_is_zero_initialized(self) -> bool:
        return bool(
            torch.count_nonzero(self.out_proj.weight.detach()).item() == 0
            and torch.count_nonzero(self.out_proj.bias.detach()).item() == 0
        )

    def forward(
        self,
        dynamic_hidden: torch.Tensor,
        static_patch_rows: torch.Tensor,
        *,
        latent_t: int,
        patch_grid_h: int,
        patch_grid_w: int,
        structure_gate: float,
    ) -> torch.Tensor:
        expected_rows = int(latent_t) * int(patch_grid_h) * int(patch_grid_w)
        if dynamic_hidden.ndim != 2 or dynamic_hidden.shape != (expected_rows, self.hidden_size):
            raise ValueError("adapter dynamic target-video rows do not match the active H3 patch grid")
        if static_patch_rows.ndim != 2 or static_patch_rows.shape[0] != expected_rows:
            raise ValueError("adapter static Base rows do not match the active H3 patch grid")

        dynamic = F.silu(self.dynamic_proj(self.dynamic_norm(dynamic_hidden)))
        static = F.silu(self.static_proj(static_patch_rows.to(dynamic.dtype)))
        gate = float(max(0.0, min(1.0, structure_gate)))
        anchor = static * gate + dynamic * (1.0 - gate)
        fused = F.silu(self.fuse_proj(torch.cat((dynamic, anchor, dynamic - anchor), dim=-1)))
        grid = fused.reshape(latent_t, patch_grid_h, patch_grid_w, -1).permute(3, 0, 1, 2).unsqueeze(0)
        local = grid + F.silu(self.pointwise(self.depthwise(grid)))
        rows = local.squeeze(0).permute(1, 2, 3, 0).reshape(expected_rows, -1)
        return self.out_proj(self.out_norm(rows))


@dataclass(slots=True)
class BaseVideoAdapterProvider:
    module: StateAwareBaseVideoAdapter
    config: BaseVideoAdapterConfig
    architecture_digest: str
    architecture: dict[str, Any]
    trained: bool = False
    checkpoint_sha256: str = ""
    note: str = "zero-init scaffold"
    api: int = ADAPTER_API

    def __post_init__(self) -> None:
        if self.api != ADAPTER_API:
            raise ValueError(f"unsupported BaseVideo Adapter provider API {self.api}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "config": self.config.to_dict(),
            "architecture_digest": self.architecture_digest,
            "architecture": self.architecture,
            "trained": bool(self.trained),
            "checkpoint_sha256": self.checkpoint_sha256,
            "note": self.note,
            "zero_output_projection": self.module.output_is_zero_initialized(),
        }


@dataclass(slots=True)
class BaseVideoAdapterStats:
    model_calls: int = 0
    block_calls: int = 0
    applied_blocks: int = 0
    zero_init_bypass_blocks: int = 0
    m4_tile_fallback_blocks: int = 0
    static_cache_hits: int = 0
    static_cache_builds: int = 0
    device_moves: int = 0
    residual_rms_sum: float = 0.0
    residual_rms_max: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        denom = max(1, self.applied_blocks)
        return {
            "model_calls": self.model_calls,
            "block_calls": self.block_calls,
            "applied_blocks": self.applied_blocks,
            "zero_init_bypass_blocks": self.zero_init_bypass_blocks,
            "m4_tile_fallback_blocks": self.m4_tile_fallback_blocks,
            "static_cache_hits": self.static_cache_hits,
            "static_cache_builds": self.static_cache_builds,
            "device_moves": self.device_moves,
            "residual_rms_mean": self.residual_rms_sum / denom,
            "residual_rms_max": self.residual_rms_max,
        }


@dataclass(slots=True)
class _ActiveAdapterCall:
    sigma: float
    sigma_start: float
    latent_t: int
    latent_h: int
    latent_w: int
    branch: str


class BaseVideoAdapterRuntime:
    def __init__(
        self,
        provider: BaseVideoAdapterProvider,
        base_video: torch.Tensor,
        *,
        strength: float = 1.0,
    ):
        if not torch.is_tensor(base_video) or base_video.ndim != 5:
            raise TypeError("BaseVideo Adapter requires the clean H3 Base Bx24xTxHxW latent")
        if base_video.shape[0] != 1 or base_video.shape[1] != provider.module.latent_channels:
            raise ValueError("BaseVideo Adapter Base latent batch/channels do not match the provider")
        if not 0.0 <= float(strength) <= 4.0:
            raise ValueError("BaseVideo Adapter strength must be in [0, 4]")
        self.provider = provider
        self.base_video = base_video.detach().to(device="cpu").contiguous()
        self.strength = float(strength)
        self.stats = BaseVideoAdapterStats()
        self._active: _ActiveAdapterCall | None = None
        self._static_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def begin_call(self, video_x: torch.Tensor, timestep: torch.Tensor, options: dict[str, Any]) -> None:
        if self._active is not None:
            raise RuntimeError("nested BaseVideo Adapter model calls are not supported")
        if video_x.ndim != 5 or video_x.shape[0] != 1:
            raise ValueError("BaseVideo Adapter expects batch-one H3 video tensors")
        sigma = float(timestep.flatten()[0].detach().float().cpu().item()) / 1000.0
        sample_sigmas = options.get("sample_sigmas")
        sigma_start = 1.0
        if torch.is_tensor(sample_sigmas) and sample_sigmas.numel():
            sigma_start = max(1e-8, float(sample_sigmas.flatten()[0].detach().float().cpu().item()))

        branch = "dense"
        tiled = options.get("h3_icr_tiled_renderer")
        if tiled is not None:
            h, w = int(video_x.shape[-2]), int(video_x.shape[-1])
            if (h, w) == (int(getattr(tiled, "prior_h", -1)), int(getattr(tiled, "prior_w", -1))):
                branch = "m4_global_prior"
            else:
                branch = "m4_hr_tile"

        self.stats.model_calls += 1
        self._active = _ActiveAdapterCall(
            sigma=sigma,
            sigma_start=sigma_start,
            latent_t=int(video_x.shape[-3]),
            latent_h=int(video_x.shape[-2]),
            latent_w=int(video_x.shape[-1]),
            branch=branch,
        )

    def end_call(self) -> None:
        self._active = None

    def clear_cache(self) -> None:
        self._static_cache.clear()

    def to(self, device_or_dtype: Any):
        self.provider.module.to(device_or_dtype)
        self.clear_cache()
        return self

    def _ensure_module_device(self, device: torch.device, dtype: torch.dtype) -> None:
        parameter = next(self.provider.module.parameters())
        if parameter.device != device or parameter.dtype != dtype:
            self.provider.module.to(device=device, dtype=dtype)
            self.clear_cache()
            self.stats.device_moves += 1

    @staticmethod
    def _resize_spatial(video: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if video.shape[-2:] == (target_h, target_w):
            return video
        b, c, t, h, w = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        out = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _patchify(video: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
        b, c, t_full, h_full, w_full = video.shape
        pt, ph, pw = patch_size
        if t_full % pt or h_full % ph or w_full % pw:
            raise ValueError("BaseVideo Adapter static latent is not aligned to H3 patch_size")
        t, h, w = t_full // pt, h_full // ph, w_full // pw
        rows = video.reshape(b, c, t, pt, h, ph, w, pw)
        rows = torch.einsum("nctrhpwq->nthwcrpq", rows)
        return rows.reshape(b * t * h * w, c * pt * ph * pw)

    def _static_rows(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        active = self._active
        if active is None:
            raise RuntimeError("BaseVideo Adapter has no active H3 call")
        # M4 HR tiles need an explicit global tile rectangle so the Base stream
        # can crop the matching region. Until that metadata contract lands, do
        # not feed the complete Base scene into every tile.
        if active.branch == "m4_hr_tile":
            return None
        if int(self.base_video.shape[2]) != active.latent_t:
            raise ValueError("BaseVideo Adapter Base/target temporal latent length differs")
        key = (active.latent_t, active.latent_h, active.latent_w, str(device), str(dtype))
        cached = self._static_cache.get(key)
        if cached is not None:
            self.stats.static_cache_hits += 1
            return cached
        base = self.base_video.to(device=device, dtype=dtype)
        resized = self._resize_spatial(base, active.latent_h, active.latent_w)
        rows = self._patchify(resized, self.provider.module.patch_size)
        self._static_cache[key] = rows
        self.stats.static_cache_builds += 1
        return rows

    @staticmethod
    def _video_segment(layout: Any) -> tuple[int, int]:
        matches = [(int(a), int(b)) for a, b, kind in getattr(layout, "segments", ()) if kind == "video"]
        if len(matches) != 1:
            raise RuntimeError("BaseVideo Adapter expected one native H3 target-video segment")
        return matches[0]

    def after_block(self, block_index: int, args: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
        self.stats.block_calls += 1
        if not self.provider.trained:
            self.stats.zero_init_bypass_blocks += 1
            return out
        active = self._active
        if active is None:
            raise RuntimeError("BaseVideo Adapter block executed outside an active H3 model call")
        if active.branch == "m4_hr_tile":
            self.stats.m4_tile_fallback_blocks += 1
            return out
        img = out.get("img")
        layout = args.get("layout")
        if not torch.is_tensor(img) or img.ndim != 2 or layout is None:
            raise RuntimeError("BaseVideo Adapter received an incompatible native H3 block contract")
        va, vb = self._video_segment(layout)
        dynamic = img[va:vb]
        self._ensure_module_device(dynamic.device, dynamic.dtype)
        static_rows = self._static_rows(dynamic.device, dynamic.dtype)
        if static_rows is None:
            self.stats.m4_tile_fallback_blocks += 1
            return out
        pt, ph, pw = self.provider.module.patch_size
        grid_t = active.latent_t // pt
        grid_h = active.latent_h // ph
        grid_w = active.latent_w // pw
        ratio = max(0.0, min(1.0, active.sigma / max(active.sigma_start, 1e-8)))
        gate = self.provider.config.gate_floor + (1.0 - self.provider.config.gate_floor) * (
            ratio ** self.provider.config.gate_power
        )
        residual = self.provider.module(
            dynamic,
            static_rows,
            latent_t=grid_t,
            patch_grid_h=grid_h,
            patch_grid_w=grid_w,
            structure_gate=gate,
        )
        residual = residual * self.strength
        rms = float(residual.float().square().mean().sqrt().item())
        self.stats.applied_blocks += 1
        self.stats.residual_rms_sum += rms
        self.stats.residual_rms_max = max(self.stats.residual_rms_max, rms)
        if rms != 0.0:
            img[va:vb].add_(residual)
        return out

    def report(self) -> dict[str, Any]:
        return {
            "api": ADAPTER_API,
            "provider": self.provider.to_dict(),
            "strength": self.strength,
            "m4_hr_tile_support": False,
            "stats": self.stats.to_dict(),
        }


class BaseVideoAdapterBlockPatch:
    def __init__(self, runtime: BaseVideoAdapterRuntime, block_index: int, previous: Any = None):
        self.runtime = runtime
        self.block_index = int(block_index)
        self.previous = previous

    def __call__(self, args, extra_args):
        if self.previous is None:
            out = extra_args["original_block"](args)
        else:
            out = self.previous(args, extra_args)
        return self.runtime.after_block(self.block_index, args, out)

    def to(self, device_or_dtype):
        self.runtime.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.runtime.clear_cache()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        # Zero-init scaffold has no separately-managed checkpoint yet. A trained
        # provider loader must wrap adapter weights in a ComfyUI ModelPatcher and
        # expose it here before trained checkpoints are considered production-safe.
        if hasattr(self.previous, "models"):
            return self.previous.models()
        return []


def base_video_adapter_diffusion_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = transformer_options or {}
    runtime = options.get(ADAPTER_RUNTIME_KEY)
    if not isinstance(runtime, BaseVideoAdapterRuntime) or not isinstance(x, (list, tuple)) or len(x) != 2:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    video_x = x[0]
    runtime.begin_call(video_x, timestep, options)
    try:
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    finally:
        runtime.end_call()


def create_zero_init_base_adapter_provider(
    model: Any,
    *,
    injection_blocks: str | tuple[int, ...] | list[int] = "12,24,36,45,48",
    adapter_dim: int = 256,
    gate_floor: float = 0.15,
    gate_power: float = 1.0,
    temporal_kernel: int = 3,
    spatial_kernel: int = 3,
    model_id: str = "",
) -> BaseVideoAdapterProvider:
    inner = locate_native_h3(model)
    architecture = h3_architecture_descriptor(inner, model_id)
    layer_count = int(architecture["layers"])
    blocks = parse_injection_blocks(injection_blocks, layer_count=layer_count)
    config = BaseVideoAdapterConfig(
        injection_blocks=blocks,
        adapter_dim=int(adapter_dim),
        gate_floor=float(gate_floor),
        gate_power=float(gate_power),
        temporal_kernel=int(temporal_kernel),
        spatial_kernel=int(spatial_kernel),
    )
    if architecture["video_channels"] != 24 or tuple(architecture["patch_size"]) != (1, 2, 2):
        raise ValueError("M6 scaffold currently targets the native H3 24-channel, 1x2x2 video-patch contract")
    module = StateAwareBaseVideoAdapter(
        hidden_size=int(architecture["hidden_size"]),
        latent_channels=int(architecture["video_channels"]),
        patch_size=tuple(architecture["patch_size"]),
        config=config,
    )
    return BaseVideoAdapterProvider(
        module=module,
        config=config,
        architecture_digest=_digest(architecture),
        architecture=architecture,
        trained=False,
        note="zero-init scaffold; output projection is zero and no trained adapter weights are loaded",
    )


def patch_base_video_adapter(
    model: Any,
    base_video: torch.Tensor,
    provider: BaseVideoAdapterProvider,
    *,
    strength: float = 1.0,
) -> tuple[Any, BaseVideoAdapterRuntime]:
    if not isinstance(provider, BaseVideoAdapterProvider) or provider.api != ADAPTER_API:
        raise TypeError("invalid H3 ICR BaseVideo Adapter provider")
    inner = locate_native_h3(model)
    model_id = str(provider.architecture.get("model_id", ""))
    actual_digest = h3_architecture_digest(inner, model_id)
    if actual_digest != provider.architecture_digest:
        raise ValueError("BaseVideo Adapter architecture fingerprint does not match this native H3 MODEL")
    layer_count = len(getattr(inner, "blocks", ()))
    for block in provider.config.injection_blocks:
        if block < 0 or block >= layer_count:
            raise ValueError("BaseVideo Adapter injection block is incompatible with this H3 MODEL")

    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("BaseVideo Adapter expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    runtime = BaseVideoAdapterRuntime(provider, base_video, strength=strength)

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    transformer[ADAPTER_RUNTIME_KEY] = runtime
    options["transformer_options"] = transformer
    patched.model_options = options

    add_wrapper = getattr(patched, "add_wrapper_with_key", None)
    if not callable(add_wrapper):
        raise TypeError("ComfyUI MODEL does not expose add_wrapper_with_key required by M6")
    add_wrapper("diffusion_model", ADAPTER_WRAPPER_KEY, base_video_adapter_diffusion_wrapper)

    for block_index in provider.config.injection_blocks:
        blocks_replace = (
            patched.model_options.get("transformer_options", {})
            .get("patches_replace", {})
            .get("dit", {})
        )
        previous = blocks_replace.get(("double_block", int(block_index)))
        setter = getattr(patched, "set_model_patch_replace", None)
        if not callable(setter):
            raise TypeError("ComfyUI MODEL does not expose set_model_patch_replace required by M6")
        setter(
            BaseVideoAdapterBlockPatch(runtime, int(block_index), previous),
            "dit",
            "double_block",
            int(block_index),
        )
    return patched, runtime
