from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .tiling import (
    TilingPlan,
    WeightedTileAccumulator,
    h3_video_token_count,
    plan_spatial_tiles,
    select_global_video_positions,
    tile_blend_weight,
)

TILED_CONFIG_KEY = "h3_icr_tiled_renderer"
TILED_STATS_KEY = "h3_icr_tiled_renderer_stats"
TILED_WRAPPER_KEY = "h3_icr_tiled_2k"
SPECTRUM_PREFIX = "spectrum_h3_"


@dataclass(frozen=True, slots=True)
class H3TiledRendererConfig:
    prior_h: int
    prior_w: int
    tile_h: int
    tile_w: int
    overlap_h: int
    overlap_w: int
    prior_strength: float = 0.30
    patch_h: int = 2
    patch_w: int = 2
    max_tiles: int = 16

    def __post_init__(self) -> None:
        if min(self.prior_h, self.prior_w, self.tile_h, self.tile_w) <= 0:
            raise ValueError("prior and tile geometry must be positive")
        if self.prior_strength < 0.0:
            raise ValueError("prior_strength must be non-negative")
        if self.max_tiles < 1:
            raise ValueError("max_tiles must be positive")
        for value, patch, name in (
            (self.prior_h, self.patch_h, "prior_h"),
            (self.prior_w, self.patch_w, "prior_w"),
            (self.tile_h, self.patch_h, "tile_h"),
            (self.tile_w, self.patch_w, "tile_w"),
            (self.overlap_h, self.patch_h, "overlap_h"),
            (self.overlap_w, self.patch_w, "overlap_w"),
        ):
            if value % patch:
                raise ValueError(f"{name} must align to the H3 patch grid")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "prior_h": self.prior_h,
            "prior_w": self.prior_w,
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
            "overlap_h": self.overlap_h,
            "overlap_w": self.overlap_w,
            "prior_strength": self.prior_strength,
            "max_tiles": self.max_tiles,
        }


@dataclass(slots=True)
class TiledRendererStats:
    calls: int = 0
    tiled_calls: int = 0
    dense_bypass_calls: int = 0
    prior_calls: int = 0
    tile_model_calls: int = 0
    spectrum_prior_calls: int = 0
    last_tile_count: int = 0
    max_tile_count: int = 0
    last_target_h: int = 0
    last_target_w: int = 0
    last_full_video_tokens: int = 0
    last_tile_video_tokens: int = 0
    last_keyframe_count: int = 0

    def record_plan(
        self,
        plan: TilingPlan,
        *,
        latent_t: int,
        spectrum_prior: bool,
        keyframe_count: int = 0,
    ) -> None:
        self.calls += 1
        self.tiled_calls += 1
        self.prior_calls += 1
        self.tile_model_calls += len(plan.tiles)
        self.last_tile_count = len(plan.tiles)
        self.max_tile_count = max(self.max_tile_count, len(plan.tiles))
        self.last_target_h = plan.latent_h
        self.last_target_w = plan.latent_w
        self.last_full_video_tokens = h3_video_token_count(latent_t, plan.latent_h, plan.latent_w)
        self.last_tile_video_tokens = h3_video_token_count(latent_t, plan.tile_h, plan.tile_w)
        self.last_keyframe_count = int(keyframe_count)
        if spectrum_prior:
            self.spectrum_prior_calls += 1

    def record_bypass(self, h: int, w: int) -> None:
        self.calls += 1
        self.dense_bypass_calls += 1
        self.last_target_h = h
        self.last_target_w = w

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "tiled_calls": self.tiled_calls,
            "dense_bypass_calls": self.dense_bypass_calls,
            "prior_calls": self.prior_calls,
            "tile_model_calls": self.tile_model_calls,
            "spectrum_prior_calls": self.spectrum_prior_calls,
            "last_tile_count": self.last_tile_count,
            "max_tile_count": self.max_tile_count,
            "last_target_h": self.last_target_h,
            "last_target_w": self.last_target_w,
            "last_full_video_tokens": self.last_full_video_tokens,
            "last_tile_video_tokens": self.last_tile_video_tokens,
            "last_keyframe_count": self.last_keyframe_count,
        }


def _resize_video_spatial(video: torch.Tensor, target_h: int, target_w: int, mode: str) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("H3 video must be BxCxTxHxW")
    b, c, t, h, w = video.shape
    if (h, w) == (target_h, target_w):
        return video
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    kwargs = {} if mode in {"area", "nearest"} else {"align_corners": False}
    resized = F.interpolate(flat, size=(target_h, target_w), mode=mode, **kwargs)
    return resized.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def _resize_mask_max(mask: torch.Tensor | None, target_h: int, target_w: int) -> torch.Tensor | None:
    if mask is None:
        return None
    if not torch.is_tensor(mask) or mask.ndim != 5:
        raise TypeError("H3 tiled denoise_mask must be a 5D tensor")
    if mask.shape[-2:] == (target_h, target_w):
        return mask
    return F.adaptive_max_pool3d(mask, (mask.shape[-3], target_h, target_w))


def _slice_mask(mask: torch.Tensor | None, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor | None:
    if mask is None:
        return None
    if not torch.is_tensor(mask) or mask.ndim != 5:
        raise TypeError("H3 tiled denoise_mask must be a 5D tensor")
    return mask[..., y0:y1, x0:x1]


def _native_module(inner: Any):
    module_name = type(inner).__module__
    if module_name != "comfy.ldm.minimax.model":
        raise TypeError(f"tiled H3 renderer needs native MiniMaxH3Model, got {module_name}.{type(inner).__name__}")
    module = importlib.import_module(module_name)
    if not hasattr(module, "PackedLayout"):
        raise RuntimeError("native MiniMax H3 module does not expose PackedLayout")
    return module


def _layout_signature_matches(layout: Any, expected: tuple[int, int, int, int, int]) -> bool:
    signature = tuple(getattr(layout, "signature", ()))
    return len(signature) >= 5 and signature[:5] == expected


def _make_layout(module, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int, payload):
    kwargs = {"keyframes": payload.get("keyframes"), "refs": payload.get("refs")}
    parameters = inspect.signature(module.PackedLayout).parameters
    if "frame_count" in parameters and payload.get("frame_count") is not None:
        kwargs["frame_count"] = payload.get("frame_count")
    return module.PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)


def _one_segment(layout: Any, kind: str) -> tuple[int, int]:
    matches = [(int(a), int(b)) for a, b, segment_kind in layout.segments if segment_kind == kind]
    if len(matches) != 1:
        raise RuntimeError(f"H3 tiled renderer expected exactly one {kind!r} segment")
    return matches[0]


def _video_keyframes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [kf for kf in (payload.get("keyframes") or ()) if kf.get("latent") is not None]


def _validate_full_keyframes(payload: dict[str, Any], full_h: int, full_w: int) -> None:
    for index, keyframe in enumerate(_video_keyframes(payload)):
        latent = keyframe.get("latent")
        if not torch.is_tensor(latent) or latent.ndim != 5:
            raise TypeError(f"minimax_keyframes[{index}] latent must be a 5D H3 video tensor")
        if latent.shape[0] != 1:
            raise ValueError("H3 tiled keyframes require batch size one")
        if tuple(latent.shape[-2:]) != (full_h, full_w):
            raise ValueError(
                "M4 tiled keyframes must use the full target latent geometry before tiling; "
                f"got {tuple(latent.shape[-2:])}, expected {(full_h, full_w)}"
            )


def _rebuild_cond_latents(payload: dict[str, Any]) -> None:
    keyframes = payload.get("keyframes") or ()
    refs = payload.get("refs") or ()
    payload["cond_video_latents"] = [
        item["latent"] for item in (*keyframes, *refs) if item.get("latent") is not None
    ]
    payload["cond_audio_latents"] = [
        item["audio_latent"] for item in (*keyframes, *refs) if item.get("audio_latent") is not None
    ]


def _payload_with_resized_keyframes(
    payload: dict[str, Any],
    target_h: int,
    target_w: int,
) -> dict[str, Any]:
    out = dict(payload)
    keyframes = []
    for keyframe in payload.get("keyframes") or ():
        item = dict(keyframe)
        latent = item.get("latent")
        if latent is not None:
            item["latent"] = _resize_video_spatial(latent, target_h, target_w, "area")
        keyframes.append(item)
    out["keyframes"] = keyframes
    _rebuild_cond_latents(out)
    out.pop("layout", None)
    return out


def _payload_with_tiled_keyframes(
    payload: dict[str, Any],
    tile,
) -> dict[str, Any]:
    out = dict(payload)
    keyframes = []
    for keyframe in payload.get("keyframes") or ():
        item = dict(keyframe)
        latent = item.get("latent")
        if latent is not None:
            item["latent"] = latent[..., tile.y0 : tile.y1, tile.x0 : tile.x1]
        keyframes.append(item)
    out["keyframes"] = keyframes
    _rebuild_cond_latents(out)
    out.pop("layout", None)
    return out


def rewrite_tile_layout_global_positions(
    full_layout: Any,
    tile_layout: Any,
    *,
    latent_t: int,
    full_h: int,
    full_w: int,
    tile,
    keyframes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    patch_h: int = 2,
    patch_w: int = 2,
) -> None:
    full_segments = list(full_layout.segments)
    tile_segments = list(tile_layout.segments)
    if [kind for _, _, kind in full_segments] != [kind for _, _, kind in tile_segments]:
        raise RuntimeError("full and tile H3 layouts have different segment kinds")

    full_va, full_vb = _one_segment(full_layout, "video")
    tile_va, tile_vb = _one_segment(tile_layout, "video")
    video_keyframes = [kf for kf in (keyframes or ()) if kf.get("latent") is not None]
    cond_index = 0

    for (fa, fb, kind), (ta, tb, tile_kind) in zip(full_segments, tile_segments):
        if kind != tile_kind:
            raise RuntimeError("H3 tile layout segment order changed unexpectedly")
        if kind == "video":
            continue
        if kind == "cond":
            if cond_index >= len(video_keyframes):
                raise RuntimeError("H3 layout exposes more visual keyframe segments than the payload")
            keyframe_t = int(video_keyframes[cond_index]["latent"].shape[2])
            selected = select_global_video_positions(
                full_layout.position_ids[fa:fb],
                latent_t=keyframe_t,
                full_h=full_h,
                full_w=full_w,
                tile=tile,
                patch_h=patch_h,
                patch_w=patch_w,
            )
            if selected.shape[0] != tb - ta:
                raise RuntimeError("global MM-RoPE selection does not match tiled keyframe rows")
            tile_layout.position_ids[ta:tb] = selected
            cond_index += 1
            continue
        if (fb - fa) != (tb - ta):
            raise RuntimeError(f"non-video H3 packed segment {kind!r} changed size during spatial tiling")
        tile_layout.position_ids[ta:tb] = full_layout.position_ids[fa:fb]

    if cond_index != len(video_keyframes):
        raise RuntimeError("H3 payload contains visual keyframes that are missing from the packed layout")

    selected = select_global_video_positions(
        full_layout.position_ids[full_va:full_vb],
        latent_t=latent_t,
        full_h=full_h,
        full_w=full_w,
        tile=tile,
        patch_h=patch_h,
        patch_w=patch_w,
    )
    if selected.shape[0] != tile_vb - tile_va:
        raise RuntimeError("global MM-RoPE row selection does not match tile video rows")
    tile_layout.position_ids[tile_va:tile_vb] = selected


def _child_options_without_spectrum(options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if not key.startswith(SPECTRUM_PREFIX)}


def _contains_easycache(transformer_options: dict[str, Any]) -> bool:
    if "easycache" in transformer_options:
        return True
    wrappers = transformer_options.get("wrappers", {})
    diffusion = wrappers.get("diffusion_model", {}) if isinstance(wrappers, dict) else {}
    return isinstance(diffusion, dict) and "easycache" in diffusion


def _resolve_full_layout(module, context, video_x, audio_x, payload):
    expected = (context.shape[1], video_x.shape[2], video_x.shape[3], video_x.shape[4], audio_x.shape[-1])
    layout = payload.get("layout")
    if layout is None or not _layout_signature_matches(layout, expected):
        layout = _make_layout(module, *expected, payload)
    return layout


def tiled_diffusion_model_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = transformer_options or {}
    config = options.get(TILED_CONFIG_KEY)
    stats = options.get(TILED_STATS_KEY)
    if not isinstance(config, H3TiledRendererConfig) or not isinstance(stats, TiledRendererStats):
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        raise TypeError("H3 tiled renderer expects [video, audio] model input")
    if _contains_easycache(options):
        raise RuntimeError("EasyCache is not yet topology-safe with H3 ICR spatial tiling")

    video_x, audio_x = x
    if video_x.ndim != 5 or audio_x.ndim != 4 or video_x.shape[0] != 1 or audio_x.shape[0] != 1:
        raise ValueError("H3 tiled renderer expects batch-one native H3 AV tensors")

    inner = executor.class_obj
    module = _native_module(inner)
    patch = tuple(getattr(inner, "patch_size", (1, 2, 2)))
    patch_h, patch_w = int(patch[-2]), int(patch[-1])
    full_h, full_w = int(video_x.shape[-2]), int(video_x.shape[-1])
    if full_h % patch_h or full_w % patch_w:
        raise ValueError("full H3 target latent must align to the DiT patch grid before tiling")

    if config.tile_h >= full_h and config.tile_w >= full_w:
        stats.record_bypass(full_h, full_w)
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    plan = plan_spatial_tiles(
        full_h,
        full_w,
        tile_h=min(config.tile_h, full_h),
        tile_w=min(config.tile_w, full_w),
        overlap_h=min(config.overlap_h, max(0, min(config.tile_h, full_h) - patch_h)),
        overlap_w=min(config.overlap_w, max(0, min(config.tile_w, full_w) - patch_w)),
        patch_h=patch_h,
        patch_w=patch_w,
    )
    if len(plan.tiles) > config.max_tiles:
        raise RuntimeError(f"H3 tiled plan needs {len(plan.tiles)} tiles, exceeding max_tiles={config.max_tiles}")

    payload = dict(minimax_payload or {})
    _validate_full_keyframes(payload, full_h, full_w)
    _rebuild_cond_latents(payload)
    full_layout = _resolve_full_layout(module, context, video_x, audio_x, payload)

    prior_video_x = _resize_video_spatial(video_x, config.prior_h, config.prior_w, "area")
    prior_payload = _payload_with_resized_keyframes(payload, config.prior_h, config.prior_w)
    prior_payload["layout"] = _make_layout(
        module,
        context.shape[1],
        prior_video_x.shape[2],
        prior_video_x.shape[3],
        prior_video_x.shape[4],
        audio_x.shape[-1],
        prior_payload,
    )
    prior_kwargs = dict(kwargs)
    if "denoise_mask" in prior_kwargs:
        prior_kwargs["denoise_mask"] = _resize_mask_max(
            prior_kwargs.get("denoise_mask"), config.prior_h, config.prior_w
        )
    prior_output = executor(
        [prior_video_x, audio_x],
        timestep,
        context,
        options,
        minimax_payload=prior_payload,
        **prior_kwargs,
    )
    if not isinstance(prior_output, (list, tuple)) or len(prior_output) != 2:
        raise TypeError("native H3 global prior call returned an unexpected output")
    prior_video_out, prior_audio_out = prior_output
    prior_hr = _resize_video_spatial(prior_video_out, full_h, full_w, "bilinear").to(torch.float32)

    accumulator = WeightedTileAccumulator(tuple(video_x.shape), device=video_x.device, dtype=torch.float32)
    tile_options = _child_options_without_spectrum(options)
    for tile in plan.tiles:
        tile_video_x = video_x[..., tile.y0 : tile.y1, tile.x0 : tile.x1]
        tile_payload = _payload_with_tiled_keyframes(payload, tile)
        tile_layout = _make_layout(
            module,
            context.shape[1],
            tile_video_x.shape[2],
            tile_video_x.shape[3],
            tile_video_x.shape[4],
            audio_x.shape[-1],
            tile_payload,
        )
        rewrite_tile_layout_global_positions(
            full_layout,
            tile_layout,
            latent_t=video_x.shape[2],
            full_h=full_h,
            full_w=full_w,
            tile=tile,
            keyframes=payload.get("keyframes"),
            patch_h=patch_h,
            patch_w=patch_w,
        )
        tile_payload["layout"] = tile_layout
        tile_kwargs = dict(kwargs)
        if "denoise_mask" in tile_kwargs:
            tile_kwargs["denoise_mask"] = _slice_mask(
                tile_kwargs.get("denoise_mask"), tile.y0, tile.y1, tile.x0, tile.x1
            )
        tile_output = executor(
            [tile_video_x, audio_x],
            timestep,
            context,
            tile_options,
            minimax_payload=tile_payload,
            **tile_kwargs,
        )
        if not isinstance(tile_output, (list, tuple)) or len(tile_output) != 2:
            raise TypeError("native H3 tile call returned an unexpected output")
        tile_video_out = tile_output[0].to(torch.float32)
        weight = tile_blend_weight(tile, device=video_x.device, dtype=torch.float32)
        accumulator.add(tile, tile_video_out, weight)

    fused_video = accumulator.finalize(prior=prior_hr, prior_strength=config.prior_strength)
    spectrum_prior = any(key.startswith(SPECTRUM_PREFIX) for key in options)
    stats.record_plan(
        plan,
        latent_t=int(video_x.shape[2]),
        spectrum_prior=spectrum_prior,
        keyframe_count=len(_video_keyframes(payload)),
    )
    return [fused_video.to(video_x.dtype), prior_audio_out]


def _install_outermost_diffusion_wrapper(transformer_options: dict[str, Any]) -> None:
    wrappers = dict(transformer_options.get("wrappers", {}))
    diffusion = dict(wrappers.get("diffusion_model", {}))
    if "easycache" in diffusion:
        raise RuntimeError("Apply the H3 tiled renderer without EasyCache; tile-local cache state is not implemented")
    diffusion.pop(TILED_WRAPPER_KEY, None)
    wrappers["diffusion_model"] = {TILED_WRAPPER_KEY: [tiled_diffusion_model_wrapper], **diffusion}
    transformer_options["wrappers"] = wrappers


def patch_tiled_renderer(
    model: Any,
    base_video: torch.Tensor,
    *,
    tile_h: int,
    tile_w: int,
    overlap_h: int,
    overlap_w: int,
    prior_strength: float,
    max_tiles: int = 16,
) -> tuple[Any, H3TiledRendererConfig, TiledRendererStats]:
    if not torch.is_tensor(base_video) or base_video.ndim != 5:
        raise TypeError("base_video must be the clean Bx24xTxHxW H3 Base latent")
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("H3 tiled renderer expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")

    config = H3TiledRendererConfig(
        prior_h=int(base_video.shape[-2]),
        prior_w=int(base_video.shape[-1]),
        tile_h=int(tile_h),
        tile_w=int(tile_w),
        overlap_h=int(overlap_h),
        overlap_w=int(overlap_w),
        prior_strength=float(prior_strength),
        max_tiles=int(max_tiles),
    )
    stats = TiledRendererStats()

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    transformer[TILED_CONFIG_KEY] = config
    transformer[TILED_STATS_KEY] = stats
    _install_outermost_diffusion_wrapper(transformer)
    options["transformer_options"] = transformer
    patched.model_options = options
    return patched, config, stats


def tiled_renderer_report(model: Any) -> dict[str, Any] | None:
    options = getattr(model, "model_options", None)
    if not isinstance(options, dict):
        return None
    transformer = options.get("transformer_options", {})
    if not isinstance(transformer, dict):
        return None
    config = transformer.get(TILED_CONFIG_KEY)
    stats = transformer.get(TILED_STATS_KEY)
    if not isinstance(config, H3TiledRendererConfig) or not isinstance(stats, TiledRendererStats):
        return None
    return {"config": config.to_dict(), "stats": stats.to_dict()}
