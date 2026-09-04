from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from .contracts import unwrap_av
from .fidelity import fourier_lowpass, resize_video
from .tiling import plan_spatial_tiles
from .validation import canonical_descriptor, fingerprint, validate_manifest_integrity

METRICS_API = 1
METRICS_SCHEMA = "h3_icr_latent_metrics/v1"
BUNDLE_API = 1
BUNDLE_SCHEMA = "h3_icr_validation_result/v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _rms(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().square().mean().sqrt().item())


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().abs().max().item())


def _spatial_gradient_rms(video: torch.Tensor) -> float:
    terms = []
    if video.shape[-1] > 1:
        terms.append(video[..., 1:] - video[..., :-1])
    if video.shape[-2] > 1:
        terms.append(video[..., 1:, :] - video[..., :-1, :])
    if not terms:
        return 0.0
    energy = sum(term.detach().float().square().mean() for term in terms) / len(terms)
    return float(energy.sqrt().item())


def _temporal_delta(video: torch.Tensor) -> torch.Tensor:
    if video.shape[2] < 2:
        return video[:, :, :0]
    return video[:, :, 1:] - video[:, :, :-1]


def _renderer_config(renderer: Any) -> Any | None:
    if renderer is None:
        return None
    if isinstance(renderer, dict):
        return renderer.get("config")
    return getattr(renderer, "config", None)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _boundary_rms_x(video: torch.Tensor, positions: list[int]) -> float:
    values = []
    for pos in positions:
        if 0 < pos < video.shape[-1]:
            values.append(video[..., pos] - video[..., pos - 1])
    if not values:
        return 0.0
    merged = torch.stack([value.detach().float().square().mean() for value in values])
    return float(merged.mean().sqrt().item())


def _boundary_rms_y(video: torch.Tensor, positions: list[int]) -> float:
    values = []
    for pos in positions:
        if 0 < pos < video.shape[-2]:
            values.append(video[..., pos, :] - video[..., pos - 1, :])
    if not values:
        return 0.0
    merged = torch.stack([value.detach().float().square().mean() for value in values])
    return float(merged.mean().sqrt().item())


def _seam_metrics(video: torch.Tensor, renderer: Any) -> dict[str, Any] | None:
    config = _renderer_config(renderer)
    if config is None:
        return None
    tile_h = int(_config_value(config, "tile_h", 0))
    tile_w = int(_config_value(config, "tile_w", 0))
    overlap_h = int(_config_value(config, "overlap_h", 0))
    overlap_w = int(_config_value(config, "overlap_w", 0))
    patch_h = int(_config_value(config, "patch_h", 2))
    patch_w = int(_config_value(config, "patch_w", 2))
    full_h, full_w = int(video.shape[-2]), int(video.shape[-1])
    if min(tile_h, tile_w) <= 0 or tile_h >= full_h and tile_w >= full_w:
        return {
            "active": False,
            "reason": "renderer_not_tiling_this_geometry",
        }
    try:
        plan = plan_spatial_tiles(
            full_h,
            full_w,
            tile_h=min(tile_h, full_h),
            tile_w=min(tile_w, full_w),
            overlap_h=min(overlap_h, max(0, min(tile_h, full_h) - patch_h)),
            overlap_w=min(overlap_w, max(0, min(tile_w, full_w) - patch_w)),
            patch_h=patch_h,
            patch_w=patch_w,
        )
    except Exception as exc:
        return {
            "active": False,
            "reason": f"plan_failed:{type(exc).__name__}",
        }

    x_positions = sorted({tile.x0 for tile in plan.tiles if tile.x0 > 0})
    y_positions = sorted({tile.y0 for tile in plan.tiles if tile.y0 > 0})
    dx = video[..., 1:] - video[..., :-1] if full_w > 1 else video[..., :0]
    dy = video[..., 1:, :] - video[..., :-1, :] if full_h > 1 else video[..., :0, :]
    global_x = _rms(dx)
    global_y = _rms(dy)
    seam_x = _boundary_rms_x(video, x_positions)
    seam_y = _boundary_rms_y(video, y_positions)
    eps = 1e-12
    return {
        "active": True,
        "tile_count": len(plan.tiles),
        "x_boundary_count": len(x_positions),
        "y_boundary_count": len(y_positions),
        "boundary_x_rms": seam_x,
        "boundary_y_rms": seam_y,
        "global_neighbor_x_rms": global_x,
        "global_neighbor_y_rms": global_y,
        "boundary_x_ratio": seam_x / max(global_x, eps) if x_positions else 0.0,
        "boundary_y_ratio": seam_y / max(global_y, eps) if y_positions else 0.0,
    }


def evaluate_latent_output(
    output_samples: Any,
    base_samples: Any,
    *,
    renderer: Any = None,
    lowpass_cutoff: float = 0.25,
) -> dict[str, Any]:
    if not 0.0 < float(lowpass_cutoff) <= 1.0:
        raise ValueError("validation lowpass_cutoff must be in (0, 1]")
    output_video, output_audio = unwrap_av(output_samples)
    base_video, base_audio = unwrap_av(base_samples)
    if output_video.ndim != 5 or base_video.ndim != 5:
        raise ValueError("validation metrics expect H3 BxCxTxHxW video latents")
    if output_video.shape[:3] != base_video.shape[:3]:
        raise ValueError("output/Base video B/C/T geometry differs")
    if not bool(torch.isfinite(output_video).all().item()) or not bool(torch.isfinite(base_video).all().item()):
        raise ValueError("validation video latent contains NaN/Inf")

    base_h, base_w = int(base_video.shape[-2]), int(base_video.shape[-1])
    out_h, out_w = int(output_video.shape[-2]), int(output_video.shape[-1])
    down = resize_video(output_video, base_h, base_w, mode="area")
    residual_lr = down - base_video.to(device=down.device, dtype=down.dtype)
    low_residual = fourier_lowpass(residual_lr, float(lowpass_cutoff))

    baseline_hr = resize_video(down, out_h, out_w, mode="bicubic")
    detail = output_video - baseline_hr
    output_delta_lr = _temporal_delta(down)
    base_delta = _temporal_delta(base_video.to(device=down.device, dtype=down.dtype))
    temporal_mismatch = output_delta_lr - base_delta
    detail_delta = _temporal_delta(detail)

    audio_shape_equal = tuple(output_audio.shape) == tuple(base_audio.shape)
    audio_exact = False
    audio_rmse = math.inf
    audio_max_abs = math.inf
    if audio_shape_equal:
        base_audio_on_output = base_audio.to(device=output_audio.device, dtype=output_audio.dtype)
        audio_diff = output_audio - base_audio_on_output
        audio_exact = bool(torch.equal(output_audio, base_audio_on_output))
        audio_rmse = _rms(audio_diff)
        audio_max_abs = _max_abs(audio_diff)

    metrics: dict[str, Any] = {
        "api": METRICS_API,
        "schema": METRICS_SCHEMA,
        "geometry": {
            "base_video_shape": [int(value) for value in base_video.shape],
            "output_video_shape": [int(value) for value in output_video.shape],
            "base_audio_shape": [int(value) for value in base_audio.shape],
            "output_audio_shape": [int(value) for value in output_audio.shape],
            "spatial_scale_h": out_h / base_h,
            "spatial_scale_w": out_w / base_w,
        },
        "base_compatibility": {
            "measurement_rmse": _rms(residual_lr),
            "low_frequency_rmse": _rms(low_residual),
            "temporal_delta_rmse": _rms(temporal_mismatch),
        },
        "detail": {
            "hr_residual_rms": _rms(detail),
            "hr_residual_spatial_gradient_rms": _spatial_gradient_rms(detail),
            "hr_residual_temporal_delta_rms": _rms(detail_delta),
            "output_spatial_gradient_rms": _spatial_gradient_rms(output_video),
        },
        "audio": {
            "shape_equal": audio_shape_equal,
            "exact": audio_exact,
            "rmse": audio_rmse,
            "max_abs": audio_max_abs,
        },
        "output_fingerprint": fingerprint(output_samples, strict=True, path="$.output_samples"),
    }
    seams = _seam_metrics(output_video, renderer)
    if seams is not None:
        metrics["m4_seams"] = seams
    metrics["metrics_id"] = _digest(metrics)
    return metrics


def validate_metrics_integrity(metrics: dict[str, Any]) -> None:
    if not isinstance(metrics, dict):
        raise TypeError("validation metrics must be a dictionary")
    if metrics.get("api") != METRICS_API or metrics.get("schema") != METRICS_SCHEMA:
        raise ValueError("unsupported H3 ICR latent metrics schema/API")
    expected = metrics.get("metrics_id")
    core = dict(metrics)
    core.pop("metrics_id", None)
    if expected != _digest(core):
        raise ValueError("validation metrics_id does not match its content")


def build_validation_result_bundle(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    *,
    reports: dict[str, Any] | None = None,
    notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_manifest_integrity(manifest)
    validate_metrics_integrity(metrics)
    reports_desc = canonical_descriptor(reports or {}, strict=True, path="$.reports")
    notes_desc = canonical_descriptor(notes or {}, strict=True, path="$.notes")
    bundle: dict[str, Any] = {
        "api": BUNDLE_API,
        "schema": BUNDLE_SCHEMA,
        "run_id": str(manifest["run_id"]),
        "metrics_id": str(metrics["metrics_id"]),
        "manifest": manifest,
        "metrics": metrics,
        "reports": reports_desc,
        "notes": notes_desc,
    }
    bundle["bundle_id"] = _digest(bundle)
    return bundle


def validate_bundle_integrity(bundle: dict[str, Any]) -> None:
    if not isinstance(bundle, dict):
        raise TypeError("validation result bundle must be a dictionary")
    if bundle.get("api") != BUNDLE_API or bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported H3 ICR validation result bundle schema/API")
    validate_manifest_integrity(bundle.get("manifest"))
    validate_metrics_integrity(bundle.get("metrics"))
    if bundle.get("run_id") != bundle["manifest"].get("run_id"):
        raise ValueError("validation bundle run_id differs from manifest")
    if bundle.get("metrics_id") != bundle["metrics"].get("metrics_id"):
        raise ValueError("validation bundle metrics_id differs from metrics")
    expected = bundle.get("bundle_id")
    core = dict(bundle)
    core.pop("bundle_id", None)
    if expected != _digest(core):
        raise ValueError("validation bundle_id does not match its content")
