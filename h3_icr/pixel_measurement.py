from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .contracts import make_av_container, unwrap_av


@dataclass(frozen=True, slots=True)
class PixelMeasurementConfig:
    strength: float = 0.05
    apply_every: int = 4
    max_correction_rms_ratio: float = 0.02
    measurement_max_side: int = 384
    frame_stride: int = 2
    edge_weight: float = 0.25
    temporal_weight: float = 0.10
    verify_after: bool = False
    allow_full_vae: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("pixel measurement strength must be in [0, 1]")
        if self.apply_every < 1:
            raise ValueError("pixel measurement apply_every must be positive")
        if not 0.0 < self.max_correction_rms_ratio <= 0.25:
            raise ValueError("pixel measurement RMS cap must be in (0, 0.25]")
        if self.measurement_max_side < 64:
            raise ValueError("measurement_max_side must be >= 64")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if self.edge_weight < 0.0 or self.temporal_weight < 0.0:
            raise ValueError("pixel measurement auxiliary weights must be non-negative")

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "strength": self.strength,
            "apply_every": self.apply_every,
            "max_correction_rms_ratio": self.max_correction_rms_ratio,
            "measurement_max_side": self.measurement_max_side,
            "frame_stride": self.frame_stride,
            "edge_weight": self.edge_weight,
            "temporal_weight": self.temporal_weight,
            "verify_after": self.verify_after,
            "allow_full_vae": self.allow_full_vae,
        }


@dataclass(slots=True)
class PixelMeasurementStats:
    calls: int = 0
    applied: int = 0
    verified_after: int = 0
    decoder_calls: int = 0
    pixel_rmse_sum: float = 0.0
    pixel_rmse_after_sum: float = 0.0
    edge_rmse_sum: float = 0.0
    temporal_rmse_sum: float = 0.0
    gradient_rms_sum: float = 0.0
    correction_ratio_sum: float = 0.0
    correction_ratio_max: float = 0.0

    def skipped(self) -> None:
        self.calls += 1

    def record(
        self,
        *,
        pixel_rmse: float,
        edge_rmse: float,
        temporal_rmse: float,
        gradient_rms: float,
        correction_ratio: float,
        pixel_rmse_after: float | None,
        decoder_calls: int,
    ) -> None:
        self.calls += 1
        self.applied += 1
        self.decoder_calls += int(decoder_calls)
        self.pixel_rmse_sum += float(pixel_rmse)
        self.edge_rmse_sum += float(edge_rmse)
        self.temporal_rmse_sum += float(temporal_rmse)
        self.gradient_rms_sum += float(gradient_rms)
        self.correction_ratio_sum += float(correction_ratio)
        self.correction_ratio_max = max(self.correction_ratio_max, float(correction_ratio))
        if pixel_rmse_after is not None:
            self.verified_after += 1
            self.pixel_rmse_after_sum += float(pixel_rmse_after)

    def to_dict(self) -> dict[str, float | int]:
        denom = max(1, self.applied)
        verified = max(1, self.verified_after)
        return {
            "calls": self.calls,
            "applied": self.applied,
            "verified_after": self.verified_after,
            "decoder_calls": self.decoder_calls,
            "pixel_rmse_mean": self.pixel_rmse_sum / denom,
            "pixel_rmse_after_mean": self.pixel_rmse_after_sum / verified if self.verified_after else 0.0,
            "edge_rmse_mean": self.edge_rmse_sum / denom,
            "temporal_rmse_mean": self.temporal_rmse_sum / denom,
            "gradient_rms_mean": self.gradient_rms_sum / denom,
            "correction_rms_ratio_mean": self.correction_ratio_sum / denom,
            "correction_rms_ratio_max": self.correction_ratio_max,
        }


def _is_full_h3_vae(first_stage_model: Any) -> bool:
    return (
        type(first_stage_model).__module__ == "comfy.ldm.minimax.vae"
        and type(first_stage_model).__name__ == "MiniMaxH3VideoVAE"
    )


def validate_h3_pixel_vae(vae: Any, *, allow_full_vae: bool) -> str:
    first = getattr(vae, "first_stage_model", None)
    if first is None or not callable(getattr(first, "decode", None)):
        raise TypeError("pixel measurement requires a ComfyUI VAE with a differentiable decoder")
    latent_channels = int(getattr(vae, "latent_channels", getattr(first, "latent_channels", 0)) or 0)
    is_h3_proxy = bool(getattr(first, "is_h3", False))
    is_full = _is_full_h3_vae(first)
    if latent_channels != 24 or not (is_h3_proxy or is_full):
        raise ValueError("pixel measurement requires a MiniMax H3-compatible 24-channel video VAE/TAE")
    if is_full and not allow_full_vae:
        raise ValueError(
            "full MiniMax H3 VisualVAE gradient is disabled by default; use the lightweight taeh3/TAE proxy "
            "or explicitly enable allow_full_vae for a high-cost experiment"
        )
    return "full_minimax_h3_vae" if is_full else "h3_tae_proxy"


def _area_downsample_latent(video: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("pixel measurement expects BxCxTxHxW video latents")
    if video.shape[-2:] == (target_h, target_w):
        return video
    b, c, t, h, w = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = F.interpolate(flat, size=(target_h, target_w), mode="area")
    return out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def _normalize_pixels(decoded: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(decoded) or decoded.ndim != 5:
        raise ValueError("H3 pixel decoder must return a 5D video tensor")
    # Native/TAE first-stage decoders return B,C,T,H,W. Keep a guarded
    # channel-last path for controlled test/proxy implementations.
    if decoded.shape[1] in (1, 3, 4):
        return decoded[:, :3]
    if decoded.shape[-1] in (1, 3, 4):
        return decoded[..., :3].permute(0, 4, 1, 2, 3).contiguous()
    raise ValueError("cannot resolve RGB channel axis from H3 pixel decoder output")


def _decoder_context(vae: Any):
    try:
        import comfy.model_management
    except Exception:
        return contextlib.nullcontext()
    device = getattr(vae, "device", None)
    if device is None:
        return contextlib.nullcontext()
    return comfy.model_management.cuda_device_context(device)


def decode_h3_pixels_differentiable(vae: Any, latent: torch.Tensor) -> torch.Tensor:
    first = vae.first_stage_model
    prepare = getattr(vae, "prepare_decode", None)
    if callable(prepare):
        prepare(tuple(latent.shape))
    device = getattr(vae, "device", latent.device)
    dtype = getattr(vae, "vae_dtype", latent.dtype)
    work = latent.to(device=device, dtype=dtype)
    with _decoder_context(vae):
        decoded = first.decode(work)
    decoded = _normalize_pixels(decoded)
    process_output = getattr(vae, "process_output", None)
    if callable(process_output):
        decoded = process_output(decoded)
    if not bool(torch.isfinite(decoded).all().item()):
        raise ValueError("H3 pixel decoder returned NaN/Inf")
    return decoded


def reduce_pixel_measurement(pixels: torch.Tensor, *, max_side: int, frame_stride: int) -> torch.Tensor:
    pixels = _normalize_pixels(pixels)
    pixels = pixels[:, :, ::frame_stride]
    h, w = pixels.shape[-2:]
    scale = min(1.0, float(max_side) / max(h, w))
    target_h = max(1, int(round(h * scale)))
    target_w = max(1, int(round(w * scale)))
    if (target_h, target_w) == (h, w):
        return pixels
    b, c, t, _, _ = pixels.shape
    flat = pixels.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = F.interpolate(flat, size=(target_h, target_w), mode="area")
    return out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def _measurement_losses(pred: torch.Tensor, reference: torch.Tensor, config: PixelMeasurementConfig):
    if pred.shape != reference.shape:
        raise ValueError(f"pixel measurement shape mismatch: {tuple(pred.shape)} vs {tuple(reference.shape)}")
    pred_f = pred.float()
    ref_f = reference.to(device=pred.device, dtype=torch.float32)
    residual = pred_f - ref_f
    pixel_mse = residual.square().mean()

    edge_terms = []
    if pred.shape[-1] > 1:
        edge_terms.append(((pred_f[..., 1:] - pred_f[..., :-1]) - (ref_f[..., 1:] - ref_f[..., :-1])).square().mean())
    if pred.shape[-2] > 1:
        edge_terms.append(((pred_f[..., 1:, :] - pred_f[..., :-1, :]) - (ref_f[..., 1:, :] - ref_f[..., :-1, :])).square().mean())
    edge_mse = sum(edge_terms) / len(edge_terms) if edge_terms else pixel_mse.new_zeros(())

    if pred.shape[2] > 1:
        temporal_mse = (
            (pred_f[:, :, 1:] - pred_f[:, :, :-1]) - (ref_f[:, :, 1:] - ref_f[:, :, :-1])
        ).square().mean()
    else:
        temporal_mse = pixel_mse.new_zeros(())

    total = pixel_mse + float(config.edge_weight) * edge_mse + float(config.temporal_weight) * temporal_mse
    return total, pixel_mse.sqrt(), edge_mse.sqrt(), temporal_mse.sqrt()


def build_reference_measurement(
    vae: Any,
    base_video: torch.Tensor,
    config: PixelMeasurementConfig,
) -> torch.Tensor:
    validate_h3_pixel_vae(vae, allow_full_vae=config.allow_full_vae)
    with torch.inference_mode():
        decoded = decode_h3_pixels_differentiable(vae, base_video.detach())
        reduced = reduce_pixel_measurement(
            decoded,
            max_side=config.measurement_max_side,
            frame_stride=config.frame_stride,
        )
    return reduced.detach().to(device="cpu", dtype=torch.float16)


def pixel_measurement_step(
    high_clean: torch.Tensor,
    low_reference_latent: torch.Tensor,
    reference_pixels: torch.Tensor,
    vae: Any,
    config: PixelMeasurementConfig,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if high_clean.ndim != 5 or low_reference_latent.ndim != 5:
        raise ValueError("pixel measurement expects BxCxTxHxW latent tensors")
    if high_clean.shape[:3] != low_reference_latent.shape[:3]:
        raise ValueError("pixel measurement B/C/T latent geometry differs")
    if config.strength == 0.0:
        return high_clean, {
            "pixel_rmse": 0.0,
            "edge_rmse": 0.0,
            "temporal_rmse": 0.0,
            "gradient_rms": 0.0,
            "correction_rms_ratio": 0.0,
            "pixel_rmse_after": 0.0,
            "verified_after": 0,
            "decoder_calls": 0,
        }

    validate_h3_pixel_vae(vae, allow_full_vae=config.allow_full_vae)
    with torch.inference_mode(False), torch.enable_grad():
        probe = high_clean.detach().float().requires_grad_(True)
        low_probe = _area_downsample_latent(
            probe,
            low_reference_latent.shape[-2],
            low_reference_latent.shape[-1],
        )
        decoded = decode_h3_pixels_differentiable(vae, low_probe)
        reduced = reduce_pixel_measurement(
            decoded,
            max_side=config.measurement_max_side,
            frame_stride=config.frame_stride,
        )
        loss, pixel_rmse, edge_rmse, temporal_rmse = _measurement_losses(reduced, reference_pixels, config)
        (gradient,) = torch.autograd.grad(loss, probe, create_graph=False, retain_graph=False)

    dims = tuple(range(1, high_clean.ndim))
    gradient_rms = gradient.detach().float().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)
    error_scale = pixel_rmse.detach().float().clamp_min(1e-8)
    correction = -float(config.strength) * gradient.detach().float() * (error_scale / gradient_rms)

    reference_rms = torch.maximum(
        high_clean.detach().float().square().mean(dim=dims, keepdim=True).sqrt(),
        low_reference_latent.detach().to(high_clean.device).float().square().mean(dim=dims, keepdim=True).sqrt(),
    ).clamp_min(1e-8)
    correction_rms = correction.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)
    clamp_scale = torch.clamp(
        reference_rms * float(config.max_correction_rms_ratio) / correction_rms,
        max=1.0,
    )
    bounded = correction * clamp_scale
    corrected = (high_clean.detach().float() + bounded).to(dtype=high_clean.dtype)
    ratio = float((bounded.square().mean(dim=dims).sqrt() / reference_rms.flatten()).mean().item())

    after_value: float | None = None
    decoder_calls = 1
    if config.verify_after:
        with torch.inference_mode():
            low_after = _area_downsample_latent(
                corrected.float(),
                low_reference_latent.shape[-2],
                low_reference_latent.shape[-1],
            )
            decoded_after = decode_h3_pixels_differentiable(vae, low_after)
            reduced_after = reduce_pixel_measurement(
                decoded_after,
                max_side=config.measurement_max_side,
                frame_stride=config.frame_stride,
            )
            _, pixel_after, _, _ = _measurement_losses(reduced_after, reference_pixels, config)
            after_value = float(pixel_after.item())
            decoder_calls += 1

    return corrected, {
        "pixel_rmse": float(pixel_rmse.detach().item()),
        "edge_rmse": float(edge_rmse.detach().item()),
        "temporal_rmse": float(temporal_rmse.detach().item()),
        "gradient_rms": float(gradient_rms.mean().item()),
        "correction_rms_ratio": ratio,
        "pixel_rmse_after": 0.0 if after_value is None else after_value,
        "verified_after": int(after_value is not None),
        "decoder_calls": decoder_calls,
    }


def patch_pixel_measurement_consistency(
    model: Any,
    base_video: torch.Tensor,
    vae: Any,
    *,
    strength: float = 0.05,
    apply_every: int = 4,
    max_correction_rms_ratio: float = 0.02,
    measurement_max_side: int = 384,
    frame_stride: int = 2,
    edge_weight: float = 0.25,
    temporal_weight: float = 0.10,
    verify_after: bool = False,
    allow_full_vae: bool = False,
) -> tuple[Any, PixelMeasurementConfig, PixelMeasurementStats, str]:
    clone = getattr(model, "clone", None)
    setter = getattr(model, "set_model_sampler_post_cfg_function", None)
    if not callable(clone) or not callable(setter):
        raise TypeError("pixel measurement consistency expects a ComfyUI MODEL/ModelPatcher")
    if not torch.is_tensor(base_video) or base_video.ndim != 5:
        raise TypeError("base_video must be the clean H3 Base Bx24xTxHxW latent")

    config = PixelMeasurementConfig(
        strength=float(strength),
        apply_every=int(apply_every),
        max_correction_rms_ratio=float(max_correction_rms_ratio),
        measurement_max_side=int(measurement_max_side),
        frame_stride=int(frame_stride),
        edge_weight=float(edge_weight),
        temporal_weight=float(temporal_weight),
        verify_after=bool(verify_after),
        allow_full_vae=bool(allow_full_vae),
    )
    decoder_kind = validate_h3_pixel_vae(vae, allow_full_vae=config.allow_full_vae)
    reference_pixels = build_reference_measurement(vae, base_video, config)
    stats = PixelMeasurementStats(decoder_calls=1)

    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")

    def apply_video(video: torch.Tensor):
        return pixel_measurement_step(video, base_video, reference_pixels, vae, config)

    def post_cfg(args):
        denoised = args["denoised"]
        should_apply = config.strength > 0.0 and (stats.calls % config.apply_every == 0)
        if not should_apply:
            stats.skipped()
            return denoised

        if (
            isinstance(denoised, (tuple, list))
            or getattr(denoised, "is_nested", False)
            or getattr(denoised, "tensors", None) is not None
        ):
            video, audio = unwrap_av(denoised)
            corrected, summary = apply_video(video)
            stats.record(
                pixel_rmse=float(summary["pixel_rmse"]),
                edge_rmse=float(summary["edge_rmse"]),
                temporal_rmse=float(summary["temporal_rmse"]),
                gradient_rms=float(summary["gradient_rms"]),
                correction_ratio=float(summary["correction_rms_ratio"]),
                pixel_rmse_after=float(summary["pixel_rmse_after"]) if summary["verified_after"] else None,
                decoder_calls=int(summary["decoder_calls"]),
            )
            return make_av_container(corrected, audio, template=denoised)

        if torch.is_tensor(denoised):
            base_model = args.get("model")
            latent_shapes = getattr(base_model, "latent_shapes", None)
            if not latent_shapes or len(latent_shapes) < 2:
                raise RuntimeError("pixel measurement cannot resolve packed H3 AV latent_shapes")
            import comfy.utils

            streams = comfy.utils.unpack_latents(denoised, latent_shapes)
            if len(streams) < 2:
                raise RuntimeError("pixel measurement expected packed H3 video+audio")
            corrected, summary = apply_video(streams[0])
            streams[0] = corrected
            packed, _ = comfy.utils.pack_latents(streams)
            stats.record(
                pixel_rmse=float(summary["pixel_rmse"]),
                edge_rmse=float(summary["edge_rmse"]),
                temporal_rmse=float(summary["temporal_rmse"]),
                gradient_rms=float(summary["gradient_rms"]),
                correction_ratio=float(summary["correction_rms_ratio"]),
                pixel_rmse_after=float(summary["pixel_rmse_after"]) if summary["verified_after"] else None,
                decoder_calls=int(summary["decoder_calls"]),
            )
            return packed

        raise TypeError(f"Unsupported H3 denoised representation: {type(denoised)!r}")

    patched.set_model_sampler_post_cfg_function(post_cfg)
    return patched, config, stats, decoder_kind
