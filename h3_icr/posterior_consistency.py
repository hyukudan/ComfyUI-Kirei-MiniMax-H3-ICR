from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .contracts import make_av_container, unwrap_av


@dataclass(frozen=True, slots=True)
class PosteriorConsistencyConfig:
    strength: float = 0.10
    apply_every: int = 2
    max_correction_rms_ratio: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError("posterior-consistency strength must be in [0, 2]")
        if self.apply_every < 1:
            raise ValueError("posterior-consistency apply_every must be positive")
        if not 0.0 < self.max_correction_rms_ratio <= 1.0:
            raise ValueError("posterior-consistency RMS cap must be in (0, 1]")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "strength": self.strength,
            "apply_every": self.apply_every,
            "max_correction_rms_ratio": self.max_correction_rms_ratio,
        }


@dataclass(slots=True)
class PosteriorConsistencyStats:
    calls: int = 0
    applied: int = 0
    error_before_sum: float = 0.0
    error_after_sum: float = 0.0
    correction_ratio_sum: float = 0.0
    correction_ratio_max: float = 0.0
    gradient_rms_sum: float = 0.0

    def record(
        self,
        *,
        applied: bool,
        error_before: float = 0.0,
        error_after: float = 0.0,
        correction_ratio: float = 0.0,
        gradient_rms: float = 0.0,
    ) -> None:
        self.calls += 1
        if not applied:
            return
        self.applied += 1
        self.error_before_sum += float(error_before)
        self.error_after_sum += float(error_after)
        self.correction_ratio_sum += float(correction_ratio)
        self.correction_ratio_max = max(self.correction_ratio_max, float(correction_ratio))
        self.gradient_rms_sum += float(gradient_rms)

    def to_dict(self) -> dict[str, float | int]:
        denom = max(1, self.applied)
        return {
            "calls": self.calls,
            "applied": self.applied,
            "measurement_error_before_mean": self.error_before_sum / denom,
            "measurement_error_after_mean": self.error_after_sum / denom,
            "correction_rms_ratio_mean": self.correction_ratio_sum / denom,
            "correction_rms_ratio_max": self.correction_ratio_max,
            "gradient_rms_mean": self.gradient_rms_sum / denom,
        }


def _area_downsample(video: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("posterior consistency expects BxCxTxHxW video tensors")
    b, c, t, h, w = video.shape
    if (h, w) == (target_h, target_w):
        return video
    work = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    down = F.interpolate(work, size=(target_h, target_w), mode="area")
    return down.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()


def posterior_measurement_step(
    high_clean: torch.Tensor,
    low_reference: torch.Tensor,
    *,
    strength: float,
    max_correction_rms_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One normalized DPS-style latent measurement correction.

    The measurement operator is H3-latent spatial area downsampling. Autograd is
    used only through that operator on a detached clean-state probe. The update
    is normalized by residual/gradient RMS and capped relative to the latent
    scale; no H3 model gradient and no VAE decode are involved.
    """
    if high_clean.ndim != 5 or low_reference.ndim != 5:
        raise ValueError("posterior consistency expects BxCxTxHxW tensors")
    if high_clean.shape[:3] != low_reference.shape[:3]:
        raise ValueError("posterior consistency B/C/T geometry differs")
    if not bool(torch.isfinite(high_clean).all().item()) or not bool(torch.isfinite(low_reference).all().item()):
        raise ValueError("posterior consistency received NaN/Inf")
    if strength <= 0.0:
        down = _area_downsample(high_clean.float(), low_reference.shape[-2], low_reference.shape[-1])
        error = float((down - low_reference.float()).square().mean().sqrt().item())
        return high_clean, {
            "measurement_error_before": error,
            "measurement_error_after": error,
            "correction_rms_ratio": 0.0,
            "gradient_rms": 0.0,
            "clamp_scale_mean": 1.0,
        }

    device = high_clean.device
    low = low_reference.detach().to(device=device, dtype=torch.float32)
    with torch.inference_mode(False), torch.enable_grad():
        probe = high_clean.detach().to(dtype=torch.float32).requires_grad_(True)
        measured = _area_downsample(probe, low.shape[-2], low.shape[-1])
        residual = measured - low
        loss = 0.5 * residual.square().sum()
        (gradient,) = torch.autograd.grad(loss, probe, create_graph=False, retain_graph=False)

    dims = tuple(range(1, high_clean.ndim))
    residual_rms = residual.detach().square().mean(dim=dims, keepdim=True).sqrt()
    gradient_rms = gradient.detach().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)
    normalized_gradient = gradient.detach() * (residual_rms / gradient_rms)
    correction = -float(strength) * normalized_gradient

    high_rms = high_clean.detach().float().square().mean(dim=dims, keepdim=True).sqrt()
    low_rms = low.square().mean(dim=dims, keepdim=True).sqrt()
    reference_rms = torch.maximum(high_rms, low_rms).clamp_min(1e-8)
    correction_rms = correction.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)
    cap_scale = torch.clamp(
        reference_rms * float(max_correction_rms_ratio) / correction_rms,
        max=1.0,
    )
    bounded = correction * cap_scale
    corrected = (high_clean.detach().float() + bounded).to(dtype=high_clean.dtype)

    before_error = float(residual.detach().square().mean().sqrt().item())
    after = _area_downsample(corrected.float(), low.shape[-2], low.shape[-1])
    after_error = float((after - low).square().mean().sqrt().item())
    bounded_rms = bounded.square().mean(dim=dims).sqrt()
    ratio = float((bounded_rms / reference_rms.flatten()).mean().item())
    return corrected, {
        "measurement_error_before": before_error,
        "measurement_error_after": after_error,
        "correction_rms_ratio": ratio,
        "gradient_rms": float(gradient_rms.mean().item()),
        "clamp_scale_mean": float(cap_scale.mean().item()),
    }


def patch_posterior_consistency(
    model: Any,
    low_reference_video: torch.Tensor,
    *,
    strength: float = 0.10,
    apply_every: int = 2,
    max_correction_rms_ratio: float = 0.05,
) -> tuple[Any, PosteriorConsistencyConfig, PosteriorConsistencyStats]:
    clone = getattr(model, "clone", None)
    setter = getattr(model, "set_model_sampler_post_cfg_function", None)
    if not callable(clone) or not callable(setter):
        raise TypeError("posterior consistency expects a ComfyUI MODEL/ModelPatcher")
    if not torch.is_tensor(low_reference_video) or low_reference_video.ndim != 5:
        raise TypeError("low_reference_video must be the clean H3 Base Bx24xTxHxW latent")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")

    config = PosteriorConsistencyConfig(
        strength=float(strength),
        apply_every=int(apply_every),
        max_correction_rms_ratio=float(max_correction_rms_ratio),
    )
    stats = PosteriorConsistencyStats()

    def apply_video(video: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        ref = low_reference_video.to(device=video.device, dtype=video.dtype)
        return posterior_measurement_step(
            video,
            ref,
            strength=config.strength,
            max_correction_rms_ratio=config.max_correction_rms_ratio,
        )

    def post_cfg(args):
        denoised = args["denoised"]
        should_apply = config.strength > 0.0 and (stats.calls % config.apply_every == 0)
        if not should_apply:
            stats.record(applied=False)
            return denoised

        if (
            isinstance(denoised, (tuple, list))
            or getattr(denoised, "is_nested", False)
            or getattr(denoised, "tensors", None) is not None
        ):
            video, audio = unwrap_av(denoised)
            corrected, summary = apply_video(video)
            stats.record(
                applied=True,
                error_before=summary["measurement_error_before"],
                error_after=summary["measurement_error_after"],
                correction_ratio=summary["correction_rms_ratio"],
                gradient_rms=summary["gradient_rms"],
            )
            return make_av_container(corrected, audio, template=denoised)

        if torch.is_tensor(denoised):
            base_model = args.get("model")
            latent_shapes = getattr(base_model, "latent_shapes", None)
            if not latent_shapes or len(latent_shapes) < 2:
                raise RuntimeError("posterior consistency cannot resolve packed H3 AV latent_shapes")
            import comfy.utils

            streams = comfy.utils.unpack_latents(denoised, latent_shapes)
            if len(streams) < 2:
                raise RuntimeError("posterior consistency expected packed H3 video+audio")
            corrected, summary = apply_video(streams[0])
            streams[0] = corrected
            packed, _ = comfy.utils.pack_latents(streams)
            stats.record(
                applied=True,
                error_before=summary["measurement_error_before"],
                error_after=summary["measurement_error_after"],
                correction_ratio=summary["correction_rms_ratio"],
                gradient_rms=summary["gradient_rms"],
            )
            return packed

        raise TypeError(f"Unsupported H3 denoised representation: {type(denoised)!r}")

    patched.set_model_sampler_post_cfg_function(post_cfg)
    return patched, config, stats
