from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class FidelityConfig:
    strength: float = 0.35
    cutoff: float = 0.25
    max_correction_rms_ratio: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError("fidelity strength must be in [0, 2]")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("fidelity cutoff must be in (0, 1]")
        if not 0.0 < self.max_correction_rms_ratio <= 2.0:
            raise ValueError("max_correction_rms_ratio must be in (0, 2]")


def resize_video(video: torch.Tensor, target_h: int, target_w: int, mode: str = "bicubic") -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError("video must be BxCxTxHxW")
    b, c, t, h, w = video.shape
    if (h, w) == (target_h, target_w):
        return video
    work = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    if mode in {"bilinear", "bicubic"}:
        out = F.interpolate(work, size=(target_h, target_w), mode=mode, align_corners=False)
    elif mode == "area":
        out = F.interpolate(work, size=(target_h, target_w), mode="area")
    else:
        raise ValueError(f"unsupported resize mode {mode!r}")
    return out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).to(video)


def fourier_lowpass(video: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Spatial radial low-pass on each B/C/T plane, preserving tensor shape."""
    if video.ndim != 5:
        raise ValueError("fourier_lowpass expects BxCxTxHxW")
    if cutoff >= 1.0:
        return video
    b, c, t, h, w = video.shape
    work = video.float().reshape(b * c * t, h, w)
    spectrum = torch.fft.rfft2(work, norm="ortho")
    fy = torch.fft.fftfreq(h, device=video.device).abs()[:, None]
    fx = torch.fft.rfftfreq(w, device=video.device).abs()[None, :]
    # cutoff is expressed against Nyquist radius: 1.0 passes every spatial frequency.
    radius = torch.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2)
    mask = (radius <= cutoff).to(spectrum.dtype)
    filtered = torch.fft.irfft2(spectrum * mask, s=(h, w), norm="ortho")
    return filtered.reshape(b, c, t, h, w).to(video)


def _bounded_correction(
    correction: torch.Tensor,
    baseline: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    dims = tuple(range(1, correction.ndim))
    corr_rms = correction.float().square().mean(dim=dims, keepdim=True).sqrt()
    base_rms = baseline.float().square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    scale = torch.clamp(base_rms * ratio / corr_rms.clamp_min(1e-8), max=1.0)
    bounded = correction * scale.to(correction)
    stats = {
        "correction_rms_ratio": float((bounded.float().square().mean(dim=dims).sqrt() / base_rms.flatten()).mean().item()),
        "clamp_scale_mean": float(scale.mean().item()),
    }
    return bounded, stats


def align_clean_hr_to_lr(
    high_clean: torch.Tensor,
    low_clean: torch.Tensor,
    config: FidelityConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Low-frequency initialization alignment inspired by flow-aligned SR methods.

    It preserves high-frequency degrees of freedom in the upscaled latent while forcing
    its downsampled low-frequency structure toward the original H3 base latent.
    """
    if high_clean.ndim != 5 or low_clean.ndim != 5:
        raise ValueError("alignment expects BxCxTxHxW tensors")
    if high_clean.shape[:3] != low_clean.shape[:3]:
        raise ValueError("alignment source/target B/C/T differ")
    if config.strength == 0.0:
        return high_clean, {"downsample_error_before": 0.0, "downsample_error_after": 0.0, "correction_rms_ratio": 0.0}

    down = resize_video(high_clean, low_clean.shape[-2], low_clean.shape[-1], mode="area")
    residual_lr = fourier_lowpass(low_clean - down, config.cutoff)
    residual_hr = resize_video(residual_lr, high_clean.shape[-2], high_clean.shape[-1], mode="bicubic")
    correction = config.strength * residual_hr
    correction, bound_stats = _bounded_correction(correction, high_clean, config.max_correction_rms_ratio)
    aligned = high_clean + correction
    after = resize_video(aligned, low_clean.shape[-2], low_clean.shape[-1], mode="area")
    before_err = float((down.float() - low_clean.float()).square().mean().sqrt().item())
    after_err = float((after.float() - low_clean.float()).square().mean().sqrt().item())
    return aligned, {
        "downsample_error_before": before_err,
        "downsample_error_after": after_err,
        **bound_stats,
    }


def project_to_low_reference(
    high_clean: torch.Tensor,
    low_reference: torch.Tensor,
    *,
    strength: float,
    cutoff: float,
    max_correction_rms_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project only low spatial frequencies of an HR clean estimate toward the H3 Base draft."""
    if high_clean.ndim != 5 or low_reference.ndim != 5:
        raise ValueError("reference projection expects BxCxTxHxW tensors")
    if high_clean.shape[:3] != low_reference.shape[:3]:
        raise ValueError("reference projection B/C/T geometry differs")
    if strength <= 0.0:
        return high_clean, {"correction_rms_ratio": 0.0, "clamp_scale_mean": 1.0, "low_error": 0.0}
    down = resize_video(high_clean, low_reference.shape[-2], low_reference.shape[-1], mode="area")
    low_error = low_reference - down
    residual_lr = fourier_lowpass(low_error, cutoff)
    correction = resize_video(residual_lr, high_clean.shape[-2], high_clean.shape[-1], mode="bicubic")
    correction = correction * float(strength)
    correction, stats = _bounded_correction(correction, high_clean, max_correction_rms_ratio)
    return high_clean + correction, {
        **stats,
        "low_error": float(low_error.float().square().mean().sqrt().item()),
    }


def fidelity_schedule(
    sigma: float,
    sigma_start: float,
    *,
    power: float = 1.0,
    floor: float = 0.0,
) -> float:
    """Structure-first schedule: strongest at second-pass start, relaxed toward sigma=0."""
    if sigma_start <= 0.0:
        return float(floor)
    fraction = max(0.0, min(1.0, float(sigma) / float(sigma_start))) ** float(power)
    return float(floor) + (1.0 - float(floor)) * fraction
