from __future__ import annotations

from dataclasses import dataclass

import torch

from .fidelity import _bounded_correction, fourier_lowpass, resize_video


@dataclass(frozen=True, slots=True)
class MeasurementConsistencyConfig:
    strength: float = 0.15
    cutoff: float = 0.35
    high_band_mix: float = 0.25
    max_correction_rms_ratio: float = 0.15
    robust_delta: float = 3.0
    max_backprojection_gain: float = 2.0
    iterations: int = 1
    schedule_power: float = 1.0
    schedule_floor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError("measurement strength must be in [0, 2]")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("measurement cutoff must be in (0, 1]")
        if not 0.0 <= self.high_band_mix <= 1.0:
            raise ValueError("high_band_mix must be in [0, 1]")
        if not 0.0 < self.max_correction_rms_ratio <= 2.0:
            raise ValueError("max_correction_rms_ratio must be in (0, 2]")
        if self.robust_delta < 0.0:
            raise ValueError("robust_delta must be non-negative")
        if not 0.0 < self.max_backprojection_gain <= 8.0:
            raise ValueError("max_backprojection_gain must be in (0, 8]")
        if not 1 <= self.iterations <= 8:
            raise ValueError("measurement iterations must be between 1 and 8")
        if self.schedule_power < 0.0:
            raise ValueError("schedule_power must be non-negative")
        if not 0.0 <= self.schedule_floor <= 1.0:
            raise ValueError("schedule_floor must be in [0, 1]")


def _rms_per_sample(value: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, value.ndim))
    return value.float().square().mean(dim=dims, keepdim=True).sqrt()


def _robust_residual(residual: torch.Tensor, delta: float) -> tuple[torch.Tensor, float]:
    if delta <= 0.0:
        return residual, 0.0
    scale = _rms_per_sample(residual).clamp_min(1e-8)
    threshold = scale * float(delta)
    weights = torch.clamp(threshold / residual.float().abs().clamp_min(1e-8), max=1.0)
    outlier_fraction = float((weights < 1.0).float().mean().item())
    return residual * weights.to(residual), outlier_fraction


def _measurement_band(residual: torch.Tensor, cutoff: float, high_band_mix: float) -> torch.Tensor:
    low = fourier_lowpass(residual, cutoff)
    if high_band_mix <= 0.0:
        return low
    return low + (residual - low) * float(high_band_mix)


def _normalized_backprojection(
    residual_lr: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    max_gain: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    lifted = resize_video(residual_lr, target_h, target_w, mode="bicubic")
    response = resize_video(lifted, residual_lr.shape[-2], residual_lr.shape[-1], mode="area")
    dims = tuple(range(1, residual_lr.ndim))
    numerator = (residual_lr.float() * response.float()).sum(dim=dims, keepdim=True)
    denominator = response.float().square().sum(dim=dims, keepdim=True).clamp_min(1e-8)
    gain = (numerator / denominator).clamp(min=0.0, max=float(max_gain))
    projected = lifted * gain.to(lifted)
    return projected, {
        "backprojection_gain_mean": float(gain.mean().item()),
        "backprojection_gain_max": float(gain.max().item()),
    }


def project_measurement_consistency(
    high_clean: torch.Tensor,
    low_reference: torch.Tensor,
    config: MeasurementConsistencyConfig,
    *,
    effective_strength: float | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Project an HR H3 clean estimate toward the observed Base latent measurement.

    D is spatial area downsampling to the Base latent grid. The residual is optionally
    robustified, split into low/high spatial bands, lifted back to HR, normalized by the
    measured D(U(r)) response, and RMS-bounded. Audio never enters this function.
    """
    if high_clean.ndim != 5 or low_reference.ndim != 5:
        raise ValueError("measurement consistency expects BxCxTxHxW tensors")
    if high_clean.shape[:3] != low_reference.shape[:3]:
        raise ValueError("measurement source/target B/C/T geometry differs")
    if not bool(torch.isfinite(high_clean).all().item()) or not bool(torch.isfinite(low_reference).all().item()):
        raise ValueError("measurement consistency received NaN/Inf")

    strength = config.strength if effective_strength is None else float(effective_strength)
    if strength < 0.0:
        raise ValueError("effective measurement strength must be non-negative")

    down_before = resize_video(high_clean, low_reference.shape[-2], low_reference.shape[-1], mode="area")
    error_before = float((down_before.float() - low_reference.float()).square().mean().sqrt().item())
    if strength == 0.0:
        return high_clean, {
            "measurement_error_before": error_before,
            "measurement_error_after": error_before,
            "correction_rms_ratio": 0.0,
            "clamp_scale_mean": 1.0,
            "backprojection_gain_mean": 0.0,
            "backprojection_gain_max": 0.0,
            "robust_outlier_fraction": 0.0,
            "iterations": 0,
        }

    current = high_clean
    correction_ratio_max = 0.0
    clamp_scale_sum = 0.0
    gain_mean_sum = 0.0
    gain_max = 0.0
    outlier_sum = 0.0

    for _ in range(config.iterations):
        down = resize_video(current, low_reference.shape[-2], low_reference.shape[-1], mode="area")
        residual = low_reference - down
        residual, outlier_fraction = _robust_residual(residual, config.robust_delta)
        residual = _measurement_band(residual, config.cutoff, config.high_band_mix)
        lifted, gain_stats = _normalized_backprojection(
            residual,
            target_h=current.shape[-2],
            target_w=current.shape[-1],
            max_gain=config.max_backprojection_gain,
        )
        correction = lifted * strength
        correction, bound_stats = _bounded_correction(
            correction,
            current,
            config.max_correction_rms_ratio,
        )
        current = current + correction
        correction_ratio_max = max(correction_ratio_max, float(bound_stats["correction_rms_ratio"]))
        clamp_scale_sum += float(bound_stats["clamp_scale_mean"])
        gain_mean_sum += float(gain_stats["backprojection_gain_mean"])
        gain_max = max(gain_max, float(gain_stats["backprojection_gain_max"]))
        outlier_sum += outlier_fraction

    down_after = resize_video(current, low_reference.shape[-2], low_reference.shape[-1], mode="area")
    error_after = float((down_after.float() - low_reference.float()).square().mean().sqrt().item())
    iterations = int(config.iterations)
    return current, {
        "measurement_error_before": error_before,
        "measurement_error_after": error_after,
        "correction_rms_ratio": correction_ratio_max,
        "clamp_scale_mean": clamp_scale_sum / iterations,
        "backprojection_gain_mean": gain_mean_sum / iterations,
        "backprojection_gain_max": gain_max,
        "robust_outlier_fraction": outlier_sum / iterations,
        "iterations": iterations,
    }
