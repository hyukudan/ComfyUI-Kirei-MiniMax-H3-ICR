from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import make_av_container, unwrap_av
from .fidelity import fidelity_schedule, project_to_low_reference


@dataclass(slots=True)
class PerStepFidelityStats:
    calls: int = 0
    applied: int = 0
    correction_ratio_sum: float = 0.0
    correction_ratio_max: float = 0.0
    last_schedule: float = 0.0

    def record(self, schedule: float, correction_ratio: float) -> None:
        self.calls += 1
        self.last_schedule = float(schedule)
        if correction_ratio > 0.0:
            self.applied += 1
            self.correction_ratio_sum += float(correction_ratio)
            self.correction_ratio_max = max(self.correction_ratio_max, float(correction_ratio))

    def to_dict(self) -> dict[str, float | int]:
        mean = self.correction_ratio_sum / self.applied if self.applied else 0.0
        return {
            "calls": self.calls,
            "applied": self.applied,
            "correction_rms_ratio_mean": mean,
            "correction_rms_ratio_max": self.correction_ratio_max,
            "last_schedule": self.last_schedule,
        }


def _scalar_sigma(value: Any) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            raise ValueError("sampler post-CFG sigma is empty")
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def patch_per_step_fidelity(
    model: Any,
    low_reference_video: torch.Tensor,
    *,
    sigma_start: float,
    strength: float,
    cutoff: float,
    max_correction_rms_ratio: float,
    schedule_power: float = 1.0,
    schedule_floor: float = 0.0,
) -> tuple[Any, PerStepFidelityStats]:
    """Attach an H3-aware post-CFG clean-state projector.

    Handles both ComfyUI's NestedTensor x0 view and the packed latent representation.
    It never modifies audio. Unknown layouts fail closed instead of silently skipping.
    """
    clone = getattr(model, "clone", None)
    setter = getattr(model, "set_model_sampler_post_cfg_function", None)
    if not callable(clone) or not callable(setter):
        raise TypeError("per-step H3 fidelity expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    stats = PerStepFidelityStats()

    def post_cfg(args):
        denoised = args["denoised"]
        sigma = _scalar_sigma(args["sigma"])
        schedule = fidelity_schedule(sigma, sigma_start, power=schedule_power, floor=schedule_floor)
        effective = float(strength) * schedule
        if effective <= 0.0:
            stats.record(schedule, 0.0)
            return denoised

        if getattr(denoised, "is_nested", False) or getattr(denoised, "tensors", None) is not None:
            video, audio = unwrap_av(denoised)
            ref = low_reference_video.to(device=video.device, dtype=video.dtype)
            corrected, summary = project_to_low_reference(
                video,
                ref,
                strength=effective,
                cutoff=cutoff,
                max_correction_rms_ratio=max_correction_rms_ratio,
            )
            stats.record(schedule, summary["correction_rms_ratio"])
            return make_av_container(corrected, audio, template=denoised)

        if torch.is_tensor(denoised):
            base_model = args.get("model")
            latent_shapes = getattr(base_model, "latent_shapes", None)
            if not latent_shapes or len(latent_shapes) < 2:
                raise RuntimeError("H3 ICR per-step fidelity cannot resolve packed AV latent_shapes")
            import comfy.utils

            streams = comfy.utils.unpack_latents(denoised, latent_shapes)
            if len(streams) < 2:
                raise RuntimeError("H3 ICR expected packed video+audio streams")
            video = streams[0]
            ref = low_reference_video.to(device=video.device, dtype=video.dtype)
            corrected, summary = project_to_low_reference(
                video,
                ref,
                strength=effective,
                cutoff=cutoff,
                max_correction_rms_ratio=max_correction_rms_ratio,
            )
            streams[0] = corrected
            packed, _ = comfy.utils.pack_latents(streams)
            stats.record(schedule, summary["correction_rms_ratio"])
            return packed

        raise TypeError(f"Unsupported H3 denoised representation: {type(denoised)!r}")

    patched.set_model_sampler_post_cfg_function(post_cfg)
    return patched, stats
