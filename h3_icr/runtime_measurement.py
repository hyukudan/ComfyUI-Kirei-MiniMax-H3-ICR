from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import make_av_container, unwrap_av
from .fidelity import fidelity_schedule
from .measurement import MeasurementConsistencyConfig, project_measurement_consistency
from .runtime_fidelity import _scalar_sigma


@dataclass(slots=True)
class MeasurementConsistencyStats:
    calls: int = 0
    applied: int = 0
    error_before_sum: float = 0.0
    error_after_sum: float = 0.0
    error_reduction_sum: float = 0.0
    correction_ratio_max: float = 0.0
    gain_sum: float = 0.0
    last_schedule: float = 0.0

    def record(self, schedule: float, summary: dict[str, float | int] | None) -> None:
        self.calls += 1
        self.last_schedule = float(schedule)
        if summary is None or int(summary.get("iterations", 0)) <= 0:
            return
        self.applied += 1
        before = float(summary["measurement_error_before"])
        after = float(summary["measurement_error_after"])
        self.error_before_sum += before
        self.error_after_sum += after
        self.error_reduction_sum += before - after
        self.correction_ratio_max = max(
            self.correction_ratio_max,
            float(summary["correction_rms_ratio"]),
        )
        self.gain_sum += float(summary["backprojection_gain_mean"])

    def to_dict(self) -> dict[str, float | int]:
        denom = self.applied if self.applied else 1
        return {
            "calls": self.calls,
            "applied": self.applied,
            "measurement_error_before_mean": self.error_before_sum / denom if self.applied else 0.0,
            "measurement_error_after_mean": self.error_after_sum / denom if self.applied else 0.0,
            "measurement_error_reduction_mean": self.error_reduction_sum / denom if self.applied else 0.0,
            "correction_rms_ratio_max": self.correction_ratio_max,
            "backprojection_gain_mean": self.gain_sum / denom if self.applied else 0.0,
            "last_schedule": self.last_schedule,
        }


def patch_measurement_consistency(
    model: Any,
    low_reference_video: torch.Tensor,
    *,
    sigma_start: float,
    config: MeasurementConsistencyConfig,
) -> tuple[Any, MeasurementConsistencyStats]:
    """Attach the latent measurement-consistency projector after CFG.

    The hook supports H3 NestedTensor x0 and packed AV representations and only edits
    the target-video stream. It is intended to be installed after the low-frequency
    fidelity projector so D(x0_HR) -> z_Base is the last structural correction.
    """
    clone = getattr(model, "clone", None)
    setter = getattr(model, "set_model_sampler_post_cfg_function", None)
    if not callable(clone) or not callable(setter):
        raise TypeError("measurement consistency expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    stats = MeasurementConsistencyStats()

    def apply_video(video: torch.Tensor, sigma: float) -> tuple[torch.Tensor, float, dict[str, float | int] | None]:
        schedule = fidelity_schedule(
            sigma,
            sigma_start,
            power=config.schedule_power,
            floor=config.schedule_floor,
        )
        effective = float(config.strength) * schedule
        if effective <= 0.0:
            return video, schedule, None
        reference = low_reference_video.to(device=video.device, dtype=video.dtype)
        corrected, summary = project_measurement_consistency(
            video,
            reference,
            config,
            effective_strength=effective,
        )
        return corrected, schedule, summary

    def post_cfg(args):
        denoised = args["denoised"]
        sigma = _scalar_sigma(args["sigma"])

        if getattr(denoised, "is_nested", False) or getattr(denoised, "tensors", None) is not None:
            video, audio = unwrap_av(denoised)
            corrected, schedule, summary = apply_video(video, sigma)
            stats.record(schedule, summary)
            return make_av_container(corrected, audio, template=denoised)

        if torch.is_tensor(denoised):
            base_model = args.get("model")
            latent_shapes = getattr(base_model, "latent_shapes", None)
            if not latent_shapes or len(latent_shapes) < 2:
                raise RuntimeError("measurement consistency cannot resolve packed H3 AV latent_shapes")
            import comfy.utils

            streams = comfy.utils.unpack_latents(denoised, latent_shapes)
            if len(streams) < 2:
                raise RuntimeError("measurement consistency expected packed video+audio streams")
            corrected, schedule, summary = apply_video(streams[0], sigma)
            streams[0] = corrected
            packed, _ = comfy.utils.pack_latents(streams)
            stats.record(schedule, summary)
            return packed

        raise TypeError(f"Unsupported H3 denoised representation: {type(denoised)!r}")

    patched.set_model_sampler_post_cfg_function(post_cfg)
    return patched, stats
