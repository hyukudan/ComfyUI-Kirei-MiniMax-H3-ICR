from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .contracts import make_av_container, target_latent_hw, unwrap_av, validate_av
from .fidelity import FidelityConfig, align_clean_hr_to_lr, resize_video

H3_LATENT_UPSCALER_API = 1
H3_LATENT_UPSCALER_KIND = "minimax_h3_learned_latent_upscaler"


@dataclass(frozen=True, slots=True)
class InitConfig:
    transfer: str = "learned_3d"
    target_width: int = 1344
    target_height: int = 768
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)

    def __post_init__(self) -> None:
        if self.transfer not in {"learned_3d", "bicubic"}:
            raise ValueError("transfer must be learned_3d or bicubic")
        target_latent_hw(self.target_width, self.target_height)


def validate_provider(provider: Any) -> None:
    api = getattr(provider, "api_version", None)
    kind = getattr(provider, "kind", None)
    method = getattr(provider, "upscale_clean_video", None)
    if api != H3_LATENT_UPSCALER_API or kind != H3_LATENT_UPSCALER_KIND or not callable(method):
        raise TypeError(
            "learned_3d requires H3_LATENT_UPSCALER API v1 from the companion MiniMax H3 latent upscaler"
        )


def upscale_and_align_clean(
    base_samples: Any,
    config: InitConfig,
    learned_upscaler: Any = None,
) -> tuple[Any, dict[str, object]]:
    validate_av(base_samples)
    source_video, source_audio = unwrap_av(base_samples)
    target_h, target_w = target_latent_hw(config.target_width, config.target_height)

    if config.transfer == "learned_3d":
        if learned_upscaler is None:
            raise ValueError("learned_3d selected but no H3_LATENT_UPSCALER provider is connected")
        validate_provider(learned_upscaler)
        upscaled = learned_upscaler.upscale_clean_video(
            source_video,
            target_h=target_h,
            target_w=target_w,
        )
    else:
        upscaled = resize_video(source_video, target_h, target_w, mode="bicubic")

    if not torch.is_tensor(upscaled) or upscaled.shape[:3] != source_video.shape[:3]:
        raise ValueError("upscaler must preserve H3 video B/C/T")
    if tuple(upscaled.shape[-2:]) != (target_h, target_w):
        raise ValueError(
            f"upscaler returned {tuple(upscaled.shape[-2:])}, expected exact target {(target_h, target_w)}"
        )
    if not upscaled.is_floating_point() or not bool(torch.isfinite(upscaled).all().item()):
        raise ValueError("upscaler returned invalid/non-finite video latent")

    aligned, stats = align_clean_hr_to_lr(upscaled, source_video.to(upscaled.device, upscaled.dtype), config.fidelity)
    audio = source_audio.to(device=aligned.device) if source_audio.device != aligned.device else source_audio
    result = make_av_container(aligned, audio, template=base_samples)
    validate_av(result)
    report = {
        "transfer": config.transfer,
        "source_video_shape": list(source_video.shape),
        "target_video_shape": list(aligned.shape),
        "audio_shape": list(audio.shape),
        "fidelity": stats,
    }
    return result, report
