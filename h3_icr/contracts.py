from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

H3_VIDEO_CHANNELS = 24
H3_AUDIO_CHANNELS = 32
H3_AUDIO_TRACKS = 2
H3_VAE_SPATIAL_FACTOR = 16
H3_DIT_SPATIAL_PATCH = 2
H3_PIXEL_ALIGNMENT = H3_VAE_SPATIAL_FACTOR * H3_DIT_SPATIAL_PATCH


@dataclass(frozen=True, slots=True)
class AVShapes:
    video: tuple[int, ...]
    audio: tuple[int, ...]


def unwrap_av(samples: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the video/audio members of an H3 AV sample container."""
    tensors = getattr(samples, "tensors", None)
    if tensors is not None:
        if not isinstance(tensors, (tuple, list)) or len(tensors) != 2:
            raise TypeError("H3 NestedTensor must contain exactly video and audio members")
        return tensors[0], tensors[1]
    if isinstance(samples, (tuple, list)) and len(samples) == 2:
        return samples[0], samples[1]
    raise TypeError("Expected an H3 AV NestedTensor or a (video, audio) pair")


def validate_av(samples: Any) -> AVShapes:
    video, audio = unwrap_av(samples)
    if not torch.is_tensor(video) or not torch.is_tensor(audio):
        raise TypeError("H3 video and audio members must be torch tensors")
    if video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(
            f"H3 video must be Bx{H3_VIDEO_CHANNELS}xTxHxW, got {tuple(video.shape)}"
        )
    if audio.ndim != 4 or audio.shape[1] != H3_AUDIO_CHANNELS or audio.shape[2] != H3_AUDIO_TRACKS:
        raise ValueError(
            f"H3 audio must be Bx{H3_AUDIO_CHANNELS}x{H3_AUDIO_TRACKS}xT, got {tuple(audio.shape)}"
        )
    if video.shape[0] != audio.shape[0]:
        raise ValueError("H3 video/audio batch sizes differ")
    if video.shape[-2] % H3_DIT_SPATIAL_PATCH or video.shape[-1] % H3_DIT_SPATIAL_PATCH:
        raise ValueError("H3 video latent H/W must be even for the DiT 2x2 spatial patch")
    if not video.is_floating_point() or not audio.is_floating_point():
        raise TypeError("H3 AV latents must use floating-point tensors")
    if not bool(torch.isfinite(video).all().item()) or not bool(torch.isfinite(audio).all().item()):
        raise ValueError("H3 AV latents contain NaN/Inf")
    return AVShapes(tuple(video.shape), tuple(audio.shape))


def target_latent_hw(width: int, height: int) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("target width/height must be positive")
    if width % H3_PIXEL_ALIGNMENT or height % H3_PIXEL_ALIGNMENT:
        raise ValueError(
            f"target width/height must be aligned to {H3_PIXEL_ALIGNMENT} pixels; got {width}x{height}"
        )
    return height // H3_VAE_SPATIAL_FACTOR, width // H3_VAE_SPATIAL_FACTOR


def make_av_container(video: torch.Tensor, audio: torch.Tensor, template: Any = None) -> Any:
    """Create an AV container matching ComfyUI at runtime and tuples in tests."""
    if template is not None and isinstance(template, tuple):
        return (video, audio)
    if template is not None and isinstance(template, list):
        return [video, audio]
    try:
        import comfy.nested_tensor
    except ImportError:
        return (video, audio)
    return comfy.nested_tensor.NestedTensor((video, audio))
