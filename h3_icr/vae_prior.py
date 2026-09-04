from __future__ import annotations

from typing import Any

import torch

from .contracts import make_av_container, unwrap_av, validate_av


def fuse_vae_prior(
    learned_samples: Any,
    vae_prior_samples: Any,
    strength: float,
) -> tuple[Any, dict[str, object]]:
    """Blend a VAE round-trip video prior into the learned initialization.

    The blend is deliberately one-shot and full-latent. H3's 24 latent channels
    do not have a documented frequency semantics, so per-channel spatial
    high-pass mixing would introduce an unsupported assumption. Audio always
    comes from ``learned_samples`` and is never blended.
    """

    validate_av(learned_samples)
    validate_av(vae_prior_samples)
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("VAE prior strength must be in [0, 1]")

    learned_video, learned_audio = unwrap_av(learned_samples)
    prior_video, _ = unwrap_av(vae_prior_samples)
    if tuple(prior_video.shape) != tuple(learned_video.shape):
        raise ValueError(
            "VAE prior video shape must exactly match the learned initialization; "
            f"got {tuple(prior_video.shape)} vs {tuple(learned_video.shape)}"
        )

    prior_video = prior_video.to(device=learned_video.device, dtype=learned_video.dtype)
    with torch.no_grad():
        delta = prior_video - learned_video
        delta_rms = float(delta.float().square().mean().sqrt().item())
        learned_rms = float(learned_video.float().square().mean().sqrt().item())
        if strength == 0.0:
            fused_video = learned_video
        elif strength == 1.0:
            fused_video = prior_video
        else:
            fused_video = torch.lerp(learned_video, prior_video, strength)

    result = make_av_container(fused_video, learned_audio, template=learned_samples)
    validate_av(result)
    return result, {
        "api": 1,
        "mode": "one_shot_full_latent",
        "strength": strength,
        "video_shape": list(fused_video.shape),
        "delta_rms": delta_rms,
        "learned_rms": learned_rms,
        "delta_to_learned_rms_ratio": delta_rms / max(learned_rms, 1.0e-12),
        "audio": "learned_initialization_bypass_exact",
    }
