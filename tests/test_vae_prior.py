from __future__ import annotations

import pytest
import torch

from h3_icr.vae_prior import fuse_vae_prior


def _samples(video_value: float, audio_value: float = 7.0):
    video = torch.full((1, 24, 3, 4, 6), video_value)
    audio = torch.full((1, 32, 2, 11), audio_value)
    return video, audio


def test_vae_prior_blends_only_video_and_preserves_audio_exactly():
    learned = _samples(2.0)
    prior = _samples(6.0, audio_value=-99.0)

    result, report = fuse_vae_prior(learned, prior, 0.25)

    assert torch.equal(result[0], torch.full_like(result[0], 3.0))
    assert result[1] is learned[1]
    assert torch.equal(result[1], learned[1])
    assert report["mode"] == "one_shot_full_latent"
    assert report["strength"] == 0.25
    assert report["audio"] == "learned_initialization_bypass_exact"


def test_vae_prior_zero_strength_is_video_identity():
    learned = _samples(2.0)
    result, _ = fuse_vae_prior(learned, _samples(6.0), 0.0)
    assert result[0] is learned[0]
    assert result[1] is learned[1]


def test_vae_prior_rejects_shape_mismatch_and_invalid_strength():
    learned = _samples(2.0)
    bad_video = torch.zeros((1, 24, 3, 4, 8))
    bad = (bad_video, learned[1])

    with pytest.raises(ValueError, match="exactly match"):
        fuse_vae_prior(learned, bad, 0.25)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuse_vae_prior(learned, _samples(6.0), 1.1)
