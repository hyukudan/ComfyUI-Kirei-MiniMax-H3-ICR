import pytest
import torch

from h3_icr.contracts import target_latent_hw, validate_av


def test_validate_h3_av_pair():
    video = torch.zeros(1, 24, 7, 48, 64)
    audio = torch.zeros(1, 32, 2, 120)
    shapes = validate_av((video, audio))
    assert shapes.video == tuple(video.shape)
    assert shapes.audio == tuple(audio.shape)


def test_reject_odd_latent_geometry():
    with pytest.raises(ValueError, match="even"):
        validate_av((torch.zeros(1, 24, 7, 47, 64), torch.zeros(1, 32, 2, 120)))


def test_target_geometry_requires_32_pixel_grid():
    assert target_latent_hw(1344, 768) == (48, 84)
    with pytest.raises(ValueError, match="aligned"):
        target_latent_hw(1350, 768)
