from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from h3_icr import latent_upscaler
from h3_icr.latent_upscaler import (
    H3_LATENT_UPSCALER_API,
    H3_LATENT_UPSCALER_KIND,
    KireiH3LatentUpscalerProvider,
    _stabilize_temporal_residual,
)


class _FakeModel:
    def to(self, **_kwargs):
        return self

    def __call__(self, video, *, target_h, target_w, scale, temporal_chunk_size):
        assert scale == pytest.approx(1.5)
        assert temporal_chunk_size == 8
        b, c, t, h, w = video.shape
        work = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        out = F.interpolate(work, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4).to(video)


def _provider() -> KireiH3LatentUpscalerProvider:
    return KireiH3LatentUpscalerProvider(
        checkpoint_path="unused.safetensors",
        device="cpu",
        precision="fp32",
        temporal_chunk_size=8,
        offload_after_upscale=False,
    )


def test_provider_contract_and_exact_target(monkeypatch):
    monkeypatch.setattr(latent_upscaler, "_load_checkpoint_model", lambda *_args: _FakeModel())
    provider = _provider()
    video = torch.randn(1, 24, 3, 8, 12)
    out = provider.upscale_clean_video(video, target_h=12, target_w=18)
    assert provider.api_version == H3_LATENT_UPSCALER_API == 1
    assert provider.kind == H3_LATENT_UPSCALER_KIND
    assert out.shape == (1, 24, 3, 12, 18)
    assert out.dtype == video.dtype
    assert out.device == video.device
    assert not torch.is_inference(out)
    assert provider.last_run["target_shape"] == [1, 24, 3, 12, 18]


@pytest.mark.parametrize(
    ("video", "target_h", "target_w", "error"),
    [
        (torch.randn(1, 23, 3, 8, 8), 12, 12, "24 H3 video channels"),
        (torch.randn(1, 24, 3, 8), 12, 12, "BxCxTxHxW"),
        (torch.randn(1, 24, 3, 8, 8), 4, 12, "does not support spatial downscaling"),
    ],
)
def test_provider_rejects_invalid_geometry(video, target_h, target_w, error):
    with pytest.raises((TypeError, ValueError), match=error):
        _provider().upscale_clean_video(video, target_h=target_h, target_w=target_w)


def test_checkpoint_rejects_wrong_architecture():
    state = {
        "conv_in.weight": torch.zeros(16, 24, 3, 3, 3),
        "conv_out.weight": torch.zeros(24, 16, 3, 3, 3),
    }
    with pytest.raises(ValueError, match="24->512"):
        latent_upscaler._validate_checkpoint(state)


def test_non_safetensors_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="only accept safetensors"):
        latent_upscaler._read_checkpoint(Path("model.ckpt"))


def test_temporal_stability_smooths_only_learned_residual():
    source = torch.zeros(1, 24, 4, 2, 2)
    source[:, :, 2:] = 1.0
    baseline = F.interpolate(source, size=(4, 4, 4), mode="trilinear", align_corners=False)
    alternating = torch.tensor([1.0, -1.0, 1.0, -1.0]).view(1, 1, 4, 1, 1)
    output = baseline + alternating
    stable = _stabilize_temporal_residual(output, source, strength=1.0)
    assert torch.equal(
        F.interpolate(source, size=(4, 4, 4), mode="trilinear", align_corners=False),
        baseline,
    )
    assert (stable - baseline).square().mean() < (output - baseline).square().mean()


def test_temporal_stability_rejects_invalid_strength():
    tensor = torch.zeros(1, 24, 2, 2, 2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _stabilize_temporal_residual(tensor, tensor, strength=1.1)
