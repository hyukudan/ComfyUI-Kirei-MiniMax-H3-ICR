import torch
import torch.nn.functional as F

from h3_icr.posterior_consistency import (
    patch_posterior_consistency,
    posterior_measurement_step,
)


def _upsample(low, h, w):
    b, c, t, lh, lw = low.shape
    work = low.permute(0, 2, 1, 3, 4).reshape(b * t, c, lh, lw)
    high = F.interpolate(work, size=(h, w), mode="bilinear", align_corners=False)
    return high.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def test_posterior_measurement_step_reduces_area_downsample_error():
    torch.manual_seed(7)
    low = torch.randn(1, 24, 2, 4, 6)
    high = _upsample(low, 8, 12)
    high = high + 0.20 * torch.randn_like(high)

    corrected, stats = posterior_measurement_step(
        high,
        low,
        strength=0.20,
        max_correction_rms_ratio=0.50,
    )

    assert corrected.shape == high.shape
    assert stats["measurement_error_after"] < stats["measurement_error_before"]
    assert stats["correction_rms_ratio"] <= 0.50 + 1e-6
    assert stats["gradient_rms"] > 0.0


def test_posterior_measurement_step_respects_small_rms_cap():
    torch.manual_seed(8)
    low = torch.randn(1, 24, 1, 4, 4)
    high = 4.0 * torch.randn(1, 24, 1, 8, 8)

    _corrected, stats = posterior_measurement_step(
        high,
        low,
        strength=2.0,
        max_correction_rms_ratio=0.01,
    )
    assert stats["correction_rms_ratio"] <= 0.01001
    assert stats["clamp_scale_mean"] < 1.0


class FakeModel:
    def __init__(self):
        self.post_cfg = []

    def clone(self):
        return FakeModel()

    def set_model_sampler_post_cfg_function(self, fn):
        self.post_cfg.append(fn)


def test_posterior_hook_preserves_audio_and_honors_apply_every():
    torch.manual_seed(9)
    base_video = torch.randn(1, 24, 1, 4, 4)
    high_video = _upsample(base_video, 8, 8) + 0.1 * torch.randn(1, 24, 1, 8, 8)
    audio = torch.randn(1, 32, 2, 5)

    patched, config, stats = patch_posterior_consistency(
        FakeModel(),
        base_video,
        strength=0.2,
        apply_every=2,
        max_correction_rms_ratio=0.2,
    )
    assert config.apply_every == 2
    assert len(patched.post_cfg) == 1
    hook = patched.post_cfg[0]

    first = hook({"denoised": (high_video, audio), "sigma": torch.tensor([0.5])})
    second = hook({"denoised": (high_video, audio), "sigma": torch.tensor([0.4])})

    assert torch.equal(first[1], audio)
    assert torch.equal(second[1], audio)
    assert not torch.equal(first[0], high_video)
    assert torch.equal(second[0], high_video)
    assert stats.calls == 2
    assert stats.applied == 1
    report = stats.to_dict()
    assert report["measurement_error_after_mean"] < report["measurement_error_before_mean"]
