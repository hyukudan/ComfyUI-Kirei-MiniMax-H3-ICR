import torch

from h3_icr.fidelity import resize_video
from h3_icr.measurement import MeasurementConsistencyConfig
from h3_icr.runtime_measurement import patch_measurement_consistency


class FakeModel:
    def __init__(self):
        self.post_cfg = []

    def clone(self):
        clone = FakeModel()
        clone.post_cfg = list(self.post_cfg)
        return clone

    def set_model_sampler_post_cfg_function(self, fn):
        self.post_cfg.append(fn)


def test_measurement_hook_preserves_audio_and_reduces_error():
    torch.manual_seed(21)
    low = torch.randn(1, 24, 2, 6, 8)
    high = resize_video(low, 12, 16, "bicubic") + torch.randn(1, 24, 2, 12, 16) * 0.25
    audio = torch.randn(1, 32, 2, 5)
    config = MeasurementConsistencyConfig(
        strength=0.6,
        cutoff=1.0,
        high_band_mix=1.0,
        max_correction_rms_ratio=1.0,
        robust_delta=0.0,
    )
    patched, stats = patch_measurement_consistency(
        FakeModel(),
        low,
        sigma_start=0.5,
        config=config,
    )
    assert len(patched.post_cfg) == 1
    output = patched.post_cfg[0]({"denoised": (high, audio), "sigma": torch.tensor([0.5])})
    corrected_video, corrected_audio = output
    assert torch.equal(corrected_audio, audio)
    before = resize_video(high, 6, 8, "area")
    after = resize_video(corrected_video, 6, 8, "area")
    assert (after - low).square().mean() < (before - low).square().mean()
    report = stats.to_dict()
    assert report["applied"] == 1
    assert report["measurement_error_after_mean"] < report["measurement_error_before_mean"]
