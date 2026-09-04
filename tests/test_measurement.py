import torch

from h3_icr.fidelity import resize_video
from h3_icr.measurement import MeasurementConsistencyConfig, project_measurement_consistency


def test_measurement_projection_reduces_base_grid_error():
    torch.manual_seed(17)
    low = torch.randn(1, 24, 3, 8, 10)
    high = resize_video(low, 16, 20, "bicubic") + torch.randn(1, 24, 3, 16, 20) * 0.35
    config = MeasurementConsistencyConfig(
        strength=0.8,
        cutoff=1.0,
        high_band_mix=1.0,
        max_correction_rms_ratio=1.0,
        robust_delta=0.0,
        iterations=2,
    )
    corrected, stats = project_measurement_consistency(high, low, config)
    before = resize_video(high, 8, 10, "area")
    after = resize_video(corrected, 8, 10, "area")
    assert (after - low).square().mean() < (before - low).square().mean()
    assert stats["measurement_error_after"] < stats["measurement_error_before"]
    assert stats["iterations"] == 2
    assert stats["backprojection_gain_mean"] >= 0.0


def test_measurement_projection_obeys_rms_guard():
    torch.manual_seed(18)
    low = torch.randn(1, 24, 2, 6, 8) * 10.0
    high = torch.randn(1, 24, 2, 12, 16)
    config = MeasurementConsistencyConfig(
        strength=2.0,
        cutoff=1.0,
        high_band_mix=1.0,
        max_correction_rms_ratio=0.05,
        robust_delta=0.0,
        iterations=1,
    )
    corrected, stats = project_measurement_consistency(high, low, config)
    correction = corrected - high
    correction_rms = correction.float().square().mean().sqrt()
    baseline_rms = high.float().square().mean().sqrt()
    assert correction_rms <= baseline_rms * 0.0501
    assert stats["correction_rms_ratio"] <= 0.0501


def test_measurement_strength_zero_is_exact_noop():
    torch.manual_seed(19)
    low = torch.randn(1, 24, 2, 6, 8)
    high = torch.randn(1, 24, 2, 12, 16)
    config = MeasurementConsistencyConfig(strength=0.0)
    corrected, stats = project_measurement_consistency(high, low, config)
    assert torch.equal(corrected, high)
    assert stats["iterations"] == 0


def test_high_band_mix_changes_full_measurement_constraint():
    torch.manual_seed(20)
    low = torch.randn(1, 24, 2, 8, 8)
    high = resize_video(low, 16, 16, "bicubic") + torch.randn(1, 24, 2, 16, 16) * 0.5
    low_only, _ = project_measurement_consistency(
        high,
        low,
        MeasurementConsistencyConfig(
            strength=0.5,
            cutoff=0.2,
            high_band_mix=0.0,
            max_correction_rms_ratio=1.0,
            robust_delta=0.0,
        ),
    )
    mixed, _ = project_measurement_consistency(
        high,
        low,
        MeasurementConsistencyConfig(
            strength=0.5,
            cutoff=0.2,
            high_band_mix=1.0,
            max_correction_rms_ratio=1.0,
            robust_delta=0.0,
        ),
    )
    assert not torch.allclose(low_only, mixed)
