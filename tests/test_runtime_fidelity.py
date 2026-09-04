import torch

from h3_icr.fidelity import fidelity_schedule, project_to_low_reference, resize_video


def test_structure_first_schedule_decays_to_zero():
    assert fidelity_schedule(0.5, 0.5) == 1.0
    assert fidelity_schedule(0.25, 0.5) == 0.5
    assert fidelity_schedule(0.0, 0.5) == 0.0


def test_projection_reduces_low_resolution_error():
    torch.manual_seed(4)
    low = torch.randn(1, 24, 3, 8, 10)
    high = resize_video(low, 16, 20, "bicubic") + torch.randn(1, 24, 3, 16, 20) * 0.3
    before = resize_video(high, 8, 10, "area")
    corrected, stats = project_to_low_reference(
        high, low, strength=0.7, cutoff=1.0, max_correction_rms_ratio=1.0
    )
    after = resize_video(corrected, 8, 10, "area")
    assert (after - low).square().mean() < (before - low).square().mean()
    assert stats["correction_rms_ratio"] > 0
