import torch

from h3_icr.fidelity import FidelityConfig, align_clean_hr_to_lr, fourier_lowpass, resize_video


def test_fourier_lowpass_suppresses_checkerboard():
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    checker = ((yy + xx) % 2).float() * 2 - 1
    video = checker.view(1, 1, 1, 16, 16).repeat(1, 24, 3, 1, 1)
    filtered = fourier_lowpass(video, 0.2)
    assert filtered.abs().mean() < video.abs().mean() * 0.1


def test_alignment_reduces_downsample_error():
    torch.manual_seed(2)
    low = torch.randn(1, 24, 3, 8, 10)
    high = resize_video(low, 16, 20, mode="bicubic") + 0.2 * torch.randn(1, 24, 3, 16, 20)
    aligned, stats = align_clean_hr_to_lr(
        high,
        low,
        FidelityConfig(strength=0.8, cutoff=1.0, max_correction_rms_ratio=1.0),
    )
    assert aligned.shape == high.shape
    assert stats["downsample_error_after"] < stats["downsample_error_before"]
