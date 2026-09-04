from types import SimpleNamespace

import torch

from h3_icr.validation import build_validation_manifest
from h3_icr.validation_metrics import (
    build_validation_result_bundle,
    evaluate_latent_output,
    validate_bundle_integrity,
    validate_metrics_integrity,
)


class FakeNoise:
    def __init__(self, seed=7):
        self.seed = seed


class FakeSampler:
    def __init__(self):
        self.extra_options = {}
        self.inpaint_options = {}


class FakeModel:
    model_options = {"transformer_options": {}}


def _backend():
    return {
        "api": 1,
        "kind": "fl2va_reference",
        "checkpoint_format": "full",
        "checkpoint_sha256": "a" * 64,
        "overlay_sha256": "",
        "note": "",
    }


def _base():
    torch.manual_seed(17)
    video = torch.randn(1, 24, 2, 4, 4)
    audio = torch.randn(1, 32, 2, 6)
    return video, audio


def test_identical_output_has_zero_base_error_and_exact_audio():
    video, audio = _base()
    metrics = evaluate_latent_output((video.clone(), audio.clone()), (video, audio))
    validate_metrics_integrity(metrics)
    assert metrics["base_compatibility"]["measurement_rmse"] == 0.0
    assert metrics["base_compatibility"]["low_frequency_rmse"] == 0.0
    assert metrics["base_compatibility"]["temporal_delta_rmse"] == 0.0
    assert metrics["detail"]["hr_residual_rms"] == 0.0
    assert metrics["audio"]["exact"] is True
    assert metrics["audio"]["rmse"] == 0.0


def test_hr_checkerboard_adds_detail_without_changing_base_measurement():
    video, audio = _base()
    up = torch.nn.functional.interpolate(
        video.permute(0, 2, 1, 3, 4).reshape(2, 24, 4, 4),
        size=(8, 8),
        mode="nearest",
    ).reshape(1, 2, 24, 8, 8).permute(0, 2, 1, 3, 4)
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    checker = ((yy + xx) % 2).float().mul(2.0).sub(1.0).view(1, 1, 1, 8, 8)
    output = up + checker * 0.05
    metrics = evaluate_latent_output((output, audio.clone()), (video, audio))
    assert metrics["base_compatibility"]["measurement_rmse"] < 1e-6
    assert metrics["detail"]["hr_residual_rms"] > 0.01
    assert metrics["detail"]["hr_residual_spatial_gradient_rms"] > 0.01
    assert metrics["audio"]["exact"] is True


def test_temporal_drift_is_reported_on_base_grid():
    video, audio = _base()
    output = video.clone()
    output[:, :, 1] += 0.25
    metrics = evaluate_latent_output((output, audio), (video, audio))
    assert metrics["base_compatibility"]["temporal_delta_rmse"] > 0.1


def test_audio_change_is_detected_exactly():
    video, audio = _base()
    changed_audio = audio.clone()
    changed_audio[..., 0] += 0.01
    metrics = evaluate_latent_output((video, changed_audio), (video, audio))
    assert metrics["audio"]["shape_equal"] is True
    assert metrics["audio"]["exact"] is False
    assert metrics["audio"]["rmse"] > 0.0
    assert metrics["audio"]["max_abs"] >= 0.009


def test_m4_seam_diagnostic_detects_strong_internal_boundary():
    video = torch.zeros(1, 24, 2, 8, 12)
    video[..., 6:] = 1.0
    audio = torch.zeros(1, 32, 2, 6)
    renderer = {
        "config": SimpleNamespace(
            tile_h=4,
            tile_w=6,
            overlap_h=2,
            overlap_w=2,
            patch_h=2,
            patch_w=2,
        )
    }
    metrics = evaluate_latent_output((video, audio), (video, audio), renderer=renderer)
    seams = metrics["m4_seams"]
    assert seams["active"] is True
    assert seams["x_boundary_count"] > 0
    assert seams["boundary_x_ratio"] > 1.0


def test_validation_bundle_is_stable_and_binds_manifest_to_metrics():
    video, audio = _base()
    manifest = build_validation_manifest(
        experiment_name="bundle-test",
        comparison_group="control",
        arm="A",
        model=FakeModel(),
        base_latent={"samples": (video, audio)},
        positive=[[torch.zeros(1, 1, 2), {}]],
        negative=None,
        noise=FakeNoise(),
        sampler=FakeSampler(),
        sigmas=torch.tensor([0.5, 0.0]),
        locked_settings={"target": [64, 64]},
        arm_settings={},
        backend=_backend(),
    )
    metrics = evaluate_latent_output((video.clone(), audio.clone()), (video, audio))
    first = build_validation_result_bundle(
        manifest,
        metrics,
        reports={"runtime": {"calls": 3}},
        notes={"clip": "synthetic"},
    )
    second = build_validation_result_bundle(
        manifest,
        metrics,
        reports={"runtime": {"calls": 3}},
        notes={"clip": "synthetic"},
    )
    assert first == second
    assert first["run_id"] == manifest["run_id"]
    assert first["metrics_id"] == metrics["metrics_id"]
    assert len(first["bundle_id"]) == 64
    validate_bundle_integrity(first)
