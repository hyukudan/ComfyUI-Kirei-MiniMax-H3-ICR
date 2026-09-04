import torch

from h3_icr.validation import (
    build_validation_manifest,
    canonical_descriptor,
    compare_validation_manifests,
    handle_descriptor,
    validate_manifest_integrity,
)


class FakeNoise:
    def __init__(self, seed):
        self.seed = int(seed)


class FakeSampler:
    def __init__(self, eta=0.0):
        self.sampler_function = fake_sampler_function
        self.extra_options = {"eta": float(eta)}
        self.inpaint_options = {}


def fake_sampler_function(*args, **kwargs):
    return args, kwargs


class FakeModel:
    def __init__(self, backend=None):
        transformer = {}
        if backend is not None:
            transformer["h3_icr_backend"] = dict(backend)
        self.model_options = {"transformer_options": transformer}


class FakeComfyNestedTensor:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)


def _backend(kind="fl2va_reference", checkpoint="a", overlay=""):
    return {
        "api": 1,
        "kind": kind,
        "checkpoint_format": "full",
        "checkpoint_sha256": checkpoint * 64,
        "overlay_sha256": overlay * 64 if overlay else "",
        "note": "",
    }


def _inputs():
    torch.manual_seed(11)
    video = torch.randn(1, 24, 2, 4, 6)
    audio = torch.randn(1, 32, 2, 5)
    positive = [[torch.randn(1, 3, 4), {"minimax_refs": [torch.randn(1, 24, 1, 2, 2)]}]]
    sigmas = torch.tensor([0.5, 0.3, 0.0], dtype=torch.float32)
    return {
        "model": FakeModel(),
        "base_latent": {"samples": (video, audio)},
        "positive": positive,
        "negative": None,
        "noise": FakeNoise(1234),
        "sampler": FakeSampler(eta=0.1),
        "sigmas": sigmas,
    }


def _manifest(arm="control", backend=None, arm_settings=None, locked=None, **overrides):
    values = _inputs()
    values.update(overrides)
    return build_validation_manifest(
        experiment_name="unit-validation",
        comparison_group="backend-ab",
        arm=arm,
        model=values["model"],
        base_latent=values["base_latent"],
        positive=values["positive"],
        negative=values["negative"],
        noise=values["noise"],
        sampler=values["sampler"],
        sigmas=values["sigmas"],
        locked_settings=locked or {"target": [1344, 768], "upscaler_sha256": "f" * 64},
        arm_settings=arm_settings or {},
        strict_hashing=True,
        backend=backend or _backend(),
    )


def test_manifest_is_deterministic_for_identical_content():
    values = _inputs()
    kwargs = dict(
        experiment_name="unit-validation",
        comparison_group="same-run",
        arm="A",
        model=values["model"],
        base_latent=values["base_latent"],
        positive=values["positive"],
        negative=None,
        noise=values["noise"],
        sampler=values["sampler"],
        sigmas=values["sigmas"],
        locked_settings={"x": 1},
        arm_settings={"strength": 0.2},
        strict_hashing=True,
        backend=_backend(),
    )
    first = build_validation_manifest(**kwargs)
    second = build_validation_manifest(**kwargs)
    assert first == second
    assert len(first["run_id"]) == 64
    validate_manifest_integrity(first)


def test_comfy_nested_tensor_container_has_stable_strict_fingerprint():
    video = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    first = canonical_descriptor(FakeComfyNestedTensor((video, audio)), strict=True)
    second = canonical_descriptor(FakeComfyNestedTensor((video.clone(), audio.clone())), strict=True)
    assert first == second
    assert first["type"] == "tensor_container"
    assert [child["shape"] for child in first["children"]] == [
        [1, 2, 3, 2, 2],
        [1, 2, 2, 3],
    ]


def test_comfy_nested_tensor_container_rejects_non_tensor_members():
    try:
        canonical_descriptor(FakeComfyNestedTensor((torch.zeros(1), "audio")), strict=True)
    except TypeError as exc:
        assert "non-tensor member" in str(exc)
    else:
        raise AssertionError("strict validation must reject malformed tensor containers")


def test_tensor_hash_changes_when_base_content_changes():
    values = _inputs()
    first = build_validation_manifest(
        experiment_name="unit-validation",
        comparison_group="base-lock",
        arm="A",
        model=values["model"],
        base_latent=values["base_latent"],
        positive=values["positive"],
        negative=None,
        noise=values["noise"],
        sampler=values["sampler"],
        sigmas=values["sigmas"],
        locked_settings={},
        arm_settings={},
        backend=_backend(),
    )
    video, audio = values["base_latent"]["samples"]
    changed = video.clone()
    changed[..., 0, 0] += 1.0
    second = build_validation_manifest(
        experiment_name="unit-validation",
        comparison_group="base-lock",
        arm="B",
        model=values["model"],
        base_latent={"samples": (changed, audio)},
        positive=values["positive"],
        negative=None,
        noise=values["noise"],
        sampler=values["sampler"],
        sigmas=values["sigmas"],
        locked_settings={},
        arm_settings={},
        backend=_backend(),
    )
    report = compare_validation_manifests(first, second)
    assert report["compatible"] is False
    assert report["locks_identical"] is False
    assert any(row["path"].startswith("$.locks.base_latent") for row in report["unexpected_differences"])


def test_backend_only_ab_is_valid_when_backend_path_is_explicitly_allowed():
    values = _inputs()
    common = dict(
        experiment_name="unit-validation",
        comparison_group="backend-only",
        model=values["model"],
        base_latent=values["base_latent"],
        positive=values["positive"],
        negative=None,
        noise=values["noise"],
        sampler=values["sampler"],
        sigmas=values["sigmas"],
        locked_settings={"target": [1344, 768]},
        arm_settings={},
    )
    first = build_validation_manifest(arm="FL2VA", backend=_backend("fl2va_reference", "a"), **common)
    second = build_validation_manifest(
        arm="Hybrid45",
        backend=_backend("hybrid_late_adaln", "b", "c"),
        **common,
    )
    report = compare_validation_manifests(first, second, allowed_differences="arm.backend")
    assert report["compatible"] is True
    assert report["locks_identical"] is True
    assert report["unexpected_differences"] == []
    assert any(row["path"].startswith("$.arm.backend") for row in report["accepted_differences"])


def test_sigma_change_is_rejected_even_when_backend_is_allowed():
    values = _inputs()
    common = dict(
        experiment_name="unit-validation",
        comparison_group="sigma-lock",
        model=values["model"],
        base_latent=values["base_latent"],
        positive=values["positive"],
        negative=None,
        noise=values["noise"],
        sampler=values["sampler"],
        locked_settings={},
        arm_settings={},
    )
    first = build_validation_manifest(
        arm="A",
        sigmas=values["sigmas"],
        backend=_backend("fl2va_reference", "a"),
        **common,
    )
    second = build_validation_manifest(
        arm="B",
        sigmas=torch.tensor([0.55, 0.3, 0.0]),
        backend=_backend("hybrid_late_adaln", "b", "c"),
        **common,
    )
    report = compare_validation_manifests(first, second, allowed_differences="arm.backend")
    assert report["compatible"] is False
    assert any(row["path"].startswith("$.locks.sigmas") for row in report["unexpected_differences"])


def test_declared_arm_setting_path_can_change_without_unlocking_other_state():
    values = _inputs()
    common = dict(
        experiment_name="unit-validation",
        comparison_group="m3b-strength",
        model=values["model"],
        base_latent=values["base_latent"],
        positive=values["positive"],
        negative=None,
        noise=values["noise"],
        sampler=values["sampler"],
        sigmas=values["sigmas"],
        locked_settings={"backend_sha": "a" * 64},
        backend=_backend(),
    )
    first = build_validation_manifest(arm="0.10", arm_settings={"m3b": {"strength": 0.10}}, **common)
    second = build_validation_manifest(arm="0.20", arm_settings={"m3b": {"strength": 0.20}}, **common)
    report = compare_validation_manifests(
        first,
        second,
        allowed_differences="arm.settings.m3b.strength",
    )
    assert report["compatible"] is True
    assert report["locks_identical"] is True


def test_corrupted_manifest_run_id_is_rejected():
    manifest = _manifest()
    manifest = dict(manifest)
    manifest["run_id"] = "0" * 64
    try:
        validate_manifest_integrity(manifest)
    except ValueError as exc:
        assert "run_id" in str(exc)
    else:
        raise AssertionError("corrupted validation manifest must fail")


def test_strict_canonicalization_rejects_closure_state():
    captured = 7

    def closure(value):
        return value + captured

    try:
        canonical_descriptor({"callback": closure}, strict=True)
    except TypeError as exc:
        assert "closure" in str(exc)
    else:
        raise AssertionError("strict validation must reject untracked closure state")


def test_renderer_handle_includes_prior_schedule_configuration():
    class Config:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return dict(self.payload)

    descriptor = handle_descriptor(
        {
            "api": 1,
            "config": Config({"prior_strength": 0.3}),
            "prior_schedule": Config({"floor": 0.15, "power": 1.0}),
        }
    )
    assert descriptor["config"]["prior_strength"] == 0.3
    assert descriptor["prior_schedule"] == {"floor": 0.15, "power": 1.0}
