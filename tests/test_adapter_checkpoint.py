import json
from types import SimpleNamespace

import torch

from h3_icr.adapter_checkpoint import (
    ManagedBaseVideoAdapterProvider,
    adapter_checkpoint_metadata_for_export,
    build_managed_adapter_provider,
    parse_adapter_checkpoint_metadata,
    patch_managed_base_video_adapter,
)
from h3_icr.base_video_adapter import create_zero_init_base_adapter_provider


FakeInner = type(
    "MiniMaxH3Model",
    (),
    {"__module__": "comfy.ldm.minimax.model"},
)


class FakeManagedPatcher:
    def __init__(self, module):
        self.model = module


class FakeModelPatcher:
    def __init__(self, hidden_size=16, layers=4):
        inner = FakeInner()
        inner.hidden_size = hidden_size
        inner.blocks = [object() for _ in range(layers)]
        inner.patch_size = (1, 2, 2)
        inner.latents_dim = 24
        inner.audio_latents_dim = 32
        inner.use_adaln_curves = False
        inner.dtype = torch.float32
        self.model = SimpleNamespace(diffusion_model=inner)
        self.model_options = {"transformer_options": {}}
        self.additional_models = {}

    def clone(self):
        clone = FakeModelPatcher(
            hidden_size=self.model.diffusion_model.hidden_size,
            layers=len(self.model.diffusion_model.blocks),
        )
        clone.model_options = {
            "transformer_options": dict(self.model_options.get("transformer_options", {}))
        }
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        transformer = self.model_options.setdefault("transformer_options", {})
        wrappers = transformer.setdefault("wrappers", {})
        wrappers.setdefault(wrapper_type, {})[key] = [wrapper]

    def set_model_patch_replace(self, patch, category, block_type, block_index):
        transformer = self.model_options.setdefault("transformer_options", {})
        replacements = transformer.setdefault("patches_replace", {}).setdefault(category, {})
        replacements[(block_type, block_index)] = patch

    def set_additional_models(self, key, models):
        self.additional_models[key] = list(models)


def _checkpoint_fixture():
    model = FakeModelPatcher()
    scaffold = create_zero_init_base_adapter_provider(
        model,
        injection_blocks="1,3",
        adapter_dim=32,
        model_id="fake-h3",
    )
    # Simulate a trained checkpoint: give each block-specific residual head a
    # different non-zero value while keeping the shared trunk unchanged.
    with torch.no_grad():
        scaffold.module.out_proj["1"].weight.fill_(0.01)
        scaffold.module.out_proj["1"].bias.fill_(0.02)
        scaffold.module.out_proj["3"].weight.fill_(0.03)
        scaffold.module.out_proj["3"].bias.fill_(0.04)
    state = {key: value.detach().clone() for key, value in scaffold.module.state_dict().items()}
    metadata = adapter_checkpoint_metadata_for_export(
        scaffold,
        training={"teacher": "synthetic", "steps": 7},
        note="unit-test checkpoint",
    )
    return model, scaffold, state, metadata


def test_metadata_export_roundtrip_preserves_config_and_training():
    _model, scaffold, _state, metadata = _checkpoint_fixture()
    parsed = parse_adapter_checkpoint_metadata(metadata)
    assert parsed.api == 1
    assert parsed.kind == "base_video_adapter"
    assert parsed.model_id == "fake-h3"
    assert parsed.architecture_digest == scaffold.architecture_digest
    assert parsed.config.injection_blocks == (1, 3)
    assert parsed.config.adapter_dim == 32
    assert parsed.training == {"teacher": "synthetic", "steps": 7}
    assert parsed.note == "unit-test checkpoint"


def test_metadata_rejects_unknown_config_fields():
    _model, _scaffold, _state, metadata = _checkpoint_fixture()
    config = json.loads(metadata["kirei_h3_icr_config_json"])
    config["mystery"] = 1
    metadata = dict(metadata)
    metadata["kirei_h3_icr_config_json"] = json.dumps(config)
    try:
        parse_adapter_checkpoint_metadata(metadata)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown M6 checkpoint config must fail")


def test_build_managed_provider_strict_loads_and_binds_checkpoint():
    model, scaffold, state, metadata = _checkpoint_fixture()
    sentinel = {}

    def factory(module):
        sentinel["module"] = module
        return FakeManagedPatcher(module)

    provider = build_managed_adapter_provider(
        model,
        state,
        metadata,
        checkpoint_sha256="a" * 64,
        model_patcher_factory=factory,
    )
    assert isinstance(provider, ManagedBaseVideoAdapterProvider)
    assert provider.trained is True
    assert provider.checkpoint_sha256 == "a" * 64
    assert provider.architecture_digest == scaffold.architecture_digest
    assert provider.model_patcher.model is provider.module
    assert sentinel["module"] is provider.module
    assert provider.module.out_proj["1"].weight.abs().mean().item() > 0.0
    assert provider.module.out_proj["3"].weight.abs().mean().item() > provider.module.out_proj["1"].weight.abs().mean().item()
    report = provider.to_dict()
    assert report["managed_by_comfyui"] is True
    assert report["checkpoint_metadata"]["training"]["teacher"] == "synthetic"
    assert report["residual_head_mode"] == "per_injection_block"


def test_build_managed_provider_rejects_missing_state_tensor():
    model, _scaffold, state, metadata = _checkpoint_fixture()
    state = dict(state)
    state.pop(next(iter(state)))
    try:
        build_managed_adapter_provider(
            model,
            state,
            metadata,
            checkpoint_sha256="b" * 64,
            model_patcher_factory=lambda module: FakeManagedPatcher(module),
        )
    except ValueError as exc:
        assert "state_dict keys" in str(exc)
    else:
        raise AssertionError("incomplete M6 adapter state_dict must fail")


def test_build_managed_provider_rejects_architecture_digest_mismatch():
    model, _scaffold, state, metadata = _checkpoint_fixture()
    metadata = dict(metadata)
    metadata["kirei_h3_icr_architecture_digest"] = "0" * 64
    try:
        build_managed_adapter_provider(
            model,
            state,
            metadata,
            checkpoint_sha256="c" * 64,
            model_patcher_factory=lambda module: FakeManagedPatcher(module),
        )
    except ValueError as exc:
        assert "architecture fingerprint" in str(exc)
    else:
        raise AssertionError("M6 architecture mismatch must fail")


def test_managed_apply_registers_adapter_as_comfyui_additional_model():
    model, _scaffold, state, metadata = _checkpoint_fixture()
    provider = build_managed_adapter_provider(
        model,
        state,
        metadata,
        checkpoint_sha256="d" * 64,
        model_patcher_factory=lambda module: FakeManagedPatcher(module),
    )
    patched, runtime = patch_managed_base_video_adapter(
        model,
        torch.randn(1, 24, 2, 4, 6),
        provider,
        strength=0.75,
    )
    assert patched.additional_models["h3_icr_base_video_adapter"] == [provider.model_patcher]
    assert runtime.provider is provider
    assert runtime.strength == 0.75
