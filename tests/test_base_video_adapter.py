import math
from types import SimpleNamespace

import torch

from h3_icr.base_video_adapter import (
    BaseVideoAdapterBlockPatch,
    create_zero_init_base_adapter_provider,
    infer_m4_tile_region_from_global_positions,
    parse_injection_blocks,
    patch_base_video_adapter,
)


FakeInner = type(
    "MiniMaxH3Model",
    (),
    {"__module__": "comfy.ldm.minimax.model"},
)


class FakeModelPatcher:
    def __init__(self, hidden_size=16, layers=4):
        inner = FakeInner()
        inner.hidden_size = hidden_size
        inner.blocks = [object() for _ in range(layers)]
        inner.patch_size = (1, 2, 2)
        inner.latents_dim = 24
        inner.audio_latents_dim = 32
        inner.use_adaln_curves = False
        self.model = SimpleNamespace(diffusion_model=inner)
        self.model_options = {"transformer_options": {}}

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


def _layout(video_rows):
    return SimpleNamespace(
        segments=[
            (0, 2, "text"),
            (2, 4, "audio"),
            (4, 4 + video_rows, "video"),
        ],
    )


def _axis(dim, other, patch=2):
    area = math.sqrt(dim * other)
    ratio = dim / area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _global_tile_layout(*, full_h=8, full_w=12, y0=4, x0=6, tile_h=4, tile_w=6, latent_t=2):
    h_axis = _axis(full_h, full_w)
    w_axis = _axis(full_w, full_h)
    ys = h_axis[y0 // 2 : (y0 + tile_h) // 2]
    xs = w_axis[x0 // 2 : (x0 + tile_w) // 2]
    hh, ww = torch.meshgrid(ys, xs, indexing="ij")
    frame = torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)
    video_positions = []
    for t in range(latent_t):
        pos = torch.empty(frame.shape[0], 3, dtype=torch.float64)
        pos[:, 0] = 10.0 + t
        pos[:, 1:] = frame
        video_positions.append(pos)
    video_positions = torch.cat(video_positions)
    prefix = torch.zeros(4, 3, dtype=torch.float64)
    return SimpleNamespace(
        segments=[
            (0, 2, "text"),
            (2, 4, "audio"),
            (4, 4 + video_positions.shape[0], "video"),
        ],
        position_ids=torch.cat((prefix, video_positions)),
    )


def test_parse_injection_blocks_is_sorted_unique_and_range_checked():
    assert parse_injection_blocks("3,1,3", layer_count=5) == (1, 3)
    try:
        parse_injection_blocks("5", layer_count=5)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("out-of-range adapter block must fail")


def test_zero_init_adapter_module_has_independent_exact_zero_heads():
    model = FakeModelPatcher(hidden_size=16, layers=4)
    provider = create_zero_init_base_adapter_provider(
        model,
        injection_blocks="1,3",
        adapter_dim=32,
    )
    module = provider.module
    dynamic = torch.randn(12, 16)
    static = torch.randn(12, 96)
    residual1 = module(
        dynamic,
        static,
        block_index=1,
        latent_t=2,
        patch_grid_h=2,
        patch_grid_w=3,
        structure_gate=0.75,
    )
    residual3 = module(
        dynamic,
        static,
        block_index=3,
        latent_t=2,
        patch_grid_h=2,
        patch_grid_w=3,
        structure_gate=0.75,
    )
    assert set(module.out_proj) == {"1", "3"}
    assert module.out_proj["1"] is not module.out_proj["3"]
    assert module.output_is_zero_initialized()
    assert torch.equal(residual1, torch.zeros_like(residual1))
    assert torch.equal(residual3, torch.zeros_like(residual3))
    assert provider.trained is False


def test_adapter_rejects_unconfigured_block_head():
    model = FakeModelPatcher(hidden_size=16, layers=4)
    module = create_zero_init_base_adapter_provider(model, injection_blocks="1", adapter_dim=32).module
    try:
        module(
            torch.randn(12, 16),
            torch.randn(12, 96),
            block_index=2,
            latent_t=2,
            patch_grid_h=2,
            patch_grid_w=3,
            structure_gate=0.5,
        )
    except ValueError as exc:
        assert "no configured residual head" in str(exc)
    else:
        raise AssertionError("unconfigured M6 block must fail")


def test_apply_registers_selected_blocks_and_zero_init_scaffold_is_noop():
    model = FakeModelPatcher(hidden_size=16, layers=4)
    provider = create_zero_init_base_adapter_provider(model, injection_blocks="1,3", adapter_dim=32)
    base = torch.randn(1, 24, 2, 4, 6)
    patched, runtime = patch_base_video_adapter(model, base, provider)
    transformer = patched.model_options["transformer_options"]
    replacements = transformer["patches_replace"]["dit"]
    assert set(replacements) == {("double_block", 1), ("double_block", 3)}
    assert "h3_icr_base_video_adapter" in transformer["wrappers"]["diffusion_model"]
    args = {"img": torch.randn(16, 16), "layout": _layout(12)}
    patch = replacements[("double_block", 1)]
    output = patch(args, {"original_block": lambda value: {"img": value["img"].clone()}})
    assert torch.equal(output["img"], args["img"])
    assert runtime.stats.zero_init_bypass_blocks == 1


def test_trained_flag_exercises_static_dynamic_path_but_zero_head_stays_exact():
    model = FakeModelPatcher(hidden_size=16, layers=4)
    provider = create_zero_init_base_adapter_provider(model, injection_blocks="1", adapter_dim=32)
    provider.trained = True
    base = torch.randn(1, 24, 2, 4, 6)
    patched, runtime = patch_base_video_adapter(model, base, provider)
    video_x = torch.randn(1, 24, 2, 4, 6)
    runtime.begin_call(video_x, torch.tensor([500.0]), {"sample_sigmas": torch.tensor([0.6, 0.0])})
    try:
        patch = patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 1)]
        original = torch.randn(16, 16)
        output = patch(
            {"img": original, "layout": _layout(12)},
            {"original_block": lambda value: {"img": value["img"].clone()}},
        )
    finally:
        runtime.end_call()
    assert torch.equal(output["img"], original)
    assert runtime.stats.applied_blocks == 1
    assert runtime.stats.residual_rms_max == 0.0
    assert runtime.stats.static_cache_builds == 1


def test_m4_global_mmrope_recovers_full_canvas_and_exact_tile_bounds():
    layout = _global_tile_layout()
    region = infer_m4_tile_region_from_global_positions(layout, latent_t=2, tile_h=4, tile_w=6)
    assert region == (8, 12, 4, 8, 6, 12)


def test_m4_hr_tile_uses_aligned_base_region_and_keeps_zero_init_parity():
    model = FakeModelPatcher(hidden_size=16, layers=4)
    provider = create_zero_init_base_adapter_provider(model, injection_blocks="1", adapter_dim=32)
    provider.trained = True
    base = torch.randn(1, 24, 2, 4, 6)
    patched, runtime = patch_base_video_adapter(model, base, provider)
    options = {
        "sample_sigmas": torch.tensor([0.6, 0.0]),
        "h3_icr_tiled_renderer": SimpleNamespace(prior_h=2, prior_w=4),
    }
    runtime.begin_call(torch.randn(1, 24, 2, 4, 6), torch.tensor([500.0]), options)
    try:
        patch = patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 1)]
        original = torch.randn(16, 16)
        output = patch(
            {"img": original, "layout": _global_tile_layout()},
            {"original_block": lambda value: {"img": value["img"].clone()}},
        )
    finally:
        runtime.end_call()
    assert torch.equal(output["img"], original)
    assert runtime.stats.m4_tile_aligned_blocks == 1
    assert runtime.stats.tile_region_inferences == 1
    assert runtime.stats.static_cache_builds == 1


def test_provider_architecture_fingerprint_rejects_different_h3_width():
    source = FakeModelPatcher(hidden_size=16, layers=4)
    provider = create_zero_init_base_adapter_provider(source, injection_blocks="1", adapter_dim=32)
    incompatible = FakeModelPatcher(hidden_size=32, layers=4)
    try:
        patch_base_video_adapter(incompatible, torch.randn(1, 24, 2, 4, 6), provider)
    except ValueError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("adapter architecture mismatch must fail")


def test_block_patch_preserves_existing_patch_chain():
    calls = []

    class Previous:
        def __call__(self, args, extra_args):
            calls.append("previous")
            out = extra_args["original_block"](args)
            out["img"] = out["img"] + 2.0
            return out

    model = FakeModelPatcher(hidden_size=16, layers=4)
    provider = create_zero_init_base_adapter_provider(model, injection_blocks="1", adapter_dim=32)
    assert provider.config.injection_blocks == (1,)
    runtime = SimpleNamespace(
        after_block=lambda block_index, args, out: (calls.append(("adapter", block_index)) or out),
        to=lambda value: None,
        clear_cache=lambda: None,
    )
    patch = BaseVideoAdapterBlockPatch(runtime, 1, Previous())
    original = torch.zeros(4, 16)
    out = patch({"img": original}, {"original_block": lambda value: {"img": value["img"].clone()}})
    assert torch.equal(out["img"], torch.full_like(original, 2.0))
    assert calls == ["previous", ("adapter", 1)]
