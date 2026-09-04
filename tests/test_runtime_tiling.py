from types import SimpleNamespace

import torch

from h3_icr.runtime_tiling import (
    H3TiledRendererConfig,
    TiledRendererStats,
    _child_options_without_spectrum,
    rewrite_tile_layout_global_positions,
)
from h3_icr.tiling import plan_spatial_tiles, target_patch_indices


def test_config_requires_patch_alignment():
    try:
        H3TiledRendererConfig(
            prior_h=48,
            prior_w=84,
            tile_h=47,
            tile_w=64,
            overlap_h=16,
            overlap_w=16,
        )
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("unaligned tile config must fail")


def test_spectrum_is_kept_for_prior_but_removed_from_tile_child_options():
    options = {
        "spectrum_h3_runtime": object(),
        "spectrum_h3_run_id": 9,
        "sample_sigmas": torch.tensor([0.5, 0.0]),
        "minimax_h3_sigma_shift_video": 12.0,
    }
    child = _child_options_without_spectrum(options)
    assert "spectrum_h3_runtime" not in child
    assert "spectrum_h3_run_id" not in child
    assert "sample_sigmas" in child
    assert child["minimax_h3_sigma_shift_video"] == 12.0


def test_global_layout_rewrite_keeps_non_video_positions_and_slices_video_positions():
    full = SimpleNamespace(
        segments=[(0, 2, "text"), (2, 4, "audio"), (4, 28, "video")],
        position_ids=torch.arange(28 * 3, dtype=torch.float64).reshape(28, 3),
    )
    tile = plan_spatial_tiles(4, 6, tile_h=2, tile_w=4, overlap_h=0, overlap_w=0).tiles[-1]
    tile_rows = 4 * (tile.height // 2) * (tile.width // 2)
    tile_layout = SimpleNamespace(
        segments=[(0, 2, "text"), (2, 4, "audio"), (4, 4 + tile_rows, "video")],
        position_ids=torch.full((4 + tile_rows, 3), -1.0, dtype=torch.float64),
    )
    rewrite_tile_layout_global_positions(
        full,
        tile_layout,
        latent_t=4,
        full_h=4,
        full_w=6,
        tile=tile,
    )
    assert torch.equal(tile_layout.position_ids[:4], full.position_ids[:4])
    indices = target_patch_indices(4, 4, 6, tile)
    assert torch.equal(tile_layout.position_ids[4:], full.position_ids[4:][indices])


def test_stats_report_plan_counts():
    stats = TiledRendererStats()
    plan = plan_spatial_tiles(72, 128, tile_h=48, tile_w=64, overlap_h=16, overlap_w=16)
    stats.record_plan(plan, latent_t=37, spectrum_prior=True)
    report = stats.to_dict()
    assert report["prior_calls"] == 1
    assert report["tile_model_calls"] == 6
    assert report["spectrum_prior_calls"] == 1
    assert report["last_full_video_tokens"] == 85248
    assert report["last_tile_video_tokens"] == 28416


def test_tiled_wrapper_runs_one_global_prior_then_actual_tiles(monkeypatch):
    import sys
    import types

    from h3_icr.runtime_tiling import (
        TILED_CONFIG_KEY,
        TILED_STATS_KEY,
        tiled_diffusion_model_wrapper,
    )

    module = types.ModuleType("comfy.ldm.minimax.model")

    class PackedLayout:
        def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
            assert not keyframes
            video_rows = latent_t * (latent_h // 2) * (latent_w // 2)
            audio_rows = audio_t * 2
            self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
            self.segments = [
                (0, text_len, "text"),
                (text_len, text_len + audio_rows, "audio"),
                (text_len + audio_rows, text_len + audio_rows + video_rows, "video"),
            ]
            self.seq_len = self.segments[-1][1]
            self.position_ids = torch.arange(self.seq_len * 3, dtype=torch.float64).reshape(self.seq_len, 3)

    module.PackedLayout = PackedLayout
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.model", module)

    FakeInner = type(
        "MiniMaxH3Model",
        (),
        {"__module__": "comfy.ldm.minimax.model", "patch_size": (1, 2, 2)},
    )

    class Executor:
        def __init__(self):
            self.class_obj = FakeInner()
            self.calls = []

        def __call__(self, x, timestep, context, options, minimax_payload=None, **kwargs):
            video, audio = x
            self.calls.append((tuple(video.shape), dict(options), minimax_payload["layout"]))
            is_prior = video.shape[-2:] == (2, 4)
            value = 0.0 if is_prior else 2.0
            return [torch.full_like(video, value), torch.full_like(audio, 7.0)]

    stats = TiledRendererStats()
    config = H3TiledRendererConfig(
        prior_h=2,
        prior_w=4,
        tile_h=4,
        tile_w=6,
        overlap_h=0,
        overlap_w=0,
        prior_strength=1.0,
    )
    options = {
        TILED_CONFIG_KEY: config,
        TILED_STATS_KEY: stats,
        "spectrum_h3_runtime": object(),
        "spectrum_h3_run_id": 4,
    }
    executor = Executor()
    video = torch.randn(1, 1, 2, 8, 12)
    audio = torch.randn(1, 1, 2, 3)
    context = torch.zeros(1, 2, 1)

    output = tiled_diffusion_model_wrapper(
        executor,
        [video, audio],
        torch.tensor([500.0]),
        context,
        options,
        minimax_payload={},
    )
    assert len(executor.calls) == 5
    assert executor.calls[0][0][-2:] == (2, 4)
    assert "spectrum_h3_runtime" in executor.calls[0][1]
    assert all("spectrum_h3_runtime" not in call[1] for call in executor.calls[1:])
    assert torch.allclose(output[0], torch.ones_like(video), atol=1e-6)
    assert torch.allclose(output[1], torch.full_like(audio, 7.0))
    assert stats.prior_calls == 1
    assert stats.tile_model_calls == 4
