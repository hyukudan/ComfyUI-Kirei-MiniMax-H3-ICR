from types import SimpleNamespace

import torch

from h3_icr.runtime_tiling import (
    H3TiledRendererConfig,
    TiledRendererStats,
    _child_options_without_spectrum,
    _payload_with_resized_keyframes,
    _payload_with_tiled_keyframes,
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


def test_global_layout_rewrite_slices_keyframe_cond_rows_with_full_canvas_positions():
    tile = plan_spatial_tiles(4, 6, tile_h=2, tile_w=4, overlap_h=0, overlap_w=0).tiles[-1]
    full_cond_rows = 1 * (4 // 2) * (6 // 2)
    full_video_rows = 2 * (4 // 2) * (6 // 2)
    tile_cond_rows = 1 * (tile.height // 2) * (tile.width // 2)
    tile_video_rows = 2 * (tile.height // 2) * (tile.width // 2)

    full = SimpleNamespace(
        segments=[
            (0, 2, "text"),
            (2, 2 + full_cond_rows, "cond"),
            (2 + full_cond_rows, 4 + full_cond_rows, "audio"),
            (4 + full_cond_rows, 4 + full_cond_rows + full_video_rows, "video"),
        ],
        position_ids=torch.arange(
            (4 + full_cond_rows + full_video_rows) * 3, dtype=torch.float64
        ).reshape(4 + full_cond_rows + full_video_rows, 3),
    )
    tile_layout = SimpleNamespace(
        segments=[
            (0, 2, "text"),
            (2, 2 + tile_cond_rows, "cond"),
            (2 + tile_cond_rows, 4 + tile_cond_rows, "audio"),
            (4 + tile_cond_rows, 4 + tile_cond_rows + tile_video_rows, "video"),
        ],
        position_ids=torch.full(
            (4 + tile_cond_rows + tile_video_rows, 3), -1.0, dtype=torch.float64
        ),
    )
    keyframe = {"latent": torch.zeros(1, 24, 1, 4, 6), "resolved_frame_index": 0}
    rewrite_tile_layout_global_positions(
        full,
        tile_layout,
        latent_t=2,
        full_h=4,
        full_w=6,
        tile=tile,
        keyframes=[keyframe],
    )

    cond_indices = target_patch_indices(1, 4, 6, tile)
    assert torch.equal(
        tile_layout.position_ids[2 : 2 + tile_cond_rows],
        full.position_ids[2 : 2 + full_cond_rows][cond_indices],
    )
    full_video_start = 4 + full_cond_rows
    tile_video_start = 4 + tile_cond_rows
    video_indices = target_patch_indices(2, 4, 6, tile)
    assert torch.equal(
        tile_layout.position_ids[tile_video_start:],
        full.position_ids[full_video_start:][video_indices],
    )


def test_keyframe_payloads_resize_crop_and_rebuild_condition_latents():
    keyframe_video = torch.arange(1 * 24 * 1 * 4 * 6, dtype=torch.float32).reshape(1, 24, 1, 4, 6)
    keyframe_audio = torch.ones(1, 32, 2, 5)
    ref_video = torch.zeros(1, 24, 1, 2, 2)
    payload = {
        "keyframes": [
            {
                "resolved_frame_index": 0,
                "latent": keyframe_video,
                "audio_latent": keyframe_audio,
            }
        ],
        "refs": [{"kind": "image", "latent": ref_video}],
        "cond_video_latents": [torch.full_like(keyframe_video, -1.0)],
        "cond_audio_latents": [],
    }

    prior = _payload_with_resized_keyframes(payload, 2, 4)
    assert prior["keyframes"][0]["latent"].shape[-2:] == (2, 4)
    assert prior["cond_video_latents"][0].shape[-2:] == (2, 4)
    assert prior["cond_video_latents"][1] is ref_video
    assert prior["cond_audio_latents"][0] is keyframe_audio

    tile = plan_spatial_tiles(4, 6, tile_h=2, tile_w=4, overlap_h=0, overlap_w=0).tiles[-1]
    tiled = _payload_with_tiled_keyframes(payload, tile)
    expected = keyframe_video[..., tile.y0 : tile.y1, tile.x0 : tile.x1]
    assert torch.equal(tiled["keyframes"][0]["latent"], expected)
    assert tiled["cond_video_latents"][0] is tiled["keyframes"][0]["latent"]
    assert tiled["cond_video_latents"][1] is ref_video
    assert tiled["cond_audio_latents"][0] is keyframe_audio


def test_stats_report_plan_counts():
    stats = TiledRendererStats()
    plan = plan_spatial_tiles(72, 128, tile_h=48, tile_w=64, overlap_h=16, overlap_w=16)
    stats.record_plan(plan, latent_t=37, spectrum_prior=True, keyframe_count=2)
    report = stats.to_dict()
    assert report["prior_calls"] == 1
    assert report["tile_model_calls"] == 6
    assert report["spectrum_prior_calls"] == 1
    assert report["last_full_video_tokens"] == 85248
    assert report["last_tile_video_tokens"] == 28416
    assert report["last_keyframe_count"] == 2


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
