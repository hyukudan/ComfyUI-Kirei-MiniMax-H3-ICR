import torch

from h3_icr.tiling import (
    WeightedTileAccumulator,
    plan_spatial_tiles,
    select_global_video_positions,
    target_patch_indices,
    tile_blend_weight,
)


def test_tile_plan_covers_target_with_patch_aligned_fixed_shapes():
    plan = plan_spatial_tiles(
        72,
        128,
        tile_h=48,
        tile_w=64,
        overlap_h=16,
        overlap_w=16,
    )
    assert plan.rows == 2
    assert plan.cols == 3
    assert len(plan.tiles) == 6
    coverage = torch.zeros(72, 128, dtype=torch.bool)
    for tile in plan.tiles:
        assert tile.height == 48
        assert tile.width == 64
        assert tile.y0 % 2 == tile.x0 % 2 == 0
        coverage[tile.y0 : tile.y1, tile.x0 : tile.x1] = True
    assert bool(coverage.all())


def test_weighted_tiles_reconstruct_constant_without_seams():
    plan = plan_spatial_tiles(48, 80, tile_h=32, tile_w=48, overlap_h=8, overlap_w=8)
    accumulator = WeightedTileAccumulator((1, 2, 3, 48, 80), device="cpu", dtype=torch.float32)
    for tile in plan.tiles:
        prediction = torch.ones(1, 2, 3, tile.height, tile.width)
        accumulator.add(tile, prediction, tile_blend_weight(tile, device="cpu", dtype=torch.float32))
    result = accumulator.finalize()
    assert torch.allclose(result, torch.ones_like(result), atol=1e-6)


def test_prior_regularization_is_closed_form_for_single_tile():
    plan = plan_spatial_tiles(32, 48, tile_h=32, tile_w=48, overlap_h=0, overlap_w=0)
    tile = plan.tiles[0]
    accumulator = WeightedTileAccumulator((1, 1, 1, 32, 48), device="cpu", dtype=torch.float32)
    accumulator.add(tile, torch.full((1, 1, 1, 32, 48), 2.0), torch.ones(32, 48))
    prior = torch.zeros(1, 1, 1, 32, 48)
    result = accumulator.finalize(prior=prior, prior_strength=1.0)
    assert torch.allclose(result, torch.ones_like(result))


def test_target_patch_indices_preserve_frame_major_global_order():
    plan = plan_spatial_tiles(8, 12, tile_h=4, tile_w=6, overlap_h=0, overlap_w=0)
    tile = plan.tiles[-1]
    indices = target_patch_indices(2, 8, 12, tile)
    expected_frame0 = torch.tensor([15, 16, 17, 21, 22, 23])
    expected = torch.cat([expected_frame0, expected_frame0 + 24])
    assert torch.equal(indices, expected)


def test_global_position_selection_uses_full_canvas_coordinates():
    full_positions = torch.arange(48 * 3, dtype=torch.float64).reshape(48, 3)
    tile = plan_spatial_tiles(8, 12, tile_h=4, tile_w=6, overlap_h=0, overlap_w=0).tiles[-1]
    selected = select_global_video_positions(
        full_positions,
        latent_t=2,
        full_h=8,
        full_w=12,
        tile=tile,
    )
    indices = target_patch_indices(2, 8, 12, tile)
    assert torch.equal(selected, full_positions[indices])


def test_invalid_unaligned_geometry_fails_closed():
    try:
        plan_spatial_tiles(72, 128, tile_h=47, tile_w=64, overlap_h=16, overlap_w=16)
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("unaligned tile geometry must fail")
