from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SpatialTile:
    index: int
    y0: int
    y1: int
    x0: int
    x1: int
    overlap_top: int = 0
    overlap_bottom: int = 0
    overlap_left: int = 0
    overlap_right: int = 0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass(frozen=True, slots=True)
class TilingPlan:
    latent_h: int
    latent_w: int
    tile_h: int
    tile_w: int
    requested_overlap_h: int
    requested_overlap_w: int
    patch_h: int
    patch_w: int
    tiles: tuple[SpatialTile, ...]

    @property
    def rows(self) -> int:
        return len(sorted({tile.y0 for tile in self.tiles}))

    @property
    def cols(self) -> int:
        return len(sorted({tile.x0 for tile in self.tiles}))

    @property
    def max_actual_overlap_h(self) -> int:
        return max((max(tile.overlap_top, tile.overlap_bottom) for tile in self.tiles), default=0)

    @property
    def max_actual_overlap_w(self) -> int:
        return max((max(tile.overlap_left, tile.overlap_right) for tile in self.tiles), default=0)

    def to_dict(self) -> dict[str, int]:
        return {
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
            "rows": self.rows,
            "cols": self.cols,
            "tile_count": len(self.tiles),
            "requested_overlap_h": self.requested_overlap_h,
            "requested_overlap_w": self.requested_overlap_w,
            "max_actual_overlap_h": self.max_actual_overlap_h,
            "max_actual_overlap_w": self.max_actual_overlap_w,
        }


def _validate_axis(size: int, tile: int, overlap: int, patch: int) -> None:
    if min(size, tile, patch) <= 0:
        raise ValueError("size, tile and patch must be positive")
    if size % patch or tile % patch or overlap % patch:
        raise ValueError("size, tile and overlap must align to the H3 patch grid")
    if tile > size:
        raise ValueError("tile cannot be larger than the target latent axis")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")


def _axis_starts(size: int, tile: int, overlap: int, patch: int) -> tuple[int, ...]:
    _validate_axis(size, tile, overlap, patch)
    if tile == size:
        return (0,)

    max_step = tile - overlap
    distance = size - tile
    intervals = max(1, math.ceil(distance / max_step))
    raw = [distance * index / intervals for index in range(intervals + 1)]
    starts = [int(round(value / patch)) * patch for value in raw]
    starts[0] = 0
    starts[-1] = distance

    deduped: list[int] = []
    for start in starts:
        start = min(max(0, start), distance)
        if not deduped or start != deduped[-1]:
            deduped.append(start)
    if deduped[-1] != distance:
        deduped.append(distance)
    if any((b - a) > max_step for a, b in zip(deduped, deduped[1:])):
        raise RuntimeError("aligned tile planner exceeded the requested maximum stride")
    return tuple(deduped)


def _neighbor_overlaps(starts: tuple[int, ...], tile: int) -> tuple[tuple[int, int], ...]:
    result = []
    for index, start in enumerate(starts):
        prev_overlap = 0 if index == 0 else starts[index - 1] + tile - start
        next_overlap = 0 if index == len(starts) - 1 else start + tile - starts[index + 1]
        result.append((max(0, prev_overlap), max(0, next_overlap)))
    return tuple(result)


def plan_spatial_tiles(
    latent_h: int,
    latent_w: int,
    *,
    tile_h: int,
    tile_w: int,
    overlap_h: int,
    overlap_w: int,
    patch_h: int = 2,
    patch_w: int = 2,
) -> TilingPlan:
    ys = _axis_starts(latent_h, tile_h, overlap_h, patch_h)
    xs = _axis_starts(latent_w, tile_w, overlap_w, patch_w)
    y_overlap = _neighbor_overlaps(ys, tile_h)
    x_overlap = _neighbor_overlaps(xs, tile_w)

    tiles: list[SpatialTile] = []
    index = 0
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            tiles.append(
                SpatialTile(
                    index=index,
                    y0=y0,
                    y1=y0 + tile_h,
                    x0=x0,
                    x1=x0 + tile_w,
                    overlap_top=y_overlap[yi][0],
                    overlap_bottom=y_overlap[yi][1],
                    overlap_left=x_overlap[xi][0],
                    overlap_right=x_overlap[xi][1],
                )
            )
            index += 1
    return TilingPlan(
        latent_h=latent_h,
        latent_w=latent_w,
        tile_h=tile_h,
        tile_w=tile_w,
        requested_overlap_h=overlap_h,
        requested_overlap_w=overlap_w,
        patch_h=patch_h,
        patch_w=patch_w,
        tiles=tuple(tiles),
    )


def _raised_cosine_ramp(length: int, *, rising: bool, device, dtype) -> torch.Tensor:
    if length <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    phase = (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length
    ramp = torch.sin(phase * (math.pi / 2.0)).square()
    if not rising:
        ramp = torch.flip(ramp, dims=(0,))
    return ramp.to(dtype=dtype)


def tile_blend_weight(tile: SpatialTile, *, device, dtype) -> torch.Tensor:
    wy = torch.ones(tile.height, device=device, dtype=dtype)
    wx = torch.ones(tile.width, device=device, dtype=dtype)
    if tile.overlap_top:
        wy[: tile.overlap_top] = _raised_cosine_ramp(
            tile.overlap_top, rising=True, device=device, dtype=dtype
        )
    if tile.overlap_bottom:
        wy[-tile.overlap_bottom :] = torch.minimum(
            wy[-tile.overlap_bottom :],
            _raised_cosine_ramp(tile.overlap_bottom, rising=False, device=device, dtype=dtype),
        )
    if tile.overlap_left:
        wx[: tile.overlap_left] = _raised_cosine_ramp(
            tile.overlap_left, rising=True, device=device, dtype=dtype
        )
    if tile.overlap_right:
        wx[-tile.overlap_right :] = torch.minimum(
            wx[-tile.overlap_right :],
            _raised_cosine_ramp(tile.overlap_right, rising=False, device=device, dtype=dtype),
        )
    return wy[:, None] * wx[None, :]


class WeightedTileAccumulator:
    """Streaming weighted least-squares accumulator in model-output space."""

    def __init__(self, shape: tuple[int, ...], *, device, dtype):
        if len(shape) != 5:
            raise ValueError("H3 video output shape must be BxCxTxHxW")
        self.numerator = torch.zeros(shape, device=device, dtype=dtype)
        self.denominator = torch.zeros((1, 1, 1, shape[-2], shape[-1]), device=device, dtype=dtype)
        self.tiles_added = 0

    def add(self, tile: SpatialTile, prediction: torch.Tensor, weight: torch.Tensor) -> None:
        expected = self.numerator[..., tile.y0 : tile.y1, tile.x0 : tile.x1]
        if prediction.shape != expected.shape:
            raise ValueError(
                f"tile prediction shape {tuple(prediction.shape)} does not match {tuple(expected.shape)}"
            )
        if weight.shape != (tile.height, tile.width):
            raise ValueError("tile weight has the wrong spatial shape")
        w = weight.to(device=prediction.device, dtype=prediction.dtype)[None, None, None]
        self.numerator[..., tile.y0 : tile.y1, tile.x0 : tile.x1].add_(prediction * w)
        self.denominator[..., tile.y0 : tile.y1, tile.x0 : tile.x1].add_(w)
        self.tiles_added += 1

    def finalize(
        self,
        *,
        prior: torch.Tensor | None = None,
        prior_strength: float = 0.0,
        regularizer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.tiles_added == 0:
            raise RuntimeError("cannot finalize tiled fusion before adding predictions")
        if bool((self.denominator <= 0).any().item()):
            raise RuntimeError("tile plan left uncovered target positions")

        numerator = self.numerator
        denominator = self.denominator
        if prior is not None and prior_strength > 0.0:
            if prior.shape != numerator.shape:
                raise ValueError("global prior must match the full target model-output shape")
            if regularizer is None:
                lam = torch.as_tensor(prior_strength, device=prior.device, dtype=prior.dtype)
            else:
                if regularizer.ndim == 2:
                    regularizer = regularizer[None, None, None]
                if regularizer.ndim != 5 or regularizer.shape[-2:] != numerator.shape[-2:]:
                    raise ValueError("regularizer must be HxW or broadcastable 5D spatial weights")
                lam = regularizer.to(device=prior.device, dtype=prior.dtype) * float(prior_strength)
            numerator = numerator + prior * lam
            denominator = denominator + lam
        return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)


def h3_video_token_count(
    latent_t: int,
    latent_h: int,
    latent_w: int,
    *,
    patch_h: int = 2,
    patch_w: int = 2,
) -> int:
    if min(latent_t, latent_h, latent_w, patch_h, patch_w) <= 0:
        raise ValueError("token-count geometry must be positive")
    if latent_h % patch_h or latent_w % patch_w:
        raise ValueError("token-count geometry must align to the H3 patch grid")
    return latent_t * (latent_h // patch_h) * (latent_w // patch_w)


def target_patch_indices(
    latent_t: int,
    full_h: int,
    full_w: int,
    tile: SpatialTile,
    *,
    patch_h: int = 2,
    patch_w: int = 2,
) -> torch.Tensor:
    if latent_t <= 0:
        raise ValueError("latent_t must be positive")
    if full_h % patch_h or full_w % patch_w:
        raise ValueError("full latent geometry must align to the H3 patch grid")
    if any(value % patch_h for value in (tile.y0, tile.y1)):
        raise ValueError("tile Y bounds must align to patch_h")
    if any(value % patch_w for value in (tile.x0, tile.x1)):
        raise ValueError("tile X bounds must align to patch_w")

    grid_h = full_h // patch_h
    grid_w = full_w // patch_w
    y0, y1 = tile.y0 // patch_h, tile.y1 // patch_h
    x0, x1 = tile.x0 // patch_w, tile.x1 // patch_w
    spatial = (
        torch.arange(y0, y1, dtype=torch.long)[:, None] * grid_w
        + torch.arange(x0, x1, dtype=torch.long)[None, :]
    ).reshape(-1)
    frame_rows = grid_h * grid_w
    return (torch.arange(latent_t, dtype=torch.long)[:, None] * frame_rows + spatial[None, :]).reshape(-1)


def select_global_video_positions(
    full_video_positions: torch.Tensor,
    *,
    latent_t: int,
    full_h: int,
    full_w: int,
    tile: SpatialTile,
    patch_h: int = 2,
    patch_w: int = 2,
) -> torch.Tensor:
    indices = target_patch_indices(
        latent_t,
        full_h,
        full_w,
        tile,
        patch_h=patch_h,
        patch_w=patch_w,
    )
    if full_video_positions.ndim != 2 or full_video_positions.shape[1] != 3:
        raise ValueError("full video MM-RoPE positions must be [rows, 3]")
    if indices.numel() and int(indices.max()) >= full_video_positions.shape[0]:
        raise ValueError("global position selection exceeds the full target video segment")
    return full_video_positions.index_select(0, indices.to(full_video_positions.device))
