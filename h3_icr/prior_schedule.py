from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .runtime_tiling import H3TiledRendererConfig, TILED_CONFIG_KEY

PRIOR_SCHEDULE_CONFIG_KEY = "h3_icr_tiled_prior_schedule"
PRIOR_SCHEDULE_STATS_KEY = "h3_icr_tiled_prior_schedule_stats"
PRIOR_SCHEDULE_WRAPPER_KEY = "h3_icr_tiled_prior_schedule"


@dataclass(frozen=True, slots=True)
class PriorScheduleConfig:
    floor: float = 0.15
    power: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("prior schedule floor must be in [0, 1]")
        if self.power < 0.0:
            raise ValueError("prior schedule power must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {"floor": self.floor, "power": self.power}


@dataclass(slots=True)
class PriorScheduleStats:
    calls: int = 0
    sigma_start: float = 0.0
    last_sigma: float = 0.0
    last_multiplier: float = 1.0
    last_effective_strength: float = 0.0
    min_effective_strength: float = 0.0
    max_effective_strength: float = 0.0

    def record(self, *, sigma_start: float, sigma: float, multiplier: float, effective_strength: float) -> None:
        self.calls += 1
        self.sigma_start = float(sigma_start)
        self.last_sigma = float(sigma)
        self.last_multiplier = float(multiplier)
        self.last_effective_strength = float(effective_strength)
        if self.calls == 1:
            self.min_effective_strength = float(effective_strength)
            self.max_effective_strength = float(effective_strength)
        else:
            self.min_effective_strength = min(self.min_effective_strength, float(effective_strength))
            self.max_effective_strength = max(self.max_effective_strength, float(effective_strength))

    def to_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "sigma_start": self.sigma_start,
            "last_sigma": self.last_sigma,
            "last_multiplier": self.last_multiplier,
            "last_effective_strength": self.last_effective_strength,
            "min_effective_strength": self.min_effective_strength,
            "max_effective_strength": self.max_effective_strength,
        }


def _current_video_sigma(timestep: Any) -> float:
    if torch.is_tensor(timestep):
        if timestep.numel() == 0:
            raise ValueError("H3 tiled prior schedule received an empty timestep tensor")
        sigma = float(timestep.detach().reshape(-1)[0].cpu().item()) / 1000.0
    else:
        sigma = float(timestep) / 1000.0
    if not 0.0 <= sigma <= 1.0 + 1e-6:
        raise ValueError(f"H3 tiled prior schedule received invalid video sigma {sigma!r}")
    return max(0.0, min(1.0, sigma))


def _schedule_sigma_start(options: dict[str, Any], sigma: float, stats: PriorScheduleStats) -> float:
    for key in ("sample_sigmas", "sigmas"):
        values = options.get(key)
        if torch.is_tensor(values) and values.numel() > 0:
            finite = values.detach().to(torch.float32)
            finite = finite[torch.isfinite(finite)]
            if finite.numel() > 0:
                start = float(finite.max().cpu().item())
                if start > 0.0:
                    return max(start, sigma)
    if stats.sigma_start > 0.0:
        return max(stats.sigma_start, sigma)
    return max(sigma, 1e-8)


def scheduled_prior_strength(
    base_strength: float,
    *,
    timestep: Any,
    transformer_options: dict[str, Any],
    config: PriorScheduleConfig,
    stats: PriorScheduleStats,
) -> tuple[float, float, float, float]:
    sigma = _current_video_sigma(timestep)
    sigma_start = _schedule_sigma_start(transformer_options, sigma, stats)
    ratio = max(0.0, min(1.0, sigma / sigma_start)) if sigma_start > 0.0 else 0.0
    multiplier = config.floor + (1.0 - config.floor) * (ratio ** config.power)
    effective = float(base_strength) * multiplier
    return effective, sigma_start, sigma, multiplier


def tiled_prior_schedule_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    options = transformer_options or {}
    tiled_config = options.get(TILED_CONFIG_KEY)
    schedule_config = options.get(PRIOR_SCHEDULE_CONFIG_KEY)
    stats = options.get(PRIOR_SCHEDULE_STATS_KEY)
    if (
        not isinstance(tiled_config, H3TiledRendererConfig)
        or not isinstance(schedule_config, PriorScheduleConfig)
        or not isinstance(stats, PriorScheduleStats)
    ):
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    effective, sigma_start, sigma, multiplier = scheduled_prior_strength(
        tiled_config.prior_strength,
        timestep=timestep,
        transformer_options=options,
        config=schedule_config,
        stats=stats,
    )
    stats.record(
        sigma_start=sigma_start,
        sigma=sigma,
        multiplier=multiplier,
        effective_strength=effective,
    )

    child_options = dict(options)
    child_options[TILED_CONFIG_KEY] = replace(tiled_config, prior_strength=effective)
    return executor(
        x,
        timestep,
        context,
        child_options,
        minimax_payload=minimax_payload,
        **kwargs,
    )


def _install_outermost_schedule_wrapper(transformer_options: dict[str, Any]) -> None:
    wrappers = dict(transformer_options.get("wrappers", {}))
    diffusion = dict(wrappers.get("diffusion_model", {}))
    diffusion.pop(PRIOR_SCHEDULE_WRAPPER_KEY, None)
    wrappers["diffusion_model"] = {PRIOR_SCHEDULE_WRAPPER_KEY: [tiled_prior_schedule_wrapper], **diffusion}
    transformer_options["wrappers"] = wrappers


def patch_tiled_prior_schedule(
    model: Any,
    *,
    floor: float = 0.15,
    power: float = 1.0,
) -> tuple[Any, PriorScheduleConfig, PriorScheduleStats]:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("H3 tiled prior schedule expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    if not isinstance(transformer.get(TILED_CONFIG_KEY), H3TiledRendererConfig):
        raise ValueError("Apply Kirei H3 ICR Tiled 2K Patch before the prior schedule patch")

    config = PriorScheduleConfig(floor=float(floor), power=float(power))
    stats = PriorScheduleStats()
    transformer[PRIOR_SCHEDULE_CONFIG_KEY] = config
    transformer[PRIOR_SCHEDULE_STATS_KEY] = stats
    _install_outermost_schedule_wrapper(transformer)
    options["transformer_options"] = transformer
    patched.model_options = options
    return patched, config, stats
