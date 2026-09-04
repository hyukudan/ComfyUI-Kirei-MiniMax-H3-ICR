import torch

from h3_icr.prior_schedule import (
    PRIOR_SCHEDULE_WRAPPER_KEY,
    PriorScheduleConfig,
    PriorScheduleStats,
    patch_tiled_prior_schedule,
    scheduled_prior_strength,
    tiled_prior_schedule_wrapper,
)
from h3_icr.runtime_tiling import H3TiledRendererConfig, TILED_CONFIG_KEY


def test_prior_schedule_is_full_at_start_and_decays_to_floor():
    config = PriorScheduleConfig(floor=0.20, power=1.0)
    stats = PriorScheduleStats()
    options = {"sample_sigmas": torch.tensor([0.6, 0.3, 0.0])}

    start = scheduled_prior_strength(
        0.30,
        timestep=torch.tensor([600.0]),
        transformer_options=options,
        config=config,
        stats=stats,
    )
    mid = scheduled_prior_strength(
        0.30,
        timestep=torch.tensor([300.0]),
        transformer_options=options,
        config=config,
        stats=stats,
    )
    end = scheduled_prior_strength(
        0.30,
        timestep=torch.tensor([0.0]),
        transformer_options=options,
        config=config,
        stats=stats,
    )

    assert abs(start[0] - 0.30) < 1e-7
    assert abs(mid[0] - 0.18) < 1e-7
    assert abs(end[0] - 0.06) < 1e-7


def test_prior_schedule_uses_first_observed_sigma_when_sampler_schedule_is_unavailable():
    config = PriorScheduleConfig(floor=0.0, power=1.0)
    stats = PriorScheduleStats()

    first = scheduled_prior_strength(
        0.4,
        timestep=torch.tensor([500.0]),
        transformer_options={},
        config=config,
        stats=stats,
    )
    stats.record(
        sigma_start=first[1],
        sigma=first[2],
        multiplier=first[3],
        effective_strength=first[0],
    )
    second = scheduled_prior_strength(
        0.4,
        timestep=torch.tensor([250.0]),
        transformer_options={},
        config=config,
        stats=stats,
    )

    assert abs(first[0] - 0.4) < 1e-7
    assert abs(second[0] - 0.2) < 1e-7
    assert abs(second[1] - 0.5) < 1e-7


def test_schedule_wrapper_passes_effective_tiled_config_to_inner_renderer():
    base = H3TiledRendererConfig(
        prior_h=48,
        prior_w=84,
        tile_h=48,
        tile_w=64,
        overlap_h=16,
        overlap_w=16,
        prior_strength=0.30,
    )
    config = PriorScheduleConfig(floor=0.20, power=1.0)
    stats = PriorScheduleStats()
    options = {
        TILED_CONFIG_KEY: base,
        "h3_icr_tiled_prior_schedule": config,
        "h3_icr_tiled_prior_schedule_stats": stats,
        "sample_sigmas": torch.tensor([0.6, 0.3, 0.0]),
    }

    class Executor:
        def __init__(self):
            self.strength = None

        def __call__(self, x, timestep, context, child_options, minimax_payload=None, **kwargs):
            self.strength = child_options[TILED_CONFIG_KEY].prior_strength
            return "ok"

    executor = Executor()
    result = tiled_prior_schedule_wrapper(
        executor,
        object(),
        torch.tensor([300.0]),
        object(),
        options,
        minimax_payload={},
    )
    assert result == "ok"
    assert abs(executor.strength - 0.18) < 1e-7
    assert stats.calls == 1
    assert abs(stats.last_multiplier - 0.60) < 1e-7


def test_prior_schedule_patch_requires_tiled_renderer_and_installs_outer_wrapper():
    class FakeModel:
        def __init__(self, options=None):
            self.model_options = options or {"transformer_options": {}}

        def clone(self):
            return FakeModel(
                {"transformer_options": dict(self.model_options.get("transformer_options", {}))}
            )

    try:
        patch_tiled_prior_schedule(FakeModel())
    except ValueError as exc:
        assert "Tiled 2K Patch" in str(exc)
    else:
        raise AssertionError("schedule patch must require an existing tiled renderer")

    tiled = H3TiledRendererConfig(
        prior_h=48,
        prior_w=84,
        tile_h=48,
        tile_w=64,
        overlap_h=16,
        overlap_w=16,
    )
    model = FakeModel({"transformer_options": {TILED_CONFIG_KEY: tiled}})
    patched, config, stats = patch_tiled_prior_schedule(model, floor=0.1, power=2.0)
    transformer = patched.model_options["transformer_options"]
    diffusion = transformer["wrappers"]["diffusion_model"]

    assert next(iter(diffusion)) == PRIOR_SCHEDULE_WRAPPER_KEY
    assert config.floor == 0.1
    assert config.power == 2.0
    assert stats.calls == 0
