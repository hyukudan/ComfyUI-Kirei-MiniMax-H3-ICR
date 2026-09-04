import torch

from h3_icr.tiled_nodes import H3ICRTiled2KPatch


class FakeModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}

    def clone(self):
        clone = FakeModel()
        clone.model_options = {
            "transformer_options": dict(self.model_options.get("transformer_options", {}))
        }
        return clone


def test_tiled_node_converts_pixel_geometry_to_h3_latent_geometry():
    video = torch.zeros(1, 24, 2, 48, 84)
    audio = torch.zeros(1, 32, 2, 10)
    model, renderer = H3ICRTiled2KPatch().patch(
        FakeModel(),
        {"samples": (video, audio)},
        1024,
        768,
        256,
        256,
        0.3,
        16,
    )
    config = renderer["config"]
    assert config.prior_h == 48
    assert config.prior_w == 84
    assert config.tile_h == 48
    assert config.tile_w == 64
    assert config.overlap_h == 16
    assert config.overlap_w == 16
    assert "h3_icr_tiled_renderer" in model.model_options["transformer_options"]


def test_tiled_node_rejects_non_32_pixel_geometry():
    video = torch.zeros(1, 24, 2, 48, 84)
    audio = torch.zeros(1, 32, 2, 10)
    try:
        H3ICRTiled2KPatch().patch(
            FakeModel(),
            {"samples": (video, audio)},
            1000,
            768,
            256,
            256,
            0.3,
            16,
        )
    except ValueError as exc:
        assert "multiple of 32" in str(exc)
    else:
        raise AssertionError("pixel geometry must be 32-aligned")
