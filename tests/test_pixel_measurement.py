import torch
import torch.nn.functional as F

from h3_icr.pixel_measurement import (
    PixelMeasurementConfig,
    build_reference_measurement,
    patch_pixel_measurement_consistency,
    pixel_measurement_step,
    validate_h3_pixel_vae,
)


class FakeH3Decoder:
    is_h3 = True
    latent_channels = 24

    def decode(self, z):
        # Small differentiable proxy: use the first RGB-like latent channels and
        # enlarge spatially. Shape remains B,C,T,H,W.
        rgb = z[:, :3]
        b, c, t, h, w = rgb.shape
        flat = rgb.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        out = F.interpolate(flat, scale_factor=2.0, mode="bilinear", align_corners=False)
        return out.reshape(b, t, c, h * 2, w * 2).permute(0, 2, 1, 3, 4).contiguous()


class FakeVAE:
    latent_channels = 24
    vae_dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.first_stage_model = FakeH3Decoder()
        self.process_output = lambda value: value

    def prepare_decode(self, shape):
        return 1


class FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        clone = FakeModel()
        clone.model_options = dict(self.model_options)
        return clone

    def set_model_sampler_post_cfg_function(self, fn, disable_cfg1_optimization=False):
        del disable_cfg1_optimization
        self.model_options["sampler_post_cfg_function"] = self.model_options.get(
            "sampler_post_cfg_function", []
        ) + [fn]


def _base_and_high():
    torch.manual_seed(7)
    base = torch.randn(1, 24, 3, 4, 6) * 0.1
    high = F.interpolate(
        base.permute(0, 2, 1, 3, 4).reshape(3, 24, 4, 6),
        size=(8, 12),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 3, 24, 8, 12).permute(0, 2, 1, 3, 4).contiguous()
    high = high + 0.15
    return base, high


def test_proxy_vae_contract_accepts_h3_24_channel_decoder():
    assert validate_h3_pixel_vae(FakeVAE(), allow_full_vae=False) == "h3_tae_proxy"


def test_pixel_measurement_step_reduces_verified_pixel_error():
    vae = FakeVAE()
    base, high = _base_and_high()
    config = PixelMeasurementConfig(
        strength=0.25,
        apply_every=1,
        max_correction_rms_ratio=0.20,
        measurement_max_side=128,
        frame_stride=1,
        edge_weight=0.0,
        temporal_weight=0.0,
        verify_after=True,
    )
    reference = build_reference_measurement(vae, base, config)
    corrected, summary = pixel_measurement_step(high, base, reference, vae, config)

    assert corrected.shape == high.shape
    assert summary["verified_after"] == 1
    assert summary["pixel_rmse_after"] < summary["pixel_rmse"]
    assert summary["decoder_calls"] == 2
    assert summary["correction_rms_ratio"] <= config.max_correction_rms_ratio + 1e-5


def test_pixel_measurement_respects_zero_strength():
    vae = FakeVAE()
    base, high = _base_and_high()
    config = PixelMeasurementConfig(strength=0.0, measurement_max_side=128, frame_stride=1)
    reference = build_reference_measurement(vae, base, config)
    corrected, summary = pixel_measurement_step(high, base, reference, vae, config)
    assert torch.equal(corrected, high)
    assert summary["decoder_calls"] == 0


def test_post_cfg_hook_preserves_audio_exactly():
    vae = FakeVAE()
    base, high = _base_and_high()
    audio = torch.randn(1, 32, 2, 5)
    model, _config, stats, decoder_kind = patch_pixel_measurement_consistency(
        FakeModel(),
        base,
        vae,
        strength=0.10,
        apply_every=1,
        max_correction_rms_ratio=0.10,
        measurement_max_side=128,
        frame_stride=1,
        edge_weight=0.0,
        temporal_weight=0.0,
        verify_after=False,
    )
    hook = model.model_options["sampler_post_cfg_function"][-1]
    result = hook({"denoised": (high, audio)})
    assert isinstance(result, tuple)
    assert torch.equal(result[1], audio)
    assert result[0].shape == high.shape
    assert stats.applied == 1
    assert decoder_kind == "h3_tae_proxy"
