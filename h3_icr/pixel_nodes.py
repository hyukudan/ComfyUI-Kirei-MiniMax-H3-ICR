from __future__ import annotations

import json

from .contracts import unwrap_av
from .pixel_measurement import patch_pixel_measurement_consistency


class H3ICRPixelMeasurementConsistency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_latent": ("LATENT",),
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.005}),
                "apply_every": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "max_correction_rms_ratio": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.001, "max": 0.25, "step": 0.001},
                ),
                "measurement_max_side": ("INT", {"default": 384, "min": 64, "max": 1024, "step": 32}),
                "frame_stride": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "edge_weight": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 4.0, "step": 0.05}),
                "temporal_weight": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 4.0, "step": 0.05}),
                "verify_after": ("BOOLEAN", {"default": False}),
                "allow_full_vae": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_PIXEL_MEASUREMENT")
    RETURN_NAMES = ("model", "pixel_measurement")
    FUNCTION = "patch"
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"
    DESCRIPTION = (
        "Experimental M3c pixel-space measurement constraint. It downsamples the HR predicted-clean latent "
        "to Base latent geometry, decodes that LR probe with an H3-compatible 24-channel VAE/TAE, and "
        "backpropagates pixel/edge/temporal measurement error into the HR latent. A lightweight taeh3 proxy "
        "is recommended; full VisualVAE gradients are opt-in because they can be very expensive. Audio is untouched."
    )

    def patch(
        self,
        model,
        base_latent,
        vae,
        strength,
        apply_every,
        max_correction_rms_ratio,
        measurement_max_side,
        frame_stride,
        edge_weight,
        temporal_weight,
        verify_after,
        allow_full_vae,
    ):
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing H3 AV samples")
        base_video, _ = unwrap_av(base_latent["samples"])
        patched, config, stats, decoder_kind = patch_pixel_measurement_consistency(
            model,
            base_video,
            vae,
            strength=float(strength),
            apply_every=int(apply_every),
            max_correction_rms_ratio=float(max_correction_rms_ratio),
            measurement_max_side=int(measurement_max_side),
            frame_stride=int(frame_stride),
            edge_weight=float(edge_weight),
            temporal_weight=float(temporal_weight),
            verify_after=bool(verify_after),
            allow_full_vae=bool(allow_full_vae),
        )
        return patched, {
            "api": 1,
            "config": config,
            "stats": stats,
            "decoder_kind": decoder_kind,
        }


class H3ICRPixelMeasurementReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pixel_measurement": ("H3_ICR_PIXEL_MEASUREMENT",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"

    def render(self, pixel_measurement):
        if not isinstance(pixel_measurement, dict):
            raise TypeError("invalid H3 ICR pixel-measurement handle")
        config = pixel_measurement.get("config")
        stats = pixel_measurement.get("stats")
        payload = {
            "api": int(pixel_measurement.get("api", 0)),
            "decoder_kind": pixel_measurement.get("decoder_kind", "unknown"),
            "config": config.to_dict() if hasattr(config, "to_dict") else None,
            "stats": stats.to_dict() if hasattr(stats, "to_dict") else None,
        }
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),)


NODE_CLASS_MAPPINGS = {
    "H3ICRPixelMeasurementConsistency": H3ICRPixelMeasurementConsistency,
    "H3ICRPixelMeasurementReport": H3ICRPixelMeasurementReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRPixelMeasurementConsistency": "Kirei H3 ICR Pixel Measurement [M3c Experimental]",
    "H3ICRPixelMeasurementReport": "Kirei H3 ICR Pixel Measurement Report",
}
