from __future__ import annotations

import json

from .contracts import unwrap_av
from .posterior_consistency import patch_posterior_consistency


class H3ICRPosteriorConsistency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 2.0, "step": 0.01}),
                "apply_every": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "max_correction_rms_ratio": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.005},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_POSTERIOR")
    RETURN_NAMES = ("model", "posterior")
    FUNCTION = "patch"
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"
    DESCRIPTION = (
        "Experimental latent measurement-consistency patch. It backpropagates only through spatial "
        "area-downsampling of the predicted-clean HR video so D(x0_HR) stays close to the clean H3 Base "
        "latent. No H3/VAE gradient is used, audio is untouched, and the correction is RMS-capped."
    )

    def patch(self, model, base_latent, strength, apply_every, max_correction_rms_ratio):
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing H3 AV samples")
        base_video, _ = unwrap_av(base_latent["samples"])
        patched, config, stats = patch_posterior_consistency(
            model,
            base_video,
            strength=float(strength),
            apply_every=int(apply_every),
            max_correction_rms_ratio=float(max_correction_rms_ratio),
        )
        return patched, {"api": 1, "config": config, "stats": stats}


class H3ICRPosteriorConsistencyReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"posterior": ("H3_ICR_POSTERIOR",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"

    def render(self, posterior):
        if not isinstance(posterior, dict):
            raise TypeError("invalid H3 ICR posterior-consistency handle")
        config = posterior.get("config")
        stats = posterior.get("stats")
        payload = {
            "api": int(posterior.get("api", 0)),
            "config": config.to_dict() if hasattr(config, "to_dict") else None,
            "stats": stats.to_dict() if hasattr(stats, "to_dict") else None,
        }
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),)


NODE_CLASS_MAPPINGS = {
    "H3ICRPosteriorConsistency": H3ICRPosteriorConsistency,
    "H3ICRPosteriorConsistencyReport": H3ICRPosteriorConsistencyReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRPosteriorConsistency": "Kirei H3 ICR Posterior Consistency [Experimental]",
    "H3ICRPosteriorConsistencyReport": "Kirei H3 ICR Posterior Consistency Report",
}
