from __future__ import annotations

import json

from .contracts import unwrap_av
from .prior_schedule import patch_tiled_prior_schedule
from .runtime_tiling import patch_tiled_renderer


class H3ICRTiled2KPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_latent": ("LATENT",),
                "tile_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "tile_height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 32}),
                "overlap_width": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 32}),
                "overlap_height": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 32}),
                "prior_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 4.0, "step": 0.01}),
                "max_tiles": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_TILED_RENDERER")
    RETURN_NAMES = ("model", "renderer")
    FUNCTION = "patch"
    CATEGORY = "Kirei/MiniMax H3/ICR"
    DESCRIPTION = (
        "Patch native MiniMax H3 with the experimental M4 global-LR + tiled-HR renderer. "
        "Predictions are fused in model-output space every evaluation while target and HR-keyframe "
        "MM-RoPE coordinates remain those of the full canvas. EasyCache is intentionally rejected."
    )

    def patch(
        self,
        model,
        base_latent,
        tile_width,
        tile_height,
        overlap_width,
        overlap_height,
        prior_strength,
        max_tiles,
    ):
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing samples")
        base_video, _ = unwrap_av(base_latent["samples"])
        for value, name in (
            (tile_width, "tile_width"),
            (tile_height, "tile_height"),
            (overlap_width, "overlap_width"),
            (overlap_height, "overlap_height"),
        ):
            if int(value) % 32:
                raise ValueError(f"{name} must be a multiple of 32 pixels for the H3 patch grid")
        if overlap_width >= tile_width or overlap_height >= tile_height:
            raise ValueError("tile overlap must be smaller than the corresponding tile dimension")

        patched, config, stats = patch_tiled_renderer(
            model,
            base_video,
            tile_h=int(tile_height) // 16,
            tile_w=int(tile_width) // 16,
            overlap_h=int(overlap_height) // 16,
            overlap_w=int(overlap_width) // 16,
            prior_strength=float(prior_strength),
            max_tiles=int(max_tiles),
        )
        renderer = {"api": 1, "config": config, "stats": stats}
        return patched, renderer


class H3ICRTiledPriorSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "renderer": ("H3_ICR_TILED_RENDERER",),
                "floor": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_TILED_RENDERER")
    RETURN_NAMES = ("model", "renderer")
    FUNCTION = "patch"
    CATEGORY = "Kirei/MiniMax H3/ICR"
    DESCRIPTION = (
        "Make the M4 global-prior regularization structure-first: full configured strength near the "
        "start of the H3 second pass, decaying toward a configurable floor as sigma approaches zero."
    )

    def patch(self, model, renderer, floor, power):
        patched, config, stats = patch_tiled_prior_schedule(
            model,
            floor=float(floor),
            power=float(power),
        )
        out = dict(renderer)
        out["prior_schedule"] = config
        out["prior_schedule_stats"] = stats
        return patched, out


class H3ICRTiled2KReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"renderer": ("H3_ICR_TILED_RENDERER",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR"

    def render(self, renderer):
        config = renderer.get("config")
        stats = renderer.get("stats")
        prior_schedule = renderer.get("prior_schedule")
        prior_schedule_stats = renderer.get("prior_schedule_stats")
        payload = {
            "api": int(renderer.get("api", 0)),
            "config": config.to_dict() if hasattr(config, "to_dict") else None,
            "stats": stats.to_dict() if hasattr(stats, "to_dict") else None,
            "prior_schedule": (
                prior_schedule.to_dict() if hasattr(prior_schedule, "to_dict") else None
            ),
            "prior_schedule_stats": (
                prior_schedule_stats.to_dict() if hasattr(prior_schedule_stats, "to_dict") else None
            ),
        }
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),)


NODE_CLASS_MAPPINGS = {
    "H3ICRTiled2KPatch": H3ICRTiled2KPatch,
    "H3ICRTiledPriorSchedule": H3ICRTiledPriorSchedule,
    "H3ICRTiled2KReport": H3ICRTiled2KReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRTiled2KPatch": "Kirei H3 ICR Tiled 2K Patch [Experimental]",
    "H3ICRTiledPriorSchedule": "Kirei H3 ICR Tiled Prior Schedule [Experimental]",
    "H3ICRTiled2KReport": "Kirei H3 ICR Tiled 2K Report",
}
