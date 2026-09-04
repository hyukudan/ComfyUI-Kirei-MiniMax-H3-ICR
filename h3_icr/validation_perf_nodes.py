from __future__ import annotations

import json

from .validation_perf import patch_sampler_performance


class H3ICRValidationPerformancePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "H3_ICR_VALIDATION_PERFORMANCE")
    RETURN_NAMES = ("model", "performance")
    FUNCTION = "patch"
    CATEGORY = "Kirei/MiniMax H3/ICR/Validation"
    DESCRIPTION = (
        "Passively wrap the complete ComfyUI sampler call to record wall time and, when CUDA is active, peak allocated "
        "and reserved VRAM. The wrapper delegates the sampler unchanged and records first-use separately from later calls."
    )

    def patch(self, model):
        patched, stats = patch_sampler_performance(model)
        return patched, {"api": 1, "stats": stats}


class H3ICRValidationPerformanceReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"performance": ("H3_ICR_VALIDATION_PERFORMANCE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Validation"

    def render(self, performance):
        stats = performance.get("stats") if isinstance(performance, dict) else None
        if stats is None or not hasattr(stats, "to_dict"):
            raise TypeError("invalid H3 ICR validation performance handle")
        payload = stats.to_dict()
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),)


NODE_CLASS_MAPPINGS = {
    "H3ICRValidationPerformancePatch": H3ICRValidationPerformancePatch,
    "H3ICRValidationPerformanceReport": H3ICRValidationPerformanceReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRValidationPerformancePatch": "Kirei H3 ICR Validation Performance Patch",
    "H3ICRValidationPerformanceReport": "Kirei H3 ICR Validation Performance Report",
}
