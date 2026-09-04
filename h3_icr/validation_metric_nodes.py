from __future__ import annotations

import json

from .validation import parse_json_object
from .validation_metrics import (
    build_validation_result_bundle,
    evaluate_latent_output,
)


class H3ICRLatentValidationMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_latent": ("LATENT",),
                "base_latent": ("LATENT",),
                "lowpass_cutoff": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "renderer": ("H3_ICR_TILED_RENDERER",),
            },
        }

    RETURN_TYPES = ("H3_ICR_VALIDATION_METRICS", "STRING")
    RETURN_NAMES = ("metrics", "metrics_json")
    FUNCTION = "evaluate"
    CATEGORY = "Kirei/MiniMax H3/ICR/Validation"
    DESCRIPTION = (
        "Compute deterministic latent-space validation metrics for a regenerated H3 result: Base-grid measurement "
        "compatibility, temporal drift, new HR detail energy/flicker, exact audio invariance and optional M4 tile-seam "
        "diagnostics. This is a triage tool, not a perceptual-quality oracle."
    )

    def evaluate(self, output_latent, base_latent, lowpass_cutoff, renderer=None):
        if not isinstance(output_latent, dict) or "samples" not in output_latent:
            raise TypeError("output_latent must be a ComfyUI LATENT containing H3 AV samples")
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing H3 AV samples")
        metrics = evaluate_latent_output(
            output_latent["samples"],
            base_latent["samples"],
            renderer=renderer,
            lowpass_cutoff=float(lowpass_cutoff),
        )
        return metrics, json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2)


class H3ICRValidationResultBundle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("H3_ICR_VALIDATION_MANIFEST",),
                "metrics": ("H3_ICR_VALIDATION_METRICS",),
                "reports_json": ("STRING", {"multiline": True, "default": "{}"}),
                "notes_json": ("STRING", {"multiline": True, "default": "{}"}),
            }
        }

    RETURN_TYPES = ("H3_ICR_VALIDATION_BUNDLE", "STRING", "STRING")
    RETURN_NAMES = ("bundle", "bundle_json", "bundle_id")
    FUNCTION = "bundle"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Validation"
    DESCRIPTION = (
        "Bind one canonical validation manifest to its deterministic latent metrics and optional runtime reports/notes. "
        "The bundle receives its own SHA-256 bundle_id while preserving the manifest run_id and metrics_id."
    )

    def bundle(self, manifest, metrics, reports_json, notes_json):
        reports = parse_json_object(reports_json, "reports")
        notes = parse_json_object(notes_json, "notes")
        bundle = build_validation_result_bundle(
            manifest,
            metrics,
            reports=reports,
            notes=notes,
        )
        rendered = json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2)
        return bundle, rendered, str(bundle["bundle_id"])


NODE_CLASS_MAPPINGS = {
    "H3ICRLatentValidationMetrics": H3ICRLatentValidationMetrics,
    "H3ICRValidationResultBundle": H3ICRValidationResultBundle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRLatentValidationMetrics": "Kirei H3 ICR Latent Validation Metrics",
    "H3ICRValidationResultBundle": "Kirei H3 ICR Validation Result Bundle",
}
