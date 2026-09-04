from __future__ import annotations

import json

from .validation import parse_json_object
from .validation_metrics import (
    build_validation_result_bundle,
    compare_validation_result_bundles,
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


class H3ICRCompareValidationBundles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle_a": ("H3_ICR_VALIDATION_BUNDLE",),
                "bundle_b": ("H3_ICR_VALIDATION_BUNDLE",),
                "allowed_differences": (
                    "STRING",
                    {"multiline": True, "default": "arm.settings"},
                ),
                "fail_on_unexpected_difference": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("comparable", "comparison_json")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Validation"
    DESCRIPTION = (
        "Compare two complete validation bundles. Their manifests must satisfy the same strict A/B rules before metric "
        "deltas are interpreted. Shared scalar latent metrics are reported as B-A deltas with conservative direction "
        "hints; no automatic global winner is assigned."
    )

    def compare(self, bundle_a, bundle_b, allowed_differences, fail_on_unexpected_difference):
        report = compare_validation_result_bundles(
            bundle_a,
            bundle_b,
            allowed_differences=allowed_differences,
        )
        if fail_on_unexpected_difference and not report["comparable"]:
            unexpected = report.get("manifest_comparison", {}).get("unexpected_differences", [])
            paths = [row.get("path", "?") for row in unexpected]
            preview = ", ".join(paths[:12])
            if len(paths) > 12:
                preview += f", ... (+{len(paths) - 12})"
            raise ValueError(f"H3-ICR bundle comparison is not controlled: {preview}")
        return bool(report["comparable"]), json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)


NODE_CLASS_MAPPINGS = {
    "H3ICRLatentValidationMetrics": H3ICRLatentValidationMetrics,
    "H3ICRValidationResultBundle": H3ICRValidationResultBundle,
    "H3ICRCompareValidationBundles": H3ICRCompareValidationBundles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRLatentValidationMetrics": "Kirei H3 ICR Latent Validation Metrics",
    "H3ICRValidationResultBundle": "Kirei H3 ICR Validation Result Bundle",
    "H3ICRCompareValidationBundles": "Kirei H3 ICR Compare Validation Bundles",
}
