from __future__ import annotations

import json

from .validation import (
    build_validation_manifest,
    compare_validation_manifests,
    parse_json_object,
)


class H3ICRValidationManifest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "base_latent": ("LATENT",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "experiment_name": ("STRING", {"default": "h3-icr-validation"}),
                "comparison_group": ("STRING", {"default": "strict-ab"}),
                "arm": ("STRING", {"default": "control"}),
                "locked_settings_json": ("STRING", {"multiline": True, "default": "{}"}),
                "arm_settings_json": ("STRING", {"multiline": True, "default": "{}"}),
                "strict_hashing": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "backend": ("H3_ICR_BACKEND",),
                "measurement": ("H3_ICR_MEASUREMENT",),
                "posterior": ("H3_ICR_POSTERIOR",),
                "pixel_measurement": ("H3_ICR_PIXEL_MEASUREMENT",),
                "renderer": ("H3_ICR_TILED_RENDERER",),
                "sparse_runtime": ("H3_ICR_SPARSE_RUNTIME",),
                "adapter": ("H3_ICR_BASE_ADAPTER",),
            },
        }

    RETURN_TYPES = ("H3_ICR_VALIDATION_MANIFEST", "STRING", "STRING")
    RETURN_NAMES = ("manifest", "manifest_json", "run_id")
    FUNCTION = "build"
    CATEGORY = "Kirei/MiniMax H3/ICR/Validation"
    DESCRIPTION = (
        "Build a canonical H3-ICR validation manifest. Base AV latent, conditioning, sigmas, NOISE and SAMPLER "
        "are fingerprinted automatically. locked_settings_json contains experiment invariants that must remain "
        "identical across strict A/B arms; arm_settings_json contains the treatment being varied. Optional M3/M4/M5/M6 "
        "handles record their active configuration. Strict hashing rejects unsupported conditioning/state objects instead "
        "of using non-deterministic repr strings."
    )

    def build(
        self,
        model,
        positive,
        base_latent,
        noise,
        sampler,
        sigmas,
        experiment_name,
        comparison_group,
        arm,
        locked_settings_json,
        arm_settings_json,
        strict_hashing,
        negative=None,
        backend=None,
        measurement=None,
        posterior=None,
        pixel_measurement=None,
        renderer=None,
        sparse_runtime=None,
        adapter=None,
    ):
        locked = parse_json_object(locked_settings_json, "locked_settings")
        arm_settings = parse_json_object(arm_settings_json, "arm_settings")
        manifest = build_validation_manifest(
            experiment_name=experiment_name,
            comparison_group=comparison_group,
            arm=arm,
            model=model,
            base_latent=base_latent,
            positive=positive,
            negative=negative,
            noise=noise,
            sampler=sampler,
            sigmas=sigmas,
            locked_settings=locked,
            arm_settings=arm_settings,
            strict_hashing=bool(strict_hashing),
            backend=backend,
            measurement=measurement,
            posterior=posterior,
            pixel_measurement=pixel_measurement,
            renderer=renderer,
            sparse_runtime=sparse_runtime,
            adapter=adapter,
        )
        rendered = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        return manifest, rendered, str(manifest["run_id"])


class H3ICRCompareValidationManifests:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_a": ("H3_ICR_VALIDATION_MANIFEST",),
                "manifest_b": ("H3_ICR_VALIDATION_MANIFEST",),
                "allowed_differences": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "arm.settings",
                    },
                ),
                "fail_on_unexpected_difference": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("compatible", "comparison_json")
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Validation"
    DESCRIPTION = (
        "Compare two validation manifests. The arm label may differ automatically; every other difference must be "
        "covered by an explicit allowed path such as arm.backend or arm.features.measurement_m3b. Differences under "
        "locks are never hidden: Base latent, conditioning, sigmas, noise/sampler identity and locked settings remain "
        "visible in the report. With fail_on_unexpected_difference enabled, an invalid A/B comparison stops the graph."
    )

    def compare(self, manifest_a, manifest_b, allowed_differences, fail_on_unexpected_difference):
        report = compare_validation_manifests(
            manifest_a,
            manifest_b,
            allowed_differences=allowed_differences,
        )
        if fail_on_unexpected_difference and not report["compatible"]:
            paths = [row["path"] for row in report.get("unexpected_differences", [])]
            preview = ", ".join(paths[:12])
            if len(paths) > 12:
                preview += f", ... (+{len(paths) - 12})"
            raise ValueError(f"H3-ICR validation A/B mismatch outside allowed differences: {preview}")
        return bool(report["compatible"]), json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)


NODE_CLASS_MAPPINGS = {
    "H3ICRValidationManifest": H3ICRValidationManifest,
    "H3ICRCompareValidationManifests": H3ICRCompareValidationManifests,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRValidationManifest": "Kirei H3 ICR Validation Manifest",
    "H3ICRCompareValidationManifests": "Kirei H3 ICR Compare Validation Manifests",
}
