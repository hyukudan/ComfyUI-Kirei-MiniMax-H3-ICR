from __future__ import annotations

from .backend import BACKENDS, BackendDescriptor, descriptor_from_model, tag_model
from .contracts import unwrap_av
from .fidelity import FidelityConfig
from .initializer import InitConfig, upscale_and_align_clean
from .metrics import ICRMetrics
from .reference import append_base_latent_reference
from .runtime_fidelity import patch_per_step_fidelity
from .sampling import mark_second_pass, run_second_pass, validate_partial_sigmas


class H3ICRBackendTag:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (list(BACKENDS), {"default": "hybrid_late_adaln"}),
                "checkpoint_format": (["unknown", "pruned", "full"], {"default": "unknown"}),
                "checkpoint_sha256": ("STRING", {"default": ""}),
                "overlay_sha256": ("STRING", {"default": ""}),
                "note": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_BACKEND")
    RETURN_NAMES = ("model", "backend")
    FUNCTION = "tag"
    CATEGORY = "Kirei/MiniMax H3/ICR"

    def tag(self, model, backend, checkpoint_format, checkpoint_sha256, overlay_sha256, note):
        descriptor = BackendDescriptor(
            kind=backend,
            checkpoint_format=checkpoint_format,
            checkpoint_sha256=checkpoint_sha256.strip(),
            overlay_sha256=overlay_sha256.strip(),
            note=note.strip(),
        )
        return tag_model(model, descriptor), descriptor


class H3ICRAppendBaseLatentReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "base_latent": ("LATENT",),
                "include_audio": ("BOOLEAN", {"default": False}),
                "replace_existing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "append"
    CATEGORY = "Kirei/MiniMax H3/ICR"
    DESCRIPTION = (
        "Append the clean H3 Base latent directly as a native minimax_refs block. "
        "This conditions the DiT without a VAE round-trip; it does not add Qwen visual tokens."
    )

    def append(self, positive, base_latent, include_audio=False, replace_existing=True):
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing samples")
        return (
            append_base_latent_reference(
                positive,
                base_latent["samples"],
                include_audio=bool(include_audio),
                replace_existing=bool(replace_existing),
            ),
        )


class H3ICRPrepareClean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_latent": ("LATENT",),
                "target_width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "transfer": (["learned_3d", "bicubic"], {"default": "learned_3d"}),
                "fidelity_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "fidelity_cutoff": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01}),
                "max_correction_rms_ratio": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01}),
            },
            "optional": {"learned_upscaler": ("H3_LATENT_UPSCALER",)},
        }

    RETURN_TYPES = ("LATENT", "H3_ICR_REPORT")
    RETURN_NAMES = ("clean_hr_latent", "report")
    FUNCTION = "prepare"
    CATEGORY = "Kirei/MiniMax H3/ICR"

    def prepare(
        self,
        base_latent,
        target_width,
        target_height,
        transfer,
        fidelity_strength,
        fidelity_cutoff,
        max_correction_rms_ratio,
        learned_upscaler=None,
    ):
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing samples")
        config = InitConfig(
            transfer=transfer,
            target_width=target_width,
            target_height=target_height,
            fidelity=FidelityConfig(
                strength=fidelity_strength,
                cutoff=fidelity_cutoff,
                max_correction_rms_ratio=max_correction_rms_ratio,
            ),
        )
        samples, report = upscale_and_align_clean(base_latent["samples"], config, learned_upscaler)
        out = dict(base_latent)
        out["samples"] = samples
        if "noise_mask" in out:
            out.pop("noise_mask")
            report["noise_mask"] = "dropped_source_grid_mask_v0.1"
        return out, report


class H3ICRRegenerate:
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
                "target_width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "transfer": (["learned_3d", "bicubic"], {"default": "learned_3d"}),
                "fidelity_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "fidelity_cutoff": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01}),
                "max_correction_rms_ratio": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01}),
                "per_step_fidelity_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01}),
                "per_step_fidelity_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "per_step_fidelity_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lock_audio": ("BOOLEAN", {"default": True}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "learned_upscaler": ("H3_LATENT_UPSCALER",),
                "backend": ("H3_ICR_BACKEND",),
            },
        }

    RETURN_TYPES = ("LATENT", "H3_ICR_REPORT")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "regenerate"
    CATEGORY = "Kirei/MiniMax H3/ICR"
    DESCRIPTION = (
        "Second-pass H3 regeneration. The positive conditioning should be built with the native "
        "MiniMax H3 Reference to Video node and include the H3 Base video plus original references."
    )

    def regenerate(
        self,
        model,
        positive,
        base_latent,
        noise,
        sampler,
        sigmas,
        target_width,
        target_height,
        transfer,
        fidelity_strength,
        fidelity_cutoff,
        max_correction_rms_ratio,
        per_step_fidelity_strength,
        per_step_fidelity_power,
        per_step_fidelity_floor,
        lock_audio,
        cfg,
        negative=None,
        learned_upscaler=None,
        backend=None,
    ):
        metrics = ICRMetrics()
        sigma_start = validate_partial_sigmas(sigmas)
        descriptor = backend if isinstance(backend, BackendDescriptor) else descriptor_from_model(model)
        metrics.event("backend", **descriptor.to_dict())
        metrics.event("schedule", sigma_start=sigma_start, intervals=int(sigmas.numel() - 1))

        prepared_node = H3ICRPrepareClean()
        clean, init_report = prepared_node.prepare(
            base_latent,
            target_width,
            target_height,
            transfer,
            fidelity_strength,
            fidelity_cutoff,
            max_correction_rms_ratio,
            learned_upscaler,
        )
        metrics.event("initialization", **init_report)
        prepared_model = mark_second_pass(model)
        source_video, _ = unwrap_av(base_latent["samples"])
        fidelity_stats = None
        if per_step_fidelity_strength > 0.0:
            prepared_model, fidelity_stats = patch_per_step_fidelity(
                prepared_model,
                source_video,
                sigma_start=sigma_start,
                strength=per_step_fidelity_strength,
                cutoff=fidelity_cutoff,
                max_correction_rms_ratio=max_correction_rms_ratio,
                schedule_power=per_step_fidelity_power,
                schedule_floor=per_step_fidelity_floor,
            )
            metrics.event(
                "per_step_fidelity_config",
                strength=float(per_step_fidelity_strength),
                power=float(per_step_fidelity_power),
                floor=float(per_step_fidelity_floor),
            )
        out = run_second_pass(
            clean,
            model=prepared_model,
            positive=positive,
            negative=negative,
            noise=noise,
            sampler=sampler,
            sigmas=sigmas,
            cfg=cfg,
            lock_audio=bool(lock_audio),
        )
        if fidelity_stats is not None:
            metrics.event("per_step_fidelity_result", **fidelity_stats.to_dict())
        metrics.event("complete", lock_audio=bool(lock_audio))
        return out, metrics.to_dict()


class H3ICRReportJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"report": ("H3_ICR_REPORT",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR"

    def render(self, report):
        import json

        return (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),)


NODE_CLASS_MAPPINGS = {
    "H3ICRBackendTag": H3ICRBackendTag,
    "H3ICRAppendBaseLatentReference": H3ICRAppendBaseLatentReference,
    "H3ICRPrepareClean": H3ICRPrepareClean,
    "H3ICRRegenerate": H3ICRRegenerate,
    "H3ICRReportJSON": H3ICRReportJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRBackendTag": "Kirei H3 ICR Backend Tag",
    "H3ICRAppendBaseLatentReference": "Kirei H3 ICR Append Base Latent Reference",
    "H3ICRPrepareClean": "Kirei H3 ICR Prepare Clean HR",
    "H3ICRRegenerate": "Kirei H3 ICR Regenerate",
    "H3ICRReportJSON": "Kirei H3 ICR Report JSON",
}
