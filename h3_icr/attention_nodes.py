from __future__ import annotations

import json

from .attention_profile_v2 import patch_attention_profiler_v2, propose_attention_policy_v2


class H3ICRAttentionProfiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "layer_stride": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "query_samples": ("INT", {"default": 24, "min": 4, "max": 256, "step": 4}),
                "key_samples_per_modality": ("INT", {"default": 48, "min": 4, "max": 512, "step": 4}),
                "sigma_decimals": ("INT", {"default": 3, "min": 0, "max": 6, "step": 1}),
                "max_buckets": ("INT", {"default": 2048, "min": 32, "max": 16384, "step": 32}),
                "model_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_ATTN_PROFILE")
    RETURN_NAMES = ("model", "profile")
    FUNCTION = "patch"
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"
    DESCRIPTION = (
        "M5 passive calibration profiler. Samples normalized H3 Q/K by layer, sigma and modality, "
        "and adds exact spatial/temporal/far QK pair evidence for target-video heads. It delegates "
        "to the original attention backend unchanged and never enables sparse attention."
    )

    def patch(
        self,
        model,
        layer_stride,
        query_samples,
        key_samples_per_modality,
        sigma_decimals,
        max_buckets,
        model_id,
    ):
        patched, runtime = patch_attention_profiler_v2(
            model,
            layer_stride=layer_stride,
            query_samples=query_samples,
            key_samples_per_modality=key_samples_per_modality,
            sigma_decimals=sigma_decimals,
            max_buckets=max_buckets,
            model_id=model_id,
        )
        return patched, {"api": 2, "runtime": runtime}


class H3ICRAttentionProfileReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profile": ("H3_ICR_ATTN_PROFILE",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("profile_json", "policy_proposal_json")
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"

    def render(self, profile):
        runtime = profile.get("runtime") if isinstance(profile, dict) else None
        if runtime is None or not hasattr(runtime, "report"):
            raise TypeError("invalid H3 ICR attention profile handle")
        report = runtime.report()
        proposal = propose_attention_policy_v2(report)
        return (
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
            json.dumps(proposal, ensure_ascii=False, sort_keys=True, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "H3ICRAttentionProfiler": H3ICRAttentionProfiler,
    "H3ICRAttentionProfileReport": H3ICRAttentionProfileReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRAttentionProfiler": "Kirei H3 ICR Attention Profiler [M5 Research]",
    "H3ICRAttentionProfileReport": "Kirei H3 ICR Attention Profile Report",
}
