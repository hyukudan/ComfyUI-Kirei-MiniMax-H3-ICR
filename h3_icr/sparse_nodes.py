from __future__ import annotations

import json

from .sparse_attention_v3 import patch_flex_sparse_attention_v3


class H3ICRFlexSparseAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "policy_json": ("STRING", {"multiline": True, "default": ""}),
                "model_id": ("STRING", {"default": ""}),
                "dense_tail_sigma": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.99, "step": 0.01}),
                "max_policy_sigma_distance": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "block_size": ([16, 32, 64, 128, 256], {"default": 128}),
                "min_block_sparsity": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "local_t_radius": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "local_y_radius": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "local_x_radius": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "temporal_radius": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "force_flex_kernel": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "profile_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_SPARSE_RUNTIME")
    RETURN_NAMES = ("model", "sparse_runtime")
    FUNCTION = "patch"
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"
    DESCRIPTION = (
        "Experimental M5 v3 FlexAttention BlockMask backend. It requires the current topology- and "
        "sigma-domain-bound M5 policy. Every call selects the closest calibrated branch/topology/sigma "
        "domain inside max_policy_sigma_distance; otherwise it fails back to dense attention."
    )

    def patch(
        self,
        model,
        policy_json,
        model_id,
        dense_tail_sigma,
        max_policy_sigma_distance,
        block_size,
        min_block_sparsity,
        local_t_radius,
        local_y_radius,
        local_x_radius,
        temporal_radius,
        force_flex_kernel,
        profile_json="",
    ):
        patched, runtime = patch_flex_sparse_attention_v3(
            model,
            policy_json=policy_json,
            profile_json=profile_json,
            model_id=model_id,
            dense_tail_sigma=dense_tail_sigma,
            max_policy_sigma_distance=max_policy_sigma_distance,
            block_size=int(block_size),
            min_block_sparsity=min_block_sparsity,
            local_t_radius=local_t_radius,
            local_y_radius=local_y_radius,
            local_x_radius=local_x_radius,
            temporal_radius=temporal_radius,
            force_flex_kernel=force_flex_kernel,
        )
        return patched, {"api": 3, "runtime": runtime}


class H3ICRFlexSparseReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sparse_runtime": ("H3_ICR_SPARSE_RUNTIME",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"

    def render(self, sparse_runtime):
        runtime = sparse_runtime.get("runtime") if isinstance(sparse_runtime, dict) else None
        if runtime is None or not hasattr(runtime, "report"):
            raise TypeError("invalid H3 ICR sparse runtime handle")
        return (json.dumps(runtime.report(), ensure_ascii=False, sort_keys=True, indent=2),)


NODE_CLASS_MAPPINGS = {
    "H3ICRFlexSparseAttention": H3ICRFlexSparseAttention,
    "H3ICRFlexSparseReport": H3ICRFlexSparseReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRFlexSparseAttention": "Kirei H3 ICR Flex Sparse Attention [M5 Experimental]",
    "H3ICRFlexSparseReport": "Kirei H3 ICR Flex Sparse Report",
}
