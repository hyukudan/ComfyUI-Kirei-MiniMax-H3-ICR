from __future__ import annotations

import json

from .base_video_adapter import (
    BaseVideoAdapterProvider,
    create_zero_init_base_adapter_provider,
    patch_base_video_adapter,
)
from .contracts import unwrap_av


class H3ICRBaseVideoAdapterScaffold:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "injection_blocks": ("STRING", {"default": "12,24,36,45,48"}),
                "adapter_dim": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32}),
                "gate_floor": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gate_power": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "temporal_kernel": ([1, 3, 5], {"default": 3}),
                "spatial_kernel": ([1, 3, 5], {"default": 3}),
                "model_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("H3_ICR_BASE_ADAPTER_PROVIDER", "STRING")
    RETURN_NAMES = ("adapter_provider", "provider_json")
    FUNCTION = "create"
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"
    DESCRIPTION = (
        "Create the M6 zero-init BaseVideo Adapter provider for architecture/plumbing validation. "
        "This scaffold contains no trained adapter weights: trained=false and the residual output projection "
        "is zero. Applying it must therefore preserve native H3 output exactly."
    )

    def create(
        self,
        model,
        injection_blocks,
        adapter_dim,
        gate_floor,
        gate_power,
        temporal_kernel,
        spatial_kernel,
        model_id,
    ):
        provider = create_zero_init_base_adapter_provider(
            model,
            injection_blocks=injection_blocks,
            adapter_dim=int(adapter_dim),
            gate_floor=float(gate_floor),
            gate_power=float(gate_power),
            temporal_kernel=int(temporal_kernel),
            spatial_kernel=int(spatial_kernel),
            model_id=model_id,
        )
        return provider, json.dumps(provider.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)


class H3ICRApplyBaseVideoAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_latent": ("LATENT",),
                "adapter_provider": ("H3_ICR_BASE_ADAPTER_PROVIDER",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "H3_ICR_BASE_ADAPTER")
    RETURN_NAMES = ("model", "adapter")
    FUNCTION = "apply"
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"
    DESCRIPTION = (
        "Apply an M6 BaseVideo Adapter provider to selected native H3 blocks. The current scaffold provider "
        "is intentionally untrained and bypasses residual injection. Trained providers must use the same ABI "
        "and architecture fingerprint. M4 HR-tile injection is currently fail-safe/bypassed until the explicit "
        "global tile-region contract is added."
    )

    def apply(self, model, base_latent, adapter_provider, strength):
        if not isinstance(adapter_provider, BaseVideoAdapterProvider):
            raise TypeError("adapter_provider is not an H3 ICR BaseVideo Adapter provider")
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing H3 AV samples")
        base_video, _ = unwrap_av(base_latent["samples"])
        patched, runtime = patch_base_video_adapter(
            model,
            base_video,
            adapter_provider,
            strength=float(strength),
        )
        return patched, {"api": 1, "runtime": runtime}


class H3ICRBaseVideoAdapterReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"adapter": ("H3_ICR_BASE_ADAPTER",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"

    def render(self, adapter):
        runtime = adapter.get("runtime") if isinstance(adapter, dict) else None
        if runtime is None or not hasattr(runtime, "report"):
            raise TypeError("invalid H3 ICR BaseVideo Adapter handle")
        return (json.dumps(runtime.report(), ensure_ascii=False, sort_keys=True, indent=2),)


NODE_CLASS_MAPPINGS = {
    "H3ICRBaseVideoAdapterScaffold": H3ICRBaseVideoAdapterScaffold,
    "H3ICRApplyBaseVideoAdapter": H3ICRApplyBaseVideoAdapter,
    "H3ICRBaseVideoAdapterReport": H3ICRBaseVideoAdapterReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRBaseVideoAdapterScaffold": "Kirei H3 ICR BaseVideo Adapter Scaffold [M6]",
    "H3ICRApplyBaseVideoAdapter": "Kirei H3 ICR Apply BaseVideo Adapter [M6]",
    "H3ICRBaseVideoAdapterReport": "Kirei H3 ICR BaseVideo Adapter Report",
}
