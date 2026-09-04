from __future__ import annotations

import json

import torch

from .contracts import make_av_container, target_latent_hw, unwrap_av, validate_av


class H3ICRVAEUpscalePrior:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_latent": ("LATENT",),
                "video_vae": ("VAE",),
                "target_width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "target_height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "upscale_method": (
                    ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                    {"default": "lanczos"},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("vae_prior_latent", "report_json")
    FUNCTION = "build"
    CATEGORY = "Kirei/MiniMax H3/ICR/Research"
    DESCRIPTION = (
        "Experimental premium prior: decode the clean H3 Base video, resize deterministically in RGB, and encode it "
        "at target resolution. Audio bypasses the round-trip exactly. Connect the output to Regenerate's optional "
        "vae_prior_latent input; leaving that input disconnected preserves the existing pipeline."
    )

    def build(self, base_latent, video_vae, target_width, target_height, upscale_method):
        if not isinstance(base_latent, dict) or "samples" not in base_latent:
            raise TypeError("base_latent must be a ComfyUI LATENT containing H3 AV samples")
        validate_av(base_latent["samples"])
        target_h, target_w = target_latent_hw(target_width, target_height)
        source_video, source_audio = unwrap_av(base_latent["samples"])

        # Reuse ComfyUI's canonical VAE and image-resize implementations rather
        # than maintaining a fork of model-specific encode/decode behavior.
        import nodes

        decoded = nodes.VAEDecode().decode(video_vae, base_latent)[0]
        resized = nodes.ImageScale().upscale(
            decoded,
            str(upscale_method),
            int(target_width),
            int(target_height),
            "disabled",
        )[0]
        encoded = nodes.VAEEncode().encode(video_vae, resized)[0]
        prior_video = encoded.get("samples")
        if not torch.is_tensor(prior_video):
            raise TypeError("H3 VAE encode did not return a tensor latent")
        if prior_video.ndim != 5 or prior_video.shape[:3] != source_video.shape[:3]:
            raise ValueError(
                "H3 VAE round-trip must preserve video B/C/T; "
                f"got {tuple(prior_video.shape)}, source {tuple(source_video.shape)}"
            )
        if tuple(prior_video.shape[-2:]) != (target_h, target_w):
            raise ValueError(
                f"H3 VAE round-trip returned {tuple(prior_video.shape[-2:])}, "
                f"expected {(target_h, target_w)}"
            )
        if not prior_video.is_floating_point() or not bool(torch.isfinite(prior_video).all().item()):
            raise ValueError("H3 VAE round-trip returned an invalid/non-finite video latent")

        audio = source_audio.to(device=prior_video.device)
        samples = make_av_container(prior_video, audio, template=base_latent["samples"])
        validate_av(samples)
        output = dict(base_latent)
        output["samples"] = samples
        output.pop("noise_mask", None)
        report = {
            "api": 1,
            "kind": "kirei_h3_vae_roundtrip_prior",
            "upscale_method": str(upscale_method),
            "source_video_shape": list(source_video.shape),
            "target_video_shape": list(prior_video.shape),
            "audio_shape": list(audio.shape),
            "audio": "source_bypass_exact",
            "cost_note": "one_full_video_decode_and_encode",
        }
        return output, json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)


NODE_CLASS_MAPPINGS = {
    "H3ICRVAEUpscalePrior": H3ICRVAEUpscalePrior,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ICRVAEUpscalePrior": "Kirei H3 ICR VAE Upscale Prior [Experimental]",
}
