from __future__ import annotations

import math
from typing import Any

import torch

from .contracts import make_av_container, unwrap_av, validate_av

H3_REFINEMENT_REQUEST_KEY = "h3_refinement"
H3_REFINEMENT_API = 1


def validate_partial_sigmas(sigmas: torch.Tensor) -> float:
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        raise TypeError("sigmas must be a one-dimensional torch.Tensor")
    if sigmas.numel() < 2:
        raise ValueError("sigmas must contain at least start and terminal coordinates")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("sigmas contain NaN/Inf")
    sigma_start = float(sigmas[0].detach().cpu().item())
    if not math.isfinite(sigma_start) or not 0.0 <= sigma_start < 1.0:
        raise ValueError(
            "H3 ICR needs a partial second pass with 0 <= sigmas[0] < 1. "
            "A full-noise start discards the clean learned initialization."
        )
    return sigma_start


def mark_second_pass(model: Any) -> Any:
    clone = getattr(model, "clone", None)
    get_model_object = getattr(model, "get_model_object", None)
    if not callable(clone) or not callable(get_model_object):
        raise TypeError("H3 ICR regeneration expects a ComfyUI MODEL/ModelPatcher")
    refined = clone()
    if refined is model:
        raise RuntimeError("MODEL.clone() returned original object")
    sampling = refined.get_model_object("model_sampling")
    sigma_max = getattr(sampling, "sigma_max", None)
    if sigma_max is None:
        raise ValueError("MODEL.model_sampling does not expose sigma_max")
    if torch.is_tensor(sigma_max):
        if sigma_max.numel() != 1:
            raise ValueError("sigma_max must be scalar")
        sigma_reference = float(sigma_max.detach().cpu().item())
    else:
        sigma_reference = float(sigma_max)
    if not math.isfinite(sigma_reference) or sigma_reference <= 0:
        raise ValueError("invalid H3 sigma reference")

    options = dict(getattr(refined, "model_options", {}))
    transformer = options.get("transformer_options", {})
    if not isinstance(transformer, dict):
        raise TypeError("transformer_options must be a dictionary")
    transformer = dict(transformer)
    existing = transformer.get(H3_REFINEMENT_REQUEST_KEY)
    if existing is not None and not isinstance(existing, dict):
        raise TypeError("existing h3_refinement metadata is malformed")
    merged = dict(existing or {})
    if merged.get("api", H3_REFINEMENT_API) != H3_REFINEMENT_API:
        raise ValueError("conflicting h3_refinement API")
    merged.update(
        {
            "api": H3_REFINEMENT_API,
            "active": True,
            "source": "h3_icr",
            "min_actual_prefix_steps": 0,
            "sigma_reference": sigma_reference,
        }
    )
    transformer[H3_REFINEMENT_REQUEST_KEY] = merged
    options["transformer_options"] = transformer
    refined.model_options = options
    return refined


def _make_guider(model: Any, positive: list, negative: list | None, cfg: float):
    import comfy.samplers

    if negative is None:
        class _BasicGuider(comfy.samplers.CFGGuider):
            def set_positive(self, value):
                self.inner_set_conds({"positive": value})

        guider = _BasicGuider(model)
        guider.set_positive(positive)
        return guider
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(float(cfg))
    return guider


def run_second_pass(
    clean_latent: dict,
    *,
    model: Any,
    positive: list,
    negative: list | None,
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    cfg: float = 1.0,
    lock_audio: bool = True,
) -> dict:
    """Execute the H3 partial-noise second pass through ComfyUI's native sampler path."""
    validate_partial_sigmas(sigmas)
    try:
        import comfy.model_management
        import comfy.sample
        import comfy.utils
        import latent_preview
    except Exception as exc:  # pragma: no cover - real ComfyUI runtime only
        raise RuntimeError(f"ComfyUI sampling runtime unavailable: {exc}") from exc

    if "samples" not in clean_latent:
        raise ValueError("clean LATENT is missing samples")
    validate_av(clean_latent["samples"])
    latent = dict(clean_latent)
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"], None, None)
    validate_av(latent_image)
    clean_video, clean_audio = unwrap_av(latent_image)
    latent["samples"] = latent_image

    generated_noise = noise.generate_noise(latent)
    validate_av(generated_noise)
    noise_video, noise_audio = unwrap_av(generated_noise)
    if noise_video.shape != clean_video.shape or noise_audio.shape != clean_audio.shape:
        raise ValueError("second-pass noise must exactly match the clean H3 AV latent")
    if lock_audio:
        generated_noise = make_av_container(noise_video, torch.zeros_like(noise_audio), template=generated_noise)

    guider = _make_guider(model, positive, negative, cfg)
    x0_output: dict[str, Any] = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, int(sigmas.shape[-1]) - 1, x0_output)
    samples = guider.sample(
        generated_noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=latent.get("noise_mask"),
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(getattr(noise, "seed", 0)),
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    validate_av(samples)
    sampled_video, sampled_audio = unwrap_av(samples)
    if lock_audio:
        sampled_audio = clean_audio.to(sampled_audio.device)
        samples = make_av_container(sampled_video, sampled_audio, template=samples)
    out = dict(latent)
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out
