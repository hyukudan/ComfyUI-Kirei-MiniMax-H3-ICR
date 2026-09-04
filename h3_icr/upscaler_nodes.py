from __future__ import annotations

import hashlib
import os
import urllib.parse
import urllib.request

from .latent_upscaler import KireiH3LatentUpscalerProvider


MODEL_FOLDER_KEY = "kirei_h3_upscalers"
MODEL_FOLDER_NAME = "kirei_h3_upscalers"
SHARED_MODEL_FOLDER_NAME = "latent_upscale_models"

REGISTERED_MODELS = {
    "minimax_h3_latent_upscaler_3d_bf16.safetensors": {
        "url": (
            "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/"
            "minimax_h3_latent_upscaler_3d_bf16.safetensors?download=true"
        ),
        "sha256": "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6",
        "size": 690_592_992,
        "license": "Apache-2.0",
    }
}


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme.lower() != "https":
            raise ValueError("Kirei model downloads refuse redirects to non-HTTPS URLs")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _register_model_folder():
    import folder_paths

    model_dir = os.path.join(folder_paths.models_dir, MODEL_FOLDER_NAME)
    os.makedirs(model_dir, exist_ok=True)
    if MODEL_FOLDER_KEY not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(MODEL_FOLDER_KEY, model_dir)
    return folder_paths, model_dir


def _model_names() -> list[str]:
    folder_paths, model_dir = _register_model_folder()
    names = {
        name
        for name in folder_paths.get_filename_list(MODEL_FOLDER_KEY)
        if name.lower().endswith(".safetensors")
    }
    shared_dir = os.path.join(folder_paths.models_dir, SHARED_MODEL_FOLDER_NAME)
    if os.path.isdir(shared_dir):
        for root, _dirs, files in os.walk(shared_dir):
            for filename in files:
                normalized = filename.lower()
                if normalized.startswith("minimax_h3_latent_upscaler_3d_") and normalized.endswith(
                    ".safetensors"
                ):
                    names.add(os.path.relpath(os.path.join(root, filename), shared_dir))
    names.update(REGISTERED_MODELS)
    return sorted(names) or [f"(place safetensors in {model_dir})"]


def _safe_candidate(root: str, name: str) -> str | None:
    root = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath((root, candidate)) != root:
        return None
    return candidate if os.path.isfile(candidate) else None


def _resolve_model_path(folder_paths, model_dir: str, model_name: str) -> str | None:
    local = _safe_candidate(model_dir, model_name)
    if local is not None:
        return local
    shared_dir = os.path.join(folder_paths.models_dir, SHARED_MODEL_FOLDER_NAME)
    return _safe_candidate(shared_dir, model_name)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_registered_model(model_name: str, model_dir: str) -> str:
    descriptor = REGISTERED_MODELS.get(model_name)
    if descriptor is None:
        raise ValueError(f"no verified Kirei download is registered for {model_name!r}")
    destination = os.path.join(model_dir, model_name)
    partial = destination + ".part"
    os.makedirs(model_dir, exist_ok=True)
    if os.path.isfile(destination):
        if _sha256_file(destination) == descriptor["sha256"]:
            return destination
        raise ValueError(f"existing model failed SHA-256 verification: {destination}")

    print(f"[Kirei H3 ICR] Downloading verified latent upscaler to {destination}")
    try:
        digest = hashlib.sha256()
        written = 0
        if urllib.parse.urlparse(descriptor["url"]).scheme.lower() != "https":
            raise ValueError("Kirei model downloads require HTTPS")
        request = urllib.request.Request(descriptor["url"], headers={"User-Agent": "Kirei-H3-ICR/1"})
        opener = urllib.request.build_opener(_HTTPSOnlyRedirectHandler())
        with opener.open(request, timeout=60) as response, open(partial, "wb") as handle:
            if urllib.parse.urlparse(response.geturl()).scheme.lower() != "https":
                raise ValueError("Kirei model download resolved to a non-HTTPS URL")
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if written != descriptor["size"] or digest.hexdigest() != descriptor["sha256"]:
            raise ValueError("downloaded upscaler failed size/SHA-256 verification")
        os.replace(partial, destination)
    except Exception:
        if os.path.isfile(partial):
            os.remove(partial)
        raise
    return destination


class KireiH3ICRLatentUpscalerProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_names(),),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "temporal_chunk_size": ("INT", {"default": 32, "min": 0, "max": 256, "step": 4}),
                "offload_after_upscale": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "temporal_stability": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "download_if_missing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("H3_LATENT_UPSCALER",)
    RETURN_NAMES = ("learned_upscaler",)
    FUNCTION = "build"
    CATEGORY = "Kirei/MiniMax H3/ICR"
    DESCRIPTION = (
        "Native Kirei API-v1 learned latent-upscaler provider for Prepare Clean or Regenerate. "
        "It preserves audio, supports full temporal context with chunk size 0, can stabilize only the "
        "learned temporal residual, and can explicitly download the registered SHA-256-verified weights."
    )

    def build(
        self,
        model_name,
        device,
        precision,
        temporal_chunk_size,
        offload_after_upscale,
        temporal_stability=0.30,
        download_if_missing=False,
    ):
        if str(model_name).startswith("("):
            raise ValueError("place a compatible checkpoint in ComfyUI/models/kirei_h3_upscalers")
        folder_paths, model_dir = _register_model_folder()
        path = _resolve_model_path(folder_paths, model_dir, str(model_name))
        if path is None and bool(download_if_missing):
            path = _download_registered_model(str(model_name), model_dir)
        if path is None:
            raise FileNotFoundError(
                f"upscaler {model_name!r} is not installed; place it in {model_dir} or enable "
                "download_if_missing for a registered model"
            )
        checkpoint_sha256 = _sha256_file(path)
        registered = REGISTERED_MODELS.get(str(model_name))
        if registered is not None and checkpoint_sha256 != registered["sha256"]:
            raise ValueError(f"registered upscaler failed SHA-256 verification: {path}")
        return (
            KireiH3LatentUpscalerProvider(
                checkpoint_path=path,
                device=device,
                precision=precision,
                temporal_chunk_size=temporal_chunk_size,
                temporal_stability=temporal_stability,
                offload_after_upscale=offload_after_upscale,
                checkpoint_sha256=checkpoint_sha256,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "KireiH3ICRLatentUpscalerProvider": KireiH3ICRLatentUpscalerProviderNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KireiH3ICRLatentUpscalerProvider": "Kirei H3 ICR Learned Latent Upscaler [Native]",
}
