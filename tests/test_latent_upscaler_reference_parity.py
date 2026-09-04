from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from h3_icr import latent_upscaler as kirei


CHECKPOINT = os.environ.get("KIREI_H3_UPSCALER_CHECKPOINT", "")
REFERENCE = os.environ.get("KIREI_H3_UPSCALER_REFERENCE", "")
HAS_FIXTURE = bool(CHECKPOINT and REFERENCE and torch.cuda.is_available())

pytestmark = pytest.mark.skipif(
    not HAS_FIXTURE,
    reason="reference parity requires CUDA plus local checkpoint/reference paths",
)


@pytest.fixture(scope="module")
def models():
    reference_path = Path(REFERENCE).resolve()
    comfy_root = reference_path.parents[3]
    sys.path.insert(0, str(comfy_root))
    spec = importlib.util.spec_from_file_location("kirei_lbh_reference", reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load reference module from {reference_path}")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    raw = reference._load_raw_sd(CHECKPOINT)
    state = reference._extract_upscaler_sd(raw)
    config = reference._detect_arch(state)
    reference_model = reference.LatentResizer3D(
        in_channels=config["in_channels"],
        in_blocks=config["in_blocks"],
        out_blocks=config["out_blocks"],
        channels=config["channels"],
        dropout=config["dropout"],
        attn=config["attn"],
        temporal_every=config["temporal_every"],
        temporal_kernel=config["temporal_kernel"],
    )
    reference_model.load_state_dict(state, strict=True)
    reference_model = reference_model.to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    kirei_model = kirei._load_checkpoint_model(
        Path(CHECKPOINT),
        torch.device("cuda"),
        torch.bfloat16,
    )
    return reference, reference_model, kirei_model


def _run_pair(models, shape, target_h, target_w, *, chunked):
    reference, reference_model, kirei_model = models
    torch.manual_seed(123)
    source = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    ref_mean, ref_std = reference._make_norm_tensors(torch.device("cuda"), torch.bfloat16)
    our_mean = torch.tensor(kirei.LATENTS_MEAN, device="cuda", dtype=torch.bfloat16).view(
        1, -1, 1, 1, 1
    )
    our_std = torch.tensor(kirei.LATENTS_STD, device="cuda", dtype=torch.bfloat16).view(
        1, -1, 1, 1, 1
    )
    scale = 0.5 * (target_h / shape[-2] + target_w / shape[-1])
    ref_normalized = (source - ref_mean) / ref_std
    our_normalized = (source - our_mean) / our_std
    assert torch.equal(ref_normalized, our_normalized)
    with torch.inference_mode():
        expected = reference_model(
            ref_normalized,
            scale=scale,
            target_size=(shape[2], target_h, target_w),
            enable_chunking=chunked,
        )
        actual = kirei_model(
            our_normalized,
            target_h=target_h,
            target_w=target_w,
            scale=scale,
            temporal_chunk_size=32 if chunked else 0,
        )
    assert torch.equal(expected, actual)


def test_full_context_reference_parity(models):
    _run_pair(models, (1, 24, 5, 4, 4), 6, 6, chunked=False)


def test_chunk32_reference_parity(models):
    _run_pair(models, (1, 24, 37, 4, 4), 6, 6, chunked=True)


def test_real_anisotropic_geometry_reference_parity(models):
    _run_pair(models, (1, 24, 37, 54, 48), 80, 72, chunked=False)
