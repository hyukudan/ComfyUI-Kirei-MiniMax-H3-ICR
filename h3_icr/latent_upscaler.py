from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


H3_LATENT_UPSCALER_API = 1
H3_LATENT_UPSCALER_KIND = "minimax_h3_learned_latent_upscaler"
H3_VIDEO_CHANNELS = 24

LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)


def _stabilize_temporal_residual(
    output: torch.Tensor,
    source: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    """Smooth only learned temporal residuals while preserving Base motion.

    The spatially resized source is the motion-preserving baseline.  Filtering
    the learned residual instead of the whole latent avoids averaging genuine
    Base movement between adjacent frames.
    """
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("temporal stability must be in [0, 1]")
    if strength == 0.0 or output.shape[2] < 2:
        return output
    baseline = F.interpolate(
        source,
        size=(source.shape[2], output.shape[-2], output.shape[-1]),
        mode="trilinear",
        align_corners=False,
    )
    residual = output - baseline
    padded = F.pad(residual, (0, 0, 0, 0, 1, 1), mode="replicate")
    smoothed = 0.25 * (
        padded[:, :, :-2] + 2.0 * padded[:, :, 1:-1] + padded[:, :, 2:]
    )
    return baseline + torch.lerp(residual, smoothed, strength)


def _normalization(channels: int) -> nn.GroupNorm:
    if channels % 32:
        raise ValueError("Kirei H3 latent upscaler channels must be divisible by 32")
    return nn.GroupNorm(32, channels)


class _ResBlock3D(nn.Module):
    def __init__(self, channels: int, embedding_channels: int, dropout: float = 0.1):
        super().__init__()
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_channels, 2 * channels),
        )
        self.out_norm = _normalization(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(x)
        scale, shift = self.emb_layers(embedding).to(hidden).chunk(2, dim=1)
        while scale.ndim < hidden.ndim:
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        hidden = self.out_norm(hidden) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(hidden)


class _TemporalConv3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.pwconv(self.dwconv(F.silu(self.norm(x))))
        return x + hidden


class KireiH3LatentUpscaler3D(nn.Module):
    """Strict runtime for Kirei's supported 24-channel H3 latent-upscaler ABI."""

    def __init__(
        self,
        *,
        channels: int = 512,
        in_blocks: int = 12,
        out_blocks: int = 12,
        embedding_channels: int = 64,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ):
        super().__init__()
        self.temporal_kernel = int(temporal_kernel)
        self.conv_in = nn.Conv3d(H3_VIDEO_CHANNELS, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.in_blocks = self._make_stack(
            channels, embedding_channels, in_blocks, temporal_every, temporal_kernel
        )
        self.out_blocks = self._make_stack(
            channels, embedding_channels, out_blocks, temporal_every, temporal_kernel
        )
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, H3_VIDEO_CHANNELS, 3, padding=1)

    @staticmethod
    def _make_stack(
        channels: int,
        embedding_channels: int,
        block_count: int,
        temporal_every: int,
        temporal_kernel: int,
    ) -> nn.ModuleList:
        layers: list[nn.Module] = []
        for block_index in range(block_count):
            layers.append(_ResBlock3D(channels, embedding_channels))
            if temporal_every > 0 and block_index % temporal_every == 0:
                layers.append(_TemporalConv3D(channels, temporal_kernel))
        return nn.ModuleList(layers)

    def _forward_segment(
        self,
        video: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
        scale: float,
    ) -> torch.Tensor:
        embedding = torch.tensor([[scale - 1.0]], device=video.device, dtype=video.dtype)
        embedding = self.embed(embedding).expand(video.shape[0], -1)
        hidden = self.conv_in(video)
        for block in self.in_blocks:
            hidden = block(hidden, embedding) if isinstance(block, _ResBlock3D) else block(hidden)
        hidden = F.interpolate(
            hidden,
            size=(video.shape[2], target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
        for block in self.out_blocks:
            hidden = block(hidden, embedding) if isinstance(block, _ResBlock3D) else block(hidden)
        return self.conv_out(F.silu(self.norm_out(hidden)))

    def forward(
        self,
        video: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
        scale: float,
        temporal_chunk_size: int,
    ) -> torch.Tensor:
        total_frames = int(video.shape[2])
        chunk_size = int(temporal_chunk_size)
        if chunk_size <= 0 or total_frames <= chunk_size:
            return self._forward_segment(video, target_h=target_h, target_w=target_w, scale=scale)

        overlap = max(1, self.temporal_kernel)
        padded = F.pad(video, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        result = torch.zeros(
            video.shape[0],
            video.shape[1],
            total_frames,
            target_h,
            target_w,
            device=video.device,
            dtype=video.dtype,
        )
        weights = torch.zeros(1, 1, total_frames, 1, 1, device=video.device, dtype=video.dtype)

        start = 0
        while start < total_frames:
            core_end = min(total_frames, start + chunk_size)
            out_start = max(0, start - overlap)
            out_end = min(total_frames, core_end + overlap)
            source_start = max(0, out_start - overlap)
            source_end = min(total_frames + 2 * overlap, out_end + overlap)
            segment = padded[:, :, source_start:source_end].contiguous()
            segment_out = self._forward_segment(
                segment,
                target_h=target_h,
                target_w=target_w,
                scale=scale,
            )
            valid_start = out_start + overlap - source_start
            valid_end = valid_start + out_end - out_start
            valid = segment_out[:, :, valid_start:valid_end]
            blend = torch.ones(out_end - out_start, device=video.device, dtype=video.dtype)
            if start > out_start:
                length = start - out_start
                blend[:length] = torch.arange(1, length + 1, device=video.device, dtype=video.dtype) / (
                    length + 1
                )
            if out_end > core_end:
                length = out_end - core_end
                blend[-length:] = torch.arange(length, 0, -1, device=video.device, dtype=video.dtype) / (
                    length + 1
                )
            shaped_blend = blend.view(1, 1, -1, 1, 1)
            result[:, :, out_start:out_end] += valid * shaped_blend
            weights[:, :, out_start:out_end] += shaped_blend
            start += chunk_size
        return result / weights.clamp_min(1e-8)


def _read_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Kirei H3 latent upscalers only accept safetensors checkpoints")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    if any(key.startswith("upscaler.") for key in state):
        state = {
            key.removeprefix("upscaler."): value
            for key, value in state.items()
            if key.startswith("upscaler.")
        }
    return state


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(state: dict[str, torch.Tensor]) -> None:
    conv_in = state.get("conv_in.weight")
    conv_out = state.get("conv_out.weight")
    if conv_in is None or tuple(conv_in.shape) != (512, 24, 3, 3, 3):
        raise ValueError("checkpoint does not match the Kirei H3 24->512 latent-upscaler architecture")
    if conv_out is None or tuple(conv_out.shape) != (24, 512, 3, 3, 3):
        raise ValueError("checkpoint does not match the Kirei H3 512->24 output architecture")
    if not state or any(not bool(torch.isfinite(value).all().item()) for value in state.values()):
        raise ValueError("checkpoint contains missing or non-finite tensors")


def _load_checkpoint_model(path: Path, device: torch.device, dtype: torch.dtype) -> KireiH3LatentUpscaler3D:
    state = _read_checkpoint(path)
    _validate_checkpoint(state)
    model = KireiH3LatentUpscaler3D()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "checkpoint state does not match Kirei H3 latent upscaler: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    del state
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the Kirei upscaler but is unavailable")
    if name not in {"cuda", "cpu"}:
        raise ValueError(f"unsupported Kirei upscaler device {name!r}")
    return torch.device(name)


class KireiH3LatentUpscalerProvider:
    api_version = H3_LATENT_UPSCALER_API
    kind = H3_LATENT_UPSCALER_KIND

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str = "auto",
        precision: str = "bf16",
        temporal_chunk_size: int = 32,
        temporal_stability: float = 0.0,
        offload_after_upscale: bool = True,
        checkpoint_sha256: str = "",
    ):
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.device_name = str(device)
        self.precision = str(precision)
        self.temporal_chunk_size = int(temporal_chunk_size)
        self.temporal_stability = float(temporal_stability)
        if not 0.0 <= self.temporal_stability <= 1.0:
            raise ValueError("temporal stability must be in [0, 1]")
        self.offload_after_upscale = bool(offload_after_upscale)
        self.checkpoint_sha256 = str(checkpoint_sha256).lower().strip()
        self._model: KireiH3LatentUpscaler3D | None = None
        self.last_run: dict[str, Any] | None = None

    def _get_model(self, device: torch.device, dtype: torch.dtype) -> KireiH3LatentUpscaler3D:
        if self._model is None:
            checkpoint_path = Path(self.checkpoint_path)
            if not self.checkpoint_sha256 and checkpoint_path.is_file():
                self.checkpoint_sha256 = _checkpoint_sha256(checkpoint_path)
            self._model = _load_checkpoint_model(checkpoint_path, device, dtype)
        else:
            self._model = self._model.to(device=device, dtype=dtype)
        return self._model

    def _offload(self, device: torch.device) -> None:
        if not self.offload_after_upscale or self._model is None or device.type != "cuda":
            return
        self._model.to("cpu")
        try:
            import comfy.model_management

            comfy.model_management.soft_empty_cache()
        except Exception:
            torch.cuda.empty_cache()
        gc.collect()

    def upscale_clean_video(
        self,
        video: torch.Tensor,
        *,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(video) or video.ndim != 5:
            raise TypeError("Kirei learned upscaler expects a BxCxTxHxW tensor")
        if video.shape[1] != H3_VIDEO_CHANNELS:
            raise ValueError(f"Kirei learned upscaler expects {H3_VIDEO_CHANNELS} H3 video channels")
        if not video.is_floating_point() or not bool(torch.isfinite(video).all().item()):
            raise ValueError("Kirei learned upscaler input must be finite floating point")
        target_h = int(target_h)
        target_w = int(target_w)
        source_h, source_w = map(int, video.shape[-2:])
        if target_h < source_h or target_w < source_w:
            raise ValueError("Kirei learned upscaler does not support spatial downscaling")
        if (target_h, target_w) == (source_h, source_w):
            return video

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if self.precision not in dtype_map:
            raise ValueError(f"unsupported Kirei upscaler precision {self.precision!r}")
        device = _resolve_device(self.device_name)
        dtype = dtype_map[self.precision]
        if device.type == "cpu" and dtype == torch.float16:
            dtype = torch.float32
        scale = 0.5 * (target_h / source_h + target_w / source_w)
        model = self._get_model(device, dtype)
        mean = torch.tensor(LATENTS_MEAN, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, device=device, dtype=dtype).view(1, -1, 1, 1, 1)

        with torch.inference_mode():
            source = video.to(device=device, dtype=dtype, copy=True)
            normalized = (source - mean) / std
            del source
            output = model(
                normalized,
                target_h=target_h,
                target_w=target_w,
                scale=scale,
                temporal_chunk_size=self.temporal_chunk_size,
            )
            output = _stabilize_temporal_residual(
                output,
                normalized,
                strength=self.temporal_stability,
            )
            output = output * std + mean
            del normalized
            output = output.to(device=video.device, dtype=video.dtype)
        # Leave inference-tensor mode before M3c/M3d may construct an autograd graph.
        output = output.clone()

        if tuple(output.shape[:3]) != tuple(video.shape[:3]) or tuple(output.shape[-2:]) != (
            target_h,
            target_w,
        ):
            raise RuntimeError("Kirei learned upscaler violated the exact B/C/T/target-size contract")
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("Kirei learned upscaler produced NaN/Inf")
        self.last_run = {
            "source_shape": list(video.shape),
            "target_shape": list(output.shape),
            "device": str(device),
            "precision": str(dtype).removeprefix("torch."),
            "temporal_chunk_size": self.temporal_chunk_size,
            "temporal_stability": self.temporal_stability,
            "offload_after_upscale": self.offload_after_upscale,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        self._offload(device)
        return output
