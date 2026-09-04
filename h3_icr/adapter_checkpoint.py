from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .base_video_adapter import (
    ADAPTER_API,
    BaseVideoAdapterConfig,
    BaseVideoAdapterProvider,
    StateAwareBaseVideoAdapter,
    h3_architecture_descriptor,
    locate_native_h3,
    patch_base_video_adapter,
)

CHECKPOINT_KIND = "base_video_adapter"
META_PREFIX = "kirei_h3_icr_"
REQUIRED_METADATA = (
    f"{META_PREFIX}api",
    f"{META_PREFIX}kind",
    f"{META_PREFIX}architecture_digest",
    f"{META_PREFIX}model_id",
    f"{META_PREFIX}config_json",
)


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AdapterCheckpointMetadata:
    api: int
    kind: str
    architecture_digest: str
    model_id: str
    config: BaseVideoAdapterConfig
    training: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "kind": self.kind,
            "architecture_digest": self.architecture_digest,
            "model_id": self.model_id,
            "config": self.config.to_dict(),
            "training": self.training,
            "note": self.note,
        }


@dataclass(slots=True)
class ManagedBaseVideoAdapterProvider(BaseVideoAdapterProvider):
    model_patcher: Any = None
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "managed_by_comfyui": self.model_patcher is not None,
                "checkpoint_metadata": self.checkpoint_metadata,
            }
        )
        return payload


def _require_hex_digest(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a 64-character lowercase/uppercase SHA-256 hex digest")
    return value


def _parse_json_object(value: Any, name: str, *, allow_empty: bool = False) -> dict[str, Any]:
    if value in (None, "") and allow_empty:
        return {}
    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {name} JSON: {exc}") from exc
    else:
        raise TypeError(f"{name} metadata must be a JSON object string")
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} metadata must decode to an object")
    return parsed


def parse_adapter_checkpoint_metadata(metadata: dict[str, Any] | None) -> AdapterCheckpointMetadata:
    if not isinstance(metadata, dict):
        raise ValueError("M6 adapter checkpoint is missing safetensors metadata")
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"M6 adapter checkpoint is missing required metadata: {', '.join(missing)}")
    try:
        api = int(metadata[f"{META_PREFIX}api"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid M6 adapter API metadata") from exc
    if api != ADAPTER_API:
        raise ValueError(f"unsupported M6 adapter checkpoint API {api}; expected {ADAPTER_API}")
    kind = str(metadata[f"{META_PREFIX}kind"]).strip()
    if kind != CHECKPOINT_KIND:
        raise ValueError(f"unexpected M6 checkpoint kind {kind!r}")
    architecture_digest = _require_hex_digest(
        metadata[f"{META_PREFIX}architecture_digest"],
        "architecture_digest",
    )
    model_id = str(metadata[f"{META_PREFIX}model_id"]).strip()
    config_raw = _parse_json_object(metadata[f"{META_PREFIX}config_json"], "adapter config")
    allowed = {
        "injection_blocks",
        "adapter_dim",
        "gate_floor",
        "gate_power",
        "temporal_kernel",
        "spatial_kernel",
    }
    unknown = sorted(set(config_raw) - allowed)
    if unknown:
        raise ValueError(f"unknown M6 adapter config fields: {', '.join(unknown)}")
    missing_config = sorted(allowed - set(config_raw))
    if missing_config:
        raise ValueError(f"M6 adapter config is missing fields: {', '.join(missing_config)}")
    blocks = config_raw["injection_blocks"]
    if not isinstance(blocks, (list, tuple)) or not blocks:
        raise ValueError("M6 adapter config injection_blocks must be a non-empty list")
    config = BaseVideoAdapterConfig(
        injection_blocks=tuple(int(value) for value in blocks),
        adapter_dim=int(config_raw["adapter_dim"]),
        gate_floor=float(config_raw["gate_floor"]),
        gate_power=float(config_raw["gate_power"]),
        temporal_kernel=int(config_raw["temporal_kernel"]),
        spatial_kernel=int(config_raw["spatial_kernel"]),
    )
    if tuple(sorted(set(config.injection_blocks))) != config.injection_blocks:
        raise ValueError("M6 checkpoint injection_blocks must be sorted and unique")
    training = _parse_json_object(
        metadata.get(f"{META_PREFIX}training_json", ""),
        "training",
        allow_empty=True,
    )
    note = str(metadata.get(f"{META_PREFIX}note", "")).strip()
    return AdapterCheckpointMetadata(
        api=api,
        kind=kind,
        architecture_digest=architecture_digest,
        model_id=model_id,
        config=config,
        training=training,
        note=note,
    )


def adapter_checkpoint_metadata_for_export(
    provider: BaseVideoAdapterProvider,
    *,
    training: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, str]:
    if not isinstance(provider, BaseVideoAdapterProvider):
        raise TypeError("provider must be a BaseVideoAdapterProvider")
    return {
        f"{META_PREFIX}api": str(provider.api),
        f"{META_PREFIX}kind": CHECKPOINT_KIND,
        f"{META_PREFIX}architecture_digest": provider.architecture_digest,
        f"{META_PREFIX}model_id": str(provider.architecture.get("model_id", "")),
        f"{META_PREFIX}config_json": json.dumps(provider.config.to_dict(), sort_keys=True, separators=(",", ":")),
        f"{META_PREFIX}training_json": json.dumps(training or {}, sort_keys=True, separators=(",", ":")),
        f"{META_PREFIX}note": str(note),
    }


def _state_dict_dtype(state_dict: dict[str, torch.Tensor], fallback: torch.dtype) -> torch.dtype:
    floating = [tensor.dtype for tensor in state_dict.values() if torch.is_tensor(tensor) and tensor.is_floating_point()]
    if not floating:
        return fallback
    first = floating[0]
    return first if all(dtype == first for dtype in floating) else fallback


def _validate_state_dict(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("M6 adapter checkpoint state_dict is empty or invalid")
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError("M6 adapter state_dict must contain only string -> tensor entries")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"M6 adapter tensor {key!r} contains NaN/Inf")
        result[key] = value
    return result


def build_managed_adapter_provider(
    model: Any,
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
    *,
    checkpoint_sha256: str,
    model_patcher_factory: Any = None,
) -> ManagedBaseVideoAdapterProvider:
    inner = locate_native_h3(model)
    parsed = parse_adapter_checkpoint_metadata(metadata)
    architecture = h3_architecture_descriptor(inner, parsed.model_id)
    actual_digest = _digest(architecture)
    if actual_digest != parsed.architecture_digest:
        raise ValueError("M6 checkpoint architecture fingerprint does not match this native H3 MODEL")
    layer_count = int(architecture["layers"])
    if any(block < 0 or block >= layer_count for block in parsed.config.injection_blocks):
        raise ValueError("M6 checkpoint injection blocks are incompatible with this H3 MODEL")
    if architecture["video_channels"] != 24 or tuple(architecture["patch_size"]) != (1, 2, 2):
        raise ValueError("M6 checkpoint loader currently targets native H3 24-channel, 1x2x2 video patches")

    tensors = _validate_state_dict(state_dict)
    module = StateAwareBaseVideoAdapter(
        hidden_size=int(architecture["hidden_size"]),
        latent_channels=int(architecture["video_channels"]),
        patch_size=tuple(architecture["patch_size"]),
        config=parsed.config,
    )
    expected = set(module.state_dict())
    supplied = set(tensors)
    if expected != supplied:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            "M6 checkpoint state_dict keys do not match the adapter architecture; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    module.load_state_dict(tensors, strict=True)

    managed_patcher = None
    if model_patcher_factory is not False:
        if model_patcher_factory is None:
            import comfy.model_management
            import comfy.model_patcher

            load_device = comfy.model_management.get_torch_device()
            offload_device = comfy.model_management.unet_offload_device()
            fallback_dtype = getattr(inner, "dtype", torch.float32)
            if not isinstance(fallback_dtype, torch.dtype):
                fallback_dtype = torch.float32
            module_dtype = _state_dict_dtype(tensors, fallback_dtype)
            module.to(device=offload_device, dtype=module_dtype)
            managed_patcher = comfy.model_patcher.CoreModelPatcher(
                module,
                load_device=load_device,
                offload_device=offload_device,
                size=comfy.model_management.module_size(module),
            )
        else:
            managed_patcher = model_patcher_factory(module)

    return ManagedBaseVideoAdapterProvider(
        module=module,
        config=parsed.config,
        architecture_digest=actual_digest,
        architecture=architecture,
        trained=True,
        checkpoint_sha256=_require_hex_digest(checkpoint_sha256, "checkpoint_sha256"),
        note=parsed.note or "trained M6 adapter checkpoint",
        model_patcher=managed_patcher,
        checkpoint_metadata=parsed.to_dict(),
    )


def load_managed_adapter_checkpoint(model: Any, path: str) -> ManagedBaseVideoAdapterProvider:
    if not isinstance(path, str) or not path:
        raise ValueError("M6 adapter checkpoint path is empty")
    if os.path.splitext(path)[1].lower() not in {".safetensors", ".sft"}:
        raise ValueError("M6 trained adapter loader accepts safetensors checkpoints only")
    import comfy.utils

    state_dict, metadata = comfy.utils.load_torch_file(
        path,
        safe_load=True,
        device=torch.device("cpu"),
        return_metadata=True,
    )
    return build_managed_adapter_provider(
        model,
        state_dict,
        metadata or {},
        checkpoint_sha256=file_sha256(path),
    )


def patch_managed_base_video_adapter(
    model: Any,
    base_video: torch.Tensor,
    provider: ManagedBaseVideoAdapterProvider,
    *,
    strength: float = 1.0,
) -> tuple[Any, Any]:
    if not isinstance(provider, ManagedBaseVideoAdapterProvider) or provider.model_patcher is None:
        raise TypeError("trained M6 provider is not bound to a ComfyUI model patcher")
    patched, runtime = patch_base_video_adapter(model, base_video, provider, strength=strength)
    register = getattr(patched, "set_additional_models", None)
    if not callable(register):
        raise TypeError("ComfyUI MODEL does not expose set_additional_models required by trained M6")
    register("h3_icr_base_video_adapter", [provider.model_patcher])
    return patched, runtime
