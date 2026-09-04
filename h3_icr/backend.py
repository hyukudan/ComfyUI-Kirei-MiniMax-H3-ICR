from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BACKEND_API = 1
BACKENDS = (
    "unknown",
    "fl2va_reference",
    "hybrid_late_adaln",
    "ref2va_reference",
    "hybrid_all_adaln",
)


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    kind: str = "unknown"
    checkpoint_format: str = "unknown"
    checkpoint_sha256: str = ""
    overlay_sha256: str = ""
    note: str = ""
    api: int = BACKEND_API

    def __post_init__(self) -> None:
        if self.api != BACKEND_API:
            raise ValueError(f"Unsupported backend descriptor API {self.api}")
        if self.kind not in BACKENDS:
            raise ValueError(f"Unsupported H3 ICR backend kind: {self.kind}")
        if self.checkpoint_format not in {"unknown", "pruned", "full"}:
            raise ValueError("checkpoint_format must be unknown, pruned, or full")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tag_model(model: Any, descriptor: BackendDescriptor) -> Any:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("H3 ICR backend tagging expects a ComfyUI MODEL/ModelPatcher")
    tagged = clone()
    if tagged is model:
        raise RuntimeError("MODEL.clone() returned the original object")
    options = getattr(tagged, "model_options", None)
    if not isinstance(options, dict):
        raise TypeError("MODEL.model_options must be a dictionary")
    options = dict(options)
    transformer = options.get("transformer_options", {})
    if not isinstance(transformer, dict):
        raise TypeError("MODEL transformer_options must be a dictionary")
    transformer = dict(transformer)
    transformer["h3_icr_backend"] = descriptor.to_dict()
    options["transformer_options"] = transformer
    tagged.model_options = options
    return tagged


def descriptor_from_model(model: Any) -> BackendDescriptor:
    options = getattr(model, "model_options", None)
    if not isinstance(options, dict):
        return BackendDescriptor()
    transformer = options.get("transformer_options", {})
    raw = transformer.get("h3_icr_backend") if isinstance(transformer, dict) else None
    if not isinstance(raw, dict):
        return BackendDescriptor()
    try:
        return BackendDescriptor(**raw)
    except (TypeError, ValueError):
        return BackendDescriptor(note="invalid backend metadata on MODEL")
