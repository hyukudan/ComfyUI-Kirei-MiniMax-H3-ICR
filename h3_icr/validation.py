from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any

import torch

VALIDATION_API = 1
VALIDATION_SCHEMA = "h3_icr_validation/v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _callable_descriptor(value: Any) -> dict[str, str]:
    return {
        "module": str(getattr(value, "__module__", type(value).__module__)),
        "qualname": str(getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))),
    }


def _tensor_bytes_digest(tensor: torch.Tensor, *, chunk_bytes: int = 1024 * 1024) -> str:
    if getattr(tensor, "is_nested", False):
        children = [_tensor_descriptor(child) for child in tensor.unbind()]
        return _digest({"nested": children})
    if tensor.layout != torch.strided:
        raise TypeError(f"validation tensor hashing supports strided/nested tensors, got {tensor.layout}")
    cpu = tensor.detach().contiguous().cpu()
    if cpu.is_floating_point() and not bool(torch.isfinite(cpu).all().item()):
        raise ValueError("validation tensor contains NaN/Inf")
    raw = cpu.view(torch.uint8).flatten()
    digest = hashlib.sha256()
    try:
        array = raw.numpy()
        digest.update(memoryview(array).cast("B"))
    except Exception:
        chunk_elems = max(1, int(chunk_bytes))
        for start in range(0, int(raw.numel()), chunk_elems):
            digest.update(bytes(raw[start : start + chunk_elems].tolist()))
    return digest.hexdigest()


def _tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    if getattr(tensor, "is_nested", False):
        children = [_tensor_descriptor(child) for child in tensor.unbind()]
        return {
            "type": "nested_tensor",
            "dtype": str(tensor.dtype),
            "children": children,
            "sha256": _digest(children),
        }
    return {
        "type": "tensor",
        "dtype": str(tensor.dtype),
        "shape": [int(value) for value in tensor.shape],
        "numel": int(tensor.numel()),
        "sha256": _tensor_bytes_digest(tensor),
    }


def canonical_descriptor(value: Any, *, strict: bool = True, path: str = "$") -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}")
        return value
    if torch.is_tensor(value):
        return _tensor_descriptor(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"type": "bytes", "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        result = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, (str, int, float, bool)):
                if strict:
                    raise TypeError(f"unsupported dict key {type(key)!r} at {path}")
                key = f"<{type(key).__module__}.{type(key).__qualname__}>"
            result[str(key)] = canonical_descriptor(value[key], strict=strict, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            canonical_descriptor(item, strict=strict, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if dataclasses.is_dataclass(value):
        return canonical_descriptor(dataclasses.asdict(value), strict=strict, path=path)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return canonical_descriptor(to_dict(), strict=strict, path=path)
    if callable(value):
        descriptor = _callable_descriptor(value)
        closure = getattr(value, "__closure__", None)
        if strict and closure:
            raise TypeError(
                f"unsupported callable closure at {path}: {descriptor['module']}.{descriptor['qualname']}"
            )
        return {"type": "callable", **descriptor}
    if strict:
        raise TypeError(
            f"unsupported validation object at {path}: {type(value).__module__}.{type(value).__qualname__}"
        )
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def fingerprint(value: Any, *, strict: bool = True, path: str = "$") -> dict[str, Any]:
    descriptor = canonical_descriptor(value, strict=strict, path=path)
    return {"sha256": _digest(descriptor), "descriptor": descriptor}


def parse_json_object(value: str, name: str) -> dict[str, Any]:
    if not str(value).strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} JSON must contain an object")
    canonical_descriptor(parsed, strict=True, path=f"$.{name}")
    return parsed


def noise_descriptor(noise: Any, *, strict: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "class": f"{type(noise).__module__}.{type(noise).__qualname__}",
    }
    if hasattr(noise, "seed"):
        payload["seed"] = int(noise.seed)
    state = getattr(noise, "__dict__", None)
    if isinstance(state, dict):
        safe_state = {key: value for key, value in state.items() if key != "seed"}
        if safe_state:
            payload["state"] = canonical_descriptor(safe_state, strict=strict, path="$.noise.state")
    return payload


def sampler_descriptor(sampler: Any, *, strict: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "class": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
    }
    sampler_function = getattr(sampler, "sampler_function", None)
    if callable(sampler_function):
        payload["sampler_function"] = _callable_descriptor(sampler_function)
    for key in ("extra_options", "inpaint_options"):
        if hasattr(sampler, key):
            payload[key] = canonical_descriptor(getattr(sampler, key), strict=strict, path=f"$.sampler.{key}")
    state = getattr(sampler, "__dict__", None)
    if isinstance(state, dict):
        known = {"sampler_function", "extra_options", "inpaint_options"}
        remainder = {key: value for key, value in state.items() if key not in known}
        if remainder:
            payload["state"] = canonical_descriptor(remainder, strict=strict, path="$.sampler.state")
    return payload


def _strip_runtime_stats(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_runtime_stats(child)
            for key, child in value.items()
            if key not in {"stats", "calls", "applied", "model_calls", "attention_calls"}
        }
    if isinstance(value, list):
        return [_strip_runtime_stats(child) for child in value]
    return value


def _runtime_descriptor(runtime: Any) -> dict[str, Any] | None:
    if runtime is None:
        return None
    report = getattr(runtime, "report", None)
    if callable(report):
        try:
            return canonical_descriptor(_strip_runtime_stats(report()), strict=True, path="$.runtime")
        except Exception:
            pass
    config = getattr(runtime, "config", None)
    payload: dict[str, Any] = {"class": f"{type(runtime).__module__}.{type(runtime).__qualname__}"}
    if config is not None:
        payload["config"] = canonical_descriptor(config, strict=True, path="$.runtime.config")
    return payload


def handle_descriptor(handle: Any) -> dict[str, Any] | None:
    if handle is None:
        return None
    if not isinstance(handle, dict):
        return canonical_descriptor(handle, strict=True, path="$.handle")
    payload: dict[str, Any] = {}
    for key in ("api", "decoder_kind"):
        if key in handle:
            payload[key] = canonical_descriptor(handle[key], strict=True, path=f"$.handle.{key}")
    for key in ("config", "prior_schedule"):
        if key in handle:
            payload[key] = canonical_descriptor(handle[key], strict=True, path=f"$.handle.{key}")
    if "runtime" in handle:
        payload["runtime"] = _runtime_descriptor(handle["runtime"])
    return payload


def model_research_descriptor(model: Any) -> dict[str, Any]:
    options = getattr(model, "model_options", None)
    if not isinstance(options, dict):
        return {}
    transformer = options.get("transformer_options", {})
    if not isinstance(transformer, dict):
        return {}
    result: dict[str, Any] = {}
    backend = transformer.get("h3_icr_backend")
    if isinstance(backend, dict):
        result["backend"] = canonical_descriptor(backend, strict=True, path="$.model.backend")
    known_config_keys = (
        "h3_icr_tiled_renderer",
        "h3_icr_tiled_prior_schedule",
    )
    for key in known_config_keys:
        value = transformer.get(key)
        if value is not None:
            result[key] = canonical_descriptor(value, strict=True, path=f"$.model.{key}")
    runtime_keys = (
        "h3_icr_sparse_runtime",
        "h3_icr_base_video_adapter_runtime",
    )
    for key in runtime_keys:
        value = transformer.get(key)
        descriptor = _runtime_descriptor(value)
        if descriptor is not None:
            result[key] = descriptor
    return result


def _backend_descriptor(model: Any, backend: Any) -> dict[str, Any]:
    if backend is not None:
        return canonical_descriptor(backend, strict=True, path="$.backend")
    research = model_research_descriptor(model)
    raw = research.get("backend")
    if raw is not None:
        return raw
    return {
        "kind": "unknown",
        "checkpoint_format": "unknown",
        "checkpoint_sha256": "",
        "overlay_sha256": "",
    }


def build_validation_manifest(
    *,
    experiment_name: str,
    comparison_group: str,
    arm: str,
    model: Any,
    base_latent: dict[str, Any],
    positive: Any,
    negative: Any,
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    locked_settings: dict[str, Any],
    arm_settings: dict[str, Any],
    strict_hashing: bool = True,
    backend: Any = None,
    measurement: Any = None,
    posterior: Any = None,
    pixel_measurement: Any = None,
    renderer: Any = None,
    sparse_runtime: Any = None,
    adapter: Any = None,
) -> dict[str, Any]:
    if not isinstance(base_latent, dict) or "samples" not in base_latent:
        raise TypeError("validation base_latent must be a ComfyUI LATENT containing samples")
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise TypeError("validation sigmas must be a one-dimensional tensor with at least two entries")
    if not str(experiment_name).strip() or not str(comparison_group).strip() or not str(arm).strip():
        raise ValueError("validation experiment_name, comparison_group and arm must be non-empty")

    noise_desc = noise_descriptor(noise, strict=strict_hashing)
    sampler_desc = sampler_descriptor(sampler, strict=strict_hashing)
    features = {
        "measurement_m3b": handle_descriptor(measurement),
        "posterior_m3c": handle_descriptor(posterior),
        "pixel_measurement_m3d": handle_descriptor(pixel_measurement),
        "renderer_m4": handle_descriptor(renderer),
        "sparse_m5": handle_descriptor(sparse_runtime),
        "adapter_m6": handle_descriptor(adapter),
    }
    features = {key: value for key, value in features.items() if value is not None}
    research = model_research_descriptor(model)
    research.pop("backend", None)

    manifest: dict[str, Any] = {
        "api": VALIDATION_API,
        "schema": VALIDATION_SCHEMA,
        "experiment": {
            "name": str(experiment_name).strip(),
            "comparison_group": str(comparison_group).strip(),
            "arm": str(arm).strip(),
        },
        "locks": {
            "base_latent": fingerprint(base_latent["samples"], strict=strict_hashing, path="$.base_latent.samples"),
            "positive_conditioning": fingerprint(positive, strict=strict_hashing, path="$.positive"),
            "negative_conditioning": fingerprint(negative, strict=strict_hashing, path="$.negative"),
            "sigmas": fingerprint(sigmas, strict=True, path="$.sigmas"),
            "noise": {"sha256": _digest(noise_desc), "descriptor": noise_desc},
            "sampler": {"sha256": _digest(sampler_desc), "descriptor": sampler_desc},
            "locked_settings": {
                "sha256": _digest(locked_settings),
                "descriptor": canonical_descriptor(locked_settings, strict=True, path="$.locked_settings"),
            },
        },
        "arm": {
            "backend": _backend_descriptor(model, backend),
            "features": features,
            "model_research": research,
            "settings": canonical_descriptor(arm_settings, strict=True, path="$.arm_settings"),
        },
    }
    warnings: list[str] = []
    backend_desc = manifest["arm"]["backend"]
    if isinstance(backend_desc, dict):
        if not str(backend_desc.get("checkpoint_sha256", "")).strip():
            warnings.append("backend checkpoint_sha256 is empty; exact model-weight provenance is incomplete")
        kind = str(backend_desc.get("kind", ""))
        if kind.startswith("hybrid") and not str(backend_desc.get("overlay_sha256", "")).strip():
            warnings.append("Hybrid overlay_sha256 is empty; exact overlay provenance is incomplete")
    if warnings:
        manifest["warnings"] = warnings
    manifest["run_id"] = _digest(manifest)
    return manifest


def validate_manifest_integrity(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise TypeError("validation manifest must be a dictionary")
    if manifest.get("api") != VALIDATION_API or manifest.get("schema") != VALIDATION_SCHEMA:
        raise ValueError("unsupported H3 ICR validation manifest schema/API")
    expected = manifest.get("run_id")
    core = dict(manifest)
    core.pop("run_id", None)
    actual = _digest(core)
    if expected != actual:
        raise ValueError("validation manifest run_id does not match its content")


def _diff_values(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "a": left, "b": right}]
    if isinstance(left, dict):
        diffs: list[dict[str, Any]] = []
        keys = sorted(set(left) | set(right))
        for key in keys:
            child_path = f"{path}.{key}"
            if key not in left:
                diffs.append({"path": child_path, "a": "<missing>", "b": right[key]})
            elif key not in right:
                diffs.append({"path": child_path, "a": left[key], "b": "<missing>"})
            else:
                diffs.extend(_diff_values(left[key], right[key], child_path))
        return diffs
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path, "a": left, "b": right}]
        diffs: list[dict[str, Any]] = []
        for index, (a_value, b_value) in enumerate(zip(left, right)):
            diffs.extend(_diff_values(a_value, b_value, f"{path}[{index}]"))
        return diffs
    if left != right:
        return [{"path": path, "a": left, "b": right}]
    return []


def parse_allowed_differences(value: str) -> tuple[str, ...]:
    normalized = str(value).replace("\n", ",").replace(";", ",")
    paths = [item.strip() for item in normalized.split(",") if item.strip()]
    result = []
    for path in paths:
        if not path.startswith("$."):
            path = f"$.{path.lstrip('.')}"
        if path not in result:
            result.append(path)
    return tuple(result)


def _path_allowed(path: str, allowed: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[") for prefix in allowed)


def compare_validation_manifests(
    manifest_a: dict[str, Any],
    manifest_b: dict[str, Any],
    *,
    allowed_differences: str = "",
) -> dict[str, Any]:
    validate_manifest_integrity(manifest_a)
    validate_manifest_integrity(manifest_b)
    exp_a = manifest_a.get("experiment", {})
    exp_b = manifest_b.get("experiment", {})
    if exp_a.get("name") != exp_b.get("name"):
        raise ValueError("validation manifests belong to different experiment names")
    if exp_a.get("comparison_group") != exp_b.get("comparison_group"):
        raise ValueError("validation manifests belong to different comparison groups")

    left = dict(manifest_a)
    right = dict(manifest_b)
    left.pop("run_id", None)
    right.pop("run_id", None)
    left.pop("warnings", None)
    right.pop("warnings", None)
    allowed = ("$.experiment.arm",) + parse_allowed_differences(allowed_differences)
    diffs = _diff_values(left, right)
    accepted = [diff for diff in diffs if _path_allowed(diff["path"], allowed)]
    unexpected = [diff for diff in diffs if not _path_allowed(diff["path"], allowed)]
    lock_diffs = [diff for diff in diffs if diff["path"].startswith("$.locks")]
    return {
        "api": VALIDATION_API,
        "schema": VALIDATION_SCHEMA,
        "experiment": exp_a.get("name"),
        "comparison_group": exp_a.get("comparison_group"),
        "arm_a": exp_a.get("arm"),
        "arm_b": exp_b.get("arm"),
        "run_id_a": manifest_a.get("run_id"),
        "run_id_b": manifest_b.get("run_id"),
        "compatible": len(unexpected) == 0,
        "locks_identical": len(lock_diffs) == 0,
        "allowed_paths": list(allowed),
        "accepted_differences": accepted,
        "unexpected_differences": unexpected,
    }
