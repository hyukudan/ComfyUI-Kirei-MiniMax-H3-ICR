from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

PERF_API = 1
PERF_SCHEMA = "h3_icr_sampler_performance/v1"
PERF_WRAPPER_KEY = "h3_icr_validation_performance"


@dataclass(slots=True)
class SamplerPerformanceStats:
    wall_seconds: list[float] = field(default_factory=list)
    cuda_peak_allocated_bytes: list[int] = field(default_factory=list)
    cuda_peak_reserved_bytes: list[int] = field(default_factory=list)
    cuda_devices: list[str] = field(default_factory=list)
    cuda_available_calls: int = 0
    cpu_or_unresolved_calls: int = 0

    def record(
        self,
        *,
        wall_seconds: float,
        cuda_device: str | None,
        peak_allocated_bytes: int | None,
        peak_reserved_bytes: int | None,
    ) -> None:
        self.wall_seconds.append(float(wall_seconds))
        if cuda_device is None or peak_allocated_bytes is None or peak_reserved_bytes is None:
            self.cpu_or_unresolved_calls += 1
            return
        self.cuda_available_calls += 1
        self.cuda_devices.append(cuda_device)
        self.cuda_peak_allocated_bytes.append(int(peak_allocated_bytes))
        self.cuda_peak_reserved_bytes.append(int(peak_reserved_bytes))

    def to_dict(self) -> dict[str, Any]:
        first = self.wall_seconds[0] if self.wall_seconds else None
        steady = self.wall_seconds[1:]
        return {
            "api": PERF_API,
            "schema": PERF_SCHEMA,
            "calls": len(self.wall_seconds),
            "first_wall_seconds": first,
            "steady_wall_seconds_mean": sum(steady) / len(steady) if steady else None,
            "steady_wall_seconds_min": min(steady) if steady else None,
            "steady_wall_seconds_max": max(steady) if steady else None,
            "last_wall_seconds": self.wall_seconds[-1] if self.wall_seconds else None,
            "cuda_available_calls": self.cuda_available_calls,
            "cpu_or_unresolved_calls": self.cpu_or_unresolved_calls,
            "cuda_devices": sorted(set(self.cuda_devices)),
            "peak_allocated_bytes_max": max(self.cuda_peak_allocated_bytes) if self.cuda_peak_allocated_bytes else None,
            "peak_reserved_bytes_max": max(self.cuda_peak_reserved_bytes) if self.cuda_peak_reserved_bytes else None,
            "peak_allocated_gib_max": (
                max(self.cuda_peak_allocated_bytes) / (1024**3) if self.cuda_peak_allocated_bytes else None
            ),
            "peak_reserved_gib_max": (
                max(self.cuda_peak_reserved_bytes) / (1024**3) if self.cuda_peak_reserved_bytes else None
            ),
        }


def _first_tensor_device(value: Any) -> torch.device | None:
    if torch.is_tensor(value):
        return value.device
    tensors = getattr(value, "tensors", None)
    if isinstance(tensors, (tuple, list)):
        for child in tensors:
            device = _first_tensor_device(child)
            if device is not None:
                return device
    if isinstance(value, (tuple, list)):
        for child in value:
            device = _first_tensor_device(child)
            if device is not None:
                return device
    if isinstance(value, dict):
        for child in value.values():
            device = _first_tensor_device(child)
            if device is not None:
                return device
    return None


def _resolve_cuda_device(noise: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.device | None:
    for candidate in (noise, args, kwargs):
        device = _first_tensor_device(candidate)
        if device is not None and device.type == "cuda":
            return device
    if not torch.cuda.is_available():
        return None
    try:
        import comfy.model_management

        device = comfy.model_management.get_torch_device()
        if isinstance(device, torch.device) and device.type == "cuda":
            return device
    except Exception:
        return None
    return None


def sampler_performance_wrapper(
    executor,
    guider,
    sigmas,
    extra_args,
    callback,
    noise,
    *args,
    **kwargs,
):
    model_options = extra_args.get("model_options", {}) if isinstance(extra_args, dict) else {}
    transformer = model_options.get("transformer_options", {}) if isinstance(model_options, dict) else {}
    stats = transformer.get("h3_icr_validation_performance_stats") if isinstance(transformer, dict) else None
    if not isinstance(stats, SamplerPerformanceStats):
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)

    cuda_device = _resolve_cuda_device(noise, args, kwargs)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    start = time.perf_counter()
    try:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
    finally:
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            peak_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            device_name = str(cuda_device)
        else:
            peak_allocated = None
            peak_reserved = None
            device_name = None
        stats.record(
            wall_seconds=time.perf_counter() - start,
            cuda_device=device_name,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )


def patch_sampler_performance(model: Any) -> tuple[Any, SamplerPerformanceStats]:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("validation performance patch expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    stats = SamplerPerformanceStats()

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    transformer["h3_icr_validation_performance_stats"] = stats
    options["transformer_options"] = transformer
    patched.model_options = options

    add_wrapper = getattr(patched, "add_wrapper_with_key", None)
    if not callable(add_wrapper):
        raise TypeError("ComfyUI MODEL does not expose add_wrapper_with_key required by validation performance")
    try:
        import comfy.patcher_extension

        wrapper_type = comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE
    except Exception as exc:
        raise RuntimeError("ComfyUI SAMPLER_SAMPLE wrapper API is unavailable") from exc
    add_wrapper(wrapper_type, PERF_WRAPPER_KEY, sampler_performance_wrapper)
    return patched, stats
