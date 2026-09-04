from __future__ import annotations

from typing import Any

from .attention_profile_v2 import topology_digest
from .sparse_attention import SPARSE_RUNTIME_KEY, SPARSE_WRAPPER_KEY, FlexSparseConfig, _locate_native_h3
from .sparse_attention_v2 import (
    FlexSparseOverrideV2,
    FlexSparseRuntimeV2,
    parse_and_validate_policy_v2,
    sparse_diffusion_wrapper_v2,
)


def select_sigma_domain_layers(
    policy: dict[str, Any],
    *,
    branch: str,
    topology: str,
    sigma: float,
    max_sigma_distance: float,
) -> tuple[dict[str, Any], float | None]:
    domains = policy.get("sigma_domains")
    if not isinstance(domains, list) or not domains:
        return {}, None

    selected: dict[int, tuple[float, int, dict[str, Any]]] = {}
    for row in domains:
        if not isinstance(row, dict):
            continue
        if str(row.get("branch", "")) != str(branch):
            continue
        if str(row.get("topology_digest", "")) != str(topology):
            continue
        distance = abs(float(row.get("sigma", -999.0)) - float(sigma))
        if distance > max_sigma_distance:
            continue
        layer = int(row.get("layer", -1))
        if layer < 0:
            continue
        samples = int(row.get("samples", 0))
        candidate = (distance, -samples, row)
        current = selected.get(layer)
        if current is None or candidate[:2] < current[:2]:
            selected[layer] = candidate

    if not selected:
        return {}, None
    layers = {
        str(layer): {
            "samples": int(candidate[2].get("samples", 0)),
            "heads": candidate[2].get("heads", []),
            "sigma": float(candidate[2].get("sigma", 0.0)),
        }
        for layer, candidate in sorted(selected.items())
    }
    max_distance = max(candidate[0] for candidate in selected.values())
    return layers, float(max_distance)


class FlexSparseRuntimeV3(FlexSparseRuntimeV2):
    def __init__(
        self,
        policy: dict[str, Any],
        config: FlexSparseConfig,
        architecture_digest: str,
        *,
        max_policy_sigma_distance: float,
    ):
        super().__init__(policy, config, architecture_digest)
        if not 0.0 <= max_policy_sigma_distance <= 1.0:
            raise ValueError("max_policy_sigma_distance must be in [0, 1]")
        self.max_policy_sigma_distance = float(max_policy_sigma_distance)
        self.sigma_domain_match_calls = 0
        self.sigma_domain_fallback_calls = 0
        self.last_sigma_domain_distance: float | None = None
        self.max_sigma_domain_distance = 0.0
        self._aggregate_layers = dict(policy.get("layers", {}))

    def begin_call(
        self,
        *,
        layout,
        sigma: float,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        patch_h: int,
        patch_w: int,
        branch: str,
    ) -> None:
        topology = topology_digest(layout)
        layers, distance = select_sigma_domain_layers(
            self.policy,
            branch=branch,
            topology=topology,
            sigma=sigma,
            max_sigma_distance=self.max_policy_sigma_distance,
        )
        self.policy["layers"] = layers
        if layers:
            self.sigma_domain_match_calls += 1
            self.last_sigma_domain_distance = float(distance or 0.0)
            self.max_sigma_domain_distance = max(self.max_sigma_domain_distance, self.last_sigma_domain_distance)
        else:
            self.sigma_domain_fallback_calls += 1
            self.last_sigma_domain_distance = None
        try:
            super().begin_call(
                layout=layout,
                sigma=sigma,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                patch_h=patch_h,
                patch_w=patch_w,
                branch=branch,
            )
        except BaseException:
            self.policy["layers"] = self._aggregate_layers
            raise

    def end_call(self) -> None:
        try:
            super().end_call()
        finally:
            self.policy["layers"] = self._aggregate_layers

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["api"] = 3
        report["sigma_domain_bound"] = True
        report["max_policy_sigma_distance"] = self.max_policy_sigma_distance
        report["stats"]["sigma_domain_match_calls"] = self.sigma_domain_match_calls
        report["stats"]["sigma_domain_fallback_calls"] = self.sigma_domain_fallback_calls
        report["stats"]["last_sigma_domain_distance"] = self.last_sigma_domain_distance
        report["stats"]["max_sigma_domain_distance"] = self.max_sigma_domain_distance
        return report


def parse_and_validate_policy_v3(
    policy_json: str,
    *,
    inner: Any,
    model_id: str,
    profile_json: str = "",
) -> tuple[dict[str, Any], str]:
    policy, architecture_digest = parse_and_validate_policy_v2(
        policy_json,
        inner=inner,
        model_id=model_id,
        profile_json=profile_json,
    )
    domains = policy.get("sigma_domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError(
            "Flex sparse v3 requires an M5 v3 policy with sigma_domains; "
            "run the current attention profiler/report before sparse execution"
        )
    for row in domains:
        if not isinstance(row, dict):
            raise ValueError("sigma_domains contains a non-object entry")
        required = ("branch", "topology_digest", "sigma", "layer", "heads")
        if any(key not in row for key in required):
            raise ValueError("sigma domain is missing required calibration fields")
    return policy, architecture_digest


def patch_flex_sparse_attention_v3(
    model: Any,
    *,
    policy_json: str,
    profile_json: str = "",
    model_id: str = "",
    dense_tail_sigma: float = 0.12,
    local_t_radius: int = 1,
    local_y_radius: int = 2,
    local_x_radius: int = 2,
    temporal_radius: int = 2,
    block_size: int = 128,
    min_block_sparsity: float = 5.0,
    max_policy_sigma_distance: float = 0.03,
    force_flex_kernel: bool = True,
) -> tuple[Any, FlexSparseRuntimeV3]:
    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError("Flex sparse backend expects a ComfyUI MODEL/ModelPatcher")
    patched = clone()
    if patched is model:
        raise RuntimeError("MODEL.clone() returned original object")
    inner = _locate_native_h3(patched)
    policy, architecture_digest = parse_and_validate_policy_v3(
        policy_json,
        inner=inner,
        model_id=str(model_id).strip(),
        profile_json=profile_json,
    )
    config = FlexSparseConfig(
        dense_tail_sigma=float(dense_tail_sigma),
        local_t_radius=int(local_t_radius),
        local_y_radius=int(local_y_radius),
        local_x_radius=int(local_x_radius),
        temporal_radius=int(temporal_radius),
        block_size=int(block_size),
        min_block_sparsity=float(min_block_sparsity),
        force_flex_kernel=bool(force_flex_kernel),
        model_id=str(model_id).strip(),
    )
    runtime = FlexSparseRuntimeV3(
        policy,
        config,
        architecture_digest,
        max_policy_sigma_distance=float(max_policy_sigma_distance),
    )

    options = dict(getattr(patched, "model_options", {}))
    transformer = dict(options.get("transformer_options", {}))
    previous = transformer.get("optimized_attention_override")
    if previous is not None and hasattr(previous, "container_function"):
        raise RuntimeError("Flex sparse backend cannot chain a container-style optimized_attention_override")
    transformer["optimized_attention_override"] = FlexSparseOverrideV2(runtime, previous)
    transformer[SPARSE_RUNTIME_KEY] = runtime

    wrappers = dict(transformer.get("wrappers", {}))
    diffusion = dict(wrappers.get("diffusion_model", {}))
    diffusion.pop(SPARSE_WRAPPER_KEY, None)
    diffusion[SPARSE_WRAPPER_KEY] = [sparse_diffusion_wrapper_v2]
    wrappers["diffusion_model"] = diffusion
    transformer["wrappers"] = wrappers
    options["transformer_options"] = transformer
    patched.model_options = options
    return patched, runtime
