from __future__ import annotations

from copy import deepcopy
from typing import Any

from .attention_profile import _canonical_digest, propose_attention_policy
from .attention_profile_v2 import propose_attention_policy_v2


def _pair_refine_head_rows(bucket: dict[str, Any], rows: list[dict[str, Any]], threshold: float) -> None:
    pair = bucket.get("video_pair_affinity", {})
    spatial = pair.get("spatial_minus_far", [])
    temporal = pair.get("temporal_minus_far", [])
    for row in rows:
        head = int(row.get("head", -1))
        if head < 0:
            continue
        spatial_margin = float(spatial[head]) if head < len(spatial) else 0.0
        temporal_margin = float(temporal[head]) if head < len(temporal) else 0.0
        row["exact_spatial_minus_far"] = spatial_margin
        row["exact_temporal_minus_far"] = temporal_margin
        row["classification_basis"] = "sigma_domain_sampled_mass_plus_exact_qk_pair_margin"

        self_mass = float(row.get("target_video_mass", 0.0))
        if self_mass < 0.55:
            classification = "global_or_cross_modal"
        elif spatial_margin > threshold and temporal_margin > threshold:
            classification = "local_3d_pair_candidate"
        elif spatial_margin > threshold:
            classification = "spatial_pair_candidate"
        elif temporal_margin > threshold:
            classification = "temporal_pair_candidate"
        else:
            classification = "mixed_dense"
        row["classification"] = classification


def _domain_heads(report: dict[str, Any], bucket: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    layer = str(int(bucket["layer"]))
    single_report = {
        "architecture_digest": report.get("architecture_digest", ""),
        "profile_digest": report.get("profile_digest", ""),
        "config": report.get("config", {}),
        "buckets": [bucket],
    }
    base = propose_attention_policy(single_report)
    layer_row = base.get("layers", {}).get(layer)
    if not isinstance(layer_row, dict):
        return []
    rows = deepcopy(layer_row.get("heads", []))
    _pair_refine_head_rows(bucket, rows, threshold)
    return rows


def propose_attention_policy_v3(report: dict[str, Any]) -> dict[str, Any]:
    """Add branch/topology/sigma/layer domains to the topology-bound v2 proposal."""
    proposal = propose_attention_policy_v2(report)
    topologies = proposal.get("calibrated_topologies", {})
    if not isinstance(topologies, dict) or not topologies:
        raise ValueError("M5 v3 policy requires v2 calibrated_topologies")

    threshold = float(proposal.get("pair_margin_threshold", 0.05))
    domains = []
    for bucket in report.get("buckets", []):
        if not isinstance(bucket, dict):
            continue
        branch = str(bucket.get("branch", ""))
        topology = topologies.get(branch)
        if not isinstance(topology, dict) or not topology.get("digest"):
            raise ValueError(f"missing calibrated topology for attention branch {branch!r}")
        heads = _domain_heads(report, bucket, threshold)
        if not heads:
            continue
        domains.append(
            {
                "branch": branch,
                "topology_digest": str(topology["digest"]),
                "sigma": float(bucket.get("sigma", 0.0)),
                "layer": int(bucket.get("layer", 0)),
                "samples": int(bucket.get("count", 0)),
                "seq_len": int(bucket.get("seq_len", 0)),
                "heads": heads,
            }
        )

    domains.sort(key=lambda row: (row["branch"], -row["sigma"], row["layer"]))
    proposal["api"] = 3
    proposal["policy_domain_schema"] = "branch+topology+sigma+layer"
    proposal["sigma_decimals"] = int(report.get("config", {}).get("sigma_decimals", 3))
    proposal["sigma_domains"] = domains
    proposal["sigma_domain_count"] = len(domains)
    proposal["warning"] = (
        "Sigma-domain policies are calibration proposals only. Runtime execution must fail back to dense "
        "outside the calibrated branch/topology/sigma domain and still requires decoded-media validation."
    )
    proposal["proposal_digest"] = _canonical_digest(
        {key: value for key, value in proposal.items() if key != "proposal_digest"}
    )
    return proposal
