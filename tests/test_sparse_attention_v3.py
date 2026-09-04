from types import SimpleNamespace

from h3_icr.sparse_attention import FlexSparseConfig
from h3_icr.sparse_attention_v3 import FlexSparseRuntimeV3, select_sigma_domain_layers


def _policy():
    return {
        "layers": {"0": {"heads": [{"head": 0, "classification": "mixed_dense"}]}},
        "sigma_domains": [
            {
                "branch": "dense",
                "topology_digest": "topology",
                "sigma": 0.70,
                "layer": 0,
                "samples": 4,
                "seq_len": 20,
                "heads": [{"head": 0, "classification": "local_3d_pair_candidate"}],
            },
            {
                "branch": "dense",
                "topology_digest": "topology",
                "sigma": 0.30,
                "layer": 0,
                "samples": 4,
                "seq_len": 20,
                "heads": [{"head": 0, "classification": "mixed_dense"}],
            },
        ],
        "calibrated_topologies": {"dense": {"digest": "topology", "descriptor": {}}},
    }


def test_select_sigma_domain_uses_nearest_calibrated_coordinate():
    layers, distance = select_sigma_domain_layers(
        _policy(),
        branch="dense",
        topology="topology",
        sigma=0.69,
        max_sigma_distance=0.03,
    )
    assert distance < 0.02
    assert layers["0"]["sigma"] == 0.70
    assert layers["0"]["heads"][0]["classification"] == "local_3d_pair_candidate"


def test_select_sigma_domain_fails_closed_outside_tolerance_or_topology():
    layers, distance = select_sigma_domain_layers(
        _policy(),
        branch="dense",
        topology="topology",
        sigma=0.50,
        max_sigma_distance=0.03,
    )
    assert layers == {}
    assert distance is None

    layers, _ = select_sigma_domain_layers(
        _policy(),
        branch="dense",
        topology="other",
        sigma=0.70,
        max_sigma_distance=0.03,
    )
    assert layers == {}


def test_runtime_restores_aggregate_layers_after_each_call(monkeypatch):
    import h3_icr.sparse_attention_v3 as sparse_v3

    runtime = FlexSparseRuntimeV3(
        _policy(),
        FlexSparseConfig(),
        architecture_digest="arch",
        max_policy_sigma_distance=0.03,
    )
    monkeypatch.setattr(sparse_v3, "topology_digest", lambda layout: layout.digest)
    layout = SimpleNamespace(digest="topology", seq_len=20)

    runtime.begin_call(
        layout=layout,
        sigma=0.70,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        patch_h=2,
        patch_w=2,
        branch="dense",
    )
    assert runtime.policy["layers"]["0"]["heads"][0]["classification"] == "local_3d_pair_candidate"
    runtime.end_call()
    assert runtime.policy["layers"]["0"]["heads"][0]["classification"] == "mixed_dense"
    assert runtime.sigma_domain_match_calls == 1

    runtime.begin_call(
        layout=layout,
        sigma=0.50,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        patch_h=2,
        patch_w=2,
        branch="dense",
    )
    assert runtime.policy["layers"] == {}
    runtime.end_call()
    assert runtime.sigma_domain_fallback_calls == 1
