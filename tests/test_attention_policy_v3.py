from h3_icr.attention_policy_v3 import propose_attention_policy_v3


def _bucket(sigma, spatial_margin, temporal_margin):
    return {
        "branch": "dense",
        "sigma": sigma,
        "layer": 0,
        "count": 2,
        "seq_len": 20,
        "heads": 1,
        "modality_rows": {"target_video": 8},
        "q_norm": {"target_video": [1.0]},
        "k_norm": {"target_video": [1.0]},
        "q_to_k_mass": {"target_video": {"target_video": [0.9]}},
        "video_structure": {
            "same_frame_mass": [0.8],
            "spatial_r2_same_frame_mass": [0.8],
            "temporal_r1_same_spatial_mass": [0.3],
            "local_3d_r1_mass": [0.6],
        },
        "video_pair_affinity": {
            "spatial_minus_far": [spatial_margin],
            "temporal_minus_far": [temporal_margin],
        },
        "video_pair_samples": 2,
    }


def test_v3_policy_keeps_separate_sigma_domains_for_same_topology():
    report = {
        "profile_digest": "profile",
        "architecture_digest": "arch",
        "config": {"sigma_decimals": 3},
        "calibrated_topologies": {
            "dense": {"digest": "topology", "descriptor": {"signature": [2, 2, 4, 4, 1]}}
        },
        "buckets": [
            _bucket(0.7, 0.20, 0.20),
            _bucket(0.3, 0.00, 0.00),
        ],
    }
    policy = propose_attention_policy_v3(report)
    assert policy["api"] == 3
    assert policy["policy_domain_schema"] == "branch+topology+sigma+layer"
    assert policy["sigma_domain_count"] == 2
    high, low = policy["sigma_domains"]
    assert high["sigma"] == 0.7
    assert low["sigma"] == 0.3
    assert high["topology_digest"] == "topology"
    assert high["heads"][0]["classification"] == "local_3d_pair_candidate"
    assert low["heads"][0]["classification"] == "mixed_dense"
    assert policy["proposal_digest"]
