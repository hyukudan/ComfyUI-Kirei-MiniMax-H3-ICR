import math
from types import SimpleNamespace

import torch

from h3_icr.attention_profile import AttentionProfileConfig
from h3_icr.attention_profile_v2 import (
    AttentionProfileRuntimeV2,
    AttentionProfilerOverrideV2,
    exact_video_pair_affinity,
    propose_attention_policy_v2,
)


def _layout():
    return SimpleNamespace(
        seq_len=12,
        signature=(2, 2, 4, 4, 1),
        segments=[
            (0, 2, "text"),
            (2, 4, "audio"),
            (4, 12, "video"),
        ],
    )


def _runtime(layer_stride=1):
    return AttentionProfileRuntimeV2(
        AttentionProfileConfig(
            layer_stride=layer_stride,
            query_samples=8,
            key_samples_per_modality=8,
            sigma_decimals=3,
            max_buckets=32,
            model_id="synthetic",
        ),
        {
            "model_id": "synthetic",
            "module": "comfy.ldm.minimax.model",
            "class": "MiniMaxH3Model",
            "layers": 1,
            "hidden_size": 8,
            "patch_size": (1, 2, 2),
        },
    )


def _begin(runtime):
    runtime.begin_call(
        layout=_layout(),
        sigma=0.5,
        branch="dense",
        latent_t=2,
        latent_h=4,
        latent_w=4,
        patch_h=2,
        patch_w=2,
    )


def test_profiler_override_is_passive_and_records_metrics():
    torch.manual_seed(5)
    runtime = _runtime()
    _begin(runtime)
    q = torch.randn(1, 2, 12, 4)
    k = torch.randn(1, 2, 12, 4)
    v = torch.randn(1, 2, 12, 4)

    def original(q_in, k_in, v_in, heads, **kwargs):
        assert heads == 2
        return q_in + 2.0 * k_in + 3.0 * v_in

    expected = original(q, k, v, 2)
    override = AttentionProfilerOverrideV2(runtime)
    actual = override(
        original,
        q,
        k,
        v,
        2,
        skip_reshape=True,
        transformer_options={},
    )
    runtime.end_call()

    assert torch.equal(actual, expected)
    report = runtime.report()
    assert runtime.attention_calls_seen == 1
    assert runtime.attention_calls_profiled == 1
    assert len(report["buckets"]) == 1
    assert report["buckets"][0]["video_pair_samples"] == 1
    assert "spatial_minus_far" in report["buckets"][0]["video_pair_affinity"]


def test_exact_pair_affinity_detects_spatial_neighbor_advantage_over_far_key():
    layout = SimpleNamespace(
        seq_len=36,
        signature=(2, 2, 8, 8, 1),
        segments=[
            (0, 2, "text"),
            (2, 4, "audio"),
            (4, 36, "video"),
        ],
    )
    runtime = _runtime()
    runtime.begin_call(
        layout=layout,
        sigma=0.5,
        branch="dense",
        latent_t=2,
        latent_h=8,
        latent_w=8,
        patch_h=2,
        patch_w=2,
    )

    q = torch.zeros(1, 1, 36, 6)
    k = torch.zeros_like(q)
    row = 4
    for t in range(2):
        for y in range(4):
            for x in range(4):
                feature = torch.tensor(
                    [
                        math.cos(2.0 * math.pi * x / 4.0),
                        math.sin(2.0 * math.pi * x / 4.0),
                        math.cos(2.0 * math.pi * y / 4.0),
                        math.sin(2.0 * math.pi * y / 4.0),
                        1.0 if t == 0 else -1.0,
                        0.5,
                    ],
                    dtype=torch.float32,
                )
                q[0, 0, row] = feature
                k[0, 0, row] = feature
                row += 1

    pair = exact_video_pair_affinity(q, k, runtime)
    runtime.end_call()

    assert pair
    assert pair["spatial_neighbor_score"][0] > pair["far_video_score"][0]
    assert pair["spatial_minus_far"][0] > 0.0


def test_policy_v2_contains_exact_pair_evidence_and_remains_proposal_only():
    runtime = _runtime()
    _begin(runtime)
    q = torch.randn(1, 2, 12, 4)
    k = torch.randn(1, 2, 12, 4)
    runtime.observe(q, k, 2)
    runtime.end_call()

    report = runtime.report()
    proposal = propose_attention_policy_v2(report)

    assert proposal["status"] == "proposal_only_no_sparse_kernel_enabled"
    assert proposal["late_dense_required"] is True
    assert proposal["dense_fallback_required"] is True
    assert proposal["exact_pair_evidence"] is True
    assert proposal["pair_margin_threshold"] == 0.05
    assert proposal["proposal_digest"]
