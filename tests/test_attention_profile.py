from types import SimpleNamespace

import torch

from h3_icr.attention_profile import (
    AttentionProfileConfig,
    AttentionProfileRuntime,
    AttentionProfilerOverride,
    _modality_indices,
    propose_attention_policy,
)


def _layout():
    return SimpleNamespace(
        seq_len=14,
        segments=[(0, 2, "text"), (2, 4, "ref_img"), (4, 6, "audio"), (6, 14, "video")],
    )


def test_modality_sampling_handles_non_contiguous_reference_segments():
    layout = SimpleNamespace(
        seq_len=12,
        segments=[(0, 2, "text"), (2, 4, "ref_img"), (4, 6, "audio"), (6, 8, "ref_img"), (8, 12, "video")],
    )
    indices, population = _modality_indices(layout, "visual_cond", 3, device=torch.device("cpu"))
    assert population == 4
    assert indices.numel() == 3
    assert all(int(value) in {2, 3, 6, 7} for value in indices)


def test_runtime_profiles_sampled_h3_attention_and_builds_digest():
    runtime = AttentionProfileRuntime(
        AttentionProfileConfig(layer_stride=1, query_samples=2, key_samples_per_modality=2),
        {"layers": 2, "hidden_size": 8, "patch_size": (1, 2, 2)},
    )
    runtime.begin_call(
        layout=_layout(),
        sigma=0.5,
        branch="dense",
        latent_t=2,
        latent_h=2,
        latent_w=4,
        patch_h=1,
        patch_w=1,
    )
    q = torch.randn(1, 2, 14, 4)
    k = torch.randn(1, 2, 14, 4)
    runtime.observe(q, k, 2)
    runtime.end_call()
    report = runtime.report()
    assert report["attention_calls_profiled"] == 1
    assert len(report["buckets"]) == 1
    assert report["buckets"][0]["layer"] == 0
    assert "target_video" in report["buckets"][0]["q_to_k_mass"]
    assert len(report["profile_digest"]) == 64


def test_profiler_override_delegates_without_modifying_output():
    runtime = AttentionProfileRuntime(AttentionProfileConfig(), {"layers": 1})
    override = AttentionProfilerOverride(runtime)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    def original(q, k, v, heads, **kwargs):
        return q + k + v + heads

    result = override(original, q, k, v, 2, skip_reshape=True, transformer_options={})
    assert torch.equal(result, q + k + v + 2)


def test_policy_builder_is_proposal_only_and_keeps_dense_fallback():
    report = {
        "profile_digest": "abc",
        "architecture_digest": "def",
        "buckets": [
            {
                "layer": 0,
                "heads": 2,
                "video_structure": {
                    "local_3d_r1_mass": [0.8, 0.1],
                    "spatial_r2_same_frame_mass": [0.9, 0.2],
                    "temporal_r1_same_spatial_mass": [0.4, 0.1],
                },
                "q_to_k_mass": {"target_video": {"target_video": [0.9, 0.4]}},
            }
        ],
    }
    proposal = propose_attention_policy(report)
    assert proposal["status"] == "proposal_only_no_sparse_kernel_enabled"
    assert proposal["dense_fallback_required"] is True
    assert proposal["layers"]["0"]["heads"][0]["classification"] == "local_3d_candidate"
    assert proposal["layers"]["0"]["heads"][1]["classification"] == "global_or_cross_modal"
