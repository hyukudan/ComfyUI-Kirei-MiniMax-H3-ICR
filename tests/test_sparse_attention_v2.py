from types import SimpleNamespace

import torch

from h3_icr.sparse_attention import (
    FlexSparseConfig,
    _HEAD_DENSE,
    _HEAD_LOCAL_3D,
    _HEAD_SPATIAL,
    _HEAD_TEMPORAL,
)
from h3_icr.sparse_attention_v2 import (
    FlexSparseRuntimeV2,
    _block_mask_v2,
    policy_head_codes_v2,
    topology_matches_policy,
)
from h3_icr.attention_profile_v2 import topology_digest


def _layout(signature=(2, 2, 4, 4, 1)):
    text_len, latent_t, latent_h, latent_w, audio_t = signature
    video_rows = latent_t * (latent_h // 2) * (latent_w // 2)
    audio_rows = audio_t * 2
    return SimpleNamespace(
        signature=signature,
        seq_len=text_len + audio_rows + video_rows,
        segments=[
            (0, text_len, "text"),
            (text_len, text_len + audio_rows, "audio"),
            (text_len + audio_rows, text_len + audio_rows + video_rows, "video"),
        ],
    )


def _policy(layout):
    return {
        "calibrated_topologies": {
            "dense": {
                "digest": topology_digest(layout),
                "descriptor": {},
            }
        },
        "layers": {
            "0": {
                "heads": [
                    {"head": 0, "classification": "global_or_cross_modal"},
                    {"head": 1, "classification": "local_3d_pair_candidate"},
                    {"head": 2, "classification": "spatial_pair_candidate"},
                    {"head": 3, "classification": "temporal_pair_candidate"},
                ]
            }
        },
    }


def test_v2_policy_maps_exact_pair_candidate_labels_to_sparse_codes():
    policy = _policy(_layout())
    codes = policy_head_codes_v2(policy, 0, 4, device=torch.device("cpu"))
    assert codes.tolist() == [_HEAD_DENSE, _HEAD_LOCAL_3D, _HEAD_SPATIAL, _HEAD_TEMPORAL]


def test_topology_binding_rejects_geometry_outside_calibration():
    calibrated = _layout((2, 2, 4, 4, 1))
    policy = _policy(calibrated)
    assert topology_matches_policy(policy, "dense", calibrated)
    assert not topology_matches_policy(policy, "dense", _layout((2, 2, 4, 6, 1)))
    assert not topology_matches_policy(policy, "m4_hr_tile", calibrated)


def test_blockmask_cache_is_reused_across_sigmas_for_same_topology(monkeypatch):
    import h3_icr.sparse_attention_v2 as sparse_v2

    layout = _layout()
    policy = _policy(layout)
    runtime = FlexSparseRuntimeV2(
        policy,
        FlexSparseConfig(block_size=16, min_block_sparsity=0.0),
        architecture_digest="synthetic",
    )
    calls = {"count": 0}

    class FakeMask:
        def sparsity(self):
            return 50.0

    def fake_create_block_mask(*args, **kwargs):
        calls["count"] += 1
        return FakeMask()

    monkeypatch.setattr(sparse_v2, "_get_flex_api", lambda: (fake_create_block_mask, object()))

    runtime.begin_call(
        layout=layout,
        sigma=0.60,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        patch_h=2,
        patch_w=2,
        branch="dense",
    )
    active1 = runtime._active
    mask1 = _block_mask_v2(runtime, active1, 0, 4, layout.seq_len, torch.device("cpu"))
    runtime.end_call()

    runtime.begin_call(
        layout=layout,
        sigma=0.35,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        patch_h=2,
        patch_w=2,
        branch="dense",
    )
    active2 = runtime._active
    mask2 = _block_mask_v2(runtime, active2, 0, 4, layout.seq_len, torch.device("cpu"))
    runtime.end_call()

    assert mask1 is mask2
    assert calls["count"] == 1
    assert runtime.stats.mask_builds == 1
    assert runtime.stats.mask_cache_hits == 1
