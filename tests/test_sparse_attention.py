from types import SimpleNamespace

import torch

from h3_icr.sparse_attention import (
    FlexSparseConfig,
    _HEAD_DENSE,
    _HEAD_LOCAL_3D,
    _HEAD_SPATIAL,
    _HEAD_TEMPORAL,
    _mask_allowed_scalar,
    _policy_head_codes,
    _proposal_digest,
)


def test_proposal_digest_ignores_digest_field_only():
    policy = {"api": 1, "status": "proposal_only_no_sparse_kernel_enabled", "layers": {}}
    digest = _proposal_digest(policy)
    policy["proposal_digest"] = digest
    assert _proposal_digest(policy) == digest


def test_policy_head_codes_fail_closed_on_missing_heads():
    policy = {"layers": {"0": {"heads": [{"head": 0, "classification": "local_3d_candidate"}]}}}
    assert _policy_head_codes(policy, 0, 2, device=torch.device("cpu")) is None


def test_policy_head_codes_map_candidate_classes():
    policy = {
        "layers": {
            "3": {
                "heads": [
                    {"head": 0, "classification": "global_or_cross_modal"},
                    {"head": 1, "classification": "local_3d_candidate"},
                    {"head": 2, "classification": "spatial_window_candidate"},
                    {"head": 3, "classification": "temporal_stripe_candidate"},
                ]
            }
        }
    }
    codes = _policy_head_codes(policy, 3, 4, device=torch.device("cpu"))
    assert codes.tolist() == [_HEAD_DENSE, _HEAD_LOCAL_3D, _HEAD_SPATIAL, _HEAD_TEMPORAL]


def test_sparse_mask_keeps_all_cross_modal_context_global():
    cfg = FlexSparseConfig(local_t_radius=1, local_y_radius=1, local_x_radius=1)
    kwargs = dict(
        video_start=6,
        video_stop=22,
        latent_t=2,
        latent_h=2,
        latent_w=4,
        patch_h=1,
        patch_w=1,
        config=cfg,
    )
    assert _mask_allowed_scalar(head_code=_HEAD_LOCAL_3D, q_idx=10, kv_idx=1, **kwargs)
    assert _mask_allowed_scalar(head_code=_HEAD_LOCAL_3D, q_idx=1, kv_idx=20, **kwargs)


def test_local_3d_mask_rejects_far_target_video_keys():
    cfg = FlexSparseConfig(local_t_radius=0, local_y_radius=0, local_x_radius=0)
    kwargs = dict(
        video_start=6,
        video_stop=22,
        latent_t=2,
        latent_h=2,
        latent_w=4,
        patch_h=1,
        patch_w=1,
        config=cfg,
    )
    assert _mask_allowed_scalar(head_code=_HEAD_LOCAL_3D, q_idx=6, kv_idx=6, **kwargs)
    assert not _mask_allowed_scalar(head_code=_HEAD_LOCAL_3D, q_idx=6, kv_idx=7, **kwargs)
    assert not _mask_allowed_scalar(head_code=_HEAD_LOCAL_3D, q_idx=6, kv_idx=14, **kwargs)


def test_spatial_and_temporal_candidate_geometry():
    cfg = FlexSparseConfig(local_y_radius=1, local_x_radius=1, temporal_radius=1)
    kwargs = dict(
        video_start=6,
        video_stop=22,
        latent_t=2,
        latent_h=2,
        latent_w=4,
        patch_h=1,
        patch_w=1,
        config=cfg,
    )
    assert _mask_allowed_scalar(head_code=_HEAD_SPATIAL, q_idx=6, kv_idx=7, **kwargs)
    assert not _mask_allowed_scalar(head_code=_HEAD_SPATIAL, q_idx=6, kv_idx=14, **kwargs)
    assert _mask_allowed_scalar(head_code=_HEAD_TEMPORAL, q_idx=6, kv_idx=14, **kwargs)
    assert not _mask_allowed_scalar(head_code=_HEAD_TEMPORAL, q_idx=6, kv_idx=15, **kwargs)
