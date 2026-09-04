from types import SimpleNamespace

from h3_icr.attention_profile import AttentionProfileConfig
from h3_icr.attention_profile_v2 import AttentionProfileRuntimeV2, propose_attention_policy_v2


def _layout(signature, extra_ref_rows=0):
    text_len, latent_t, latent_h, latent_w, audio_t = signature
    cursor = text_len
    segments = [(0, text_len, "text")]
    if extra_ref_rows:
        segments.append((cursor, cursor + extra_ref_rows, "ref_img"))
        cursor += extra_ref_rows
    audio_rows = audio_t * 2
    segments.append((cursor, cursor + audio_rows, "audio"))
    cursor += audio_rows
    video_rows = latent_t * (latent_h // 2) * (latent_w // 2)
    segments.append((cursor, cursor + video_rows, "video"))
    return SimpleNamespace(signature=signature, seq_len=cursor + video_rows, segments=segments)


def _runtime():
    return AttentionProfileRuntimeV2(
        AttentionProfileConfig(model_id="synthetic"),
        {"model_id": "synthetic", "module": "m", "class": "c", "layers": 1},
    )


def _begin(runtime, layout, branch):
    runtime.begin_call(
        layout=layout,
        sigma=0.5,
        branch=branch,
        latent_t=layout.signature[1],
        latent_h=layout.signature[2],
        latent_w=layout.signature[3],
        patch_h=2,
        patch_w=2,
    )


def test_same_branch_cannot_mix_different_calibration_topologies():
    runtime = _runtime()
    first = _layout((2, 2, 4, 4, 1))
    second = _layout((2, 2, 4, 6, 1))
    _begin(runtime, first, "dense")
    runtime.end_call()

    try:
        _begin(runtime, second, "dense")
    except RuntimeError as exc:
        assert "topology changed" in str(exc)
    else:
        raise AssertionError("one calibration branch must not mix packed topologies")


def test_reference_row_change_changes_topology_even_when_target_geometry_matches():
    runtime = _runtime()
    first = _layout((2, 2, 4, 4, 1), extra_ref_rows=4)
    second = _layout((2, 2, 4, 4, 1), extra_ref_rows=8)
    _begin(runtime, first, "dense")
    runtime.end_call()

    try:
        _begin(runtime, second, "dense")
    except RuntimeError:
        pass
    else:
        raise AssertionError("reference-load topology changes must require a separate calibration")


def test_different_m4_branches_may_have_distinct_calibrated_topologies_and_propagate_to_policy():
    runtime = _runtime()
    global_layout = _layout((2, 2, 4, 4, 1))
    tile_layout = _layout((2, 2, 6, 8, 1))
    _begin(runtime, global_layout, "m4_global_prior")
    runtime.end_call()
    _begin(runtime, tile_layout, "m4_hr_tile")
    runtime.end_call()

    report = runtime.report()
    proposal = propose_attention_policy_v2(report)

    assert set(report["calibrated_topologies"]) == {"m4_global_prior", "m4_hr_tile"}
    assert proposal["calibrated_topologies"] == report["calibrated_topologies"]
    assert proposal["proposal_digest"]
