import torch

from h3_icr.reference import BASE_REF_MARKER, append_base_latent_reference


def test_append_base_latent_reference_preserves_existing_refs():
    positive = [[torch.zeros(1), {"minimax_refs": [{"kind": "image", "x": 1}]}]]
    video = torch.randn(1, 24, 3, 8, 10)
    audio = torch.randn(1, 32, 2, 100)
    out = append_base_latent_reference(positive, (video, audio))
    refs = out[0][1]["minimax_refs"]
    assert len(refs) == 2
    assert refs[-1][BASE_REF_MARKER] is True
    assert refs[-1]["latent"] is video
    assert out is not positive
    assert positive[0][1]["minimax_refs"] == [{"kind": "image", "x": 1}]


def test_replacement_avoids_duplicate_base_ref():
    positive = [[torch.zeros(1), {}]]
    video = torch.randn(1, 24, 3, 8, 10)
    audio = torch.randn(1, 32, 2, 100)
    one = append_base_latent_reference(positive, (video, audio))
    two = append_base_latent_reference(one, (video, audio), replace_existing=True)
    refs = two[0][1]["minimax_refs"]
    assert sum(bool(r.get(BASE_REF_MARKER)) for r in refs) == 1
