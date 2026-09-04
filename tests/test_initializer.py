import torch

from h3_icr.fidelity import FidelityConfig
from h3_icr.initializer import InitConfig, upscale_and_align_clean


class Provider:
    api_version = 1
    kind = "minimax_h3_learned_latent_upscaler"

    def __init__(self):
        self.called = None

    def upscale_clean_video(self, video, *, target_h, target_w):
        self.called = (target_h, target_w)
        import torch.nn.functional as F

        b, c, t, h, w = video.shape
        work = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        out = F.interpolate(work, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)


def test_learned_provider_exact_target_and_audio_preserved():
    torch.manual_seed(1)
    video = torch.randn(1, 24, 3, 24, 32)
    audio = torch.randn(1, 32, 2, 100)
    provider = Provider()
    result, report = upscale_and_align_clean(
        (video, audio),
        InitConfig(
            transfer="learned_3d",
            target_width=1024,
            target_height=768,
            fidelity=FidelityConfig(strength=0.0),
        ),
        provider,
    )
    out_video, out_audio = result
    assert provider.called == (48, 64)
    assert out_video.shape[-2:] == (48, 64)
    assert torch.equal(out_audio, audio)
    assert report["transfer"] == "learned_3d"
