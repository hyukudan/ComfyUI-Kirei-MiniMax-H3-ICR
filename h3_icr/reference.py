from __future__ import annotations

from typing import Any

from .contracts import unwrap_av, validate_av

BASE_REF_MARKER = "h3_icr_base_video"


def _clone_conditioning(conditioning: list) -> list:
    out = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            raise TypeError("CONDITIONING entries must contain [embedding, metadata]")
        copied = list(entry)
        copied[1] = dict(entry[1])
        out.append(copied)
    return out


def append_base_latent_reference(
    conditioning: list,
    base_samples: Any,
    *,
    include_audio: bool = False,
    replace_existing: bool = True,
) -> list:
    """Append the clean H3 Base latent as a native `minimax_refs` block.

    This is a DiT-side exact latent reference. It does *not* modify Qwen tokens; for the
    full multimodal path also provide the decoded base video to ComfyUI's native
    MiniMax H3 Reference to Video node.
    """
    validate_av(base_samples)
    video, audio = unwrap_av(base_samples)
    ref = {
        "kind": "video_audio" if include_audio else "video",
        "latent_t": int(video.shape[2]),
        "latent_h": int(video.shape[-2]),
        "latent_w": int(video.shape[-1]),
        "latent": video,
        BASE_REF_MARKER: True,
    }
    if include_audio:
        ref["ref_audio_t"] = int(audio.shape[-1])
        ref["audio_latent"] = audio
    else:
        ref["ref_audio_t"] = 0

    out = _clone_conditioning(conditioning)
    for entry in out:
        metadata = entry[1]
        refs = list(metadata.get("minimax_refs", []))
        if replace_existing:
            refs = [r for r in refs if not (isinstance(r, dict) and r.get(BASE_REF_MARKER) is True)]
        refs.append(dict(ref))
        metadata["minimax_refs"] = refs
        metadata["h3_icr_base_reference"] = {
            "api": 1,
            "mode": "dit_latent",
            "include_audio": bool(include_audio),
        }
    return out
