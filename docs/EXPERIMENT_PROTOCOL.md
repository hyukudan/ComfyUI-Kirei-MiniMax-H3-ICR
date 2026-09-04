# Backend A/B/C/D experiment protocol

The first media gate must compare four arms without changing any variable except MODEL backend:

- A — FL2VA pure + native reference conditioning.
- B — Hybrid 45–49 + the exact same conditioning.
- C — Ref2VA pure + the exact same conditioning.
- D — Hybrid all-AdaLN, experimental/QC only.

Lock across arms:

1. H3 Base latent and decoded base-video reference;
2. prompt / Context-IR text;
3. original image/video/audio references and their order;
4. `minimax_refs` and keyframe positions;
5. target dimensions;
6. learned-upscaler checkpoint, precision and device;
7. exact generated target-noise tensor / seed;
8. sigma tensor and sampler;
9. fidelity configuration;
10. pass-1 audio latent.

Evaluate in this order:

1. draft geometry and motion fidelity;
2. identity and object correctness;
3. temporal stability / disocclusions;
4. faces, hands, small text and product details;
5. perceptual sharpness/detail;
6. wall time, VRAM and H3 NFE accounting.

A backend that is sharper but invents or changes objects loses to a less sharp backend that preserves the draft.
