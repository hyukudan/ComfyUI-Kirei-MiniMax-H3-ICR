# H3-ICR architecture

## v0.1 implemented path

```text
H3 Base clean AV LATENT
        |
        +--> H3_LATENT_UPSCALER provider (preferred) / bicubic control
        |        |
        |        +--> exact target clean video latent
        |
        +--> low-frequency initialization alignment against H3 Base
                 |
                 v
          clean target AV latent
                 |
Native MiniMaxH3ReferenceToVideo CONDITIONING
  - base video as ref_video
  - original pictures/videos/audio
  - original prompt / Context-IR text
                 |
MODEL backend (FL2VA / Hybrid / Ref2VA) -- H3 ICR Backend Tag
                 |
                 v
          partial-noise H3 sampler
                 |
                 v
             final LATENT
```

The repository never loads or merges Hybrid checkpoints itself. Hybrid is a MODEL provider chosen by the workflow.
This keeps the project backend-agnostic and avoids copying GPL-licensed loader code.

## Why partial noise

The learned-upscaled clean latent is valuable only when the second pass starts below full noise. ComfyUI's
native sampler combines the clean target latent with fresh target-grid noise according to the supplied H3 model
sampling law. `sigmas[0] == 1` is therefore rejected.

## What v0.1 does not claim

- It is not MiniMax H3-Regenerate-2K.
- It does not implement MiniMax's private sparse attention.
- It applies low-frequency fidelity guidance after model evaluations through a post-CFG projector; live full-ComfyUI validation remains pending.
- It does not yet implement tiled per-step fusion, posterior-consistency gradients, BaseVideo Adapter, or LoRA training.

Those items are isolated milestones rather than hidden half-implementations.
