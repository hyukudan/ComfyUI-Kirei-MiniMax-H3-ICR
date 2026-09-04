# H3-ICR Research Survey v1.2

**Date:** September 4, 2026  
**Scope:** research directions, public implementations and engineering ideas relevant to high-fidelity MiniMax H3 in-context regeneration.

This document is a literature and implementation survey for H3-ICR. It is **not** an audit of any third-party repository, and it does not claim that any reviewed project implements MiniMax's private H3-Regenerate-2K system.

## 1. Research question

The practical goal is to take a completed H3 Base generation and produce a higher-detail second rendering while preserving:

- shot structure and camera behavior;
- subject identity;
- object state and interactions;
- motion and timing;
- audio from pass 1;
- original prompt/reference intent.

The central distinction is between **upscaling** and **regeneration**. A conventional latent or pixel upscaler can only infer missing high-frequency information from the existing image/video. An H3 second pass can also use the original text, images, videos, audio and H3's own generative prior to synthesize missing detail.

## 2. Public implementations reviewed

Several public implementations were studied as engineering references. They are independent projects with their own licenses and design goals.

### MiniMax H3 and ComfyUI

Primary source for the public H3 contracts used by this project:

- `MiniMax-AI/MiniMax-H3` — public H3 model family and documented Base -> Regenerate-2K product path;
- `Comfy-Org/ComfyUI` — H3 AV packing, reference conditioning, keyframes, `minimax_refs`, MM-RoPE layout, flow shifts and native sampler behavior.

The public path shows that H3 can receive multimodal references and re-inject reference latents at every model evaluation. H3-ICR builds on those public contracts rather than assuming private Regenerate-2K internals.

### xmarre/MiniMax-H3-Flow-Aligned-Regenerate

Reviewed as one of several experimental H3-specific implementations. Useful ideas include:

- exact H3 trajectory capture and provenance;
- flow-coordinate matching between low- and high-resolution passes;
- explicit solver/Spectrum lifetime boundaries;
- progressive clean-state handoff between spatial grids;
- conservative low-frequency guidance;
- integration with the companion learned H3 latent upscaler.

H3-ICR does not treat this repository as authoritative or as an implementation of H3-Regenerate-2K. It is one research reference among the papers and repositories listed below.

### xmarre/Comfyui_Minimax_h3_latent_Upscaler

Reviewed for its learned 24-channel H3 latent upscaling path and its versioned `H3_LATENT_UPSCALER` provider API. H3-ICR consumes that provider as an external dependency boundary instead of copying its model implementation.

### ANe5s/ComfyUI-MiniMax-H3-Hybrid

Reviewed as evidence that H3 reference conditioning and MODEL checkpoint choice are separable. The project uses FL2VA as the base model and optionally overlays selected Ref2VA modulation tensors. Its experiments suggest late-block overlays can increase clarity while wider overlays can introduce structural artifacts.

H3-ICR therefore treats Hybrid as a **MODEL backend candidate**, not as a replacement for reference conditioning.

## 3. Papers already strongly aligned with the problem

### HiFlow — Training-free High-Resolution Image Generation with Flow-Aligned Guidance

Most useful concepts:

- align low- and high-resolution states at matching flow coordinates;
- use low-resolution structure as a prior instead of relying only on the final endpoint;
- separate direction guidance from acceleration-like trajectory information.

For H3-ICR, the strongest transferable idea is the **initialization alignment** concept: build a clean HR estimate that remains consistent with the LR draft, then add target-grid noise and regenerate. This is more central to the second-pass design than aggressively increasing acceleration guidance.

### FrescoDiffusion — 4K Image-to-Video with Prior-Regularized Tiled Diffusion

Most useful concepts:

- retain a global low-resolution spatiotemporal prior;
- process high-resolution regions/tiles locally;
- fuse local and global predictions at every diffusion step;
- avoid final-only tile stitching.

This is the main architectural reference for the planned H3-ICR 2K renderer.

### RALU — Training-free Mixed-Resolution Latent Upsampling

Key lesson: a noisy latent should not be arbitrarily resized mid-trajectory because noise statistics and resolution-dependent model behavior change with geometry.

H3-ICR follows the safer pattern:

1. obtain a clean estimate;
2. transform the clean estimate to target geometry;
3. create fresh target-grid Gaussian noise;
4. resume sampling under a well-defined H3 flow coordinate.

### Self-Cascade, CineScale and Just-in-Time

These works reinforce a common coarse-to-fine principle: early denoising primarily resolves global structure and does not necessarily need the full final-resolution compute budget. They motivate progressive or two-pass compute allocation, but their exact positional or noise formulas are not copied into H3.

## 4. Temporal consistency and correspondence research

### TokenFlow, FRESCO, MoVideo, Upscale-A-Video and LatentWarp

These works motivate:

- latent/feature correspondence across frames;
- motion-aware propagation;
- visibility and occlusion gating;
- avoiding blind detail transport through ambiguous regions.

The main lesson for H3-ICR is caution. A local latent matcher with high cosine similarity can still be ambiguous on repeated textures. Temporal transport should therefore be confidence-gated or learned; simply increasing temporal guidance strength is not a robust path to more detail.

## 5. Resolution and attention research

### simple diffusion and SD3 resolution-dependent flow

These works motivate evaluating whether SNR/flow schedules should change with observation count or target resolution. H3 already has its own shifted flow schedule, so any resolution adjustment must be treated as a relative experiment rather than replacing H3's native model sampling.

### FreeSwim, HRDiT and ResDiT

Common insight: high-resolution failure is not only a memory problem. It also involves receptive-field scale, positional behavior and the balance between local detail and global layout.

H3-ICR takes from this family:

- local HR windows should preserve native-scale spatial neighborhoods;
- some heads/layers should retain global access;
- positional treatment and attention locality should be measured rather than uniformly modified.

## 6. Additional papers with direct value for H3-ICR

### Video Super-Resolution: All You Need is a Video Diffusion Model

Useful transfer: formulate super-resolution fidelity as an inverse problem. If `D` is a known degradation operator, the HR estimate should satisfy:

```text
D(x0_hr) ~= x0_lr
```

For H3-ICR this supports two future consistency tiers:

1. latent downsample consistency;
2. periodic VAE-domain degradation consistency with bounded gradients back to the latent state.

### LiteVSR-style state-aware adapter

A particularly relevant design pattern is a frozen diffusion transformer plus an adapter that sees two sources:

- a **static** low-quality/base-video stream;
- a **dynamic** current denoising/clean-estimate stream.

A timestep-dependent gate can emphasize draft structure early and current HR detail later. This is the preferred conceptual blueprint for a future trained H3 BaseVideo Adapter.

### SparkVSR-style sparse HR keyframes

Sparse high-resolution keyframes can provide actual HR evidence at critical frames while the LR video remains the motion anchor. H3 already exposes `minimax_keyframes`, making this idea unusually compatible with the public H3 interface.

Potential use cases:

- first frame of a shot/chunk;
- face close-ups;
- hands and contact frames;
- product/logo shots;
- small text.

Every generated/detail-enhanced keyframe should pass a fidelity gate after being reduced to Base resolution. A sharper keyframe that changes identity or structure must be rejected.

### STCDiT and HiStream-style anchors/caches

Useful ideas:

- persistent first-frame or chunk anchors;
- recent-context caches;
- lower-resolution long-term context;
- updating the long-term cache from completed HR output reduced back to LR.

These are relevant for extending H3-ICR to long sequences without losing identity or shot continuity.

### UltraGen / Scale-DiT / AtlasVid-style global-local rendering

These works converge on a useful high-resolution design:

- LR global branch for semantics and motion;
- HR local branch for texture;
- asymmetric information transfer from global to local;
- local token organization that preserves physical neighborhood structure.

This supports the planned H3-ICR 2K architecture: global LR H3 prior plus HR windows/tiles fused during denoising.

### SEGA-style spectral positional adaptation

Potential experiment: adapt spatial positional scaling based on current latent frequency content instead of applying one uniform scaling factor. For H3 this would have to be restricted to spatial video/reference coordinates and tested carefully against native MM-RoPE. Text, audio and temporal coordinates should remain untouched unless measurements justify otherwise.

## 7. Sparse attention research

### Sparse VideoGen, Sparse-vDiT, CalibAtt and Re-ttention

Together these works suggest a measurement-first sparse strategy:

1. profile dense attention by layer, head, timestep and modality;
2. classify stable local/global/diagonal/reference patterns;
3. calibrate profiles over multiple prompts, aspect ratios, durations and reference loads;
4. bind profiles to the exact checkpoint/layout hash;
5. dispatch to actual sparse kernels;
6. keep a dense late tail or confidence-based fallback.

A dense attention kernel receiving a large QxK mask does **not** count as a successful sparse implementation.

For H3 specifically, text, audio and reference tokens should remain globally reachable until experiments prove a safer restriction.

## 8. High-frequency training and AIGC-specific degradation

### RealisVSR-style high-frequency losses

Wavelet/HOG-oriented losses are useful because the objective is not merely edge sharpening. A future H3 adapter should recover high-frequency bands and geometric structure while remaining constrained by LR measurement consistency and temporal losses.

### Ultra Flash-style T2V -> TV2V training

Useful principles:

- train on AIGC-like degradations and actual model rollouts, not only bicubic reductions of natural videos;
- preserve a generative TV2V formulation so the backbone can synthesize detail rather than becoming a narrow deterministic restorer;
- treat latent upsampling and high-resolution decoding capacity as distinct problems.

## 9. Trajectory-delta and editing research

### FlowEdit and path-aware velocity methods

Delta-velocity ideas may help preserve a source draft while allowing controlled detail changes. They are not part of the current MVP because they require additional model evaluations and the source/target conditioning assumptions differ from pure H3 regeneration. They remain a later research branch after the simpler second-pass teacher is validated.

## 10. Distillation research

### DUO-VSR and related one/few-step restoration distillation

The key engineering lesson is sequencing:

1. establish a high-quality multi-step teacher;
2. distill while preserving trajectory/fidelity behavior;
3. reduce to 2–4 steps;
4. add adversarial/perceptual objectives only after structural parity;
5. preference tuning last;
6. one-step inference only after the teacher and few-step student are stable.

H3-ICR should not optimize for one-step speed before proving the multi-step regeneration path.

## 11. Hybrid as a backend, not a conditioning method

The public H3 reference path in ComfyUI accepts references independently of whether the loaded checkpoint is FL2VA, Ref2VA or a compatible Hybrid composition. Therefore the correct conceptual split is:

```text
common H3 reference conditioning
          +
MODEL backend: FL2VA / Hybrid / Ref2VA
```

The first controlled media gate should compare:

| Arm | MODEL backend | Conditioning |
|---|---|---|
| A | FL2VA | identical common reference conditioning |
| B | Hybrid late AdaLN (45–49) | identical common reference conditioning |
| C | Ref2VA | identical common reference conditioning |
| D | Hybrid all-AdaLN | identical common reference conditioning, experimental only |

Hybrid 45–49 is the current **laboratory candidate**, not a universal default. The winner must be selected from full-video fidelity and correctness, not sharpness alone.

## 12. Resulting H3-ICR architecture

```text
H3 Base video + original multimodal context
        |
        +-- decoded Base video -> Qwen reference presentation
        +-- clean Base latent -> direct minimax_refs block
        +-- original image/video/audio references
        +-- optional verified sparse HR keyframes
        |
        v
Common H3 reference conditioning
        |
        +-- FL2VA MODEL
        +-- Hybrid MODEL
        +-- Ref2VA MODEL
        |
        v
Learned clean latent upscale
+ LR-consistent initialization alignment
+ fresh partial target-grid noise
        |
        v
H3 second pass
+ per-step low-frequency fidelity
+ exact audio lock
+ optional future posterior consistency
        |
        +-- dense ~1 MP baseline
        |
        +-- future 2K global-local renderer
             + per-step tiled prediction fusion
             + global LR prior
             + calibrated sparse attention
             + optional anchors/caches
```

## 13. Confidence levels

**High confidence**

- Hybrid is a MODEL backend rather than a replacement for H3 reference conditioning.
- A clean-state geometry transition plus fresh target noise is safer than arbitrary noisy-latent resizing.
- A controlled FL2VA/Hybrid/Ref2VA comparison must keep conditioning, noise and sampling identical.

**Medium confidence**

- Hybrid 45–49 is likely to be a strong detail candidate for pass 2 while retaining more FL2VA character.
- Global-LR + local-HR fusion is the right architectural family for 2K.
- Sparse HR keyframes can improve difficult local details if they pass a Base-fidelity gate.

**Unproven / experimental**

- Hybrid 45–49 is the best universal backend for H3-ICR.
- Any public implementation reproduces MiniMax's private H3-Regenerate-2K internals.
- Spectral MM-RoPE scaling or calibrated sparse attention can preserve H3 quality without model-specific tuning.
- One-step distillation can match a high-quality multi-step H3-ICR teacher.

## 14. Research policy

When a paper or repository materially influences H3-ICR:

- record it here or in `RESEARCH.md`;
- state the specific idea transferred;
- distinguish direct implementation from conceptual inspiration;
- preserve third-party licensing boundaries;
- validate any quality claim on decoded video, not only latent statistics or still-frame sharpness.
