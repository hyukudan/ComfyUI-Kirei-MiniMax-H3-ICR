# H3-ICR architecture

## Base H3-ICR path

```text
H3 Base clean AV LATENT
        |
        +--> H3_LATENT_UPSCALER provider / bicubic control
        |        |
        |        +--> exact target clean video latent
        |
        +--> low-frequency initialization alignment against H3 Base
                 |
                 v
          clean target AV latent
                 |
Native MiniMaxH3ReferenceToVideo CONDITIONING
  - decoded Base video for Qwen
  - optional exact Base latent in minimax_refs
  - original pictures/videos/audio
  - original prompt / Context-IR text
                 |
MODEL backend (FL2VA / Hybrid / Ref2VA)
                 |
                 v
          partial-noise H3 sampler
                 |
      per-step low-frequency fidelity
                 |
                 v
             final LATENT
```

The repository never loads or merges Hybrid checkpoints itself. Hybrid is a workflow-level `MODEL` provider.

## M4 global-LR + tiled-HR path

M4 patches the H3 diffusion-model evaluation rather than the final decoded video:

```text
current target video state
        |
        +--> area downsample --> global LR H3 output --------+
        |                                                     |
        +--> HR tile H3 outputs with full-canvas MM-RoPE -----+--> weighted LS fusion
                                                              |
                                                              +--> sampler update
```

M4 invariants:

- only the target video stream is spatially tiled;
- text, references and audio remain globally visible;
- target-video tile `position_ids` are selected from the full native H3 `PackedLayout`;
- HR keyframe visual latents are cropped per tile and their condition rows receive matching full-canvas MM-RoPE positions;
- the global branch receives consistently downscaled HR keyframes;
- keyframe/reference audio remains global;
- returned audio is owned by the global LR branch;
- global-prior strength can decay with sigma to release late high-frequency freedom;
- Spectrum may remain on the stable global topology while HR tile branches are forced actual;
- EasyCache currently fails closed because tile-local cache semantics have not been defined.

## M5 calibration and sparse execution path

M5 first measures native H3 attention without changing its output:

```text
native H3 diffusion call
        |
        +--> profile wrapper: active PackedLayout / sigma / branch
        |
        +--> native optimized attention
                 |
                 +--> bounded normalized Q/K sample
                 +--> per-head / per-modality statistics
                 +--> delegate to original attention backend unchanged
```

The profiler never constructs the full SxS analysis matrix. It records sampled cross-modal mass and target-video locality by layer/head/sigma, plus architecture/profile fingerprints. A proposal-only classifier identifies candidate local-3D, spatial-window, temporal-stripe and dense/global heads.

The experimental execution path consumes that fingerprinted proposal:

```text
profile + proposal
      |
      +--> proposal/profile/architecture validation
      |
      v
active PackedLayout + sigma + layer
      |
      +--> dense tail / fallback? ---- yes --> original ComfyUI attention
      |
      no
      v
head-specific target-video policy
      |
      +--> all text/ref/audio K/V remain global
      +--> dense/global heads remain fully dense
      +--> selected target-video heads use local 3D / spatial / temporal patterns
      |
      v
PyTorch FlexAttention BlockMask
      |
      +--> reject if measured block sparsity is below threshold
      |
      v
real block-sparse attention kernel [experimental]
```

M5 sparse invariants:

- a dense QxK mask does not count as sparse execution;
- policy and optional source-profile digests are verified before patching the MODEL;
- the current native H3 architecture plus `model_id` must match the calibration fingerprint;
- non-target-video queries remain dense;
- sparse target-video queries retain complete access to text, visual conditions/references and audio context;
- masks are topology-specific and cached separately for dense/M4-global/M4-tile sequence layouts;
- late sigma steps always use the original dense attention path;
- CPU, pre-existing attention masks, incomplete policies and low block sparsity fall back to dense attention;
- `BlockMask.sparsity()` and fallback rates are telemetered.

## Why partial noise

The learned-upscaled clean latent is valuable only when the second pass starts below full noise. ComfyUI's native sampler combines the clean target latent with fresh target-grid noise according to the supplied H3 model-sampling law. `sigmas[0] == 1` is therefore rejected.

## Non-claims

- Kirei H3-ICR is not MiniMax H3-Regenerate-2K.
- It does not reproduce MiniMax's private sparse-attention topology.
- M4 has contract/unit coverage but still requires controlled decoded-media validation.
- The M5 Flex path is a real block-sparse kernel integration, but it is experimental and has no accepted speed/quality claim until CUDA H3 benchmarking and decoded-media parity are complete.
- Posterior-consistency gradients, BaseVideo Adapter, detail LoRA and distillation remain later milestones.
