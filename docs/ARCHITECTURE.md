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
      optional M3b measurement consistency
                 |
                 v
             final LATENT
```

The repository never loads or merges Hybrid checkpoints itself. Hybrid is a workflow-level `MODEL` provider.

## M3b latent measurement consistency

The low-frequency fidelity projector and M3b solve related but different problems.

The existing projector constrains selected low spatial frequencies. M3b explicitly checks whether the current HR predicted-clean latent is still compatible with the Base observation:

```text
D(x0_HR) ~= z_Base
```

where `D` is spatial area downsampling to the Base latent grid.

M3b computes a robust Base-grid residual, mixes low and optional higher Base-grid bands, lifts that residual to HR, measures the actual `D(U(r))` response, normalizes the backprojection gain and applies an independent RMS-bounded correction. The operation can repeat 1–N times and uses a structure-first sigma schedule.

Invariants:

- measurement consistency operates on video only;
- audio is returned exactly unchanged by the post-CFG hook;
- B/C/T must match the clean Base latent;
- NaN/Inf inputs fail closed;
- the normalized gain and HR correction are independently bounded;
- when installed through `Regenerate`, M3b runs after the low-frequency projector so measurement compatibility is the last structural correction before the sampler step.

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
                 +--> modal mass + exact sampled QK-pair evidence
                 +--> per-head / per-sigma statistics
                 +--> delegate to original attention backend unchanged
```

The passive profiler never constructs the full S×S analysis matrix. Calibration is bound to the native packed topology for each branch.

The experimental execution path has two policy layers:

```text
profile
  |
  +--> v2 aggregate topology-bound policy
  |
  +--> v3 branch + topology + sigma + layer domains
                  |
                  v
        architecture/topology gate
                  |
          optional sigma-domain gate
                  |
     dense tail / unsupported? ---- yes --> original ComfyUI attention
                  |
                  no
                  v
        head-specific target-video policy
                  |
       +----------+-----------+
       |          |           |
   local 3D    spatial     temporal
       |          |           |
       +----------+-----------+
                  |
       all text/ref/keyframe/audio context remains global
                  |
                  v
       PyTorch FlexAttention BlockMask
                  |
       block-sparsity threshold gate
                  |
                  v
       real block-sparse attention [experimental]
```

M5 invariants:

- a dense Q×K mask does not count as sparse execution;
- proposal and optional source-profile digests are verified before patching the MODEL;
- the current native H3 architecture plus `model_id` must match calibration;
- v2 refuses sparse execution outside the calibrated packed topology;
- v3 additionally refuses a per-layer policy outside `max_policy_sigma_distance` from a calibrated sigma domain;
- no categorical sparse-pattern interpolation is performed between unobserved sigmas;
- non-target-video queries remain dense;
- sparse target-video queries retain complete access to text, visual conditions/references/keyframes and audio context;
- BlockMasks are cached by effective topology/layer/policy/device, so equal policies may reuse a mask across sigmas but different head codes cannot;
- late sigma steps always use the original dense attention path;
- CPU, pre-existing attention masks, incomplete policies and low measured block sparsity fall back to dense attention;
- `BlockMask.sparsity()`, topology fallbacks and sigma-domain fallbacks are telemetered.

## Why partial noise

The learned-upscaled clean latent is valuable only when the second pass starts below full noise. ComfyUI's native sampler combines the clean target latent with fresh target-grid noise according to the supplied H3 model-sampling law. `sigmas[0] == 1` is therefore rejected.

## Non-claims

- Kirei H3-ICR is not MiniMax H3-Regenerate-2K.
- It does not reproduce MiniMax's private sparse-attention topology.
- M3b is a latent normalized-backprojection constraint, not a claim to reproduce diffusion posterior sampling exactly.
- M4 has contract/unit coverage but still requires controlled decoded-media validation.
- The M5 Flex path is a real block-sparse kernel integration, but it is experimental and has no accepted speed/quality claim until CUDA H3 benchmarking and decoded-media parity are complete.
- Pixel/VAE-space posterior gradients, BaseVideo Adapter, detail LoRA and distillation remain later milestones.
