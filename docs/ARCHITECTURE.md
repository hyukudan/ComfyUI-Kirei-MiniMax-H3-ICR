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
          M3a low-frequency fidelity
                 |
        optional M3b normalized measurement
                 |
        optional M3c latent posterior gradient
                 |
        optional M3d proxy-decoder pixel gradient
                 |
                 v
             final LATENT
```

The repository never loads or merges Hybrid checkpoints itself. Hybrid is a workflow-level `MODEL` provider.

## M3 constraint family

M3 mechanisms are independent research arms, not a stack that should automatically be enabled together.

### M3a — low-frequency latent fidelity

M3a filters the Base residual spatially and corrects only low-frequency structure. Its structure-first schedule intentionally releases late high-frequency freedom.

Invariants:

- video only;
- audio untouched;
- spatial correction bounded by RMS;
- unknown AV layouts fail closed.

### M3b — normalized latent measurement backprojection

M3b checks:

```text
D_latent(x0_HR) ~= z_Base
```

It computes a robust Base-grid residual, mixes low and optional higher Base-grid bands, lifts the residual to HR, measures the actual `D(U(r))` response, normalizes the backprojection gain and applies an independent RMS-bounded correction. The operation can repeat internally and has its own structure-first schedule.

Invariants:

- B/C/T must match the Base latent;
- measurement and correction gains are independently bounded;
- audio is returned exactly unchanged;
- NaN/Inf and unsupported layouts fail closed.

### M3c — latent posterior gradient

M3c is a separate post-CFG MODEL patch:

```text
probe = detach(x0_HR)
loss = 0.5 * ||D_latent(probe) - z_Base||^2
g = d(loss) / d(probe)
```

Autograd exists only through `D_latent`. No H3 or VAE gradient is involved. The update is normalized by residual/gradient RMS and capped relative to the latent scale.

M3c exists primarily as an ablation against M3b: if the explicit measurement gradient does not improve the quality/cost operating point, the cheaper M3b path should win.

### M3d — proxy-decoder pixel measurement

M3d adds decoder semantics without a differentiable 2K decode:

```text
x0_HR
  -> D_latent to Base geometry
  -> H3-compatible 24-channel decoder
  -> reduced pixel measurement
  -> RGB + edge + temporal-difference loss
  -> gradient through decoder and D_latent to x0_HR
```

The Base reference is decoded through the same decoder and cached as a reduced measurement.

Recommended initial decoder is ComfyUI's lightweight H3 TAE/`taeh3`. Full `MiniMaxH3VideoVAE` backward is recognized but explicit opt-in because of its cost.

M3d invariants:

- decoder must be MiniMax H3-compatible and 24-channel;
- the differentiable decoder branch runs at Base latent geometry;
- audio never enters the decoder/loss and is copied through exactly;
- correction is conservatively RMS-capped;
- optional post-correction verification is an extra decoder call and remains off by default;
- lower proxy loss is not an acceptance criterion by itself.

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
- HR keyframe visual latents are cropped per tile and receive matching full-canvas MM-RoPE positions;
- the global branch receives consistently downscaled HR keyframes;
- keyframe/reference audio remains global;
- returned audio is owned by the global LR branch;
- global-prior strength can decay with sigma;
- Spectrum may remain on the stable global topology while HR tiles are forced actual;
- EasyCache fails closed until tile-local cache semantics are defined.

## M5 calibration and sparse execution path

M5 first measures native H3 attention without changing its output:

```text
native H3 diffusion call
        |
        +--> profile wrapper: PackedLayout / sigma / branch
        |
        +--> native optimized attention
                 |
                 +--> bounded normalized Q/K sample
                 +--> modal mass + exact QK-pair evidence
                 +--> per-head / per-sigma statistics
                 +--> delegate to original backend unchanged
```

Calibration is architecture- and topology-bound. The experimental executor uses a real PyTorch FlexAttention `BlockMask`; a dense QxK mask is not considered sparse execution.

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

- proposal/profile/architecture fingerprints are verified;
- sparse execution is refused outside calibrated packed topology;
- v3 additionally refuses per-layer policy outside its sigma tolerance;
- no categorical pattern interpolation between unobserved sigmas;
- non-target-video queries remain dense;
- target-video sparse queries keep complete text/reference/keyframe/audio visibility;
- BlockMasks are cached by effective topology/layer/policy/device;
- late sigma always returns to dense attention;
- CPU, external masks, incomplete policy and insufficient sparsity fall back to dense;
- sparse/fallback/mask/sparsity telemetry is mandatory.

## M6 state-aware BaseVideo Adapter contract

M6 is the first trained stage and must remain optional until the training-free teacher is characterized.

Conceptual data flow:

```text
clean Base latent ---------> static Base stream ---------+
                                                        |
current x0 / H3 state -----> dynamic state stream -------+--> time-conditioned adapter
                                                        |
optional verified HR KF ---> sparse keyframe stream -----+
                                                        |
                                             zero-init residual injection
                                                        |
                                           selected native H3 blocks
```

Normative M6 requirements:

- H3 backbone frozen initially;
- adapter checkpoint/provider ABI separate from H3 checkpoints;
- static Base and dynamic current-state streams are distinct;
- sigma/time conditioning shifts emphasis from structure to late detail;
- optional HR-keyframe evidence is explicit rather than silently synthesized;
- residual output path is zero-initialized so an untrained adapter is exact backbone parity;
- adapter metadata binds architecture/checkpoint assumptions;
- missing/incompatible weights fail closed;
- selected injection blocks are explicit and reproducible;
- training data includes real H3 Base rollouts/AIGC degradations;
- training loss may combine teacher/flow, measurement, temporal, identity/reference and high-frequency objectives.

No M6 runtime should claim improvement before trained checkpoints pass the same decoded-media gate as the training-free path.

## Why partial noise

The learned-upscaled clean latent is valuable only when pass 2 starts below full noise. `sigmas[0] == 1` discards that initialization and is rejected.

## Non-claims

- Kirei H3-ICR is not MiniMax H3-Regenerate-2K.
- It does not reproduce MiniMax's private sparse-attention topology.
- M3b is a normalized latent backprojection constraint, not exact DPS.
- M3c is an explicit latent-measurement gradient, not a gradient through H3.
- M3d is a proxy-decoder pixel measurement experiment, not exact pixel-space DPS.
- M4 still requires controlled decoded-media validation.
- M5 uses a real block-sparse kernel, but has no accepted speed/quality claim until CUDA parity/benchmarking is complete.
- M6 adapter weights and quality gains are not yet established.
- Distillation remains a later milestone after a validated teacher exists.
