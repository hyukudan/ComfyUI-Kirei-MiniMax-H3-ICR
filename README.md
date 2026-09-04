# ComfyUI Kirei MiniMax H3 ICR

**Kirei H3 ICR** is an experimental MiniMax H3 **In-Context Regeneration** stack for ComfyUI. It performs a second H3 pass over an H3 Base result, reusing the draft and its original multimodal context while starting from a clean high-resolution latent initialization.

The project optimizes for **correct additional detail**. A result that is sharper but changes identity, objects, text, pose, action, camera, timing or scene state is considered worse.

> This is an independent research implementation. It is not MiniMax H3-Regenerate-2K and does not claim to reproduce MiniMax's private regeneration or sparse-attention internals.

## Status

### Base path — M0 to M3a

Implemented on `main`:

- backend-agnostic second pass: FL2VA, Hybrid, Ref2VA or another compatible H3 `MODEL`;
- explicit backend provenance for controlled A/B testing;
- strict H3 joint audio/video latent contracts;
- exact H3 Base latent injection into native `minimax_refs`;
- learned `H3_LATENT_UPSCALER` API-v1 integration plus bicubic control;
- Fourier low-frequency clean-latent alignment with RMS safety limits;
- native partial-noise H3 regeneration;
- per-step predicted-clean low-frequency draft-fidelity projection;
- exact pass-1 audio lock;
- structured runtime reports and CI-backed tests.

### M3 constraint family — experimental

M3 contains four deliberately separate constraints. They should be characterized independently before being combined.

#### M3a — low-frequency latent fidelity

Implemented in the Base path. It corrects only low spatial frequencies of predicted-clean `x0` toward the Base latent and relaxes through the second-pass trajectory so late H3 steps retain freedom for microdetail.

#### M3b — normalized latent measurement backprojection

Implemented on `feature/tiled-2k-fusion` / PR #1.

M3b explicitly measures:

```text
D_latent(x0_HR) -> z_Base
```

and backprojects the Base-grid residual into the HR predicted-clean latent.

Implemented:

- area-downsample measurement operator;
- robust residual weighting;
- low/full-band residual mixing;
- normalized bicubic backprojection using measured `D(U(r))` response;
- backprojection-gain and RMS correction guards;
- optional internal re-measurement iterations;
- structure-first schedule;
- NestedTensor/packed AV support with audio untouched;
- before/after measurement telemetry.

Initial laboratory values:

```text
strength:                  0.15
cutoff:                    0.35
high_band_mix:             0.25
max_correction_rms_ratio: 0.15
robust_delta:              3.0
max_backprojection_gain:   2.0
iterations:                1
schedule_power:            1.0
schedule_floor:            0.0
```

Use **Kirei H3 ICR Measurement Consistency [Experimental]** and connect its handle to **Kirei H3 ICR Regenerate**.

See [`docs/M3_MEASUREMENT_CONSISTENCY.md`](docs/M3_MEASUREMENT_CONSISTENCY.md).

#### M3c — latent posterior gradient

M3c is a separate model patch. It differentiates only through the known latent measurement operator:

```text
loss = 0.5 * ||D_latent(x0_HR) - z_Base||^2
```

No H3 denoiser or VAE gradient is used. The gradient is RMS-normalized, corrections are capped, and audio is copied through exactly.

Initial values:

```text
strength:                 0.10
apply_every:              2
max_correction_rms_ratio: 0.05
```

Use **Kirei H3 ICR Posterior Consistency [Experimental]**.

See [`docs/M3_POSTERIOR_CONSISTENCY.md`](docs/M3_POSTERIOR_CONSISTENCY.md).

#### M3d — proxy-decoder pixel measurement

M3d adds decoder semantics without differentiably decoding a 2K latent. The HR clean estimate is first reduced to Base latent geometry, then decoded through an H3-compatible 24-channel decoder:

```text
x0_HR
  -> D_latent
  -> H3 decoder at Base geometry
  -> reduced RGB / edge / temporal measurement
  -> pixel-space loss
  -> gradient back to x0_HR
```

The Base reference is decoded through the **same** decoder and cached in reduced form.

Recommended first decoder: ComfyUI's lightweight H3 TAE/`taeh3` proxy. Full `MiniMaxH3VideoVAE` gradient is explicit opt-in because it can be very expensive.

Initial values:

```text
strength:                  0.05
apply_every:               4
max_correction_rms_ratio:  0.02
measurement_max_side:     384
frame_stride:               2
edge_weight:               0.25
temporal_weight:           0.10
verify_after:              false
allow_full_vae:            false
```

Use **Kirei H3 ICR Pixel Measurement [M3d Experimental]**.

See [`docs/M3_PIXEL_MEASUREMENT.md`](docs/M3_PIXEL_MEASUREMENT.md).

### M4 — experimental 2K renderer

Implemented on the experimental branch:

- one global low-resolution H3 model-output prior per evaluation;
- overlapping HR target-video tiles;
- model-output fusion at every H3 evaluation rather than final RGB stitching;
- weighted least-squares fusion against the global LR prior;
- sigma-aware prior decay to release late high-frequency freedom;
- exact full-canvas MM-RoPE coordinates inside every HR tile;
- global text, references and audio on every branch;
- HR `minimax_keyframes` cropped per tile and downscaled for the global branch;
- keyframe condition rows remapped to full-canvas MM-RoPE positions;
- Spectrum retained only on the stable global-prior topology while tile calls are forced actual;
- fail-closed geometry/topology rules and live telemetry.

Initial laboratory preset:

```text
target:               2048 x 1152
tile:                 1024 x 768
overlap:               256 x 256 minimum
global prior:          H3 Base latent geometry
prior strength:        0.30
prior schedule floor:  0.15
prior schedule power:  1.0
max tiles:             16
```

For a typical 124-frame H3 clip, the 2048×1152 target contains about **85,248 target-video tokens** before text, references and audio; a 1024×768 tile contains about **28,416**.

See [`docs/M4_TILED_2K_RENDERER.md`](docs/M4_TILED_2K_RENDERER.md).

### M5a — passive attention calibration v2

Implemented experimentally:

- passive normalized Q/K profiler through ComfyUI's function-style `optimized_attention_override`;
- bounded sampling without a full S×S analysis matrix;
- importance-corrected Q→K mass across text, visual conditions, audio conditions, target audio and target video;
- exact target-video QK pairs for diagonal, spatial-neighbor, temporal-neighbor and far-video evidence;
- per-head `spatial_minus_far` and `temporal_minus_far` margins;
- branch labels for `dense`, `m4_global_prior` and `m4_hr_tile`;
- canonical packed topology per branch/calibration run;
- topology digest covering target signature plus ordered segment kinds/row counts;
- architecture/profile SHA-256 fingerprints;
- passive-equivalence tests.

A changed target geometry, text/audio length, reference layout or keyframe layout changes the topology digest and requires a new calibration run.

See [`docs/M5_ATTENTION_CALIBRATION.md`](docs/M5_ATTENTION_CALIBRATION.md).

### M5b — real FlexAttention sparse executor v2

Implemented but **not validated or enabled by default**:

- real PyTorch `FlexAttention` + `BlockMask` execution;
- architecture/profile/topology validation before sparse execution;
- target-video local-3D, spatial-window and temporal-stripe patterns per head;
- complete text/reference/keyframe/audio visibility;
- all non-target-video queries remain dense;
- mandatory dense sigma tail;
- dense fallback on topology mismatch, CPU, external masks, incomplete policy or insufficient measured block sparsity;
- topology/layer/policy/device BlockMask cache;
- modern `BACKEND="TRITON"` selection when supported;
- telemetry for sparse/fallback calls, mask builds/cache hits and block sparsity.

See [`docs/M5_FLEX_SPARSE_BACKEND.md`](docs/M5_FLEX_SPARSE_BACKEND.md).

### M5c — topology + sigma-domain policy v3

The optional v3 policy emits explicit:

```text
branch + topology digest + sigma + layer
```

For each H3 call, Flex passes the topology gate and selects the nearest calibrated sigma independently per layer. If the distance exceeds `max_policy_sigma_distance`, that layer falls back to native dense attention.

Initial default:

```text
max_policy_sigma_distance: 0.03
```

No categorical interpolation is performed between unobserved sigma coordinates.

See [`docs/M5_SIGMA_DOMAIN_POLICY.md`](docs/M5_SIGMA_DOMAIN_POLICY.md).

The existence of M5 sparse code is **not a speedup or quality claim**. Real H3 CUDA wall time, VRAM, BlockMask sparsity and decoded-video parity must be measured first.

## Architecture

```text
H3 Base
  |
  +-- clean AV latent -----------------------------+
  +-- decoded Base video -> Qwen reference         |
  +-- original Context-IR / refs / audio           |
                                                   v
                                        common H3 conditioning
                                                   |
                 FL2VA / Hybrid 45-49 / Ref2VA
                                                   |
                                      learned clean HR initializer
                                                   |
                                      low-frequency init alignment
                                                   |
                                            partial H3 noise
                                                   |
                                                   v
                                           H3 second pass
                                                   |
                                         M3a low-frequency x0
                                                   |
                              optional M3b normalized measurement
                                                   |
                         optional M3c latent posterior gradient
                                                   |
                       optional M3d proxy-decoder pixel gradient
                                                   |
                              +--------------------+--------------------+
                              |                                         |
                           dense                               M4 tiled 2K+
                              |                           global LR + HR tiles
                              |                           full-canvas MM-RoPE
                              |                           HR keyframe remapping
                              |                           sigma-aware LR prior
                              |                           per-step output fusion
                              +--------------------+--------------------+
                                                   |
                                                   v
                                             regenerated H3

M5 research overlay
  PackedLayout + sigma + normalized Q/K
                  |
      modal mass + exact QK-pair evidence
                  |
       topology-bound calibration profile
                  |
          v2 aggregate policy + v3 sigma domains
                  |
 topology / architecture / sigma-domain gate
                  |
      FlexAttention BlockMask [experimental]
                  |
          dense tail / dense fallback
```

## Quality rule

Validation priority is normative:

1. Base geometry and motion fidelity;
2. identity and object correctness;
3. temporal consistency and disocclusion behavior;
4. faces, hands, small text and product detail;
5. perceptual detail / sharpness;
6. wall time and memory.

A sharper result that changes the Base draft loses.

## Recommended base workflow

1. Generate H3 Base and keep its clean AV latent.
2. Build pass-2 conditioning with native **MiniMax H3 Reference to Video**:
   - decoded Base video as a reference so Qwen sees the draft;
   - original prompt / Context-IR;
   - original reference images, videos and audio;
   - final target geometry.
3. Optionally use **Kirei H3 ICR Append Base Latent Reference** for exact DiT-side Base reference conditioning.
4. Load one controlled backend arm: FL2VA, Hybrid 45-49, Ref2VA, or all-AdaLN as a high-risk control.
5. Optionally attach **Kirei H3 ICR Backend Tag**.
6. Connect the companion learned 3D latent-upscaler provider.
7. Select at most one new M3 experimental constraint for first-pass characterization.
8. Use a partial schedule: `0 <= sigmas[0] < 1`.
9. Run **Kirei H3 ICR Regenerate**.

### Backend matrix

| Arm | MODEL backend | Purpose |
| --- | --- | --- |
| A | FL2VA | stability / texture baseline |
| B | Hybrid 45-49 | current laboratory detail candidate |
| C | Ref2VA | reference-behavior control |
| D | Hybrid all-AdaLN | higher-risk experimental control |

Everything except the backend must remain identical between arms.

## M3 node composition

M3a and M3b are integrated into the Regenerate path. M3c and M3d are MODEL patches and therefore compose through ComfyUI's ordered post-CFG function list.

For controlled tests, prefer one experimental addition at a time:

```text
control:  M3a only
arm B:    M3a + M3b
arm C:    M3a + M3c
arm D:    M3a + M3d
```

Only test M3b+M3c+M3d combinations after the individual effects are understood.

## M4 node chain

```text
MODEL
  -> Kirei H3 ICR Tiled 2K Patch
  -> Kirei H3 ICR Tiled Prior Schedule
  -> optional M5 profiler / Flex sparse patch
  -> optional M3c or M3d MODEL patch
  -> Kirei H3 ICR Regenerate
```

M4 performs one LR global H3 evaluation plus overlapping HR tile evaluations at each model call, then fuses their model outputs at the same diffusion coordinate.

## M5 workflow

### Calibration

Use:

- **Kirei H3 ICR Attention Profiler [M5 Research]**
- **Kirei H3 ICR Attention Profile Report**

Default light profile:

```text
layer stride:              5
query samples/modality:   24
key samples/modality:     48
sigma decimals:            3
max buckets:            2048
```

Use `layer_stride=1` only after the light profile is stable.

### Experimental Flex run

Use:

- **Kirei H3 ICR Flex Sparse Attention [M5 Experimental]**
- **Kirei H3 ICR Flex Sparse Report**

Initial settings:

```text
Flex block size:            128
dense tail sigma:          0.12
max policy sigma distance: 0.03
minimum block sparsity:    5%
local 3D radius T/Y/X:     1 / 2 / 2
temporal stripe radius:    2
```

A topology mismatch, sigma-domain miss or unsupported runtime condition falls back to native dense attention.

## Compatibility

### Spectrum H3

For M4, the global LR call retains Spectrum runtime metadata while HR tile calls are forced actual. A forecasted call that does not execute H3 attention contributes no Q/K sample to M5.

### EasyCache

M4 currently rejects EasyCache because sharing cache state across incompatible tile topologies is unsafe.

### Attention overrides

M5 uses ComfyUI's function-style `optimized_attention_override`. Existing container-style overrides are rejected until a dedicated compatibility adapter exists.

### H3 pixel decoder

M3d requires a 24-channel H3-compatible decoder. `taeh3` is the recommended first proxy. Full VisualVAE gradient is explicit opt-in.

## Nodes

### Base / M3

- **Kirei H3 ICR Backend Tag**
- **Kirei H3 ICR Append Base Latent Reference**
- **Kirei H3 ICR Prepare Clean HR**
- **Kirei H3 ICR Measurement Consistency [Experimental]** — M3b
- **Kirei H3 ICR Posterior Consistency [Experimental]** — M3c
- **Kirei H3 ICR Posterior Consistency Report**
- **Kirei H3 ICR Pixel Measurement [M3d Experimental]** — M3d
- **Kirei H3 ICR Pixel Measurement Report**
- **Kirei H3 ICR Regenerate**
- **Kirei H3 ICR Report JSON**

### M4 experimental

- **Kirei H3 ICR Tiled 2K Patch [Experimental]**
- **Kirei H3 ICR Tiled Prior Schedule [Experimental]**
- **Kirei H3 ICR Tiled 2K Report**

### M5 research / experimental

- **Kirei H3 ICR Attention Profiler [M5 Research]**
- **Kirei H3 ICR Attention Profile Report**
- **Kirei H3 ICR Flex Sparse Attention [M5 Experimental]**
- **Kirei H3 ICR Flex Sparse Report**

## Combined validation plan

Before PR #1 leaves draft status, compare at minimum:

- dense ~1 MP Base H3-ICR;
- M3a only;
- M3a + M3b;
- M3a + M3c;
- M3a + M3d with `taeh3`;
- selected M3 combinations only after single-constraint characterization;
- M4 2048×1152 constant versus sigma-scheduled prior;
- M4 HR keyframes on/off;
- FL2VA / Hybrid 45-49 / Ref2VA where runtime permits;
- profiler passive/no-op control;
- M5b static topology-bound sparse control;
- M5c sigma-domain sparse policy;
- topology-mismatch and sigma-domain-miss dense fallbacks;
- actual BlockMask sparsity and sparse-call fraction;
- first-use versus steady-state CUDA timings;
- peak VRAM;
- complete decoded-video inspection for identity, objects, faces, hands, text, motion, flicker, audio and tile boundaries.

No optimization becomes default merely because it lowers one metric or runs faster.

## Tests

```bash
python -m pytest
ruff check h3_icr tests
```

GitHub Actions runs tests and Ruff on every push and pull request. CPU tests cover model-independent contracts; real M3d/M4/M5 performance validation requires a current ComfyUI MiniMax H3 CUDA runtime.

## Documentation

- [`docs/SPEC_v1_2.md`](docs/SPEC_v1_2.md) — engineering specification.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current architecture and contracts.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) — controlled media-comparison protocol.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — research map.
- [`docs/RESEARCH_SURVEY_v1_2.md`](docs/RESEARCH_SURVEY_v1_2.md) — literature and public-implementation survey.
- [`docs/M3_MEASUREMENT_CONSISTENCY.md`](docs/M3_MEASUREMENT_CONSISTENCY.md) — M3b normalized latent measurement backprojection.
- [`docs/M3_POSTERIOR_CONSISTENCY.md`](docs/M3_POSTERIOR_CONSISTENCY.md) — M3c latent posterior gradient.
- [`docs/M3_PIXEL_MEASUREMENT.md`](docs/M3_PIXEL_MEASUREMENT.md) — M3d proxy-decoder pixel measurement.
- [`docs/M4_TILED_2K_RENDERER.md`](docs/M4_TILED_2K_RENDERER.md) — M4 design and validation gate.
- [`docs/M5_ATTENTION_CALIBRATION.md`](docs/M5_ATTENTION_CALIBRATION.md) — passive measurement/topology-binding design.
- [`docs/M5_FLEX_SPARSE_BACKEND.md`](docs/M5_FLEX_SPARSE_BACKEND.md) — experimental real block-sparse backend.
- [`docs/M5_SIGMA_DOMAIN_POLICY.md`](docs/M5_SIGMA_DOMAIN_POLICY.md) — optional sigma-domain sparse policy.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — milestone gates.

## Next milestones

1. combined decoded-media validation of Base H3-ICR + M3 family + M4;
2. real attention calibration on the exact H3 backends/topologies used by that benchmark;
3. compare M5b static topology-bound and M5c sigma-domain sparse policies;
4. CUDA benchmark/parity gate for Flex sparse execution;
5. decide which M3 constraint, if any, actually improves the teacher;
6. train the state-aware BaseVideo Adapter + detail LoRA only after the training-free teacher is characterized;
7. distill the validated teacher later.
