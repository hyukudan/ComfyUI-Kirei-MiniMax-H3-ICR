# ComfyUI Kirei MiniMax H3 ICR

**Kirei H3 ICR** is an experimental MiniMax H3 **In-Context Regeneration** stack for ComfyUI. It performs a second H3 pass over an H3 Base result, reusing the draft and its original multimodal context while starting from a clean high-resolution latent initialization.

The project optimizes for **correct additional detail**. A result that is sharper but changes identity, objects, text, pose, action, camera, timing or scene state is considered worse.

> This is an independent research implementation. It is not MiniMax H3-Regenerate-2K and does not claim to reproduce MiniMax's private regeneration or sparse-attention internals.

## Status

### Base path — M0 to M3

Implemented on `main`:

- backend-agnostic second pass: FL2VA, Hybrid, Ref2VA or another compatible H3 `MODEL`;
- backend provenance for controlled A/B testing;
- strict H3 joint audio/video latent contracts;
- exact H3 Base latent injection into native `minimax_refs`;
- learned `H3_LATENT_UPSCALER` API-v1 integration plus bicubic control;
- Fourier low-frequency clean-latent alignment with RMS safety limits;
- native partial-noise H3 regeneration;
- per-step predicted-clean draft-fidelity projection;
- exact pass-1 audio lock;
- structured runtime reports and CI-backed tests.

### M4 — experimental 2K renderer

Implemented on `feature/tiled-2k-fusion` / PR #1:

- one global low-resolution H3 model-output prior per evaluation;
- overlapping HR target-video tiles;
- per-evaluation model-output fusion rather than final RGB stitching;
- weighted least-squares fusion against the global prior;
- sigma-aware prior decay to release late high-frequency freedom;
- exact full-canvas MM-RoPE coordinates inside every HR tile;
- global text, references and audio on every branch;
- HR `minimax_keyframes` cropped per tile and downscaled for the global branch;
- keyframe condition rows remapped to full-canvas MM-RoPE positions;
- Spectrum retained only on the stable global-prior topology while tile calls are forced actual;
- fail-closed geometry/topology rules and live telemetry.

### M5 — attention calibration and sparse-kernel research

Implemented on the experimental branch:

- non-destructive normalized Q/K profiler by H3 layer, head, sigma and branch;
- bounded sampling without constructing a full S×S analysis matrix;
- text / visual-condition / audio-condition / target-audio / target-video accounting;
- target-video spatial, temporal and 3D-local concentration metrics;
- architecture/profile SHA-256 fingerprints;
- proposal-only per-head classification;
- **experimental real block-sparse backend using PyTorch FlexAttention `BlockMask`**;
- policy/architecture/profile fingerprint validation;
- cross-modal context kept globally visible for sparse target-video heads;
- head-specific local-3D, spatial-window and temporal-stripe masks;
- mandatory dense sigma tail;
- dense fallback when a policy, topology, device or sparsity gate is not satisfied;
- BlockMask cache and actual block-sparsity telemetry.

The Flex backend is **not validated or enabled by default**. Its existence is not a speedup claim; CUDA wall time, VRAM and decoded-video parity must be measured first.

## Core architecture

```text
H3 Base
  |
  +-- clean AV latent -----------------------------+
  +-- decoded Base video -> Qwen reference         |
  +-- original Context-IR / refs / audio           |
                                                   v
                                        common H3 conditioning
                                                   |
                   +-------------------------------+------------------+
                   |                               |                  |
                 FL2VA                       Hybrid 45-49          Ref2VA
                   |                               |                  |
                   +-------------------------------+------------------+
                                                   |
                                      learned clean HR initializer
                                                   |
                                      low-frequency draft alignment
                                                   |
                                            partial H3 noise
                                                   |
                                                   v
                                           H3 second pass
                              +--------------------+--------------------+
                              |                                         |
                           dense                               M4 tiled 2K+
                              |                           global LR + HR tiles
                              |                           full-canvas MM-RoPE
                              |                           HR keyframe remapping
                              |                           per-step output fusion
                              +--------------------+--------------------+
                                                   |
                                                   v
                                             regenerated H3

M5 research overlay
  PackedLayout + sigma + normalized Q/K
                  |
             calibration profile
                  |
          fingerprint-bound policy
                  |
       FlexAttention BlockMask [experimental]
                  |
       dense tail / fallback when required
```

## Quality ranking

Validation is normative in this order:

1. draft geometry and motion fidelity;
2. identity and object correctness;
3. temporal consistency and disocclusion behavior;
4. faces, hands, small text and product detail;
5. perceptual detail / sharpness;
6. wall time and memory.

Do not select a backend, renderer or sparse policy from a single still frame or a sharpness score.

## Recommended base workflow

1. Generate H3 Base and keep its clean AV latent.
2. Build pass-2 conditioning with native **MiniMax H3 Reference to Video**:
   - decoded Base video as a reference so Qwen sees the draft;
   - original prompt / Context-IR;
   - original reference images, videos and audio;
   - final target geometry.
3. Optionally use **Kirei H3 ICR Append Base Latent Reference** for an exact DiT-side Base-video reference without a VAE round-trip.
4. Load one backend arm: FL2VA, Hybrid 45-49, Ref2VA, or all-AdaLN as a high-risk control.
5. Optionally attach **Kirei H3 ICR Backend Tag**.
6. Connect the companion learned 3D latent-upscaler provider.
7. Use a partial schedule: `0 <= sigmas[0] < 1`.
8. Run **Kirei H3 ICR Regenerate**.

### Controlled backend matrix

| Arm | MODEL backend | Purpose |
| --- | --- | --- |
| A | FL2VA | stability / texture baseline |
| B | Hybrid 45-49 | current laboratory detail candidate |
| C | Ref2VA | reference-behavior control |
| D | Hybrid all-AdaLN | higher-risk experimental control |

All other inputs must remain identical between arms.

## M4 2K workflow

M4 replaces each full HR H3 evaluation with one global branch and overlapping HR tile evaluations:

```text
current HR state
  +-- area downsample -> global LR H3 -> global prior --------+
  +-- HR tile 1 -> H3 ----------------------------------------+
  +-- HR tile 2 -> H3 ----------------------------------------+--> weighted model-output fusion
  +-- ... ----------------------------------------------------+
  +-- HR tile N -> H3 ----------------------------------------+
```

Fusion is performed at the same diffusion coordinate:

```text
y = (sum_i w_i * tile_i + lambda(sigma) * prior_hr)
    / (sum_i w_i + lambda(sigma))
```

Recommended initial M4 chain:

```text
MODEL
  -> Kirei H3 ICR Tiled 2K Patch
  -> Kirei H3 ICR Tiled Prior Schedule
  -> Kirei H3 ICR Regenerate
```

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

For a typical 124-frame clip, the 2048×1152 target contains about **85,248 target-video tokens** before text, references and audio; a 1024×768 tile contains about **28,416**.

See [`docs/M4_TILED_2K_RENDERER.md`](docs/M4_TILED_2K_RENDERER.md).

## M5 attention workflow

### 1. Profile

Patch a controlled run with:

- **Kirei H3 ICR Attention Profiler [M5 Research]**
- **Kirei H3 ICR Attention Profile Report**

The report produces:

- complete calibration JSON;
- proposal-only policy JSON.

Default light-profile settings:

```text
layer stride:              5
query samples/modality:   24
key samples/modality:     48
sigma decimals:            3
max buckets:            2048
```

A full calibration can later use `layer_stride=1`.

### 2. Experimental Flex sparse run

The generated policy can be supplied to:

- **Kirei H3 ICR Flex Sparse Attention [M5 Experimental]**
- **Kirei H3 ICR Flex Sparse Report**

The backend validates the policy digest and current H3 architecture fingerprint. Supplying the original profile JSON adds source-profile verification.

Initial sparse settings:

```text
Flex block size:          128
dense tail sigma:        0.12
minimum block sparsity:  5%
local 3D radius T/Y/X:   1 / 2 / 2
temporal stripe radius:  2
```

Policy behavior:

| Head classification | Target-video K/V | Text / refs / audio |
| --- | --- | --- |
| global / mixed | dense | dense |
| local 3D | local T/Y/X | global |
| spatial window | same T + local Y/X | global |
| temporal stripe | local T at same Y/X | global |

Non-target-video queries stay dense. Late steps always return to the original ComfyUI attention path.

PyTorch `BlockMask.sparsity()` is recorded so a configuration can demonstrate that blocks were actually skipped. If sparsity is too low, Kirei falls back to dense attention rather than claiming a sparse speedup.

See [`docs/M5_ATTENTION_CALIBRATION.md`](docs/M5_ATTENTION_CALIBRATION.md) and [`docs/M5_FLEX_SPARSE_BACKEND.md`](docs/M5_FLEX_SPARSE_BACKEND.md).

## Compatibility

### Spectrum H3

For M4, the global LR call retains Spectrum runtime metadata while HR tile calls are forced actual. A forecasted call that does not execute H3 attention naturally contributes no Q/K sample to M5.

### Attention overrides

M5 profiler and Flex sparse execution use ComfyUI's function-style `optimized_attention_override` and are designed to chain. Existing container-style overrides are rejected until a dedicated adapter exists.

### EasyCache

M4 currently rejects EasyCache because sharing cache state across incompatible tile topologies is unsafe.

## Nodes

### Base
- **Kirei H3 ICR Backend Tag**
- **Kirei H3 ICR Append Base Latent Reference**
- **Kirei H3 ICR Prepare Clean HR**
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

## Validation plan

Before merging PR #1 we will validate Base H3-ICR, M4 and M5 together. At minimum:

- dense ~1 MP baseline;
- M4 2048×1152 constant and sigma-scheduled prior;
- FL2VA / Hybrid 45-49 / Ref2VA where runtime permits;
- M4 with and without verified HR keyframes;
- profiler no-op/output-neutral control;
- Flex sparse all-dense/fallback control;
- candidate sparse policies with measured BlockMask sparsity;
- peak VRAM and wall time, including first-use overhead separately;
- complete decoded-video inspection for identity, objects, hands, faces, text, motion, flicker and tile boundaries.

No sparse policy becomes a default merely because it runs faster.

## Tests

```bash
python -m pytest
ruff check h3_icr tests
```

GitHub Actions runs tests and Ruff on every push and pull request. Unit tests use CPU Torch; actual Flex sparse performance requires CUDA and the real H3 runtime.

## Documentation

- [`docs/SPEC_v1_2.md`](docs/SPEC_v1_2.md) — engineering specification.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current architecture and contracts.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) — controlled media-comparison protocol.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — research map.
- [`docs/RESEARCH_SURVEY_v1_2.md`](docs/RESEARCH_SURVEY_v1_2.md) — literature and public-implementation survey.
- [`docs/M4_TILED_2K_RENDERER.md`](docs/M4_TILED_2K_RENDERER.md) — M4 design and validation gate.
- [`docs/M5_ATTENTION_CALIBRATION.md`](docs/M5_ATTENTION_CALIBRATION.md) — M5 measurement design.
- [`docs/M5_FLEX_SPARSE_BACKEND.md`](docs/M5_FLEX_SPARSE_BACKEND.md) — experimental real block-sparse backend.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — milestone gates.

## Next milestones

1. combined decoded-media validation of Base H3-ICR + M4;
2. real attention calibration on the target H3 backends and geometries;
3. CUDA benchmark/parity gate for the Flex sparse backend;
4. posterior/measurement-consistency experiments;
5. refine sparse policy by sigma/topology from real measurements;
6. train the state-aware BaseVideo Adapter + detail LoRA only after the training-free teacher is characterized;
7. distill the validated teacher later.
