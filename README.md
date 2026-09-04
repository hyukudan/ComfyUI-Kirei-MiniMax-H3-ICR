# ComfyUI Kirei MiniMax H3 ICR

**Kirei H3 ICR** is an experimental MiniMax H3 **In-Context Regeneration** stack for ComfyUI. It performs a second H3 pass over an H3 Base result, reusing the draft, original multimodal context and reference material while starting from a clean high-resolution latent initialization.

The project is designed around one priority: **generate additional detail without changing what happened in the Base video**.

> This is an independent research implementation. It is not MiniMax H3-Regenerate-2K and does not claim to reproduce MiniMax's private regeneration or sparse-attention internals.

## Project status

### Base H3-ICR path

Implemented on `main`:

- backend-agnostic second pass: FL2VA, Hybrid, Ref2VA or another compatible H3 `MODEL`;
- explicit backend provenance for controlled A/B testing;
- strict H3 joint audio/video latent validation;
- direct H3 Base latent injection into native `minimax_refs`;
- `H3_LATENT_UPSCALER` API-v1 integration with the companion learned 3D latent upscaler;
- bicubic latent control arm;
- Fourier low-frequency initialization alignment with RMS safety limits;
- partial-noise native H3 regeneration;
- per-step predicted-clean fidelity projection with a structure-first decay schedule;
- exact pass-1 audio lock;
- structured runtime reports;
- CI-backed unit tests and Ruff checks.

### M4 experimental 2K path

Under active development on `feature/tiled-2k-fusion` / PR #1:

- one global low-resolution H3 prior call per model evaluation;
- overlapping high-resolution target-video tiles;
- model-output fusion **during every H3 evaluation**, not final RGB stitching;
- weighted least-squares fusion against the global LR prior;
- sigma-aware prior scheduling: strong structure lock near the start of pass 2, weaker prior near sigma 0 for late detail synthesis;
- exact full-canvas MM-RoPE target coordinates inside every HR tile;
- global text, references and audio on every branch;
- Spectrum runtime retained only on the stable global prior branch while HR tile calls are forced actual;
- native H3 HR keyframes cropped per tile and remapped to full-canvas MM-RoPE positions;
- keyframes downscaled consistently for the global LR branch;
- tile planner, overlap windows and renderer telemetry;
- explicit fail-closed behavior for unsupported topology or geometry.

M4 is **not production-validated yet**. It will remain experimental until decoded H3 media tests show that it preserves the Base video's identities, objects, motion, timing and scene state.

## Core architecture

```text
H3 Base generation
    |
    +-- clean AV latent ---------------------------+
    |                                              |
    +-- decoded Base video --> Qwen reference      |
    +-- original prompt / Context-IR               |
    +-- original images / videos / audio           |
                                                   v
                                       common H3 conditioning
                                                   |
                      +----------------------------+---------------------------+
                      |                            |                           |
                    FL2VA                    Hybrid 45-49                  Ref2VA
                      |                            |                           |
                      +----------------------------+---------------------------+
                                                   |
                                         learned clean HR init
                                                   |
                                        low-frequency alignment
                                                   |
                                           partial H3 noise
                                                   |
                                                   v
                                           H3 second pass
                                                   |
                              +--------------------+--------------------+
                              |                                         |
                       dense / ~1 MP                          M4 tiled / ~2K+
                              |                                         |
                    per-step fidelity                global LR prior + HR tiles
                              |                         + global MM-RoPE coords
                              |                         + HR keyframe remapping
                              |                         + sigma-aware prior decay
                              |                         + per-step output fusion
                              +--------------------+--------------------+
                                                   |
                                                   v
                                             regenerated H3
```

## Quality rule

A result that is sharper but changes the Base draft **loses**.

Validation priority is:

1. draft geometry and motion fidelity;
2. identity and object correctness;
3. temporal consistency and disocclusion behavior;
4. faces, hands, small text and product detail;
5. perceptual detail / sharpness;
6. wall time and memory.

Do not choose a backend or renderer from a single still image or a sharpness metric.

## Recommended base workflow

1. Generate H3 Base normally and keep the clean AV latent.
2. Build second-pass conditioning with ComfyUI's native **MiniMax H3 Reference to Video** node:
   - include the decoded H3 Base video as a reference video so Qwen sees the draft;
   - include the original reference images, videos and audio again;
   - reuse the original prompt / Context-IR content;
   - configure conditioning for the final target geometry.
3. Optionally append the exact clean Base latent with **Kirei H3 ICR Append Base Latent Reference**. This adds a DiT-side `minimax_refs` block without a VAE round-trip.
4. Load one controlled backend arm:
   - FL2VA pure;
   - Hybrid 45-49 — current laboratory detail candidate;
   - Ref2VA pure;
   - all-AdaLN only as a higher-risk experimental arm.
5. Optionally tag the backend with **Kirei H3 ICR Backend Tag**.
6. Connect the companion **MiniMax H3 Latent Upscaler Provider (3D)**.
7. Use a partial H3 sigma schedule: `0 <= sigmas[0] < 1`.
8. Run **Kirei H3 ICR Regenerate**.

## Backend comparison matrix

The same conditioning, Base latent, target noise, sigmas, sampler, references and audio must be reused across all arms.

| Arm | Model backend | Purpose |
| --- | --- | --- |
| A | FL2VA | stability / texture baseline |
| B | Hybrid 45-49 | current detail candidate |
| C | Ref2VA | reference-behavior control |
| D | Hybrid all-AdaLN | higher-risk experimental control |

The current expectation is **not** an accepted result. The winner must be selected from complete decoded videos.

## Experimental M4 2K renderer

M4 avoids treating every crop as an independent video.

For each H3 model evaluation:

```text
current HR noisy video
    |
    +-- area downsample --> global LR H3 --> global model-output prior
    |
    +-- HR tile 1 --> H3 --+
    +-- HR tile 2 --> H3 --+
    +-- ...                 +--> weighted model-output fusion --> sampler
    +-- HR tile N --> H3 --+
```

The fusion is performed at the same diffusion coordinate:

```text
y = (sum_i w_i * tile_i + lambda(sigma) * prior_hr)
    / (sum_i w_i + lambda(sigma))
```

This is deliberately **not** a final-frame seam blend.

### Prior schedule

A constant global-prior weight is useful for structure, but it can suppress the high-frequency freedom we want from late H3 steps. M4 therefore exposes an optional structure-first schedule:

```text
m(sigma) = floor + (1 - floor) * (sigma / sigma_start)^power
lambda(sigma) = prior_strength * m(sigma)
```

Recommended initial values:

```text
prior_strength:       0.30
prior_schedule_floor: 0.15
prior_schedule_power: 1.0
```

At the start of the second pass the renderer uses the full configured prior strength. Toward sigma 0, the regularizer approaches `prior_strength * floor`, leaving more freedom for H3 to synthesize microdetail while retaining a small global anchor.

The schedule is a separate patch on purpose, so validation can compare constant-prior and scheduled-prior runs without changing the underlying tiled renderer.

Recommended M4 node order:

```text
MODEL
  -> Kirei H3 ICR Tiled 2K Patch
  -> Kirei H3 ICR Tiled Prior Schedule
  -> Kirei H3 ICR Regenerate
```

### Global MM-RoPE coordinates

Native H3 creates target spatial MM-RoPE coordinates from the target shape. A naive tile call would therefore make each crop behave as if it occupied the complete frame.

The M4 renderer builds the full-canvas H3 layout first and then assigns each tile the exact target-video `position_ids` that belong to its region of the full canvas. The same policy is applied to spatial HR keyframe rows.

If exact global coordinate mapping cannot be established, M4 fails instead of silently falling back to local tile coordinates.

### HR keyframes

M4 supports native target-grid `minimax_keyframes` when their visual latent uses the full target geometry before tiling.

For each global LR call:

- visual keyframe latents are spatially reduced to the global prior geometry;
- keyframe audio remains global and unchanged;
- H3 condition-latent lists are rebuilt to match the transformed keyframes.

For each HR tile:

- visual keyframes are cropped to the same tile;
- keyframe audio remains global;
- keyframe condition rows use the corresponding full-canvas MM-RoPE coordinates.

This prepares the renderer for sparse HR anchor experiments inspired by video-SR work such as SparkVSR, while keeping the H3-native conditioning contract.

### Initial 2K laboratory preset

```text
target:               2048 x 1152
tile:                 1024 x 768
requested overlap:    256 x 256
global prior:          H3 Base latent geometry
prior strength:       0.30
prior schedule floor: 0.15
prior schedule power: 1.0
max tiles:             16
```

For a typical 124-frame H3 clip, the full 2048x1152 target is about **85,248 target video tokens** before text, references and audio. A 1024x768 tile is about **28,416 target video tokens**. The initial plan normally uses six HR tiles plus one global LR call per H3 model evaluation.

## Spectrum and cache compatibility

### Spectrum H3

Current M4 policy:

- the single global LR prior call retains Spectrum runtime metadata;
- HR tile child calls remove `spectrum_h3_*` runtime fields and execute as actual H3 calls;
- forecast history is never shared across different tile topologies.

This is an intentionally conservative first interoperability mode and still requires decoded-media validation.

### EasyCache

EasyCache is currently rejected by M4. Tile-local cache semantics have not been implemented, and sharing a cache across different tile topologies would be unsafe.

## Nodes

### Base H3-ICR

- **Kirei H3 ICR Backend Tag** — attaches explicit backend provenance to a cloned `MODEL`.
- **Kirei H3 ICR Append Base Latent Reference** — adds the exact clean Base latent to `minimax_refs` without a VAE round-trip.
- **Kirei H3 ICR Prepare Clean HR** — learned/bicubic clean upscale plus low-frequency draft alignment.
- **Kirei H3 ICR Regenerate** — integrated clean initialization and native second H3 sampling pass.
- **Kirei H3 ICR Report JSON** — renders the structured ICR report.

### M4 experimental

- **Kirei H3 ICR Tiled 2K Patch [Experimental]** — patches a native H3 `MODEL` with global-LR + tiled-HR model-output rendering.
- **Kirei H3 ICR Tiled Prior Schedule [Experimental]** — decays the global prior regularization from structure-first to detail-friendly as sigma decreases.
- **Kirei H3 ICR Tiled 2K Report** — exposes live tile/prior/token/keyframe/prior-schedule telemetry.

The tiled patch is applied first, the optional prior schedule second, and the resulting `MODEL` is then passed to **Kirei H3 ICR Regenerate**.

## Companion integration

Preferred learned initializer:

- `xmarre/Comfyui_Minimax_h3_latent_Upscaler`;
- use its `H3_LATENT_UPSCALER` API-v1 provider.

The Hybrid backend is intentionally external. This repository consumes the resulting ComfyUI `MODEL`; it does not vendor the Hybrid loader.

## Validation plan

The first combined validation should compare:

- dense H3-ICR around ~1 MP;
- M4 2048x1152 with constant prior;
- M4 2048x1152 with sigma-aware prior schedule;
- FL2VA, Hybrid 45-49 and Ref2VA arms where runtime cost permits;
- M4 with and without verified HR keyframes;
- multiple `prior_strength` values around `0.30`;
- schedule floors around `0.10-0.25` and powers around `0.5-2.0`.

Inspect the complete video for:

- tile seams;
- identity drift between tile regions;
- incorrect hands/faces/text crossing tile boundaries;
- local exposure or color discontinuity;
- motion discontinuity;
- hallucinated objects or changed actions;
- temporal flicker;
- actual detail gain;
- VRAM and wall time.

## Tests

```bash
python -m pytest
ruff check h3_icr tests
```

GitHub Actions runs the unit suite and Ruff on every push and pull request. The model-independent tests do not require a ComfyUI installation; actual H3 media validation requires a current ComfyUI MiniMax H3 runtime and the target model checkpoints.

## Documentation

- [`docs/SPEC_v1_2.md`](docs/SPEC_v1_2.md) — normative engineering specification.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current architecture and contracts.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) — controlled media-comparison protocol.
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — research map.
- [`docs/RESEARCH_SURVEY_v1_2.md`](docs/RESEARCH_SURVEY_v1_2.md) — literature and public-implementation survey.
- [`docs/M4_TILED_2K_RENDERER.md`](docs/M4_TILED_2K_RENDERER.md) — M4 tiled renderer design and validation gate.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — staged roadmap.

## Roadmap

Current order:

1. validate Base H3-ICR backend matrix on decoded media;
2. validate M4 global-LR + tiled-HR 2K rendering, sigma-aware prior scheduling and HR keyframes;
3. add measurement-consistency / posterior-consistency experiments;
4. profile dense H3 attention by layer, head, timestep and modality;
5. add calibrated real sparse kernels with explicit fallback;
6. train the state-aware BaseVideo Adapter + detail LoRA only after the training-free teacher is characterized;
7. distill the validated teacher to fewer steps later.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for milestone gates.
