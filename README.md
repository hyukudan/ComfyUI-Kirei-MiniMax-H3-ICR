# ComfyUI Kirei MiniMax H3 ICR

**Kirei H3 ICR** is an experimental MiniMax H3 **In-Context Regeneration** stack for ComfyUI. It performs a second H3 pass over an H3 Base result, reusing the draft and its original multimodal context while starting from a clean high-resolution latent initialization.

The project optimizes for **correct additional detail**. A result that is sharper but changes identity, objects, text, pose, action, camera, timing or scene state is considered worse.

> This is an independent research implementation. It is not MiniMax H3-Regenerate-2K and does not claim to reproduce MiniMax's private regeneration or sparse-attention internals.

## Status at a glance

| Stage | Purpose | Code status | Media validation |
| --- | --- | --- | --- |
| M0–M2 | H3-ICR contracts, initialization and second-pass harness | implemented on `main` | pending controlled matrix |
| M3a | low-frequency latent fidelity | implemented on `main` | pending |
| M3b | normalized latent measurement backprojection | experimental branch | pending |
| M3c | latent posterior-gradient control | experimental branch | pending |
| M3d | proxy-decoder pixel measurement | experimental branch | pending |
| M4 | global-LR + tiled-HR 2K renderer | experimental branch | pending |
| M5a | passive H3 attention calibration | experimental branch | pending CUDA profiling |
| M5b/M5c | real FlexAttention sparse execution / sigma-domain policy | experimental branch | pending CUDA parity + speed gate |
| M6 | state-aware BaseVideo Adapter runtime/checkpoint ABI | scaffold + loader implemented | **no trained checkpoint shipped** |
| M7 | distillation | planned | not started |

`main` intentionally remains the small, clean Base H3-ICR implementation. The larger research stack is accumulated in `feature/tiled-2k-fusion` / draft PR #1 until we validate the complete decoded-media pipeline.

## Quality rule

Validation priority is normative:

1. Base geometry and motion fidelity;
2. identity and object correctness;
3. temporal consistency and disocclusion behavior;
4. faces, hands, small text and product detail;
5. perceptual detail / sharpness;
6. wall time and memory.

A sharper result that changes the Base draft **loses**.

## Installation

Clone the repository into ComfyUI custom nodes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hyukudan/ComfyUI-Kirei-MiniMax-H3-ICR.git
cd ComfyUI-Kirei-MiniMax-H3-ICR
git checkout feature/tiled-2k-fusion
```

The repository includes the native Kirei latent-upscaler runtime and ComfyUI provider node. It does
not vendor the 691 MB checkpoint, MiniMax H3, Hybrid or Spectrum weights.

For development tests outside ComfyUI:

```bash
pip install -e . --no-deps
pytest
ruff check h3_icr tests
```

The CPU test suite covers model-independent contracts. Actual H3 generation, M3d decoder gradients, M4 performance, M5 FlexAttention and M6 trained adapters require a current ComfyUI MiniMax H3 runtime.

## Native learned clean initializer

The experimental branch includes its own strict `H3_LATENT_UPSCALER` API-v1 runtime and provider:

- use **Kirei H3 ICR Learned Latent Upscaler [Native]**;
- select `minimax_h3_latent_upscaler_3d_bf16.safetensors` and either place it in
  `ComfyUI/models/kirei_h3_upscalers/` or explicitly enable `download_if_missing`;
- downloads are written atomically and accepted only after the registered byte-size and SHA-256 match;
- the model is loaded lazily, spatial output is exact-target, temporal length is preserved and optional
  chunking/offload keep the H3 pass within the normal ComfyUI lifecycle.

No external upscaler node implementation is required by the Kirei graph.

The currently registered bootstrap checkpoint is the public
[LBH-123-AI MiniMax H3 Latent Upscaler BF16](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler),
listed by its model card as Apache-2.0. Its expected SHA-256 is
`4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`. The runtime/provider is
part of this repository; the checkpoint remains a separately attributed third-party model. See
[`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md).

The author model card reports training on roughly 80,000 paired samples, including video and 2K
image pairs, with explicit 1.5x examples and additional continuous scales between 1x and 4x. Kirei
has not independently audited that training corpus or provenance; this is recorded as an author
claim rather than a verified property of the weights.

Temporal controls:

- `temporal_chunk_size=0` runs the complete clip in one pass and is recommended when VRAM allows;
- `temporal_chunk_size=32` is the memory-oriented default for longer clips;
- `temporal_stability=0.30` is the experimental-branch provisional default. It smooths only the
  learned high-resolution residual, retaining the Base motion path instead of blurring the whole
  latent. Promotion outside this branch requires decoded-video human A/B plus an unstabilized arm;
- use `0.0` for exact checkpoint output or when measuring an untouched control arm.

### Experimental VAE round-trip prior

The branch also contains **Kirei H3 ICR VAE Upscale Prior [Experimental]**. It provides a premium,
slower initialization arm without depending on another custom-node repository:

```text
clean H3 Base AV latent
  -> decode video with the H3 VAE
  -> deterministic Lanczos resize in RGB
  -> encode at target resolution
  -> one-shot full-latent blend with the learned 3D initialization
  -> normal Kirei H3 ICR second pass
```

Connect its `vae_prior_latent` output to the optional input of **Kirei H3 ICR Regenerate** and start
with `vae_prior_strength=0.25`. Audio bypasses decode, resize, encode and fusion exactly. Leaving the
prior disconnected preserves the learned-only path. The blend is deliberately full-latent: Kirei
does not assume that the 24 H3 channels have separable spatial-frequency meanings.

This is an experimental quality mode rather than a replacement for the learned upscaler. It adds one
full H3 video decode and encode, can be expensive on long or high-resolution clips, and must be judged
with decoded-video temporal metrics as well as still-frame sharpness. The first real 1152x1280 gate
showed that the VAE/Lanczos round-trip retained more decoded sharpness and Base fidelity than the raw
learned initialization while remaining close to Base temporal motion.

On the first controlled 8-step portrait clip, with learned temporal stability fixed at `0.30`:

| VAE-prior strength | PSNR to resized Base vs learned-only | Laplacian sharpness vs learned-only | decoded warp error vs learned-only | interpretation |
|---:|---:|---:|---:|---|
| `0.25` | +1.24 dB | +1.96% | +2.08% | balanced research preset |
| `0.50` | +2.09 dB | +10.49% | +5.55% | high-detail arm; temporal cost is already borderline |

No obvious halo or duplicate-face artifact appeared in the sampled frames. Keep `0.25` as the suggested
starting point and `0.50` as a comparison arm, not a production default. These are single-clip results;
promotion still requires varied motion, camera movement, texture and scene-cut validation.

Optional reference-parity test (requires CUDA, the checkpoint and a local checkout of the public
reference runtime):

```bash
KIREI_H3_UPSCALER_CHECKPOINT=/path/to/minimax_h3_latent_upscaler_3d_bf16.safetensors \
KIREI_H3_UPSCALER_REFERENCE=/path/to/minimax_h3_latent_upscaler_3d.py \
pytest tests/test_latent_upscaler_reference_parity.py -q
```

The test covers full-context, chunked and the real anisotropic 54x48 to 80x72 geometry. It is
skipped in ordinary CPU CI because the third-party 691 MB checkpoint is deliberately not vendored.

Hybrid is intentionally external. Kirei H3-ICR consumes the resulting ComfyUI `MODEL`; it does not copy or vendor the Hybrid loader.

## Base H3-ICR — M0 to M3a

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

### Recommended Base workflow

1. Generate H3 Base and keep its clean AV latent.
2. Build pass-2 conditioning with native **MiniMax H3 Reference to Video**:
   - decoded Base video as a reference so Qwen sees the draft;
   - original prompt / Context-IR;
   - original reference images, videos and audio;
   - final target geometry.
3. Optionally use **Kirei H3 ICR Append Base Latent Reference** for exact DiT-side Base reference conditioning.
4. Load one controlled backend arm: FL2VA, Hybrid 45-49, Ref2VA, or all-AdaLN as a high-risk control.
5. Optionally attach **Kirei H3 ICR Backend Tag**.
6. Connect **Kirei H3 ICR Learned Latent Upscaler [Native]** from this repository.
7. Optionally connect **Kirei H3 ICR VAE Upscale Prior [Experimental]** for a controlled premium arm.
8. Use a partial schedule: `0 <= sigmas[0] < 1`.
9. Run **Kirei H3 ICR Regenerate**.

### Backend comparison matrix

The Base latent, conditioning, references, target noise, sigmas, sampler, fidelity parameters and audio must remain identical between arms.

| Arm | MODEL backend | Purpose |
| --- | --- | --- |
| A | FL2VA | stability / texture baseline |
| B | Hybrid 45-49 | current laboratory detail candidate |
| C | Ref2VA | reference-behavior control |
| D | Hybrid all-AdaLN | higher-risk experimental control |

The current preference for Hybrid 45-49 is a **hypothesis**, not an accepted result.

# M3 constraint family

M3 contains four deliberately separate constraints. They should be characterized independently before combinations are considered.

## M3a — low-frequency latent fidelity

Implemented in the Base path. It corrects only low spatial frequencies of predicted-clean `x0` toward the Base latent and relaxes through pass 2 so late H3 steps retain freedom for microdetail.

## M3b — normalized latent measurement backprojection

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

## M3c — latent posterior gradient

M3c is a separate MODEL patch. It differentiates only through the known latent measurement operator:

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

## M3d — proxy-decoder pixel measurement

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

## M3 ablation order

Prefer one experimental constraint at a time:

```text
control: M3a only
arm B:   M3a + M3b
arm C:   M3a + M3c
arm D:   M3a + M3d
```

Only test M3b+M3c+M3d combinations after their individual effects are understood.

# M4 — global-LR + tiled-HR 2K renderer

M4 avoids treating every spatial crop as an independent video.

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

This is deliberately **not** final RGB stitching.

Implemented experimentally:

- one global LR H3 model-output prior per evaluation;
- overlapping HR target-video tiles;
- weighted least-squares model-output fusion;
- sigma-aware prior decay to release late high-frequency freedom;
- exact full-canvas MM-RoPE target coordinates inside every HR tile;
- global text, references and audio on every branch;
- HR `minimax_keyframes` cropped per tile and downscaled for the global branch;
- keyframe condition rows remapped to full-canvas MM-RoPE positions;
- Spectrum retained only on the stable global-prior topology while tile calls are forced actual;
- fail-closed geometry/topology rules and live telemetry.

### Initial 2K laboratory preset

```text
target:               2048 x 1152
tile:                 1024 x 768
requested overlap:    256 x 256
global prior:          H3 Base latent geometry
prior strength:        0.30
prior schedule floor:  0.15
prior schedule power:  1.0
max tiles:             16
```

For a typical 124-frame H3 clip, the 2048×1152 target contains about **85,248 target-video tokens** before text, references and audio; a 1024×768 tile contains about **28,416**.

### M4 node chain

```text
MODEL
  -> Kirei H3 ICR Tiled 2K Patch
  -> Kirei H3 ICR Tiled Prior Schedule
  -> optional M5 profiler / Flex sparse patch
  -> optional M6 BaseVideo Adapter
  -> optional M3c or M3d MODEL patch
  -> Kirei H3 ICR Regenerate
```

See [`docs/M4_TILED_2K_RENDERER.md`](docs/M4_TILED_2K_RENDERER.md).

# M5 — attention calibration and real sparse execution

## M5a — passive attention calibration v2

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

## M5b — real FlexAttention sparse executor v2

Implemented but **not validated or enabled by default**:

- real PyTorch `FlexAttention` + `BlockMask` execution;
- architecture/profile/topology validation before sparse execution;
- target-video local-3D, spatial-window and temporal-stripe patterns per head;
- complete text/reference/keyframe/audio visibility;
- all non-target-video queries remain dense;
- mandatory dense sigma tail;
- dense fallback on topology mismatch, CPU, external masks, incomplete policy or insufficient measured block sparsity;
- topology/layer/policy/device BlockMask cache;
- modern Triton FlexAttention backend selection when supported;
- telemetry for sparse/fallback calls, mask builds/cache hits and block sparsity.

See [`docs/M5_FLEX_SPARSE_BACKEND.md`](docs/M5_FLEX_SPARSE_BACKEND.md).

## M5c — topology + sigma-domain policy v3

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

# M6 — state-aware BaseVideo Adapter

M6 is the first trained stage. The current repository implements its runtime ABI, exact zero-init scaffold, M4 alignment and trained-checkpoint loader, but **ships no trained adapter weights and makes no quality claim**.

## Adapter architecture

The selected H3 target-video hidden rows form the dynamic stream. The clean Base latent forms the static stream:

```text
dynamic H3 hidden -> norm/projection -------+
                                            |
aligned Base patches -> static projection --+--> sigma-conditioned fusion
                                            |
                                            v
                                   local 3D depthwise mixer
                                            |
                                   pointwise feature mixer
                                            |
                               ZERO-INIT residual projection
                                            |
                                            v
                                    selected H3 block output
```

The local mixer is linear in target-video token count; the scaffold does not introduce quadratic Base-to-target cross-attention.

Default plumbing-only injection blocks:

```text
12,24,36,45,48
```

These are not claimed to be optimal training layers.

## Zero-init parity

**Kirei H3 ICR BaseVideo Adapter Scaffold [M6]** creates a provider with:

```text
trained = false
out_proj = exactly zero
```

Untrained providers are runtime-bypassed. Tests additionally force the module path and require its zero projection to return an exact zero residual.

## M4 tile alignment

M6 does not resize the complete Base scene into each M4 tile. It reconstructs the full target latent geometry and tile rectangle directly from the tile's **global H3 MM-RoPE `position_ids`**, then:

```text
Base latent
  -> resize once to full target latent geometry
  -> crop exact global tile rectangle
  -> native 1x2x2 patch rows
  -> M6 static stream
```

Ambiguous/missing MM-RoPE geometry fails rather than silently using a wrong Base crop.

## Trained checkpoint loader

Place trained adapter checkpoints in:

```text
ComfyUI/models/kirei_h3_adapters/
```

Use **Kirei H3 ICR Load BaseVideo Adapter [M6]**.

Only safetensors checkpoints are accepted. The loader validates:

- API = 1;
- kind = `base_video_adapter`;
- native H3 architecture SHA-256;
- checkpoint `model_id`;
- exact adapter configuration;
- sorted/unique/in-range injection blocks;
- exact state-dict keys and tensor finiteness;
- complete checkpoint file SHA-256.

Required safetensors metadata:

```text
kirei_h3_icr_api
kirei_h3_icr_kind
kirei_h3_icr_architecture_digest
kirei_h3_icr_model_id
kirei_h3_icr_config_json
```

Optional:

```text
kirei_h3_icr_training_json
kirei_h3_icr_note
```

A loaded adapter is wrapped as a ComfyUI `CoreModelPatcher` and registered on the patched H3 MODEL as an **additional model**, so sampler preparation includes it in normal ComfyUI load/offload decisions.

The module uses the active H3 dtype when available and normal `get_torch_device()` / `unet_offload_device()` placement.

See [`docs/M6_BASE_VIDEO_ADAPTER.md`](docs/M6_BASE_VIDEO_ADAPTER.md).

# Architecture

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
                                optional M6 static Base + dynamic H3 adapter
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

# Compatibility notes

## Spectrum H3

For M4, the global LR call retains Spectrum runtime metadata while HR tile calls are forced actual. A forecasted call that does not execute H3 attention contributes no Q/K sample to M5.

## EasyCache

M4 currently rejects EasyCache because sharing cache state across incompatible tile topologies is unsafe.

## Attention overrides

M5 uses ComfyUI's function-style `optimized_attention_override`. Existing container-style overrides are rejected until a dedicated compatibility adapter exists.

## H3 pixel decoder

M3d requires a 24-channel H3-compatible decoder. `taeh3` is the recommended first proxy. Full VisualVAE gradient is explicit opt-in.

## M6 model management

Zero-init scaffold providers own no additional checkpoint. Trained M6 providers are registered as ComfyUI additional models; they should not be manually pinned to GPU outside the model-management lifecycle.

# Nodes

## Base / M3

- **Kirei H3 ICR Backend Tag**
- **Kirei H3 ICR Append Base Latent Reference**
- **Kirei H3 ICR Prepare Clean HR**
- **Kirei H3 ICR Learned Latent Upscaler [Native]**
- **Kirei H3 ICR Measurement Consistency [Experimental]** — M3b
- **Kirei H3 ICR Posterior Consistency [Experimental]** — M3c
- **Kirei H3 ICR Posterior Consistency Report**
- **Kirei H3 ICR Pixel Measurement [M3d Experimental]** — M3d
- **Kirei H3 ICR Pixel Measurement Report**
- **Kirei H3 ICR Regenerate**
- **Kirei H3 ICR Report JSON**

## M4 experimental

- **Kirei H3 ICR Tiled 2K Patch [Experimental]**
- **Kirei H3 ICR Tiled Prior Schedule [Experimental]**
- **Kirei H3 ICR Tiled 2K Report**

## M5 research / experimental

- **Kirei H3 ICR Attention Profiler [M5 Research]**
- **Kirei H3 ICR Attention Profile Report**
- **Kirei H3 ICR Flex Sparse Attention [M5 Experimental]**
- **Kirei H3 ICR Flex Sparse Report**

## M6 scaffold / trained ABI

- **Kirei H3 ICR BaseVideo Adapter Scaffold [M6]**
- **Kirei H3 ICR Load BaseVideo Adapter [M6]**
- **Kirei H3 ICR Apply BaseVideo Adapter [M6]**
- **Kirei H3 ICR BaseVideo Adapter Report**

# Combined validation plan

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
- M6 zero-init exact parity;
- M6 trained checkpoint lifecycle once a real checkpoint exists;
- M6 dense and M4 tile Base-alignment telemetry;
- complete decoded-video inspection for identity, objects, faces, hands, text, motion, flicker, audio and tile boundaries.

No optimization becomes default merely because it lowers one metric or runs faster.

# Tests

```bash
python -m pytest
ruff check h3_icr tests
```

GitHub Actions runs tests and Ruff on every push and pull request. CPU tests cover model-independent contracts; real M3d/M4/M5/M6 performance validation requires a current ComfyUI MiniMax H3 CUDA runtime.

# Documentation

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
- [`docs/M6_BASE_VIDEO_ADAPTER.md`](docs/M6_BASE_VIDEO_ADAPTER.md) — M6 runtime, checkpoint ABI and training direction.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — milestone gates.

# Next milestones

1. combined decoded-media validation of Base H3-ICR + M3 family + M4;
2. real attention calibration on the exact H3 backends/topologies used by that benchmark;
3. compare M5b static topology-bound and M5c sigma-domain sparse policies;
4. CUDA benchmark/parity gate for Flex sparse execution;
5. decide which M3 constraint, if any, actually improves the training-free teacher;
6. create/train the first M6 adapter checkpoint against the validated teacher and test its ComfyUI offload lifecycle;
7. add optional detail LoRA only if the adapter alone leaves a measurable high-frequency gap;
8. distill the validated teacher later.
