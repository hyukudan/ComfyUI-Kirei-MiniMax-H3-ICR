# Roadmap

## M0 — contracts and reproducible backend matrix — implemented in v0.1

- backend-agnostic MODEL tag;
- strict H3 AV contracts;
- exact target geometry;
- learned provider API v1 / bicubic control;
- partial-sigma contract;
- exact audio lock on sampler 2;
- JSON metrics/report.

## M1 — initialization fidelity — implemented in v0.1

- learned clean-video upscale;
- low-frequency Fourier initialization alignment;
- RMS correction guard;
- downsample-error telemetry.

## M2 — in-context regenerate harness — implemented in v0.1

- native reference conditioning is an input to the integrated second pass;
- FL2VA / Hybrid 45–49 / Ref2VA can be run without changing this repository;
- identical-arm benchmark configuration documented.

## M3a — per-step low-frequency fidelity — implemented experimentally

- H3 NestedTensor/packed post-CFG projector;
- low-frequency Base-draft correction per H3 evaluation;
- structure-first sigma schedule;
- correction RMS telemetry.

Remaining before acceptance:

- live ComfyUI validation with target samplers/backends;
- interaction tests with M4 and the optional M3 measurement constraints;
- confirm that late detail is not suppressed.

## M3b — normalized latent measurement backprojection — implemented experimentally

- explicit `D_latent(x0_HR) -> z_Base` constraint;
- area-downsample measurement operator;
- robust Base-grid residual weighting;
- configurable low/full-band residual mix;
- normalized bicubic residual lift using measured `D(U(r))` response;
- backprojection-gain clamp;
- optional internal re-measurement iterations;
- independent HR RMS correction guard;
- structure-first schedule;
- video-only NestedTensor/packed H3 runtime;
- audio invariant preserved;
- integrated optional `H3_ICR_MEASUREMENT` input on `Kirei H3 ICR Regenerate`;
- before/after measurement-error telemetry.

Remaining before acceptance:

- decoded-media comparison against M3a-only and constraint-off controls;
- tune strength, band mix and internal iteration count on dense ~1 MP and M4 2K;
- verify that lower Base-grid error does not suppress valid HR detail.

## M3c — latent posterior gradient — implemented experimentally

- explicit autograd gradient through `D_latent` only;
- no H3-model or VAE gradient;
- residual/gradient RMS normalization;
- correction cap relative to HR/Base latent scale;
- configurable cadence with `apply_every`;
- AV-container and packed-H3 support;
- exact audio pass-through;
- measurement-error and gradient telemetry;
- separate MODEL patch so it can be ablated against M3b.

Remaining before acceptance:

- determine whether the explicit latent gradient adds anything over M3b's cheaper normalized backprojection;
- measure wall-time overhead versus M3b;
- reject M3c if its decoded-media operating point is not better than M3b.

## M3d — proxy-decoder pixel measurement — implemented experimentally

- `x0_HR -> D_latent -> H3-compatible decoder` measurement path;
- differentiable decoder branch runs at Base latent geometry, not 2K geometry;
- 24-channel MiniMax H3 decoder validation;
- lightweight H3 TAE/`taeh3` is the recommended first proxy;
- full `MiniMaxH3VideoVAE` gradient is explicit opt-in;
- cached reduced Base pixel reference;
- RGB, spatial-edge and temporal-difference measurement losses;
- configurable spatial measurement size and temporal stride;
- pixel-RMSE-normalized gradient;
- conservative latent RMS correction cap;
- optional second decode for measured post-correction RMSE;
- AV-container and packed-H3 support;
- audio is copied unchanged;
- dedicated nodes, telemetry, documentation and CPU contract tests.

Initial laboratory values:

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

Remaining before acceptance:

- real `taeh3` CUDA backward validation inside ComfyUI;
- compare M3d against M3a/M3b/M3c under identical seeds and media;
- quantify proxy-decoder overhead and retained VRAM;
- test 256/384 measurement size and frame stride 1/2/4;
- determine whether edge or temporal terms materially improve decoded media;
- test full VisualVAE backward only if operationally feasible;
- reject M3d if proxy loss improves without visible fidelity/detail benefit.

## M4 — 2K renderer — experimental implementation in PR #1

Implemented on `feature/tiled-2k-fusion`:

- spatial tile planner over target-video rows only;
- global dynamic LR H3 prior;
- overlap-weighted model-output fusion at every H3 evaluation;
- sigma-aware global-prior schedule with configurable floor/power;
- full global text/reference/audio context;
- exact full-canvas MM-RoPE coordinates for target-video tile rows;
- native HR keyframe crop/downscale handling and global MM-RoPE remapping;
- Spectrum retained only on the stable global-prior branch; tile calls forced actual;
- live renderer and prior-schedule telemetry;
- 2048x1152 lab preset;
- fail-closed handling for unsupported topology/geometry.

Remaining before acceptance:

- decoded-media validation against dense ~1 MP H3-ICR;
- tune prior strength, prior floor/power and tile geometry from fidelity/VRAM measurements;
- verify HR-keyframe propagation on faces, hands, products and text;
- decide whether a dense/wider-context final tail is still required;
- compare dynamic LR prior against optional pass-1 trajectory replay if trajectory capture is available;
- investigate topology-safe cache semantics rather than enabling EasyCache blindly.

## M5a — passive calibrated attention v2 — implemented experimentally

- output-neutral function-style `optimized_attention_override` profiler;
- bounded sampled Q/K analysis without materializing SxS attention;
- importance-corrected modality mass for text / visual condition / audio condition / target audio / target video;
- exact sampled target-video QK pairs: diagonal, spatial neighbor, temporal neighbor and far-video;
- exact `spatial_minus_far` / `temporal_minus_far` margins per head;
- layer/head/sigma/M4-branch buckets;
- one canonical packed topology per branch per calibration run;
- topology descriptor includes native target signature and ordered segment kinds/row counts;
- architecture, topology and complete-profile SHA-256 fingerprints;
- proposal-only head classification using both modal mass and exact-pair evidence.

Remaining before acceptance:

- collect real profiles across FL2VA / Hybrid / Ref2VA;
- separate calibration by target geometry, duration and packed reference/keyframe topology;
- full `layer_stride=1` runs after light-profile stability is confirmed;
- quantify run-to-run / seed stability of head classification;
- verify profiler overhead and output neutrality on real CUDA H3.

## M5b — real FlexAttention sparse executor v2 — implemented experimentally

- real PyTorch FlexAttention + `BlockMask` execution;
- current and legacy proposal-label compatibility;
- proposal/architecture/profile fingerprint validation;
- runtime branch-specific packed-topology validation;
- dense topology fallback outside the calibrated domain;
- per-head local-3D / spatial / temporal target-video patterns;
- all text/reference/keyframe/audio and non-target-video links remain global/dense;
- mandatory dense sigma tail;
- dense fallback on non-CUDA, external masks, incomplete/dense policies or low measured block sparsity;
- topology/layer/policy/device BlockMask cache;
- modern PyTorch Triton Flex backend with legacy compatibility;
- telemetry for sparse calls, fallbacks, mask builds/cache hits and block sparsity.

Remaining before acceptance:

- real CUDA equivalence controls on target H3 checkpoints;
- benchmark first-use compilation/mask-build overhead separately from steady state;
- peak VRAM and wall-time measurements on dense ~1 MP and M4 2K;
- actual BlockMask sparsity by layer/head/topology;
- topology/policy fallback rates;
- tune dense-tail sigma and local radii from decoded media;
- complete decoded-video parity before any speedup/default claim.

## M5c — topology + sigma-domain policy v3 — implemented experimentally

- retains v2 aggregate layer policy for analysis;
- emits explicit `branch + topology digest + sigma + layer` domains;
- preserves exact QK-pair evidence per sigma domain;
- Flex runtime selects nearest calibrated sigma independently per layer;
- configurable `max_policy_sigma_distance`, default 0.03;
- sigma-domain miss -> native dense fallback;
- no categorical interpolation between unobserved sigma coordinates;
- BlockMask reuse remains possible when effective head codes are identical;
- sigma-domain match/fallback/distance telemetry.

Remaining before acceptance:

- compare v2 static topology-bound versus v3 sigma-domain policy on the same decoded-media benchmark;
- determine useful sigma sampling density from real H3 schedules;
- measure fallback frequency at tolerance 0.03;
- prefer v3 only if it improves parity or the speed/quality operating point.

## M6a — state-aware BaseVideo Adapter runtime scaffold — implemented

Implemented:

- frozen-H3-compatible residual adapter module;
- **static stream:** aligned clean Base latent patch rows;
- **dynamic stream:** current native H3 target-video hidden rows;
- sigma-conditioned static/dynamic structure-to-detail gating;
- linear-cost local 3D depthwise + pointwise feature mixer;
- exactly zero-initialized output projection;
- `trained=false` provider is bypassed before adapter compute;
- default plumbing blocks `12,24,36,45,48` with no optimality claim;
- native H3 architecture descriptor + SHA-256 binding;
- existing `double_block` patch-chain composition;
- static Base caches by geometry/device/dtype;
- exact M4 tile Base-region reconstruction from the tile's existing full-canvas MM-RoPE `position_ids`;
- ambiguous/non-native tile geometry fails closed;
- nodes and telemetry;
- exact zero-init parity and M4 tile-region unit tests.

## M6b — trained adapter checkpoint ABI + ComfyUI offload loader — implemented

Implemented:

- safetensors-only loader under `models/kirei_h3_adapters`;
- metadata API v1;
- checkpoint kind / model_id / architecture digest validation;
- exact adapter config validation;
- sorted/unique/in-range injection-block validation;
- strict state-dict key matching;
- NaN/Inf tensor rejection;
- complete checkpoint file SHA-256 provenance;
- active H3 dtype binding;
- adapter wrapped in ComfyUI `CoreModelPatcher` using normal load/offload devices;
- managed provider registration through `MODEL.set_additional_models`;
- loader/application/report nodes;
- metadata, strict-state, architecture mismatch and additional-model registration tests.

No trained checkpoint is shipped and no M6 quality gain is claimed.

Remaining before trained M6 acceptance:

- create/train first adapter checkpoint against the selected training-free teacher;
- real ComfyUI additional-model lifecycle/offload validation on CUDA;
- dense ~1 MP decoded-media comparison against the teacher;
- M4 2K decoded-media validation of MM-RoPE-derived Base tile crops;
- ablate injection blocks, adapter width, local kernels and gate schedule;
- measure adapter VRAM/wall-time overhead;
- verify identity/object/action/timing parity and temporal stability;
- add optional verified HR-keyframe stream only after Base+dynamic adapter behavior is measured.

## M6c — optional detail LoRA — planned

Only consider after the BaseVideo Adapter is trained and evaluated. Add a detail LoRA if a repeatable high-frequency gap remains that the adapter alone cannot close without harming fidelity.

## M7 — distillation

- establish a validated multi-step teacher first;
- progressive trajectory-preserving distillation;
- 2–4 step student before one-step experiments;
- optional one-step adversarial/preference refinement only after trajectory/fidelity parity is demonstrated.
