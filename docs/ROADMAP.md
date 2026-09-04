# Roadmap

## M0 — contracts and reproducible backend matrix — implemented in v0.1
- backend-agnostic MODEL tag
- strict H3 AV contracts
- exact target geometry
- learned provider API v1 / bicubic control
- partial-sigma contract
- exact audio lock on sampler 2
- JSON metrics/report

## M1 — initialization fidelity — implemented in v0.1
- learned clean-video upscale
- low-frequency Fourier initialization alignment
- RMS correction guard
- downsample error telemetry

## M2 — in-context regenerate harness — implemented in v0.1
- native reference conditioning is an input to the integrated second pass
- FL2VA / Hybrid 45–49 / Ref2VA can be run without changing this repository
- identical-arm benchmark configuration documented

## M3 — per-step fidelity — first implementation in v0.1
- H3 NestedTensor/packed post-CFG projector implemented
- low-frequency draft projector per H3 evaluation implemented
- structure-first schedule implemented and telemetered
- remaining: live ComfyUI validation with Spectrum / SA-Solver and optional latent posterior consistency `D(x0_HR) -> x0_LR`

## M4 — 2K renderer — experimental implementation in PR #1
Implemented on `feature/tiled-2k-fusion`:
- spatial tile planner over target-video rows only
- global dynamic LR H3 prior
- overlap-weighted model-output fusion at every H3 evaluation
- sigma-aware global-prior schedule with configurable floor/power
- full global text/reference/audio context
- exact full-canvas MM-RoPE coordinates for target-video tile rows
- native HR keyframe crop/downscale handling and global MM-RoPE remapping
- Spectrum retained only on the stable global prior branch; tile calls forced actual
- live renderer and prior-schedule telemetry
- 2048x1152 lab preset
- fail-closed handling for unsupported topology/geometry

Remaining before M4 is accepted:
- decoded-media validation against dense ~1 MP H3-ICR
- tune prior strength, prior floor/power and tile geometry from measured fidelity and VRAM
- verify HR-keyframe propagation on faces, hands, products and text
- decide whether a dense/wider-context final tail is still required after media tests
- evaluate whether pass-1 trajectory replay improves the global prior over the current dynamic LR branch
- investigate safe cache semantics instead of enabling EasyCache blindly

## M5a — passive calibrated attention v2 — implemented experimentally
Implemented on the experimental branch:
- passive function-style `optimized_attention_override` profiler
- output-neutral/no-op test contract
- bounded sampled Q/K analysis without materializing SxS attention
- importance-corrected modality mass for text / visual condition / audio condition / target audio / target video
- exact sampled target-video QK pairs: diagonal, spatial neighbor, temporal neighbor and far-video
- exact `spatial_minus_far` / `temporal_minus_far` margins per head
- layer/head/sigma/M4-branch buckets
- one canonical packed topology per branch per calibration run
- topology descriptor includes native target signature plus ordered segment kinds/row counts
- architecture, topology and complete-profile SHA-256 fingerprints
- proposal-only head classification using both modal mass and exact-pair evidence

Remaining before M5a is accepted:
- collect real profiles across FL2VA / Hybrid / Ref2VA
- separate calibration by target geometry, duration and packed reference/keyframe topology
- full `layer_stride=1` runs after light-profile stability is confirmed
- quantify run-to-run / seed stability of head classification
- verify profiler overhead and output neutrality on the real CUDA H3 runtime

## M5b — real FlexAttention sparse executor v2 — implemented experimentally
Implemented on the experimental branch:
- real PyTorch FlexAttention + `BlockMask` execution path
- current and legacy proposal-label compatibility
- proposal/architecture/profile fingerprint validation
- runtime branch-specific packed-topology validation
- dense topology fallback outside the calibrated domain
- per-head local-3D / spatial / temporal target-video patterns
- all text/reference/keyframe/audio and non-target-video links remain global/dense
- mandatory dense sigma tail
- dense fallback on non-CUDA, external masks, incomplete/dense policies or low measured block sparsity
- topology/layer/policy/device BlockMask cache
- modern PyTorch `BACKEND="TRITON"` selection with legacy `FORCE_USE_FLEX_ATTENTION` compatibility
- runtime telemetry for sparse calls, all fallback classes, mask builds/cache hits and measured BlockMask sparsity

Remaining before M5b is accepted:
- real CUDA equivalence controls on the target H3 checkpoints
- benchmark first-use compilation/mask-build overhead separately from steady-state execution
- peak VRAM and wall-time measurements on dense ~1 MP and M4 2K
- actual BlockMask sparsity per layer/head topology
- topology-fallback and policy-fallback rates
- tune dense-tail sigma and local radii from decoded media rather than statistics alone
- complete decoded-video parity before any speedup/default claim
- consider alternative sparse kernels only if FlexAttention cannot deliver a useful measured gain on the target GPUs

## M5c — topology + sigma-domain policy v3 — implemented experimentally
Implemented as an optional policy layer over M5a/M5b:
- retains the v2 aggregate layer policy for analysis
- emits explicit `branch + topology digest + sigma + layer` domains from existing profiler buckets
- preserves exact QK pair evidence independently inside every sigma domain
- Flex runtime selects the nearest calibrated sigma independently per layer
- configurable `max_policy_sigma_distance`, default 0.03
- sigma-domain miss -> empty sparse layer map -> native dense fallback
- no categorical interpolation between unobserved sigma coordinates
- aggregate v2 layer map restored after every model call
- BlockMask reuse remains possible across sigmas only when the effective per-head policy codes are identical
- sigma-domain match/fallback and distance telemetry

Remaining before M5c is accepted:
- compare v2 static topology-bound policy against v3 sigma-domain policy on the same decoded-media benchmark
- determine useful sigma sampling density from real H3 schedules
- measure how frequently v3 falls back to dense at the default 0.03 tolerance
- tune tolerance only from parity/speed measurements, not to maximize sparse-call rate
- verify that sigma-specific policies improve parity or the speed/quality operating point before preferring v3 over v2

## M6 — BaseVideo Adapter + detail LoRA
- frozen H3 backbone
- static base-video stream + dynamic current-x0 stream
- timestep-dependent cross attention
- zero-init residual injection
- sparse HR keyframe anchors
- AIGC-oriented degradations

## M7 — distillation
- establish multi-step teacher first
- 2–4 step student
- one-step only after trajectory/fidelity parity is demonstrated
