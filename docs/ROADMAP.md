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

## M5 — calibrated attention — profiler and experimental real sparse backend implemented
Implemented on the experimental branch:
- non-destructive `optimized_attention_override` profiler
- bounded sampled Q/K analysis without materializing SxS attention
- layer/head/sigma/branch buckets
- text / visual-condition / audio-condition / target-audio / target-video modality accounting
- target-video same-frame, spatial-local, temporal-local and 3D-local concentration metrics
- architecture and complete-profile SHA-256 fingerprints
- proposal-only head classification
- experimental PyTorch FlexAttention `BlockMask` backend
- proposal/architecture/profile fingerprint validation
- per-head local-3D / spatial-window / temporal-stripe target-video masks
- all non-video context globally visible to sparse target-video queries
- mandatory dense sigma tail
- dense fallback for missing policy/topology, non-CUDA execution, existing attention masks or low measured block sparsity
- BlockMask caching and real block-sparsity telemetry

Remaining before M5 is accepted:
- collect real profiles across FL2VA / Hybrid / Ref2VA, prompts, reference loads, durations, aspects and M4 branches
- verify profiler output neutrality on the real H3 runtime
- run CUDA equivalence controls for the Flex backend
- benchmark first-use and steady-state wall time separately
- measure peak VRAM and `BlockMask.sparsity()` per layer/topology
- derive policies by sigma/topology rather than only aggregate layer classification
- validate dense-tail threshold and local radii from media results
- decoded-media parity before any speedup/default claim
- consider alternative kernels only if FlexAttention cannot deliver useful sparse execution on the target GPUs

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
