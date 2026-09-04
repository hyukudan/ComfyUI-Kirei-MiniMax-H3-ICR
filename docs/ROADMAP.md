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
- remaining: live ComfyUI validation with Spectrum / SA-Solver and optional latent posterior consistency D(x0_HR) -> x0_LR

## M4 — 2K renderer
- spatial tile planner over video rows only
- global LR prior
- overlap blend of x0/velocity at every diffusion step
- full global text/reference/audio context
- stable MM-RoPE global coordinates
- dense final tail

## M5 — calibrated attention
- per layer/head/timestep modality profiling
- real sparse kernel backend only; dense masks do not count
- checkpoint/layout-bound profile hashes
- dense fallback and late densification

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
