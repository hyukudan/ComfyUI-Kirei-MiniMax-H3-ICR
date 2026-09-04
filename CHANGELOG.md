# Changelog

## Unreleased

- Bundle the native Kirei H3 learned latent-upscaler provider in this repository.
- Add explicit, atomic and SHA-256-verified download of the registered BF16 checkpoint.
- Reuse checkpoints already present in either `kirei_h3_upscalers` or `latent_upscale_models`.
- Add full-sequence inference (`temporal_chunk_size=0`) and learned-residual temporal stabilization.
- Support real ComfyUI `NestedTensor` containers in strict validation fingerprints.

## 0.1.0 — 2026-09-04

- Initial Kirei research implementation.
- Backend-agnostic FL2VA / Hybrid / Ref2VA model provenance tagging.
- Exact H3 AV contracts and target geometry validation.
- Direct clean H3 Base latent injection into `minimax_refs` (DiT-side path).
- Learned `H3_LATENT_UPSCALER` v1 integration with bicubic control.
- Fourier low-frequency initialization alignment and RMS guard.
- Per-step clean-state fidelity projection with structure-first sigma schedule.
- Partial-noise native ComfyUI H3 second pass and exact audio lock.
- Structured report and controlled backend ablation configuration.
- 12 model-independent unit tests.
