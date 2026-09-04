# ComfyUI Kirei MiniMax H3 ICR

Kirei research implementation of **H3 In-Context Regeneration (H3-ICR)**: a second-pass MiniMax H3 workflow that
reuses the H3 Base video, the original multimodal context, and a clean learned HR latent initialization to let H3
synthesize substantially more spatial detail without throwing away the first pass.

> This is not MiniMax's private H3-Regenerate-2K implementation.

## v0.1 status

Implemented now:

- backend-agnostic second pass: FL2VA, Hybrid, Ref2VA or another compatible H3 MODEL;
- explicit backend metadata for A/B testing;
- strict H3 joint audio/video latent validation;
- exact 32-pixel target grid contract;
- `H3_LATENT_UPSCALER` v1 provider integration from the companion learned 3D upscaler;
- bicubic control path;
- Fourier low-frequency initialization alignment with an RMS safety guard;
- per-step H3 clean-state fidelity projection (NestedTensor and packed AV paths) with a structure-first decay schedule;
- native ComfyUI partial-noise sampler path;
- exact pass-1 audio preservation;
- structured run report;
- unit tests for the model-independent core.

## Recommended graph

1. Generate H3 Base normally and keep its clean AV latent.
2. Build **second-pass conditioning** with ComfyUI's native `MiniMax H3 Reference to Video`:
   - include the H3 Base decoded video as a reference video;
   - include the original reference images/videos/audio again;
   - reuse the original prompt / Context-IR content;
   - configure the node for the final target geometry.
3. Load one backend arm:
   - FL2VA pure;
   - Hybrid 45–49 (current lab candidate);
   - Ref2VA pure;
   - all-AdaLN only as an experimental arm.
4. Optional: pass MODEL through **H3 ICR Backend Tag**.
5. Connect the companion **MiniMax H3 Latent Upscaler Provider (3D)**.
6. Use a partial sigma schedule (`sigmas[0] < 1`).
7. Run **H3 ICR Regenerate**.

Do not decide the winner from sharpness alone. Rank arms by draft fidelity, identity/object correctness, temporal
stability, small text/faces/hands, then perceptual detail.

## Nodes

- `H3 ICR Backend Tag` — attaches explicit backend provenance to a MODEL clone.
- `H3 ICR Append Base Latent Reference` — injects the exact clean H3 Base latent into `minimax_refs` without a VAE round-trip (DiT-side only).
- `H3 ICR Prepare Clean HR` — learned/bicubic clean upscale + low-frequency draft alignment.
- `H3 ICR Regenerate` — integrated clean initialization + native second H3 sampling pass.
- `H3 ICR Report JSON` — renders the structured report.

## Companion integration

Preferred learned initializer:

- `xmarre/Comfyui_Minimax_h3_latent_Upscaler`
- use its `H3_LATENT_UPSCALER` API-v1 provider.

Hybrid backend is intentionally external. This repo consumes the resulting `MODEL`; it does not copy or vendor the
GPL Hybrid loader.

## Tests

```bash
python -m pytest
```

The unit tests do not require ComfyUI. The actual sampler node requires a current ComfyUI runtime with MiniMax H3.

## Next milestones

See `docs/ROADMAP.md`. Per-step fidelity guidance is implemented in v0.1. The next engineering target is **Fresco-style per-step tiled 2K fusion**, followed by calibrated sparse attention and the trained BaseVideo Adapter once the baseline matrix is measured.
