# Research transfer ledger

The broader literature and implementation review is in `RESEARCH_SURVEY_v1_2.md`.

Engineering priorities derived from the research survey:

1. **MiniMax H3 public regeneration path** — feed the Base video and its original multimodal context back into H3 for the second pass.
2. **Backend comparison** — treat FL2VA, Hybrid and Ref2VA as interchangeable MODEL backends under identical reference conditioning.
3. **HiFlow-style initialization alignment** — align the clean HR initializer to the Base draft before adding partial target-grid noise.
4. **FrescoDiffusion-style 2K fusion** — combine HR tile predictions with a global LR prior at every denoising coordinate, not only after decoding.
5. **RALU transition discipline** — never resize an arbitrary noisy state across spatial geometries without explicit noise semantics.
6. **State-aware BaseVideo Adapter** — future learned path using a static draft stream plus the current denoising state with timestep-dependent gating.
7. **Sparse HR anchors** — optional verified keyframes for faces, text, hands and high-value objects.
8. **Calibrated sparse attention** — profile and calibrate per layer/head/timestep, then use real sparse kernels with explicit fallback.
9. **Teacher-first distillation** — stabilize the multi-step regeneration teacher before reducing it to 2–4 steps or one step.
