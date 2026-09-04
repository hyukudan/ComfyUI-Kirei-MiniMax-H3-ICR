# Kirei MiniMax H3 ICR — Engineering Specification v1.2

Status: research / experimental  
Target: ComfyUI MiniMax H3  
Primary goal: regenerate an H3 Base result at higher spatial fidelity by feeding the draft and its original multimodal context back into H3, while preserving motion, identity, object state, timing and pass-1 audio.

> H3-ICR is an independent implementation. It is not MiniMax H3-Regenerate-2K and does not claim to reproduce MiniMax's private sparse-attention or regeneration internals.

## 1. Design objective

The project must optimize **correct additional detail**, not raw sharpness. A treatment loses if it is sharper but changes the draft's people, products, text, pose, action, camera, timing or scene state.

Quality ranking is therefore normative:

1. draft geometry and motion fidelity;
2. identity and object correctness;
3. temporal consistency and disocclusion behavior;
4. faces, hands, small text and product detail;
5. perceptual detail / sharpness;
6. wall time and memory.

## 2. Public H3 model contract

The implementation assumes current ComfyUI MiniMax H3 semantics:

- video latent: `B x 24 x T x H x W`;
- audio latent: `B x 32 x 2 x Ta`;
- video and audio are sampled jointly through the H3 AV model;
- video VAE spatial factor is 16;
- DiT video patch is spatial `2 x 2`, therefore a 32-pixel output alignment is the safe public canvas contract;
- prompt/Qwen conditioning, keyframes and `minimax_refs` are supplied through normal ComfyUI conditioning;
- audio has its own flow shift but rides through the packed AV sampling contract;
- the target second pass must use `0 <= sigmas[0] < 1`; full-noise start is invalid because it discards the learned clean initializer.

Unsupported or ambiguous layouts must fail closed.

## 3. Backend architecture

Reference conditioning and model backend are independent axes.

### 3.1 Conditioning

Build one common positive conditioning object with ComfyUI's native MiniMax H3 reference path. It should contain, when available:

- the decoded H3 Base video as a reference video so Qwen sees the draft;
- the clean H3 Base VisualVAE latent as a direct `minimax_refs` block to avoid an unnecessary decode/encode round-trip on the DiT-side reference path;
- the original prompt / Context-IR text;
- the original images, videos and audio in the same order used for pass 1;
- optional verified HR keyframes for future sparse-anchor experiments.

The same conditioning object must be reused across backend comparisons.

### 3.2 Backends

Required ablation arms:

- `fl2va_reference` — FL2VA model with the common reference conditioning;
- `hybrid_late_adaln` — external Hybrid 45–49 model with the same conditioning; current laboratory candidate;
- `ref2va_reference` — Ref2VA control arm;
- `hybrid_all_adaln` — higher-risk experimental arm only.

This repository must not vendor or reproduce the GPL Hybrid loader. It consumes the resulting ComfyUI `MODEL`.

Backend metadata is attached to a cloned `MODEL` and included in the run report. Source `MODEL` objects must never be mutated in place.

## 4. Clean HR initialization

Input is the clean pass-1 H3 AV latent.

### 4.1 Spatial transfer

Preferred path:

```text
clean H3 Base video latent
    -> H3_LATENT_UPSCALER API v1 provider
    -> exact target latent H/W
```

Control path: bicubic latent interpolation.

The provider must:

- preserve B/C/T;
- return exactly the requested target H/W;
- return a finite floating tensor;
- never receive or spatially transform audio.

### 4.2 Initialization alignment

The clean HR latent is aligned to the Base draft before re-noising. Let `U` be the learned/bicubic transfer and `D` an area downsample back to source geometry. The initial correction is:

```text
r_lr = LPF(z_lr - D(z_hr))
r_hr = resize(r_lr, target)
z_hr_aligned = z_hr + strength * r_hr
```

The correction is bounded by a per-sample RMS ratio. The filter is spatial only; time and channels are preserved. High-frequency degrees of freedom are deliberately left available to H3.

Report at minimum:

- source and target latent geometry;
- transfer mode;
- downsample error before/after alignment;
- correction RMS ratio;
- clamp scale.

## 5. Second-pass sampling

The second pass uses ComfyUI's native H3 sampling path.

Required behavior:

1. clone/mark the model as an H3 refinement invocation;
2. preserve a full-trajectory sigma reference in transformer options for compatible external patches;
3. generate fresh noise directly at final AV geometry;
4. begin from a partial H3 sigma schedule;
5. sample against the common positive conditioning;
6. use positive-only H3 guider semantics unless a deliberate negative conditioning is connected;
7. preserve callback/previews through the native sampler path;
8. remove obsolete source downscale metadata from the returned LATENT.

No manual arbitrary resize of a noisy H3 state is permitted.

## 6. Audio invariant

Default is `lock_audio=true`.

When locked:

- pass-2 audio noise is exactly zero;
- video is allowed to regenerate while audio remains available to H3's cross-modal path;
- after sampling, the returned audio latent is replaced with the clean pass-1 audio latent;
- audio never enters the spatial upscaler or spatial fidelity projector.

The implementation should eventually expose a debug hash/equality assertion for exact audio preservation.

## 7. Per-step draft fidelity

The first implemented policy is a post-CFG clean-state projector over H3's predicted clean video.

For each eligible model evaluation:

```text
x0_hr -> downsample -> compare against z_lr
      -> spatial low-pass residual
      -> lift residual to HR
      -> bounded correction
```

The correction schedule is structure-first:

```text
w(sigma) = floor + (1-floor) * clamp(sigma / sigma_start, 0, 1)^power
```

Thus draft structure is strongest at the beginning of pass 2 and relaxes toward sigma 0 so H3 can synthesize late microdetail.

The projector must support both representations observed around ComfyUI's sampler hooks:

- H3 AV NestedTensor clean states;
- packed AV tensors with live `latent_shapes`.

Unknown layouts must error rather than silently disabling fidelity.

Telemetry:

- calls;
- applied calls;
- mean/max correction RMS ratio;
- last schedule value.

## 8. Mandatory A/B/C/D media gate

Only the backend may change between arms. Lock:

- pass-1 H3 Base latent;
- decoded Base video reference;
- original prompt and context;
- original reference order;
- resulting positive conditioning;
- `minimax_refs` / keyframes;
- target dimensions;
- learned-upscaler checkpoint/device/precision;
- target noise tensor / seed;
- sigma tensor;
- sampler;
- fidelity parameters;
- audio latent.

Do not select a backend from a single still frame or a sharpness metric. Inspect the full decoded video.

The current hypothesis is Hybrid 45–49 > FL2VA > Ref2VA for visual detail, but this is explicitly **not** an accepted result until the controlled H3-ICR media gate is completed.

## 9. M4 — 2K renderer

A 2K path must not simply run independent tiles and blend final RGB frames.

Normative design:

- keep a global LR spatiotemporal prior;
- split only the target video spatial grid into overlapping HR tiles/windows;
- keep text, references and audio globally available;
- preserve global MM-RoPE coordinates for each tile;
- evaluate tile outputs at the same diffusion coordinate;
- fuse overlapping tile `x0` or velocity predictions **at every diffusion step**;
- include the global LR prior as a regularizer in the fusion;
- use overlap weights / least-squares style reconstruction rather than hard seams;
- end with a dense or wider-context tail when feasible.

This milestone is inspired by FrescoDiffusion/global-local high-resolution work but must remain H3-native in AV packing, flow coordinates and MM-RoPE.

Acceptance:

- no visible tile seams;
- no systematic inter-tile identity drift;
- same object/action state as the Base draft;
- measured memory reduction or enablement of otherwise infeasible final geometry;
- decoded-media comparison against dense ~1 MP baseline.

## 10. M5 — calibrated attention

Sparse attention is not accepted merely because a dense attention backend receives a QxK mask.

Required research procedure:

1. profile dense H3 by layer/head/timestep/modality;
2. classify useful local/global/diagonal/reference patterns;
3. calibrate over multiple prompts, durations, aspects and reference loads;
4. bind every profile to checkpoint/layout hashes;
5. dispatch to real sparse kernels;
6. keep text/audio/reference paths global unless measurements justify otherwise;
7. densify late steps or fall back to native attention when confidence is insufficient.

Quality regressions must be measured on decoded video, not only attention statistics.

## 11. M6 — BaseVideo Adapter and detail LoRA

Only after the training-free baseline is characterized.

Backbone remains frozen initially.

Adapter concept:

- **static stream:** Base/draft video features;
- **dynamic stream:** current target `x0` / denoising state;
- optional HR keyframe stream;
- timestep-conditioned cross attention moving from structure alignment toward detail synthesis;
- zero-initialized residual injection so the untrained adapter begins at backbone parity;
- selected blocks only until measurements justify wider injection.

Training data should include H3/AIGC-like degradations and actual H3 Base rollouts, not just bicubic downsampling of camera footage.

Loss stack should combine:

- teacher/flow objective;
- LR measurement consistency;
- temporal consistency;
- identity/object/reference losses where available;
- frequency/wavelet/HOG-style detail recovery;
- penalties for hallucinated structure.

## 12. Sparse HR keyframes

Optional future control path inspired by sparse-keyframe VSR methods.

Candidate HR keyframes may target:

- first frame / chunk anchor;
- faces;
- hands;
- text;
- products or high-value objects.

Every generated/detail-enhanced keyframe must pass a degradation gate: after reducing it to Base resolution it must remain sufficiently consistent with the corresponding Base frame. Failed anchors are rejected rather than injected.

## 13. Posterior consistency

Experimental extension after the basic per-step projector.

Tier 1: latent measurement residual `D_latent(x0_hr) - z_lr`.  
Tier 2: periodic VAE-domain degradation consistency with gradients back to latent state.

Both need bounded corrections, cadence controls and telemetry. They must never modify locked audio.

## 14. Distillation

Do not optimize for one-step generation before a high-quality multi-step teacher exists.

Order:

1. establish multi-step H3-ICR teacher;
2. preserve trajectory/fidelity while reducing to 2–4 steps;
3. add distribution/perceptual/adversarial objectives only after structural parity;
4. preference tuning last;
5. preserve Base-video constraints, HR anchors and audio invariants throughout.

## 15. Testing and CI

Model-independent tests must run without ComfyUI and cover:

- AV shape validation;
- target alignment;
- backend metadata immutability;
- direct Base reference insertion;
- learned-provider exact target contract;
- Fourier filtering/alignment;
- per-step schedule/projection;
- partial-sigma rejection.

GitHub Actions runs pytest and Ruff on pushes and pull requests.

Runtime integration tests are a separate gate because H3 weights and ComfyUI are not part of lightweight CI.

## 16. Failure policy

Fail closed on:

- malformed H3 AV tensors;
- odd latent H/W;
- invalid target grid;
- provider API/kind mismatch;
- non-finite upscaler output;
- full-noise pass-2 start;
- incompatible refinement metadata;
- unknown packed-layout fidelity state;
- any future sparse/tiled backend that cannot prove its expected geometry/profile.

A fallback must be explicit and reported; it must never masquerade as the requested experimental path.

## 17. Current implementation boundary

Implemented in v0.1:

- M0 contracts/backend matrix;
- M1 learned/bicubic clean initialization + Fourier alignment;
- M2 backend-agnostic in-context regenerate harness;
- first M3 per-step fidelity projector;
- exact default audio lock;
- run reports;
- unit tests and CI.

Not implemented yet:

- posterior-gradient consistency;
- Fresco-style tiled 2K per-step fusion;
- calibrated real sparse kernels;
- BaseVideo Adapter / detail LoRA training;
- automatic sparse HR keyframe generation/gating;
- distillation.

Those are deliberate next milestones, not implied capabilities of v0.1.
