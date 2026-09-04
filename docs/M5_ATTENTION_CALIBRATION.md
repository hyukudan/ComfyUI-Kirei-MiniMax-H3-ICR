# M5 — H3 Attention Profiling and Calibration

Status: measurement infrastructure implemented; sparse execution is **not** enabled.

## Purpose

The goal of M5 is to discover whether native MiniMax H3 attention contains stable per-layer, per-head and per-timestep structure that can be exploited by a real sparse kernel without changing decoded media.

Kirei H3-ICR deliberately separates **measurement** from **optimization**. A dense attention call plus a QxK mask is not considered a sparse implementation.

## Instrumentation point

Current ComfyUI attention supports `optimized_attention_override` through `transformer_options`. The Kirei profiler installs a function-style override that receives normalized H3 Q/K tensors, collects a bounded sample, and then delegates to the original attention implementation unchanged.

It does not modify Q, K, V, logits, masks, outputs, or the selected attention backend.

A diffusion-model wrapper supplies the active native H3 `PackedLayout`, sigma, target geometry, and M4 branch identity. Only attention calls whose sequence length matches that active H3 packed layout are treated as H3 DiT blocks; unrelated or text-refiner attention calls are skipped.

## Sampling strategy

The profiler never constructs a full SxS attention matrix. For every selected layer it:

- samples a bounded number of query rows per modality;
- samples a bounded number of key rows per modality;
- evaluates Q·K only for that sampled cross-product;
- applies `log(population/sample_count)` to sampled-key logits before softmax so modality-mass estimates account for different packed segment sizes.

Default research settings:

```text
layer stride:              5
query samples/modality:   24
key samples/modality:     48
sigma decimals:            3
max calibration buckets: 2048
```

After a light profile is shown to be stable, a full calibration run can use `layer_stride=1`.

## Modalities

Native H3 packed rows are grouped as:

- `text`;
- `visual_cond` — keyframe and reference image/video rows;
- `audio_cond` — condition/reference audio rows;
- `target_audio`;
- `target_video`.

The profiler estimates importance-corrected sampled attention mass from every sampled query modality to every sampled key modality.

## Target-video structural metrics

For target-video queries, sampled target-video keys are mapped back to the native H3 patch grid. Per head, the profiler estimates:

- same-frame mass;
- spatial radius <=2 patches inside the same frame;
- temporal radius <=1 latent step at the same spatial position;
- 3D local radius <=1 in time/Y/X.

These are calibration signals, not proof that a sparse pattern is safe.

## M4 interaction

Calls are labeled as:

- `dense`;
- `m4_global_prior`;
- `m4_hr_tile`.

If Spectrum forecasts a global-prior step without executing H3 attention, that step naturally produces no Q/K sample. HR M4 tile calls are currently forced real and therefore remain profileable.

## Fingerprints

Each report includes:

- optional user-supplied `model_id`;
- native H3 architectural fields;
- an architecture SHA-256 digest;
- a complete profile SHA-256 digest.

A future sparse policy must bind itself to these fingerprints and to its calibrated layout domain. It must not silently transfer to an incompatible checkpoint or topology.

## Proposal-only classification

The report node can derive conservative candidate labels:

- `local_3d_candidate`;
- `spatial_window_candidate`;
- `temporal_stripe_candidate`;
- `global_or_cross_modal`;
- `mixed_dense`.

These labels are suggestions for kernel experiments only. The generated policy explicitly states `proposal_only_no_sparse_kernel_enabled`.

## Required next M5 gate

Before any speedup claim, the sparse executor must provide:

1. a real block/window/sparse kernel backend;
2. checkpoint/layout/profile hash validation;
3. explicit dense fallback;
4. late-step densification;
5. a native-equivalence/no-op mode;
6. decoded-media comparison on the controlled H3-ICR benchmark matrix;
7. VRAM and wall-time telemetry demonstrating a real gain.

No quality conclusion should be drawn from attention statistics alone.
