# M5 — H3 Attention Profiling and Calibration

Status: passive v2 measurement infrastructure implemented; experimental sparse execution is implemented separately and remains unvalidated.

## Purpose

M5 measures whether native MiniMax H3 attention contains stable per-layer, per-head, per-sigma and per-topology structure that can be exploited by a real sparse kernel without changing decoded media.

Kirei H3-ICR deliberately separates **measurement** from **execution**. The profiler itself is passive. It never makes an attention connection sparse, and a dense attention call with a QxK mask is not considered a sparse implementation.

## Instrumentation point

Current ComfyUI attention supports `optimized_attention_override` through `transformer_options`. The Kirei profiler installs a function-style override that receives normalized H3 Q/K tensors after H3's Q/K normalization and MM-RoPE application, collects bounded measurements, and then delegates to the original attention implementation unchanged.

It does not modify Q, K, V, logits, masks, outputs or the selected attention backend.

A diffusion-model wrapper supplies the active native H3 `PackedLayout`, sigma, target geometry and M4 branch identity. Only attention calls whose sequence length matches that active H3 packed layout are treated as H3 DiT blocks; unrelated/text-refiner attention calls are skipped.

## Two complementary evidence sources

M5 v2 deliberately uses two forms of evidence instead of deriving a sparse policy from one sampled statistic.

### 1. Importance-corrected sampled modality mass

For every selected layer it:

- samples a bounded number of query rows per modality;
- samples a bounded number of key rows per modality;
- evaluates Q·K only for that sampled cross-product;
- applies `log(population/sample_count)` to sampled-key logits before softmax so estimated modality mass accounts for different packed segment sizes.

The packed modalities are:

- `text`;
- `visual_cond` — target keyframes and reference image/video rows;
- `audio_cond` — condition/reference audio rows;
- `target_audio`;
- `target_video`.

This estimates whether a head is predominantly target-video self-attention or depends strongly on global/cross-modal context.

### 2. Exact sampled QK pair margins inside target video

For sampled target-video queries, v2 also selects deterministic **exact paired keys** from the native H3 target-video grid:

- the same row (`diagonal_score`);
- an adjacent spatial patch (`spatial_neighbor_score`);
- the same spatial coordinate in an adjacent latent-time position (`temporal_neighbor_score`);
- a deliberately distant video position (`far_video_score`).

The report stores:

```text
spatial_minus_far  = spatial_neighbor_score  - far_video_score
temporal_minus_far = temporal_neighbor_score - far_video_score
```

These margins are QK affinity probes, not full softmax attention mass. They complement rather than replace the sampled modality-mass estimate.

The current policy proposal uses both sources:

- low target-video mass -> keep the head dense/global;
- positive spatial and temporal margins -> candidate local 3D head;
- positive spatial margin only -> candidate spatial head;
- positive temporal margin only -> candidate temporal head;
- otherwise -> mixed/dense.

The current pair-margin threshold (`0.05`) is a research heuristic used only to prioritize experiments. It is not a proven safe sparsity threshold.

## Default calibration settings

```text
layer stride:              5
query samples/modality:   24
key samples/modality:     48
sigma decimals:            3
max calibration buckets: 2048
```

After a light profile is stable, a full calibration can use `layer_stride=1`.

## Branches and packed-topology binding

Calls are labeled as:

- `dense`;
- `m4_global_prior`;
- `m4_hr_tile`.

M5 v2 binds every branch to one canonical packed topology for a calibration run. The topology descriptor contains:

- the native five-field `PackedLayout.signature` (`text_len`, target latent T/H/W, target audio T);
- the ordered packed segment table as `(kind, row_count)`.

The descriptor is SHA-256 digested and emitted as `calibrated_topologies`.

This means a change in any of the following changes the calibration domain:

- target geometry/aspect;
- temporal latent length;
- text length;
- audio length;
- number/type/size of reference rows;
- number/type/size of keyframe rows.

If one profiling run sees two different packed topologies under the same branch name, it fails and asks for separate calibration runs rather than averaging incompatible measurements.

M4 can legitimately have distinct calibrated topologies for `m4_global_prior` and `m4_hr_tile`; both are recorded separately.

## Spectrum interaction

If Spectrum forecasts a global-prior step without executing H3 attention, that step naturally contributes no Q/K sample. HR M4 tile calls are currently forced real and therefore remain profileable.

## Fingerprints

Each v2 report includes:

- optional user-supplied `model_id`;
- native H3 architectural fields;
- architecture SHA-256 digest;
- branch-specific packed-topology descriptors and digests;
- exact-pair evidence;
- complete profile SHA-256 digest.

The v2 policy copies the calibrated topology map and binds its `source_profile_digest` and `architecture_digest` to the report.

A sparse executor must therefore validate both the model architecture and the active packed topology. A matching resolution alone is not sufficient if the packed conditioning layout changed.

## Proposal-only classification

The v2 report derives conservative labels:

- `local_3d_pair_candidate`;
- `spatial_pair_candidate`;
- `temporal_pair_candidate`;
- `global_or_cross_modal`;
- `mixed_dense`.

Legacy proposal labels remain understood by the experimental executor so old calibration experiments can still be compared, but new profiles should use the v2 labels.

The policy still explicitly states:

```text
proposal_only_no_sparse_kernel_enabled
```

Producing this proposal does not activate sparse execution.

## Passive-equivalence gate

Profiler validation includes a no-op contract: with the profiler installed, the attention output returned to H3 must be exactly the output returned by the wrapped original backend for the same Q/K/V inputs. The profiler may add measurement overhead but may not change values.

This is tested independently from the sparse executor.

## Relationship to the Flex sparse backend

The experimental Flex backend is described in [`M5_FLEX_SPARSE_BACKEND.md`](M5_FLEX_SPARSE_BACKEND.md). It consumes the v2 proposal, checks architecture and packed-topology fingerprints, and only then maps candidate heads to real `BlockMask` patterns.

Profiler measurements alone are not a quality conclusion. The decoded-media and CUDA performance gate remains mandatory.
