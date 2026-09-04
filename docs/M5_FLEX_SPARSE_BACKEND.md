# M5 — Experimental FlexAttention Sparse Backend

Status: v2 real block-sparse kernel path implemented; real H3 CUDA/media benchmarking is still pending.

## Why FlexAttention

PyTorch FlexAttention accepts a `BlockMask` that controls the block-sparsity pattern executed by the attention kernel. Kirei uses this path because it can skip complete Q/K blocks; applying a dense attention kernel to a large boolean/additive mask does not satisfy the M5 sparse-backend requirement.

The sparse path is CUDA-only. CPU and unsupported situations fall back to the existing ComfyUI attention implementation.

## Calibration contract

The executor does not invent a sparse policy from runtime geometry. It consumes the proposal JSON produced by the current M5 v2 passive profiler.

Before patching a `MODEL`, v2 validates:

- proposal content against `proposal_digest`;
- proposal `architecture_digest` against the current native H3 architecture plus the supplied `model_id`;
- presence of branch-specific `calibrated_topologies`;
- optionally, the supplied profile JSON against `profile_digest`;
- optionally, that the proposal `source_profile_digest` matches the supplied profile.

Architecture/profile mismatches fail closed at patch time.

At runtime, every native H3 call also checks its actual `PackedLayout` topology against the calibrated topology for that branch. If it is not an exact match, that call executes **dense** and increments `dense_topology_fallback_calls`.

The topology digest covers target signature plus the ordered packed segment kinds and row counts, so changing refs/keyframes/text/audio load can invalidate a policy even when target resolution is unchanged.

## Head policies

The executor understands both the original proposal labels and the current exact-pair v2 labels:

| Calibration class | Target-video attention | Cross-modal / non-video context |
| --- | --- | --- |
| `global_or_cross_modal` | dense | dense |
| `mixed_dense` | dense | dense |
| `local_3d_candidate` / `local_3d_pair_candidate` | local T/Y/X window | global |
| `spatial_window_candidate` / `spatial_pair_candidate` | same latent time + local Y/X | global |
| `temporal_stripe_candidate` / `temporal_pair_candidate` | local T at same Y/X | global |

Non-target-video queries remain dense. Sparse target-video queries can still attend to the complete text, reference/keyframe and audio prefix. This deliberately sacrifices theoretical sparsity to preserve H3's multimodal conditioning path.

## Dense tail

The backend automatically returns to the original attention implementation when:

```text
sigma <= dense_tail_sigma
```

Initial default: `0.12`.

Late densification is mandatory because the final denoising region is where small visual details are resolved and where an inaccurate sparse policy is most likely to produce visible artifacts.

## Real BlockMask execution

The sparse path imports:

```text
torch.nn.attention.flex_attention.create_block_mask
torch.nn.attention.flex_attention.flex_attention
```

It creates a head-specific `BlockMask` and dispatches the Q/K/V tensors through `flex_attention`. When configured, the kernel options request the FlexAttention path explicitly.

This is distinct from passing a QxK mask into a dense attention backend.

## BlockMask cache v2

BlockMask geometry does not change merely because sigma changes. v2 therefore caches masks by:

- M4/dense branch;
- calibrated topology digest;
- layer;
- head count and head policy;
- sequence geometry;
- latent T/H/W;
- device;
- block size.

**Sigma is intentionally not part of the cache key.** The same safe mask can be reused across denoising coordinates until the dense-tail gate takes over.

Runtime telemetry records mask builds and cache hits so the CUDA benchmark can include mask-construction overhead separately from steady-state attention cost.

## Minimum sparsity gate

The backend records `BlockMask.sparsity()`. If the generated mask is below `min_block_sparsity`, the call stays dense because launching the sparse path is unlikely to justify its overhead.

Initial default:

```text
min_block_sparsity = 5%
```

This is only a runtime-efficiency gate. It is not a quality threshold.

## M4 interaction

M4 produces distinct packed topologies for:

- `m4_global_prior`;
- `m4_hr_tile`.

The profiler calibrates them independently. The executor refuses to transfer one branch's topology policy to the other.

The target-video rows in M4 retain full-canvas MM-RoPE coordinates. Sparse neighborhood tests use the local packed target grid only to decide **which target-video Q/K pairs are computed**; they do not modify position IDs or RoPE.

If target geometry, reference rows or keyframe rows differ from calibration, the branch falls back to dense.

## Profiler chaining

Both M5 profiling and Flex sparse execution use ComfyUI's function-style `optimized_attention_override` contract. They can chain: the profiler observes normalized Q/K and delegates to the Flex dispatcher.

Container-style attention overrides remain rejected until a dedicated adapter exists.

## Dense/fallback rules

Sparse execution is skipped when any of the following applies:

- the call is not the active native H3 packed sequence;
- the current branch/topology is outside the policy's calibration domain;
- the policy has no complete head classification for the layer;
- all heads for the layer are dense;
- the call is inside the dense sigma tail;
- the current device is not CUDA;
- another attention mask is already active;
- the generated BlockMask does not meet the minimum block-sparsity gate;
- FlexAttention is unavailable.

These cases use the original ComfyUI attention path. Kirei does not silently replace a failed sparse configuration with another approximation.

## Telemetry

The v2 report includes:

- attention calls;
- real sparse calls;
- dense-tail calls;
- dense policy fallbacks;
- dense runtime fallbacks;
- dense topology fallbacks;
- BlockMask builds/cache hits;
- last/max measured BlockMask sparsity;
- architecture digest;
- source profile/proposal digests;
- calibrated topology map.

## Required validation before acceptance

The existence of a FlexAttention path is not evidence of useful acceleration. Acceptance requires real H3 CUDA runs on the target GPU and must report:

1. profiler passive/no-op parity;
2. output parity in an all-dense/fallback control configuration;
3. actual `BlockMask.sparsity()` by layer and topology;
4. mask-build versus cache-hit rates;
5. topology-fallback rate;
6. peak VRAM;
7. kernel/model wall time excluding and including first-use compilation/mask construction;
8. complete decoded-video comparison against dense H3-ICR;
9. faces, hands, text, identity, motion, audio and tile-boundary behavior;
10. performance for dense ~1 MP and M4 2K paths.

No sparse configuration should become the default until it wins this gate.
