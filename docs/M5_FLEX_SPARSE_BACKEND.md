# M5 — Experimental FlexAttention Sparse Backend

Status: kernel path implemented; real H3 CUDA/media benchmarking is still pending.

## Why FlexAttention

PyTorch FlexAttention accepts a `BlockMask` that controls the block-sparsity pattern executed by the attention kernel. Kirei uses this path only because it can skip complete Q/K blocks; applying a dense attention kernel to a large boolean/additive mask does not satisfy the M5 sparse-backend requirement.

The backend is CUDA-only in its sparse path. CPU and unsupported situations fall back to the existing ComfyUI attention implementation.

## Calibration contract

The backend does not invent a sparse pattern on its own. It consumes the proposal JSON produced from an M5 attention profile.

Before patching a MODEL it verifies:

- proposal content against `proposal_digest`;
- proposal `architecture_digest` against the current native H3 architecture plus the supplied `model_id`;
- optionally, the complete profile JSON against `profile_digest`;
- optionally, that the proposal `source_profile_digest` matches that profile.

A mismatch fails closed.

## Head policies

Current proposal classes map to runtime behavior as follows:

| Profile class | Target-video attention | Cross-modal / non-video context |
| --- | --- | --- |
| `global_or_cross_modal` | dense | dense |
| `mixed_dense` | dense | dense |
| `local_3d_candidate` | local T/Y/X window | global |
| `spatial_window_candidate` | same latent time + local Y/X | global |
| `temporal_stripe_candidate` | local T at same Y/X | global |

Non-target-video queries remain dense. Sparse target-video queries can still attend to the complete text, reference/keyframe and audio prefix. This deliberately sacrifices some theoretical sparsity to preserve H3's multimodal conditioning path.

## Dense tail

The backend automatically returns to the original attention implementation when:

```text
sigma <= dense_tail_sigma
```

Initial default: `0.12`.

Late densification is mandatory because the final denoising region is where small visual details are resolved and where an inaccurate sparse policy is most likely to produce visible artifacts.

## BlockMask

Initial default block size: `128`.

The mask is head-specific. For target-video rows it derives latent `(t, y, x)` coordinates from the native H3 packed-video row order. All non-video packed rows remain globally visible.

Block masks are cached by branch, layer, topology, device, head policy and block size. Runtime telemetry records:

- total attention calls;
- sparse calls;
- dense-tail calls;
- policy/runtime fallbacks;
- mask builds/cache hits;
- last/max `BlockMask.sparsity()`.

If measured block sparsity is below `min_block_sparsity`, the call stays dense because launching a sparse kernel would be unlikely to justify the complexity.

## M4 interaction

M4 uses several native H3 topologies:

- global LR prior;
- HR tiles.

The sparse runtime keys its mask cache by active topology and M4 branch. It never reuses a BlockMask across incompatible sequence geometries.

The M4 target-video rows retain their full-canvas MM-RoPE coordinates. Sparse neighborhood tests use local packed-grid coordinates only to decide **which target-video connections may be computed**; they do not alter H3 position IDs or RoPE.

## Profiler chaining

Both M5 profiling and Flex sparse execution use ComfyUI's function-style `optimized_attention_override` contract. The wrappers are designed to chain: profiling can observe normalized Q/K and then delegate to the sparse dispatcher. Container-style overrides are rejected until an explicit compatibility adapter exists.

## Safety / fallback rules

Sparse execution is skipped when:

- the attention call is not the active native H3 packed sequence;
- the policy has no complete head classification for that layer;
- every head for the layer is classified dense;
- the call is inside the dense sigma tail;
- the current device is not CUDA;
- another attention mask is already active;
- the generated BlockMask does not meet the configured minimum block sparsity.

These cases call the original ComfyUI attention backend rather than approximating silently.

## Required validation before acceptance

The existence of a FlexAttention kernel path is not evidence of a useful speedup. The acceptance gate requires real H3 runs on the target GPU and must report:

1. output parity in an all-dense/no-sparse control configuration;
2. actual `BlockMask.sparsity()` by layer/topology;
3. peak VRAM;
4. kernel/model wall time excluding and including first-use compilation/mask build;
5. complete decoded-video comparison against dense H3-ICR;
6. faces, hands, text, identity, motion and tile-boundary behavior;
7. fallback rate and dense-tail share;
8. performance for dense ~1 MP and M4 2K paths.

No sparse configuration should become the default until it wins this gate.
