# M6 — State-Aware BaseVideo Adapter

Status: zero-init runtime scaffold implemented; no trained adapter checkpoint is provided or claimed.

## Purpose

M6 is the first trained stage of Kirei H3-ICR. The training-free M0–M5 stack tries to recover detail while constraining a second H3 pass toward the Base draft. M6 adds a learnable residual path that explicitly distinguishes:

- a **static Base stream** carrying the frame-aligned draft;
- a **dynamic stream** carrying the current H3 block state;
- a sigma-conditioned structure-to-detail gate;
- optional future verified HR-keyframe evidence.

The H3 backbone remains frozen initially.

## Why a scaffold exists before training

A trained adapter should not be introduced before its runtime contract is proven. The current M6 implementation therefore creates a provider with:

```text
trained = false
out_proj.weight = 0
out_proj.bias = 0
```

and the runtime bypasses residual injection for that provider.

The scaffold validates:

- native H3 architecture binding;
- selected block registration;
- composition with an existing `patches_replace` chain;
- provider ABI and metadata;
- zero-init parity;
- Base latent ownership/caching;
- sigma/branch runtime metadata;
- dense/global-prior Base alignment;
- M4 HR-tile Base-region reconstruction from global MM-RoPE.

It is **not** a quality feature and should not be benchmarked as one.

## Provider API v1

`H3_ICR_BASE_ADAPTER_PROVIDER` contains:

- API version;
- `BaseVideoAdapterConfig`;
- native H3 architecture descriptor;
- architecture SHA-256 digest;
- adapter module;
- `trained` flag;
- optional checkpoint SHA-256;
- provenance/note field.

Current architecture descriptor binds:

```text
model_id
native module/class
number of H3 blocks
hidden width
video patch size
video latent channels
audio latent channels
AdaLN-curve format
```

An architecture mismatch fails before a MODEL is patched.

A future checkpoint loader must preserve this provider ABI rather than inventing a second application path.

## Current adapter module

The scaffold uses a linear-cost local state-aware module.

Inputs at a selected native H3 block:

```text
dynamic_hidden: H3 target-video hidden rows
static_base:    clean H3 Base latent aligned to the active target/tile region
```

The module contains:

```text
dynamic H3 hidden
   -> LayerNorm -> dynamic projection ----+
                                            |
aligned Base patch rows -> static projection+--> sigma gate / fusion
                                            |
                                            v
                                     local 3D depthwise mixer
                                            |
                                     pointwise feature mixer
                                            |
                                        output norm
                                            |
                                 ZERO-INITIALIZED out projection
                                            |
                                            v
                                     H3 hidden residual
```

The local 3D mixer is intentionally linear in token count. The first scaffold does not introduce a quadratic Base-to-target attention matrix.

## Static / dynamic sigma gate

The aligned Base stream is emphasized earlier in pass 2 and the current H3 state gains relative weight later:

```text
r = clamp(sigma / sigma_start, 0, 1)
structure_gate = floor + (1 - floor) * r^power
anchor = structure_gate * static + (1 - structure_gate) * dynamic
```

Initial scaffold values:

```text
gate_floor: 0.15
gate_power: 1.0
```

These are architecture defaults, not trained or validated values.

## Initial injection blocks

The scaffold node defaults to:

```text
12,24,36,45,48
```

This only provides early/mid/late plumbing coverage. It is **not** a claim that these are optimal H3 adapter layers.

Training/ablation must compare narrower and wider layer sets.

## Zero-init invariant

The final residual projection is initialized exactly to zero. In addition, providers marked `trained=false` are bypassed before adapter computation during block injection.

Therefore the zero-init scaffold has two safety layers:

1. runtime bypass for untrained providers;
2. zero output projection even if the module is explicitly exercised in a test.

Unit tests require exact tensor equality for both paths.

## Existing block-patch composition

M6 follows the native ComfyUI MiniMax-H3 patch composition pattern:

```text
previous patch, if any
    -> original/previous block output
    -> M6 after-block residual
```

It never silently discards an existing `double_block` replacement.

## M4 tile alignment through global MM-RoPE

M4 already assigns every HR tile the exact target-video `position_ids` selected from the full-canvas H3 `PackedLayout`. M6 reuses that information instead of adding a second tile-coordinate API.

For a native H3 spatial axis:

```text
step = 32 * patch / sqrt(full_h * full_w)
```

M6 reads the first target-video frame's global H/W coordinates from the tile layout, infers the full latent area from the coordinate step, enumerates valid H/W factor pairs, and requires exactly one full-canvas geometry whose native H3 axis coordinates contain the tile axes as contiguous subsequences.

It then obtains:

```text
full_h, full_w
y0, y1
x0, x1
```

in latent units.

The static Base stream is:

```text
clean Base latent
  -> resize once to inferred full target latent H/W
  -> crop exact global tile rectangle
  -> native 1x2x2 patchify
  -> static adapter rows
```

This avoids the incorrect alternative of resizing the complete Base scene into every HR tile.

Safety rules:

- missing/non-native `position_ids` fail;
- inconsistent spatial MM-RoPE step fails;
- ambiguous/no full-canvas factorization fails;
- inferred crop must exactly match the active tile geometry;
- inferred region is cached for all selected adapter blocks in one H3 tile call;
- resized Base tensors and patch rows are cached by geometry/device/dtype.

Unit tests reconstruct a known full canvas and tile solely from global MM-RoPE and require the exact original tile bounds.

## Memory-management requirement

The zero-init scaffold owns no trained checkpoint and therefore does not yet expose an additional ComfyUI `ModelPatcher` from `models()`.

A trained provider loader must:

- wrap adapter weights in ComfyUI model-management/offload semantics;
- expose that patcher through the block-patch `models()` chain;
- support load/offload device transitions without orphan GPU residency;
- avoid silently moving a large trained adapter on every block call.

A trained provider is not production-safe until this is implemented.

## Future trained checkpoint format

The first checkpoint format should contain at minimum:

```text
api
architecture_digest
model_id
adapter config
injection blocks
training-data / teacher identifier
state_dict
checkpoint_sha256 / provenance metadata
```

Loading must fail closed if architecture or config is incompatible.

## Training direction

Initial training should freeze H3 and optimize only the adapter (and later an optional detail LoRA).

Data should include:

- real H3 Base rollouts;
- Base-to-HR teacher pairs from the validated training-free pipeline;
- H3/AIGC-like degradation rather than only bicubic camera-video downsampling;
- hard identity/object/text/hand cases;
- motion and disocclusion cases.

Candidate losses:

- teacher/flow residual objective;
- M3b-style latent measurement consistency;
- temporal consistency;
- reference/identity correctness where labels are available;
- high-frequency wavelet/HOG terms for detail;
- optional keyframe propagation loss.

## Required validation gates

Before enabling trained M6 by default:

1. zero-init scaffold exact parity;
2. checkpoint/architecture binding tests;
3. ComfyUI offload/lifecycle validation;
4. dense ~1 MP decoded-media comparison against the best training-free teacher;
5. real M4 2K validation of global-MM-RoPE Base cropping;
6. identity/object/action/timing parity;
7. faces/hands/text/detail improvement;
8. temporal stability;
9. VRAM and wall-time overhead;
10. ablation of injection blocks and gate schedule.

No M6 improvement is claimed by the current zero-init scaffold.
