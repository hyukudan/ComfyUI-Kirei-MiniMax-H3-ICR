# M6 — State-Aware BaseVideo Adapter

Status: zero-init runtime scaffold and trained-checkpoint loading ABI implemented; no trained adapter checkpoint is shipped or quality claim is made.

## Purpose

M6 is the first trained stage of Kirei H3-ICR. The training-free M0–M5 stack tries to recover detail while constraining a second H3 pass toward the Base draft. M6 adds a learnable residual path that explicitly distinguishes:

- a **static Base stream** carrying the frame-aligned draft;
- a **dynamic stream** carrying the current H3 block state;
- a sigma-conditioned structure-to-detail gate;
- optional future verified HR-keyframe evidence.

The H3 backbone remains frozen initially.

## Why a scaffold exists before training

A trained adapter should not be introduced before its runtime contract is proven. The scaffold provider is created with:

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

A managed trained provider additionally contains:

- validated checkpoint metadata;
- a ComfyUI `CoreModelPatcher` owning the adapter module.

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
   -> LayerNorm -> dynamic projection -----+
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

This avoids resizing the complete Base scene into every HR tile.

Safety rules:

- missing/non-native `position_ids` fail;
- inconsistent spatial MM-RoPE step fails;
- ambiguous/no full-canvas factorization fails;
- inferred crop must exactly match the active tile geometry;
- inferred region is cached for all selected adapter blocks in one H3 tile call;
- resized Base tensors and patch rows are cached by geometry/device/dtype.

Unit tests reconstruct a known full canvas and tile solely from global MM-RoPE and require the exact original tile bounds.

## Trained checkpoint format v1

The runtime loader accepts **safetensors only** (`.safetensors` / `.sft`) from:

```text
ComfyUI/models/kirei_h3_adapters/
```

The state dict must contain exactly the `StateAwareBaseVideoAdapter` tensor keys. Missing or unexpected tensors fail before the provider is created, and floating tensors containing NaN/Inf are rejected.

Required safetensors metadata:

```text
kirei_h3_icr_api
kirei_h3_icr_kind
kirei_h3_icr_architecture_digest
kirei_h3_icr_model_id
kirei_h3_icr_config_json
```

Optional metadata:

```text
kirei_h3_icr_training_json
kirei_h3_icr_note
```

Required values:

```text
kirei_h3_icr_api  = "1"
kirei_h3_icr_kind = "base_video_adapter"
```

`config_json` must contain exactly:

```json
{
  "injection_blocks": [12, 24, 36, 45, 48],
  "adapter_dim": 256,
  "gate_floor": 0.15,
  "gate_power": 1.0,
  "temporal_kernel": 3,
  "spatial_kernel": 3
}
```

Unknown or missing config fields fail closed. Injection blocks must be sorted, unique and valid for the active native H3 model.

The loader calculates the complete checkpoint file SHA-256 and stores it in the provider report.

## Architecture and checkpoint binding

The checkpoint's architecture digest is compared against the descriptor derived from the **actual MODEL connected to the loader node**, including the checkpoint-supplied `model_id`.

A trained checkpoint cannot therefore silently load on a different H3 hidden width, block count, patch contract, latent channel contract or AdaLN format.

The checkpoint state dict is then loaded with exact-key/strict semantics.

## ComfyUI memory management

A successfully loaded trained adapter is wrapped in a ComfyUI `CoreModelPatcher` using:

```text
load_device    = get_torch_device()
offload_device = unet_offload_device()
```

The module is converted to the active H3 model dtype when H3 exposes one; otherwise the loader uses the checkpoint floating dtype.

When the adapter is applied, its patcher is registered through:

```text
MODEL.set_additional_models("h3_icr_base_video_adapter", [adapter_patcher])
```

This is important: sampler preparation includes the MODEL's nested additional models, so adapter weights participate in normal ComfyUI load/offload decisions rather than living as an unmanaged GPU module.

The trained loader does not vendor or alter the H3 checkpoint.

## Nodes

### Zero-init ABI scaffold

**Kirei H3 ICR BaseVideo Adapter Scaffold [M6]**

Creates an untrained API-v1 provider for exact-parity/plumbing tests.

### Trained checkpoint loader

**Kirei H3 ICR Load BaseVideo Adapter [M6]**

Loads and validates a safetensors checkpoint from `models/kirei_h3_adapters` and returns a managed trained provider.

### Application

**Kirei H3 ICR Apply BaseVideo Adapter [M6]**

Accepts either the zero-init provider or a managed trained provider. Managed providers are registered as ComfyUI additional models automatically.

### Report

**Kirei H3 ICR BaseVideo Adapter Report**

Reports architecture/checkpoint provenance, managed/offload state, branch/tile statistics and residual RMS.

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
2. safetensors metadata/strict-state architecture binding tests;
3. real ComfyUI additional-model load/offload lifecycle validation;
4. dense ~1 MP decoded-media comparison against the best training-free teacher;
5. real M4 2K validation of global-MM-RoPE Base cropping;
6. identity/object/action/timing parity;
7. faces/hands/text/detail improvement;
8. temporal stability;
9. VRAM and wall-time overhead;
10. ablation of injection blocks and gate schedule.

No M6 quality improvement is claimed until a trained checkpoint passes these gates.
