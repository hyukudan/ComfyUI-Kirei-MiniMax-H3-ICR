# M6 — State-Aware BaseVideo Adapter

Status: zero-init runtime scaffold and trained-checkpoint loading ABI implemented; no trained adapter checkpoint is shipped and no quality improvement is claimed.

## Purpose

M6 is the first trained stage of Kirei H3-ICR. It adds a learnable residual path that distinguishes:

- a **static Base stream** carrying the frame-aligned draft;
- a **dynamic stream** carrying the current H3 block state;
- a sigma-conditioned structure-to-detail gate;
- block-specific residual semantics;
- optional future verified HR-keyframe evidence.

The H3 backbone remains frozen initially.

## Zero-init scaffold before training

The scaffold provider is created with:

```text
trained = false
out_proj[block].weight = 0
out_proj[block].bias = 0
```

for every configured injection block, and the runtime bypasses residual injection entirely while `trained=false`.

The scaffold validates:

- native H3 architecture binding;
- selected block registration;
- composition with existing `patches_replace` chains;
- provider ABI and metadata;
- exact zero-init parity;
- Base latent ownership/caching;
- sigma/branch runtime metadata;
- dense/global-prior Base alignment;
- M4 HR-tile Base-region reconstruction from global MM-RoPE.

It is not a quality feature.

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

A managed trained provider additionally contains validated checkpoint metadata and a ComfyUI `CoreModelPatcher` owning the adapter module.

The architecture descriptor binds:

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

An architecture mismatch fails before the H3 MODEL is patched.

## Adapter architecture

At a selected native H3 block:

```text
dynamic_hidden: H3 target-video hidden rows
static_base:    clean H3 Base latent aligned to the active target/tile region
```

The adapter is:

```text
dynamic H3 hidden
   -> LayerNorm -> shared dynamic projection -----+
                                                     |
aligned Base patches -> shared static projection ---+--> sigma-conditioned fusion
                                                     |
                                                     v
                                          shared local 3D depthwise mixer
                                                     |
                                          shared pointwise feature mixer
                                                     |
                                             shared output norm
                                                     |
                         +---------------------------+---------------------------+
                         |                           |                           |
                 zero-init head block 12    zero-init head block 24    ... block N
                         |                           |                           |
                         +---------------------------+---------------------------+
                                                     |
                                             H3 hidden residual
```

The feature trunk is shared, but **every configured injection block owns an independent residual head**. This lets training learn different output semantics for early/mid/late H3 representations without duplicating the expensive static/dynamic trunk.

The local 3D mixer is linear in video-token count; the scaffold does not introduce quadratic Base-to-target cross-attention.

## Static / dynamic sigma gate

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

These are architectural hypotheses, not trained or tuned values.

## Initial injection blocks

The scaffold defaults to:

```text
12,24,36,45,48
```

This only provides early/mid/late plumbing coverage. It is not a claim that these are the optimal training layers.

## Zero-init invariant

There are two independent no-op guarantees:

1. an untrained provider is bypassed before module computation;
2. even if the module path is forced, every block-specific output head is exactly zero initialized.

Unit tests require exact tensor equality and also verify that different injection blocks own distinct residual-head modules.

## Existing block-patch composition

M6 follows the native ComfyUI MiniMax-H3 patch-composition pattern:

```text
previous patch, if any
    -> original/previous block output
    -> M6 after-block residual
```

Existing `double_block` replacements are never silently discarded.

## M4 tile alignment through global MM-RoPE

M4 assigns each HR tile target-video `position_ids` selected from the full-canvas native H3 `PackedLayout`. M6 reuses those coordinates rather than adding a second tile-coordinate API.

For a native H3 spatial axis:

```text
step = 32 * patch / sqrt(full_h * full_w)
```

M6 reads the tile's global H/W axis values, infers the full latent area, enumerates valid H/W factor pairs, and requires exactly one full-canvas geometry whose native H3 axis coordinates contain the tile axes as contiguous subsequences.

It obtains:

```text
full_h, full_w
y0, y1
x0, x1
```

and builds the static stream as:

```text
clean Base latent
  -> resize once to inferred full target latent H/W
  -> crop exact global tile rectangle
  -> native 1x2x2 patchify
  -> static adapter rows
```

Safety rules:

- missing/non-native `position_ids` fail;
- inconsistent spatial MM-RoPE step fails;
- ambiguous/no full-canvas factorization fails;
- inferred crop must match active tile geometry exactly;
- inferred region is reused by all selected adapter blocks in the active tile call;
- resized Base tensors and patch rows are cached by geometry/device/dtype.

Unit tests reconstruct a known full canvas and tile only from global MM-RoPE and require the original tile bounds exactly.

## Trained checkpoint format v1

The loader accepts safetensors only (`.safetensors` / `.sft`) from:

```text
ComfyUI/models/kirei_h3_adapters/
```

The state dict must contain exactly the current `StateAwareBaseVideoAdapter` keys, including **one residual head for every `injection_blocks` entry**. A historical/shared-head or otherwise incompatible state dict therefore fails strict loading even if its high-level config happens to match.

Floating tensors containing NaN/Inf are rejected.

Required metadata:

```text
kirei_h3_icr_api
kirei_h3_icr_kind
kirei_h3_icr_architecture_digest
kirei_h3_icr_model_id
kirei_h3_icr_config_json
```

Optional:

```text
kirei_h3_icr_training_json
kirei_h3_icr_note
```

Required values:

```text
kirei_h3_icr_api  = "1"
kirei_h3_icr_kind = "base_video_adapter"
```

`config_json` contains exactly:

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

## Architecture/checkpoint binding

The checkpoint architecture digest is compared with the descriptor derived from the **actual MODEL connected to the loader**, including the checkpoint-supplied `model_id`.

A checkpoint therefore cannot silently load on a different H3 hidden width, block count, patch contract, latent channel contract or AdaLN format.

Strict state-dict loading provides the second compatibility gate for the actual M6 module layout.

## ComfyUI memory management

A trained adapter is wrapped in a ComfyUI `CoreModelPatcher` using:

```text
load_device    = get_torch_device()
offload_device = unet_offload_device()
```

The module uses the active H3 dtype when the model exposes one; otherwise the checkpoint floating dtype is used.

When applied, the adapter patcher is registered through:

```text
MODEL.set_additional_models("h3_icr_base_video_adapter", [adapter_patcher])
```

Sampler preparation includes nested additional models, so trained adapter weights participate in normal ComfyUI load/offload decisions instead of remaining as an unmanaged GPU module.

## Nodes

**Kirei H3 ICR BaseVideo Adapter Scaffold [M6]**  
Creates the untrained API-v1 exact-parity provider.

**Kirei H3 ICR Load BaseVideo Adapter [M6]**  
Loads and validates a safetensors checkpoint from `models/kirei_h3_adapters` and returns a managed trained provider.

**Kirei H3 ICR Apply BaseVideo Adapter [M6]**  
Accepts either scaffold or trained provider. Managed providers are registered as ComfyUI additional models.

**Kirei H3 ICR BaseVideo Adapter Report**  
Reports architecture/checkpoint provenance, residual-head mode, managed state, branch/tile statistics and residual RMS.

## Training direction

Initial training should freeze H3 and optimize only M6. A detail LoRA should be deferred until the adapter itself has been characterized.

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
2. per-block residual-head contract tests;
3. safetensors metadata/strict-state architecture binding tests;
4. real ComfyUI additional-model load/offload lifecycle validation;
5. dense ~1 MP decoded-media comparison against the best training-free teacher;
6. real M4 2K validation of global-MM-RoPE Base cropping;
7. identity/object/action/timing parity;
8. faces/hands/text/detail improvement;
9. temporal stability;
10. VRAM/wall-time overhead and injection-block/gate ablation.

No M6 quality improvement is claimed until a trained checkpoint passes these gates.
