# Validation Harness — Canonical H3-ICR Experiment Manifests

Status: implementation in progress on the experimental branch. The harness exists to make later decoded-media validation reproducible and to prevent accidental multi-variable comparisons.

## Why this exists

Kirei H3-ICR now contains several independent research axes:

- FL2VA / Hybrid / Ref2VA backend;
- M3 fidelity/measurement constraints;
- M4 tiled 2K rendering and keyframes;
- M5 profiling/sparse execution;
- M6 adapter state.

A visual A/B is meaningless if two arms accidentally use different Base latents, reference ordering, sigma tensors, sampler options, seeds or initializer checkpoints.

The validation harness therefore separates every run into:

```text
LOCKS       values that must not change inside a strict A/B group
ARM         the explicitly declared treatment being varied
RUN ID      SHA-256 of the complete canonical manifest
```

## Node: Validation Manifest

**Kirei H3 ICR Validation Manifest** accepts the actual workflow objects plus two small JSON objects supplied by the experiment author.

Automatically fingerprinted locks:

- clean H3 Base AV latent contents;
- positive conditioning contents;
- negative conditioning contents;
- exact sigma tensor;
- NOISE class/state/seed;
- SAMPLER class/function/options;
- `locked_settings_json`.

Automatically captured arm state where connected/installed:

- backend descriptor;
- M3b measurement handle;
- M3c posterior handle;
- M3d pixel-measurement handle;
- M4 renderer/prior-schedule handle;
- M5 sparse runtime handle;
- M6 adapter handle;
- known Kirei MODEL research patches.

`arm_settings_json` stores experiment-specific treatment values that are not represented by a native handle.

The output is:

```text
H3_ICR_VALIDATION_MANIFEST
manifest_json
run_id
```

## Canonical hashing

### Tensors

Tensor hashes are independent of CUDA/CPU placement. The fingerprint contains:

```text
dtype
shape
numel
SHA-256 of contiguous CPU tensor bytes
```

Nested tensors are hashed recursively.

Non-finite floating tensors fail validation.

### Conditioning

Conditioning is traversed recursively. Supported canonical values include:

- tensors;
- dictionaries with scalar keys;
- lists/tuples;
- finite scalars and strings;
- dataclasses;
- objects exposing a deterministic `to_dict()`;
- stateless named callables where supported by the sampler descriptor.

Strict mode intentionally rejects unsupported opaque objects instead of hashing `repr()`, because Python object representations can contain process-specific memory addresses.

### NOISE

The descriptor records the concrete class, `seed` when present, and reproducible object state.

For the normal ComfyUI `RandomNoise` path this captures the exact integer seed.

### SAMPLER

The descriptor records:

- concrete sampler class;
- sampler-function module/qualified name;
- `extra_options`;
- `inpaint_options`;
- other canonical sampler state where present.

A sampler callback with untracked closure state is not acceptable in strict mode.

## User-declared locked settings

The harness cannot infer every external artifact from a ComfyUI graph. `locked_settings_json` is therefore required for values such as:

```json
{
  "target": [1344, 768],
  "initializer": "learned_3d",
  "latent_upscaler_sha256": "...",
  "reference_order_contract": "base-video, original-images, original-videos, original-audio",
  "comfyui_commit": "...",
  "h3_checkpoint_sha256": "..."
}
```

Inside a strict comparison group this object must remain identical.

Do not put a treatment value here if the experiment is specifically changing it.

## Arm settings

`arm_settings_json` contains the treatment not represented elsewhere, for example:

```json
{
  "m3b": {
    "strength": 0.15
  }
}
```

or:

```json
{
  "m4": {
    "keyframes": "verified_hr_on"
  }
}
```

The comparator will still reject these differences unless their exact path is explicitly allowed.

## Node: Compare Validation Manifests

**Kirei H3 ICR Compare Validation Manifests** validates both run IDs before comparison.

The arm label is allowed to differ automatically. Every other difference must match one of the user-supplied path prefixes.

Examples:

### Backend-only A/B

```text
allowed_differences:
arm.backend
```

If sigma, Base latent, conditioning, noise or sampler changes, the comparison fails.

### M3b strength A/B

```text
allowed_differences:
arm.settings.m3b.strength
```

### M3a versus M3d

```text
allowed_differences:
arm.features.pixel_measurement_m3d
arm.settings.m3
```

### M5 policy comparison

```text
allowed_differences:
arm.features.sparse_m5
arm.settings.m5
```

Paths are dot prefixes. `arm.settings.m5` permits children of that object but does not unlock any `locks.*` value.

With `fail_on_unexpected_difference=true`, the graph raises before the invalid A/B result can be treated as evidence.

## Provenance warnings

The manifest can report incomplete backend provenance, for example a missing checkpoint SHA-256 or Hybrid overlay SHA-256.

Warnings do not themselves alter the A/B compatibility result; the underlying backend descriptor still does.

For publication-quality runs, resolve all provenance warnings.

## Strict A/B versus media baseline

Not every useful comparison is a strict A/B.

### Strict A/B

Use strict manifests when the target geometry and all locked inputs can be identical.

Examples:

- FL2VA vs Hybrid vs Ref2VA at the same target geometry;
- M3a vs M3b vs M3c vs M3d;
- constant versus sigma-scheduled M4 prior at the same 2K target;
- M5 profiler off/on no-op test;
- M5 dense vs sparse at the same topology;
- M6 adapter off vs zero-init scaffold.

### Media baseline

Dense ~1 MP versus M4 2048×1152 intentionally changes target geometry/token topology. That comparison is valuable, but it is not a strict A/B and should not be described as one.

Use separate comparison groups and report the shared Base/source evidence plus the intentional geometry difference explicitly.

## Staged validation matrix

The repository contains:

```text
configs/validation_matrix_v1.json
```

The planned order is:

1. **G1 backend strict A/B** — FL2VA / Hybrid 45-49 / Ref2VA;
2. **G2 M3 strict A/B** — M3a, M3a+M3b, M3a+M3c, M3a+M3d;
3. **G3 M4 2K internal strict A/B** — constant/scheduled prior and verified HR keyframes;
4. **G4 resolution media baseline** — best dense ~1 MP versus best M4 2K, explicitly non-strict;
5. **G5 profiler no-op** — native attention versus passive M5a;
6. **G6 sparse** — native dense versus M5b/M5c;
7. **G7 M6 zero-init** — adapter off versus scaffold exact-parity gate;
8. **G8 trained M6** — only after G1–G7 characterization and a real trained checkpoint.

This staged sequence avoids a huge factorial search and makes causal interpretation possible.

## Decoded-media gate

Manifest equality does not decide quality. It only proves that the comparison is controlled.

Decoded-media ranking remains:

1. Base geometry and motion fidelity;
2. identity and object correctness;
3. temporal consistency/disocclusion;
4. faces, hands, text and product detail;
5. perceptual detail/sharpness;
6. wall time and VRAM.

A sharper result that changes the Base draft loses.

## Performance reporting

For M4/M5/M6 performance tests, record at minimum:

- run ID;
- first-use wall time;
- steady-state wall time after compilation/cache warmup;
- peak allocated/reserved VRAM;
- target token count;
- M4 tile count;
- M5 actual BlockMask sparsity and sparse-call fraction;
- dense fallback reason counts;
- M6 adapter checkpoint SHA-256 and residual RMS telemetry.

Do not combine compilation/mask-build first-use cost with steady-state speed without reporting both.

## Run-ID integrity

`run_id` is SHA-256 over the complete manifest before the run-id field is inserted.

The comparator recomputes this digest. Editing any manifest field manually without recomputing the run ID causes an integrity failure before A/B comparison.
