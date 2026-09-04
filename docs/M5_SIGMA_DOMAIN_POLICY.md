# M5c — Sigma-Domain Sparse Policy v3

Status: optional experimental policy layer implemented; decoded-media/CUDA validation is pending.

## Motivation

M5 v2 binds a sparse policy to one canonical packed topology per branch. That prevents a policy calibrated with one target/reference/keyframe layout from being transferred to another layout.

However, H3 attention may also change across the denoising trajectory. A head that is strongly local at high sigma does not automatically remain safe to sparsify at a later sigma.

M5c preserves the v2 topology gate and adds a second runtime gate:

```text
branch + topology digest + sigma + layer
```

No interpolation of head classes is performed outside the calibrated sigma neighborhood.

## Policy generation

The passive v2 profiler already records buckets by:

- branch;
- rounded sigma;
- layer.

The v3 report converts each bucket into a `sigma_domain` while retaining:

- the branch name;
- the branch topology digest;
- the calibrated sigma coordinate;
- the layer;
- sample count;
- sequence length;
- per-head classification;
- exact spatial/temporal QK-pair margins.

The aggregate v2 `layers` policy is kept in the JSON for analysis and compatibility, but the v3 Flex runtime does **not** use it when sigma domains are present.

## Runtime selection

For every active H3 diffusion-model call, the v3 runtime:

1. verifies the v2 branch topology gate;
2. finds v3 domains with the same branch and topology digest;
3. selects, independently per layer, the nearest calibrated sigma domain;
4. accepts it only when:

```text
abs(runtime_sigma - calibrated_sigma) <= max_policy_sigma_distance
```

5. exposes only those selected layer/head rows to the existing v2 Flex executor for that call;
6. restores the aggregate policy after the call.

Default:

```text
max_policy_sigma_distance = 0.03
```

If no matching domain exists, the active layer map is empty and the v2 executor falls back to native dense attention.

## Why no sigma interpolation yet

Interpolating categorical sparse patterns between two calibrated sigmas would introduce a new unvalidated policy. For example, a head could transition from spatial-local to mixed/global behavior between samples.

The initial v3 contract therefore chooses the nearest explicitly observed domain or stays dense. A later interpolation strategy would require its own media-parity evidence.

## BlockMask cache behavior

The existing v2 BlockMask cache key contains:

- branch;
- topology digest;
- layer;
- head count;
- sequence length;
- target geometry;
- device;
- block size;
- selected per-head policy codes.

It intentionally does not require sigma itself. Therefore two sigma domains reuse one BlockMask only when they resolve to the same effective per-head policy. If classifications differ, the policy-code tuple changes and a separate BlockMask is built.

## Telemetry

The v3 sparse report adds:

- `sigma_domain_match_calls`;
- `sigma_domain_fallback_calls`;
- `last_sigma_domain_distance`;
- `max_sigma_domain_distance`;
- configured `max_policy_sigma_distance`.

This makes it possible to distinguish a true sparse run from a run that spent most of its trajectory outside calibration and correctly stayed dense.

## Validation gate

M5c is more selective than v2 static execution, but it is not automatically safer. Before acceptance, compare:

1. v2 static topology-bound policy;
2. v3 sigma-domain policy;
3. fully dense H3;
4. sparse-call fraction and sigma-domain fallback fraction;
5. actual BlockMask sparsity;
6. first-use and steady-state CUDA time;
7. peak VRAM;
8. full decoded-video parity.

The v3 policy should only be preferred if the additional sigma selectivity improves media parity or allows a better speed/quality operating point.
