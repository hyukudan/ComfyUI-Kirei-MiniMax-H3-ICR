# Validation Metrics and Result Bundles

Status: deterministic latent-space triage implemented on the experimental branch. These metrics support controlled experiment ranking; they do **not** replace decoded-media inspection.

## Purpose

The canonical validation manifest proves that two runs are comparable. The latent metrics answer a different question:

> Given one completed H3-ICR output, how compatible is it with the Base draft, how much new HR detail exists, how stable is that detail over time, did audio remain invariant, and are there obvious M4 tile-boundary discontinuities?

The output metrics are deterministic and require no external perceptual model.

## Node: Latent Validation Metrics

Use **Kirei H3 ICR Latent Validation Metrics** after the second-pass output.

Inputs:

```text
output_latent
base_latent
lowpass_cutoff = 0.25
optional renderer handle
```

Outputs:

```text
H3_ICR_VALIDATION_METRICS
metrics_json
```

The metric payload receives its own `metrics_id` SHA-256.

## Base compatibility

The output video latent is spatially area-downsampled to the Base latent grid:

```text
z_measure = D_latent(z_output_HR)
residual  = z_measure - z_Base
```

Reported values:

### `measurement_rmse`

Full Base-grid latent RMSE.

Lower means the regenerated HR result reproduces the Base observation more closely after spatial measurement.

It must **not** be minimized blindly: a model can lower this error by suppressing useful HR detail.

### `low_frequency_rmse`

The Base-grid residual after the same spatial Fourier low-pass family used by the fidelity path.

This is a broad structure/layout diagnostic.

### `temporal_delta_rmse`

Compares frame-to-frame latent changes after the output is measured back onto the Base grid:

```text
Delta D(z_output_HR) - Delta z_Base
```

This helps detect motion/timing drift even when per-frame Base error looks acceptable.

## HR detail metrics

The Base-compatible baseline at output resolution is constructed as:

```text
z_baseline_HR = U_bicubic(D_area(z_output_HR))
detail_HR     = z_output_HR - z_baseline_HR
```

This is not a claim that the baseline is the true HR signal. It is a deterministic decomposition of information not represented by a simple Base-grid measurement.

Reported values:

### `hr_residual_rms`

Amount of HR latent information beyond the measured Base-grid baseline.

### `hr_residual_spatial_gradient_rms`

Spatial variation inside that HR residual. It is useful for distinguishing a flat residual from genuinely high-frequency structure.

### `hr_residual_temporal_delta_rms`

Temporal change of the HR residual. Very high values can indicate flickering detail, but motion/disocclusion can also raise it.

### `output_spatial_gradient_rms`

Overall output latent spatial-gradient energy, recorded as context.

No single detail metric is an optimization target by itself.

## Audio invariant

The metrics report:

```text
shape_equal
exact
rmse
max_abs
```

For the normal H3-ICR audio-lock path, `exact=true` is the expected result.

If output/Base audio shapes differ, `rmse` and `max_abs` are `null`; shape mismatch already constitutes a failed audio invariant and avoids non-standard JSON infinities.

A visually improved arm that unexpectedly changes locked pass-1 audio fails the experiment.

## M4 seam diagnostic

When an M4 renderer handle is connected, the metric node reconstructs the same spatial tile plan from the output latent geometry and renderer configuration.

For every internal tile-start boundary it measures the latent neighbor discontinuity and normalizes it by the ordinary global neighboring-gradient RMS:

```text
boundary_x_ratio = RMS(boundary x jump) / RMS(all x-neighbor jumps)
boundary_y_ratio = RMS(boundary y jump) / RMS(all y-neighbor jumps)
```

The report also records tile count and boundary counts.

This is a **diagnostic**, not a proof of a seam. A real object edge can coincide with a tile boundary. Persistent ratios elevated across clips/seeds are much more suspicious than one isolated value.

## Output fingerprint

The complete output AV latent is hashed using the same canonical tensor hashing used by the validation manifest.

This gives every metrics object an exact output identity in addition to the `metrics_id`.

## Node: Validation Result Bundle

Use **Kirei H3 ICR Validation Result Bundle** to bind:

```text
canonical manifest
+ latent metrics
+ optional runtime reports JSON
+ optional notes JSON
```

Outputs:

```text
H3_ICR_VALIDATION_BUNDLE
bundle_json
bundle_id
```

The bundle preserves:

```text
run_id
metrics_id
```

and adds its own `bundle_id` SHA-256 over the complete result record.

This is the preferred unit for storing one experimental arm result.

## Node: Compare Validation Bundles

Use **Kirei H3 ICR Compare Validation Bundles** only after each arm has been wrapped in a result bundle.

The comparator first applies the same strict manifest rules used by **Compare Validation Manifests**. Metric deltas are meaningful only when that comparison is controlled.

Inputs:

```text
bundle_a
bundle_b
allowed_differences
fail_on_unexpected_difference = true
```

Examples:

```text
backend comparison:
arm.backend

M3 strength comparison:
arm.settings.m3.strength
```

If Base latent, conditioning, sigma tensor, NOISE, SAMPLER or any other locked field changes unexpectedly, `comparable=false`; with fail mode enabled the node raises before the metric deltas can be treated as evidence.

For shared scalar metrics the report computes:

```text
delta_b_minus_a = metric_B - metric_A
relative_delta  = delta / abs(metric_A)   # only when A is non-zero
```

Direction hints are intentionally conservative:

- Base-compatibility errors: lower means closer to the Base measurement;
- locked-audio RMSE/max error: zero is expected;
- M4 seam ratios: lower is less suspicious, but still not a quality score;
- detail metrics: diagnostic only; there is no monotonic "more is better" rule.

Boolean flags such as `audio.exact` are reported separately as flag changes.

The comparison always returns:

```text
winner: null
```

Kirei H3-ICR deliberately does not turn heterogeneous latent diagnostics into an automatic global quality score.

## Recommended reports to attach

Depending on the arm, `reports_json` should include the relevant runtime telemetry:

- Base H3-ICR regeneration report;
- M3b measurement statistics;
- M3c posterior statistics;
- M3d pixel-measurement statistics;
- M4 tile/prior schedule report;
- M5 profiler or Flex sparse report;
- M6 adapter report;
- wall time and VRAM measurements collected externally.

Keep large binary outputs outside the JSON bundle and refer to them by a stable filename/hash.

## Interpretation rule

The automatic metrics are triage signals. Final ranking remains:

1. Base geometry and motion fidelity;
2. identity and object correctness;
3. temporal consistency/disocclusion;
4. faces, hands, text and product detail;
5. perceptual detail/sharpness;
6. wall time and memory.

A sharper output that changes the Base draft loses even if `hr_residual_rms` is higher.

## First real validation sequence

With manifests, latent metrics and bundle comparison implemented, the next runtime work should be:

1. G1: FL2VA / Hybrid 45-49 / Ref2VA at identical Base/conditioning/noise/sigmas;
2. select the best fidelity backend from decoded media plus controlled bundle deltas;
3. G2: M3a / M3a+M3b / M3a+M3c / M3a+M3d on that backend;
4. only then move the best teacher candidate to M4 2K and later M5/M6.

Do not train M6 before the teacher arm is selected from decoded media plus controlled metrics.
