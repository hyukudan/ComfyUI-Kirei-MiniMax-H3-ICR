# M3b — Latent Measurement Consistency

Status: optional experimental implementation; real H3 decoded-media validation is pending.

## Goal

The existing H3-ICR per-step fidelity projector preserves the Base draft mainly through low spatial frequencies. M3b adds a complementary measurement constraint:

```text
D(x0_HR) ~= z_Base
```

where:

- `x0_HR` is the current predicted-clean high-resolution H3 video latent;
- `z_Base` is the clean pass-1 H3 Base video latent;
- `D` is spatial area downsampling back to the Base latent grid.

The constraint does not try to make the HR latent equal to an upscaled Base latent. High-resolution degrees of freedom remain available as long as they are compatible with the observed Base-grid measurement.

## Normalized backprojection

For each enabled post-CFG evaluation:

```text
r_lr = z_Base - D(x0_HR)
r_robust = robust(r_lr)
r_band = LPF(r_robust) + high_band_mix * (r_robust - LPF(r_robust))
u_hr = U(r_band)
response_lr = D(u_hr)
g = <r_band, response_lr> / ||response_lr||^2
x0_HR <- x0_HR + bounded(strength * g * u_hr)
```

`U` is bicubic spatial lifting. The scalar gain `g` compensates for the measured response of the actual `D(U(.))` pair rather than assuming the lift is an exact adjoint or inverse.

The gain is clamped by `max_backprojection_gain` and the final HR correction is independently bounded by an RMS ratio relative to the current HR estimate.

## Band control

`high_band_mix` distinguishes M3b from the existing low-frequency projector:

- `0.0` — only the low-frequency residual is backprojected;
- `1.0` — the complete Base-grid residual is backprojected;
- intermediate values constrain structure strongly while allowing some Base-grid high-frequency mismatch.

Initial default:

```text
cutoff:         0.35
high_band_mix: 0.25
```

## Robust residual

A very large local residual can represent an outlier, VAE mismatch, occlusion-sensitive region or another unstable constraint. When `robust_delta > 0`, residual elements larger than `robust_delta * residual_RMS` are smoothly downweighted before backprojection.

Initial default:

```text
robust_delta: 3.0
```

Set it to `0` to disable robust weighting.

## Iterations

The projection can be repeated inside one post-CFG call. Each iteration re-measures `D(x0_HR)` after the previous correction.

Initial default: `1`.

More iterations are not assumed to be better. They increase constraint strength and cost, and must be validated against detail loss.

## Sigma schedule

M3b uses the same structure-first schedule family as the existing fidelity projector:

```text
w(sigma) = floor + (1 - floor) * (sigma / sigma_start)^power
```

and applies:

```text
effective_strength = strength * w(sigma)
```

This allows strong Base-measurement consistency near the start of pass 2 while reducing interference with late detail synthesis.

Initial defaults:

```text
strength:       0.15
schedule_floor: 0.00
schedule_power: 1.00
```

## Ordering

When both constraints are enabled in `Kirei H3 ICR Regenerate`, the intended order is:

```text
predicted clean x0
    -> existing low-frequency fidelity projector
    -> M3b measurement-consistency projector
    -> sampler
```

M3b is therefore the final structural correction and directly reports whether the Base-grid measurement improved.

## Audio invariant

M3b only receives the H3 video latent. For NestedTensor and packed AV paths, the audio member is returned unchanged.

## Telemetry

The integrated H3-ICR report records:

- calls / applied calls;
- mean Base-grid error before projection;
- mean Base-grid error after projection;
- mean error reduction;
- maximum correction RMS ratio;
- mean normalized backprojection gain;
- last sigma schedule value.

The pure projector also reports robust-outlier fraction, clamp scale and iteration count.

## Initial experimental node

Use:

**Kirei H3 ICR Measurement Consistency [Experimental]**

and connect its output to the optional `measurement_consistency` input on **Kirei H3 ICR Regenerate**.

Initial configuration:

```text
strength:                  0.15
cutoff:                    0.35
high_band_mix:             0.25
max_correction_rms_ratio: 0.15
robust_delta:              3.0
max_backprojection_gain:   2.0
iterations:                1
schedule_power:            1.0
schedule_floor:            0.0
```

## Validation gate

Compare at minimum:

1. existing low-frequency fidelity only;
2. M3b only;
3. low-frequency fidelity + M3b;
4. multiple `high_band_mix` values;
5. measurement strength around the initial 0.15 setting;
6. dense ~1 MP and M4 2K paths.

A lower latent measurement error is not sufficient. The complete decoded video must still improve or preserve identity, object state, motion, temporal stability and fine detail.
