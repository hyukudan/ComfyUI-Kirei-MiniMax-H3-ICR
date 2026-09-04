# M3B — Latent Posterior / Measurement Consistency

Status: experimental, off by default, awaiting real H3 media validation.

## Motivation

The existing per-step fidelity projector intentionally corrects only **low spatial frequencies** of the H3 predicted-clean HR video toward the Base draft. That protects structure while leaving higher-frequency degrees of freedom available to H3.

Posterior consistency answers a different question:

> Does the complete HR clean estimate still reproduce the observed H3 Base latent when passed through the known LR measurement operator?

The initial measurement is therefore:

```text
D(x0_HR) ~= z_Base
```

where `D` is spatial H3-latent area downsampling and `z_Base` is the clean pass-1 H3 video latent.

## Current implementation

The experimental patch computes one DPS-style correction without differentiating through H3 or the VAE:

```text
probe = detach(x0_HR)
measurement = area_downsample(probe)
residual = measurement - z_Base
loss = 0.5 * ||residual||^2
g = d(loss) / d(probe)
```

Autograd exists only through the spatial measurement operator.

The raw gradient magnitude depends on the resize geometry, so the update is normalized by residual and gradient RMS:

```text
g_norm = g * RMS(residual) / RMS(g)
correction = -strength * g_norm
```

The correction is then capped relative to the larger RMS scale of the HR estimate and Base observation.

No H3 model parameters, H3 activations or VAE decoder are part of this gradient.

## Why it is separate from low-frequency fidelity

The two mechanisms have intentionally different roles:

### Low-frequency per-step fidelity

- filters the Base residual spatially;
- preserves global layout / motion / identity structure;
- is strongest early and can relax toward sigma 0;
- deliberately does **not** force the entire LR measurement residual to zero.

### Posterior consistency

- uses the full latent measurement residual;
- tests the explicit inverse-problem constraint `D(x0_HR) -> z_Base`;
- can catch drift that survives the low-frequency projector;
- costs an additional autograd pass through spatial downsampling;
- is therefore applied only every N post-CFG calls by default.

The two can be tested independently or chained. Posterior consistency is not enabled automatically by `H3 ICR Regenerate`.

## Initial laboratory settings

```text
strength:                 0.10
apply_every:              2
max_correction_rms_ratio: 0.05
```

These are conservative hypotheses, not tuned defaults.

## Audio invariant

The patch only changes the video member of the H3 AV clean state. Audio is copied through exactly and is never included in the spatial measurement gradient.

## Representation support

The hook supports both H3 clean-state forms used around ComfyUI sampler callbacks:

- NestedTensor / `(video, audio)` AV containers;
- packed AV tensors when `latent_shapes` are available from the active H3 model.

Unknown packed layouts fail rather than silently disabling the constraint.

## Telemetry

The report records:

- post-CFG calls;
- applied corrections;
- mean measurement error before/after;
- mean/max correction RMS ratio;
- mean measurement-gradient RMS.

A valid configuration should reduce measurement error on applied calls. That alone is not sufficient for acceptance: decoded-video detail and temporal behavior remain the quality gate.

## Recommended A/B gate

Use the same Base latent, conditioning, backend, noise, sampler and sigmas and compare:

1. low-frequency fidelity only;
2. posterior consistency only;
3. both;
4. neither.

For each arm inspect:

- `D(x0_HR)` latent error;
- draft identity/object/motion fidelity;
- detail retained at faces, hands, products and text;
- temporal flicker;
- wall-time overhead.

If measurement error improves but the decoded video becomes flatter, less detailed or less temporally stable, the posterior treatment loses.

## Future pixel-space DPS experiment

A later, more expensive experiment may decode selected clean states, apply a known pixel-space degradation operator and differentiate the measurement residual back into the H3 latent through the VAE decoder. That path is **not implemented** in the current MVP and should only be considered if latent-space consistency gives a clear benefit.
