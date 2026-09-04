# M3c — Latent Posterior Gradient Consistency

Status: experimental, off by default, awaiting real H3 media validation.

## Motivation

The low-frequency fidelity projector and M3b normalized measurement backprojection already protect the Base draft in latent space. M3c asks a narrower inverse-problem question:

> Does the complete HR clean estimate still reproduce the observed H3 Base latent through the known LR measurement operator, and does an explicit measurement gradient help beyond direct backprojection?

The measurement is:

```text
D(x0_HR) ~= z_Base
```

where `D` is spatial H3-latent area downsampling and `z_Base` is the clean pass-1 H3 video latent.

## Current implementation

M3c computes a DPS-style latent measurement correction without differentiating through H3 or the VAE:

```text
probe = detach(x0_HR)
measurement = area_downsample(probe)
residual = measurement - z_Base
loss = 0.5 * ||residual||^2
g = d(loss) / d(probe)
```

Autograd exists only through the spatial measurement operator.

The update is normalized by residual and gradient RMS:

```text
g_norm = g * RMS(residual) / RMS(g)
correction = -strength * g_norm
```

The correction is capped relative to the larger RMS scale of the HR estimate and Base observation. No H3 parameters, H3 activations or VAE decoder are part of this gradient.

## Relationship to M3a/M3b/M3d

### M3a — low-frequency latent fidelity
Protects global layout, motion and identity structure while deliberately leaving high frequencies free.

### M3b — normalized latent measurement backprojection
Uses the Base-grid residual directly, with measured backprojection normalization, robust weighting and optional internal iterations.

### M3c — latent posterior gradient
Uses autograd only through `D_latent`; it is a separate control for whether an explicit inverse-problem gradient helps beyond M3b's direct residual lift.

### M3d — proxy-decoder pixel measurement
Adds decoder semantics by differentiating pixel/edge/temporal loss through an H3-compatible decoder at Base latent geometry. See `M3_PIXEL_MEASUREMENT.md`.

These mechanisms must be ablated independently before combining them.

## Initial laboratory settings

```text
strength:                 0.10
apply_every:              2
max_correction_rms_ratio: 0.05
```

These are conservative hypotheses, not tuned defaults.

## Audio invariant

The patch only changes the video member of the H3 AV clean state. Audio is copied through exactly and is never included in the measurement gradient.

## Representation support

The hook supports:

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

A lower latent measurement error is not sufficient for acceptance. Decoded-video detail, temporal behavior and Base fidelity remain the gate.

## Recommended A/B gate

Keep Base latent, conditioning, backend, noise, sampler and sigmas identical and compare:

1. M3a only;
2. M3a + M3b;
3. M3a + M3c;
4. M3c only;
5. all measurement constraints off.

Inspect:

- `D(x0_HR)` latent error;
- identity/object/motion fidelity;
- faces, hands, products and text;
- temporal flicker;
- wall-time overhead.

If measurement error improves but decoded video becomes flatter, less detailed or less stable, M3c loses.

## Pixel-space follow-up

The separate M3d experiment is now implemented. It avoids differentiable 2K decoding by applying `D_latent` first and then differentiating a reduced pixel measurement through an H3-compatible decoder at Base latent geometry. The lightweight `taeh3` decoder is the recommended first proxy; full VisualVAE gradients remain explicit opt-in.
