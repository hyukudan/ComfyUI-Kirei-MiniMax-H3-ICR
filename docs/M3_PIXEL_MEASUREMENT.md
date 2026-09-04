# M3d — Proxy-Decoder Pixel Measurement Consistency

Status: experimental, off by default, awaiting real H3 media validation.

## Purpose

M3d tests whether an HR predicted-clean H3 latent still reproduces the Base video in **decoded pixel space**, while avoiding a differentiable 2K VisualVAE decode.

The measurement path is:

```text
x0_HR
  -> spatial area downsample to Base latent geometry
  -> H3-compatible differentiable decoder
  -> reduced pixel measurement
  -> compare against the Base decoded through the same decoder
```

The pixel residual is differentiated back through the decoder and the latent downsample operator into `x0_HR`.

This is a pixel-space measurement experiment. It is **not claimed to be an exact implementation of diffusion posterior sampling** and it does not differentiate through the H3 denoiser.

## Why the decoder runs at Base latent geometry

A differentiable full-resolution H3 VisualVAE decode at every selected denoising call would be extremely expensive. M3d therefore first applies the known latent measurement operator:

```text
z_probe = D_latent(x0_HR)
```

and decodes `z_probe` at the Base latent geometry.

This gives the pixel-domain loss access to decoder semantics while keeping the expensive differentiable decoder branch at the smaller Base resolution.

## Decoder policy

The node requires a MiniMax H3-compatible 24-channel video `VAE`.

Recommended initial path:

- use ComfyUI's lightweight H3 TAE/`taeh3` decoder as a differentiable proxy;
- keep `allow_full_vae=false`.

The full `MiniMaxH3VideoVAE` is recognized, but gradient use is blocked by default. It can be enabled explicitly only for controlled high-cost experiments.

The reference and predicted measurement are always decoded by the **same** decoder, so the loss measures consistency inside one decoder domain rather than comparing two different codec representations.

## Reduced pixel measurement

The decoded Base reference is cached once in a reduced measurement form. Defaults:

```text
measurement_max_side: 384
frame_stride:          2
```

Spatial reduction and temporal subsampling reduce retained reference memory and loss cost. They do not reduce the cost of the proxy decode itself.

## Loss

```text
L = L_pixel
  + edge_weight * L_spatial_edge
  + temporal_weight * L_temporal_difference
```

- `L_pixel`: RGB MSE;
- `L_spatial_edge`: first-difference error in X/Y;
- `L_temporal_difference`: adjacent measured-frame difference error.

Initial weights:

```text
edge_weight:     0.25
temporal_weight: 0.10
```

## Gradient normalization and safety cap

Let `g = dL / d(x0_HR)`:

```text
correction = -strength * g * pixel_RMSE / RMS(g)
```

The correction is then bounded relative to the HR/Base latent RMS scale.

Initial settings:

```text
strength:                 0.05
apply_every:              4
max_correction_rms_ratio: 0.02
verify_after:             false
```

`verify_after=true` performs a second proxy decode after correction to measure actual pixel-RMSE change. It is off by default because it doubles decoder calls on applied hooks.

## Audio invariant

M3d changes only the predicted-clean video latent. H3 audio is copied through unchanged in NestedTensor/AV-container and packed-AV paths.

## Relationship to the other M3 constraints

### M3a — low-frequency latent fidelity
Protects broad structure and motion while leaving high frequencies free.

### M3b — normalized latent measurement backprojection
Constrains `D_latent(x0_HR)` toward the clean Base latent without autograd.

### M3c — latent posterior gradient
Uses autograd only through the latent area-downsample measurement operator.

### M3d — proxy-decoder pixel measurement
Constrains decoded appearance, spatial edges and temporal differences of `D_latent(x0_HR)` in a selected H3 decoder domain.

Do not enable all constraints by default. Their overlap must be measured in controlled ablations.

## Recommended validation matrix

Keep backend, Base latent, references, target noise, sigmas and sampler identical and compare:

1. M3a only;
2. M3a + M3b;
3. M3a + M3c;
4. M3a + M3d;
5. selected combinations only after each individual arm is characterized;
6. all measurement constraints off.

For M3d also compare:

- `taeh3` proxy versus full VisualVAE only if full-VAE backward is operational;
- `apply_every` 2 / 4 / 8;
- strength around 0.02 / 0.05 / 0.10;
- edge and temporal weights on/off;
- 384 versus 256 measurement max side.

Acceptance requires decoded-media improvement. Lower proxy-decoder loss alone is not a success criterion.

## Failure conditions

Reject a treatment if it:

- suppresses valid HR detail;
- flattens texture or faces;
- introduces temporal pumping;
- materially increases wall time without a visible fidelity gain;
- changes Base identity, objects, action, camera or timing;
- modifies audio;
- silently accepts a non-H3 or non-24-channel decoder.
