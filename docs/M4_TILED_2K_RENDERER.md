# M4 — Global-LR + Tiled-HR 2K Renderer

Status: experimental implementation on `feature/tiled-2k-fusion`.

## Goal

Run the MiniMax H3 second-pass regenerator at resolutions whose full packed video token count is too expensive for a normal dense call, without treating spatial tiles as independent videos.

The first target is 2048x1152. With a typical 124-frame H3 clip, the visual latent has about 37 temporal latent positions. The full target therefore has roughly 85,248 video tokens before text, references and audio. A 1024x768 tile has roughly 28,416 video tokens.

## Design

Each H3 model evaluation is replaced by two levels:

1. **One global low-resolution branch** at the H3 Base latent geometry.
2. **Several overlapping high-resolution spatial tiles** at the final geometry.

Text, native H3 references and audio stay global in every call. Only the target video latent and target-grid visual keyframes are spatially transformed.

The global branch produces a model-output prior. The HR tile predictions are accumulated with overlap weights and fused with that prior in model-output space at the same diffusion coordinate:

```text
argmin_y  sum_i w_i || y - tile_i ||^2 + lambda || y - prior_hr ||^2
```

For the current pointwise prior formulation, the closed-form update is:

```text
y = (sum_i w_i * tile_i + lambda * prior_hr) / (sum_i w_i + lambda)
```

This is deliberately done for every H3 model evaluation. It is not a final RGB seam blend.

## Global MM-RoPE coordinates

A naive tile call is invalid for H3 because native `PackedLayout` derives spatial MM-RoPE coordinates from the local target shape. The tile would therefore behave as if it occupied the complete canvas.

M4 builds a normal tile layout for row topology, then replaces the target-video `position_ids` with the exact rows selected from the full-canvas H3 layout. Non-spatial rows are copied from the full layout. Visual keyframe condition rows are handled in the same way.

As a result:

- a tile keeps the original full-canvas spatial coordinates;
- visual keyframe rows keep their original full-canvas coordinates;
- text and reference positions are unchanged;
- target audio retains the full-canvas position convention;
- local target-video and keyframe rows remain shape-compatible with H3 patchification.

If this mapping cannot be established exactly, the renderer fails closed.

## Tile planner

Tiles operate in H3 video-latent units and must align to the DiT spatial patch size. The ComfyUI node exposes pixel dimensions in 32-pixel steps, which maps to the public H3 VAE x16 plus DiT 2x2 patch contract.

The planner uses fixed-size tiles. When the final boundary would otherwise create a small edge tile, starts are redistributed across the axis. Actual overlap may therefore be larger than the requested minimum; it is reported in telemetry.

Overlap weights use raised-cosine ramps only on tile edges that have neighbors. This keeps the image boundary fully weighted and prevents uncovered positions.

## Global prior

M4 computes the global branch by area-downsampling the current HR noisy video state to the clean H3 Base latent geometry and evaluating H3 once at that lower resolution. The resulting video model output is bilinearly lifted to the full target geometry and used as the weighted least-squares regularizer.

This is a global dynamic prior, not yet a replay of the exact pass-1 H3 trajectory. A future mode can consume captured pass-1 trajectory states when they are available and proven beneficial.

The global branch also supplies the returned audio model output. Tile audio outputs are deliberately ignored, because every tile sees the same global audio stream and audio must not be spatially fused.

## HR keyframes

Native target-grid `minimax_keyframes` are supported when their visual latent is encoded at the full target geometry before M4 is applied.

The renderer first validates that each visual keyframe has the same full H/W latent geometry as the target. It then transforms the payload per branch:

### Global LR branch

- visual keyframe latents are area-resized to the global prior geometry;
- keyframe audio latents remain unchanged and global;
- `cond_video_latents` and `cond_audio_latents` are rebuilt so H3's packed condition rows match the transformed keyframes and original refs;
- a native low-resolution `PackedLayout` is built for the prior call.

### HR tile branches

- each visual keyframe latent is cropped to exactly the same spatial tile as the target video;
- keyframe audio remains unchanged and global;
- condition-latent lists are rebuilt for the tile payload;
- every `cond` segment receives the exact full-canvas MM-RoPE rows corresponding to that tile region and keyframe latent time span.

This creates a clean path for future sparse HR anchor experiments: detail can be injected through native H3 keyframes without making each tile interpret the keyframe as a separate local canvas.

## Spectrum interaction

The renderer installs itself as the outermost native H3 diffusion-model wrapper.

When Spectrum H3 metadata is present:

- the single global LR prior call keeps the original Spectrum runtime metadata;
- HR tile child calls remove `spectrum_h3_*` runtime fields and therefore execute as actual H3 calls;
- one forecast history is never reused across different tile topologies.

This makes the initial compatibility scope explicit: **Spectrum may accelerate the stable global prior branch, while tile calls stay actual**. This still requires real H3 validation before being treated as a supported production combination.

## Explicit M4 limits

The renderer currently fails closed for:

- EasyCache;
- non-native MiniMax H3 model implementations;
- batch sizes other than one;
- target-grid visual keyframes whose latent geometry does not exactly match the full target before tiling;
- geometry that is not aligned to the H3 patch grid;
- plans that exceed `max_tiles`.

`minimax_refs` remain global and supported because their packed row sizes do not depend on the target tile geometry.

## Initial 2K laboratory preset

```text
target:             2048 x 1152
tile:               1024 x 768
requested overlap:  256 x 256
prior geometry:     H3 Base latent geometry
prior strength:     0.30
max tiles:          16
```

For a 124-frame clip this normally means six HR tile evaluations plus one global LR evaluation per H3 model call. The exact tile count and actual overlap are reported at runtime.

## Telemetry

The renderer records:

- total wrapper calls;
- tiled versus dense-bypass calls;
- global prior calls;
- tile model calls;
- Spectrum-prior calls;
- last/max tile count;
- last visual keyframe count;
- final target latent geometry;
- full target video-token count;
- per-tile video-token count.

Use `Kirei H3 ICR Tiled 2K Report` to inspect the live renderer statistics during M4 validation.

## Validation gate

Do not merge M4 into the default path solely because it runs or reduces peak memory. The controlled media gate must compare it against the dense ~1 MP H3-ICR baseline and inspect the complete video for:

1. draft geometry and motion;
2. identity/object state;
3. cross-tile continuity;
4. seams or local exposure/color changes;
5. faces, hands and text crossing tile boundaries;
6. temporal stability;
7. detail gain;
8. HR-keyframe propagation quality and hallucination risk;
9. VRAM and wall time.

A sharper result that changes the Base draft loses.

## Research lineage

The high-level prior-regularized tiled-denoising direction is inspired by **FrescoDiffusion: 4K Image-to-Video with Prior-Regularized Tiled Diffusion** (Caselles-Dupre et al., 2026). Kirei H3-ICR does not copy its Wan-specific implementation or claim algorithmic identity. The implementation here is derived independently around public MiniMax H3 AV packing, flow coordinates and MM-RoPE behavior.
