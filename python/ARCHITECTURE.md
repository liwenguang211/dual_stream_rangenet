# DS-RangeNet v3 Architecture

This document describes the implementation in `python/ds_rangenet_v3.py`.
It is aligned with the corrected paper version that uses a 16-channel
range-image input.

## Input And Output

Input tensor:

```text
x: [B, 16, H, W]
```

Channel layout:

```text
Stream-1 material/intensity, 5 channels
  ch[0]  d/dmax
  ch[1]  local mean intensity
  ch[2]  boundary strength B
  ch[3]  high-frequency saliency / intensity curvature H_curv
  ch[4]  local intensity standard deviation

Stream-2 geometry, 11 channels
  ch[5:8]    surface normal nx, ny, nz
  ch[8:11]   coordinates x, y, z
  ch[11:14]  voxel-PCA descriptors L, P, S
  ch[14]     isotropic entropy
  ch[15]     relative elevation z_rel
```

Output tensor:

```text
logits: [B, 9, H, W]
```

## Network Flow

```text
Input [B,16,H,W]
  ├─ material/intensity stream [B,5,H,W]
  └─ geometry stream           [B,11,H,W]

Each stream encoder:
  s0 stride 1  -> [B,  32, H,   W]
  s1 stride 2  -> [B,  64, H/2, W/2]
  s2 stride 2  -> [B, 128, H/4, W/4]
  s3 stride 2  -> [B, 256, H/8, W/8]

Shallow fusion:
  f0 = CBAMFusion(e0, g0)
  f1 = CBAMFusion(e1, g1)
  f2 = CBAMFusion(e2, g2)

Bottleneck:
  e3_bar = ASPP_intensity(e3)
  g3_bar = ASPP_geometry(g3)
  f3 = IGCA(e3_bar, g3_bar, curvature=ch[3])

Decoder:
  f3 -> up with f2 -> up with f1 -> up with f0 -> segmentation head
```

## IGCA

IGCA follows the corrected paper definition:

- Affinity is estimated within the guiding modality.
- Values are transported from the complementary stream.
- A pairwise intensity-curvature bias is added before row-wise softmax:

```text
B_ICB[i,j] = -gamma * |c_i - c_j|
```

The attention is applied after 2x spatial pooling. The transported residuals
are projected back to 256 channels, upsampled to the bottleneck resolution,
and gated by learnable `alpha` and `beta`, both initialized to zero.

## Key Classes

| Class | Role |
| --- | --- |
| `DSRangeNetConfig` | Reproducible configuration for model controls |
| `StreamEncoder` | Four-scale per-stream encoder |
| `ASPP` | Per-stream atrous bottleneck |
| `CBAMFusion` | Shallow multi-scale fusion |
| `IGCrossAttention` | Pooled IGCA with pairwise ICB |
| `ConventionalCrossAttention` | Reviewer-control baseline for generic cross-attention |
| `DualStreamRangeNetV3` | Complete 16-channel segmentation network |
| `build_model` | Factory for ablation and reviewer-response variants |
| `CombinedLoss` | Focal + Dice loss |

## Standard Shape Example

For `[1, 16, 64, 512]`:

```text
input              [1,  16, 64, 512]
stream split       [1,   5, 64, 512] and [1, 11, 64, 512]
s0                 [1,  32, 64, 512]
s1                 [1,  64, 32, 256]
s2                 [1, 128, 16, 128]
s3                 [1, 256,  8,  64]
IGCA pooled tokens [1, 512, 64]
logits             [1,   9, 64, 512]
```

## Reviewer-Response Controls

The model file includes controlled variants that map directly to the main
review concerns.

```python
from ds_rangenet_v3 import build_model

full = build_model("full")
cbam_only = build_model("cbam_only")
no_attention = build_model("no_attention")
igca_no_icb = build_model("igca_no_icb")
igca_g2i_only = build_model("igca_g2i_only")
igca_i2g_only = build_model("igca_i2g_only")
conventional = build_model("conventional_bidir")
standard_conv = build_model("standard_conv")
intensity_only = build_model("intensity_only")
geometry_only = build_model("geometry_only")
```

Supported analysis helpers:

| Helper | Reviewer concern addressed |
| --- | --- |
| `linear_cka` | Quantifies representation similarity |
| `normalized_cross_covariance` | Quantifies cross-modal dependence |
| `complementarity_report` | Reports input/encoder/fusion complementarity |
| `apply_corruption` | Range noise, intensity noise, dropout, scan-line dropout, block occlusion |
| `build_model("standard_conv")` | DSConv accuracy-efficiency control |
| `build_model("conventional_bidir")` | Conventional cross-attention control |
| `build_model("igca_no_icb")` | Isolates intensity-curvature bias contribution |
