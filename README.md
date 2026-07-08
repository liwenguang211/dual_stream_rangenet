# DS-RangeNet v3

Lightweight dual-stream LiDAR range-image semantic segmentation for industrial
indoor environments.

This repository is aligned with the corrected paper version and reviewer
response implementation. The network uses a 16-channel input, separates
material-intensity and geometry streams, and includes the reviewer-requested
controls for IGCA, modality complementarity, robustness, and DSConv baselines.

## Model

Default model:

```text
DS-RangeNet v3 full
Input:  [B, 16, H, W]
Output: [B,  9, H, W]
```

Input channels:

```text
Stream-1 material/intensity, 5 channels
  ch[0]  d/dmax
  ch[1]  local mean intensity
  ch[2]  intensity boundary strength B
  ch[3]  high-frequency saliency / intensity curvature H_curv
  ch[4]  local intensity standard deviation

Stream-2 geometry, 11 channels
  ch[5:8]    surface normal nx, ny, nz
  ch[8:11]   coordinates x, y, z
  ch[11:14]  voxel-PCA descriptors L, P, S
  ch[14]     isotropic entropy
  ch[15]     relative elevation z_rel
```

Architecture:

```text
16-channel range image
  |-- material/intensity encoder: 5 ch -> 32 -> 64 -> 128 -> 256
  `-- geometry encoder:          11 ch -> 32 -> 64 -> 128 -> 256

shallow fusion: CBAMFusion at f0, f1, f2
bottleneck:     per-stream ASPP + pooled IGCA + pairwise ICB
decoder:        U-Net style upsampling with fused skips
head:           DSConv + 1x1 Conv -> 9 classes
```

Classes:

```text
0 background
1 ground
2 roof
3 side_facade
4 front_facade
5 beam
6 column
7 window
8 dynamic
```

## Key Files

```text
python/ds_rangenet_v3.py      Main model and reviewer-response controls
python/test_ds_rangenet.py    Model-family smoke tests
python/export_onnx.py         ONNX export and verification
python/model.py               Compatibility import wrapper
python/ARCHITECTURE.md        Detailed architecture notes
include/model_config.h        C++ deployment constants
src/RangeNetInferencer.cpp    Projection, 16-channel preprocessing, ORT inference
draw_arch.py                  Architecture diagram generator
```

## Reproducibility Evidence Chain

The `reproducibility/` directory holds the re-runnable evidence requested in
review (paper note item 4): five-seed raw-result CSVs with a paired
significance test, CKA/cross-covariance extraction with its layer/frame/token
config, corruption parameters and batch scripts, cross-sensor
(SemanticPOSS/SemanticKITTI) configs and evaluation-log templates, the
Standard/Hybrid/DSConv checkpoint manifest, and a 60-minute Jetson telemetry
logger plus environment manifest. Numeric result files ship as templates with
explicit schemas and `FILL_ME` cells to be populated by the provided scripts.
See `reproducibility/README.md` for the reviewer-requirement-to-artifact map.

The UBPC-9 dataset and raw point clouds/labels are hosted on Baidu NetDisk:

```text
Share name : DataSets
Link       : https://pan.baidu.com/s/1a6ETzXrYIpeoofpxK_8rdQ
Access code: 5577
```

## Reviewer-Response Controls

The main model file includes controlled variants for the concerns raised in
review.

```python
from ds_rangenet_v3 import build_model

full = build_model("full")
cbam_only = build_model("cbam_only")
igca_only = build_model("igca_only")
no_attention = build_model("no_attention")
igca_no_icb = build_model("igca_no_icb")
igca_g2i_only = build_model("igca_g2i_only")
igca_i2g_only = build_model("igca_i2g_only")
conventional_g2i = build_model("conventional_g2i")
conventional_bidir = build_model("conventional_bidir")
standard_conv = build_model("standard_conv")
intensity_only = build_model("intensity_only")
geometry_only = build_model("geometry_only")
```

Reviewer mapping:

| Concern | Code support |
| --- | --- |
| IGCA novelty vs conventional cross-attention | `conventional_g2i`, `conventional_bidir` |
| Pairwise intensity-curvature bias contribution | `igca_no_icb` |
| Directional contribution of IGCA branches | `igca_g2i_only`, `igca_i2g_only` |
| Modality complementarity evidence | `linear_cka`, `normalized_cross_covariance`, `complementarity_report` |
| Robustness under corruptions | `apply_corruption` |
| DSConv accuracy-efficiency trade-off | `standard_conv` |
| Single-stream controls | `intensity_only`, `geometry_only` |

## Python Usage

Run the smoke tests:

```bash
cd dual_stream_rangenet
python python/test_ds_rangenet.py
```

Create the default model:

```python
import torch
from python.ds_rangenet_v3 import DualStreamRangeNetV3, IN_TOTAL

model = DualStreamRangeNetV3().eval()
x = torch.randn(1, IN_TOTAL, 64, 512)
with torch.no_grad():
    logits = model(x)
print(logits.shape)  # [1, 9, 64, 512]
```

Return intermediate features for complementarity analysis:

```python
from python.ds_rangenet_v3 import build_model, complementarity_report

model = build_model("full").eval()
report = complementarity_report(model, x)
print(report)
```

Apply robustness corruptions:

```python
from python.ds_rangenet_v3 import apply_corruption

x_noisy = apply_corruption(x, "range_noise")
x_drop = apply_corruption(x, "point_dropout")
x_block = apply_corruption(x, "block_occlusion")
```

Supported corruptions:

```text
range_noise
intensity_noise
point_dropout
scanline_dropout
block_occlusion
```

## ONNX Export

Export the full model:

```bash
python python/export_onnx.py \
  --weights weights/best.pth \
  --output models/dual_stream_rangenet_v3.onnx
```

Export a reviewer-control variant:

```bash
python python/export_onnx.py \
  --variant conventional_bidir \
  --output models/dual_stream_rangenet_v3_conventional_bidir.onnx
```

Verify an existing ONNX model:

```bash
python python/export_onnx.py \
  --verify models/dual_stream_rangenet_v3.onnx
```

The exported ONNX input is fixed shape:

```text
range_image: [1, 16, H, W]
logits:      [1,  9, H, W]
```

## C++ Inference

The C++ runtime builds the same 16-channel input defined by the paper:

```text
Point cloud [x, y, z, intensity]
  -> spherical projection
  -> material-intensity statistics
  -> normals and voxel-PCA geometry descriptors
  -> RangeImage [1, 16, H, W]
  -> ONNX Runtime inference
  -> pixel labels
  -> back projection and KNN fill
```

Build:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Run:

```bash
./build/bin/rangenet_inference \
  --model models/dual_stream_rangenet_v3.onnx \
  --input data/test.bin \
  --output output
```

Benchmark:

```bash
./build/bin/rangenet_inference \
  --model models/dual_stream_rangenet_v3.onnx \
  --benchmark
```

## Sensor Projection Notes

Projection keeps the nearest return when multiple points fall into one pixel.
The primary pipeline does not use input-stage interpolation or hole filling.
Post-prediction KNN fill is applied after pixel labels are back-projected to
the original point cloud.

The C++ preprocessor computes:

```text
material-intensity:
  normalized range, local mean intensity, boundary strength,
  intensity curvature, local intensity standard deviation

geometry:
  image-space normals, xyz coordinates,
  0.2 m voxel-PCA L/P/S descriptors, eigen entropy, relative elevation
```

## Regenerate Architecture Figure

```bash
python draw_arch.py
```

This writes:

```text
arch_diagram.png
arch_diagram_v2.png
```

## Important Notes

- Old 8-channel model files were removed.
- Legacy Lite ONNX names should not be used.
- New weights must be trained or exported for the 16-channel model.
- The repository currently expects the final deployment model at:

```text
models/dual_stream_rangenet_v3.onnx
```
