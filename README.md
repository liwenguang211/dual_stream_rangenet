# DS-RangeNet v3

Lightweight dual-stream LiDAR range-image semantic segmentation for industrial
indoor environments.

This repository implements the network described in Chapter 3 of the
doctoral dissertation:
*Lightweight Multi-Dimensional Semantic Extraction via
Geometry–Intensity-Curvature Deep Coupling*.

The network uses a 16-channel input, separates geometry and
intensity-curvature streams, and includes reviewer-requested controls for
IGCA, modality complementarity, robustness, and DSConv baselines.

## Model

Default model:

```text
DS-RangeNet v3 full
Input:  [B, 16, H, W]
Output: [B,  9, H, W]
```

### Input channels

Input channels follow Table `tab:stream_channels` in the dissertation.
The 16 channels are split into two physically motivated streams:

```text
Stream-1 material/intensity, 5 channels
  ch[0]  d/dmax                                    normalized range
  ch[1]  \widehat C_{I,k}                          local intensity curvature (MLS fitted)
  ch[2]  B_{C,k}                                   curvature gradient boundary strength
  ch[3]  H_{C,k}                                   curvature Laplacian high-frequency saliency
  ch[4]  \Sigma_{C,k}                              intensity-curvature covariance

Stream-2 geometry, 11 channels
  ch[5:7]    n_x, n_y, n_z                         surface normal (voxel PCA)
  ch[8:10]   x, y, z                               normalized spatial coordinates
  ch[11]     L_\lambda                             linearity
  ch[12]     P_\lambda                             planarity
  ch[13]     S_\lambda                             scattering
  ch[14]     H_\lambda                             isotropic entropy
  ch[15]     z_k^{\mathrm{rel}}                    relative elevation
```

### Architecture

Architecture follows Fig. `fig:overall_arch` in the dissertation:

```text
16-channel range image
  |-- material/intensity-curvature encoder: 5 ch  -> 32 -> 64 -> 128 -> 256
  `-- geometry encoder:                  11 ch -> 32 -> 64 -> 128 -> 256

shallow fusion: CBAMFusion at scales s0, s1, s2 (Eq. eq:cbam_fusion)
bottleneck:     per-stream ASPP (Eq. eq:aspp) + pooled IGCA (Eq. eq:igca_out)
                + pairwise ICB (Eq. eq:icb_pairwise)
decoder:        U-Net style upsampling with fused skips (Eq. eq:upblock)
head:           DSConv + 1x1 Conv -> 9 classes (Eq. eq:softmax_prob)
```

### Classes (UBPC-9)

Nine semantic classes, defined in Section `sec:dataset_annotation`:

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
ARCHITECTURE.md               Detailed architecture notes
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

### Dataset

The UBPC-9 dataset and raw point clouds/labels are hosted on Baidu NetDisk:

```text
Share name : DataSets
Link       : https://pan.baidu.com/s/1a6ETzXrYIpeoofpxK_8rdQ
Access code: 5577
```

UBPC-9 covers three indoor scenarios: large-span steel-structure workshops,
mixed indoor scenes (long corridors, glass partitions), and underground
parking garages. Data are collected at 0.5 m/s with a MID-360 LiDAR,
de-skewed using per-point timestamps, and accumulated in 0.5 s windows
projected to 64x512 range images. The dataset contains 12,000 frames
(8,400 train / 1,800 val / 1,800 test), split by complete acquisition
sequences to avoid temporal leakage.

## Reviewer-Response Controls

The main model file includes controlled variants for the concerns raised in
review. These correspond to the ablation experiments in Sections
`sec:ablation` and `sec:attention_control_ch3` of the dissertation.

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

### Reviewer mapping

| Concern | Code support | Dissertation reference |
| --- | --- | --- |
| IGCA novelty vs conventional cross-attention | `conventional_g2i`, `conventional_bidir` | Table `tab:attention_control_ch3` |
| Pairwise intensity-curvature bias contribution | `igca_no_icb` | Table `tab:ablation_component` (config I vs H) |
| Directional contribution of IGCA branches | `igca_g2i_only`, `igca_i2g_only` | Table `tab:ablation_igca` |
| Modality complementarity evidence | `linear_cka`, `normalized_cross_covariance`, `complementarity_report` | Section `sec:complementarity_ch3` |
| Robustness under corruptions | `apply_corruption` | Section `sec:robustness_ch3` |
| DSConv accuracy-efficiency trade-off | `standard_conv` | Table `tab:dsconv_tradeoff_ch3` |
| Single-stream controls | `intensity_only`, `geometry_only` | Table `tab:ablation_component` (A, B) |

## Python Usage

### Smoke tests

```bash
cd dual_stream_rangenet
python python/test_ds_rangenet.py
```

### Create the default model

```python
import torch
from python.ds_rangenet_v3 import DualStreamRangeNetV3, IN_TOTAL

model = DualStreamRangeNetV3().eval()
x = torch.randn(1, IN_TOTAL, 64, 512)
with torch.no_grad():
    logits = model(x)
print(logits.shape)  # [1, 9, 64, 512]
```

### Complementarity analysis

```python
from python.ds_rangenet_v3 import build_model, complementarity_report

model = build_model("full").eval()
report = complementarity_report(model, x)
print(report)
```

### Robustness corruptions

```python
from python.ds_rangenet_v3 import apply_corruption

x_noisy = apply_corruption(x, "range_noise")
x_drop  = apply_corruption(x, "point_dropout")
x_block = apply_corruption(x, "block_occlusion")
```

Supported corruptions (Section `sec:robustness_ch3`):

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

The C++ runtime builds the same 16-channel input defined by the dissertation
(Section `sec:feature_decoupling`):

```text
Point cloud [x, y, z, intensity]
  -> spherical projection (Eq. eq:proj_u_ch3, eq:proj_v_ch3)
  -> material-intensity statistics (Eq. eq:int_curvature)
  -> normals and voxel-PCA geometry descriptors (Eq. eq:geo_feature_vec)
  -> RangeImage [1, 16, H, W]
  -> ONNX Runtime inference
  -> pixel labels
  -> back projection and KNN fill
```

### Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### Run

```bash
./build/bin/rangenet_inference \
  --model models/dual_stream_rangenet_v3.onnx \
  --input data/test.bin \
  --output output
```

### Benchmark

```bash
./build/bin/rangenet_inference \
  --model models/dual_stream_rangenet_v3.onnx \
  --benchmark
```

## Sensor Projection Notes

Projection keeps the nearest return when multiple points fall into one pixel.
The primary pipeline does not use input-stage interpolation or hole filling.
Post-prediction KNN fill is applied after pixel labels are back-projected to
the original point cloud (Section `sec:decoder`).

The C++ preprocessor computes the following features:

```text
material-intensity:
  normalized range, local intensity curvature (MLS),
  curvature gradient boundary strength,
  curvature Laplacian high-frequency saliency,
  intensity-curvature covariance

geometry:
  image-space normals, xyz coordinates,
  0.2 m voxel-PCA linearity/planarity/scattering,
  isotropic entropy, relative elevation
```

## Degeneration-Aware Interface

The segmentation head outputs three degeneration indicators for the backend
optimizer (Section `sec:degeneration_index`):

```text
rho_miss  : surfel missing rate       (Eq. eq:surfel_missing_rate)
psi_uni  : normal vector uniformity    (Eq. eq:normal_uniformity)
phi_ff   : front-facade coverage      (Eq. eq:semantic_ff_coverage)
```

These are combined into a composite degeneration index D (Eq. eq:degeneration_index)
and mapped to a LiDAR-IMU relative weight alpha (Eq. eq:alpha_schedule),
with first-order exponential smoothing (Eq. eq:alpha_smooth).

## Regenerate Architecture Figure

```bash
python draw_arch.py
```

This writes:

```text
arch_diagram.png
arch_diagram_v2.png
```

## Training Configuration

Training protocol follows Table `tab:baseline_protocol_ch3`:

```text
Hardware:      NVIDIA RTX 3090
Input:         64 x 512 range image
Batch size:     8
Optimizer:      AdamW (lr=1e-3, weight_decay=1e-4)
Schedule:       Linear warmup (10 epochs) + cosine annealing
Total epochs:   150
Loss:           0.6 * Focal(gamma=2) + 0.4 * Dice(e=1)
Jetson deploy:  AGX Orin 32GB, MAXN, FP16, batch 1
```

## Experimental Results Summary

UBPC-9 test set (Section `sec:comparison`):

```text
Method          Params(M)  mIoU(%)  Latency(ms)  Memory(MB)
RangeNet++(DN53)  50.3      63.7        35         1124
FIDNet             8.0      62.9        26          578
CENet              6.8      64.1        24          601
RangeFormer        38.2      67.3        62         1856
DS-RangeNet        5.69     73.2        37          536
```

Ablation path (Table `tab:ablation_component`):

```text
A (RI only, 2ch)    mIoU 50.8
B (Geo only, 6ch)    mIoU 58.3
C (Dual, 8ch)        mIoU 62.5
D (+VoxPCA, Geo 11ch) mIoU 64.8
E (+ICurv, RI 5ch)   mIoU 66.3
F (+CBAM, 16ch)      mIoU 69.1
G (+IGCA G->I)       mIoU 71.0
H (+IGCA bidirectional)mIoU 72.4
I (+ICB, full)        mIoU 73.2
```

## Important Notes

- Old 8-channel model files were removed.
- Legacy Lite ONNX names should not be used.
- New weights must be trained or exported for the 16-channel model.
- The repository currently expects the final deployment model at:

```text
models/dual_stream_rangenet_v3.onnx
```

## Citation

If you use this code or the UBPC-9 dataset, please cite:

```bibtex
@phdthesis{li2026lightweight,
  author = {Wenguang Li},
  title  = {Lightweight Multi-Dimensional Semantic Extraction via
            Geometry--Intensity-Curvature Deep Coupling},
  school = {Shandong University},
  year   = {2026},
  note   = {Chapter 3: DS-RangeNet. Code and data:
            \url{https://github.com/liwenguang211/dual_stream_rangenet}}
}
```

## License

- Source code: MIT License
- UBPC-9 dataset: research-only (see Baidu NetDisk terms)
- Pre-trained checkpoints: subject to the same terms as the dataset
