# RS-Helios32 Zero-Shot Transfer: Detailed Notes

## Paper Reference
Table tab:cross_sensor_test, row 2: "RS-Helios32 (zero-shot)"
Section sec:generalization (Line C)
Paper text: "Zero-shot transfer to RS-Helios32 loses 4.7pp after the same KNN
protocol (71.6% → 66.9%)."

## Key Numbers (from paper)

| Metric | Value | Source |
|---------|-------|--------|
| Mid-360 valid pixels | 82.4% | paper text line ~1369 |
| RS-Helios32 valid pixels | 63.1% | paper text line ~1371 |
| Mid-360 mIoU pre-KNN | 71.0% | Table tab:cross_sensor_test |
| Mid-360 mIoU post-KNN | 71.6% | Table tab:cross_sensor_test |
| Helios32 mIoU pre-KNN | 66.2% | Table tab:cross_sensor_test |
| Helios32 mIoU post-KNN | 66.9% | Table tab:cross_sensor_test |
| Drop (post-KNN) | -4.7pp | Table tab:cross_sensor_test |
| KNN effect (Helios) | +0.7pp | paper text "~0.6-0.8pp" |
| KNN effect (Mid-360) | +0.6pp | paper text "~0.6-0.8pp" |

## Why 4.7pp Drop?

The paper identifies three compounding factors:

1. **Beam count reduction (40→32)**: Sparser vertical sampling → more empty
   cells (63.1% valid vs 82.4%). Voxel-PCA and intensity-curvature descriptors
   computed in 3D are robust to this, but raw-channel features lose detail.

2. **Different scan pattern**: Mid-360 uses non-repetitive rosette scan
   (irregular coverage over 0.2s window); Helios32 uses mechanical spinning
   rings (regular but coarser vertical spacing). The projection collision rule
   (nearest_return) handles both, but the resulting range-image texture differs.

3. **Sensor-specific intensity distribution**: Helios32 intensity calibration
   differs from Mid-360. The 5-channel material-intensity stream uses
   z-score normalization fitted on Mid-360 training statistics, which may
   be suboptimal for Helios32's intensity range.

## What Was NOT Done (Important Caveats)

- **No input interpolation**: Empty cells remain zero-filled (not bilinearly
  interpolated). This is documented in the config and eval log.
- **No domain adaptation**: No adversarial training, no moment matching,
  no self-training on target data.
- **No hyperparameter tuning on target**: The same lr=1e-3, wd=1e-4, 150
  epochs schedule from UBPC-9 training is used.
- **No ensembling**: Single checkpoint evaluation, not multi-seed averaging.

## Files in this Directory

| File | Purpose |
|------|---------|
| rshelios32_zero_shot.yaml | Full config (sensor, projection, labels, protocol) |
| rshelios32_finetune.yaml | 10% fine-tuning protocol (5 subsets) |
| logs/rshelios32_zero_shot_eval.txt | Detailed eval log (per-class IoU, holes) |
| logs/rshelios32_ft_10pct_all.txt | 5-seed aggregated fine-tuning results |
| checkpoint_manifest.csv | Unified manifest (17 rows, all experiments) |

## Reproducibility Checklist

- [x] Sensor FOV and resolution documented
- [x] Projection method (spherical, nearest_return) specified
- [x] No input-stage hole filling (policy explicit)
- [x] Validity mask usage documented
- [x] KNN post-processing parameters (k, window, sigma) specified
- [x] 16-channel construction documented per sensor
- [x] Label mapping (UBPC-9 schema) provided
- [x] Checkpoint SHA-256 in manifest
- [x] Evaluator version/commit recorded
- [x] Per-class IoU disclosed
- [x] Valid pixel ratio and hole ratio reported
- [x] Pre/post-KNN values both reported
- [x] Domain gap interpretation stated ("cross-sensor" not "unseen-env")
- [x] Shared physical sites acknowledged (S1, S2, S5, S6, S8)
- [x] Temporal-segment sampling (not random frames) for 10% ft.
- [x] Fixed test set across all 5 fine-tuning seeds
- [x] No test frame in adaptation or validation subsets
