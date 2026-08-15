# Reproducibility Guide — DS-RangeNet v3

This document maps **every table and headline number in the paper** to the exact
command, config file, and raw result file needed to regenerate it. The reviewer
path is:

```
data split  →  preprocessing  →  training  →  inference  →  raw logs  →  paper tables
```

Nothing in the paper is hand-entered. All LaTeX tables are produced by
`results/tables/build_tables.py`, which reads **only** the raw CSVs under
`results/raw/` and `results/summaries/`. If a raw value is missing it appears as
`FILL_ME` — it is never invented.

---

## 0. One-time setup

```bash
# option A: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# option B: conda (also pins CUDA 11.8)
conda env create -f environment.yml
conda activate ds-rangenet
```

Download the UBPC-9 dataset and checkpoints per
[`data/README_UBPC9.md`](data/README_UBPC9.md) and
[`checkpoints/README.md`](checkpoints/README.md), then verify integrity:

```bash
sha256sum -c checkpoints/sha256sums.txt
```

---

## 1. Verify the split is physically leakage-free (do this first)

```bash
python scripts/verify_site_disjoint.py --splits data/splits --loeo data/splits/loeo
pytest tests/test_split_disjoint.py
```

Expected: no physical building (site) appears in more than one of train/val/test,
for the primary split and for all 9 LOEO folds. Source of truth:
`loeo_supporting_materials/ubpc9_splits_v3.json`.

Primary split (from the split config): train = S1,S2,S4,S5,S8,S9 (8600 frames);
val = S3 (1300); test = S6,S7 (2100); total 12000.

---

## 2. Preprocessing (raw point cloud → 16-channel range image)

```bash
python scripts/preprocess_ubpc9.py \
  --config configs/dataset/ubpc9.yaml \
  --out data/processed
```

Produces the 64×512 spherical projection with the 5-channel material/intensity-
curvature stream and the 11-channel geometry stream (16 channels total).
Determinism is checked by `pytest tests/test_16ch_preprocessing.py`.

---

## 3. Table-by-table reproduction

Legend: **cmd** = command to regenerate results, **cfg** = config, **raw** = raw
log file that stores every real inference value, **value** = key number(s) from
the raw repo logs already committed.

### Table: Main comparison (primary test split S6+S7)

- **cmd:** `python scripts/run_controlled_baselines.py --config configs/train/default.yaml`
- **cfg:** `configs/models/{ds_rangenet,single_stream_16ch,cenet_16ch}.yaml`
- **raw:** `results/raw/main_comparison.csv`, `results/raw/per_class_iou.csv`
- **value (5-seed mean mIoU):** DS-RangeNet v3 = **73.25**, SalsaNext = **68.60**,
  CENet = **65.95** (means over seeds {1337,2024,42,7,2718}, from
  `reproducibility/seeds/results/per_seed_miou.csv`).
- **primary per-class IoU (DS-RangeNet v3):** background 88.5, ground 89.6,
  roof 77.1, side_facade 78.0, front_facade 73.0, dynamic 62.4, beam 60.2,
  column 63.5, window 66.8.

### Table: Five-seed statistics + significance

- **cmd:** `python scripts/run_seed_statistics.py --config configs/train/default.yaml`
- **raw:** `results/raw/five_seed_results.csv` (all 15 rows, one per model×seed)
- **summary:** `results/summaries/statistical_tests.csv` (paired t-test / Wilcoxon)
- **value:** DS-RangeNet v3 mIoU range 73.23–73.28; the mean/std and p-values are
  recomputed by `build_tables.py`, never hard-coded.

### Table: Ablation (conv type / lightweighting)

- **cmd:** `python scripts/run_ablation.py --config configs/train/default.yaml`
- **value (params):** standard 48.3 M, hybrid 7.8 M, **DSConv 5.69 M** (reported).
- See [`MODEL_ZOO.md`](MODEL_ZOO.md) for checkpoints and SHA256.

### Table: Modality controls

- **cmd:** `python scripts/run_modality_controls.py --config configs/modality/controls.yaml`
- **raw:** `results/raw/modality_controls.csv`
- **value (mean mIoU):** geometry_only 58.30, reflectance_only 56.80,
  early_fusion 67.90, intensity_missing 65.40, intensity_corrupted 66.10,
  geometry_sparse 67.30, cross_frame_mismatch 64.67, DS-RangeNet baseline 73.20
  (from `modality_controls/results_summary.csv`). Independent-model rows retrain;
  same-checkpoint-perturbation rows reuse the baseline checkpoint.

### Table: Representation analysis (CKA / cross-covariance)

- **cmd:** `python scripts/run_representation_analysis.py --config configs/train/default.yaml`
- **value:** input modalities CKA 0.11, independent encoders 0.182,
  after CBAM 0.413, after IGCA 0.631, early-concat fusion 0.718
  (from `reproducibility/cka/results/cka_summary.csv`).

### Table: LOEO 9-fold cross-validation

- **cmd:** `python scripts/run_loeo.py --config configs/loeo/nine_folds.yaml`
- **cfg:** `data/splits/loeo/fold_S1..S9.yaml`
- **raw:** `results/raw/loeo_folds.csv`
- **value (per-fold mIoU):** 70.4, 70.6, 71.0, 71.2, 71.5, 71.8, 70.2, 70.8, 71.1;
  macro-avg **70.96** (reported 71.0), std 0.52 pp, drop from primary **2.24 pp**
  (reported 2.2). Protocol: 7 train + 1 val + 1 test, disjoint; val = S3 (S4 when
  S3 is test); KNN k=3.

### Table: Corruption robustness (3 severities)

- **cmd:** `python scripts/run_corruption.py --config configs/robustness/three_severity.yaml`
- **raw:** `results/raw/robustness_runs.csv` (**every** run stored, not just means)
- **summary:** `results/summaries/robustness_summary.csv`
- **note:** clean baseline + 5 corruption families × 3 severities, full model
  seed 0; source `reproducibility/corruption/results/corruption_results.csv`.
  Corruptions act on raw data, then the 16-channel features are recomputed.

### Table: Cross-sensor transfer

- **cmd:** `python scripts/run_cross_sensor.py --config configs/cross_sensor/mid360.yaml`
- **raw:** `results/raw/cross_sensor_runs.csv`
- **checkpoints:** SemanticKITTI (64-beam) and SemanticPOSS (40-beam), seed 1337,
  KNN k=5 — see [`MODEL_ZOO.md`](MODEL_ZOO.md).

### Table / figures: Jetson AGX Orin deployment

- **cmd:** `python scripts/export_onnx.py` then `python scripts/benchmark_jetson.py`
- **raw:** `deployment/{latency_raw.csv,memory_raw.csv,power_raw.csv,endurance_60min.csv}`
- **value:** latency ≈ 37 ms, memory ≈ 536 MB, power ≈ 18.5 W, no throttling over
  60 min (3600 samples in `reproducibility/deployment/jetson_telemetry.csv`).

---

## 4. Regenerate all LaTeX tables

```bash
python results/tables/build_tables.py --raw results/raw --out results/tables/generated
pytest tests/test_table_recalculation.py
```

`build_tables.py` reads only raw/summary CSVs. `test_table_recalculation.py`
asserts that each reported mean equals the mean recomputed from the raw rows, so
no manual adjustment can slip in.

---

## 5. Determinism contract

- Global training seeds: {1337, 2024, 42, 7, 2718}.
- Per-frame corruption RNG: `seed + 70003 + frame_id` (see `src/corruptions/`).
- Intensity-missing at 100% is bit-exact reproducible
  (`tests/test_corruption_determinism.py`).
- KNN post-prediction refinement: k=3 (primary & LOEO), k=5 (cross-sensor).
- No fine-tuning or checkpoint selection is ever performed on a test site.
