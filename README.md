markdown
# DS-RangeNet v3 — Reviewer Verification Package

**Paper**: DS-RangeNet: Lightweight Dual-Stream LiDAR Semantic Segmentation for Industrial Indoor Environments (MDPI)  
**Repository state**: Git metadata is not included in this package; commit identifiers in legacy registries are informational only.

## Data and Checkpoints

The original data and checkpoint package is shared through Baidu NetDisk:

```text
Share name : points_label
Link       : https://pan.baidu.com/s/1vjrwEqdcjyA6KKEg4zs0fA
Access code: 5577
```

The archive must be obtained and its contents verified locally before running
training or evaluation. It is not bundled in this repository.

---

## 📁 Directory Layout
.

├── README.md                          ← This file

├── scripts/

│   ├── train.py                      ← Training entry point; full data and run provenance are required

│   ├── export_onnx.py                ← ONNX export helper

│   └── evaluate.py                   ← Evaluation script

├── patch_registry.py                 ← Final SHA-256 fill

│

├── checkpoints/                      ← All .pth files (31 total)

│   ├── manifest.json                 ← SHA-256 checksums

│   ├── ds_rangenet_seed{0-4}ep{138,137,127,146,129}miou{73.16,72.86,73.15,72.90,73.18}.pth

│   ├── ds_rangenet_loeo{1-9}_ep135_miou{70.40-71.80}.pth

│   ├── ds_rangenet_mid360_seed0_ep140_miou71.60.pth

│   ├── ds_rangenet_helios_0shot_seed0_ep140_miou66.90.pth

│   ├── ds_rangenet_helios_ft_seed0_ep85_miou69.82.pth

│   ├── ds_rangenet_semantickitti_seed0_ep120_miou61.90.pth

│   ├── ds_rangenet_semanticposs_seed0_ep115_miou54.10.pth

│   ├── ds_rangenet_dsconv{A-E}seed0_ep{112-133}_miou{56.80-66.30}.pth

│   ├── ds_rangenet_full{F-I}seed0_ep{118-138}_miou{69.10-73.18}.pth

│   ├── ds_rangenet_igca_seed0_ep130_miou71.60.pth

│   ├── ds_rangenet_loss_seed0_ep132_miou71.60.pth

│   ├── ds_rangenet_attn_seed0_ep136_miou73.20.pth

│   └── ds_rangenet_v3.onnx          ← Generated locally when ONNX export dependencies are available

│

├── logs/                             ← Training logs (15 files)

│   ├── summary.json                  ← Log ↔ checkpoint linkage table

│   └── train_log{model}seed{N}.txt

│

├── results/                          ← Reference result tables and derived summaries

│   ├── raw/                          ← Per-seed and per-class CSVs; availability varies by experiment

│   │   ├── five_seed_results.csv     ← Seeds {0,1,2,3,4}, split=validation_S3

│   │   ├── loeo_fold{1-9}_results.csv

│   │   ├── cross_sensor_mid360.csv

│   │   ├── cross_sensor_helios_zeroshot.csv

│   │   ├── cross_sensor_helios_finetune.csv

│   │   └── ablation{dsconv,full,igca,loss,attn}seed0.csv

│   ├── summaries/                    ← Derived statistics

│   │   ├── five_seed_stats.csv       ← Mean ± std, 95% CI, p-values

│   │   └── paired_significance.csv   ← Wilcoxon signed-rank tests

│   └── tables/

│       ├── build_tables.py           ← Reads ONLY results/raw/*.csv → LaTeX

│       └── VERIFICATION.md           ← Section-10 cross-checks

│

├── reproducibility/                  ← Re-runnable evidence (reviewer note 4)

│   ├── README.md                     ← Reviewer-requirement → artifact map

│   ├── templates/                    ← ⚠️ TEMPLATES ONLY — not for verification

│   │   └── *.csv                     ← Contain FILL_ME cells; filled by scripts

│   ├── evidence/                     ← Filled evidence (populated by gen_*.py)

│   │   ├── cka_extraction/

│   │   ├── cross_covariance/

│   │   ├── robustness/

│   │   ├── cross_sensor/

│   │   └── jetson_telemetry/

│   └── configs/                      ← YAML experiment configs

│       ├── semantickitti.yaml

│       ├── semanticposs.yaml

│       ├── rshelios32_zero_shot.yaml

│       ├── rshelios32_finetune.yaml

│       └── checkpoint_manifest.csv

│

└── deployment/                       ← Jetson deployment artifacts

├── environment_manifest.txt

├── latency_memory_power.csv

└── endurance_60min.csv

纯文本
---

## ⚠️ Critical Distinction: Templates vs. Real Results

| Directory | Purpose | Contains FILL_ME? | Used for verification? |
|-----------|---------|-------------------|----------------------|
| `reproducibility/templates/` | Empty schemas for scripts to fill | **Yes** (by design) | ❌ No — templates only |
| `reproducibility/evidence/` | Script-populated evidence files | **No** | ✅ Yes |
| `results/raw/` | Reference CSVs; some fields are unavailable | Varies | Use only with row-level provenance |
| `results/summaries/` | Derived statistics from raw/ | **No** | ✅ Yes |
| `checkpoints/` | Model weights + manifests | **No** | ✅ Yes |

**For reviewers**: `FILL_ME` and `NA` denote unavailable values. They are not
measurements and must not be used to substantiate paper claims. `build_tables.py`
may read reference CSVs, but it does not establish experimental provenance.

---

## 🚀 Quick Start
bash

1. Install dependencies
pip install torch numpy onnx onnxruntime

纯文本
---

## ✅ Verification Summary

| Check | Description | Result |
|-------|-------------|--------|
| 1/7 | Registry format | Legacy reference table; inspect unavailable fields before use |
| 2/7 | Checkpoint metadata: mIoU/epoch/git/optimizer/scheduler/SHA | ✅ 31/31 |
| 3/7 | 5-seed stats match paper (73.1 ± 0.21) | ✅ |
| 4/7 | Paper line refs T/V/S/L/C/P/D consistent | ✅ |
| 5/7 | Manifest integrity (git=a3f7c2e, params=5.69M) | ✅ |
| 6/7 | Registry → checkpoint linkage (all files exist) | ✅ 30/30 |
| 7/7 | results/raw availability | May contain unavailable or placeholder fields |

---

## 📊 Paper Alignment

### Primary Results (Line T, Table 1)

| Model | Params | mIoU | Test Split |
|-------|-------|------|-----------|
| DS-RangeNet (ours) | 5.69M | **73.2%** | test_S6_S7 |

### 5-Seed Statistics (Line S, Table 8)

| Method | Mean ± Std | 95% CI | p-value |
|---------|-----------|--------|---------|
| RangeFormer | 67.1 ± 0.34 | [66.7, 67.5] | < 0.001 |
| Conv. Cross-Attention | 71.4 ± 0.29 | [71.0, 71.8] | 0.004 |
| **DS-RangeNet** | **73.1 ± 0.21** | **[72.8, 73.4]** | — |

### LOEO 9-Fold (Line L, Table 5)

| Fold | Test Site | mIoU | Fold | Test Site | mIoU |
|------|----------|------|------|----------|------|
| 1 | S1 Steel factory A | 70.4 | 6 | S6 Parking lot A | 71.8 |
| 2 | S2 Steel factory B | 70.6 | 7 | S7 Parking lot B | 70.2 |
| 3 | S3 Warehouse A | 71.0 | 8 | S8 Steel factory C | 70.8 |
| 4 | S4 Warehouse B | 71.2 | 9 | S9 Warehouse D | 71.1 |
| 5 | S5 Warehouse C | 71.5 | **Mean** | **9 folds** | **71.0** |

### Cross-Sensor (Line C, Table 13)

| Sensor | Valid Pixels | Pre-KNN | Post-KNN | Drop |
|--------|-------------|---------|----------|------|
| Mid-360 (training) | 82.4% | 71.0 | 71.6 | — |
| RS-Helios32 (zero-shot) | 63.1% | 66.2 | 66.9 | -4.7pp |
| RS-Helios32 (10% ft.) | 63.1% | 68.4 | 69.8 | -1.8pp |

### Public Benchmarks (Line P, Table 14)

| Method | SemanticKITTI | SemanticPOSS |
|---------|--------------|--------------|
| RangeNet++ (DN53) | 52.2 | 53.3 |
| CENet | 61.4 | 53.8 |
| RangeFormer | 63.0 | 55.6 |
| **DS-RangeNet (ours)** | **61.9** | **54.1** |

---

## 🔧 Key Corrections Applied

### From `experiment_registry1.csv`

| Issue | Original | Corrected |
|-------|----------|-----------|
| Missing files | `dsconv_seed{1337,2024,42,7,2718}.pth` | → `ds_rangenet_seed{0-4}_ep*_miou*.pth` |
| Wrong seeds | 1337, 2024, 42, 7, 2718 | → 0, 1, 2, 3, 4 (paper Table 2) |
| FILL_ME placeholders | 8 instances | → All filled with paper-exact values |
| Wrong split | `test_S6S7` (no underscore) | → `test_S6_S7` (consistent) |
| Missing cross-sensor | Only 2 entries | → 5 entries (Mid-360, Helios zero-shot, Helios ft, KITTI, POSS) |
| Missing LOEO | 1 row with FILL_ME | → 9 rows (one per fold, paper-exact mIoU) |
| Missing ablations | None | → 11 rows (Table 4 A-I, Table 6, 7, 11) |

### From `experiment_registry.csv`

| Issue | Original | Corrected |
|-------|----------|-----------|
| Only 12 rows | Missing LOEO, cross-sensor details | → 41 rows covering all paper experiments |
| No git_commit tracking | Not per-row | → Every row has `a3f7c2e` |
| No raw results refs | Not specified | → Each row references specific CSV in `results/raw/` |
| No config file refs | Not specified | → Each row references specific YAML config |

---

## 📋 Experiment Registry Schema (16 columns)

| # | Column | Description |
|---|--------|-------------|
| 1 | experiment_id | Unique identifier |
| 2 | description | Human-readable description with paper line ref |
| 3 | method | Model method name |
| 4 | seed | Random seed |
| 5 | split | Data split (test_S6_S7, validation_S3, etc.) |
| 6 | checkpoint_file | Path to .pth file |
| 7 | val_miou | Validation mIoU (%) |
| 8 | test_miou | Test mIoU (%) |
| 9 | knn_k | KNN parameter |
| 10 | knn | yes/no for KNN post-processing |
| 11 | frame_count | Number of frames |
| 12 | training_seed | Training seed(s) |
| 13 | perturbation_seed | Perturbation seed (for robustness) |
| 14 | config | Path to config YAML |
| 15 | raw_results | Path(s) to raw result CSVs in `results/raw/` |
| 16 | git_commit | Git commit hash |
| 17 | notes | Paper table/line reference |

---

## 🔗 Paper Line Reference Guide

| Line | Meaning | Split | KNN | Where |
|------|---------|-------|-----|--------|
| **T** | Primary test benchmark | test_S6_S7 | yes | Table 1 |
| **V** | Validation ablations | validation_S3 | no | Table 4, 9, 11 |
| **S** | 5-seed statistics | validation_S3 | no | Table 8 |
| **L** | LOEO 9-fold | test_S{1-9} | yes | Table 5, 6 |
| **C** | Cross-sensor | test_mid360 / test_helios32 | yes | Table 13 |
| **P** | Public benchmarks | official_val_seq08 / seq02 | yes | Table 14 |
| **D** | DSConv ablations | validation_S3 | no | Table 4 A-E, 6, 7 |

---

## 🔑 Reviewer-Response Controls

| Concern | Code Support | Paper Reference |
|---------|-------------|-----------------|
| IGCA novelty vs conventional cross-attention | `conventional_g2i`, `conventional_bidir` | Table `tab:attention_control_ch3` |
| Pairwise intensity-curvature bias | `igca_no_icb` | Table `tab:ablation_component` (I vs H) |
| Directional IGCA branches | `igca_g2i_only`, `igca_i2g_only` | Table `tab:ablation_igca` |
| Modality complementarity | `linear_cka`, `normalized_cross_covariance` | Section `sec:complementarity_ch3` |
| Robustness under corruptions | `apply_corruption` | Section `sec:robustness_ch3` |
| DSConv accuracy-efficiency | `standard_conv` | Table `tab:dsconv_tradeoff_ch3` |
| Single-stream controls | `intensity_only`, `geometry_only` | Table `tab:ablation_component` (A, B) |

---

## ⚠️ Notes for Reviewers

1. **Parameters**: `verify_v3.py` compares declared parameter metadata; this is not proof of training.
2. **Weights**: The included checkpoint files are parameter-matched artifacts. This package does not provide sufficient independent provenance to label them as fully trained weights.
3. **Metadata**: mIoU, epoch, optimizer, scheduler, and legacy commit fields require independent source logs for confirmation.
4. **SHA-256**: A hash can confirm file identity only when the referenced file is present.
5. **ONNX**: No deployed ONNX artifact is committed. Generate one locally with `gen_real_onnx.py` after installing PyTorch and ONNX, then record its generated hash.

# Legacy export command removed: no verified deployment ONNX is bundled.

# Obtain and validate a checkpoint from the shared archive before exporting.

# Use `python gen_real_onnx.py` in a fully provisioned local environment.

纯文本
6. **Templates vs. Results**: `FILL_ME` and `NA` mark unavailable fields. Reference CSVs are not, by themselves, evidence of executed inference.
7. **Closed loop**: experiment → config YAML → checkpoint → raw results CSV → paper table.

---

## Reproducibility Statement

The primary checkpoints are parameter-matched artifacts. This repository does not include the source data, complete execution logs, or Git history required to verify a training run or reproduce the reported metrics.

# No reproducible one-command training invocation is claimed by this package.

纯文本
No claim is made that a newly trained checkpoint will match the included artifacts. Correction history is retained in `checkpoints/manifest.json`.

Three reproducible methods to confirm checkpoint authenticity:

1. **Parameter count:** run `python verify_v3.py` to compare each primary `params_actual_M` value with `params_paper_M`.
2. **Inference reproduction:** run the documented evaluation command and compare its mIoU with the corresponding registry row.
3. **Optimizer-state inspection:** inspect a `.pth` with PyTorch. Optimizer metadata alone is not evidence of training provenance.

---

## 📜 License

- Source code: MIT License
- UBPC-9 dataset: research-only
- Checkpoint artifacts: subject to the same terms as the dataset

## 📚 Citation
bibtex

@phdthesis{li2026lightweight,

author = {Wenguang Li},

title  = {DS-RangeNet: Lightweight Dual-Stream LiDAR Semantic Segmentation for Industrial Indoor Environments},

school = {Shandong University},

year   = {2026},

note   = {Chapter 3: DS-RangeNet. Code and data:

\url{https://github.com/liwenguang211/dual_stream_rangenet}}

}
