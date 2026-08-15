# Reproducibility Evidence Chain

This directory provides the reproducibility evidence chain requested in review
(paper modification note, item 4). It contains runnable scripts, configuration
files, and result templates so that every reviewer requirement can be traced to
a concrete, re-runnable artifact.

Numeric result files (per-seed mIoU, Jetson telemetry, official evaluation
logs) are shipped as templates with an explicit header/schema and empty data
rows. They must be filled by running the provided scripts on the actual
trained models. Placeholder cells are marked `FILL_ME` and are intentionally
not populated with fabricated numbers.

---

## 1. Directory Layout

```
reproducibility/
├── README.md                     this file
├── verify_chain.py               chain verification: refuses FILL_ME
├── seeds/
│   ├── run_seeds.py             train/evaluate 5 seeds, write per-seed CSV
│   ├── paired_significance.py    paired t-test + Wilcoxon on per-seed CSVs
│   └── results/
│       ├── per_seed_miou.csv     template: per-seed, per-class mIoU (15 data rows)
│       └── per_seed_summary.csv  template: mean/std per model over seeds
├── cka/
│   ├── extract_cka.py          CKA / normalized cross-covariance extraction
│   ├── cka_config.yaml          layers, frame ids, token sampling, seeds
│   └── results/
│       ├── cka_per_seed.csv      template: per-seed CKA/cross-cov per layer pair
│       └── cka_summary.csv      FILLED: 5-stage CKA/ρ_cc summary (paper appendix)
├── corruption/
│   ├── corruption_config.yaml   corruption kinds, severities, seeds
│   ├── run_corruption.py        batch corruption generation + evaluation
│   └── results/
│       └── corruption_results.csv  template: mIoU per corruption/severity/seed
├── cross_sensor/
│   ├── semanticposs.yaml        SemanticPOSS projection + class mapping config
│   ├── semantickitti.yaml      SemanticKITTI projection + class mapping config
│   ├── checkpoint_manifest.csv  template: dataset/model/seed → checkpoint + sha256
│   └── eval_log_template.txt   official evaluator log format
├── deployment/
│   ├── checkpoint_manifest.csv  template: Standard/Hybrid/DSConv params + sha256
│   ├── jetson_telemetry.csv    template: 60-min telemetry, 1 Hz sampling
│   ├── log_jetson_telemetry.py telemetry logger for Jetson (tegrastats parser)
│   └── environment_manifest.yaml template: TensorRT/CUDA/power-mode/clocks/ambient temp
└── degeneration/
    └── degeneration_statistics.csv FILLED: Pearson r(D, λ_min) = −0.81, 3-level D stats
```

---

## 2. Reviewer Requirement → Evidence Artifact Map

### Reviewer 1 (R1) — Public dataset, multi-seed, statistical significance

| Requirement | Evidence artifact | How to verify |
|---|---|---|
| Public dataset, multi-seed | `seeds/run_seeds.py`, `seeds/results/per_seed_miou.csv` | Run script with 5 seeds; inspect 15 data rows (3 models × 5 seeds) |
| Statistical significance | `seeds/paired_significance.py` | Reads `per_seed_miou.csv`; refuses to run if `FILL_ME` present |
| Modality complementarity (CKA / cross-covariance) | `cka/extract_cka.py`, `cka/cka_config.yaml`, `cka/results/cka_per_seed.csv`, `cka/results/cka_summary.csv` | Run with seeds `{0,1,2,3,4}`; summary already FILLED from paper appendix |
| Robustness under corruptions | `corruption/corruption_config.yaml`, `corruption/run_corruption.py`, `corruption/results/corruption_results.csv` | Run with seeds `{1337,2024,42,7,2718}`; severity ∈ {1,2,3} |

### Reviewer 2 (R2) — Baseline fairness, long-term deployment, source/weights

| Requirement | Evidence artifact | How to verify |
|---|---|---|
| Baseline fairness (DSConv vs Standard) | `deployment/checkpoint_manifest.csv` | 3 rows: Standard (48.3M) / Hybrid / DSConv (5.69M); fill `params_M` from `model.param_summary()` |
| Long-term deployment (60-min Jetson) | `deployment/log_jetson_telemetry.py`, `deployment/jetson_telemetry.csv`, `deployment/environment_manifest.yaml` | Run on Jetson device; parse tegrastats at 1 Hz for 3600 s |
| Source/weights availability | `deployment/checkpoint_manifest.csv`, `cross_sensor/checkpoint_manifest.csv` | SHA-256 digests; GitHub release links |

### Reviewer 3 (R3) — Cross-sensor projection, holes, filling, DSConv trade-off

| Requirement | Evidence artifact | How to verify |
|---|---|---|
| Cross-sensor projection (SemanticPOSS 40-beam / SemanticKITTI 64-beam) | `cross_sensor/semanticposs.yaml`, `cross_sensor/semantickitti.yaml` | Projection params: height, width, FOV, nearest_return policy |
| Holes and filling policy | `cross_sensor/semanticposs.yaml` (`input_stage_hole_filling: false`, `validity_mask: true`, `knn_postproc.k: 5`) | No input-stage interpolation; KNN post-prediction fill applied after back-projection |
| DSConv accuracy-efficiency trade-off | `deployment/checkpoint_manifest.csv` (Standard / Hybrid / DSConv rows) | Params: 48.3M / 7.8M / 5.69M; latency vs mIoU on Jetson Orin |

---

## 3. Tool Versions

| Tool | Minimum version | Purpose |
|---|---|---|
| Python | 3.9 | Script runtime |
| PyTorch | 2.0 | Model definition, inference, CKA extraction |
| CUDA | 11.8 | GPU training (A100-80G server) |
| ONNX Runtime | 1.15 | C++ deployment inference |
| TensorRT | 8.5 | Optional: Jetson deployment optimization |
| CMake | 3.16 | C++ runtime build |
| GCC | 9.4 | C++17 compilation |
| tegrastats | Jetson-included | 1 Hz telemetry (GPU/CPU/RAM/power/temp) |

---

## 4. Step-by-Step Guide

### Step 1 — Install dependencies

```bash
cd dual_stream_rangenet
pip install -r requirements.txt
```

### Step 2 — Train / export models

Train the five-seed suite (full model, SalsaNext baseline, CENet baseline):

```bash
# From dual_stream_rangenet/python/
python train.py --config configs/ubpc9_train.yaml --seed 1337
python train.py --config configs/ubpc9_train.yaml --seed 2024
# ... repeat for seeds {1337, 2024, 42, 7, 2718}
```

Export ONNX for deployment:

```bash
python export_onnx.py --weights checkpoints/full_seed1337.pth \
    --output models/dual_stream_rangenet_v3.onnx
python export_onnx.py --variant conventional_bidir \
    --output models/dual_stream_rangenet_v3_conventional_bidir.onnx
```

### Step 3 — Run five-seed evaluation

```bash
cd reproducibility/seeds
python run_seeds.py \
    --checkpoints-dir ../deployment/checkpoints \
    --frames-dir /path/to/ubpc9/test_frames
```

This populates `seeds/results/per_seed_miou.csv` (15 data rows: 3 models × 5 seeds).

### Step 4 — Compute statistical significance

```bash
cd reproducibility/seeds
python paired_significance.py \
    --csv results/per_seed_miou.csv \
    --reference full \
    --split test \
    --out results/significance_report.csv

# Regenerate per_seed_summary.csv
python paired_significance.py --summary
```

### Step 5 — Extract CKA / cross-covariance

```bash
cd reproducibility/cka
python extract_cka.py \
    --config cka_config.yaml \
    --checkpoints-dir ../deployment/checkpoints \
    --frames-dir /path/to/ubpc9/val_frames
```

Seeds in `cka_config.yaml`: `{0, 1, 2, 3, 4}`.
Layer pairs: `input_modalities`, `independent_encoders`, `after_aspp`, `after_fusion`.
Token sampling: 4096 per frame, deterministic via `seed + rng_offset`.

### Step 6 — Run corruption robustness

```bash
cd reproducibility/corruption
python run_corruption.py \
    --config corruption_config.yaml \
    --checkpoints-dir ../deployment/checkpoints \
    --frames-dir /path/to/ubpc9/test_frames \
    --eval test_ds_rangenet:eval_frames
```

Seeds: `{1337, 2024, 42, 7, 2718}`. Severities: `{1, 2, 3}`.
Corruptions: `range_noise`, `intensity_noise`, `point_dropout`, `scanline_dropout`, `block_occlusion`.
Per-frame RNG = `seed + rng_offset (70003) + frame_id` for reproducibility.

### Step 7 — Cross-sensor zero-shot transfer

```bash
# SemanticPOSS (Pandora 40-beam)
python test_ds_rangenet.py \
    --config cross_sensor/semanticposs.yaml \
    --checkpoint deployment/checkpoints/full_semanticposs_seed1337.pth

# SemanticKITTI (Velodyne HDL-64E)
python test_ds_rangenet.py \
    --config cross_sensor/semantickitti.yaml \
    --checkpoint deployment/checkpoints/full_semantickitti_seed1337.pth
```

Key config flags: `input_stage_hole_filling: false`, `validity_mask: true`,
`knn_postproc.k: 5`, `keep: nearest_return`.

### Step 8 — Jetson 60-minute telemetry

Run **on the Jetson Orin device** (tegrastats unavailable off-device):

```bash
cd reproducibility/deployment
python log_jetson_telemetry.py \
    --minutes 60 \
    --interval-ms 1000 \
    --out jetson_telemetry.csv
# Optional: merge inference latency sidecar
python log_jetson_telemetry.py \
    --minutes 60 --latency-file infer_latency.csv
```

Then fill `environment_manifest.yaml` from:
- `cat /etc/nv_tegra_release` → `device`, `jetpack`, `l4t`
- `nvpmodel -q` → `nvpmodel_mode`, `power_budget_w`
- `jetson_clocks --show` → `cpu_clock_mhz`, `gpu_clock_mhz`, `emc_clock_mhz`
- Room thermometer → `ambient_temp_c`

---

## 5. Script Conventions

### 5.1 CLI flags

| Flag | Script | Purpose |
|---|---|---|
| `--config` | `extract_cka.py`, `run_corruption.py` | YAML config path |
| `--checkpoints-dir` | `extract_cka.py`, `run_corruption.py` | Directory with `.pth` files |
| `--frames-dir` | `extract_cka.py`, `run_corruption.py` | Directory with `frame_<id>.pt` tensors |
| `--eval` | `run_corruption.py` | `module:function` for mIoU hook |
| `--csv` | `paired_significance.py` | Path to `per_seed_miou.csv` |
| `--reference` | `paired_significance.py` | Reference model name (default: `full`) |
| `--split` | `paired_significance.py` | `val` or `test` |
| `--out` | most scripts | Output CSV path |
| `--minutes` | `log_jetson_telemetry.py` | Telemetry duration (default 60) |
| `--interval-ms` | `log_jetson_telemetry.py` | Sampling interval (default 1000) |
| `--latency-file` | `log_jetson_telemetry.py` | Optional sidecar latency CSV |

### 5.2 Output discipline

- Every script writes only to its own `results/` folder.
- Scripts **never** fabricate numbers. If `FILL_ME` cells are present,
  `paired_significance.py` and `verify_chain.py` abort with a clear message.
- `log_jetson_telemetry.py` exits immediately if `tegrastats` is not in PATH.

### 5.3 Determinism

| Source | How it is fixed |
|---|---|
| Training seeds | Explicit `--seed` argument, set via YAML |
| CKA token sampling | `torch.Generator().manual_seed(seed + rng_offset)` |
| Corruption per-frame RNG | `torch.Generator().manual_seed(seed + 70003 + frame_id)` |
| PyTorch determinism | Set `torch.backends.cudnn.deterministic = True` in training script |

### 5.4 Error handling

| Condition | Behavior |
|---|---|
| `FILL_ME` cell in input CSV | Exit with message: "Refusing to run: still contains FILL_ME" |
| `tegrastats` not found | Exit: "tegrastats not found. Run on Jetson device." |
| `--eval` not in `module:function` form | Exit: "--eval must be 'module:function'" |
| YAML parse failure | Fall back to mini-parser; warn "YAML not available, using fallback" |
| Checkpoint not found | Exit: "Checkpoint {path} does not exist" |

---

## 6. Template Filling Reference

### 6.1 FILLED templates (no action needed)

| File | Status | Key values |
|---|---|---|
| `cka/results/cka_summary.csv` | ✅ FILLED | 5 stages: input 0.110→early_concat 0.718; mIoU 62.5→73.2 |
| `degeneration/degeneration_statistics.csv` | ✅ FILLED | Pearson r(D, λ_min) = −0.81; 3-level RPE |

### 6.2 Templates to fill (run scripts first)

| File | Filled by | Key columns |
|---|---|---|
| `seeds/results/per_seed_miou.csv` | `run_seeds.py` | model, seed, split, mIoU, per-class IoU (9 cols) |
| `seeds/results/per_seed_summary.csv` | `paired_significance.py --summary` | model, split, n_seeds, mIoU_mean, mIoU_std, min, max |
| `seeds/results/significance_report.csv` | `paired_significance.py` | model, n_pairs, ref_mean, model_mean, mean_gap, t_stat, t_pvalue, wilcoxon_pvalue |
| `cka/results/cka_per_seed.csv` | `extract_cka.py` | seed, layer_pair, n_frames, tokens_per_frame, linear_cka, normalized_cross_cov |
| `corruption/results/corruption_results.csv` | `run_corruption.py` | model, seed, corruption, severity, mIoU, mIoU_drop_vs_clean, per-class IoU |
| `deployment/jetson_telemetry.csv` | `log_jetson_telemetry.py` | timestamp_s, elapsed_s, fps, latency_ms, gpu_util_pct, gpu_freq_mhz, cpu_util_pct, ram_used_mb, power_total_mw, soc_temp_c, gpu_temp_c |
| `deployment/checkpoint_manifest.csv` | Author | model, conv_type, variant_flag, seed, params_M, checkpoint_path, onnx_path, sha256_pth, sha256_onnx |
| `deployment/environment_manifest.yaml` | Author (from Jetson) | device, jetpack, l4t, cuda, cudnn, tensorrt, onnxruntime, nvpmodel_mode, clocks, ambient_temp_c |
| `cross_sensor/checkpoint_manifest.csv` | Author | dataset, model, seed, sensor_beams, checkpoint_path, sha256, prediction_dir, official_eval_log |

### 6.3 SHA-256 checksums

Compute for every artifact:

```bash
sha256sum checkpoints/full_seed1337.pth
sha256sum models/dual_stream_rangenet_v3.onnx
```

Record the digest in both `checkpoint_manifest.csv` files.

---

## 7. Pre-Submission Checklist

### ✅ Completed (author-side)

- [x] `cka/results/cka_summary.csv` — FILLED with paper appendix values
- [x] `degeneration/degeneration_statistics.csv` — FILLED (Pearson r = −0.81)
- [x] `verify_chain.py` — refuses to run on templates containing `FILL_ME`
- [x] `paired_significance.py` — paired t-test + Wilcoxon, dependency-free
- [x] `extract_cka.py` — uses `linear_cka` / `normalized_cross_covariance` from `ds_rangenet_v3.py`
- [x] `run_corruption.py` — calls `apply_corruption` from `ds_rangenet_v3.py`
- [x] `log_jetson_telemetry.py` — parses tegrastats, refuses off-device execution
- [x] Cross-sensor YAMLs — projection params, hole-filling policy, label maps documented
- [x] `eval_log_template.txt` — official evaluator output format specified

### ⏳ To be filled (reviewer-action or post-training)

- [ ] `seeds/results/per_seed_miou.csv` — run `run_seeds.py` after training
- [ ] `seeds/results/per_seed_summary.csv` — run `paired_significance.py --summary`
- [ ] `seeds/results/significance_report.csv` — run `paired_significance.py`
- [ ] `cka/results/cka_per_seed.csv` — run `extract_cka.py` after training
- [ ] `corruption/results/corruption_results.csv` — run `run_corruption.py`
- [ ] `deployment/jetson_telemetry.csv` — run on Jetson device for 60 min
- [ ] `deployment/checkpoint_manifest.csv` — fill params_M + SHA-256 after training
- [ ] `deployment/environment_manifest.yaml` — fill from Jetson `nvpmodel -q` etc.
- [ ] `cross_sensor/checkpoint_manifest.csv` — fill SHA-256 after cross-sensor training
- [ ] `cross_sensor/eval_log_template.txt` — paste official evaluator output

---

## 8. Contact & Citation

**Corresponding authors:**
- Jun Zhou (zhoujun@sdu.edu.cn)
- Panling Huang (hfpl@sdu.edu.cn)

**Repository:** https://github.com/liwenguang211/dual_stream_rangenet

**BibTeX:**

```bibtex
@misc{dsrangenet2026,
  title  = {{DS-RangeNet}: Lightweight Dual-Stream LiDAR Semantic Segmentation
            for Industrial Indoor Environments},
  author = {Li, Wenguang and Ren, Jiying and Ou, Jinshun and
            Ma, Yongxin and Zhou, Jun and Huang, Panling},
  year   = {2026},
  howpublished = {\url{https://github.com/liwenguang211/dual_stream_rangenet}},
  note   = {Code and reproducibility evidence at reproducibility/}
}
```

---

## 9. License

- **Code** (Python, C++): MIT License
- **Model weights** (`checkpoints/*.pth`, `models/*.onnx`): MIT License
- **UBPC-9 dataset**: Custom research-only license; raw data hosted on Baidu NetDisk
  (Share: DataSets, Link: https://pan.baidu.com/s/1a6ETzXrYIpeoofpxK_8rdQ, Access code: 5577)
- **Pre-trained weights release**: upon acceptance

---

*Last updated: 2026-07-21*
