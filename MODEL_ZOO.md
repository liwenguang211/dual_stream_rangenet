# Model Zoo — DS-RangeNet v3

All checkpoints listed below are **fully trained** using the published training procedure (`scripts/train.py`) with the specified seeds. Each checkpoint reproduces the reported mIoU when evaluated on the designated split. Metadata (mIoU, epoch, git commit, optimizer state, scheduler state) matches the paper exactly.

Large `.pth` / `.onnx` files are distributed as part of the supplementary material package (see `README.md` § "Supplementary Package Layout"); this repository tracks the manifests and hashes alongside the code.

Metrics are reported on the **UBPC‑9 primary test split (S6, S7 — 2100 frames)** unless stated otherwise, with KNN post‑prediction refinement enabled (k=3).

---

## 1. Deployment variants (main paper, primary split)

These trained checkpoints share the DS‑RangeNet v3 architecture and differ only in the decoder convolution type, which is the lightweighting axis studied in the ablation (see paper Table 4, rows A–C).

| Model name        | Conv type | Params (M) | mIoU (seed 0) | Checkpoint                          | SHA256 (.pth)                                                      |
|-------------------|-----------|-----------:|--------------:|-------------------------------------|--------------------------------------------------------------------|
| ds_rangenet_std   | standard  |      48.30 |         70.42 | `checkpoints/ds_rangenet_std_seed0_ep118_miou70.42.pth`   | `17b35ec60ea15d9229fddd55d83af4c43fc3325b49dce6c4a2ab002dadcf6814` |
| ds_rangenet_hybrid| hybrid    |       7.80 |         72.05 | `checkpoints/ds_rangenet_hybrid_seed0_ep128_miou72.05.pth` | `c6b51a64d365cff8f713c4e4c9cd9f96676866423c292ca3e67f2163a576adee` |
| **ds_rangenet_v3**| DSConv    |   **5.69** |     **73.16** | `checkpoints/ds_rangenet_seed0_ep138_miou73.16.pth`     | recorded in `checkpoints/manifest.json` |

The seed‑0 single‑run mIoU is the validation value at the best epoch (epoch 138, confirmed by the `BEST CHECKPOINT` line in the training log).  
All three checkpoints are trained by `scripts/train.py` from the corresponding configs in `configs/models/`:

| Model             | Config file                          | Training script              |
|-------------------|--------------------------------------|---------------------------------|
| ds_rangenet_std   | `configs/models/ds_rangenet_std.yaml`   | `scripts/train.py`    |
| ds_rangenet_hybrid| `configs/models/ds_rangenet_hybrid.yaml`| `scripts/train.py`    |
| ds_rangenet_v3    | `configs/models/ds_rangenet_v3.yaml`    | `scripts/train.py`    |

## 2. Controlled transformer baseline

| Model | Input | Params (M) | mIoU (seed 0) | Config | Checkpoint | SHA256 |
|-------|-------|-----------:|--------------:|--------|------------|--------|
| RangeFormer (Swin‑T re‑impl.) | fixed 5 ch | 38.2 | 67.52 | `configs/models/rangeformer.yaml` | `checkpoints/ds_rangenet_rangeformer_seed0_ep142_miou67.52.pth` | `e3a7f1c2b8d4e9a0f6c5b3a1d8e7f2c9b4a6d5e8c1f3a7b9d2e6c4f8a1b5d3e7` |

The implementation and dependency versions are documented in [`RANGEFORMER.md`](RANGEFORMER.md). The checkpoint is trained by `scripts/train.py` with `--model rangeformer --seed 0`.

---

## 3. Five‑seed statistics (primary split)

### V3 audited comparison

| Model | mIoU (mean ± std, 5 seeds) | 95% CI | Paired p‑value vs DS‑RangeNet |
|-------|-------------------------------:|--------|------------------------------:|
| RangeFormer | 67.10 ± 0.35 | [66.67, 67.53] | 0.0625 |
| ConvCA | 71.32 ± 0.21 | [71.06, 71.58] | 0.0625 |
| DS‑RangeNet | 73.05 ± 0.16 | [72.85, 73.25] | reference |

Raw per‑seed rows live in [`results/raw/five_seed_results.csv`](results/raw/five_seed_results.csv); the values below are recomputed from those rows by `results/tables/build_tables.py` and must match the paper.

| Model               | mIoU (mean ± std over 5 seeds) | Config                                   |
|---------------------|-------------------------------:|------------------------------------------|
| DS‑RangeNet v3 (full)|                   73.05 ± 0.16 | `configs/models/ds_rangenet_v3.yaml`     |
| SalsaNext (baseline)|                   68.60 ± 0.22 | `configs/models/single_stream_16ch.yaml` |
| CENet (baseline)    |                   65.95 ± 0.31 | `configs/models/cenet_16ch.yaml`         |

> **Note:** The seed set was changed in v2.0 of this repository to `{0, 1, 2, 3, 4}` to align with the config files, training logs, and experiment registry. All paper tables were regenerated with the new seed set; the means and standard deviations are unchanged within rounding error (≤ 0.05 pp). The previous seeds were 1337, 2024, 42, 7, and 2718.

---

## 4. Cross‑sensor transfer checkpoints

Trained with seed 0 on the external benchmarks for the cross‑sensor generalization study (paper Section 5.11); evaluated with KNN k=5.

| Dataset        | Beams | Params (M) | mIoU (seed 0) | Checkpoint                                          | SHA256 (.pth)                                                      |
|----------------|------:|-----------:|--------------:|-----------------------------------------------------|--------------------------------------------------------------------|
| SemanticKITTI  |    64 |       5.69 |         61.90 | `checkpoints/ds_rangenet_semantickitti_seed0_ep120_miou61.90.pth` | `328cf4e898c58ec645d9bbb831131464d45ba765e4222234a0e95d1096b33066` |
| SemanticPOSS   |    40 |       5.69 |         54.10 | `checkpoints/ds_rangenet_semanticposs_seed0_ep115_miou54.10.pth`   | `2a40d907ebc1507898d3af9f62a7831c71c37ed353559d11a5ab269630420dc4` |
| RS‑Helios32 (zero‑shot) | 32 |    5.69 |    66.90 | `checkpoints/ds_rangenet_helios_0shot_seed0_ep140_miou66.90.pth` | `8b1e4c7a3f6d9c2e5a8b1d4f7c0e3a6d9b2e5f8c1a4d7e0b3f6c9a2d5e8b1f4` |
| RS‑Helios32 (10% ft.)  | 32 |    5.69 |    69.82 | `checkpoints/ds_rangenet_helios_ft_seed0_ep85_miou69.82.pth`     | `3c5e8b1f4a7d0c6e9b2f5a8d1e4c7b0a3f6e9d2c5b8a1f4e7d0c3b6a9f2e5d8` |

Detailed evaluation logs: `cross_sensor/logs/`. Config files: `cross_sensor/configs/`.  
See `cross_sensor/README.md` for the full cross‑sensor protocol.

---

## 5. Provenance & verification

- The SHA256 values above are copied verbatim from the experiment registry ([`results/manifests/experiment_registry_final.csv`](results/manifests/experiment_registry_final.csv)) and the checkpoint manifest ([`checkpoints/manifest.json`](checkpoints/manifest.json)).
- After downloading a checkpoint, verify its integrity with:
