# Reproducibility Evidence Chain

This directory provides the reproducibility evidence chain requested in review
(paper modification note, item 4). It contains runnable scripts, configuration
files, and result templates so that every reviewer requirement can be traced to
a concrete, re-runnable artifact.

Numeric result files (per-seed mIoU, Jetson telemetry, official evaluation
logs) are shipped as **templates with an explicit header/schema and empty data
rows**. They must be filled by running the provided scripts on the actual
trained models. Placeholder cells are marked `FILL_ME` and are intentionally
not populated with fabricated numbers.

## Directory layout

```text
reproducibility/
  README.md                     this file
  seeds/
    run_seeds.py                train/evaluate 5 seeds, write per-seed CSV
    paired_significance.py      paired t-test + Wilcoxon on per-seed CSVs
    results/
      per_seed_miou.csv         template: per-seed, per-class mIoU
      per_seed_summary.csv      template: mean/std per model over seeds
  cka/
    extract_cka.py              CKA / normalized cross-covariance extraction
    cka_config.yaml             layers, frame ids, token sampling, seeds
    results/
      cka_per_seed.csv          template: per-seed CKA/cross-cov per layer pair
  corruption/
    corruption_config.yaml      corruption kinds, severities, seeds
    run_corruption.py           batch corruption generation + evaluation
    results/
      corruption_results.csv    template: mIoU per corruption/severity/seed
  cross_sensor/
    semanticposs.yaml           SemanticPOSS projection + class mapping config
    semantickitti.yaml          SemanticKITTI projection + class mapping config
    checkpoint_manifest.csv     template: dataset/model/seed -> checkpoint + sha256
    eval_log_template.txt       official evaluator log format
  deployment/
    checkpoint_manifest.csv     Standard/Hybrid/DSConv checkpoint manifest
    jetson_telemetry.csv        template: 60-min telemetry, 1 Hz sampling
    log_jetson_telemetry.py     telemetry logger for Jetson (tegrastats parser)
    environment_manifest.yaml   TensorRT/CUDA/power-mode/clocks/ambient temp
```

## Reviewer requirement -> evidence mapping

| Reviewer requirement | Evidence artifact |
| --- | --- |
| R1: public dataset, multi-seed, statistical significance | `seeds/run_seeds.py`, `seeds/results/per_seed_miou.csv`, `seeds/paired_significance.py` |
| R1: modality complementarity (CKA / cross-covariance) | `cka/extract_cka.py`, `cka/cka_config.yaml`, `cka/results/cka_per_seed.csv` |
| R1: robustness under corruptions | `corruption/corruption_config.yaml`, `corruption/run_corruption.py`, `corruption/results/corruption_results.csv` |
| R2: baseline fairness, long-term deployment, source/weights | `deployment/checkpoint_manifest.csv`, `deployment/jetson_telemetry.csv`, `deployment/environment_manifest.yaml` |
| R2/R3: cross-sensor projection, holes, filling | `cross_sensor/semanticposs.yaml`, `cross_sensor/semantickitti.yaml`, `cross_sensor/eval_log_template.txt` |
| R3: DSConv accuracy-efficiency trade-off | `deployment/checkpoint_manifest.csv` (Standard / Hybrid / DSConv rows) |

## How to fill the templates

1. Train the reported models with the five seeds listed in `cka_config.yaml`
   and `corruption_config.yaml` (`{1337, 2024, 42, 7, 2718}`).
2. Run `seeds/run_seeds.py` to populate `seeds/results/per_seed_miou.csv`, then
   `seeds/paired_significance.py` to compute p-values against each baseline.
3. Run `cka/extract_cka.py` and `corruption/run_corruption.py` to populate their
   result CSVs.
4. On the Jetson device, run `deployment/log_jetson_telemetry.py` for 60 minutes
   and fill `deployment/environment_manifest.yaml` from `jetson_release` and
   `nvpmodel -q`.
5. Compute checkpoint SHA-256 sums and record them in the two
   `checkpoint_manifest.csv` files.

Every script accepts `--help` and writes only to its own `results/` folder.
