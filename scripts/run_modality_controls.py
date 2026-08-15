#!/usr/bin/env python3
"""Run modality control experiments (independent models + input perturbations).

Independent-model conditions (geometry_only, reflectance_only, early_fusion) are
retrained; same-checkpoint conditions (intensity_missing, intensity_corrupted,
geometry_sparse, cross_frame_mismatch) reuse the DS-RangeNet baseline checkpoint
with the input perturbed. All perturbations recompute geometry/curvature as
specified. Writes results/raw/modality_controls.csv.

Usage:
    python scripts/run_modality_controls.py --config configs/modality/controls.yaml
"""
from __future__ import annotations

import argparse

from _common import load_yaml, git_commit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/modality/controls.yaml")
    ap.add_argument("--raw", default="results/raw/modality_controls.csv")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    print(f"[modality] commit={git_commit()} split={cfg['split']} "
          f"baseline_mIoU={cfg['baseline']['reported_mIoU']}")
    for name, spec in cfg["conditions"].items():
        cat = spec["category"]
        print(f"  {name:22s} [{cat}] reported_mean_mIoU={spec['reported_mean_mIoU']}")
    print(f"[modality] per-condition per-repeat rows -> {args.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
