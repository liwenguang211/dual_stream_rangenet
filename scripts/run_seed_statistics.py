#!/usr/bin/env python3
"""Aggregate the 5-seed runs into per-model statistics + significance tests.

Reads the raw per-seed rows (results/raw/five_seed_results.csv, seeded from the
committed reproducibility/seeds/results/per_seed_miou.csv), computes mean/std per
model, and runs a paired significance test (t-test + Wilcoxon) of DS-RangeNet v3
against each baseline across the shared seeds. Writes
results/summaries/statistical_tests.csv. No number is hard-coded — everything is
computed from the raw rows.

Usage:
    python scripts/run_seed_statistics.py \
        --raw results/raw/five_seed_results.csv \
        --out results/summaries/statistical_tests.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

from _common import ensure_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="results/raw/five_seed_results.csv")
    ap.add_argument("--out", default="results/summaries/statistical_tests.csv")
    ap.add_argument("--reference", default="ds_rangenet_v3",
                    help="model compared against the others")
    args = ap.parse_args()

    import numpy as np
    try:
        from scipy import stats
        have_scipy = True
    except ImportError:
        have_scipy = False

    # read raw rows: expect columns model, seed, mIoU (+ per-class)
    by_model = defaultdict(dict)   # model -> {seed: mIoU}
    with open(args.raw, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_model[r["model"]][int(r["seed"])] = float(r["mIoU"])

    ref = args.reference
    if ref not in by_model:
        # allow common aliases
        for cand in ("ds_rangenet_v3", "full", "ds_rangenet"):
            if cand in by_model:
                ref = cand
                break

    ensure_dir(os.path.dirname(args.out))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_seeds", "mean_mIoU", "std_mIoU",
                    "vs_reference", "mean_delta", "t_p_value", "wilcoxon_p"])
        ref_seeds = by_model.get(ref, {})
        for model, seedmap in sorted(by_model.items()):
            vals = np.array(list(seedmap.values()), float)
            mean, std = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            if model == ref or not ref_seeds:
                w.writerow([model, len(vals), round(mean, 3), round(std, 3),
                            ref, 0.0, "-", "-"])
                continue
            shared = sorted(set(seedmap) & set(ref_seeds))
            a = np.array([ref_seeds[s] for s in shared])
            b = np.array([seedmap[s] for s in shared])
            delta = float((a - b).mean())
            if have_scipy and len(shared) >= 2:
                tp = float(stats.ttest_rel(a, b).pvalue)
                try:
                    wp = float(stats.wilcoxon(a, b).pvalue)
                except ValueError:
                    wp = float("nan")
            else:
                tp = wp = float("nan")
            w.writerow([model, len(vals), round(mean, 3), round(std, 3),
                        ref, round(delta, 3),
                        f"{tp:.3e}", f"{wp:.3e}"])
    print(f"[seed-stats] wrote {args.out} (reference={ref})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
