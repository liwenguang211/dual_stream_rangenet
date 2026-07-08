#!/usr/bin/env python3
"""Paired statistical-significance test over per-seed mIoU results.

Reviewer 1 requires multi-seed results with statistical significance. This
script reads ``results/per_seed_miou.csv`` (five seeds per model) and, for each
baseline, runs a paired comparison against the ``full`` DS-RangeNet model using
the *same* seeds (paired samples). It reports the mean gap, a paired t-test, and
a Wilcoxon signed-rank test.

Usage:
    python paired_significance.py \
        --csv results/per_seed_miou.csv \
        --reference full \
        --split test \
        --out results/significance_report.csv

    python paired_significance.py --summary   # regenerate per_seed_summary.csv

The script refuses to run on unfilled templates (cells still equal to
``FILL_ME``) so that no fabricated statistics are ever produced.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

PLACEHOLDER = "FILL_ME"
HERE = os.path.dirname(os.path.abspath(__file__))


def _read_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for r in reader:
            rows.append(r)
    return rows


def _load_miou(path: str, split: str) -> Dict[str, Dict[int, float]]:
    """Return {model: {seed: mIoU}} for the requested split."""
    table: Dict[str, Dict[int, float]] = defaultdict(dict)
    for r in _read_rows(path):
        if r.get("split") != split:
            continue
        val = r["mIoU"].strip()
        if val == PLACEHOLDER or val == "":
            raise SystemExit(
                f"Refusing to run: {path} still contains {PLACEHOLDER}. "
                "Fill the template with measured mIoU values first."
            )
        table[r["model"]][int(r["seed"])] = float(val)
    return table


def _paired(ref: Dict[int, float], other: Dict[int, float]) -> Tuple[List[float], List[float]]:
    seeds = sorted(set(ref) & set(other))
    if len(seeds) < 2:
        raise SystemExit("Need at least 2 shared seeds for a paired test.")
    return [ref[s] for s in seeds], [other[s] for s in seeds]


def _t_sf(t: float, df: int) -> float:
    """Two-sided survival function of Student-t via a numeric incomplete beta.

    Kept dependency-free on purpose so the script runs on a bare Jetson image.
    """
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    f = d
    for i in range(1, 200):
        m = i // 2
        if i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= d * c
        if abs(1.0 - d * c) < 1e-10:
            break
    return front * (f - 1.0)


def paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    if sd == 0.0:
        return math.inf if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    t = mean / (sd / math.sqrt(n))
    return t, _t_sf(abs(t), n - 1)


def wilcoxon(a: List[float], b: List[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value (normal approximation)."""
    diffs = [x - y for x, y in zip(a, b) if x != y]
    n = len(diffs)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0.0:
        return 1.0
    z = (w_plus - mean) / sd
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def run(csv_path: str, reference: str, split: str, out_path: str) -> None:
    table = _load_miou(csv_path, split)
    if reference not in table:
        raise SystemExit(f"Reference model {reference!r} not found in {csv_path}.")
    ref = table[reference]
    header = ["model", "n_pairs", "ref_mean", "model_mean", "mean_gap", "t_stat", "t_pvalue", "wilcoxon_pvalue"]
    out_rows = []
    for model, seeds in sorted(table.items()):
        if model == reference:
            continue
        a, b = _paired(ref, seeds)
        t, p = paired_t_test(a, b)
        w = wilcoxon(a, b)
        out_rows.append([
            model, len(a), round(statistics.mean(a), 4), round(statistics.mean(b), 4),
            round(statistics.mean(a) - statistics.mean(b), 4),
            round(t, 4), f"{p:.3e}", f"{w:.3e}",
        ])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(out_rows)
    print(f"Wrote {out_path} ({len(out_rows)} comparisons vs {reference}).")


def summarize(csv_path: str, out_path: str) -> None:
    rows = defaultdict(lambda: defaultdict(list))
    for r in _read_rows(csv_path):
        v = r["mIoU"].strip()
        if v in (PLACEHOLDER, ""):
            continue
        rows[(r["model"], r["split"])][0].append(float(v))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "n_seeds", "mIoU_mean", "mIoU_std", "mIoU_min", "mIoU_max"])
        for (model, split), buckets in sorted(rows.items()):
            vals = buckets[0]
            if not vals:
                continue
            w.writerow([
                model, split, len(vals), round(statistics.mean(vals), 4),
                round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                round(min(vals), 4), round(max(vals), 4),
            ])
    print(f"Wrote {out_path}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(HERE, "results", "per_seed_miou.csv"))
    ap.add_argument("--reference", default="full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "significance_report.csv"))
    ap.add_argument("--summary", action="store_true", help="Regenerate per_seed_summary.csv and exit.")
    args = ap.parse_args()
    if args.summary:
        summarize(args.csv, os.path.join(HERE, "results", "per_seed_summary.csv"))
        return
    run(args.csv, args.reference, args.split, args.out)


if __name__ == "__main__":
    main()
