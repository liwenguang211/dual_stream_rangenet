#!/usr/bin/env python3
"""Train/evaluate DS-RangeNet across the five reported seeds and log raw mIoU.

Reviewer 1 requires the raw CSV for every seed. This driver fixes the seed,
builds the requested model variant, runs the user-supplied train/eval hooks, and
appends one row per (model, seed) to ``results/per_seed_miou.csv``.

The heavy lifting (dataset loading, optimization, official mIoU evaluation) is
project-specific, so it is injected through a callable rather than duplicated
here. This keeps the evidence-chain script small, auditable, and free of any
hidden or fabricated numbers.

Usage:
    python run_seeds.py --model full --split test \
        --seeds 1337 2024 42 7 2718 \
        --train-eval my_pipeline:train_and_eval

``--train-eval`` names ``module:function``. The function must have signature
``fn(model, seed, split) -> dict`` returning per-class IoU plus ``mIoU``. If no
hook is given, the script prints the deterministic setup it *would* run and
exits without writing numbers.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_PY = os.path.normpath(os.path.join(HERE, "..", "..", "python"))
if REPO_PY not in sys.path:
    sys.path.insert(0, REPO_PY)

CLASS_ORDER = [
    "background", "ground", "roof", "side_facade", "front_facade",
    "beam", "column", "window", "dynamic",
]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_hook(spec: str):
    module_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise SystemExit("--train-eval must be 'module:function'")
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def append_row(csv_path: str, model: str, seed: int, split: str, result: dict) -> None:
    header = ["model", "seed", "split", "mIoU"] + CLASS_ORDER
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        row = [model, seed, split, round(float(result["mIoU"]), 4)]
        row += [round(float(result[c]), 4) for c in CLASS_ORDER]
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="full", help="build_model variant name")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--train-eval", default=None, help="module:function train/eval hook")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "per_seed_miou.csv"))
    args = ap.parse_args()

    from ds_rangenet_v3 import build_model  # noqa: E402

    hook = load_hook(args.train_eval) if args.train_eval else None
    for seed in args.seeds:
        set_deterministic(seed)
        model = build_model(args.model)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[seed {seed}] variant={args.model} params={n_params:.3f}M deterministic=on")
        if hook is None:
            print("  no --train-eval hook given; not writing numbers (dry run).")
            continue
        result = hook(model, seed, args.split)
        append_row(args.out, args.model, seed, args.split, result)
        print(f"  mIoU={result['mIoU']:.4f} -> appended to {args.out}")


if __name__ == "__main__":
    main()
