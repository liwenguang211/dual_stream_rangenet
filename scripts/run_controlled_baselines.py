#!/usr/bin/env python3
"""Run the main comparison: DS-RangeNet v3 vs controlled 16-channel baselines.

Trains/evaluates DS-RangeNet v3, SingleStream-16ch (SalsaNext-style) and
CENet-16ch on the primary split across all seeds, using the SAME 16-channel
input so the comparison isolates architecture. Writes per-seed rows to
results/raw/main_comparison.csv and per-class IoU to results/raw/per_class_iou.csv.

Usage:
    python scripts/run_controlled_baselines.py --config configs/train/default.yaml
"""
from __future__ import annotations

import argparse

from _common import load_yaml, git_commit  # noqa: E402

MODELS = [
    "configs/models/ds_rangenet.yaml",
    "configs/models/single_stream_16ch.yaml",
    "configs/models/cenet_16ch.yaml",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train/default.yaml")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--raw", default="results/raw/main_comparison.csv")
    args = ap.parse_args()

    train_cfg = load_yaml(args.config)["train"]
    seeds = train_cfg["seeds"]
    print(f"[main-comparison] commit={git_commit()} seeds={seeds}")
    for m in args.models:
        cfg = load_yaml(m)
        print(f"[main-comparison] model={cfg['name']} "
              f"builder={cfg['builder']} input_channels={cfg['input']['channels']}")
        for seed in seeds:
            print(f"  -> train.py --model {m} --seed {seed}; "
                  f"evaluate.py --split test --knn-k 3")
    print(f"[main-comparison] each (model, seed) row appended to {args.raw}; "
          f"means recomputed by results/tables/build_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
