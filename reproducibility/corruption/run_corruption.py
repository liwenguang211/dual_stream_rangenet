#!/usr/bin/env python3
"""Batch robustness evaluation under range-image corruptions.

Reviewer 1 asks for the corruption parameters and generation scripts behind the
robustness results. This script reads ``corruption_config.yaml``, applies each
corruption at each severity to the configured frames using a fixed per-frame
RNG, evaluates mIoU through a user-supplied hook, and writes one CSV row per
(model, seed, corruption, severity), plus a clean baseline row.

Corruptions are produced by ``ds_rangenet_v3.apply_corruption`` so the reported
robustness comes from the exact transform shipped in the model file.

Usage:
    python run_corruption.py --config corruption_config.yaml \
        --checkpoints-dir ../deployment/checkpoints \
        --frames-dir /path/to/ubpc9/test \
        --eval my_pipeline:eval_frames

``--eval`` names ``module:function`` with signature
``fn(model, frames, frame_ids) -> float`` returning mIoU. Without it, the script
prints the resolved corruption plan and exits without writing numbers.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_PY = os.path.normpath(os.path.join(HERE, "..", "..", "python"))
CKA_DIR = os.path.normpath(os.path.join(HERE, "..", "cka"))
for p in (REPO_PY, CKA_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_config(path: str) -> dict:
    from extract_cka import load_config as _lc  # reuse the yaml loader + fallback
    return _lc(path)


def load_hook(spec: str):
    module_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise SystemExit("--eval must be 'module:function'")
    return getattr(importlib.import_module(module_name), fn_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "corruption_config.yaml"))
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--eval", default=None, help="module:function mIoU eval hook")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_csv = args.out or os.path.join(HERE, cfg.get("output_csv", "results/corruption_results.csv"))
    kinds = list(cfg["corruptions"])
    print(f"variant={cfg['model_variant']} seeds={cfg['seeds']}")
    print(f"corruptions={kinds} severities={cfg['severities']}")

    if not (args.frames_dir and args.checkpoints_dir and args.eval):
        print("frames-dir/checkpoints-dir/eval not all given; dry run, no numbers written.")
        return

    import torch
    from ds_rangenet_v3 import build_model, apply_corruption

    hook = load_hook(args.eval)
    frame_ids = cfg["frame_ids"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    def load_frames():
        return [torch.load(os.path.join(args.frames_dir, f"frame_{i}.pt")).unsqueeze(0).float()
                for i in frame_ids]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "corruption", "severity", "mIoU", "mIoU_drop_vs_clean"])
        for seed in cfg["seeds"]:
            model = build_model(cfg["model_variant"]).eval()
            ckpt = os.path.join(args.checkpoints_dir, f"{cfg['model_variant']}_seed{seed}.pth")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            clean_frames = load_frames()
            clean_miou = float(hook(model, clean_frames, frame_ids))
            w.writerow([cfg["model_variant"], seed, "clean", 0, round(clean_miou, 4), 0.0])
            for kind in kinds:
                for sev in cfg["severities"]:
                    corrupted = []
                    for fid, x in zip(frame_ids, clean_frames):
                        g = torch.Generator().manual_seed(int(seed) + int(cfg["rng_offset"]) + int(fid))
                        corrupted.append(apply_corruption(x, kind, severity=float(sev), generator=g))
                    miou = float(hook(model, corrupted, frame_ids))
                    w.writerow([cfg["model_variant"], seed, kind, sev,
                                round(miou, 4), round(clean_miou - miou, 4)])
                    print(f"  seed={seed} {kind} sev={sev}: mIoU={miou:.4f} drop={clean_miou - miou:.4f}")
    print(f"Wrote {out_csv}.")


if __name__ == "__main__":
    main()
