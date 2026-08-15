#!/usr/bin/env python3
"""Preprocess UBPC-9: raw point clouds -> cached 16-channel range images.

Reads the dataset config, iterates the frames referenced by the split files,
builds the (16, H, W) tensor + projected label map with
src.data.build_16ch_input, and writes one .npz per frame under processed_dir.

Usage:
    python scripts/preprocess_ubpc9.py --config configs/dataset/ubpc9.yaml \
        --out data/processed
"""
from __future__ import annotations

import argparse
import os

from _common import load_yaml, ensure_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/dataset/ubpc9.yaml")
    ap.add_argument("--out", default=None, help="override processed_dir")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #frames")
    args = ap.parse_args()

    import numpy as np
    from src.data import build_16ch_input
    from src.data.ubpc9_dataset import _read_split

    cfg = load_yaml(args.config)
    data_root = cfg["data_root"]
    out_dir = ensure_dir(args.out or cfg["processed_dir"])
    H, W = cfg["resolution"]
    max_range = float(cfg.get("max_range_m", 50.0))

    all_seqs = []
    for key in ("train", "val", "test"):
        sf = cfg["splits"][key]
        if os.path.exists(sf):
            all_seqs += _read_split(sf)

    done = 0
    for seq in all_seqs:
        vel_dir = os.path.join(data_root, seq.split("/", 1)[0], "velodyne")
        if not os.path.isdir(vel_dir):
            print(f"[skip] sequence not present: {seq}")
            continue
        for fn in sorted(os.listdir(vel_dir)):
            if not fn.endswith(".bin"):
                continue
            stem = os.path.splitext(fn)[0]
            raw = np.fromfile(os.path.join(vel_dir, fn), np.float32).reshape(-1, 4)
            built = build_16ch_input(raw[:, :3], raw[:, 3], H, W, max_range)
            out_path = os.path.join(out_dir, seq, stem + ".npz")
            ensure_dir(os.path.dirname(out_path))
            np.savez_compressed(out_path, tensor=built["tensor"],
                                point_index=built["point_index"],
                                mask=built["mask"])
            done += 1
            if args.limit and done >= args.limit:
                print(f"[done] preprocessed {done} frames (limit reached)")
                return 0
    print(f"[done] preprocessed {done} frames -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
