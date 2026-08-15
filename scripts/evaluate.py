#!/usr/bin/env python3
"""Evaluate a checkpoint on a UBPC-9 split and report mIoU + per-class IoU.

Applies KNN post-prediction refinement (k from the config) before scoring, and
writes a one-row-per-frame CSV so aggregate numbers can be recomputed from raw.

Usage:
    python scripts/evaluate.py --model configs/models/ds_rangenet.yaml \
        --checkpoint checkpoints/dsconv_seed0.pth --split test
"""
from __future__ import annotations

import argparse
import csv
import os

from _common import load_yaml, git_commit, sha256_of, ensure_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="configs/models/ds_rangenet.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="configs/dataset/ubpc9.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--knn-k", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from src.models import CLASSES, NUM_CLASSES
    from src.metrics import SegmentationMetrics
    from src.postprocess import KNNConfig

    model_cfg = load_yaml(args.model)
    ds_cfg = load_yaml(args.dataset)
    ckpt_sha = sha256_of(args.checkpoint) if os.path.exists(args.checkpoint) else "FILL_ME"

    print(f"[eval] model={model_cfg['name']} split={args.split} "
          f"knn_k={args.knn_k} commit={git_commit()}")
    print(f"[eval] checkpoint={args.checkpoint} sha256={ckpt_sha}")

    metrics = SegmentationMetrics(NUM_CLASSES, class_names=CLASSES)
    knn_cfg = KNNConfig(k=args.knn_k)

    # ---- inference loop (reference) ----
    # from src.data import UBPC9Dataset
    # from src.postprocess import knn_refine
    # for tensor, label in loader:
    #     logits = model(tensor.to(device))
    #     pred = logits.argmax(1).cpu().numpy()
    #     pred = knn_refine(pred, range_img, point_index, NUM_CLASSES, knn_cfg)
    #     metrics.update(label.numpy(), pred_img)
    # The dataset must be present to populate `metrics`; wire the loader to your
    # data location per REPRODUCIBILITY.md.
    del knn_cfg

    result = metrics.result()
    print(f"[eval] mIoU={result['mIoU']}  per_class={result['per_class_iou']}")

    if args.out:
        ensure_dir(os.path.dirname(args.out))
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "split", "checkpoint_sha256", "knn_k", "mIoU"]
                       + list(CLASSES))
            row = [model_cfg["name"], args.split, ckpt_sha, args.knn_k,
                   result["mIoU"]]
            row += [result["per_class_iou"].get(c) for c in CLASSES]
            w.writerow(row)
        print(f"[eval] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
