#!/usr/bin/env python3
"""Train and evaluate DS-RangeNet with the canonical UBPC-9 9-fold LOEO split.

Each fold uses seven physical sites for training, one for validation, and one
for testing. A model is trained from scratch, selected only by validation mIoU,
and evaluated once on the held-out test site with KNN refinement (k=3).

Run from the repository root:
    python loeo_supporting_materials/run_loeo_9fold.py --folds 1,2,3,4,5,6,7,8,9
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._common import load_yaml, set_seed, sha256_of  # noqa: E402
from src.data import UBPC9Dataset  # noqa: E402
from src.metrics import SegmentationMetrics  # noqa: E402
from src.models import build_model, CombinedLoss, CLASSES, NUM_CLASSES  # noqa: E402
from src.postprocess.knn_refinement import KNNConfig, knn_refine  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def fold_partition(config: dict, fold_id: int) -> dict:
    fold = config["loeo_9fold"]["folds"][f"fold_{fold_id}"]
    train = set(fold["train_sites"])
    val = fold["val_site"]
    test = fold["test_site"]
    all_sites = set(config["sites"])
    if len(train) != 7 or train & {val, test} or val == test:
        raise ValueError(f"Fold {fold_id} is not a disjoint 7+1+1 partition")
    if train | {val, test} != all_sites:
        raise ValueError(f"Fold {fold_id} does not cover all nine sites")
    return {**fold, "train_sites": sorted(train)}


def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    if warmup_epochs <= 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=1e-3,
                      total_iters=min(warmup_epochs, epochs))
    return SequentialLR(optimizer, [warmup, cosine],
                        milestones=[min(warmup_epochs, epochs)])


def evaluate(model, loader, device, ignore_label: int, knn: KNNConfig | None):
    metrics = SegmentationMetrics(NUM_CLASSES, list(CLASSES), ignore_label)
    model.eval()
    with torch.no_grad():
        for tensor, label in loader:
            logits = model(tensor.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
            labels = label.numpy()
            for pred, gt, sample in zip(predictions, labels, tensor.numpy()):
                if knn is None:
                    metrics.update(gt, pred)
                    continue
                valid = gt != ignore_label
                point_index = np.full(gt.shape, -1, np.int64)
                point_index[valid] = np.arange(valid.sum(), dtype=np.int64)
                range_img = sample[0] * 50.0
                refined = knn_refine(pred, range_img, point_index,
                                     NUM_CLASSES, knn)
                metrics.update(gt[valid], refined)
    return metrics.result()


def build_datasets(data_cfg: dict, fold_file: Path):
    common = dict(
        data_root=data_cfg["data_root"],
        split_file=str(fold_file),
        processed_dir=data_cfg.get("processed_dir"),
        height=data_cfg["resolution"][0],
        width=data_cfg["resolution"][1],
    )
    datasets = {role: UBPC9Dataset(split_role=role, **common)
                for role in ("train", "val", "test")}
    missing = [role for role, dataset in datasets.items() if len(dataset) == 0]
    if missing:
        raise RuntimeError(
            f"Fold data not found for {missing}; data_root={data_cfg['data_root']!r}, "
            f"split={fold_file}. Download UBPC-9 or correct data_root."
        )
    return datasets


def train_fold(fold_id: int, partition: dict, fold_file: Path,
               data_cfg: dict, train_cfg: dict, args, output_dir: Path) -> dict:
    fold_seed = args.seed + fold_id - 1
    set_seed(fold_seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = build_datasets(data_cfg, fold_file)
    loaders = {
        role: DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=(role == "train"), num_workers=args.workers,
                         pin_memory=device.type == "cuda")
        for role, dataset in datasets.items()
    }

    model = build_model(variant="full").to(device)
    criterion = CombinedLoss(alpha=0.6)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=train_cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, args.epochs,
                               train_cfg.get("warmup_epochs", 0))
    checkpoint = output_dir / f"fold_{fold_id}_best.pth"
    best_val = -1.0
    best_epoch = 0

    print(f"\n[fold {fold_id}] train={partition['train_sites']} "
          f"val={partition['val_site']} test={partition['test_site']} seed={fold_seed}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for tensor, label in loaders["train"]:
            tensor, label = tensor.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tensor), label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()

        val_result = evaluate(model, loaders["val"], device,
                              data_cfg.get("ignore_label", 255), knn=None)
        val_miou = val_result["mIoU"]
        print(f"[fold {fold_id}] epoch {epoch:03d}/{args.epochs} "
              f"loss={total_loss / max(len(loaders['train']), 1):.4f} "
              f"val_mIoU={val_miou:.2f}")
        if val_miou > best_val:
            best_val, best_epoch = val_miou, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "fold": fold_id,
                "seed": fold_seed,
                "epoch": epoch,
                "val_mIoU": val_miou,
                "train_sites": partition["train_sites"],
                "val_site": partition["val_site"],
                "test_site": partition["test_site"],
            }, checkpoint)

    # The held-out test site is evaluated exactly once, after model selection.
    saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    knn_cfg = KNNConfig(k=args.knn_k, search=args.knn_search,
                        cutoff=args.knn_cutoff, sigma=args.knn_sigma)
    test_result = evaluate(model, loaders["test"], device,
                           data_cfg.get("ignore_label", 255), knn=knn_cfg)
    return {
        "fold": fold_id,
        "seed": fold_seed,
        "train_sites": ";".join(partition["train_sites"]),
        "val_site": partition["val_site"],
        "test_site": partition["test_site"],
        "train_frames": len(datasets["train"]),
        "val_frames": len(datasets["val"]),
        "test_frames": len(datasets["test"]),
        "best_epoch": best_epoch,
        "val_mIoU_no_knn": best_val,
        "test_mIoU": test_result["mIoU"],
        **{f"IoU_{name}": value
           for name, value in test_result["per_class_iou"].items()},
        "knn_k": args.knn_k,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_of(str(checkpoint)),
    }


def write_results(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "loeo_supporting_materials" /
                        "ubpc9_splits_v3.json")
    parser.add_argument("--dataset-config", type=Path,
                        default=REPO_ROOT / "configs/dataset/ubpc9.yaml")
    parser.add_argument("--train-config", type=Path,
                        default=REPO_ROOT / "configs/train/default.yaml")
    parser.add_argument("--splits-dir", type=Path,
                        default=REPO_ROOT / "data/splits/loeo")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "loeo_logs")
    parser.add_argument("--raw", type=Path,
                        default=REPO_ROOT / "results/raw/loeo_folds_measured.csv")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--knn-search", type=int, default=5)
    parser.add_argument("--knn-cutoff", type=float, default=1.0)
    parser.add_argument("--knn-sigma", type=float, default=1.0)
    args = parser.parse_args()
    train = load_yaml(str(args.train_config))["train"]
    args.epochs = args.epochs or train["epochs"]
    args.batch_size = args.batch_size or train["batch_size"]
    args.lr = args.lr or float(train["lr"])
    return args, train


def main() -> int:
    args, train_cfg = parse_args()
    config = load_json(args.config)
    data_cfg = load_yaml(str(args.dataset_config))
    # Resolve repository-relative data paths independently of the launch cwd.
    for key in ("data_root", "processed_dir"):
        if data_cfg.get(key) and not os.path.isabs(data_cfg[key]):
            data_cfg[key] = str(REPO_ROOT / data_cfg[key])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_ids = [int(value) for value in args.folds.split(",")]
    rows = []
    for fold_id in fold_ids:
        if fold_id not in range(1, 10):
            raise ValueError(f"Invalid fold: {fold_id}")
        partition = fold_partition(config, fold_id)
        fold_file = args.splits_dir / f"fold_S{fold_id}.yaml"
        rows.append(train_fold(fold_id, partition, fold_file, data_cfg,
                               train_cfg, args, args.output_dir))
        write_results(rows, args.raw)

    values = np.array([row["test_mIoU"] for row in rows], dtype=float)
    print(f"\n[LOEO] folds={len(rows)} macro_mIoU={values.mean():.2f}%")
    if len(values) > 1:
        print(f"[LOEO] sample_std={values.std(ddof=1):.2f} pp")
    print(f"[LOEO] measured rows written to {args.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
