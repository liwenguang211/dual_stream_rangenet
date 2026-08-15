#!/usr/bin/env python3
"""Run the 9-fold LOEO (Leave-One-Environment-Out) cross-validation.

For each fold:
  1. Load the UBPC-9 dataset split (7 train / 1 val / 1 test site, mutually disjoint).
  2. Build the real DS-RangeNet model (build_model from src.models).
  3. Train with the paper's training recipe (AdamW + CosineAnnealing + CombinedLoss).
  4. Select the best validation checkpoint.
  5. Evaluate on the held-out test site with KNN post-processing (k=3).
  6. Compute per-class IoU from the confusion matrix.

Results are written to:
  - results/raw/loeo_folds.csv        (one row per fold, detailed)
  - results/summaries/loeo_summary.csv (fold summary + macro avg + SD + drop)
  - results/raw/loeo_perclass_details.csv (per-class IoU across folds)

The macro-average, sample SD, and drop-from-primary are computed here and
recomputed by build_tables.py for paper Table 5.

Usage:
    python scripts/run_loeo.py --config configs/loeo/nine_folds.yaml
    python scripts/run_loeo.py --config configs/loeo/nine_folds.yaml --output_dir results
    python scripts/run_loeo.py --skip_training   # use existing checkpoints only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# Torch imports (required — no fallback / no simulation mode)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ============================================================
# Project imports (real model + data + metrics + losses + KNN)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_model           # DS-RangeNet (5.69M params)
from src.data import UBPC9Dataset            # Real UBPC-9 point-cloud dataset
from src.metrics import SegmentationMetrics  # Confusion matrix -> per-class IoU
from src.losses import CombinedLoss         # CE + Dice (paper exact)
from src.knn import knn_postprocess        # KNN k=3 post-prediction refinement

# ============================================================
# Constants
# ============================================================

CLASSES = ['background', 'ground', 'roof', 'side_facade',
           'front_facade', 'dynamic', 'beam', 'column', 'window']
NUM_CLASSES = len(CLASSES)
IGNORE_LABEL = 255

# Paper reference values (for the final consistency check)
PAPER_PRIMARY_MIOU = 73.20      # Line T
PAPER_LOEO_MACRO  = 70.96       # Line L (Table 5 macro avg)
PAPER_LOEO_SD      = 0.52        # sample SD across 9 folds
PAPER_LOEO_DROP    = 2.24        # 73.20 - 70.96


# ============================================================
# Utilities
# ============================================================

def load_yaml(path):
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit():
    """Return current git commit hash (short)."""
    try:
        import subprocess
        r = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return 'unknown'


def set_seed(seed: int):
    """Deterministic seeding for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================================
# Dataset construction (real UBPC-9)
# ============================================================

def build_fold_datasets(fold_cfg: dict, data_cfg: dict, fold_id: int):
    """
    Build train / val / test datasets for one LOEO fold.

    Parameters
    ----------
    fold_cfg : dict
        Config for this fold (train_sites, val_site, test_site, test_frames, ...).
    data_cfg : dict
        Global data config (data_root, processed_dir, resolution, num_classes, ...).
    fold_id : int
        Fold number (1-9) for seeding.

    Returns
    -------
    train_ds, val_ds, test_ds : UBPC9Dataset
    """
    data_root = str(REPO_ROOT / data_cfg.get("data_root", "data/ubpc9"))
    processed_dir = data_cfg.get("processed_dir")
    if processed_dir:
        processed_dir = str(REPO_ROOT / processed_dir)

    height = data_cfg["resolution"][0]
    width  = data_cfg["resolution"][1]

    train_sites = fold_cfg.get("train_sites", [])
    val_site   = fold_cfg.get("val_site", "")
    test_site  = fold_cfg.get("test_site", "")

    # UBPC9Dataset accepts a list of site names for the split
    train_ds = UBPC9Dataset(
        data_root=data_root,
        split="train",
        sites=train_sites,
        processed_dir=processed_dir,
        height=height,
        width=width,
        seed=1000 + fold_id,
    )
    val_ds = UBPC9Dataset(
        data_root=data_root,
        split="val",
        sites=[val_site],
        processed_dir=processed_dir,
        height=height,
        width=width,
        seed=2000 + fold_id,
    )
    test_ds = UBPC9Dataset(
        data_root=data_root,
        split="test",
        sites=[test_site],
        processed_dir=processed_dir,
        height=height,
        width=width,
        seed=3000 + fold_id,
    )

    return train_ds, val_ds, test_ds


# ============================================================
# Model construction (real DS-RangeNet)
# ============================================================

def build_fold_model(model_cfg: dict, device: torch.device) -> nn.Module:
    """
    Build the real DS-RangeNet using the project's model factory.

    The factory is called with the paper-exact configuration:
      variant='dual_stream', dsconv=True, cbam=True,
      in_channels=16, num_classes=9.

    This produces the 5.69M-parameter model used in the paper.
    """
    model = build_model(
        variant=model_cfg.get("variant", "dual_stream"),
        in_channels=model_cfg.get("in_channels", 16),
        num_classes=model_cfg.get("num_classes", NUM_CLASSES),
        dsconv=model_cfg.get("dsconv", True),
        cbam=model_cfg.get("cbam", True),
        aspp_rates=model_cfg.get("aspp_rates", [1, 6, 12, 18]),
    )
    model.to(device)
    return model


# ============================================================
# Training one fold
# ============================================================

def train_fold(
    fold_id: int,
    fold_cfg: dict,
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
    skip_training: bool = False,
) -> tuple[nn.Module, int, float]:
    """
    Train one LOEO fold.

    Returns
    -------
    model : nn.Module
        Model loaded with the best validation checkpoint.
    best_epoch : int
        Epoch with highest validation mIoU.
    best_val_miou : float
        Best validation mIoU (%).
    """
    set_seed(1000 + fold_id)

    # ---- Data ----
    train_ds, val_ds, _ = build_fold_datasets(fold_cfg, data_cfg, fold_id)
    batch_size = train_cfg.get("batch_size", 8)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
    )

    print(f"  [fold {fold_id}] train={len(train_ds)} frames, "
          f"val={len(val_ds)} frames, bs={batch_size}")

    # ---- Checkpoint path ----
    ckpt_path = checkpoint_dir / f"loeo_fold{fold_id}_best.pth"

    # ---- Skip training if requested and checkpoint exists ----
    if skip_training and ckpt_path.exists():
        print(f"  [fold {fold_id}] SKIP training (checkpoint exists: {ckpt_path})")
        model = build_fold_model(model_cfg, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = ckpt.get("epoch", 0)
        best_val_miou = ckpt.get("val_miou", 0.0)
        return model, best_epoch, best_val_miou

    # ---- Model / optim / scheduler / loss ----
    model = build_fold_model(model_cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.get("epochs", 150),
    )
    criterion = CombinedLoss(alpha=train_cfg.get("loss_alpha", 0.6))

    epochs = train_cfg.get("epochs", 150)
    best_val = 0.0
    best_ep = 0

    metrics = SegmentationMetrics(NUM_CLASSES, ignore_label=IGNORE_LABEL)

    for ep in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for tensor, label in train_loader:
            tensor = tensor.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(tensor)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # ---- Validation (real evaluation) ----
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for tensor, label in val_loader:
                tensor = tensor.to(device)
                logits = model(tensor)
                pred = logits.argmax(dim=1).cpu().numpy()
                metrics.update(label.numpy(), pred)

        val_result = metrics.result()
        val_miou = val_result["mIoU"]

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  [fold {fold_id}] ep {ep:>3d}  "
              f"loss={avg_train_loss:.4f}  val_mIoU={val_miou:.2f}%  lr={lr_now:.2e}")

        if val_miou > best_val:
            best_val = val_miou
            best_ep = ep
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": ep,
                "val_miou": val_miou,
                "fold_id": fold_id,
                "git_commit": git_commit(),
            }, ckpt_path)
            print(f"  [fold {fold_id}]  ✓ new best -> {ckpt_path}")

    print(f"  [fold {fold_id}] BEST ep={best_ep} val_mIoU={best_val:.2f}%")

    # Reload best checkpoint
    model = build_fold_model(model_cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    return model, best_ep, best_val


# ============================================================
# Evaluate one fold on the held-out test site
# ============================================================

def evaluate_fold(
    fold_id: int,
    fold_cfg: dict,
    data_cfg: dict,
    model: nn.Module,
    device: torch.device,
    knn_k: int,
) -> dict:
    """
    Evaluate the trained model on the held-out test site.

    Steps:
      1. Forward pass -> logits -> argmax predictions.
      2. KNN post-processing (k=3) to refine predictions.
      3. Accumulate confusion matrix -> per-class IoU.

    Returns
    -------
    dict with fold results (test_miou, per_class_iou, test_frames, ...)
    """
    _, _, test_ds = build_fold_datasets(fold_cfg, data_cfg, fold_id)
    batch_size = data_cfg.get("batch_size", 8)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
    )

    print(f"  [fold {fold_id}] EVAL  test={fold_cfg.get('test_site','?')}  "
          f"frames={len(test_ds)}  knn_k={knn_k}")

    metrics = SegmentationMetrics(NUM_CLASSES, ignore_label=IGNORE_LABEL)
    model.eval()

    with torch.no_grad():
        for tensor, label in test_loader:
            tensor = tensor.to(device)
            logits = model(tensor)
            pred = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)

            # KNN post-processing (paper exact: k=3)
            if knn_k > 0:
                pred = knn_postprocess(pred, k=knn_k)

            metrics.update(label.numpy(), pred)

    result = metrics.result()
    test_miou = result["mIoU"]
    per_class = result["per_class_iou"]   # list of 9 floats

    print(f"  [fold {fold_id}] >>> Test mIoU (held-out) = {test_miou:.2f}%")
    pc_str = ", ".join(f"{c}={v:.1f}" for c, v in zip(CLASSES, per_class))
    print(f"  [fold {fold_id}] >>> Per-class IoU: {pc_str}")

    return {
        "fold": fold_id,
        "test_site": fold_cfg.get("test_site", ""),
        "test_environment": fold_cfg.get("test_environment", ""),
        "val_site": fold_cfg.get("val_site", ""),
        "train_sites": fold_cfg.get("train_sites", []),
        "test_frames": len(test_ds),
        "test_miou": round(test_miou, 2),
        "per_class_iou": [round(v, 2) for v in per_class],
    }


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/loeo/nine_folds.yaml")
    ap.add_argument("--raw", default="results/raw/loeo_folds.csv")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from config")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Override batch_size from config")
    ap.add_argument("--checkpoint_dir", default="checkpoints/loeo")
    ap.add_argument("--skip_training", action="store_true",
                    help="Skip training if checkpoint already exists")
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9")
    args = ap.parse_args()

    # ---- Load config ----
    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        return 1

    config = load_yaml(args.config)

    data_cfg   = config.get("data", {})
    model_cfg  = config.get("model", {})
    train_cfg  = config.get("train", {})
    knn_cfg    = config.get("knn", {})

    # CLI overrides
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    epochs   = train_cfg.get("epochs", 150)
    batch_sz = train_cfg.get("batch_size", 8)
    knn_k    = knn_cfg.get("k", 3)

    commit = git_commit()
    protocol = config.get("protocol", "7_train + 1_validation + 1_test")

    print(f"{'='*60}")
    print(f"  DS-RangeNet — 9-Fold LOEO Cross-Validation")
    print(f"{'='*60}")
    print(f"  commit={commit}")
    print(f"  protocol={protocol}")
    print(f"  epochs={epochs}  batch_size={batch_sz}  knn_k={knn_k}")
    print(f"  device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  skip_training={args.skip_training}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_ids = [int(x) for x in args.folds.split(",")]

    # ---- Prepare directories ----
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.raw).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run each fold ----
    results = []
    t0 = time.time()

    for fid in fold_ids:
        fold_key = f"fold_{fid}"
        if fold_key not in config.get("folds", {}):
            print(f"  [fold {fid}] SKIPPED (not in config)")
            continue

        fold_cfg = config["folds"][fold_key]

        print(f"\n{'='*60}")
        print(f"  FOLD {fid} — Test: {fold_cfg.get('test_site','?')} "
              f"({fold_cfg.get('test_environment','?')})")
        print(f"  Train sites: {fold_cfg.get('train_sites', [])}")
        print(f"  Val site:    {fold_cfg.get('val_site','?')}")

        # Train
        model, best_ep, best_val = train_fold(
            fold_id=fid,
            fold_cfg=fold_cfg,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            device=device,
            checkpoint_dir=checkpoint_dir,
            skip_training=args.skip_training,
        )

        # Evaluate on held-out test site
        eval_result = evaluate_fold(
            fold_id=fid,
            fold_cfg=fold_cfg,
            data_cfg=data_cfg,
            model=model,
            device=device,
            knn_k=knn_k,
        )
        eval_result["best_epoch"] = best_ep
        eval_result["best_val_miou"] = round(best_val, 2)
        results.append(eval_result)

    elapsed = time.time() - t0

    # ============================================================
    # Write outputs
    # ============================================================

    # ---- 1. Raw detailed CSV (every fold) ----
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "test_site", "test_environment", "val_site",
            "train_sites", "test_frames", "best_epoch",
            "best_val_miou", "test_miou", "per_class_iou",
            "mode", "elapsed_s",
        ])
        for r in results:
            writer.writerow([
                r["fold"], r["test_site"], r["test_environment"],
                r["val_site"], "_".join(r["train_sites"]),
                r["test_frames"], r["best_epoch"],
                r["best_val_miou"], r["test_miou"],
                ";".join(f"{v:.1f}" for v in r["per_class_iou"]),
                "real-trained", f"{elapsed:.1f}",
            ])

    # ---- 2. Summary statistics ----
    miou_list = [r["test_miou"] for r in results]
    macro = float(np.mean(miou_list))
    sd = float(np.std(miou_list, ddof=1))
    drop = PAPER_PRIMARY_MIOU - macro

    # ---- 3. Summary CSV ----
    summary_path = output_dir / "loeo_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "test_site", "test_environment",
            "val_site", "train_sites", "test_miou",
        ])
        for r in results:
            writer.writerow([
                r["fold"], r["test_site"], r["test_environment"],
                r["val_site"], "_".join(r["train_sites"]), r["test_miou"],
            ])
        writer.writerow(["macro_avg", "", "", "", "", f"{macro:.2f}"])
        writer.writerow(["sample_sd_pp", "", "", "", "", f"{sd:.2f}"])
        writer.writerow(["drop_from_primary_pp", "", "", "", "", f"{drop:.2f}"])

    # ---- 4. Per-class details CSV ----
    pc_path = output_dir / "loeo_perclass_details.csv"
    with open(pc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["class"]
            + [f"fold_{r['fold']}" for r in results]
            + ["loeo_avg_iou", "primary_iou", "delta_pp"]
        )

        # Paper primary per-class IoU (Line T, Table 7)
        primary_vals = [88.50, 89.60, 77.10, 78.00, 73.00,
                        61.38, 60.20, 63.50, 66.80]

        for ci, cls_name in enumerate(CLASSES):
            row = [cls_name]
            fold_vals = [r["per_class_iou"][ci] for r in results]
            row += [f"{v:.2f}" for v in fold_vals]
            avg = float(np.mean(fold_vals))
            delta = avg - primary_vals[ci]
            row += [f"{avg:.2f}", f"{primary_vals[ci]:.2f}", f"{delta:.2f}"]
            writer.writerow(row)

        # Macro row
        all_avgs = []
        for ci in range(NUM_CLASSES):
            all_avgs.append(np.mean([r["per_class_iou"][ci] for r in results]))
        macro_pc = float(np.mean(all_avgs))
        writer.writerow([
            "macro_avg", "", "", "", "", "", "", "", "",
            f"{macro_pc:.2f}", f"{PAPER_PRIMARY_MIOU:.2f}",
            f"{macro_pc - PAPER_PRIMARY_MIOU:.2f}",
        ])

    # ============================================================
    # Print summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY — 9-Fold LOEO")
    print(f"{'='*60}")
    for r in results:
        print(f"  Fold {r['fold']} ({r['test_site']:>2s}): "
              f"test_mIoU={r['test_miou']:.2f}%  "
              f"val_best={r['best_val_miou']:.2f}% @ ep {r['best_epoch']}")
    print(f"{'-'*60}")
    print(f"  Macro avg mIoU:  {macro:.2f}%   (paper: {PAPER_LOEO_MACRO:.2f})")
    print(f"  Sample SD:       {sd:.2f} pp    (paper: {PAPER_LOEO_SD:.2f})")
    print(f"  Drop from 73.2%: {drop:.2f} pp    (paper: {PAPER_LOEO_DROP:.2f})")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print(f"{'='*60}")

    print(f"\n[OK] Raw folds CSV    -> {raw_path}")
    print(f"[OK] Summary CSV       -> {summary_path}")
    print(f"[OK] Per-class CSV     -> {pc_path}")

    # ============================================================
    # Paper consistency checks
    # ============================================================
    tol_macro = 0.10      # 0.10 pp tolerance on macro
    tol_sd    = 0.10      # 0.10 pp tolerance on SD
    tol_drop  = 0.10

    checks = [
        (abs(macro - PAPER_LOEO_MACRO) < tol_macro,
         f"Macro {macro:.2f} ~ paper {PAPER_LOEO_MACRO:.2f}  "
         f"(diff={abs(macro-PAPER_LOEO_MACRO):.3f})"),
        (abs(sd - PAPER_LOEO_SD) < tol_sd,
         f"SD {sd:.2f} ~ paper {PAPER_LOEO_SD:.2f}  "
         f"(diff={abs(sd-PAPER_LOEO_SD):.3f})"),
        (abs(drop - PAPER_LOEO_DROP) < tol_drop,
         f"Drop {drop:.2f} ~ paper {PAPER_LOEO_DROP:.2f}  "
         f"(diff={abs(drop-PAPER_LOEO_DROP):.3f})"),
    ]

    print(f"\n{'='*60}")
    print(f"  PAPER CONSISTENCY CHECKS")
    print(f"{'='*60}")
    all_pass = True
    for ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {msg}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n  >>> ALL CHECKS PASSED — results match paper values <<<")
    else:
        print(f"\n  >>> WARNING: some values differ from paper <<<")

    # ---- Write verification report ----
    report = {
        "macro_avg": round(macro, 2),
        "paper_macro": PAPER_LOEO_MACRO,
        "macro_diff": round(macro - PAPER_LOEO_MACRO, 3),
        "sample_sd": round(sd, 2),
        "paper_sd": PAPER_LOEO_SD,
        "sd_diff": round(sd - PAPER_LOEO_SD, 3),
        "drop_from_primary": round(drop, 2),
        "paper_drop": PAPER_LOEO_DROP,
        "drop_diff": round(drop - PAPER_LOEO_DROP, 3),
        "all_pass": all_pass,
        "elapsed_s": round(elapsed, 1),
        "git_commit": commit,
    }
    report_path = output_dir / "loeo_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Verification report -> {report_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
