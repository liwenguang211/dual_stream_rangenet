#!/usr/bin/env python3
"""Train a DS-RangeNet v3 model (or a controlled baseline) on a UBPC-9 split.

Checkpoint selection uses the best validation mIoU on the val site; no selection
or fine-tuning is ever done on a test site (train/default.yaml). The chosen
checkpoint's SHA256 is printed so it can be recorded in the experiment registry.

Usage:
    python scripts/train.py --model configs/models/ds_rangenet.yaml \
        --train configs/train/default.yaml --seed 1337

If configs/dataset/ubpc9.yaml's data_root has no downloaded sequences, pass
--smoke-test to run a couple of epochs on synthetic tensors of the correct
shape (16, H, W) instead, to sanity-check the loop itself (forward/loss/
backward/checkpoint save) without requiring the real dataset on disk.
"""
from __future__ import annotations

import argparse
import os

from _common import load_yaml, set_seed, git_commit, ensure_dir, sha256_of  # noqa: E402


def build_from_config(model_cfg: dict):
    """Dispatch to the right builder named in the model config."""
    import src.models as M
    builder = model_cfg["builder"].split(".")[-1]
    fn = getattr(M, builder)
    if builder == "build_model":
        return fn(variant=model_cfg.get("variant", "full"))
    return fn()


class _SmokeTestDataset:
    """Synthetic (16, H, W) tensors + label maps, for loop verification only."""

    def __init__(self, n: int, height: int, width: int, num_classes: int, seed: int):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.n = n
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.g = g

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        import torch
        tensor = torch.randn(16, self.height, self.width, generator=self.g)
        label = torch.randint(0, self.num_classes, (self.height, self.width),
                              generator=self.g)
        # sprinkle some ignore-label pixels, matching real data's out-of-projection gaps
        ignore_mask = torch.rand(self.height, self.width, generator=self.g) < 0.1
        label[ignore_mask] = 255
        return tensor, label


def build_loaders(ds_cfg: dict, train_cfg: dict, smoke_test: bool, seed: int):
    from torch.utils.data import DataLoader

    height, width = ds_cfg["resolution"]
    num_classes = ds_cfg["num_classes"]
    batch_size = train_cfg["batch_size"]

    if smoke_test:
        train_ds = _SmokeTestDataset(32, height, width, num_classes, seed)
        val_ds = _SmokeTestDataset(8, height, width, num_classes, seed + 1)
    else:
        from src.data import UBPC9Dataset
        train_ds = UBPC9Dataset(ds_cfg["data_root"], ds_cfg["splits"]["train"],
                                processed_dir=ds_cfg.get("processed_dir"),
                                height=height, width=width)
        val_ds = UBPC9Dataset(ds_cfg["data_root"], ds_cfg["splits"]["val"],
                              processed_dir=ds_cfg.get("processed_dir"),
                              height=height, width=width)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise RuntimeError(
                f"No frames found under data_root={ds_cfg['data_root']!r}. "
                f"train={len(train_ds)} val={len(val_ds)} frames. "
                "Point configs/dataset/ubpc9.yaml's data_root at the downloaded "
                "UBPC-9 sequences, or pass --smoke-test to verify the training "
                "loop with synthetic data instead."
            )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    return train_loader, val_loader


def make_scheduler(opt, epochs: int, warmup_epochs: int):
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    cosine = CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1))
    if warmup_epochs <= 0:
        return cosine
    warmup = LinearLR(opt, start_factor=1e-3, total_iters=warmup_epochs)
    return SequentialLR(opt, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="configs/models/ds_rangenet.yaml")
    ap.add_argument("--train", default="configs/train/default.yaml")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Use synthetic data to verify the training loop; "
                         "does not require the real UBPC-9 dataset on disk.")
    args = ap.parse_args()

    set_seed(args.seed)
    model_cfg = load_yaml(args.model)
    train_cfg_full = load_yaml(args.train)
    train_cfg = train_cfg_full["train"]
    ds_cfg = load_yaml(train_cfg_full["dataset"])
    epochs = args.epochs or train_cfg["epochs"]
    if args.smoke_test and args.epochs is None:
        epochs = 2

    import torch
    from torch import optim
    from src.models import CombinedLoss, NUM_CLASSES, CLASSES
    from src.metrics import SegmentationMetrics

    model = build_from_config(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loader, val_loader = build_loaders(ds_cfg, train_cfg, args.smoke_test,
                                             args.seed)

    opt = optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                      weight_decay=train_cfg["weight_decay"])
    scheduler = make_scheduler(opt, epochs, train_cfg.get("warmup_epochs", 0))
    criterion = CombinedLoss(alpha=0.6)
    ignore_label = ds_cfg.get("ignore_label", 255)

    best_val = -1.0
    ckpt_name = f"{model_cfg['name']}_seed{args.seed}.pth"
    ckpt_path = os.path.join(ensure_dir(args.out), ckpt_name)

    print(f"[train] model={model_cfg['name']} seed={args.seed} "
          f"epochs={epochs} device={device} commit={git_commit()} "
          f"smoke_test={args.smoke_test}")
    print(f"[train] optimizer=AdamW lr={train_cfg['lr']} loss=0.6*focal+0.4*dice")
    print(f"[train] checkpoint selection = best val mIoU (val site), "
          f"no test selection/finetuning")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for tensor, label in train_loader:
            tensor = tensor.to(device)
            label = label.to(device)
            opt.zero_grad()
            logits = model(tensor)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        metrics = SegmentationMetrics(NUM_CLASSES, class_names=CLASSES,
                                      ignore_label=ignore_label)
        with torch.no_grad():
            for tensor, label in val_loader:
                tensor = tensor.to(device)
                logits = model(tensor)
                pred = logits.argmax(dim=1).cpu().numpy()
                metrics.update(label.numpy(), pred)
        val_result = metrics.result()
        val_miou = val_result["mIoU"]

        print(f"[train] epoch {epoch:>3d}/{epochs} | train_loss {avg_train_loss:.4f} "
              f"| val_mIoU {val_miou:.2f}% | lr {opt.param_groups[0]['lr']:.2e}")

        if val_miou > best_val:
            best_val = val_miou
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_mIoU": val_miou,
                "seed": args.seed,
                "model_name": model_cfg["name"],
                "variant": model_cfg.get("variant", "full"),
            }, ckpt_path)

    if os.path.exists(ckpt_path):
        print(f"[train] best val_mIoU={best_val:.2f}% -> saved {ckpt_path} "
              f"sha256={sha256_of(ckpt_path)}")
    else:
        print("[train] no checkpoint saved (best_val never improved past -1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
