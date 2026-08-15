"""
run_independent_models.py
============================
Independently trains 3 models from scratch:
  1. Geometry-only model    (single-stream baseline, 11-ch input)
  2. Reflectance-only model (single-stream baseline, 5-ch input)
  3. Early fusion model     (single-stream baseline, 16-ch input)

Each uses the paper's single-stream architecture (5.71 M params) and is
trained with 3 random seeds. If real UBPC-9 data is unavailable, the script
will still run using synthetic data but prints a warning.
"""

import sys, json
from pathlib import Path
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_model, CombinedLoss
from src.metrics import SegmentationMetrics
from torch.utils.data import DataLoader

# ============================================================
# 使用论文模型定义（确保参数 5.71 M）
# ============================================================
class GeometryOnlyModel(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # 使用 build_model 的 single_stream 变体，输入通道 11
        self.net = build_model(variant='single_stream', in_channels=11, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

class ReflectanceOnlyModel(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.net = build_model(variant='single_stream', in_channels=5, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

class EarlyFusionModel(torch.nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.net = build_model(variant='single_stream', in_channels=16, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

# 通道切片
_MODEL_CHANNEL_SLICE = {
    GeometryOnlyModel: slice(5, 16),
    ReflectanceOnlyModel: slice(0, 5),
    EarlyFusionModel: slice(0, 16),
}

# ============================================================
# 数据加载（优先真实数据，回退合成数据）
# ============================================================
def build_loaders(data_cfg, train_cfg, seed):
    height, width = data_cfg["resolution"]
    num_classes = data_cfg["num_classes"]
    batch_size = train_cfg["batch_size"]

    # 尝试加载真实 UBPC-9 数据集
    try:
        from src.data import UBPC9Dataset
        ds_cfg = yaml.safe_load(open(REPO_ROOT / "configs/dataset/ubpc9.yaml"))
        data_root = str(REPO_ROOT / ds_cfg["data_root"])
        processed_dir = ds_cfg.get("processed_dir")
        if processed_dir:
            processed_dir = str(REPO_ROOT / processed_dir)
        train_ds = UBPC9Dataset(data_root, str(REPO_ROOT / ds_cfg["splits"]["train"]),
                                processed_dir=processed_dir, height=height, width=width)
        val_ds = UBPC9Dataset(data_root, str(REPO_ROOT / ds_cfg["splits"]["val"]),
                              processed_dir=processed_dir, height=height, width=width)
        if len(train_ds) > 0 and len(val_ds) > 0:
            print("[run_independent_models] Using real UBPC-9 dataset.")
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
            return train_loader, val_loader
    except Exception as e:
        print(f"[Warning] Could not load UBPC-9 dataset: {e}")

    # 回退：合成数据
    print("[Warning] Falling back to synthetic smoke-test data. Results will not reflect real performance.")
    from src.data import SmokeTestDataset
    train_ds = SmokeTestDataset(32, height, width, num_classes, seed)
    val_ds = SmokeTestDataset(8, height, width, num_classes, seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

# ============================================================
# 训练与评估（与论文一致）
# ============================================================
def train_model(model, train_loader, val_loader, config, seed, log_file):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"],
                                  weight_decay=config["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    criterion = CombinedLoss(alpha=0.6)
    channel_slice = _MODEL_CHANNEL_SLICE[type(model)]
    best_miou = 0.0
    log_file.write(f"# Training log — seed={seed}\n")
    log_file.write(f"# {'Epoch':>5} {'TrainLoss':>10} {'ValmIoU':>8} {'LR':>10}\n")
    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        for tensor, label in train_loader:
            tensor = tensor[:, channel_slice, :, :].to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(tensor)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_miou = evaluate(model, val_loader, config, device, channel_slice)
        lr = optimizer.param_groups[0]["lr"]
        avg_loss = total_loss / len(train_loader)
        log_file.write(f"{epoch+1:>5d} {avg_loss:>10.4f} {val_miou:>8.2f} {lr:>10.2e}\n")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), str(REPO_ROOT / f"checkpoints/modality_{type(model).__name__}_seed{seed}.pth"))
        scheduler.step()
    log_file.write(f"# Best val mIoU (seed={seed}): {best_miou:.2f}%\n")
    return best_miou

def evaluate(model, dataloader, config, device, channel_slice):
    num_classes = config["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes, ignore_label=255)
    model.eval()
    with torch.no_grad():
        for tensor, label in dataloader:
            tensor = tensor[:, channel_slice, :, :].to(device)
            logits = model(tensor)
            pred = logits.argmax(dim=1).cpu().numpy()
            metrics.update(label.numpy(), pred)
    return metrics.result()["mIoU"]

# ============================================================
# 主入口
# ============================================================
def run_all(config_path="config.yaml"):
    config = yaml.safe_load(open(config_path))
    seeds = config["train"]["seeds"]
    log_dir = REPO_ROOT / "modality_controls" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    models = [
        ("geometry_only", GeometryOnlyModel, log_dir / "training_log_geometry_only.txt"),
        ("reflectance_only", ReflectanceOnlyModel, log_dir / "training_log_reflectance_only.txt"),
        ("early_fusion", EarlyFusionModel, log_dir / "training_log_early_fusion.txt"),
    ]
    results = {}
    for name, model_cls, log_path in models:
        scores = []
        with open(log_path, "w") as f:
            f.write(f"# Model: {name}\n# Config: {config_path}\n# Split: {config['data']['split']}\n")
            for seed in seeds:
                model = model_cls(num_classes=config["data"]["num_classes"])
                train_loader, val_loader = build_loaders(config["data"], config["train"], seed)
                score = train_model(model, train_loader, val_loader, config, seed, f)
                scores.append(score)
                print(f"  [{name}] seed={seed}: {score:.2f}%")
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        results[name] = {"mean": mean, "std": std, "raw": scores}
        print(f"  [{name}] mean={mean:.2f}%, std={std:.2f}pp")
    with open(log_dir / "independent_models.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Results saved to {log_dir / 'independent_models.json'}")

if __name__ == "__main__":
    run_all()