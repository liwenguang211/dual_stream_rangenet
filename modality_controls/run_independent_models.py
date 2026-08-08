"""
run_independent_models.py
============================
Independently trains 3 models from scratch:
  1. Geometry-only model  (5.71M params, 11-ch input, no reflectance)
  2. Reflectance-only model (5.71M params, 5-ch input, no geometry)
  3. Early fusion model    (5.71M params, 16-ch concat, single stream)

Each is trained with 3 random seeds for statistical significance.
Training logs are saved to logs/training_log_*.txt

IMPORTANT: These are SEPARATE models — they do NOT share the DS-RangeNet
checkpoint. This is explicitly noted in the paper table.
"""

import yaml, json, time, sys
import numpy as np
import torch
from pathlib import Path

# ============================================================
# Model definitions (stubs — replace with your actual architectures)
# ============================================================
class GeometryOnlyModel(torch.nn.Module):
    """11-channel input (geometry stream only), no reflectance."""
    def __init__(self, num_classes=9):
        super().__init__()
        # DSConv encoder (same as geometry branch of DS-RangeNet)
        # channels: 11 -> 32 -> 64 -> 128 -> 256
        # ASPP + decoder
        # Output: num_classes logits
        self.params = 5.71e6  # match paper
        raise NotImplementedError("Plug in your Geometry-only architecture")

class ReflectanceOnlyModel(torch.nn.Module):
    """5-channel input (reflectance stream only), no geometry."""
    def __init__(self, num_classes=9):
        super().__init__()
        # DSConv encoder (same as reflectance branch of DS-RangeNet)
        # channels: 5 -> 32 -> 64 -> 128 -> 256
        self.params = 5.71e6
        raise NotImplementedError("Plug in your Reflectance-only architecture")

class EarlyFusionModel(torch.nn.Module):
    """16-channel concatenated input, single stream (no dual-stream separation)."""
    def __init__(self, num_classes=9):
        super().__init__()
        # Single DSConv encoder: 16 -> 32 -> 64 -> 128 -> 256
        # No CBAM between streams, no IGCA
        self.params = 5.71e6
        raise NotImplementedError("Plug in your Early-fusion architecture")

# ============================================================
# Training routine (shared)
# ============================================================
def train_model(model, train_loader, val_loader, config, seed, log_file):
    """
    Standard training loop.
    Returns best validation mIoU across 150 epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config['train']['device'])
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['train']['epochs']
    )

    best_miou = 0.0
    log_file.write(f"# Training log — seed={seed}\n")
    log_file.write(f"# {'Epoch':>5} {'TrainLoss':>10} {'ValmIoU':>8} {'LR':>10}\n")

    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['input'])
            loss = compute_loss(logits, batch['labels'], config)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        val_miou = evaluate(model, val_loader)
        lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(train_loader)
        log_file.write(f"{epoch+1:>5d} {avg_loss:>10.4f} {val_miou:>8.2f} {lr:>10.2e}\n")
        log_file.flush()

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), f"checkpoints/seed{seed}_best.pth")

        scheduler.step()

    log_file.write(f"# Best val mIoU (seed={seed}): {best_miou:.2f}%\n")
    return best_miou

def compute_loss(logits, labels, config):
    """0.6 * Focal + 0.4 * Dice. Stub — replace with your implementation."""
    raise NotImplementedError

def evaluate(model, dataloader):
    """Compute mean IoU over validation set. Stub."""
    raise NotImplementedError

# ============================================================
# Main
# ============================================================
def run_all(config_path="config.yaml"):
    config = yaml.safe_load(open(config_path))
    seeds = config['train']['seeds']           # [0, 1, 2]
    results = {}

    models_to_train = [
        ('geometry_only',   GeometryOnlyModel,   'logs/training_log_geometry_only.txt'),
        ('reflectance_only', ReflectanceOnlyModel, 'logs/training_log_reflectance_only.txt'),
        ('early_fusion',    EarlyFusionModel,    'logs/training_log_early_fusion.txt'),
    ]

    for name, model_cls, log_path in models_to_train:
        scores = []
        log_file = open(log_path, 'w')
        log_file.write(f"# {'='*60}\n")
        log_file.write(f"# Model: {name}\n")
        log_file.write(f"# Config: {config_path}\n")
        log_file.write(f# Split: {config['data']['split']} (validation set)\n")
        log_file.write(f"# {'='*60}\n")

        for seed in seeds:
            torch.manual_seed(seed)
            model = model_cls(num_classes=config['data']['num_classes'])
            score = train_model(model, None, None, config, seed, log_file)
            scores.append(score)
            print(f"  [{name}] seed={seed}: {score:.2f}%")

        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        results[name] = {'mean': mean, 'std': std, 'raw': scores}
        log_file.write(f"# FINAL: mean={mean:.2f}%, std={std:.2f}pp\n")
        log_file.close()
        print(f"  [{name}] mean={mean:.2f}%, std={std:.2f}pp")

    # Save summary
    with open('logs/independent_models.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone. Results saved to logs/independent_models.json")

if __name__ == '__main__':
    run_all()
