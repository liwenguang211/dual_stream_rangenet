"""
apply_perturbations.py
============================
Applies 4 inference-time perturbations to a FIXED DS-RangeNet checkpoint.
All perturbations use the SAME model weights; only inputs are modified.

Key design decisions (per reviewer feedback):
  - Intensity formula: I' = clip(1.2*I + eps, 0, 1)
  - Cross-frame: offset frames from the SAME sequence (not different scenes)
  - Geometry sparse: voxel-PCA and curvature are REcomputed after point removal
  - Each perturbation is repeated 3 times with different random seeds
"""

import numpy as np
import torch
import yaml
import os, sys, time
from pathlib import Path

# ============================================================
# 1. Intensity missing — zero-mask reflectance channels
# ============================================================
def apply_intensity_missing(batch, config):
    """
    Zero-mask the 5 reflectance channels (indices 0-4) in the full model.
    Geometry channels (5-15) are kept unchanged.
    """
    REFLECTANCE_CHANNELS = slice(0, 5)
    perturbed = batch.clone()
    perturbed[:, REFLECTANCE_CHANNELS, :, :] = 0.0
    return perturbed

# ============================================================
# 2. Intensity corrupted — shift + Gaussian noise, clipped
# ============================================================
def apply_intensity_corrupted(batch, config, seed):
    """
    I' = clip(1.2 * I + eps, 0, 1)
    where eps ~ N(0, 0.05^2) in the normalized [0,1] intensity range.
    Only applied to reflectance channels (0-4).
    """
    rng = np.random.default_rng(seed)
    REFLECTANCE_CHANNELS = slice(0, 5)
    CLIP_MIN, CLIP_MAX = 0.0, 1.0
    SHIFT = config['perturbations']['intensity_corrupted']['shift_factor']
    NOISE_STD = config['perturbations']['intensity_corrupted']['noise_std']

    perturbed = batch.clone()
    refl = perturbed[:, REFLECTANCE_CHANNELS, :, :]

    # Generate noise on the same device/dtype as the input
    noise = torch.tensor(
        rng.normal(0, NOISE_STD, size=refl.shape),
        dtype=refl.dtype, device=refl.device
    )

    shifted = SHIFT * refl + noise
    clipped = torch.clamp(shifted, CLIP_MIN, CLIP_MAX)
    perturbed[:, REFLECTANCE_CHANNELS, :, :] = clipped
    return perturbed

# ============================================================
# 3. Geometry sparse — random point removal + REcompute geometry
# ============================================================
def apply_geometry_sparse(points, config, seed):
    """
    Randomly remove 30% of points, then RECOMPUTE voxel-PCA and curvature.
    This ensures the perturbation is complete (not just removing input points
    while keeping stale descriptors).
    """
    rng = np.random.default_rng(seed)
    REMOVE_FRAC = config['perturbations']['geometry_sparse']['removal_fraction']

    n = len(points)
    keep_mask = rng.random(n) > REMOVE_FRAC
    sparse_points = points[keep_mask]

    # --- REcompute geometry descriptors from scratch ---
    voxels = voxelize(sparse_points, voxel_size=0.2)
    voxel_pca = compute_voxel_pca(voxels)        # L, P, S, H_lambda
    normals = compute_normals(sparse_points)
    rel_z = compute_relative_elevation(sparse_points)

    # --- REcompute intensity curvature on remaining points ---
    curvature = compute_intensity_curvature(sparse_points)

    return {
        'points': sparse_points,
        'voxel_pca': voxel_pca,
        'normals': normals,
        'rel_z': rel_z,
        'curvature': curvature,
    }

# ============================================================
# 4. Cross-frame mismatch — same sequence, random offset
# ============================================================
def apply_cross_frame_mismatch(frame_t, sequence, config, seed):
    """
    Geometry from frame t, reflectance from frame t+offset.
    offset is randomly chosen from {+1, +5, +10, +20} frames.
    CRITICAL: both frames come from the SAME sequence (not different scenes),
    so the degradation isolates spatial misalignment, not scene change.

    Procedure:
      1. Load frame t -> extract geometry channels (5-15)
      2. Load frame t+offset (same sequence) -> extract reflectance channels (0-4)
      3. Merge into a single 16-channel input
      4. Run inference
    """
    rng = np.random.default_rng(seed)
    OFFSETS = config['perturbations']['cross_frame_mismatch']['offset_pool']
    offset = int(rng.choice(OFFSETS))   # e.g. 1, 5, 10, or 20

    # Frame t: geometry only
    geom = extract_geometry_channels(frame_t)          # channels 5-15

    # Frame t+offset (same sequence): reflectance only
    frame_other = load_frame(sequence, frame_index=frame_t.index + offset)
    refl = extract_reflectance_channels(frame_other)   # channels 0-4

    # Merge — this is the "misaligned" input
    merged = torch.cat([refl, geom], dim=1)           # 5 + 11 = 16 ch
    return merged, offset

# ============================================================
# Helper stubs (replace with your actual implementations)
# ============================================================
def voxelize(points, voxel_size=0.2):
    """Hash-based voxelization. Replace with your actual implementation."""
    raise NotImplementedError("Plug in your voxelization code here.")

def compute_voxel_pca(voxels):
    """Compute L_lambda, P_lambda, S_lambda, H_lambda. Stub."""
    raise NotImplementedError("Plug in your voxel-PCA code here.")

def compute_normals(points):
    """Surface normals via PCA or SVD. Stub."""
    raise NotImplementedError

def compute_relative_elevation(points):
    """z_rel = (z - z_min) / (z_max - z_min + eps). Stub."""
    raise NotImplementedError

def compute_intensity_curvature(points):
    """MLS curvature + boundary strength. Stub."""
    raise NotImplementedError

def extract_geometry_channels(frame):
    """Return channels 5-15 from a frame. Stub."""
    raise NotImplementedError

def extract_reflectance_channels(frame):
    """Return channels 0-4 from a frame. Stub."""
    raise NotImplementedError

def load_frame(sequence, frame_index):
    """Load a specific frame from a sequence. Stub."""
    raise NotImplementedError

# ============================================================
# Main evaluation loop
# ============================================================
def evaluate_perturbation(name, apply_fn, dataloader, model, config, seed):
    """Run inference and return mean IoU over the validation set."""
    model.eval()
    all_ious = []
    with torch.no_grad():
        for batch in dataloader:
            if name == 'intensity_missing':
                perturbed = apply_intensity_missing(batch, config)
            elif name == 'intensity_corrupted':
                perturbed = apply_intensity_corrupted(batch, config, seed)
            elif name == 'geometry_sparse':
                perturbed = apply_geometry_sparse(batch, config, seed)
            elif name == 'cross_frame_mismatch':
                perturbed, offset = apply_cross_frame_mismatch(
                    batch, batch['sequence'], config, seed
                )
            else:
                raise ValueError(f"Unknown perturbation: {name}")

            logits = model(perturbed)
            iou = compute_miou(logits, batch['labels'])
            all_ious.append(iou.item())
    return float(np.mean(all_ious))

def compute_miou(logits, labels):
    """Per-class IoU averaged over classes. Stub — replace with your metric."""
    raise NotImplementedError

# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))
    model = torch.load(config['baseline']['checkpoint'])
    model.eval()

    dataloader = None  # Replace with your validation dataloader

    results = {}
    for name in ['intensity_missing', 'intensity_corrupted',
                  'geometry_sparse', 'cross_frame_mismatch']:
        scores = []
        for seed in config['train']['seeds']:  # 3 seeds
            iou = evaluate_perturbation(name, None, dataloader, model, config, seed)
            scores.append(iou)
            print(f"  [{name}] seed={seed}: {iou:.1f}%")
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        results[name] = {'mean': mean, 'std': std, 'raw': scores}
        print(f"  [{name}] mean={mean:.2f}%, std={std:.2f}pp")

    # Save
    import json
    with open('logs/perturbation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDone. Results saved to logs/perturbation_results.json")
