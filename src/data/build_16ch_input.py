"""Assemble the 16-channel range-image tensor consumed by DS-RangeNet v3.

Channel order is AUTHORITATIVE and must match python/ds_rangenet_v3.py exactly:

  material / intensity stream (channels 0-4):
    0  range_norm            (range / max_range)
    1  intensity_mean
    2  intensity_boundary
    3  intensity_curvature   <- CURVATURE_CHANNEL, feeds the IGCA ICB
    4  intensity_std

  geometry stream (channels 5-15):
    5  normal_x
    6  normal_y
    7  normal_z
    8  x
    9  y
    10 z
    11 linearity
    12 planarity
    13 scattering
    14 eigen_entropy
    15 relative_elevation    (z - per-frame ground height estimate)

The layout, and the fact that channel 3 is the curvature channel, is verified
against the canonical model constants in tests/test_16ch_preprocessing.py.
"""
from __future__ import annotations

import numpy as np

from .spherical_projection import project_to_range_image
from .voxel_pca import local_pca_features
from .intensity_curvature import intensity_curvature_features

# authoritative names, index == channel
CHANNEL_LAYOUT = [
    "range_norm",           # 0
    "intensity_mean",       # 1
    "intensity_boundary",   # 2
    "intensity_curvature",  # 3  == CURVATURE_CHANNEL
    "intensity_std",        # 4
    "normal_x",             # 5
    "normal_y",             # 6
    "normal_z",             # 7
    "x",                    # 8
    "y",                    # 9
    "z",                    # 10
    "linearity",            # 11
    "planarity",            # 12
    "scattering",           # 13
    "eigen_entropy",        # 14
    "relative_elevation",   # 15
]
CURVATURE_CHANNEL = 3
NUM_CHANNELS = 16


def build_16ch_input(points: np.ndarray, intensity: np.ndarray,
                     height: int = 64, width: int = 512,
                     max_range: float = 50.0, window: int = 5) -> dict:
    """Raw points -> (16, H, W) float32 tensor + validity mask.

    Returns a dict with:
      ``tensor``       (16, H, W) float32 in CHANNEL_LAYOUT order
      ``mask``         (H, W) bool, valid pixels
      ``point_index``  (H, W) int, index into ``points`` for each pixel
    """
    proj = project_to_range_image(points, intensity, height=height, width=width,
                                  max_range=max_range)
    mask = proj["mask"]
    H, W = height, width

    inten_feat = intensity_curvature_features(proj["intensity"], mask, k=window)
    geo_feat = local_pca_features(proj["xyz"], mask, k=window)

    xyz = proj["xyz"]
    # relative elevation: z minus a robust per-frame ground estimate (5th pct of z)
    z = xyz[..., 2]
    ground = np.percentile(z[mask], 5) if mask.any() else 0.0
    rel_elev = (z - ground) * mask

    ch = np.zeros((NUM_CHANNELS, H, W), np.float32)
    ch[0] = proj["range"] / max_range
    ch[1] = inten_feat["intensity_mean"]
    ch[2] = inten_feat["intensity_boundary"]
    ch[3] = inten_feat["intensity_curvature"]
    ch[4] = inten_feat["intensity_std"]
    ch[5] = geo_feat["normal"][..., 0]
    ch[6] = geo_feat["normal"][..., 1]
    ch[7] = geo_feat["normal"][..., 2]
    ch[8] = xyz[..., 0]
    ch[9] = xyz[..., 1]
    ch[10] = xyz[..., 2]
    ch[11] = geo_feat["linearity"]
    ch[12] = geo_feat["planarity"]
    ch[13] = geo_feat["scattering"]
    ch[14] = geo_feat["eigen_entropy"]
    ch[15] = rel_elev

    ch *= mask[None, :, :]  # zero out invalid pixels across all channels
    return {"tensor": ch, "mask": mask, "point_index": proj["point_index"]}
