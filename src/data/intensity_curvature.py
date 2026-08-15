"""Intensity-derived material features for the 5-channel material/intensity stream.

From the projected intensity image we compute four statistics over a local
KxK range-image window (plus the normalized range that anchors the stream):

    intensity_mean      - local mean reflectance (material tone)
    intensity_std       - local std (surface roughness / texture)
    intensity_boundary  - gradient magnitude (material edges)
    intensity_curvature - second-order response (Laplacian of intensity),
                          normalized to [0, 1]

The ``intensity_curvature`` channel is the one consumed by the IGCA
Intensity-Curvature Bias (ICB); in the 16-channel tensor it sits at index 3
(CURVATURE_CHANNEL) — see :mod:`src.data.build_16ch_input`.
"""
from __future__ import annotations

import numpy as np


def _sobel(img: np.ndarray):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = kx.T
    gx = _conv2d_same(img, kx)
    gy = _conv2d_same(img, ky)
    return gx, gy


def _laplacian(img: np.ndarray):
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    return _conv2d_same(img, lap)


def _conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, np.float32)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def _local_stat(img: np.ndarray, mask: np.ndarray, k: int):
    """Masked local mean and std over a KxK window."""
    H, W = img.shape
    r = k // 2
    mean = np.zeros_like(img, np.float32)
    std = np.zeros_like(img, np.float32)
    for v in range(H):
        for u in range(W):
            if not mask[v, u]:
                continue
            v0, v1 = max(0, v - r), min(H, v + r + 1)
            u0, u1 = max(0, u - r), min(W, u + r + 1)
            win = img[v0:v1, u0:u1][mask[v0:v1, u0:u1]]
            if win.size == 0:
                continue
            mean[v, u] = win.mean()
            std[v, u] = win.std()
    return mean, std


def intensity_curvature_features(intensity_img: np.ndarray, mask: np.ndarray,
                                 k: int = 5, eps: float = 1e-8) -> dict:
    """intensity_img/(mask): (H,W) -> dict of (H,W) material feature maps."""
    inten = intensity_img.astype(np.float32)
    mean, std = _local_stat(inten, mask, k)
    gx, gy = _sobel(inten)
    boundary = np.sqrt(gx * gx + gy * gy) * mask
    curv = np.abs(_laplacian(inten)) * mask
    cmax = curv.max() + eps
    curvature = curv / cmax                       # normalized to [0,1] for ICB
    return {
        "intensity_mean": mean,
        "intensity_std": std,
        "intensity_boundary": boundary,
        "intensity_curvature": curvature.astype(np.float32),
    }
