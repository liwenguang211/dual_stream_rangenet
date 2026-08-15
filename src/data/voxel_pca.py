"""Local PCA / eigenvalue geometric features on the projected range image.

For every valid pixel we gather its neighbours in a KxK range-image window,
compute the 3x3 covariance of their xyz coordinates, and derive the classic
eigenvalue features (lambda1 >= lambda2 >= lambda3 >= 0):

    linearity   = (l1 - l2) / l1
    planarity   = (l2 - l3) / l1
    scattering  = l3 / l1
    eigenentropy= -sum(e_i * log(e_i)),  e_i = l_i / sum(l)

plus the surface normal (eigenvector of the smallest eigenvalue). These populate
the geometry stream channels ``normal_x/y/z, linearity, planarity, scattering,
eigen_entropy`` — see :mod:`src.data.build_16ch_input` for the exact ordering.

The neighbourhood is taken in image space (fast, deterministic) which is the
standard range-image approximation to a 3D local neighbourhood.
"""
from __future__ import annotations

import numpy as np


def _window_offsets(k: int):
    r = k // 2
    return [(dv, du) for dv in range(-r, r + 1) for du in range(-r, r + 1)]


def local_pca_features(xyz_img: np.ndarray, mask: np.ndarray,
                       k: int = 5, eps: float = 1e-8) -> dict:
    """xyz_img: (H,W,3), mask: (H,W) -> dict of (H,W) geometric feature maps."""
    H, W, _ = xyz_img.shape
    offsets = _window_offsets(k)

    normals = np.zeros((H, W, 3), np.float32)
    linearity = np.zeros((H, W), np.float32)
    planarity = np.zeros((H, W), np.float32)
    scattering = np.zeros((H, W), np.float32)
    eigen_entropy = np.zeros((H, W), np.float32)

    vs, us = np.nonzero(mask)
    for v, u in zip(vs, us):
        pts = []
        for dv, du in offsets:
            nv, nu = v + dv, u + du
            if 0 <= nv < H and 0 <= nu < W and mask[nv, nu]:
                pts.append(xyz_img[nv, nu])
        if len(pts) < 3:
            continue
        P = np.asarray(pts, np.float64)
        P = P - P.mean(axis=0, keepdims=True)
        cov = (P.T @ P) / max(len(pts) - 1, 1)
        w, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
        w = np.clip(w[::-1], 0.0, None)        # l1>=l2>=l3
        vecs = vecs[:, ::-1]
        l1, l2, l3 = w + eps
        linearity[v, u] = (l1 - l2) / l1
        planarity[v, u] = (l2 - l3) / l1
        scattering[v, u] = l3 / l1
        s = w.sum() + eps
        e = w / s
        eigen_entropy[v, u] = float(-np.sum(e * np.log(e + eps)))
        n = vecs[:, 2]                         # normal = smallest-eigenvalue vec
        if n[2] < 0:                            # orient consistently (+z up)
            n = -n
        normals[v, u] = n

    return {
        "normal": normals,
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "eigen_entropy": eigen_entropy,
    }
