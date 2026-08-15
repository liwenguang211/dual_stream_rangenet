"""KNN post-prediction refinement (range-image -> point labels).

The standard range-image KNN clean-up used by DS-RangeNet v3 and its baselines:
for each 3D point, gather the predicted labels of the K nearest projected pixels
(in range-image space, weighted by range difference) and take a majority vote.
This removes projection/back-projection artefacts along depth discontinuities.

Protocol (fixed across all experiments so results are comparable):
  * k = 3 for the primary split and all LOEO folds
  * k = 5 for cross-sensor transfer (sparser/denser external sensors)
The k used for every reported number is recorded in the configs and the
experiment registry; see tests/test_knn_protocol.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KNNConfig:
    k: int = 3
    search: int = 5          # side length of the search window (search x search)
    cutoff: float = 1.0      # max range difference (m) for a neighbour to count
    sigma: float = 1.0       # gaussian range-weight bandwidth (m)


def knn_refine(pred_img: np.ndarray, range_img: np.ndarray,
               point_index: np.ndarray, num_classes: int,
               cfg: KNNConfig | None = None) -> np.ndarray:
    """Refine per-pixel predictions and gather to per-point labels.

    pred_img:    (H, W) int predicted class per pixel
    range_img:   (H, W) float range per pixel
    point_index: (H, W) int index into the original point array (-1 if empty)
    returns:     (N,) refined label per original point (N = max index + 1)
    """
    cfg = cfg or KNNConfig()
    H, W = pred_img.shape
    r = cfg.search // 2
    n_points = int(point_index.max()) + 1 if point_index.max() >= 0 else 0
    out = np.zeros(max(n_points, 0), np.int64)

    vs, us = np.nonzero(point_index >= 0)
    for v, u in zip(vs, us):
        pid = point_index[v, u]
        r0 = range_img[v, u]
        # collect neighbours in the search window
        cand_lbl = []
        cand_w = []
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                nv, nu = v + dv, u + du
                if not (0 <= nv < H and 0 <= nu < W):
                    continue
                if point_index[nv, nu] < 0:
                    continue
                dr = abs(range_img[nv, nu] - r0)
                if dr > cfg.cutoff:
                    continue
                cand_lbl.append(pred_img[nv, nu])
                cand_w.append(np.exp(-(dr * dr) / (2 * cfg.sigma ** 2)))
        if not cand_lbl:
            out[pid] = pred_img[v, u]
            continue
        # keep the k nearest (largest weight) neighbours
        order = np.argsort(cand_w)[::-1][:cfg.k]
        votes = np.zeros(num_classes, np.float64)
        for idx in order:
            votes[cand_lbl[idx]] += cand_w[idx]
        out[pid] = int(np.argmax(votes))
    return out
