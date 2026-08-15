"""Range noise: additive Gaussian perturbation of per-point range.

Each point is displaced along its own viewing ray by delta ~ N(0, sigma^2), so
the (x, y, z) coordinates are scaled by (r + delta) / r. Intensity is untouched.
Acts on raw points; the caller recomputes the 16-channel features afterwards.

severity == gaussian sigma in metres (e.g. 0.02 / 0.05 / 0.10 for L/M/H).
"""
from __future__ import annotations

import numpy as np

from . import RawFrame, frame_rng


def range_noise(frame: RawFrame, sigma_m: float, base_seed: int = 0) -> RawFrame:
    rng = frame_rng(base_seed, frame.frame_id)
    out = frame.copy()
    r = np.linalg.norm(out.points, axis=1)
    r_safe = np.maximum(r, 1e-6)
    delta = rng.normal(0.0, sigma_m, size=r.shape)
    scale = (r_safe + delta) / r_safe
    out.points = out.points * scale[:, None]
    return out
