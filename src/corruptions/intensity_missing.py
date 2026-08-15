"""Intensity missing: zero out a fraction of per-point reflectance values.

Simulates dropped/failed reflectance returns while geometry is intact. A seeded
random subset of ``fraction`` points has its intensity set to 0. Geometry is
untouched; the caller recomputes the intensity-derived channels.

Determinism note: at fraction == 1.0 EVERY intensity is zeroed, so the result is
independent of the RNG and therefore bit-exact reproducible across machines.
tests/test_corruption_determinism.py asserts this.

severity == fraction zeroed in [0, 1] (e.g. 0.30 / 0.60 / 1.00 for L/M/H).
"""
from __future__ import annotations

import numpy as np

from . import RawFrame, frame_rng


def intensity_missing(frame: RawFrame, fraction: float,
                      base_seed: int = 0) -> RawFrame:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    out = frame.copy()
    if fraction >= 1.0:
        out.intensity = np.zeros_like(out.intensity)   # deterministic, no RNG
        return out
    rng = frame_rng(base_seed, frame.frame_id)
    n = out.intensity.shape[0]
    drop = rng.random(n) < fraction
    out.intensity[drop] = 0.0
    return out
