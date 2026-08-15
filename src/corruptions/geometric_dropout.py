"""Geometric dropout: random removal of a fraction of raw points.

Simulates sparser returns / occlusion. Removes ``fraction`` of points uniformly
at random (seeded), keeping points, intensity and timestamps aligned. The caller
recomputes geometry descriptors and intensity curvature on the surviving points,
which is the whole point of doing this in the raw domain.

severity == fraction removed in [0, 1) (e.g. 0.10 / 0.30 / 0.50 for L/M/H).
"""
from __future__ import annotations

import numpy as np

from . import RawFrame, frame_rng


def geometric_dropout(frame: RawFrame, fraction: float,
                      base_seed: int = 0) -> RawFrame:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    rng = frame_rng(base_seed, frame.frame_id)
    n = frame.points.shape[0]
    keep = rng.random(n) >= fraction
    out = frame.copy()
    out.points = out.points[keep]
    out.intensity = out.intensity[keep]
    out.timestamps = out.timestamps[keep]
    return out
