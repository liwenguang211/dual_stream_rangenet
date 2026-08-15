"""Raw-domain corruptions for robustness evaluation.

CONTRACT (enforced across all modules here):
  * Every corruption takes a RawFrame (points, intensity, timestamps) and returns
    a corrupted RawFrame. The 16-channel features are RECOMPUTED afterwards by
    the caller (scripts/run_corruption.py) via src.data.build_16ch_input.
    Corruptions never touch the finished 16-channel tensor.
  * Randomness is fully seeded and per-frame:
        rng_seed = base_seed + RNG_OFFSET + frame_id      (RNG_OFFSET = 70003)
    so a given (seed, frame) always yields the identical corruption. 100%
    intensity-missing is deterministic by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

RNG_OFFSET = 70003


@dataclass
class RawFrame:
    points: np.ndarray          # (N, 3) xyz
    intensity: np.ndarray       # (N,)
    timestamps: np.ndarray      # (N,) per-point time in seconds within the sweep
    frame_id: int = 0

    def copy(self) -> "RawFrame":
        return replace(
            self,
            points=self.points.copy(),
            intensity=self.intensity.copy(),
            timestamps=self.timestamps.copy(),
        )


def frame_rng(base_seed: int, frame_id: int) -> np.random.Generator:
    """Deterministic per-frame generator: base_seed + 70003 + frame_id."""
    return np.random.default_rng(base_seed + RNG_OFFSET + frame_id)


from .range_noise import range_noise
from .geometric_dropout import geometric_dropout
from .intensity_calibration import intensity_calibration
from .intensity_missing import intensity_missing
from .motion_distortion import motion_distortion
from .combined import combined, COMPOSITIONS

__all__ = [
    "RawFrame",
    "frame_rng",
    "RNG_OFFSET",
    "range_noise",
    "geometric_dropout",
    "intensity_calibration",
    "intensity_missing",
    "motion_distortion",
    "combined",
    "COMPOSITIONS",
]
