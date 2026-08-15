"""Combined corruption: Light / Medium / Heavy compositions.

Each composition applies several raw-domain corruptions in a fixed order using
the SAME per-frame seed, so the whole pipeline is deterministic. The order is
range -> geometry -> intensity so that geometric dropout happens before the
intensity perturbations act on the surviving points.

The three severities are explicitly defined below (not inferred), matching the
robustness config configs/robustness/three_severity.yaml.
"""
from __future__ import annotations

import numpy as np

from . import RawFrame
from .range_noise import range_noise
from .geometric_dropout import geometric_dropout
from .intensity_calibration import intensity_calibration
from .intensity_missing import intensity_missing
from .motion_distortion import motion_distortion

# Explicit Light / Medium / Heavy compositions.
COMPOSITIONS = {
    "light": {
        "range_noise": {"sigma_m": 0.02},
        "geometric_dropout": {"fraction": 0.10},
        "intensity_calibration": {"gain": 1.10, "bias": 0.02, "noise_std": 0.02},
        "motion_distortion": {"speed_m_s": 0.5},
    },
    "medium": {
        "range_noise": {"sigma_m": 0.05},
        "geometric_dropout": {"fraction": 0.30},
        "intensity_calibration": {"gain": 1.20, "bias": 0.05, "noise_std": 0.05},
        "motion_distortion": {"speed_m_s": 1.5},
    },
    "heavy": {
        "range_noise": {"sigma_m": 0.10},
        "geometric_dropout": {"fraction": 0.50},
        "intensity_calibration": {"gain": 1.40, "bias": 0.10, "noise_std": 0.10},
        "motion_distortion": {"speed_m_s": 3.0},
    },
}

# fixed application order (geometry-altering first, then intensity)
_ORDER = ["motion_distortion", "range_noise", "geometric_dropout",
          "intensity_calibration", "intensity_missing"]

_FUNCS = {
    "range_noise": range_noise,
    "geometric_dropout": geometric_dropout,
    "intensity_calibration": intensity_calibration,
    "intensity_missing": intensity_missing,
    "motion_distortion": motion_distortion,
}


def combined(frame: RawFrame, severity: str, base_seed: int = 0) -> RawFrame:
    if severity not in COMPOSITIONS:
        raise ValueError(f"severity must be one of {sorted(COMPOSITIONS)}")
    spec = COMPOSITIONS[severity]
    out = frame
    for name in _ORDER:
        if name in spec:
            out = _FUNCS[name](out, base_seed=base_seed, **spec[name])
    return out
