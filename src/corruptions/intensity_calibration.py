"""Intensity miscalibration: affine gain/bias plus Gaussian noise.

Models a sensor whose reflectance calibration has drifted:

    I' = clip(gain * I + bias + eps, 0, 1),   eps ~ N(0, noise_std^2)

gain, bias and noise_std are all explicit (no hidden defaults) so the corruption
is fully specified by the config. Geometry (xyz) is untouched; the caller
recomputes the intensity-derived channels (mean/std/boundary/curvature).

Params are passed explicitly, e.g. Medium: gain=1.2, bias=0.05, noise_std=0.05.
"""
from __future__ import annotations

import numpy as np

from . import RawFrame, frame_rng


def intensity_calibration(frame: RawFrame, gain: float, bias: float,
                          noise_std: float, base_seed: int = 0,
                          clip_range=(0.0, 1.0)) -> RawFrame:
    rng = frame_rng(base_seed, frame.frame_id)
    out = frame.copy()
    eps = rng.normal(0.0, noise_std, size=out.intensity.shape)
    lo, hi = clip_range
    out.intensity = np.clip(gain * out.intensity + bias + eps, lo, hi)
    return out
