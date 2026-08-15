"""Corruptions are bit-exact reproducible for a fixed (seed, frame_id).

Also checks the contract that 100% intensity-missing zeroes everything without
consuming RNG (i.e. is deterministic regardless of seed).
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

corr = pytest.importorskip("src.corruptions")


def _frame(n=256, seed=0):
    rng = np.random.default_rng(seed)
    return corr.RawFrame(
        points=rng.standard_normal((n, 3)).astype(np.float32),
        intensity=rng.random(n).astype(np.float32),
        timestamps=np.linspace(0.0, 0.1, n).astype(np.float32),
        frame_id=17,
    )


def test_frame_rng_formula():
    g1 = corr.frame_rng(1337, 5)
    g2 = np.random.default_rng(1337 + corr.RNG_OFFSET + 5)
    assert np.array_equal(g1.random(10), g2.random(10))


# Each corruption with a representative "medium" parameter set (explicit params,
# matching configs/robustness/three_severity.yaml).
CALLS = {
    "range_noise": {"sigma_m": 0.05},
    "geometric_dropout": {"fraction": 0.30},
    "intensity_calibration": {"gain": 1.20, "bias": 0.05, "noise_std": 0.05},
    "intensity_missing": {"fraction": 0.60},
    "motion_distortion": {"speed_m_s": 1.5},
    "combined": {"severity": "medium"},
}


@pytest.mark.parametrize("name", list(CALLS))
def test_repeated_call_is_bit_exact(name):
    fn = getattr(corr, name)
    f = _frame()
    a = fn(f.copy(), base_seed=1337, **CALLS[name])
    b = fn(f.copy(), base_seed=1337, **CALLS[name])
    assert np.array_equal(a.points, b.points), f"{name}: points differ across identical calls"
    assert np.array_equal(a.intensity, b.intensity), f"{name}: intensity differs"


def test_different_seed_changes_result():
    f = _frame()
    a = corr.range_noise(f.copy(), sigma_m=0.05, base_seed=1)
    b = corr.range_noise(f.copy(), sigma_m=0.05, base_seed=2)
    assert not np.array_equal(a.points, b.points)


def test_full_intensity_missing_is_deterministic_zero():
    f = _frame()
    out1 = corr.intensity_missing(f.copy(), fraction=1.0, base_seed=1)
    out2 = corr.intensity_missing(f.copy(), fraction=1.0, base_seed=999)
    assert np.allclose(out1.intensity, 0.0), "100% missing must zero all intensity"
    assert np.array_equal(out1.intensity, out2.intensity), "100% missing must be seed-independent"
