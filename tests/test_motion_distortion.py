"""Motion distortion must be genuine PER-POINT timestamp interpolation.

The key property (vs a single global shift) is that displacement grows with a
point's timestamp within the sweep: the last point moves more than the first,
and a point at t=0 does not move at all.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

corr = pytest.importorskip("src.corruptions")


def _ordered_frame(n=200):
    # points on a line at unit range, timestamps increasing across the sweep
    pts = np.zeros((n, 3), np.float32)
    pts[:, 0] = 1.0  # x = 1m so range is well-defined and nonzero
    ts = np.linspace(0.0, 0.1, n).astype(np.float32)  # 100 ms sweep
    return corr.RawFrame(points=pts, intensity=np.ones(n, np.float32),
                         timestamps=ts, frame_id=3)


def test_displacement_is_monotonic_in_timestamp():
    f = _ordered_frame()
    out = corr.motion_distortion(f.copy(), speed_m_s=3.0, sweep_ms=100.0, base_seed=42)
    disp = np.linalg.norm(out.points - f.points, axis=1)
    # first point (t=min) should be ~unmoved; later points move progressively more
    assert disp[0] == pytest.approx(0.0, abs=1e-5), "point at sweep start must not move"
    assert disp[-1] > disp[0], "last point must move more than the first"
    # overall trend increasing: correlation of disp with index is strongly positive
    idx = np.arange(len(disp))
    assert np.corrcoef(idx, disp)[0, 1] > 0.9, "displacement must grow with timestamp"


def test_zero_speed_no_motion_up_to_rotation():
    f = _ordered_frame()
    out = corr.motion_distortion(f.copy(), speed_m_s=0.0, sweep_ms=100.0, base_seed=42)
    # zero speed -> zero translation and (default) zero yaw rate -> no displacement
    assert np.allclose(out.points, f.points, atol=1e-5)


def test_intensity_untouched():
    f = _ordered_frame()
    out = corr.motion_distortion(f.copy(), speed_m_s=1.5, sweep_ms=100.0, base_seed=1)
    assert np.array_equal(out.intensity, f.intensity)
