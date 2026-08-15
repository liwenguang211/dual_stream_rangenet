"""Motion distortion via per-point timestamp interpolation (rolling-shutter).

A moving LiDAR captures points at different instants across a sweep, so the
sensor pose differs per point. This corruption reproduces that skew:

  1. Each point has a timestamp t_i in [0, sweep] (normalized to [0, 1] as
     alpha_i = t_i / sweep).
  2. A constant body velocity ``v`` (m/s) and yaw rate ``omega`` (rad/s) are
     applied for duration alpha_i * sweep, giving a per-point rigid transform
     T_i = (R(omega * dt_i), v * dt_i).
  3. Point i is displaced by T_i, so points late in the sweep are shifted more
     than early ones — exactly the deskew that an ego-motion compensator would
     remove.

This is genuine PER-POINT interpolation over the sweep, not a single global
shift. Geometry moves; intensity is unchanged. The caller recomputes features.

severity scales the linear speed (m/s); omega defaults proportional to speed.
"""
from __future__ import annotations

import numpy as np

from . import RawFrame, frame_rng


def _rot_z(theta: np.ndarray) -> np.ndarray:
    """Stack of Z-rotation matrices for an array of angles -> (N,3,3)."""
    c, s = np.cos(theta), np.sin(theta)
    n = theta.shape[0]
    R = np.zeros((n, 3, 3), np.float64)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def motion_distortion(frame: RawFrame, speed_m_s: float, sweep_ms: float = 100.0,
                      yaw_rate_rad_s: float | None = None,
                      base_seed: int = 0) -> RawFrame:
    out = frame.copy()
    sweep_s = sweep_ms / 1000.0

    # per-point elapsed time within the sweep
    t = out.timestamps.astype(np.float64)
    if t.size == 0:
        return out
    span = t.max() - t.min()
    if span <= 0:
        # no timestamps available: synthesize a linear sweep by point order
        dt = np.linspace(0.0, sweep_s, num=t.size)
    else:
        dt = (t - t.min()) / span * sweep_s

    # slight random heading so distortion is not identical every frame, seeded
    rng = frame_rng(base_seed, frame.frame_id)
    heading = rng.uniform(0.0, 2.0 * np.pi)
    if yaw_rate_rad_s is None:
        yaw_rate_rad_s = 0.05 * speed_m_s      # gentle turn proportional to speed

    # per-point translation from constant velocity over dt
    vx = speed_m_s * np.cos(heading)
    vy = speed_m_s * np.sin(heading)
    trans = np.stack([vx * dt, vy * dt, np.zeros_like(dt)], axis=1)

    # per-point rotation from constant yaw rate over dt
    R = _rot_z(yaw_rate_rad_s * dt)
    p = out.points.astype(np.float64)
    rotated = np.einsum("nij,nj->ni", R, p)
    out.points = (rotated + trans).astype(frame.points.dtype)
    return out
