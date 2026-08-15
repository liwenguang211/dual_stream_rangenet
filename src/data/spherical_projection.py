"""Spherical projection of a MID-360 point cloud onto a 64x512 range image.

Each 3D point (x, y, z, intensity) is mapped to a pixel (u, v):
    r     = sqrt(x^2 + y^2 + z^2)
    yaw   = atan2(y, x)                        in [-pi, pi]
    pitch = arcsin(z / r)                       in [fov_down, fov_up]
    u = 0.5 * (1 - yaw / pi) * W                (column)
    v = (1 - (pitch - fov_down) / fov)  * H     (row)

When several points fall in the same pixel the nearest (smallest range) point
wins, which is the standard range-image convention. The projection returns the
per-pixel point attributes plus a validity mask and, crucially, the index of the
original point in each pixel so that raw-domain corruptions can be applied to the
points and then re-projected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SphericalProjection:
    height: int = 64
    width: int = 512
    fov_up_deg: float = 52.0       # Livox MID-360 vertical FoV upper bound
    fov_down_deg: float = -7.0     # lower bound
    max_range: float = 50.0

    def project(self, points: np.ndarray, intensity: np.ndarray) -> dict:
        """points: (N,3) xyz; intensity: (N,) -> dict of (H,W) maps + mask/index."""
        assert points.ndim == 2 and points.shape[1] == 3
        H, W = self.height, self.width
        fov_up = np.deg2rad(self.fov_up_deg)
        fov_down = np.deg2rad(self.fov_down_deg)
        fov = fov_up - fov_down

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.linalg.norm(points, axis=1)
        valid = (r > 1e-6) & (r < self.max_range)

        yaw = np.arctan2(y, x)
        pitch = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))

        u = 0.5 * (1.0 - yaw / np.pi) * W
        v = (1.0 - (pitch - fov_down) / fov) * H
        u = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
        v = np.clip(np.floor(v).astype(np.int64), 0, H - 1)

        # nearest-point-wins: sort by decreasing range so nearer points overwrite
        order = np.argsort(-r)
        range_img = np.zeros((H, W), np.float32)
        xyz_img = np.zeros((H, W, 3), np.float32)
        intensity_img = np.zeros((H, W), np.float32)
        mask = np.zeros((H, W), np.bool_)
        point_index = np.full((H, W), -1, np.int64)

        for i in order:
            if not valid[i]:
                continue
            vi, ui = v[i], u[i]
            range_img[vi, ui] = r[i]
            xyz_img[vi, ui] = points[i]
            intensity_img[vi, ui] = intensity[i]
            mask[vi, ui] = True
            point_index[vi, ui] = i

        return {
            "range": range_img,
            "xyz": xyz_img,
            "intensity": intensity_img,
            "mask": mask,
            "point_index": point_index,
        }


def project_to_range_image(points: np.ndarray, intensity: np.ndarray,
                           height: int = 64, width: int = 512,
                           **kwargs) -> dict:
    proj = SphericalProjection(height=height, width=width, **kwargs)
    return proj.project(points, intensity)
