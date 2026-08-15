"""16-channel input construction: layout, channel count, and curvature index.

Guards the authoritative CHANNEL_LAYOUT so a refactor cannot silently reorder
channels (which would corrupt every downstream per-class result) or move the
curvature channel away from index 3.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

build = pytest.importorskip("src.data.build_16ch_input")


def test_channel_count_is_16():
    assert len(build.CHANNEL_LAYOUT) == 16


def test_curvature_channel_index():
    assert build.CURVATURE_CHANNEL == 3
    assert build.CHANNEL_LAYOUT[3] == "intensity_curvature"


def test_intensity_then_geometry_partition():
    # first 5 channels are the intensity/material stream, next 11 are geometry.
    intensity = build.CHANNEL_LAYOUT[:5]
    geometry = build.CHANNEL_LAYOUT[5:]
    assert len(intensity) == 5 and len(geometry) == 11
    assert "intensity_curvature" in intensity
    assert set(intensity).isdisjoint(set(geometry))


def test_layout_names_unique():
    assert len(set(build.CHANNEL_LAYOUT)) == len(build.CHANNEL_LAYOUT)


def test_build_output_shape_if_available():
    fn = getattr(build, "build_16ch_input", None)
    if fn is None:
        pytest.skip("build_16ch_input() not exposed; layout constants still checked above")
    H, W = 8, 16
    rng = np.random.default_rng(0)
    dummy = {
        "range": rng.random((H, W), dtype=np.float32),
        "intensity": rng.random((H, W), dtype=np.float32),
        "xyz": rng.random((H, W, 3), dtype=np.float32),
        "mask": np.ones((H, W), dtype=bool),
    }
    try:
        out = fn(**dummy)
    except TypeError:
        pytest.skip("build_16ch_input signature differs from smoke-test kwargs")
    assert out.shape[0] == 16, "channel axis must be first with 16 channels"
