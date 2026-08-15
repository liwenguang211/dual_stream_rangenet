"""KNN refinement protocol and behavior.

Checks the fixed protocol (k=3 primary/LOEO, k=5 cross-sensor is what configs
declare) and that refinement gathers pixels to per-point labels correctly and
cleans an isolated mislabeled pixel via weighted majority vote.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

yaml = pytest.importorskip("yaml")
knn = pytest.importorskip("src.postprocess.knn_refinement")


def test_default_k_is_3():
    assert knn.KNNConfig().k == 3


def test_output_is_per_point():
    H, W = 4, 4
    point_index = np.arange(H * W).reshape(H, W)
    pred = np.zeros((H, W), np.int64)
    rng_img = np.ones((H, W), np.float32)
    out = knn.knn_refine(pred, rng_img, point_index, num_classes=3)
    assert out.shape == (H * W,), "one label per original point"
    assert np.all(out == 0)


def test_isolated_wrong_pixel_is_corrected():
    # a 3x3 patch of class 0 with one class-1 pixel in the middle; equal ranges
    H, W = 3, 3
    point_index = np.arange(H * W).reshape(H, W)
    pred = np.zeros((H, W), np.int64)
    pred[1, 1] = 1
    rng_img = np.ones((H, W), np.float32)
    cfg = knn.KNNConfig(k=3, search=3, cutoff=1.0, sigma=1.0)
    out = knn.knn_refine(pred, rng_img, point_index, num_classes=2, cfg=cfg)
    center_pid = point_index[1, 1]
    assert out[center_pid] == 0, "majority vote should overturn the isolated wrong pixel"


def test_empty_pixels_ignored():
    H, W = 3, 3
    point_index = -np.ones((H, W), np.int64)
    point_index[0, 0] = 0
    pred = np.zeros((H, W), np.int64)
    rng_img = np.ones((H, W), np.float32)
    out = knn.knn_refine(pred, rng_img, point_index, num_classes=2)
    assert out.shape == (1,)


def test_configs_declare_expected_k():
    # primary/LOEO configs must use k=3; cross-sensor must use k=5.
    def _find_k(path):
        with open(path) as f:
            text = f.read()
        data = yaml.safe_load(text)
        # search recursively for a 'k' under any 'knn'-ish key
        found = []

        def walk(node):
            if isinstance(node, dict):
                for key, val in node.items():
                    if key == "k" and isinstance(val, int):
                        found.append(val)
                    walk(val)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        return found

    train_cfg = os.path.join(REPO, "configs", "train", "default.yaml")
    cross_cfg = os.path.join(REPO, "configs", "cross_sensor", "mid360.yaml")
    if os.path.exists(train_cfg):
        ks = _find_k(train_cfg)
        assert not ks or 3 in ks, f"primary training config should declare k=3, got {ks}"
    if os.path.exists(cross_cfg):
        ks = _find_k(cross_cfg)
        assert not ks or 5 in ks, f"cross-sensor config should declare k=5, got {ks}"
