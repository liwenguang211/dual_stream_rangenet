"""Segmentation metrics: IoU math, ignore-label handling, and known values."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

cm_mod = pytest.importorskip("src.metrics.confusion_matrix")
seg = pytest.importorskip("src.metrics.segmentation_metrics")


def test_perfect_prediction_gives_100_miou():
    gt = np.array([0, 1, 2, 0, 1, 2])
    pred = gt.copy()
    conf = cm_mod.build_confusion_matrix(gt, pred, num_classes=3)
    assert seg.mean_iou(conf) == pytest.approx(1.0)


def test_ignore_label_excluded():
    # ignored pixels (255) must not affect the matrix at all
    gt = np.array([0, 1, 255, 255])
    pred = np.array([0, 1, 0, 1])
    conf = cm_mod.build_confusion_matrix(gt, pred, num_classes=2, ignore_label=255)
    assert conf.sum() == 2, "ignored pixels must be dropped before accumulation"
    assert seg.mean_iou(conf) == pytest.approx(1.0)


def test_known_iou_value():
    # class 0: 2 TP; one class-0 gt predicted as 1 -> FN=1; one class-1 gt as 0 -> FP=1
    # IoU_0 = 2 / (2 + 1 + 1) = 0.5
    gt = np.array([0, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 0, 1])
    conf = cm_mod.build_confusion_matrix(gt, pred, num_classes=2)
    iou = seg.per_class_iou(conf)
    assert iou[0] == pytest.approx(0.5, abs=1e-6)


def test_absent_class_excluded_from_mean():
    # class 2 never appears in gt or pred -> nan -> excluded from mIoU
    gt = np.array([0, 1, 0, 1])
    pred = np.array([0, 1, 0, 1])
    conf = cm_mod.build_confusion_matrix(gt, pred, num_classes=3)
    iou = seg.per_class_iou(conf)
    assert np.isnan(iou[2])
    assert seg.mean_iou(conf) == pytest.approx(1.0)


def test_result_reports_percent_and_names():
    names = ["a", "b"]
    m = seg.SegmentationMetrics(num_classes=2, class_names=names)
    gt = np.array([0, 1, 0, 1])
    m.update(gt, gt.copy())
    res = m.result()
    assert res["mIoU"] == pytest.approx(100.0)
    assert set(res["per_class_iou"]) == set(names)
