"""Per-class IoU and mIoU from a confusion matrix.

IoU_c = TP_c / (TP_c + FP_c + FN_c), computed from the accumulated confusion
matrix. mIoU is the unweighted mean over classes that appear in the ground truth
(classes with an all-zero row and column are excluded so absent classes do not
drag the mean). Results are reported in percent to match the paper tables.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .confusion_matrix import ConfusionMatrix


def per_class_iou(conf: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / (denom + eps), np.nan)
    return iou


def mean_iou(conf: np.ndarray) -> float:
    iou = per_class_iou(conf)
    valid = ~np.isnan(iou)
    if not valid.any():
        return 0.0
    return float(np.nanmean(iou[valid]))


class SegmentationMetrics:
    """Convenience accumulator returning percent IoU/mIoU."""

    def __init__(self, num_classes: int, class_names: Optional[list] = None,
                 ignore_label: int = 255):
        self.cm = ConfusionMatrix(num_classes, ignore_label)
        self.class_names = class_names

    def update(self, gt, pred):
        self.cm.update(gt, pred)
        return self

    def result(self) -> dict:
        iou = per_class_iou(self.cm.matrix) * 100.0
        miou = mean_iou(self.cm.matrix) * 100.0
        names = self.class_names or [str(i) for i in range(len(iou))]
        return {
            "mIoU": round(miou, 2),
            "per_class_iou": {n: (round(float(v), 2) if not np.isnan(v) else None)
                              for n, v in zip(names, iou)},
        }
