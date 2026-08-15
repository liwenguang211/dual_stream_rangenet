"""Accumulating confusion matrix for semantic segmentation.

Rows = ground truth, columns = prediction. The ignore label (default 255) is
excluded before accumulation, so ignored pixels never affect any metric.
"""
from __future__ import annotations

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_label: int = 255):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mat = np.zeros((num_classes, num_classes), np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray) -> "ConfusionMatrix":
        gt = np.asarray(gt).reshape(-1)
        pred = np.asarray(pred).reshape(-1)
        keep = gt != self.ignore_label
        gt, pred = gt[keep], pred[keep]
        idx = gt * self.num_classes + pred
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += binc.reshape(self.num_classes, self.num_classes)
        return self

    def reset(self):
        self.mat[...] = 0

    @property
    def matrix(self) -> np.ndarray:
        return self.mat


def build_confusion_matrix(gt: np.ndarray, pred: np.ndarray,
                           num_classes: int, ignore_label: int = 255) -> np.ndarray:
    cm = ConfusionMatrix(num_classes, ignore_label)
    cm.update(gt, pred)
    return cm.matrix
