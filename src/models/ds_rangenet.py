"""DS-RangeNet v3 — canonical dual-stream model (re-export).

The full, tested implementation lives in ``python/ds_rangenet_v3.py``:
  - 16-channel input (5-channel material/intensity + 11-channel geometry)
  - shallow CBAM fusion, per-stream ASPP
  - pooled IGCA with pairwise Intensity-Curvature Bias (ICB)
  - DSConv decoder, 9-class head
  - all controlled variants (conventional cross-attention, IGCA-no-ICB,
    unidirectional IGCA, standard-conv control, modality controls)

This module simply re-exports those symbols so ``from src.models.ds_rangenet
import build_model`` works, without forking the architecture code.
"""
from __future__ import annotations

from .._canonical import ds_rangenet_v3 as _m

DualStreamRangeNetV3 = _m.DualStreamRangeNetV3
DSRangeNetConfig = _m.DSRangeNetConfig
build_model = _m.build_model

CLASSES = _m.CLASSES
NUM_CLASSES = _m.NUM_CLASSES
INTENSITY_CHANNELS = _m.INTENSITY_CHANNELS
GEOMETRY_CHANNELS = _m.GEOMETRY_CHANNELS
IN_INTENSITY = _m.IN_INTENSITY
IN_GEO = _m.IN_GEO
IN_TOTAL = _m.IN_TOTAL
CURVATURE_CHANNEL = _m.CURVATURE_CHANNEL

complementarity_report = _m.complementarity_report
linear_cka = _m.linear_cka
normalized_cross_covariance = _m.normalized_cross_covariance
apply_corruption = _m.apply_corruption

CombinedLoss = _m.CombinedLoss
FocalLoss = _m.FocalLoss
DiceLoss = _m.DiceLoss

__all__ = [
    "DualStreamRangeNetV3",
    "DSRangeNetConfig",
    "build_model",
    "CLASSES",
    "NUM_CLASSES",
    "INTENSITY_CHANNELS",
    "GEOMETRY_CHANNELS",
    "IN_INTENSITY",
    "IN_GEO",
    "IN_TOTAL",
    "CURVATURE_CHANNEL",
    "complementarity_report",
    "linear_cka",
    "normalized_cross_covariance",
    "apply_corruption",
    "CombinedLoss",
    "FocalLoss",
    "DiceLoss",
]
