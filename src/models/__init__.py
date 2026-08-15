"""Model definitions for DS-RangeNet v3 and its controlled baselines.

Everything here re-exports the canonical, tested implementation from
``python/ds_rangenet_v3.py`` (via :mod:`src._canonical`) so there is a single
source of truth for the architecture, parameter counts, and analysis utilities.
"""
from .ds_rangenet import (
    DualStreamRangeNetV3,
    DSRangeNetConfig,
    build_model,
    CLASSES,
    NUM_CLASSES,
    INTENSITY_CHANNELS,
    GEOMETRY_CHANNELS,
    IN_INTENSITY,
    IN_GEO,
    IN_TOTAL,
    CURVATURE_CHANNEL,
    complementarity_report,
    linear_cka,
    normalized_cross_covariance,
    CombinedLoss,
    FocalLoss,
    DiceLoss,
)
from .single_stream_16ch import build_single_stream_16ch
from .cenet_16ch import build_cenet_16ch
from .rangeformer_adapter import build_rangeformer

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
    "CombinedLoss",
    "FocalLoss",
    "DiceLoss",
    "build_single_stream_16ch",
    "build_cenet_16ch",
    "build_rangeformer",
]
