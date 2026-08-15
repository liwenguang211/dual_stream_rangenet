"""Reusable building blocks for DS-RangeNet v3.

Re-exports the canonical, tested modules from ``python/ds_rangenet_v3.py``:
depthwise-separable conv (DSConv), CBAM, and the Intensity-Guided Cross
Attention (IGCA) block with its Intensity-Curvature Bias (ICB).
"""
from .dsconv import DSConv2d, StandardConv2d, make_conv
from .cbam import CBAM, ChannelAttention, SpatialAttention
from .igca import IGCrossAttention, ConventionalCrossAttention

__all__ = [
    "DSConv2d",
    "StandardConv2d",
    "make_conv",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "IGCrossAttention",
    "ConventionalCrossAttention",
]
