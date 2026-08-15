"""CBAM (Convolutional Block Attention Module) — re-export.

Used for the shallow-stage fusion (fuse0/1/2) of DS-RangeNet v3: channel
attention followed by spatial attention. Canonical definitions:
python/ds_rangenet_v3.py.
"""
from __future__ import annotations

from ..._canonical import ds_rangenet_v3 as _m

CBAM = _m.CBAM
ChannelAttention = _m.ChannelAttention
SpatialAttention = _m.SpatialAttention

__all__ = ["CBAM", "ChannelAttention", "SpatialAttention"]
