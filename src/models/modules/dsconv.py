"""Depthwise-separable convolution and its standard-conv control (re-export).

DSConv2d is the depthwise (groups=in_ch) + pointwise (1x1) block that makes the
5.69 M DS-RangeNet v3 decoder lightweight; StandardConv2d is the full-conv
control used in the 48.3 M ablation row. ``make_conv`` selects between them.
Canonical definitions: python/ds_rangenet_v3.py.
"""
from __future__ import annotations

from ..._canonical import ds_rangenet_v3 as _m

DSConv2d = _m.DSConv2d
StandardConv2d = _m.StandardConv2d
make_conv = _m.make_conv

__all__ = ["DSConv2d", "StandardConv2d", "make_conv"]
