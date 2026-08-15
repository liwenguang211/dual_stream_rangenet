"""Single-stream 16-channel baseline (SalsaNext-style controlled baseline).

Controlled baseline for the main comparison: takes the SAME 16-channel input as
DS-RangeNet v3 (5 material/intensity + 11 geometry, concatenated) but processes
it with a single encoder-decoder — no dual stream, no CBAM/IGCA fusion. This
isolates the contribution of the dual-stream intensity-guided fusion. Reported
5-seed mean mIoU on the primary split: 68.60
(reproducibility/seeds/results/per_seed_miou.csv).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._canonical import ds_rangenet_v3 as _m

# reuse the canonical building blocks so the baseline shares conv/BN conventions
ResDSBlock = _m.ResDSBlock
make_conv = _m.make_conv
UpBlock = _m.UpBlock
IN_TOTAL = _m.IN_TOTAL
NUM_CLASSES = _m.NUM_CLASSES


class SingleStream16Ch(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, base: int = 32,
                 in_ch: int = IN_TOTAL, conv_type: str = "ds"):
        super().__init__()
        c0, c1, c2, c3 = base, base * 2, base * 4, base * 8
        self.stem = ResDSBlock(in_ch, c0, stride=1, conv_type=conv_type)
        self.s1 = ResDSBlock(c0, c1, stride=2, conv_type=conv_type)
        self.s2 = ResDSBlock(c1, c2, stride=2, conv_type=conv_type)
        self.s3 = ResDSBlock(c2, c3, stride=2, conv_type=conv_type)
        self.aspp = _m.ASPP(c3, conv_type=conv_type)
        self.up3 = UpBlock(c3, c2, c2, conv_type=conv_type)
        self.up2 = UpBlock(c2, c1, c1, conv_type=conv_type)
        self.up1 = UpBlock(c1, c0, c0, conv_type=conv_type)
        self.head = nn.Sequential(make_conv(conv_type, c0, c0),
                                  nn.Conv2d(c0, num_classes, 1))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        e0 = self.stem(x)
        e1 = self.s1(e0)
        e2 = self.s2(e1)
        e3 = self.aspp(self.s3(e2))
        d2 = self.up3(e3, e2)
        d1 = self.up2(d2, e1)
        d0 = self.up1(d1, e0)
        logits = self.head(d0)
        if return_features:
            return {"e3": e3, "decoder": d0, "logits": logits}
        return logits


def build_single_stream_16ch(num_classes: int = NUM_CLASSES,
                             conv_type: str = "ds", **kwargs) -> SingleStream16Ch:
    return SingleStream16Ch(num_classes=num_classes, conv_type=conv_type, **kwargs)
