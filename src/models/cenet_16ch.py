"""CENet-style 16-channel baseline (controlled comparison).

CENet processes the range image with a single stream and a lightweight
context aggregation head. Here it is fed the SAME 16-channel input as
DS-RangeNet v3 so the comparison isolates architecture rather than input
features. Reported 5-seed mean mIoU on the primary split: 65.95
(reproducibility/seeds/results/per_seed_miou.csv).

This is a compact re-implementation of the CENet backbone idea (basic residual
encoder + multi-scale context head), not the original authors' code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._canonical import ds_rangenet_v3 as _m

ResDSBlock = _m.ResDSBlock
make_conv = _m.make_conv
IN_TOTAL = _m.IN_TOTAL
NUM_CLASSES = _m.NUM_CLASSES


class ContextHead(nn.Module):
    """Multi-scale context aggregation, in the spirit of CENet's head."""

    def __init__(self, ch: int, num_classes: int, scales=(1, 2, 3)):
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, s, dilation=s, bias=False),
                nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
                nn.ReLU(inplace=True),
            ) for s in scales
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * len(scales), ch, 1, bias=False),
            nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], dim=1)
        return self.classifier(self.fuse(y))


class CENet16Ch(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, base: int = 32,
                 in_ch: int = IN_TOTAL, conv_type: str = "ds"):
        super().__init__()
        c0, c1, c2, c3 = base, base * 2, base * 4, base * 8
        self.stem = ResDSBlock(in_ch, c0, stride=1, conv_type=conv_type)
        self.s1 = ResDSBlock(c0, c1, stride=2, conv_type=conv_type)
        self.s2 = ResDSBlock(c1, c2, stride=2, conv_type=conv_type)
        self.s3 = ResDSBlock(c2, c3, stride=2, conv_type=conv_type)
        self.head = ContextHead(c3, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        in_size = x.shape[2:]
        e0 = self.stem(x)
        e1 = self.s1(e0)
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        logits = self.head(e3)
        logits = F.interpolate(logits, size=in_size, mode="bilinear",
                               align_corners=False)
        if return_features:
            return {"e3": e3, "logits": logits}
        return logits


def build_cenet_16ch(num_classes: int = NUM_CLASSES,
                     conv_type: str = "ds", **kwargs) -> CENet16Ch:
    return CENet16Ch(num_classes=num_classes, conv_type=conv_type, **kwargs)
