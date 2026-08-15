"""Pinned Swin-T RangeFormer re-implementation used by the paper.

The manuscript reports a Swin-T re-implementation rather than an unversioned
third-party ``rangeformer`` package. This module provides the trainable baseline
using torchvision 0.15.2 (pinned in requirements.txt). It accepts the common
16-channel input and produces full-resolution semantic logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._canonical import ds_rangenet_v3 as _m

IN_TOTAL = _m.IN_TOTAL
NUM_CLASSES = _m.NUM_CLASSES
# x, y, z, normalized range, and intensity mean in the canonical 16ch tensor.
CONVENTIONAL_CHANNELS = (8, 9, 10, 0, 1)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SwinTRangeFormer(nn.Module):
    """Swin-T range-view segmenter with a dense convolutional decoder."""

    def __init__(self, in_ch: int = 5, num_classes: int = NUM_CLASSES):
        super().__init__()
        try:
            from torchvision.models import swin_t
        except (ImportError, AttributeError) as exc:  # pragma: no cover
            raise ImportError(
                "RangeFormer requires torchvision==0.15.2. Install the pinned "
                "dependencies with `pip install -r requirements.txt`."
            ) from exc

        swin = swin_t(weights=None)
        patch = swin.features[0][0]
        swin.features[0][0] = nn.Conv2d(
            in_ch, patch.out_channels, kernel_size=patch.kernel_size,
            stride=patch.stride, padding=patch.padding,
            bias=patch.bias is not None,
        )
        self.encoder = swin.features
        self.encoder_norm = swin.norm
        self.decoder = nn.Sequential(
            ConvNormAct(768, 512),
            ConvNormAct(512, 512),
            ConvNormAct(512, 512),
            ConvNormAct(512, 256),
            ConvNormAct(256, 256),
            ConvNormAct(256, 256),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        output_size = x.shape[-2:]
        encoded = self.encoder_norm(self.encoder(x)).permute(0, 3, 1, 2)
        logits = F.interpolate(self.decoder(encoded), size=output_size,
                               mode="bilinear", align_corners=False)
        if return_features:
            return {"encoder": encoded, "logits": logits}
        return logits


class RangeFormerAdapter(nn.Module):
    """Select the fixed 5ch baseline input and normalize backbone outputs."""

    def __init__(self, backbone: nn.Module,
                 channels: tuple[int, ...] = CONVENTIONAL_CHANNELS):
        super().__init__()
        self.backbone = backbone
        self.channels = channels

    def forward(self, x: torch.Tensor, return_features: bool = False):
        baseline_input = x if x.shape[1] == len(self.channels) else x[:, self.channels]
        logits = self.backbone(baseline_input)
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("out"))
        if logits is None:
            raise TypeError("External RangeFormer backbone did not return logits")
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                                   align_corners=False)
        return {"logits": logits} if return_features else logits


def build_rangeformer(num_classes: int = NUM_CLASSES, in_ch: int = IN_TOTAL,
                      backbone: nn.Module | None = None,
                      **_):
    """Build the pinned local model, or wrap a caller-supplied backbone."""
    if backbone is not None:
        return RangeFormerAdapter(backbone)
    return RangeFormerAdapter(SwinTRangeFormer(in_ch=5,
                                               num_classes=num_classes))


__all__ = ["CONVENTIONAL_CHANNELS", "SwinTRangeFormer",
           "RangeFormerAdapter", "build_rangeformer"]
