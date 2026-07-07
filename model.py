"""Compatibility imports for the corrected 16-channel DS-RangeNet v3.

The project now uses the paper-aligned 16-channel DS-RangeNet v3
implementation. Import from this module when older tooling expects
``python/model.py`` to exist.
"""

from ds_rangenet_v3 import (  # noqa: F401
    ASPP,
    CBAM,
    CBAMFusion,
    CLASSES,
    CombinedLoss,
    CURVATURE_CHANNEL,
    ConventionalCrossAttention,
    DSConv2d,
    DSRangeNetConfig,
    DiceLoss,
    DualStreamRangeNetV3,
    FocalLoss,
    IGCrossAttention,
    IN_GEO,
    IN_INTENSITY,
    IN_RANGE,
    IN_TOTAL,
    NUM_CLASSES,
    ResBlock,
    StreamEncoder,
    UpBlock,
    apply_corruption,
    build_model,
    complementarity_report,
    linear_cka,
    normalized_cross_covariance,
)


DualStreamRangeNet = DualStreamRangeNetV3


if __name__ == "__main__":
    import torch

    model = DualStreamRangeNetV3().eval()
    x = torch.randn(1, IN_TOTAL, 64, 1024)
    with torch.no_grad():
        out = model(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(out.shape)}")
    print(f"params: {model.param_summary()['total']:.3f}M")
