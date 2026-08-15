"""Smoke tests for the paper-aligned DS-RangeNet v3 model family."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ds_rangenet_v3 import (
    CombinedLoss,
    IN_GEO,
    IN_INTENSITY,
    IN_TOTAL,
    NUM_CLASSES,
    apply_corruption,
    build_model,
    complementarity_report,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    check(IN_INTENSITY == 5, "material/intensity stream must have 5 channels")
    check(IN_GEO == 11, "geometry stream must have 11 channels")
    check(IN_TOTAL == 16, "total input must have 16 channels")

    x = torch.randn(1, IN_TOTAL, 64, 128)
    target = torch.randint(0, NUM_CLASSES, (1, 64, 128))

    variants = [
        "full",
        "cbam_only",
        "igca_only",
        "no_attention",
        "igca_no_icb",
        "igca_g2i_only",
        "igca_i2g_only",
        "conventional_g2i",
        "conventional_bidir",
        "standard_conv",
        "intensity_only",
        "geometry_only",
    ]

    for name in variants:
        model = build_model(name).eval()
        with torch.no_grad():
            y = model(x)
        check(tuple(y.shape) == (1, NUM_CLASSES, 64, 128), f"{name}: bad output shape {tuple(y.shape)}")

    full = build_model("full").eval()
    with torch.no_grad():
        feats = full(x, return_features=True)
    check("logits" in feats and "f3" in feats, "feature dictionary incomplete")

    report = complementarity_report(full, x)
    check("independent_encoders" in report, "complementarity report missing encoder stage")

    for kind in ["range_noise", "intensity_noise", "point_dropout", "scanline_dropout", "block_occlusion"]:
        xc = apply_corruption(x, kind)
        check(tuple(xc.shape) == tuple(x.shape), f"{kind}: corruption changed shape")

    loss = CombinedLoss()(full(x), target)
    check(torch.isfinite(loss), "loss must be finite")

    print("All DS-RangeNet v3 model-family checks passed.")
    print(f"Input channels: {IN_TOTAL} = {IN_INTENSITY} material-intensity + {IN_GEO} geometry")
    print(f"Full model params: {full.param_summary()['total']:.3f}M")


if __name__ == "__main__":
    main()
