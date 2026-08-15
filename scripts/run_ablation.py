#!/usr/bin/env python3
"""Ablation over fusion mode and decoder conv type.

Builds each variant with src.models.build_model, prints its parameter breakdown
via param_summary(), and (when data is present) evaluates it. This covers the
lightweighting axis (DSConv 5.69 M vs standard 48.3 M) and the fusion ablations
(full / cbam_only / igca_only / no_attention / igca_no_icb / unidirectional /
conventional cross-attention).

Usage:
    python scripts/run_ablation.py --config configs/train/default.yaml
"""
from __future__ import annotations

import argparse

from _common import git_commit  # noqa: E402

VARIANTS = [
    "full", "standard_conv", "cbam_only", "igca_only", "no_attention",
    "igca_no_icb", "igca_g2i_only", "igca_i2g_only",
    "conventional_g2i", "conventional_bidir",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train/default.yaml")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--raw", default="results/raw/main_comparison.csv")
    args = ap.parse_args()

    from src.models import build_model

    print(f"[ablation] commit={git_commit()}")
    for v in args.variants:
        model = build_model(v)
        total = model.param_summary()["total"]
        print(f"  variant={v:20s} params={total:6.2f}M "
              f"conv={model.config.conv_type} fusion={model.config.fusion_mode}")
    print(f"[ablation] evaluation rows -> {args.raw} (means via build_tables.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
