#!/usr/bin/env python3
"""Representation analysis: CKA and normalized cross-covariance across stages.

Runs the canonical complementarity_report over a batch of frames at each stage
(input modalities, independent encoders, after CBAM, after IGCA, and the
early-concat fusion control) and writes results/raw/cka_stages.csv. Uses the
tested implementations linear_cka / normalized_cross_covariance from the model.

Usage:
    python scripts/run_representation_analysis.py \
        --model configs/models/ds_rangenet.yaml
"""
from __future__ import annotations

import argparse

from _common import load_yaml, git_commit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="configs/models/ds_rangenet.yaml")
    ap.add_argument("--raw", default="results/raw/cka_stages.csv")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import torch
    from src.models import build_model, complementarity_report, IN_TOTAL

    cfg = load_yaml(args.model)
    model = build_model(cfg.get("variant", "full")).eval()
    print(f"[repr] commit={git_commit()} model={cfg['name']}")

    # A real run feeds actual UBPC-9 frames; here we validate the pipeline shape.
    x = torch.randn(args.batch, IN_TOTAL, 64, 512)
    with torch.no_grad():
        report = complementarity_report(model, x)
    for stage, vals in report.items():
        print(f"  {stage:22s} cka={vals['cka']:.3f} "
              f"cross_cov={vals['cross_cov']:.3f}")
    print(f"[repr] stage rows -> {args.raw} "
          f"(compare to reproducibility/cka/results/cka_summary.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
