#!/usr/bin/env python3
"""Export a trained DS-RangeNet v3 checkpoint to ONNX for Jetson deployment.

Exports with a fixed 16-channel 64x512 input, opset 13, and prints the ONNX
SHA256 so it can be recorded alongside the checkpoint (see MODEL_ZOO.md).

Usage:
    python scripts/export_onnx.py --model configs/models/ds_rangenet.yaml \
        --checkpoint checkpoints/dsconv_seed0.pth --out models/ds_rangenet_v3.onnx
"""
from __future__ import annotations

import argparse
import os

from _common import load_yaml, sha256_of, ensure_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="configs/models/ds_rangenet.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/ds_rangenet_v3.onnx")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()

    import torch
    from src.models import build_model, IN_TOTAL

    cfg = load_yaml(args.model)
    model = build_model(cfg.get("variant", "full")).eval()
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    else:
        print(f"[export] WARNING: checkpoint {args.checkpoint} not found; "
              f"exporting randomly-initialized weights (shape check only)")

    ensure_dir(os.path.dirname(args.out) or ".")
    dummy = torch.randn(1, IN_TOTAL, args.height, args.width)
    torch.onnx.export(
        model, dummy, args.out, opset_version=13,
        input_names=["input_16ch"], output_names=["logits"],
        dynamic_axes={"input_16ch": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[export] wrote {args.out}")
    print(f"[export] onnx sha256={sha256_of(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
