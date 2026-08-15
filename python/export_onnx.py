"""Export and verify DS-RangeNet v3 ONNX models."""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from ds_rangenet_v3 import DualStreamRangeNetV3, IN_TOTAL, NUM_CLASSES, build_model


RANGE_H = 64
RANGE_W = 1024
OPSET = 12


def load_model(weights_path: str, variant: str) -> DualStreamRangeNetV3:
    model = build_model(variant)
    if weights_path and os.path.isfile(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {weights_path}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("No weights supplied; exporting randomly initialized architecture.")
    model.eval()
    return model


def export_onnx(model: DualStreamRangeNetV3, output_path: str, simplify: bool = True) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dummy = torch.randn(1, IN_TOTAL, RANGE_H, RANGE_W)

    print(f"Exporting ONNX opset={OPSET}: {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["range_image"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=OPSET,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )

    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX check passed ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

    if simplify:
        try:
            import onnxsim

            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, output_path)
                print(f"Simplified ONNX saved ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
            else:
                print("onnxsim reported failure; keeping original ONNX.")
        except ImportError:
            print("onnxsim is not installed; skipping simplification.")
    return output_path


def verify_onnx(onnx_path: str):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, IN_TOTAL, RANGE_H, RANGE_W).astype(np.float32)

    for _ in range(3):
        sess.run(None, {"range_image": dummy})

    t0 = time.perf_counter()
    runs = 20
    for _ in range(runs):
        out = sess.run(None, {"range_image": dummy})
    elapsed = (time.perf_counter() - t0) / runs * 1000

    logits = out[0]
    labels = logits.argmax(axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    print("ONNX Runtime verification passed")
    print(f"  Output shape: {logits.shape}")
    print(f"  Average CPU latency: {elapsed:.1f} ms")
    print(f"  Class distribution: {dict(zip(unique.astype(int), counts.astype(int)))}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export DS-RangeNet v3 to ONNX")
    parser.add_argument("--weights", type=str, default="", help="PyTorch checkpoint path")
    parser.add_argument("--output", type=str, default="models/dual_stream_rangenet_v3.onnx")
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        help="Model variant: full, cbam_only, igca_no_icb, conventional_bidir, standard_conv, etc.",
    )
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--verify", type=str, default="", help="Verify an existing ONNX file")
    parser.add_argument("--height", type=int, default=RANGE_H)
    parser.add_argument("--width", type=int, default=RANGE_W)
    return parser.parse_args()


def main():
    args = parse_args()
    global RANGE_H, RANGE_W
    RANGE_H, RANGE_W = args.height, args.width

    if args.verify:
        verify_onnx(args.verify)
        return

    model = load_model(args.weights, args.variant)
    onnx_path = export_onnx(model, args.output, simplify=not args.no_simplify)
    verify_onnx(onnx_path)
    print(f"Export complete: {onnx_path}")


if __name__ == "__main__":
    main()
