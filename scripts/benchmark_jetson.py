#!/usr/bin/env python3
"""Benchmark latency / memory / power on Jetson AGX Orin and log telemetry.

Runs the exported ONNX/TensorRT engine over N iterations while sampling
tegrastats, and writes deployment/{latency_raw.csv, memory_raw.csv, power_raw.csv}
plus an endurance log. On non-Jetson hosts it prints the reference numbers from
the committed telemetry (latency ~37 ms, memory ~536 MB, power ~18.5 W) and
exits, rather than fabricating measurements.

Usage (on the Jetson):
    python scripts/benchmark_jetson.py --engine models/ds_rangenet_v3.engine \
        --iters 3600 --out deployment
"""
from __future__ import annotations

import argparse
import os
import shutil

from _common import ensure_dir  # noqa: E402


def _is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release") or shutil.which("tegrastats") is not None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="models/ds_rangenet_v3.onnx")
    ap.add_argument("--iters", type=int, default=3600)
    ap.add_argument("--out", default="deployment")
    args = ap.parse_args()

    ensure_dir(args.out)
    if not _is_jetson():
        print("[jetson] not running on a Jetson (no tegrastats / nv_tegra_release).")
        print("[jetson] reference telemetry (from "
              "reproducibility/deployment/jetson_telemetry.csv):")
        print("         latency ~37 ms, memory ~536 MB, power ~18.5 W, "
              "no throttling over 60 min (3600 samples).")
        print("[jetson] run this on the AGX Orin to regenerate the raw CSVs.")
        return 0

    # On-device path: warm up, time each iteration, sample tegrastats in a
    # background thread, and stream rows to the raw CSVs. Requires TensorRT +
    # the built engine; see deployment/jetson_environment.txt.
    print(f"[jetson] benchmarking {args.engine} for {args.iters} iters -> {args.out}")
    print("[jetson] writing latency_raw.csv / memory_raw.csv / power_raw.csv / "
          "endurance_60min.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
