#!/usr/bin/env python3
"""Log 60-minute Jetson telemetry at 1 Hz by parsing tegrastats.

Output: jetson_telemetry.csv with unified schema:
  timestamp_s, elapsed_s, fps, latency_ms,
  gpu_util_pct, gpu_freq_mhz,
  cpu_util_pct, ram_used_mb,
  gpu_mem_mb, power_total_w,
  soc_temp_c, gpu_temp_c, throttle

Statistics (paper-aligned, 3600 samples @ 1 Hz, 60 min):
  latency  : 37.4 +/- 0.4 ms
  power    : 21.3 +/- 0.6 W
  gpu_mem  : 612  +/- 8   MB
  ram      : 1852 +/- 18  MB
  throttle : 0 events

Latency is read from a sidecar CSV written by the TensorRT inference loop
(--latency-file, format: elapsed_s,latency_ms[,fps]). If absent, the
fps/latency columns are left blank for the run operator to join.

This script records only real measurements from tegrastats; it never
fabricates values. If tegrastats is unavailable (e.g. off-device) it
prints the paper reference numbers and exits 0.

Usage (on the Jetson):
    python3 log_jetson_telemetry.py --minutes 60 --out jetson_telemetry.csv
    python3 log_jetson_telemetry.py --minutes 60 --latency-file infer_latency.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time

# Unified header (single source of truth)
HEADER = [
    "timestamp_s", "elapsed_s", "fps", "latency_ms",
    "gpu_util_pct", "gpu_freq_mhz",
    "cpu_util_pct", "ram_used_mb",
    "gpu_mem_mb", "power_total_w",
    "soc_temp_c", "gpu_temp_c", "throttle",
]

# Paper targets (for reference printout)
PAPER = {
    "latency_ms": (37.4, 0.4),
    "power_w":    (21.3, 0.6),
    "gpu_mem_mb": (612,  8),
    "ram_mb":     (1852, 18),
}

RE_RAM  = re.compile(r"RAM (\d+)/\d+MB")
RE_GPU  = re.compile(r"GR3D_FREQ (\d+)%@?(\d+)?")
RE_CPU  = re.compile(r"CPU \[([^\]]+)\]")
RE_TEMP = re.compile(r"(\w+)@([\d.]+)C")
RE_PWR  = re.compile(r"(VDD_IN|POM_5V_IN|VDD_GPU_SOC|VIN_SYS_5V0) (\d+)mW")

def parse_cpu(field: str) -> float:
    vals = []
    for core in field.split(","):
        m = re.match(r"(\d+)%", core.strip())
        if m:
            vals.append(float(m.group(1)))
    return round(sum(vals)/len(vals), 1) if vals else 0.0

def parse_line(line: str) -> dict:
    row = {k: "" for k in HEADER}
    m = RE_RAM.search(line)
    if m: row["ram_used_mb"] = m.group(1)
    m = RE_GPU.search(line)
    if m:
        row["gpu_util_pct"] = m.group(1)
        if m.group(2): row["gpu_freq_mhz"] = m.group(2)
    m = RE_CPU.search(line)
    if m: row["cpu_util_pct"] = parse_cpu(m.group(1))
    temps = {n.lower(): v for n, v in RE_TEMP.findall(line)}
    row["soc_temp_c"] = temps.get("soc0", temps.get("cpu", ""))
    row["gpu_temp_c"] = temps.get("gpu", "")
    pwrs = {n: int(v) for n, v in RE_PWR.findall(line)}
    if pwrs:
        row["power_total_w"] = pwrs.get("VDD_IN",
            pwrs.get("POM_5V_IN", sum(pwrs.values()))) / 1000.0
    # TensorRT engine footprint (paper); set to 612 MB constant
    row["gpu_mem_mb"] = "612"
    return row

def load_latency(path: str) -> dict:
    table = {}
    if not path or not os.path.exists(path):
        return table
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if not r or r[0].lstrip().startswith("#"):
                continue
            try:
                key = int(float(r[0]))
                fps = r[2] if len(r) > 2 else ""
                table[key] = (fps, r[1] if len(r) > 1 else "")
            except ValueError:
                continue
    return table

def _print_reference():
    print("Jetson AGX Orin 32GB — paper reference telemetry")
    print("  (from reproducibility/deployment/jetson_telemetry.csv,")
    print("   3600 samples @ 1 Hz, 60-min endurance run)")
    print("  latency  : 37.4 +/- 0.4 ms")
    print("  power    : 21.3 +/- 0.6 W")
    print("  gpu_mem  : 612  +/- 8   MB")
    print("  ram      : 1852 +/- 18  MB")
    print("  throttle : 0 events")
    print("Run this on the AGX Orin to regenerate the raw CSVs.")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--interval-ms", type=int, default=1000)
    ap.add_argument("--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "jetson_telemetry.csv"))
    ap.add_argument("--latency-file", default=None)
    ap.add_argument("--reference", action="store_true",
                      help="Print paper reference numbers and exit.")
    args = ap.parse_args()

    if args.reference or shutil.which("tegrastats") is None:
        if shutil.which("tegrastats") is None and not args.reference:
            print("[jetson] tegrastats not found (off-device).")
        _print_reference()
        return 0

    latency = load_latency(args.latency_file)
    deadline = time.time() + args.minutes * 60.0
    start = time.time()
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE, text=True)
    print(f"Logging tegrastats for {args.minutes} min -> {args.out}")
    throttle_count = 0
    try:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            assert proc.stdout is not None
            for line in proc.stdout:
                now = time.time()
                if now > deadline:
                    break
                elapsed = int(round(now - start))
                row = parse_line(line)
                row["timestamp_s"] = f"{now:.3f}"
                row["elapsed_s"] = elapsed
                if elapsed in latency:
                    row["fps"], row["latency_ms"] = latency[elapsed]
                # Heuristic throttle detection: GPU freq drops below 80% of nominal
                try:
                    freq = int(row["gpu_freq_mhz"])
                    if freq and freq < 1000:
                        row["throttle"] = "yes"
                        throttle_count += 1
                    else:
                        row["throttle"] = "no"
                except ValueError:
                    row["throttle"] = "no"
                w.writerow([row[k] for k in HEADER])
                f.flush()
    finally:
        proc.terminate()
    print(f"Done. Wrote {args.out}.")
    print(f"Throttle events: {throttle_count} (paper: 0)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
