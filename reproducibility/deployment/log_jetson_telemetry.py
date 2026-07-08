#!/usr/bin/env python3
"""Log 60-minute Jetson telemetry at 1 Hz by parsing tegrastats.

Reviewer 2 asks for a long-term (60-minute) Jetson telemetry CSV. This script
runs ``tegrastats`` at a 1 s interval, parses each line for GPU/CPU utilization,
GPU frequency, RAM, power, and temperatures, and writes one row per second to
``jetson_telemetry.csv``. Per-frame fps/latency can be merged in from a sidecar
file written by the inference loop (``--latency-file``); otherwise those columns
are left blank for the run operator to join.

Usage (on the Jetson):
    python3 log_jetson_telemetry.py --minutes 60 --out jetson_telemetry.csv

    # optional: merge inference latency written as "elapsed_s,latency_ms,fps" lines
    python3 log_jetson_telemetry.py --minutes 60 --latency-file infer_latency.csv

This script only records real measurements from tegrastats; it never synthesizes
values. If tegrastats is unavailable (e.g. running off-device) it exits with a
clear message instead of writing fake data.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time

HEADER = [
    "timestamp_s", "elapsed_s", "fps", "latency_ms", "gpu_util_pct", "gpu_freq_mhz",
    "cpu_util_pct", "ram_used_mb", "power_total_mw", "soc_temp_c", "gpu_temp_c",
]

RE_RAM = re.compile(r"RAM (\d+)/\d+MB")
RE_GPU = re.compile(r"GR3D_FREQ (\d+)%@?(\d+)?")
RE_CPU = re.compile(r"CPU \[([^\]]+)\]")
RE_TEMP = re.compile(r"(\w+)@([\d.]+)C")
RE_POWER = re.compile(r"(VDD_IN|POM_5V_IN|VDD_GPU_SOC|VIN_SYS_5V0) (\d+)mW")


def parse_cpu(field: str) -> float:
    loads = []
    for core in field.split(","):
        m = re.match(r"(\d+)%", core)
        if m:
            loads.append(float(m.group(1)))
    return round(sum(loads) / len(loads), 1) if loads else 0.0


def parse_line(line: str) -> dict:
    row = {k: "" for k in HEADER}
    m = RE_RAM.search(line)
    if m:
        row["ram_used_mb"] = m.group(1)
    m = RE_GPU.search(line)
    if m:
        row["gpu_util_pct"] = m.group(1)
        if m.group(2):
            row["gpu_freq_mhz"] = m.group(2)
    m = RE_CPU.search(line)
    if m:
        row["cpu_util_pct"] = parse_cpu(m.group(1))
    temps = {name.lower(): val for name, val in RE_TEMP.findall(line)}
    row["soc_temp_c"] = temps.get("soc0", temps.get("cpu", ""))
    row["gpu_temp_c"] = temps.get("gpu", "")
    powers = {name: int(val) for name, val in RE_POWER.findall(line)}
    if powers:
        # prefer a total-input rail if present, else sum the reported rails
        row["power_total_mw"] = powers.get("VDD_IN", powers.get("POM_5V_IN", sum(powers.values())))
    return row


def load_latency(path: str) -> dict:
    """Map int(elapsed_s) -> (fps, latency_ms) from a sidecar CSV."""
    table = {}
    if not path or not os.path.exists(path):
        return table
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if not r or r[0].lstrip().startswith("#"):
                continue
            try:
                table[int(float(r[0]))] = (r[2] if len(r) > 2 else "", r[1] if len(r) > 1 else "")
            except ValueError:
                continue
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--interval-ms", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "jetson_telemetry.csv"))
    ap.add_argument("--latency-file", default=None)
    args = ap.parse_args()

    if shutil.which("tegrastats") is None:
        raise SystemExit("tegrastats not found. Run this on the Jetson device; refusing to write synthetic data.")

    latency = load_latency(args.latency_file)
    deadline = time.time() + args.minutes * 60.0
    start = time.time()
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE, text=True,
    )
    print(f"Logging tegrastats for {args.minutes} min -> {args.out}")
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
                row["timestamp_s"] = round(now, 3)
                row["elapsed_s"] = elapsed
                if elapsed in latency:
                    row["fps"], row["latency_ms"] = latency[elapsed]
                w.writerow([row[k] for k in HEADER])
                f.flush()
    finally:
        proc.terminate()
    print(f"Done. Wrote {args.out}.")


if __name__ == "__main__":
    main()
