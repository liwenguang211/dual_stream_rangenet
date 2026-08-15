#!/usr/bin/env python3
"""
cross_validate.py
=================
Cross-validates three sources of truth:
1. ubpc9_splits_v3.json   (canonical config)
2. loeo_summary.csv        (9-fold summary)
3. perclass_loeo_details.csv (per-class table)
4. training_log_9fold.txt  (human-readable log)

All four must agree or the script exits 1.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
errors = []

def check(cond, tag, msg=''):
    if cond:
        print(f"  [PASS] {tag}")
    else:
        print(f"  [FAIL] {tag}  -> {msg}")
        errors.append(f"{tag}: {msg}")

# ============================================================
print("="*60)
print("1. JSON vs loeo_summary.csv")
print("="*60)

with open(ROOT / 'ubpc9_splits_v3.json') as f:
    C = json.load(f)

csv_rows = []
with open(ROOT / 'loeo_summary.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_rows.append(row)

# Find macro/sd/drop rows (they use column 2,3,4 as their value)
csv_macro = None
csv_sd = None
csv_drop = None
fold_rows = []
for r in csv_rows:
    if r['fold'] == 'macro_avg':
        # Value is in test_miou column or val_site column
        v = r['test_miou'] if r['test_miou'] else r['val_site']
        csv_macro = float(v)
    elif r['fold'] == 'sample_sd_pp':
        v = r['test_miou'] if r['test_miou'] else r['val_site']
        csv_sd = float(v)
    elif r['fold'] == 'population_sd_pp':
        pass  # not needed for check
    elif r['fold'] == 'drop_from_primary_pp':
        v = r['test_miou'] if r['test_miou'] else r['val_site']
        csv_drop = float(v)
    else:
        fold_rows.append(r)

json_miou = C['loeo_9fold']['miou_list']
json_macro = C['loeo_9fold']['macro_avg_miou']
json_sd = C['loeo_9fold']['sample_std_pp']
json_drop = C['loeo_9fold']['drop_from_primary_pp']

# Check each fold mIoU
for r in fold_rows:
    fid = int(r['fold'])
    csv_m = float(r['test_miou'])
    json_m = json_miou[fid - 1]
    check(abs(csv_m - json_m) < 0.01,
          f"Fold {fid}: CSV {csv_m} == JSON {json_m}",
          f"CSV={csv_m} JSON={json_m}")

check(abs(csv_macro - json_macro) < 0.01,
      f"Macro: CSV {csv_macro} == JSON {json_macro}",
      f"CSV={csv_macro} JSON={json_macro}")

check(abs(csv_sd - json_sd) < 0.01,
      f"Sample SD: CSV {csv_sd} == JSON {json_sd}",
      f"CSV={csv_sd} JSON={json_sd}")

check(abs(csv_drop - json_drop) < 0.01,
      f"Drop: CSV {csv_drop} == JSON {json_drop}",
      f"CSV={csv_drop} JSON={json_drop}")

# ============================================================
print("\n" + "="*60)
print("2. JSON vs perclass_loeo_details.csv")
print("="*60)

pc_json = C['perclass']
json_classes = pc_json['classes']
json_primary = pc_json['primary_iou']
json_loeo = pc_json['loeo_avg_iou']
json_delta = pc_json['delta_pp']

csv_pc = []
with open(ROOT / 'perclass_loeo_details.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['class'] == 'macro_avg':
            csv_pc_macro_primary = float(row['primary_iou'])
            csv_pc_macro_loeo = float(row['loeo_avg_iou'])
            csv_pc_macro_delta = float(row['delta_pp'])
        else:
            csv_pc.append(row)

for i, cls in enumerate(json_classes):
    csv_row = next(r for r in csv_pc if r['class'] == cls)
    c_prim = float(csv_row['primary_iou'])
    c_loeo = float(csv_row['loeo_avg_iou'])
    c_delta = float(csv_row['delta_pp'])
    check(abs(c_prim - json_primary[i]) < 0.01,
          f"Class {cls}: primary CSV={c_prim} == JSON={json_primary[i]}",
          f"mismatch")
    check(abs(c_loeo - json_loeo[i]) < 0.01,
          f"Class {cls}: loeo CSV={c_loeo} == JSON={json_loeo[i]}",
          f"mismatch")
    check(abs(c_delta - json_delta[i]) < 0.01,
          f"Class {cls}: delta CSV={c_delta} == JSON={json_delta[i]}",
          f"mismatch")

# Macro from perclass CSV
csv_pc_macro_calc = sum(float(r['loeo_avg_iou']) for r in csv_pc) / len(csv_pc)
check(abs(csv_pc_macro_calc - json_macro) < 0.01,
      f"PerClass macro avg {csv_pc_macro_calc:.4f} == JSON {json_macro}",
      f"mismatch")

# ============================================================
print("\n" + "="*60)
print("3. JSON vs training_log_9fold.txt")
print("="*60)

log_text = (ROOT / 'training_log_9fold.txt').read_text()

# Extract per-fold mIoU from log
for i in range(1, 10):
    # Match: ">>> Test mIoU (held-out S1) = 70.4%"
    pattern = rf">>> Test mIoU \(held-out S{i}\) = ([\d.]+)%"
    m = re.search(pattern, log_text)
    if m:
        log_miou = float(m.group(1))
        json_m = json_miou[i-1]
        check(abs(log_miou - json_m) < 0.01,
              f"Fold {i}: log {log_miou}% == JSON {json_m}%",
              f"log={log_miou} JSON={json_m}")
    else:
        # Try alternate pattern without >>>
        pattern2 = rf"Test mIoU \(held-out S{i}\) = ([\d.]+)%"
        m2 = re.search(pattern2, log_text)
        if m2:
            log_miou = float(m2.group(1))
            json_m = json_miou[i-1]
            check(abs(log_miou - json_m) < 0.01,
                  f"Fold {i}: log {log_miou}% == JSON {json_m}%",
                  f"log={log_miou} JSON={json_m}")
        else:
            check(False, f"Fold {i}: mIoU found in log", "not found")

# Extract macro from log (format: "Macro 71.0%   (unrounded 70.96%)")
m_macro = re.search(r"Macro\s+\d+\.\d+%\s*\(unrounded\s*([\d.]+)%\)", log_text)
if m_macro:
    log_macro = float(m_macro.group(1))
    check(abs(log_macro - json_macro) < 0.01,
          f"Log macro {log_macro}% == JSON {json_macro}%",
          f"log={log_macro} JSON={json_macro}")
else:
    # Try simpler pattern
    m_macro2 = re.search(r"Macro\s*(\d+\.\d+)%", log_text)
    if m_macro2:
        log_macro = float(m_macro2.group(1))
        check(abs(log_macro - json_macro) < 0.05,
              f"Log macro {log_macro}% ~ JSON {json_macro}%",
              f"log={log_macro} JSON={json_macro}")
    else:
        check(False, "Log macro found in log", "not found")

# Extract SD from log
m_sd = re.search(r"Sample SD\s*=\s*([\d.]+)\s*pp", log_text)
if m_sd:
    log_sd = float(m_sd.group(1))
    check(abs(log_sd - json_sd) < 0.05,
          f"Log SD {log_sd}pp ~ JSON {json_sd}pp",
          f"log={log_sd} JSON={json_sd}")

# ============================================================
print("\n" + "="*60)
print("4. JSON vs primary_results.csv")
print("="*60)

pr = {}
with open(ROOT / 'primary_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pr[row['metric']] = row['value']

check(int(pr['total_frames']) == 12000, "Primary frames = 12000", f"got {pr['total_frames']}")
check(int(pr['lidar_hz']) == 10, "Primary Hz = 10", f"got {pr['lidar_hz']}")
check(float(pr['agv_speed_m_per_s']) == 1.5, "Primary speed = 1.5", f"got {pr['agv_speed_m_per_s']}")
check(float(pr['total_trajectory_m']) == 1800, "Primary traj = 1800", f"got {pr['total_trajectory_m']}")
check(float(pr['primary_miou']) == 73.2, "Primary mIoU = 73.2", f"got {pr['primary_miou']}")

# ============================================================
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

if errors:
    print(f"  FAILED ({len(errors)} error(s))")
    for e in errors:
        print(f"    x {e}")
    sys.exit(1)
else:
    print("  ALL CROSS-VALIDATIONS PASSED")
    print("  JSON = CSV = Training Log (all numbers agree)")
    sys.exit(0)
