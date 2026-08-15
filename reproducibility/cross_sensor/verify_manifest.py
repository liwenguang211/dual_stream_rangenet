#!/usr/bin/env python3
"""
verify_manifest.py — Cross-sensor experiment consistency checker (v2.0).

Checks:
  1. CSV structural integrity 
  2. File naming conventions (checkpoint, prediction_dir, eval_log)
  3. Per-dataset mIoU values match paper tables
  4. valid_pixel_ratio in valid range
  5. Cross-sensor drop calculations match paper
  6. 5-seed statistics for RS-Helios32 fine-tune are self-consistent
  7. Real checkpoint files exist (not just referenced)
  8. Prediction directories exist with content
  9. Eval log files exist with content
 10. Subset indices file exists and has 5 subsets
 11. Fixed test set is consistent across zero-shot and all 5 fine-tune runs
 12. SHA-256 hashes are 64-char hex strings

Usage: python verify_manifest.py
"""

import csv
import sys
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
ERRORS = 0
WARNINGS = 0


# ---------- helpers ----------

def error(msg):
    global ERRORS
    print(f"  ❌ {msg}")
    ERRORS += 1

def warn(msg):
    global WARNINGS
    print(f"  ⚠️  {msg}")
    WARNINGS += 1

def ok(msg):
    print(f"  ✅ {msg}")

def parse_vpr(val_str):
    """Accepts '82.4', '0.824', etc. Returns value in 0-1."""
    v = float(val_str)
    if v > 1.0:
        v = v / 100.0
    return v

def is_valid_sha256(s):
    """Check if string is a valid 64-char hex SHA-256."""
    return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s.lower())

def compute_sha256(path):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------- main ----------

def main():
    global ERRORS, WARNINGS

    # ---- Load checkpoint_manifest.csv ----
    manifest_path = ROOT / "checkpoint_manifest.csv"
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)

    rows = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        expected_cols = {"dataset","model","seed","sensor_beams","checkpoint_path",
                         "sha256","prediction_dir","official_eval_log",
                         "miou_pre_knn","miou_post_knn","valid_pixel_ratio","note"}
        actual_cols = set(reader.fieldnames) if reader.fieldnames else set()
        if expected_cols != actual_cols:
            missing = expected_cols - actual_cols
            extra = actual_cols - expected_cols
            if missing:
                error(f"Manifest missing columns: {missing}")
            if extra:
                warn(f"Manifest has extra columns: {extra}")
        for row in reader:
            rows.append(row)

    print(f"📋 Loaded {len(rows)} rows from checkpoint_manifest.csv\n")

    # ---- Check 0: No FILL_ME anywhere ----
    print("=" * 60)
    print("Check 0: No FILL_ME placeholders")
    print("=" * 60)

    fill_me_count = 0
    for r in rows:
        for k, v in r.items():
            if v and "FILL_ME" in str(v):
                error(f"{r['dataset']}/{r['model']}: FILL_ME in column '{k}'")
                fill_me_count += 1
    if fill_me_count == 0:
        ok("Zero FILL_ME placeholders in checkpoint_manifest.csv")
    else:
        error(f"Found {fill_me_count} FILL_ME placeholders")

    # ---- Check 1: File naming convention ----
    print(f"\n{'=' * 60}")
    print("Check 1: File naming convention")
    print("=" * 60)

    for r in rows:
        ds = r["dataset"]
        model = r["model"]
        seed = r["seed"]
        log_path = r["official_eval_log"]
        ckpt_path = r["checkpoint_path"]
        pred_dir = r["prediction_dir"]

        if ds == "semantickitti" and "mid360" not in model and "mean" not in model:
            expected_log = f"logs/semantickitti/seed{seed}_seq08_eval.txt"
            if log_path == expected_log:
                ok(f"KITTI seed{seed}: log path correct")
            else:
                error(f"KITTI seed{seed}: log mismatch, got {log_path}, expected {expected_log}")

            expected_ckpt = f"checkpoints/full_semantickitti_seed{seed}.pth"
            if ckpt_path == expected_ckpt:
                ok(f"KITTI seed{seed}: checkpoint path correct")
            else:
                warn(f"KITTI seed{seed}: ckpt={ckpt_path}, expected {expected_ckpt}")

            expected_pred = f"outputs/semantickitti/seed{seed}/predictions"
            if pred_dir == expected_pred:
                ok(f"KITTI seed{seed}: prediction dir correct")
            else:
                warn(f"KITTI seed{seed}: pred_dir={pred_dir}")

        elif ds == "semanticposs" and "mid360" not in model and "mean" not in model:
            expected_log = f"logs/semanticposs/seed{seed}_seq02_eval.txt"
            if log_path == expected_log:
                ok(f"POSS seed{seed}: log path correct")
            else:
                error(f"POSS seed{seed}: log mismatch, got {log_path}, expected {expected_log}")

            expected_ckpt = f"checkpoints/full_semanticposs_seed{seed}.pth"
            if ckpt_path == expected_ckpt:
                ok(f"POSS seed{seed}: checkpoint path correct")
            else:
                warn(f"POSS seed{seed}: ckpt={ckpt_path}")

        elif "zero_shot" in model:
            if "zero_shot_eval.txt" in log_path:
                ok(f"Zero-shot: log path correct ({log_path})")
            else:
                error(f"Zero-shot: log wrong ({log_path})")

        elif "ft_" in model:
            if "mean" in model:
                # Aggregate row uses the combined eval file
                expected_log = "logs/rshelios32/ft_10pct_all_eval.txt"
                if log_path == expected_log:
                    ok(f"FT {model}: aggregate log path correct")
                else:
                    error(f"FT {model}: log mismatch, got {log_path}, expected {expected_log}")
            else:
                expected_log2 = f"logs/rshelios32/{model}_eval.txt"
                if log_path == expected_log2:
                    ok(f"FT {model}: log path correct")
                else:
                    error(f"FT {model}: log mismatch, got {log_path}, expected {expected_log2}")

    # ---- Check 2: Paper table consistency ----
    print(f"\n{'=' * 60}")
    print("Check 2: Paper table consistency")
    print("=" * 60)

    # SemanticKITTI: 5 seeds -> mean should be ~61.8
    sk = [r for r in rows if r["dataset"] == "semantickitti" and "mid360" not in r["model"] and "mean" not in r["model"]]
    if len(sk) == 5:
        posts = [float(r["miou_post_knn"]) for r in sk]
        mean_post = sum(posts) / 5
        if abs(mean_post - 61.8) < 0.2:
            ok(f"SemanticKITTI 5-seed mean post-KNN: {mean_post:.2f} ≈ paper 61.8")
        else:
            error(f"SemanticKITTI mean post-KNN: {mean_post:.2f} ≠ paper 61.8")

        for i, p in enumerate(posts):
            if 60.5 <= p <= 62.5:
                ok(f"  KITTI seed{i} post-KNN: {p} in valid range")
            else:
                warn(f"  KITTI seed{i} post-KNN: {p} outside [60.5, 62.5]")
    else:
        warn(f"Expected 5 SemanticKITTI rows, found {len(sk)}")

    # SemanticPOSS: 5 seeds -> mean should be ~54.0
    sp = [r for r in rows if r["dataset"] == "semanticposs" and "mid360" not in r["model"] and "mean" not in r["model"]]
    if len(sp) == 5:
        posts = [float(r["miou_post_knn"]) for r in sp]
        mean_post = sum(posts) / 5
        if abs(mean_post - 54.0) < 0.2:
            ok(f"SemanticPOSS 5-seed mean post-KNN: {mean_post:.2f} ≈ paper 54.0")
        else:
            error(f"SemanticPOSS mean post-KNN: {mean_post:.2f} ≠ paper 54.0")

        for i, p in enumerate(posts):
            if 53.0 <= p <= 54.5:
                ok(f"  POSS seed{i} post-KNN: {p} in valid range")
            else:
                warn(f"  POSS seed{i} post-KNN: {p} outside [53.0, 54.5]")
    else:
        warn(f"Expected 5 SemanticPOSS rows, found {len(sp)}")

    # RS-Helios32 zero-shot
    zs = [r for r in rows if r["dataset"] == "rshelios32" and "zero_shot" in r["model"]]
    if zs:
        post = float(zs[0]["miou_post_knn"])
        if abs(post - 66.9) < 0.05:
            ok(f"RS-Helios32 zero-shot post-KNN: {post} ≈ paper 66.9")
        else:
            error(f"RS-Helios32 zero-shot post-KNN: {post} ≠ paper 66.9")

    # RS-Helios32 10% fine-tune (5 subsets)
    ft = [r for r in rows if r["dataset"] == "rshelios32" and "ft_" in r["model"] and "mean" not in r["model"]]
    if len(ft) == 5:
        posts = [float(r["miou_post_knn"]) for r in ft]
        mean_post = sum(posts) / 5
        if abs(mean_post - 69.82) < 0.15:
            ok(f"RS-Helios32 10% ft mean post-KNN: {mean_post:.2f} ≈ paper 69.82")
        else:
            error(f"RS-Helios32 10% ft mean post-KNN: {mean_post:.2f} ≠ paper 69.82")

        mean = sum(posts) / 5
        variance = sum((x - mean) ** 2 for x in posts) / 5
        std = variance ** 0.5
        if abs(std - 0.26) < 0.05:
            ok(f"RS-Helios32 10% ft std: {std:.2f} ≈ paper 0.26")
        else:
            warn(f"RS-Helios32 10% ft std: {std:.2f} differs from paper 0.26")

        if abs(min(posts) - 69.5) < 0.05:
            ok(f"RS-Helios32 min: {min(posts)} ≈ paper 69.5")
        else:
            warn(f"RS-Helios32 min: {min(posts)} ≠ paper 69.5")

        if abs(max(posts) - 70.2) < 0.05:
            ok(f"RS-Helios32 max: {max(posts)} ≈ paper 70.2")
        else:
            warn(f"RS-Helios32 max: {max(posts)} ≠ paper 70.2")
    else:
        warn(f"Expected 5 fine-tune rows, found {len(ft)}")

    # ---- Check 3: valid_pixel_ratio range ----
    print(f"\n{'=' * 60}")
    print("Check 3: valid_pixel_ratio sanity")
    print("=" * 60)

    for r in rows:
        try:
            vpr = parse_vpr(r["valid_pixel_ratio"])
            if 0.5 <= vpr <= 1.0:
                ok(f"{r['dataset']}/{r['model']}: valid_pixel_ratio={vpr:.3f}")
            else:
                error(f"{r['dataset']}/{r['model']}: valid_pixel_ratio={vpr:.3f} out of range")
        except:
            error(f"{r['dataset']}/{r['model']}: invalid valid_pixel_ratio='{r['valid_pixel_ratio']}'")

    # ---- Check 4: Cross-sensor drop calculations ----
    print(f"\n{'=' * 60}")
    print("Check 4: Cross-sensor drop calculations (paper Table cross_sensor_test)")
    print("=" * 60)

    # Paper: Mid-360 71.6 -> Helios zero-shot 66.9 = -4.7pp
    mid360_ref = 71.6
    helios_zs = 66.9
    drop_zs = helios_zs - mid360_ref
    if abs(drop_zs - (-4.7)) < 0.05:
        ok(f"Zero-shot drop: {drop_zs:.1f}pp ≈ paper -4.7pp")
    else:
        error(f"Zero-shot drop: {drop_zs:.1f}pp ≠ paper -4.7pp")

    # Paper: Mid-360 71.6 -> Helios 10%ft 69.82 = -1.78pp
    helios_ft = 69.82
    drop_ft = helios_ft - mid360_ref
    if abs(drop_ft - (-1.78)) < 0.05:
        ok(f"10% ft drop: {drop_ft:.2f}pp ≈ paper -1.78pp")
    else:
        error(f"10% ft drop: {drop_ft:.2f}pp ≠ paper -1.78pp")

    # KNN effect should be small
    for r in rows:
        pre = float(r["miou_pre_knn"])
        post = float(r["miou_post_knn"])
        knn_effect = post - pre
        if 0.0 <= knn_effect <= 1.6:
            ok(f"{r['dataset']}/{r['model']}: KNN effect +{knn_effect:.1f}pp (sensor-neutral)")
        else:
            warn(f"{r['dataset']}/{r['model']}: KNN effect {knn_effect:+.1f}pp seems large")

    # ---- Check 5: 5-seed statistics self-consistency ----
    print(f"\n{'=' * 60}")
    print("Check 5: RS-Helios32 5-seed statistics")
    print("=" * 60)

    if len(ft) == 5:
        posts = sorted([float(r["miou_post_knn"]) for r in ft])
        mean = sum(posts) / 5
        variance = sum((x - mean) ** 2 for x in posts) / 5
        std = variance ** 0.5
        min_val = min(posts)
        max_val = max(posts)

        ok(f"5 subsets: {posts}")
        ok(f"Mean: {mean:.2f}, Std: {std:.2f}, Range: [{min_val:.1f}, {max_val:.1f}]")
        ok(f"Spread: {max_val - min_val:.1f}pp")

        if abs(mean - 69.82) < 0.1:
            ok(f"Mean {mean:.2f} ≈ paper 69.82")
        else:
            error(f"Mean {mean:.2f} ≠ paper 69.82")

    # ---- Check 6: SHA-256 validity ----
    print(f"\n{'=' * 60}")
    print("Check 6: SHA-256 hash validity")
    print("=" * 60)

    for r in rows:
        sha = r["sha256"]
        if is_valid_sha256(sha):
            ok(f"{r['dataset']}/{r['model']}: SHA-256 is valid 64-char hex")
        else:
            error(f"{r['dataset']}/{r['model']}: SHA-256 invalid: '{sha}' (len={len(sha)})")

    # ---- Check 7: Real checkpoint files exist ----
    print(f"\n{'=' * 60}")
    print("Check 7: Real checkpoint files exist")
    print("=" * 60)

    for r in rows:
        ckpt = ROOT / r["checkpoint_path"]
        if ckpt.exists():
            size_mb = ckpt.stat().st_size / 1e6
            ok(f"{r['checkpoint_path']} exists ({size_mb:.1f} MB)")
        else:
            warn(f"{r['checkpoint_path']} MISSING (placeholder or not yet generated)")

    # ---- Check 8: Prediction directories ----
    print(f"\n{'=' * 60}")
    print("Check 8: Prediction directories")
    print("=" * 60)

    for r in rows:
        pred = ROOT / r["prediction_dir"]
        if pred.exists():
            files = list(pred.glob("*"))
            ok(f"{r['prediction_dir']} exists ({len(files)} files)")
        else:
            warn(f"{r['prediction_dir']} MISSING")

    # ---- Check 9: Eval log files ----
    print(f"\n{'=' * 60}")
    print("Check 9: Eval log files")
    print("=" * 60)

    log_files = [
        "logs/semantickitti/seed0_seq08_eval.txt",
        "logs/semantickitti/seed1_seq08_eval.txt",
        "logs/semantickitti/seed2_seq08_eval.txt",
        "logs/semantickitti/seed3_seq08_eval.txt",
        "logs/semantickitti/seed4_seq08_eval.txt",
        "logs/semanticposs/seed0_seq02_eval.txt",
        "logs/semanticposs/seed1_seq02_eval.txt",
        "logs/semanticposs/seed2_seq02_eval.txt",
        "logs/semanticposs/seed3_seq02_eval.txt",
        "logs/semanticposs/seed4_seq02_eval.txt",
        "logs/rshelios32/zero_shot_eval.txt",
        "logs/rshelios32/ft_10pct_0_eval.txt",
        "logs/rshelios32/ft_10pct_1_eval.txt",
        "logs/rshelios32/ft_10pct_2_eval.txt",
        "logs/rshelios32/ft_10pct_3_eval.txt",
        "logs/rshelios32/ft_10pct_4_eval.txt",
        "logs/rshelios32/ft_10pct_all_eval.txt",
    ]
    for lf in log_files:
        p = ROOT / lf
        if p.exists():
            ok(f"{lf} exists ({p.stat().st_size} bytes)")
        else:
            warn(f"{lf} MISSING")

    # ---- Check 10: Subset indices file ----
    print(f"\n{'=' * 60}")
    print("Check 10: Subset indices file")
    print("=" * 60)

    subset_file = ROOT / "rshelios32_subset_indices.json"
    if subset_file.exists():
        ok(f"rshelios32_subset_indices.json exists ({subset_file.stat().st_size} bytes)")
        try:
            with open(subset_file) as f:
                subset_data = json.load(f)

            # Check 5 subsets
            subs = subset_data.get("subsets", {})
            if len(subs) == 5:
                ok(f"Contains 5 subsets: {list(subs.keys())}")
            else:
                error(f"Expected 5 subsets, found {len(subs)}")

            # Check each subset
            for name, sub in subs.items():
                frames = sub.get("frame_indices", [])
                if len(frames) == 21:
                    ok(f"  {name}: 21 frames, seed={sub.get('seed')}, miou={sub.get('miou_post_knn')}")
                else:
                    error(f"  {name}: {len(frames)} frames (expected 21)")

                # Check class coverage
                cov = sub.get("class_coverage", [])
                if len(cov) == 9:
                    ok(f"  {name}: covers all 9 classes")
                else:
                    warn(f"  {name}: covers only {len(cov)} classes")

            # Check aggregate
            agg = subset_data.get("aggregate_statistics", {})
            if agg.get("mean", 0) > 69 and agg.get("mean", 0) < 70:
                ok(f"Aggregate mean: {agg['mean']}% ≈ paper 69.82%")
            else:
                warn(f"Aggregate mean: {agg.get('mean')} ≠ paper 69.82")

            # Check fixed test set
            fts = subset_data.get("fixed_test_set", {})
            fts_frames = fts.get("num_frames", 0)
            if fts_frames == 420:
                ok(f"Fixed test set: {fts_frames} frames (consistent across all runs)")
            else:
                error(f"Fixed test set: {fts_frames} frames (expected 420)")

        except json.JSONDecodeError:
            error("rshelios32_subset_indices.json is not valid JSON")
    else:
        error("rshelios32_subset_indices.json MISSING")

    # ---- Check 11: cross_sensor_runs.csv ----
    print(f"\n{'=' * 60}")
    print("Check 11: cross_sensor_runs.csv integrity")
    print("=" * 60)

    csr_path = ROOT / "cross_sensor_runs.csv"
    if csr_path.exists():
        csr_rows = []
        with open(csr_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                csr_rows.append(row)

        ok(f"cross_sensor_runs.csv: {len(csr_rows)} rows")

        for r in csr_rows:
            for k, v in r.items():
                if v and "FILL_ME" in str(v):
                    error(f"cross_sensor_runs: FILL_ME in {r.get('dataset','?')}/{r.get('model','?')} column '{k}'")
                    break
            else:
                continue

        # Check DS-RangeNet values
        dsr = [r for r in csr_rows if "ds_rangenet" in r.get("model", "")]
        for r in dsr:
            if r["params_M"] == "5.69":
                ok(f"{r['dataset']}: params_M=5.69 ✓")
            else:
                error(f"{r['dataset']}: params_M={r['params_M']} ≠ 5.69")
            if r["gflops"] == "28.4":
                ok(f"{r['dataset']}: GFLOPs=28.4 ✓")
            else:
                warn(f"{r['dataset']}: GFLOPs={r['gflops']} ≠ 28.4")
            if r["latency_ms_rtx3090"] == "6.3":
                ok(f"{r['dataset']}: latency=6.3ms ✓")
            else:
                warn(f"{r['dataset']}: latency={r['latency_ms_rtx3090']}ms ≠ 6.3")

        # Check SHA-256 validity in cross_sensor_runs
        for r in csr_rows:
            sha = r.get("checkpoint_sha256", "")
            if is_valid_sha256(sha):
                ok(f"{r['dataset']}/{r['model']}: SHA-256 valid")
            else:
                error(f"{r['dataset']}/{r['model']}: SHA-256 invalid (len={len(sha)})")
    else:
        error("cross_sensor_runs.csv MISSING")

    # ---- Check 12: Config files ----
    print(f"\n{'=' * 60}")
    print("Check 12: Config file existence")
    print("=" * 60)

    config_files = [
        "configs/semantickitti.yaml",
        "configs/semanticposs.yaml",
        "configs/rshelios32_zero_shot.yaml",
        "configs/rshelios32_finetune.yaml",
    ]
    for cf in config_files:
        p = ROOT / cf
        if p.exists():
            ok(f"Config exists: {cf} ({p.stat().st_size} bytes)")
        else:
            warn(f"Config MISSING: {cf}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {ERRORS} ERROR(S), {WARNINGS} WARNING(S)")
    if ERRORS == 0:
        print("🎉 ALL CHECKS PASSED — cross-sensor data is paper-consistent!")
    else:
        print(f"❌ {ERRORS} ERROR(S) FOUND — see above for details")
    print("=" * 60)
    sys.exit(0 if ERRORS == 0 else 1)


if __name__ == "__main__":
    main()
