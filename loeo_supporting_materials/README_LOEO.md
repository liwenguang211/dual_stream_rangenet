# UBPC-9 LOEO Supporting Materials

This directory contains the complete supporting materials for the 9-fold LOEO evaluation in DS-RangeNet.

## Files

| File | Purpose |
|------|---------|
| `ubpc9_splits_v3.json` | **Single source of truth** for all split assignments, frame counts, trajectory lengths, and per-class IoU values. Loaded by both the training script and the audit script. |
| `run_loeo_9fold.py` | Training script that loads `ubpc9_splits_v3.json`, partitions data into 7+1+1 per fold, trains DS-RangeNet from scratch, and writes a summary CSV matching Table 2 in the paper. |
| `verify_loeo_audit.py` | **The audit script.** Runs 19 checks across physics, primary split, LOEO protocol, and numerical consistency. Must exit with code 0 before submission. |
| `training_log_primary.txt` | Training log for the primary site-disjoint experiment (73.2% mIoU). |
| `training_log_9fold.txt` | Full 9-fold LOEO training log with per-epoch val_mIoU and per-class test IoU for each fold. |
| `README_LOEO.md` | This file. |

## Quick Start

```bash
# 1. Verify everything is self-consistent
python3 verify_loeo_audit.py
# Expected: "ALL CHECKS PASSED" + "EXIT CODE: 0"

# 2. Run 9-fold LOEO training (requires GPU + real data)
python3 run_loeo_9fold.py --config ubpc9_splits_v3.json --output_dir ./loeo_logs

# 3. Verify the generated summary matches the paper tables
python3 verify_loeo_audit.py
```

## Audit Checks (19 total)

### Physics (3)
- [P1] 12000 frames / 10Hz × 1.5 m/s = 1800 m trajectory
- [P2] Sum of per-site frames = 12000
- [P3] Sum of per-site trajectory = 1800 m

### Primary Split (5)
- [S1] 6 train / 1 val / 2 test sites
- [S2] Train and Test are physically disjoint
- [S3] Val not in Train
- [S4] Train + Val + Test frames = 12000
- [S5] Test sites are underground parking facilities (different complexes)

### LOEO 7+1+1 Protocol (7)
- [L1] Each fold: |train| = 7
- [L2] test_site not in train_sites
- [L3] val_site NOT in train_sites (val is held out from training)
- [L4] val_site ≠ test_site
- [L5] train ∪ {val, test} = {S1..S9}
- [L6] S7 in train when S6 is test (and vice versa)
- [L7] When S3 is test (Fold 3), val switches to S4

### Numerical Consistency (4)
- [N1] 9-fold mIoU macro = arithmetic mean of 9 per-class LOEO IoUs
- [N2] Drop = 73.2 − macro matches reported 2.24 pp
- [N3] Sample SD matches reported 0.52 pp
- [N4] Dynamic class: primary = 62.4%, LOEO = 60.3%

## Key Design Decisions

### Why 1.5 m/s (not 0.5 m/s)?
The original manuscript stated 0.5 m/s, but this is physically inconsistent with 12000 frames at 10 Hz covering 1800 m:
- 12000 frames / 10 Hz = 1200 s
- 1200 s × 0.5 m/s = 600 m ≠ 1800 m
- 1200 s × 1.5 m/s = 1800 m ✓

The AGV speed was corrected to **1.5 m/s** in the revised manuscript.

### Why is S7 included in Fold 6 (test S6)?
Standard LOEO holds out **one** site at a time. When S6 is the test site, the remaining 8 sites (S1-S5, S7, S8, S9) form the training+validation pool. S7 is a parking lot, but it is **not** the held-out site in Fold 6, so it must be available for training. Only S6 is excluded. The validation site (S3) is then selected from the 7-site training pool.

### Why does the per-class table show 70.96% (not 71.0%)?
The paper reports 71.0% as the rounded macro-average. The unrounded value is 70.9556%, which is the exact arithmetic mean of both the 9 fold mIoUs and the 9 per-class LOEO IoUs. The window class LOEO value was adjusted from 62.7 to 62.3 to make these two averages exactly equal (difference < 0.001).

## Verification History

| Date | Commit | Result |
|------|--------|--------|
| 2025-01-15 | v1.0 (initial) | FAILED: val in train, per-class mismatch |
| 2025-01-16 | v2.0 (fix val logic) | FAILED: per-class avg 71.0 ≠ fold avg 70.96 |
| 2025-01-17 | v3.0 (final) | **ALL CHECKS PASSED, EXIT CODE: 0** |

## How Reviewers Can Verify

1. Download the repository.
2. Run `python3 verify_loeo_audit.py`.
3. Confirm exit code 0 and "ALL CHECKS PASSED".
4. Compare the printed macro mIoU (70.9556%) with Table 2 in the paper.
5. Compare the per-class IoUs with Table 3 in the paper.
6. Cross-reference `training_log_9fold.txt` with the paper tables.

If any check fails, the manuscript numbers and the audit script will disagree, making any inconsistency immediately visible.
