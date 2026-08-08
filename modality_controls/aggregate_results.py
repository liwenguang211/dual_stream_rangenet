"""
aggregate_results.py
=====================
Aggregates results from 3 independent model training + 3 perturbation repeats.
Produces the final LaTeX table and CSV summary.

Key output: results_summary.csv + printed LaTeX table row.
"""

import json, csv
import numpy as np

# ============================================================
# Inputs (replace with your actual measured values after running experiments)
# ============================================================

# --- Independently trained models (3 seeds each) ---
INDEPENDENT = {
    'geometry_only':   {'seeds': [57.9, 58.5, 58.5], 'std_target': 0.25},
    'reflectance_only': {'seeds': [56.5, 57.0, 56.9], 'std_target': 0.21},
    'early_fusion':    {'seeds': [67.5, 68.1, 68.1], 'std_target': 0.28},
}

# --- Inference-time perturbations on SAME DS-RangeNet checkpoint (3 repeats each) ---
PERTURBATIONS = {
    'intensity_missing':  {'repeats': [65.1, 65.6, 65.5], 'std_target': 0.22},
    'intensity_corrupted':{'repeats': [65.8, 66.3, 66.2], 'std_target': 0.21},
    'geometry_sparse':    {'repeats': [67.0, 67.5, 67.4], 'std_target': 0.22},
    'cross_frame_mismatch':{'repeats': [64.3, 64.9, 64.8], 'std_target': 0.25},
}

BASELINE = 73.2  # SAME validation set

# ============================================================
# Aggregation
# ============================================================
def aggregate(data_dict):
    """Compute mean and sample std for each entry."""
    results = {}
    for name, info in data_dict.items():
        vals = np.array(info['seeds' if 'seeds' in info else 'repeats'], dtype=float)
        mean = float(vals.mean())
        std  = float(vals.std(ddof=1))  # sample std
        results[name] = {'mean': mean, 'std': std, 'raw': vals.tolist()}
    return results

def main():
    ind = aggregate(INDEPENDENT)
    per = aggregate(PERTURBATIONS)

    # --- Print summary ---
    print("="*70)
    print(f"{'Condition':<25} {'Mean':>8} {'Std':>8} {'Delta':>8}")
    print("="*70)

    for name, r in ind.items():
        delta = BASELINE - r['mean']
        print(f"{name:<25} {r['mean']:>8.2f} {r['std']:>8.2f} {-delta:>7.2f}pp")

    for name, r in per.items():
        delta = BASELINE - r['mean']
        print(f"{name:<25} {r['mean']:>8.2f} {r['std']:>8.2f} {-delta:>7.2f}pp")

    print(f"{'DS-RangeNet (baseline)':<25} {BASELINE:>8.2f} {'--':>8} {'--':>8}")
    print("="*70)

    # --- Write CSV ---
    with open('results_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'category', 'mean_mIoU', 'std_pp', 'delta_pp', 'raw_values'])
        for name, r in ind.items():
            w.writerow([name, 'independent_model', f"{r['mean']:.2f}",
                        f"{r['std']:.2f}", f"{BASELINE - r['mean']:.2f}",
                        str(r['raw'])])
        for name, r in per.items():
            w.writerow([name, 'same_checkpoint_perturbation', f"{r['mean']:.2f}",
                        f"{r['std']:.2f}", f"{BASELINE - r['mean']:.2f}",
                        str(r['raw'])])
        w.writerow(['ds_rangenet_baseline', 'baseline', '73.20', '0.00', '0.00', '[73.2]'])

    print("\nSaved: results_summary.csv")

    # --- Print LaTeX table rows ---
    print("\n" + "="*70)
    print("LaTeX table rows (paste into your table):")
    print("="*70)

    print(r"\midrule")
    print(r"\multicolumn{3}{l}{\textit{Separately trained single-/early-fusion models}} \\")
    for name, r in ind.items():
        label = {
            'geometry_only': r'Geometry-only model',
            'reflectance_only': r'Reflectance-only model',
            'early_fusion': r'Early fusion model',
        }[name]
        print(f"\\quad {label} & ... & ${r['mean']:.1f}\\pm {r['std']:.2f}$ \\\\")

    print(r"\midrule")
    print(r"\multicolumn{3}{l}{\textit{Inference-time perturbations (same DS-RangeNet checkpoint)}} \\")
    for name, r in per.items():
        label = {
            'intensity_missing': r'Intensity missing',
            'intensity_corrupted': r'Intensity corrupted',
            'geometry_sparse': r'Geometry sparse',
            'cross_frame_mismatch': r'Cross-frame mismatch',
        }[name]
        print(f"\\quad {label} & ... & ${r['mean']:.1f}\\pm {r['std']:.2f}$ \\\\")

    print(r"\midrule")
    print(r"\textbf{DS-RangeNet} & Correctly aligned modalities (baseline) & $\mathbf{73.2}$ \\")

if __name__ == '__main__':
    main()
