"""
run_all.py
===========
Orchestrates the full pipeline:
  1. Train 3 independent models (3 seeds each) → logs/
  2. Run 4 inference-time perturbations (3 repeats each) → logs/
  3. Aggregate results → results_summary.csv
  4. Verify self-consistency → exit code 0/1

Usage:
  python run_all.py --config config.yaml
"""

import subprocess, sys, os

def main():
    print("="*60)
    print("DS-RangeNet — Modality Control Experiments (v2)")
    print("="*60)

    # Step 1: Train independent models
    print("\n[Step 1/4] Training independent models...")
    subprocess.run([sys.executable, 'run_independent_models.py', '--config', 'config.yaml'], check=True)

    # Step 2: Run perturbations
    print("\n[Step 2/4] Running inference-time perturbations...")
    subprocess.run([sys.executable, 'apply_perturbations.py', '--config', 'config.yaml'], check=True)

    # Step 3: Aggregate
    print("\n[Step 3/4] Aggregating results...")
    subprocess.run([sys.executable, 'aggregate_results.py'], check=True)

    # Step 4: Verify
    print("\n[Step 4/4] Running self-consistency verification...")
    result = subprocess.run([sys.executable, 'verify_results.py'], capture_output=False)
    return result.returncode

if __name__ == '__main__':
    rc = main()
    sys.exit(rc)