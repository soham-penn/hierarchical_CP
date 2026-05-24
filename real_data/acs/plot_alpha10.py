#!/usr/bin/env python3
"""Generate plots for alpha=0.1 results."""

from pathlib import Path
import sys

# Update the module paths before importing
script_path = Path(__file__).resolve()
sys.path.insert(0, str(script_path.parent))

import plot_true_marginal_paper_style as plotter

# Override paths for alpha=0.1
BASE_DIR = script_path.parent.parent
plotter.RESULTS_FILE = BASE_DIR / "acs/results/true_marginal_alpha10/acs_true_marg_alpha10_detailed.csv"
plotter.OUTPUT_DIR = BASE_DIR / "acs/NEW_PLOTS/true_marginal_alpha10/paper_style"
plotter.SUMMARY_DIR = BASE_DIR / "acs/results/true_marginal_alpha10/summaries"
plotter.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plotter.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import pandas as pd

    print("=" * 70)
    print("ACS TRUE MARGINAL COVERAGE - Paper Plots (ALPHA=0.1)")
    print("=" * 70)

    if not plotter.RESULTS_FILE.exists():
        print(f"\nError: Results file not found: {plotter.RESULTS_FILE}")
        exit(1)

    print(f"\nLoading results from: {plotter.RESULTS_FILE}")
    df = pd.read_csv(plotter.RESULTS_FILE)

    print(f"  Total rows: {len(df)}")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  o values: {sorted(df['o'].unique())}")
    print(f"  Replicates: {df['replicate'].nunique()}")

    # Generate plots
    plotter.generate_all_plots(df)

    # Generate summary tables
    plotter.generate_summary_tables(df)

    print(f"\nOutput directory: {plotter.OUTPUT_DIR}")
    print(f"Summary directory: {plotter.SUMMARY_DIR}")
    print("\nDone!")
