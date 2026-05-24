#!/usr/bin/env python3
"""Generate summary statistics for stratified ACS experiment results.

Creates a summary table with mean coverage, median width, and standard deviations
for each method and o value combination.

Input:
    real_data/acs/results/stratified/acs_stratified_raw_results.csv

Output:
    real_data/acs/results/stratified/acs_stratified_summary.csv
    real_data/acs/results/stratified/acs_stratified_summary_long.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent.parent
RESULTS_FILE = BASE_DIR / "acs/results/stratified/acs_stratified_raw_results.csv"
SUMMARY_FILE = BASE_DIR / "acs/results/stratified/acs_stratified_summary.csv"
SUMMARY_LONG_FILE = BASE_DIR / "acs/results/stratified/acs_stratified_summary_long.csv"


def main():
    """Generate summary statistics."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    print(f"Loading results from: {RESULTS_FILE}")
    df = pd.read_csv(RESULTS_FILE)

    # Convert to numeric
    df["o"] = pd.to_numeric(df["o"], errors="coerce")
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["width_income"] = pd.to_numeric(df["width_income"], errors="coerce")

    print(f"Loaded {len(df)} rows")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"O values: {sorted(df['o'].unique())}")
    print(f"Replicates: {df['replicate'].nunique()}")

    # Compute summary statistics
    summary = df.groupby(["method", "o"], as_index=False).agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        coverage_median=("coverage", "median"),
        width_mean=("width", "mean"),
        width_std=("width", "std"),
        width_median=("width", "median"),
        width_income_mean=("width_income", "mean"),
        width_income_std=("width_income", "std"),
        width_income_median=("width_income", "median"),
        n_replicates=("replicate", "nunique"),
    )

    # Round to reasonable precision
    for col in summary.columns:
        if col not in ["method", "o", "n_replicates"]:
            summary[col] = summary[col].round(6)

    # Sort by method and o
    summary = summary.sort_values(["method", "o"]).reset_index(drop=True)

    # Save long-format summary
    print(f"\nSaving long-format summary to: {SUMMARY_LONG_FILE}")
    summary.to_csv(SUMMARY_LONG_FILE, index=False)
    print(f"Saved {len(summary)} rows")

    # Create wide-format pivot tables for coverage and width
    print(f"\nCreating pivot tables...")

    # Coverage pivot: rows = (method, o), columns = statistic
    cov_pivot = summary.pivot_table(
        index=["method", "o"],
        values=["coverage_mean", "coverage_std", "coverage_median"],
        aggfunc="first",
    )

    # Width pivot: rows = (method, o), columns = statistic
    width_pivot = summary.pivot_table(
        index=["method", "o"],
        values=["width_income_mean", "width_income_std", "width_income_median"],
        aggfunc="first",
    )

    # Combine coverage and width into one table
    combined = pd.concat([cov_pivot, width_pivot], axis=1)

    # Reorder columns for readability
    combined = combined[
        [
            "coverage_mean",
            "coverage_std",
            "coverage_median",
            "width_income_mean",
            "width_income_std",
            "width_income_median",
        ]
    ]

    # Save wide-format summary
    print(f"\nSaving wide-format summary to: {SUMMARY_FILE}")
    combined.to_csv(SUMMARY_FILE)
    print(f"Saved summary with {len(combined)} rows")

    # Print preview
    print("\n" + "=" * 80)
    print("SUMMARY PREVIEW (first 20 rows)")
    print("=" * 80)
    print(combined.head(20).to_string())
    print("\n" + "=" * 80)

    # Print summary statistics by method (averaged over all o)
    print("\nAVERAGE STATISTICS BY METHOD (across all o values)")
    print("=" * 80)
    method_avg = df.groupby("method", as_index=False).agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        width_income_mean=("width_income", "mean"),
        width_income_median=("width_income", "median"),
    ).round(4)
    print(method_avg.to_string(index=False))
    print("=" * 80)

    print("\nDone! Summary files saved:")
    print(f"  - {SUMMARY_FILE}")
    print(f"  - {SUMMARY_LONG_FILE}")


if __name__ == "__main__":
    main()
