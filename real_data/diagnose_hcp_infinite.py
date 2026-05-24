"""
Diagnostic script to understand why HCP has so many infinite intervals.

This script will:
1. Load one replicate of ACS data
2. Manually compute the donor-style split at o=0
3. Count the number of calibration groups
4. Compute the weight on infinity: 1/(K_cal + 1)
5. Compare with alpha to see if infinite intervals are expected
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from acs.data_processing import load_and_clean_acs_pums
from methods.donor_hcp import get_donor_style_train_cal_split

def main():
    print("Loading ACS data...")
    base_dir = Path(__file__).parent
    df = load_and_clean_acs_pums(base_dir / 'acs_pums')

    print(f"Loaded {len(df)} observations")

    # Configuration (matching experiment)
    group_col = 'PUMA'
    target_col = 'log1p_income'
    alpha = 0.1
    alpha_sel = 0.1
    seed = 42
    K_total = 20  # Total number of groups to select

    rng = np.random.default_rng(seed)

    # Sample K groups
    unique_groups = df[group_col].unique()
    print(f"Total unique groups: {len(unique_groups)}")

    # Select K groups with at least 21 observations
    eligible_groups = []
    for g in unique_groups:
        grp_mask = (df[group_col] == g).values
        if grp_mask.sum() >= 21:
            eligible_groups.append(g)

    print(f"Groups with >= 21 observations: {len(eligible_groups)}")

    # Sample K_total groups for calibration
    calib_groups = rng.choice(eligible_groups, size=K_total, replace=False).tolist()

    # Collect sample sizes
    sample_sizes = []
    for g in calib_groups:
        grp_mask = (df[group_col] == g).values
        sample_sizes.append(grp_mask.sum())

    print(f"\nCalibration groups selected: {K_total}")
    print(f"Sample sizes: min={min(sample_sizes)}, max={max(sample_sizes)}, mean={np.mean(sample_sizes):.1f}")
    print(f"Sample size distribution:")
    print(f"  Q1={np.percentile(sample_sizes, 25):.0f}, "
          f"Q2={np.percentile(sample_sizes, 50):.0f}, "
          f"Q3={np.percentile(sample_sizes, 75):.0f}")

    # Compute donor-style split at o=0
    print(f"\n{'='*60}")
    print(f"Computing donor-style split with o=0, alpha_selection={alpha_sel}")
    print(f"{'='*60}")

    train_idx, calib_idx = get_donor_style_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel
    )

    K_train = len(train_idx)
    K_cal = len(calib_idx)

    print(f"\nSplit results:")
    print(f"  Training groups: {K_train}")
    print(f"  Calibration groups: {K_cal}")
    print(f"  Total: {K_train + K_cal} (should be {K_total})")

    # Compute weight on infinity
    weight_inf = 1.0 / (K_cal + 1)

    print(f"\n{'='*60}")
    print(f"HCP Interval Analysis")
    print(f"{'='*60}")
    print(f"Number of calibration groups K_cal: {K_cal}")
    print(f"Weight on infinity: 1/(K_cal+1) = 1/{K_cal+1} = {weight_inf:.4f}")
    print(f"Alpha: {alpha}")
    print(f"\nIf weight_inf < alpha ({weight_inf:.4f} < {alpha}), intervals should be FINITE")
    print(f"If weight_inf >= alpha ({weight_inf:.4f} >= {alpha}), intervals will be INFINITE")

    if weight_inf >= alpha:
        print(f"\n*** FOUND THE ISSUE! ***")
        print(f"Weight on infinity ({weight_inf:.4f}) >= alpha ({alpha})")
        print(f"This means HCP quantile will select the infinity score!")
        print(f"With only {K_cal} calibration groups, infinite intervals are EXPECTED.")
    else:
        print(f"\n*** NO ISSUE DETECTED ***")
        print(f"Weight on infinity ({weight_inf:.4f}) < alpha ({alpha})")
        print(f"Intervals should be finite. Need to investigate further.")

    # Let's run multiple replicates to see distribution
    print(f"\n{'='*60}")
    print(f"Running 100 replicates to check variability...")
    print(f"{'='*60}")

    K_cal_values = []
    for rep in range(100):
        rng_rep = np.random.default_rng(seed + rep)

        # Sample new calibration groups
        calib_groups_rep = rng_rep.choice(eligible_groups, size=K_total, replace=False).tolist()

        # Collect sample sizes
        sizes_rep = []
        for g in calib_groups_rep:
            grp_mask = (df[group_col] == g).values
            sizes_rep.append(grp_mask.sum())

        # Compute split
        _, calib_idx_rep = get_donor_style_train_cal_split(
            sample_sizes=sizes_rep,
            o_observed=0,
            alpha_selection=alpha_sel
        )

        K_cal_values.append(len(calib_idx_rep))

    K_cal_values = np.array(K_cal_values)

    print(f"\nDistribution of K_cal across 100 replicates:")
    print(f"  Min: {K_cal_values.min()}")
    print(f"  Q1: {np.percentile(K_cal_values, 25):.1f}")
    print(f"  Median: {np.percentile(K_cal_values, 50):.1f}")
    print(f"  Q3: {np.percentile(K_cal_values, 75):.1f}")
    print(f"  Max: {K_cal_values.max()}")
    print(f"  Mean: {K_cal_values.mean():.2f}")

    # Count how many have infinite intervals
    weights_inf = 1.0 / (K_cal_values + 1)
    infinite_count = (weights_inf >= alpha).sum()

    print(f"\nReplicates with infinite HCP intervals:")
    print(f"  {infinite_count}/100 ({100*infinite_count/100:.1f}%)")
    print(f"  Threshold: K_cal <= {int(1/alpha - 1)} gives infinite intervals")
    print(f"  (because 1/(K_cal+1) >= {alpha} when K_cal <= {int(1/alpha - 1)})")

    # Show histogram
    unique_vals, counts = np.unique(K_cal_values, return_counts=True)
    print(f"\nHistogram of K_cal:")
    for val, cnt in zip(unique_vals, counts):
        weight = 1.0 / (val + 1)
        infinite_marker = " *** INFINITE ***" if weight >= alpha else ""
        print(f"  K_cal={val:2d}: {cnt:3d} replicates ({100*cnt/100:5.1f}%) "
              f"[weight_inf={weight:.4f}]{infinite_marker}")

if __name__ == '__main__':
    main()
