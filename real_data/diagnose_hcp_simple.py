"""
Simple diagnostic to understand why HCP has so many infinite intervals.

This uses the donor-style split logic directly without loading data.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from methods.donor_hcp import get_donor_style_train_cal_split

def main():
    # Configuration matching ACS experiments
    alpha = 0.1
    alpha_sel = 0.1
    K_total = 20  # Total calibration groups

    print("="*60)
    print("HCP Infinite Interval Diagnostic")
    print("="*60)
    print(f"Configuration: K={K_total}, alpha={alpha}, alpha_selection={alpha_sel}")
    print()

    # Simulate different sample size scenarios
    scenarios = [
        ("Homogeneous (all size 100)", [100] * K_total),
        ("Homogeneous (all size 50)", [50] * K_total),
        ("Heterogeneous (uniform 20-200)", list(range(20, 200, 9))),
        ("Heterogeneous (realistic ACS)", [50, 60, 45, 80, 100, 120, 65, 75, 90, 110,
                                            55, 70, 85, 95, 105, 115, 125, 135, 60, 75]),
    ]

    for scenario_name, sample_sizes in scenarios:
        print(f"\n{scenario_name}")
        print(f"  Sample sizes: min={min(sample_sizes)}, max={max(sample_sizes)}, mean={np.mean(sample_sizes):.1f}")

        # Run 1000 replicates to check distribution
        K_cal_values = []
        for rep in range(1000):
            # Note: donor-style split has randomness from tie-breaking
            np.random.seed(rep)  # For reproducibility

            train_idx, calib_idx = get_donor_style_train_cal_split(
                sample_sizes=sample_sizes,
                o_observed=0,
                alpha_selection=alpha_sel
            )

            K_cal_values.append(len(calib_idx))

        K_cal_values = np.array(K_cal_values)

        # Compute weight on infinity for each
        weights_inf = 1.0 / (K_cal_values + 1)
        infinite_count = (weights_inf >= alpha).sum()

        print(f"  Distribution of K_cal:")
        print(f"    Min={K_cal_values.min()}, Max={K_cal_values.max()}, Mean={K_cal_values.mean():.2f}")
        print(f"  Infinite intervals: {infinite_count}/1000 ({100*infinite_count/1000:.1f}%)")

        # Show detailed histogram
        unique_vals, counts = np.unique(K_cal_values, return_counts=True)
        if len(unique_vals) <= 10:
            print(f"  Histogram:")
            for val, cnt in zip(unique_vals, counts):
                weight = 1.0 / (val + 1)
                infinite_marker = " ***INF***" if weight >= alpha else ""
                print(f"    K_cal={val:2d}: {cnt:4d} ({100*cnt/1000:5.1f}%) [weight={weight:.4f}]{infinite_marker}")

    # Now let's investigate the specific case that matches the user's calculation
    print(f"\n{'='*60}")
    print(f"User's Expectation Analysis")
    print(f"{'='*60}")
    print(f"User expected: K_cal = K/2 = {K_total}/2 = {K_total//2}")
    print(f"This would give weight_inf = 1/(K_cal+1) = 1/{K_total//2 + 1} = {1.0/(K_total//2 + 1):.4f}")
    print(f"Since {1.0/(K_total//2 + 1):.4f} < {alpha}, intervals should be FINITE")
    print()
    print(f"BUT: Let's check what actually happens...")

    # Check homogeneous case at o=0
    sample_sizes = [100] * K_total
    train_idx, calib_idx = get_donor_style_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel
    )

    K_cal = len(calib_idx)
    weight_inf = 1.0 / (K_cal + 1)

    print(f"\nHomogeneous case (all N_j=100, o=0):")
    print(f"  K_train = {len(train_idx)}")
    print(f"  K_cal = {K_cal}")
    print(f"  weight_inf = 1/(K_cal+1) = 1/{K_cal+1} = {weight_inf:.4f}")
    print(f"  alpha = {alpha}")

    if weight_inf >= alpha:
        print(f"\n  *** ISSUE CONFIRMED ***")
        print(f"  K_cal={K_cal} is LESS than expected K/2={K_total//2}")
        print(f"  This makes weight_inf ({weight_inf:.4f}) >= alpha ({alpha})")
        print(f"  Result: HCP intervals are INFINITE")
    else:
        print(f"\n  *** NO ISSUE ***")
        print(f"  K_cal={K_cal} gives weight_inf ({weight_inf:.4f}) < alpha ({alpha})")
        print(f"  Result: HCP intervals should be FINITE")

    # Let's trace through the donor-style split logic manually
    print(f"\n{'='*60}")
    print(f"Manual Trace of Donor-Style Split Logic")
    print(f"{'='*60}")

    # Manually compute S_tilde
    N = np.array(sample_sizes, dtype=int)
    o_observed = 0

    # From donor_hcp.py logic:
    # S_tilde = {j : N_j >= (1 - alpha_selection) * (N_j + o_observed)}
    # At o=0: N_j >= (1 - alpha_selection) * N_j
    # This simplifies to: N_j >= (1 - 0.1) * N_j = 0.9 * N_j
    # Which is always true for N_j > 0!

    threshold = (1 - alpha_sel) * (N + o_observed)
    S_tilde = np.where(N >= threshold)[0]

    print(f"S_tilde selection:")
    print(f"  Condition: N_j >= (1-alpha_sel)*(N_j+o) = {1-alpha_sel}*(N_j+{o_observed})")
    print(f"  At o=0: N_j >= {1-alpha_sel}*N_j")
    print(f"  For N_j=100: 100 >= {1-alpha_sel}*100 = {(1-alpha_sel)*100}")
    print(f"  S_tilde = ALL {len(S_tilde)} groups (indices {S_tilde[:5]}...)")

    print(f"\nDonor selection:")
    print(f"  Randomly pick one group from S_tilde as donor")
    print(f"  donor ∈ training set")
    print(f"  All OTHER groups in S_tilde → calibration set")
    print(f"  Result: K_cal = |S_tilde| - 1 = {len(S_tilde)} - 1 = {len(S_tilde)-1}")

    actual_K_cal = len(S_tilde) - 1
    actual_weight = 1.0 / (actual_K_cal + 1)

    print(f"\nFinal calculation:")
    print(f"  K_cal = {actual_K_cal}")
    print(f"  weight_inf = 1/(K_cal+1) = 1/{actual_K_cal+1} = {actual_weight:.4f}")
    print(f"  Compare with alpha = {alpha}")
    print(f"  Infinite intervals? {actual_weight >= alpha} (weight_inf {'>=':'<'[actual_weight < alpha]} alpha)")

if __name__ == '__main__':
    main()
