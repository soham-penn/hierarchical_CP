#!/usr/bin/env python3
"""
Diagnose why D-HCP intervals miss the target more often at o=20 vs o=0.

Run a few replicates and save:
- Interval bounds (lower, upper)
- True target value
- Predicted mu_hat
- Coverage (0/1)

This will help us understand:
1. Is the target unusual/outlier?
2. Is mu_hat severely biased?
3. Are intervals too narrow?
4. Is there a systematic pattern?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs
from methods.mu_methods import create_mu_method_ols_global_only, create_mu_method_ols_offset
from methods.donor_hcp import compute_donor_hcp_randomized_interval

# Load data
acs_data_path = Path(__file__).parent / "acs" / "data" / "acs_data_all50states.csv"
df = load_and_clean_acs_pums(str(acs_data_path), states_keep=["CA"])
X = build_design_matrix_acs(df)

# Filter to eligible groups (size >= 20)
group_col = 'puma'
group_sizes = df.groupby(group_col).size()
eligible_groups = group_sizes[group_sizes >= 20].index.tolist()

print(f"Total eligible PUMAs: {len(eligible_groups)}")

# Run diagnostic on first 10 replicates
n_replicates = 10
o_values = [0, 5, 10, 15, 20]
target_index = 20
alpha = 0.2
alpha_sel = 0.5
seed = 456

results = []

for rep_idx in range(n_replicates):
    print(f"\n{'='*80}")
    print(f"Replicate {rep_idx}")
    print(f"{'='*80}")

    rng = np.random.default_rng(seed + rep_idx)

    # Select 20 calibration PUMAs + 1 test PUMA (with at least 21 observations)
    calib_groups = rng.choice(eligible_groups, size=20, replace=False).tolist()

    # Test group must have at least 21 observations
    calib_set = set(calib_groups)
    test_candidates = []
    for g in eligible_groups:
        if g not in calib_set:
            grp_mask = (df[group_col] == g).values
            grp_size = grp_mask.sum()
            if grp_size >= 21:
                test_candidates.append(g)

    if len(test_candidates) == 0:
        print("  No valid test candidates!")
        continue

    test_group = rng.choice(test_candidates)

    # Get data
    group_data = {}
    for grp in calib_groups + [test_group]:
        grp_mask = (df[group_col] == grp).values
        grp_idx = np.where(grp_mask)[0]
        group_data[grp] = {
            'X': X[grp_idx],
            'Y': df.iloc[grp_idx]['y'].values,
        }

    # Build Z_calibration
    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))
    U_test = np.zeros((1, 1))

    # Extract target
    x_target = group_data[test_group]['X'][target_index]
    true_y = group_data[test_group]['Y'][target_index]

    print(f"  Test group: {test_group}, size: {len(group_data[test_group]['Y'])}")
    print(f"  Target value (log1p income): {true_y:.4f}")
    print(f"  Target value (income): ${np.expm1(true_y):,.0f}")

    # Create mu_method
    mu_hcp = create_mu_method_ols_offset()

    # Test for each o value
    for o in o_values:
        # Build Z_test
        Z_test = [
            {'X': group_data[test_group]['X'][i], 'Y': group_data[test_group]['Y'][i]}
            for i in range(o)
        ]
        Z_test.append({'X': x_target, 'Y': true_y})

        # Compute D-HCP interval
        dhcp_seed = seed + rep_idx * 1009 + o
        try:
            res_dhcp = compute_donor_hcp_randomized_interval(
                U_calibration=U_calibration_full,
                Z_calibration=Z_calibration_full,
                U_test=U_test, Z_test=Z_test,
                o_observed=o, alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_hcp,
                random_seed=dhcp_seed,
                test_index_target=o,
            )

            interval = res_dhcp['interval']
            mu_hat = res_dhcp['mu_hat']
            lower, upper = interval
            width = upper - lower
            covered = 1 if lower <= true_y <= upper else 0

            # Convert to income scale for easier interpretation
            lower_income = np.expm1(lower) if not np.isinf(lower) else -np.inf
            upper_income = np.expm1(upper) if not np.isinf(upper) else np.inf
            mu_hat_income = np.expm1(mu_hat)
            true_income = np.expm1(true_y)
            width_income = upper_income - lower_income if np.isfinite(width) else np.inf

            # Compute bias and relative position
            bias = mu_hat - true_y
            bias_income = mu_hat_income - true_income

            if covered == 0:
                if true_y < lower:
                    miss_direction = "below"
                    distance = lower - true_y
                else:
                    miss_direction = "above"
                    distance = true_y - upper
            else:
                miss_direction = "covered"
                distance = 0.0

            print(f"    o={o:2d}: covered={covered}, mu_hat=${mu_hat_income:,.0f}, " +
                  f"width=${width_income:,.0f}, bias=${bias_income:,.0f}, {miss_direction}")

            results.append({
                'replicate': rep_idx,
                'test_group': test_group,
                'o': o,
                'true_y': true_y,
                'true_income': true_income,
                'mu_hat': mu_hat,
                'mu_hat_income': mu_hat_income,
                'lower': lower,
                'upper': upper,
                'lower_income': lower_income,
                'upper_income': upper_income,
                'width': width,
                'width_income': width_income,
                'bias': bias,
                'bias_income': bias_income,
                'covered': covered,
                'miss_direction': miss_direction,
                'distance': distance,
            })

        except Exception as e:
            print(f"    o={o:2d}: ERROR - {e}")

# Save results
results_df = pd.DataFrame(results)
out_file = Path(__file__).parent / "diagnostics" / "interval_diagnosis.csv"
out_file.parent.mkdir(exist_ok=True)
results_df.to_csv(out_file, index=False)

print(f"\n{'='*80}")
print(f"SUMMARY STATISTICS")
print(f"{'='*80}")

for o in o_values:
    subset = results_df[results_df['o'] == o]
    if len(subset) > 0:
        cov_mean = subset['covered'].mean()
        bias_mean = subset['bias_income'].mean()
        bias_std = subset['bias_income'].std()
        width_mean = subset['width_income'].mean()
        n_below = (subset['miss_direction'] == 'below').sum()
        n_above = (subset['miss_direction'] == 'above').sum()

        print(f"\no={o:2d}:")
        print(f"  Coverage: {cov_mean:.2f} ({subset['covered'].sum()}/{len(subset)})")
        print(f"  Bias: ${bias_mean:,.0f} ± ${bias_std:,.0f}")
        print(f"  Width: ${width_mean:,.0f}")
        print(f"  Misses: {n_below} below, {n_above} above")

print(f"\nResults saved to: {out_file}")
