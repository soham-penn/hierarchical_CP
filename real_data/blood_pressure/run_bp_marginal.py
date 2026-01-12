"""
Blood Pressure Marginal Coverage Experiments

This script implements marginal coverage evaluation where:
1. Use treatment arm only
2. For each test clinic, use ALL observations except the last one as calibration
3. Order observations by baseline SBP (to get percentiles)
4. Test coverage at 0th, 25th, 50th, 75th percentile points
5. Predict the NEXT observation after each percentile
6. Aggregate coverage across clinics by percentile

Experiment Design:
- 32 clinics total in treatment arm
- Training: 17 clinics (fixed)
- Test: 15 clinics
- Alpha: 0.2 (80% coverage target)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import methods
sys.path.append(str(Path(__file__).parent.parent.parent))

from methods import (
    create_mu_method_ols_offset,
    create_mu_method_ols_global_only
)
from methods.hcp_plus import compute_hcp_plus_interval
from methods.hcp_sample import compute_hcp_sample_interval
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius
)
from scores import absolute_residual_score

from data_processing import (
    load_and_clean_bp_data,
    build_design_matrix_bp
)

sys.path.append(str(Path(__file__).parent.parent))
from format_results import csv_to_markdown_table, add_data_summary_to_markdown


def run_marginal_experiment_one_clinic(
    df, X, training_clinics, test_clinic,
    alpha=0.2,
    alpha_selection=0.5,
    n_subsample_rep=50,
    mu_method_baseline=None,
    mu_method_hcp=None
):
    """
    Run marginal coverage experiment for ONE test clinic.

    For the test clinic:
    1. Order observations by baseline_sbp (for percentile calculation)
    2. Use all but last observation for calibration
    3. Test coverage at 0th, 25th, 50th, 75th percentile + 1
    4. Record coverage indicator for each percentile

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned BP data
    X : np.ndarray
        Design matrix
    training_clinics : list
        Clinics to use for calibration (fixed)
    test_clinic : int/str
        Clinic to test on
    alpha : float
        Miscoverage level
    alpha_selection : float
        Selection parameter
    n_subsample_rep : int
        Subsampling repetitions
    mu_method_baseline : dict
        Baseline μ-method
    mu_method_hcp : dict
        HCP μ-method

    Returns:
    --------
    pd.DataFrame : Results with one row per (percentile, method)
    """
    # Get test clinic data (original order, no permutation)
    test_df = df[df['clinic_id'] == test_clinic].reset_index(drop=True)
    n_test = len(test_df)

    if n_test < 5:
        print(f"  Warning: Test clinic {test_clinic} has only {n_test} observations, skipping")
        return pd.DataFrame()

    print(f"  Test clinic {test_clinic}: {n_test} observations (baseline SBP range: {test_df['baseline_sbp'].min():.1f}-{test_df['baseline_sbp'].max():.1f})")

    # Sort by baseline_sbp to find which observations are at SBP percentiles
    # Create mapping from sorted position to original position
    test_df_sorted = test_df.sort_values('baseline_sbp').reset_index(drop=False)
    # test_df_sorted['index'] now contains the original positions in test_df

    # Create BASE calibration groups from training clinics (fixed across all percentiles)
    Z_calibration_base = []
    cal_clinics_used_base = []

    for clinic in training_clinics:
        clinic_df = df[df['clinic_id'] == clinic]
        if len(clinic_df) < 1:
            continue

        Z_group = []
        clinic_indices = clinic_df.index.tolist()
        for idx in clinic_indices:
            Z_group.append({
                'X': X[idx, :],
                'Y': df.iloc[idx]['y']
            })

        Z_calibration_base.append(Z_group)
        cal_clinics_used_base.append(clinic)

    # Store test indices for use in percentile loop
    test_indices = test_df.index.tolist()
    U_test = np.zeros((1, 1))

    # Determine which observations to test (at baseline SBP percentiles)
    # Sort by SBP to find percentiles, then map back to original indices
    percentiles = [0, 25, 50, 75]
    percentile_indices = {}

    for pct in percentiles:
        if pct == 0:
            sorted_idx = 0
        else:
            sorted_idx = int(np.percentile(np.arange(n_test), pct))

        # Map from sorted position to original position in test_df
        original_idx = test_df_sorted.iloc[sorted_idx]['index']

        # Check if we have enough observations for history
        if original_idx < 1:  # Need at least 1 history observation
            continue

        percentile_indices[pct] = original_idx

    print(f"    Testing at baseline SBP percentiles: {list(percentile_indices.keys())}")

    # Run experiments for each percentile
    all_results = []

    for pct, target_index in percentile_indices.items():
        # Calibration: ONLY the 17 FIXED training clinics
        Z_calibration = list(Z_calibration_base)
        n_cal = len(Z_calibration)
        U_calibration = np.zeros((n_cal, 1))

        # target_index = index of the observation we want to predict
        # History = observations 0 through target_index-1 (target_index observations)
        # o_observed = target_index (number of history observations)
        # Z_test = history + target = observations 0 through target_index (inclusive)

        # At 0th percentile: target_index=0, no history, predict observation 0
        # At 25th percentile (10 obs): target_index=2, history=[0,1], predict observation 2

        o_observed = target_index  # Number of observations in history

        # Build Z_test: history (indices 0 to target_index-1) + target (index target_index)
        Z_test = []
        for i in range(target_index + 1):  # Includes target
            if i >= n_test:
                break
            idx = test_indices[i]
            Z_test.append({
                'X': X[idx, :],
                'Y': df.iloc[idx]['y']
            })

        # Get target observation
        target_idx = test_indices[target_index]
        true_y = df.iloc[target_idx]['y']
        x_target = X[target_idx, :]

        # Baseline methods: split training groups, compute fixed radius
        K = n_cal
        K0 = K // 2
        train_idx = list(range(K0))
        calib_idx = list(range(K0, K))

        # Fit baseline model
        model_baseline = mu_method_baseline['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=train_idx
        )

        # Compute scores
        scores_list = []
        for j in calib_idx:
            Zj = Z_calibration[j]
            Uj = U_calibration[j, :]
            yj = np.array([z['Y'] for z in Zj])
            muj = np.array([
                mu_method_baseline['predict_global'](
                    model_global=model_baseline,
                    x_vector=z['X'],
                    u_vector=Uj
                ) for z in Zj
            ])
            scores = absolute_residual_score(yj, muj)
            scores_list.append(scores)

        # Compute interval radii
        T_hcp = compute_hcp_interval_radius(scores_list, alpha)
        T_pool = compute_pooling_interval_radius(scores_list, alpha)
        T_sub = compute_subsampling_once_interval_radius(scores_list, alpha)
        T_rep = compute_repeated_subsampling_interval_radius(scores_list, alpha, n_subsample_rep)

        # Baseline prediction (global estimate only, no within-group offset)
        mu_hat_baseline = mu_method_baseline['predict_global'](
            model_global=model_baseline,
            x_vector=x_target,
            u_vector=U_test[0, :]
        )

        # Helper function
        def interval_from_radius(center, radius):
            if np.isinf(radius):
                return (-np.inf, np.inf)
            return (center - radius, center + radius)

        # HCP++ interval
        try:
            res_pp = compute_hcp_plus_interval(
                U_calibration=U_calibration,
                Z_calibration=Z_calibration,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o_observed,
                alpha=alpha,
                alpha_selection=alpha_selection,
                mu_method=mu_method_hcp
            )
            int_pp = res_pp['interval']
            mu_hat_hcp_methods = res_pp.get('mu_hat', mu_hat_baseline)
        except Exception as e:
            print(f"      Warning: HCP++ failed at pct={pct}: {e}")
            int_pp = (-np.inf, np.inf)
            mu_hat_hcp_methods = mu_hat_baseline

        # HCP.sample interval (uses same mu as HCP++)
        try:
            res_hs = compute_hcp_sample_interval(
                U_calibration=U_calibration,
                Z_calibration=Z_calibration,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o_observed,
                alpha=alpha,
                test_index_target=target_index,
                alpha_selection=alpha_selection,
                mu_method=mu_method_hcp
            )
            int_hs = res_hs['interval']
        except Exception as e:
            print(f"      Warning: HCP.sample failed at pct={pct}: {e}")
            int_hs = (-np.inf, np.inf)

        # Baseline intervals (all use baseline mu estimate)
        int_hcp = interval_from_radius(mu_hat_baseline, T_hcp)
        int_pool = interval_from_radius(mu_hat_baseline, T_pool)
        int_sub = interval_from_radius(mu_hat_baseline, T_sub)
        int_rep = interval_from_radius(mu_hat_baseline, T_rep)

        # Check coverage
        def check_coverage(interval, true_value):
            return interval[0] <= true_value <= interval[1]

        def compute_width(interval):
            if np.isfinite(interval[0]) and np.isfinite(interval[1]):
                return interval[1] - interval[0]
            return np.nan

        # Record results
        for method, interval, mu_val in [
            ('HCP++', int_pp, mu_hat_hcp_methods),
            ('HCP.sample', int_hs, mu_hat_hcp_methods),
            ('HCP', int_hcp, mu_hat_baseline),
            ('Pooling', int_pool, mu_hat_baseline),
            ('Subsampling', int_sub, mu_hat_baseline),
            ('Repeated', int_rep, mu_hat_baseline)
        ]:
            all_results.append({
                'test_clinic': test_clinic,
                'percentile': pct,
                'n_observed': o_observed + 1,
                'method': method,
                'covered': int(check_coverage(interval, true_y)),
                'width': compute_width(interval),
                'infinite': int(not (np.isfinite(interval[0]) and np.isfinite(interval[1]))),
                'lower': interval[0],
                'upper': interval[1],
                'true_y': true_y,
                'mu_hat': mu_val,
                'n_cal_groups': n_cal,
                'n_test_observed': len(Z_test),
                'n_test_total': n_test
            })

    return pd.DataFrame(all_results)


def main():
    """Main function - run marginal experiments."""
    import argparse

    parser = argparse.ArgumentParser(description='Run BP marginal coverage experiments')
    parser.add_argument('bp_csv', type=str, help='Path to BP CSV file')
    parser.add_argument('--output_dir', type=str, default='results/marginal',
                       help='Output directory for results')
    parser.add_argument('--n_test_clinics', type=int, default=15,
                       help='Number of test clinics (default: 15)')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Miscoverage level (default: 0.2 for 80%% coverage)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 80)
    print("BLOOD PRESSURE MARGINAL COVERAGE EXPERIMENTS")
    print("=" * 80)

    # Load and filter data
    print("\n1. Loading and cleaning BP data...")
    print("   Treatment arm only, no additional filters")

    df = load_and_clean_bp_data(
        args.bp_csv,
        treatment_arm_only=True,
        outcome_type='followup',
        min_clinic_size=5
    )

    # Reset index
    df = df.reset_index(drop=True)

    # Build design matrix
    print("\n2. Building design matrix...")
    X = build_design_matrix_bp(df)
    print(f"   Design matrix shape: {X.shape}")

    # Select training and test clinics
    print("\n3. Selecting training and test clinics...")
    clinic_counts = df.groupby('clinic_id').size().sort_values(ascending=False)
    print(f"   Total clinics: {len(clinic_counts)}")
    print(f"   Clinic sizes:\n{clinic_counts}")

    # Test clinics: top N by size
    test_clinics = clinic_counts.head(args.n_test_clinics).index.tolist()
    print(f"\n   Test clinics (top {args.n_test_clinics} by size): {test_clinics}")

    # Training clinics: remaining
    training_clinics = [c for c in clinic_counts.index if c not in test_clinics]
    print(f"   Training clinics ({len(training_clinics)}): {training_clinics}")

    # Create μ-methods (using OLS)
    print("\n4. Creating μ-estimation methods (OLS)...")
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    # Run marginal experiments
    print(f"\n5. Running marginal experiments ({len(test_clinics)} test clinics)...")
    all_results = []

    for test_clinic in test_clinics:
        print(f"\n  Running marginal coverage for clinic {test_clinic}...")
        results = run_marginal_experiment_one_clinic(
            df=df,
            X=X,
            training_clinics=training_clinics,
            test_clinic=test_clinic,
            alpha=args.alpha,
            alpha_selection=0.5,
            n_subsample_rep=50,
            mu_method_baseline=mu_baseline,
            mu_method_hcp=mu_hcp
        )
        all_results.append(results)

    # Combine results
    print("\n6. Combining and saving results...")
    full_results = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = output_dir / 'bp_marginal_detailed.csv'
    full_results.to_csv(detailed_path, index=False)
    print(f"   Detailed results saved to {detailed_path}")
    print(f"   Total predictions: {len(full_results)} rows")

    # Compute summary statistics
    print("\n7. Computing summary statistics...")

    # Coverage by (percentile, method)
    by_percentile = full_results.groupby(['percentile', 'method'])['covered'].agg(['sum', 'count', 'mean']).round(4)
    by_percentile.columns = ['n_covered', 'n_total', 'coverage']

    # Pivot to get coverage by percentile for each method
    coverage_by_pct = by_percentile['coverage'].unstack(level=0)

    # Overall coverage by method (average across all percentiles)
    overall = full_results.groupby('method').agg({
        'covered': 'mean',
        'width': ['mean', 'median'],
        'infinite': 'mean'
    }).round(4)
    overall.columns = ['overall_coverage', 'mean_width', 'median_width', 'prop_infinite']

    # Combine
    target_coverage = 1.0 - args.alpha
    summary = pd.concat([coverage_by_pct, overall], axis=1)
    summary['target_coverage'] = target_coverage
    summary['coverage_diff'] = summary['overall_coverage'] - target_coverage

    # Save summary
    summary_path = output_dir / 'bp_marginal_summary.csv'
    summary.to_csv(summary_path)
    print(f"   Summary saved to {summary_path}")

    # Generate markdown table
    md_path = output_dir / 'bp_marginal_summary.md'
    csv_to_markdown_table(
        str(summary_path),
        str(md_path),
        "Blood Pressure Marginal Coverage Results",
        target_coverage
    )

    # Add data summary table (no filtered data file for BP, use main df)
    temp_data_path = output_dir / 'bp_data_for_summary.csv'
    df.to_csv(temp_data_path, index=False)
    add_data_summary_to_markdown(
        str(md_path),
        str(temp_data_path),
        group_col='clinic_id',
        outcome_col='y',
        original_outcome_col='followup_sbp',
        is_test_group_func=lambda clinic: clinic in test_clinics
    )
    temp_data_path.unlink()  # Clean up temp file

    # Display results
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(summary)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nFiles saved to {output_dir}/:")
    print(f"  - bp_marginal_detailed.csv ({len(full_results)} predictions)")
    print(f"  - bp_marginal_summary.csv (method summaries)")


if __name__ == "__main__":
    main()
