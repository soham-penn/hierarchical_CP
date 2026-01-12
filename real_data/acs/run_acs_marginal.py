"""
ACS Marginal Coverage Experiments

This script implements marginal coverage evaluation where:
1. For each test state, use ALL observations except the last one as calibration
2. Order observations by income (to get percentiles)
3. Test coverage at 0th, 25th, 50th, 75th percentile points
4. Predict the NEXT observation after each percentile
5. Aggregate coverage across states by percentile

Experiment Design:
- All 50 states in ACS data
- Filter by top 25% income
- Training: 34 states (fixed)
- Test: 15-16 emerging destination states
- Alpha: 0.1 (90% coverage target)
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
    load_and_clean_acs_pums,
    build_design_matrix_acs,
    EMERGING_STATES
)

sys.path.append(str(Path(__file__).parent.parent))
from format_results import csv_to_markdown_table, add_data_summary_to_markdown


def run_marginal_experiment_one_state(
    df, X, training_states, test_state,
    alpha=0.1,
    alpha_selection=0.5,
    n_subsample_rep=50,
    mu_method_baseline=None,
    mu_method_hcp=None
):
    """
    Run marginal coverage experiment for ONE test state.

    For the test state:
    1. Order observations by income (for percentile calculation)
    2. Use all but last observation for calibration
    3. Test coverage at 0th, 25th, 50th, 75th percentile + 1
    4. Record coverage indicator for each percentile

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ACS data
    X : np.ndarray
        Design matrix
    training_states : list
        States to use for calibration (fixed)
    test_state : str
        State to test on
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
    # Get test state data, ordered by income
    test_df = df[df['state_abb'] == test_state].sort_values('income').reset_index(drop=True)
    n_test = len(test_df)

    if n_test < 5:
        print(f"  Warning: Test state {test_state} has only {n_test} observations, skipping")
        return pd.DataFrame()

    print(f"  Test state {test_state}: {n_test} observations (income range: ${test_df['income'].min():,.0f}-${test_df['income'].max():,.0f})")

    # Create calibration groups from training states ONLY
    # Baseline methods should NOT see any test state data
    Z_calibration = []
    cal_states_used = []

    for state in training_states:
        state_df = df[df['state_abb'] == state]
        if len(state_df) < 1:
            continue

        Z_group = []
        state_indices = state_df.index.tolist()
        for idx in state_indices:
            Z_group.append({
                'X': X[idx, :],
                'Y': df.iloc[idx]['y']
            })

        Z_calibration.append(Z_group)
        cal_states_used.append(state)

    test_indices = test_df.index.tolist()

    n_cal_groups = len(Z_calibration)
    n_cal_total_obs = sum(len(Z) for Z in Z_calibration)
    print(f"    Calibration: {n_cal_groups} groups, {n_cal_total_obs} total observations")

    # U vectors (constant 0)
    U_calibration = np.zeros((n_cal_groups, 1))
    U_test = np.zeros((1, 1))

    # Determine percentile indices (0th, 25th, 50th, 75th)
    percentiles = [0, 25, 50, 75]
    percentile_indices = {}

    for pct in percentiles:
        if pct == 0:
            idx = 0
        else:
            idx = int(np.percentile(np.arange(n_test - 1), pct))  # Percentile of n_test-1 observations

        # Check if next observation exists
        if idx + 1 >= n_test:
            continue

        percentile_indices[pct] = idx

    print(f"    Testing percentiles: {list(percentile_indices.keys())}")

    # Run experiments for each percentile
    all_results = []

    for pct, target_index in percentile_indices.items():
        # target_index = index of the observation we want to predict
        # History = observations 0 through target_index-1 (target_index observations)
        # o_observed = target_index (number of history observations)
        # Z_test = history + target = observations 0 through target_index (inclusive)

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

        # Baseline methods: split groups, compute fixed radius
        K = len(Z_calibration)
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
                'test_state': test_state,
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
                'n_cal_groups': n_cal_groups,
                'n_cal_total_obs': n_cal_total_obs,
                'n_test_total': n_test
            })

    return pd.DataFrame(all_results)


def main():
    """Main function - run marginal experiments."""
    import argparse

    parser = argparse.ArgumentParser(description='Run ACS marginal coverage experiments')
    parser.add_argument('pums_csv', type=str, help='Path to ACS PUMS CSV file')
    parser.add_argument('--output_dir', type=str, default='results/marginal',
                       help='Output directory for results')
    parser.add_argument('--top_income_pct', type=float, default=25.0,
                       help='Top X percent by income to keep (default: 25)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Miscoverage level (default: 0.1 for 90%% coverage)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 80)
    print("ACS MARGINAL COVERAGE EXPERIMENTS")
    print("=" * 80)

    # Load and filter data
    print(f"\n1. Loading and filtering ACS data...")
    print(f"   Top {args.top_income_pct}% by income")

    if args.top_income_pct > 0:
        top_income_quantile = 1.0 - (args.top_income_pct / 100.0)
    else:
        top_income_quantile = None

    # Load ALL states (not just emerging)
    df = load_and_clean_acs_pums(
        args.pums_csv,
        states_keep=None,  # Keep all states
        top_income_quantile=top_income_quantile
    )

    # Save filtered data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_data_path = output_dir.parent.parent / 'data' / f'acs_filtered_marginal_top{int(args.top_income_pct)}pct.csv'
    filtered_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filtered_data_path, index=False)
    print(f"\n   Saved filtered data to {filtered_data_path}")

    # Reset index
    df = df.reset_index(drop=True)

    # Build design matrix
    print("\n2. Building design matrix...")
    X = build_design_matrix_acs(df)
    print(f"   Design matrix shape: {X.shape}")

    # Select training and test states
    print("\n3. Selecting training and test states...")
    state_counts = df.groupby('state_abb').size().sort_values(ascending=False)
    print(f"   Total states: {len(state_counts)}")
    print(f"   Top 20 states by count:\n{state_counts.head(20)}")

    # Test states: emerging destination states (that have data)
    test_states = [s for s in EMERGING_STATES if s in state_counts.index]
    print(f"\n   Test states (emerging destinations, {len(test_states)}): {test_states}")

    # Training states: all others
    training_states = [s for s in state_counts.index if s not in test_states]
    print(f"   Training states ({len(training_states)}): {training_states[:10]}... (showing first 10)")

    # Create μ-methods (using OLS)
    print("\n4. Creating μ-estimation methods (OLS)...")
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    # Run marginal experiments
    print(f"\n5. Running marginal experiments ({len(test_states)} test states)...")
    all_results = []

    for test_state in test_states:
        print(f"\n  Running marginal coverage for {test_state}...")
        results = run_marginal_experiment_one_state(
            df=df,
            X=X,
            training_states=training_states,
            test_state=test_state,
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
    detailed_path = output_dir / 'acs_marginal_detailed.csv'
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
    summary_path = output_dir / 'acs_marginal_summary.csv'
    summary.to_csv(summary_path)
    print(f"   Summary saved to {summary_path}")

    # Generate markdown table
    md_path = output_dir / 'acs_marginal_summary.md'
    csv_to_markdown_table(
        str(summary_path),
        str(md_path),
        "ACS Marginal Coverage Results",
        target_coverage
    )

    # Add data summary table
    add_data_summary_to_markdown(
        str(md_path),
        str(filtered_data_path),
        group_col='state_abb',
        outcome_col='y',
        original_outcome_col='income',
        is_test_group_func=lambda state: state in test_states
    )

    # Display results
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(summary)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nFiles saved to {output_dir}/:")
    print(f"  - acs_marginal_detailed.csv ({len(full_results)} predictions)")
    print(f"  - acs_marginal_summary.csv (method summaries)")
    print(f"\nFiltered data saved to:")
    print(f"  - {filtered_data_path} ({len(df)} observations)")


if __name__ == "__main__":
    main()
