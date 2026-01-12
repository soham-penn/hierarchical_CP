"""
ACS Sequential Online Conformal Prediction Experiments

This script implements sequential/online conformal prediction where:
1. Test data is ordered by year of entry (YOEP)
2. We start with 0 test observations, predict for 1st observation
3. Add 1st observation, predict for 2nd observation
4. Continue sequentially through all test observations
5. Compute average coverage across all predictions

Experiment Design:
- Training: 19 emerging states (fixed)
- Test: 5 emerging states (one at a time)
- For each test state: N predictions (one per observation)
- Output: Detailed results + summary by method
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


def run_all_methods_one_prediction(
    U_cal, Z_cal, U_test, Z_test,
    o_observed,
    alpha=0.1,
    alpha_selection=0.5,
    n_subsample_rep=50,
    mu_method_baseline=None,
    mu_method_hcp=None
):
    """
    Run all 6 methods for ONE prediction (the (o+1)-th observation).

    Parameters:
    -----------
    U_cal : ndarray
        Calibration group-level covariates
    Z_cal : list
        Calibration observations
    U_test : ndarray
        Test group-level covariates
    Z_test : list
        Test observations (first o_observed points)
    o_observed : int
        Number of observed points in test group so far
    alpha : float
        Miscoverage level
    alpha_selection : float
        Selection parameter for HCP++
    n_subsample_rep : int
        Number of subsampling repetitions
    mu_method_baseline : dict
        Baseline μ-method
    mu_method_hcp : dict
        HCP μ-method

    Returns:
    --------
    dict : Results for all 6 methods for this one prediction
    """
    K = len(Z_cal)
    if K < 2:
        raise ValueError("Need at least 2 calibration groups")

    # Baseline: split groups into train/calib halves
    K0 = K // 2
    train_idx = list(range(K0))
    calib_idx = list(range(K0, K))

    # Fit baseline model
    model_baseline = mu_method_baseline['fit_global'](
        U_matrix=U_cal,
        Z_list=Z_cal,
        group_index_vector=train_idx
    )

    # Compute scores for baseline methods
    scores_list = []
    for j in calib_idx:
        Zj = Z_cal[j]
        Uj = U_cal[j, :]
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

    # Compute interval radii for baseline methods
    T_hcp = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, n_subsample_rep
    )

    # Target observation index (o+1 = the NEXT observation we want to predict)
    test_index = o_observed  # 0-indexed

    # Get true value and features for target
    true_y = Z_test[test_index]['Y']
    x_target = Z_test[test_index]['X']

    # Baseline prediction
    mu_hat = mu_method_baseline['predict_global'](
        model_global=model_baseline,
        x_vector=x_target,
        u_vector=U_test[0, :]
    )

    # Helper function to create interval from radius
    def interval_from_radius(center, radius):
        if np.isinf(radius):
            return (-np.inf, np.inf)
        return (center - radius, center + radius)

    # HCP++ interval
    try:
        res_pp = compute_hcp_plus_interval(
            U_calibration=U_cal,
            Z_calibration=Z_cal,
            U_test=U_test,
            Z_test=Z_test[:o_observed+1],  # Include target observation
            o_observed=o_observed,
            alpha=alpha,
            alpha_selection=alpha_selection,
            mu_method=mu_method_hcp
        )
        int_pp = res_pp['interval']
    except Exception as e:
        print(f"    Warning: HCP++ failed: {e}")
        int_pp = (-np.inf, np.inf)

    # HCP.sample interval
    try:
        res_hs = compute_hcp_sample_interval(
            U_calibration=U_cal,
            Z_calibration=Z_cal,
            U_test=U_test,
            Z_test=Z_test[:o_observed+1],  # Include target for HCP.sample
            o_observed=o_observed,
            alpha=alpha,
            test_index_target=test_index,
            alpha_selection=alpha_selection,
            mu_method=mu_method_hcp
        )
        int_hs = res_hs['interval']
    except Exception as e:
        print(f"    Warning: HCP.sample failed: {e}")
        int_hs = (-np.inf, np.inf)

    # Baseline intervals
    int_hcp = interval_from_radius(mu_hat, T_hcp)
    int_pool = interval_from_radius(mu_hat, T_pool)
    int_sub = interval_from_radius(mu_hat, T_sub)
    int_rep = interval_from_radius(mu_hat, T_rep)

    # Helper functions
    def check_coverage(interval, true_value):
        return interval[0] <= true_value <= interval[1]

    def compute_width(interval):
        if np.isfinite(interval[0]) and np.isfinite(interval[1]):
            return interval[1] - interval[0]
        return np.nan

    # Compile results
    results = {}
    for method, interval in [
        ('HCP++', int_pp),
        ('HCP.sample', int_hs),
        ('HCP', int_hcp),
        ('Pooling', int_pool),
        ('Subsampling', int_sub),
        ('Repeated', int_rep)
    ]:
        results[method] = {
            'covered': check_coverage(interval, true_y),
            'width': compute_width(interval),
            'infinite': not (np.isfinite(interval[0]) and np.isfinite(interval[1])),
            'lower': interval[0],
            'upper': interval[1],
            'true_y': true_y,
            'mu_hat': mu_hat
        }

    return results


def run_sequential_experiment_one_state(
    df, X, training_states, test_state,
    alpha=0.1,
    alpha_selection=0.5,
    n_subsample_rep=50,
    mu_method_baseline=None,
    mu_method_hcp=None
):
    """
    Run sequential online prediction for ONE test state.

    For each observation in the test state (ordered by year of entry):
    - Use all previous observations as context
    - Predict for the next observation
    - Record coverage indicator

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
    pd.DataFrame : Results with one row per (test_obs_index, method)
    """
    # Get test state data, ordered by year of entry
    test_df = df[df['state_abb'] == test_state].sort_values('yoep').reset_index(drop=True)
    n_test = len(test_df)

    if n_test == 0:
        print(f"  Warning: No data for test state {test_state}")
        return pd.DataFrame()

    print(f"  Test state {test_state}: {n_test} observations (YOEP range: {test_df['yoep'].min()}-{test_df['yoep'].max()})")

    # Create calibration groups (one per training state, ALL observations)
    Z_calibration = []
    cal_states_used = []

    for state in training_states:
        state_df = df[df['state_abb'] == state]
        if len(state_df) < 1:  # Only skip if truly no data
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

    n_cal_groups = len(Z_calibration)
    n_cal_total_obs = sum(len(Z) for Z in Z_calibration)
    print(f"  Calibration: {n_cal_groups} groups, {n_cal_total_obs} total observations")

    # U vectors (constant 0 to avoid leakage)
    U_calibration = np.zeros((n_cal_groups, 1))
    U_test = np.zeros((1, 1))

    # Sequential prediction loop
    all_results = []
    test_indices = test_df.index.tolist()

    for o_observed in range(n_test):
        if (o_observed + 1) % 10 == 0 or o_observed == n_test - 1:
            print(f"    Prediction {o_observed + 1}/{n_test}")

        # Build test group: all observations up to (but not including) current one
        Z_test = []
        for i in range(o_observed + 1):  # Need to include target for data structure
            idx = test_indices[i]
            Z_test.append({
                'X': X[idx, :],
                'Y': df.iloc[idx]['y']
            })

        # Run all methods
        try:
            results = run_all_methods_one_prediction(
                U_cal=U_calibration,
                Z_cal=Z_calibration,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o_observed,
                alpha=alpha,
                alpha_selection=alpha_selection,
                n_subsample_rep=n_subsample_rep,
                mu_method_baseline=mu_method_baseline,
                mu_method_hcp=mu_method_hcp
            )

            # Record results for all methods
            year_entry = test_df.iloc[o_observed]['yoep']
            for method, res in results.items():
                all_results.append({
                    'test_state': test_state,
                    'test_obs_index': o_observed,
                    'year_of_entry': year_entry,
                    'n_observed_so_far': o_observed,
                    'method': method,
                    'covered': int(res['covered']),
                    'width': res['width'],
                    'infinite': int(res['infinite']),
                    'lower': res['lower'],
                    'upper': res['upper'],
                    'true_y': res['true_y'],
                    'mu_hat': res['mu_hat'],
                    'n_cal_groups': n_cal_groups,
                    'n_cal_total_obs': n_cal_total_obs,
                    'n_test_total': n_test
                })
        except Exception as e:
            print(f"    Error in prediction {o_observed + 1}: {e}")
            continue

    return pd.DataFrame(all_results)


def main():
    """Main function - run sequential experiments."""
    import argparse

    parser = argparse.ArgumentParser(description='Run ACS sequential online CP experiments')
    parser.add_argument('pums_csv', type=str, help='Path to ACS PUMS CSV file')
    parser.add_argument('--output_dir', type=str, default='results_final',
                       help='Output directory for results')
    parser.add_argument('--top_income_pct', type=float, default=10.0,
                       help='Top X percent by income to keep (default: 10)')
    parser.add_argument('--n_training_states', type=int, default=19,
                       help='Number of training states (default: 19)')
    parser.add_argument('--test_states', type=str, nargs='+', default=None,
                       help='Test states (default: auto-select 5)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Miscoverage level (default: 0.1 for 90%% coverage)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 80)
    print("ACS SEQUENTIAL ONLINE CONFORMAL PREDICTION EXPERIMENTS")
    print("=" * 80)

    # Load and filter data
    print("\n1. Loading and filtering ACS data...")
    print(f"   Top {args.top_income_pct}% by income")

    # Handle income filter: 0 means no filter, otherwise use top X%
    if args.top_income_pct > 0:
        top_income_quantile = 1.0 - (args.top_income_pct / 100.0)
    else:
        top_income_quantile = None  # No income filter

    df = load_and_clean_acs_pums(
        args.pums_csv,
        states_keep=EMERGING_STATES,
        top_income_quantile=top_income_quantile
    )

    # Save filtered data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_data_path = output_dir.parent / 'data' / f'acs_filtered_top{int(args.top_income_pct)}pct.csv'
    filtered_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filtered_data_path, index=False)
    print(f"\n   Saved filtered data to {filtered_data_path}")

    # Reset index so positions match between df and X
    df = df.reset_index(drop=True)

    # Build design matrix
    print("\n2. Building design matrix...")
    X = build_design_matrix_acs(df)
    print(f"   Design matrix shape: {X.shape}")

    # Select training and test states
    print("\n3. Selecting training and test states...")
    state_counts = df.groupby('state_abb').size().sort_values(ascending=False)
    print(f"   State counts:\n{state_counts}")

    # Determine test states first (either specified or auto-select)
    if args.test_states is not None:
        test_states = args.test_states
        print(f"\n   Test states (specified, {len(test_states)}): {test_states}")
    else:
        # Auto-select: use states ranked N+1 to N+5
        test_states = state_counts.index[args.n_training_states:args.n_training_states+5].tolist()
        print(f"\n   Test states (auto-selected, {len(test_states)}): {test_states}")

    # Training states: Top N states EXCLUDING test states
    available_for_training = [s for s in state_counts.index if s not in test_states]
    training_states = available_for_training[:args.n_training_states]

    print(f"   Training states ({len(training_states)}): {training_states}")

    # Verify no overlap
    overlap = set(training_states) & set(test_states)
    if overlap:
        raise ValueError(f"Training and test states overlap: {overlap}")

    # Create μ-methods (using OLS)
    print("\n4. Creating μ-estimation methods (OLS)...")
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    # Run sequential experiments
    print(f"\n5. Running sequential experiments ({len(test_states)} test states)...")
    all_results = []

    for test_state in test_states:
        print(f"\n  Running sequential prediction for {test_state}...")
        results = run_sequential_experiment_one_state(
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
    detailed_path = output_dir / 'acs_sequential_detailed.csv'
    full_results.to_csv(detailed_path, index=False)
    print(f"   Detailed results saved to {detailed_path}")
    print(f"   Total predictions: {len(full_results)} rows")

    # Compute summary statistics
    print("\n7. Computing summary statistics...")

    # Coverage by (test_state, method)
    by_state = full_results.groupby(['test_state', 'method']).agg({
        'covered': 'mean',
        'width': 'median',
        'infinite': 'sum'
    }).round(4)
    by_state.columns = ['coverage', 'median_width', 'n_infinite']

    # Pivot to get coverage by state for each method
    coverage_by_state = by_state['coverage'].unstack(level=0)

    # Overall coverage by method
    overall = full_results.groupby('method').agg({
        'covered': 'mean',
        'width': 'median',
        'infinite': 'sum'
    }).round(4)
    overall.columns = ['overall_coverage', 'overall_median_width', 'total_infinite']

    # Combine
    target_coverage = 1.0 - args.alpha
    summary = pd.concat([coverage_by_state, overall], axis=1)
    summary['target_coverage'] = target_coverage
    summary['coverage_diff'] = summary['overall_coverage'] - target_coverage

    # Save summary
    summary_path = output_dir / 'acs_sequential_summary.csv'
    summary.to_csv(summary_path)
    print(f"   Summary saved to {summary_path}")

    # Display results
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(summary)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nFiles saved to {output_dir}/:")
    print(f"  - acs_sequential_detailed.csv ({len(full_results)} predictions)")
    print(f"  - acs_sequential_summary.csv (method summaries)")
    print(f"\nFiltered data saved to:")
    print(f"  - {filtered_data_path} ({len(df)} observations)")


if __name__ == "__main__":
    main()
