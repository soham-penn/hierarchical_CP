#!/usr/bin/env python3
"""
Add naive pooled-income quantile baseline to ACS true marginal experiments.

This script:
1. Loads existing ACS true marginal experiment results
2. Re-runs each trial to extract the pooled calibration incomes
3. Computes unconditional income quantile intervals
4. Saves results with method name 'Pooled-Income-Quantile'

Output:
- Raw results: acs/results/true_marginal_alphaXX/acs_true_marg_pooled_quantile_alphaXX_detailed.csv
- Summary: acs/results/true_marginal_alphaXX/acs_true_marg_pooled_quantile_alphaXX_summary.csv
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs


# ==================== Helper functions ====================

def stratified_sample_groups(eligible_groups, strata, selection_seed, n_groups_to_select):
    """Stratified sampling of groups (matching main experiment)."""
    rng = np.random.default_rng(selection_seed)
    n_strata = len(strata)
    n_per_stratum = n_groups_to_select // n_strata
    remainder = n_groups_to_select % n_strata

    selected = []
    for i, (_, grp_list) in enumerate(sorted(strata.items())):
        n_from_this = n_per_stratum + (1 if i < remainder else 0)
        n_from_this = min(n_from_this, len(grp_list))
        selected.extend(rng.choice(grp_list, size=n_from_this, replace=False).tolist())

    return selected


def create_ba_plus_strata(eligible_groups, group_share_baplus, n_strata):
    """Create strata based on BA+ share (matching main experiment)."""
    shares = []
    groups = []
    for g in eligible_groups:
        if g in group_share_baplus:
            shares.append(group_share_baplus[g])
            groups.append(g)

    shares = np.array(shares)
    groups = np.array(groups)

    quantiles = np.linspace(0, 1, n_strata + 1)
    bins = np.quantile(shares, quantiles)
    bins[0] -= 0.001
    bins[-1] += 0.001

    labels = np.digitize(shares, bins) - 1

    strata = {}
    for i in range(n_strata):
        strata[i] = groups[labels == i].tolist()

    return strata


def run_one_trial_pooled_quantile(df, X, eligible_groups, strata, group_col, config, trial_idx):
    """
    Run one trial of the pooled-income quantile baseline.

    This replicates the same sampling procedure as the main experiment,
    then computes pooled income quantiles from calibration PUMAs only.
    """
    rng = np.random.default_rng(config['seed'] + trial_idx)
    alpha = config['alpha']
    n_calib_groups = config.get('n_puma_groups', 20)

    # Select calibration groups via stratified sampling (same as main experiment)
    selection_seed = config['seed'] + trial_idx * 1009
    calib_groups = stratified_sample_groups(
        eligible_groups=eligible_groups,
        strata=strata,
        selection_seed=selection_seed,
        n_groups_to_select=n_calib_groups,
    )

    # Select one test group uniformly from remaining groups with at least 21 observations
    calib_set = set(calib_groups)
    min_test_size = 21
    test_candidates = []
    for g in eligible_groups:
        if g not in calib_set:
            grp_mask = (df[group_col] == g).values
            grp_size = grp_mask.sum()
            if grp_size >= min_test_size:
                test_candidates.append(g)

    if len(test_candidates) == 0:
        return None

    test_group = rng.choice(test_candidates)

    # Get original observations
    group_data = {}
    for grp in calib_groups + [test_group]:
        grp_mask = (df[group_col] == grp).values
        grp_idx = np.where(grp_mask)[0]
        group_data[grp] = {
            'X': X[grp_idx],
            'Y': df.iloc[grp_idx]['y'].values,  # log1p(income)
        }

    # Check test group has enough observations
    if len(group_data[test_group]['Y']) < min_test_size:
        return None

    # Fixed target: always the 20th observation (index 20, the 21st point)
    target_index = 20
    y_target = group_data[test_group]['Y'][target_index]

    # Pool ALL incomes from calibration groups ONLY (do NOT include test group)
    pooled_incomes = []
    for grp in calib_groups:
        pooled_incomes.extend(group_data[grp]['Y'].tolist())

    pooled_incomes = np.array(pooled_incomes)

    # Compute quantile-based interval
    q_lo = np.quantile(pooled_incomes, alpha / 2)
    q_hi = np.quantile(pooled_incomes, 1 - alpha / 2)

    # Check coverage
    covered = int(q_lo <= y_target <= q_hi)

    # Interval width in log1p space
    width_log = q_hi - q_lo

    # Interval width in actual income space
    # log1p(income) = y, so income = expm1(y)
    # width_income = expm1(q_hi) - expm1(q_lo)
    income_lo = np.expm1(q_lo)
    income_hi = np.expm1(q_hi)
    width_income = income_hi - income_lo

    return {
        'trial_id': trial_idx,
        'coverage': covered,
        'width': width_log,
        'width_income': width_income,
        'lower': q_lo,
        'upper': q_hi,
        'y_target': y_target,
        'income_target': np.expm1(y_target),
        'n_pooled': len(pooled_incomes),
    }


# Worker initialization for multiprocessing
_WORKER_SHARED = None

def _init_worker(df, X, eligible_groups, strata, group_col, config):
    global _WORKER_SHARED
    _WORKER_SHARED = (df, X, eligible_groups, strata, group_col, config)


def _run_one_trial_worker(trial_idx):
    df, X, eligible_groups, strata, group_col, config = _WORKER_SHARED
    result = run_one_trial_pooled_quantile(df, X, eligible_groups, strata, group_col, config, trial_idx)
    return trial_idx, result


def run_pooled_quantile_experiments(df, X, eligible_groups, strata, group_col, config):
    """Run pooled-income quantile baseline experiments."""
    B = config['B']
    n_workers = int(config.get('n_workers', 1))

    results = []

    print(f"  Running {B} pooled-quantile trials with {n_workers} workers")

    if n_workers <= 1:
        _init_worker(df, X, eligible_groups, strata, group_col, config)
        for b in range(B):
            if (b + 1) % 100 == 0 or (b + 1) <= 10:
                print(f"    Trial {b+1}/{B}")
            trial_idx, result = _run_one_trial_worker(b)
            if result is not None:
                results.append(result)
    else:
        completed = 0
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(df, X, eligible_groups, strata, group_col, config),
        ) as pool:
            for trial_idx, result in pool.imap_unordered(_run_one_trial_worker, range(B)):
                if result is not None:
                    results.append(result)
                completed += 1
                if completed % 100 == 0 or completed <= 10 or completed == B:
                    print(f"    Trial {completed}/{B}")

    return results


def save_results(results, output_dir, alpha):
    """Save raw results and summary statistics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Add metadata columns
    df['method'] = 'Pooled-Income-Quantile'
    df['alpha'] = alpha
    df['nominal_coverage'] = 1 - alpha

    # Since this method is o-independent, we'll duplicate across o values for plotting compatibility
    # But we'll also save an o-independent version
    o_values = [0, 5, 10, 15, 20]  # Match main experiment

    # Save o-independent version (with o = -1 to indicate o-independent)
    df_independent = df.copy()
    df_independent['o'] = -1  # Special marker for o-independent

    alpha_str = f"alpha{int(alpha*100):02d}"
    output_csv_independent = output_dir / f'acs_true_marg_pooled_quantile_{alpha_str}_o_independent_detailed.csv'
    df_independent.to_csv(output_csv_independent, index=False)
    print(f"\n  Saved o-independent results: {output_csv_independent}")

    # Save duplicated version (same interval for all o values, for plotting)
    df_duplicated_list = []
    for o in o_values:
        df_o = df.copy()
        df_o['o'] = o
        df_duplicated_list.append(df_o)

    df_duplicated = pd.concat(df_duplicated_list, ignore_index=True)
    output_csv_duplicated = output_dir / f'acs_true_marg_pooled_quantile_{alpha_str}_detailed.csv'
    df_duplicated.to_csv(output_csv_duplicated, index=False)
    print(f"  Saved duplicated results (for plotting): {output_csv_duplicated}")

    # Compute summary statistics
    n_trials = len(results)
    coverage_mean = df['coverage'].mean()
    coverage_se = np.sqrt(coverage_mean * (1 - coverage_mean) / n_trials)

    summary = {
        'method': 'Pooled-Income-Quantile',
        'alpha': alpha,
        'nominal_coverage': 1 - alpha,
        'n_trials': n_trials,
        'coverage_mean': coverage_mean,
        'coverage_se': coverage_se,
        'width_income_mean': df['width_income'].mean(),
        'width_income_std': df['width_income'].std(),
        'width_income_median': df['width_income'].median(),
        'width_log_mean': df['width'].mean(),
        'width_log_std': df['width'].std(),
        'width_log_median': df['width'].median(),
        'n_pooled_mean': df['n_pooled'].mean(),
        'n_pooled_std': df['n_pooled'].std(),
    }

    df_summary = pd.DataFrame([summary])
    output_csv_summary = output_dir / f'acs_true_marg_pooled_quantile_{alpha_str}_summary.csv'
    df_summary.to_csv(output_csv_summary, index=False)
    print(f"  Saved summary: {output_csv_summary}")

    # Print summary
    print(f"\n  Summary statistics:")
    print(f"    Trials: {n_trials}")
    print(f"    Nominal coverage: {1-alpha:.1%}")
    print(f"    Empirical coverage: {coverage_mean:.4f} ± {coverage_se:.4f}")
    print(f"    Width (income): median={df['width_income'].median():.0f}, mean={df['width_income'].mean():.0f}")
    print(f"    Width (log): median={df['width'].median():.3f}, mean={df['width'].mean():.3f}")
    print(f"    Pooled sample size: mean={df['n_pooled'].mean():.1f}")

    return df_duplicated, df_summary


def main():
    parser = argparse.ArgumentParser(description='Add pooled-income quantile baseline to ACS true marginal experiments')
    parser.add_argument('--B', type=int, default=1000, help='Number of trials')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level (default: 0.1 for 90%% coverage)')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--n_puma_groups', type=int, default=20, help='Number of calibration PUMAs per trial')

    args = parser.parse_args()

    print("=" * 70)
    print("ACS True Marginal: Pooled-Income Quantile Baseline")
    print("=" * 70)

    # Load data (matching main experiment setup)
    base_dir = Path(__file__).parent

    print(f"\nLoading ACS data...")
    df = load_and_clean_acs_pums(
        str(base_dir / 'acs/data/acs_data_all50states.csv'),
        states_keep=['CA'],  # Match main experiment
        age_min=25,
        age_max=54,
        yoep_min_year=2012,
        min_hours=40,
        min_income=10000.0,
    )

    df = df.dropna(subset=['puma']).copy()
    df['puma'] = df['puma'].astype(int)

    print(f"  Loaded {len(df)} observations")

    # Build design matrix
    X = build_design_matrix_acs(df)
    print(f"  Design matrix shape: {X.shape}")

    # Target column
    group_col = 'puma'
    df['y'] = np.log1p(df['PINCP'])  # log1p(income)

    # Filter eligible groups (groups with at least 21 observations)
    min_group_size = 21
    group_counts = df[group_col].value_counts()
    eligible_groups = group_counts[group_counts >= min_group_size].index.tolist()

    print(f"\n  Eligible PUMAs (>= {min_group_size} obs): {len(eligible_groups)}")
    print(f"  Min group size: {group_counts[eligible_groups].min()}")
    print(f"  Max group size: {group_counts[eligible_groups].max()}")
    print(f"  Median group size: {group_counts[eligible_groups].median():.0f}")

    # Create BA+ share stratification
    print(f"\n  Computing BA+ share stratification...")
    group_share_baplus = {}
    for g in eligible_groups:
        grp_mask = (df[group_col] == g).values
        grp_df = df.loc[grp_mask]
        share = (grp_df['SCHL'] >= 21).mean()  # SCHL >= 21 means BA or higher
        group_share_baplus[g] = share

    strata = create_ba_plus_strata(
        eligible_groups=eligible_groups,
        group_share_baplus=group_share_baplus,
        n_strata=5,
    )

    print(f"  Number of strata: {len(strata)}")
    for k in sorted(strata.keys()):
        print(f"    Stratum {k}: {len(strata[k])} PUMAs")

    # Config
    config = {
        'B': args.B,
        'seed': 456,  # Match main experiment seed
        'alpha': args.alpha,
        'n_puma_groups': args.n_puma_groups,
        'n_workers': args.n_workers,
    }

    print(f"\nRunning pooled-income quantile baseline:")
    print(f"  Trials: {args.B}")
    print(f"  Alpha: {args.alpha} (nominal coverage: {1-args.alpha:.1%})")
    print(f"  Calibration PUMAs per trial: {args.n_puma_groups}")
    print(f"  Workers: {args.n_workers}")
    print()

    # Run experiments
    results = run_pooled_quantile_experiments(
        df=df,
        X=X,
        eligible_groups=eligible_groups,
        strata=strata,
        group_col=group_col,
        config=config,
    )

    # Save results
    alpha_str = f"alpha{int(args.alpha*100):02d}"
    output_dir = base_dir / f'acs/results/true_marginal_{alpha_str}'

    df_results, df_summary = save_results(results, output_dir, args.alpha)

    print(f"\nDone!")
    print(f"  Results saved to: {output_dir}")
    print(f"  Use these files for plotting alongside other ACS true marginal results.")


if __name__ == '__main__':
    main()
