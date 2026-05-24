"""
ACS TRUE MARGINAL coverage experiments.

Key difference from bootstrap experiments:
- Each replicate: randomly select 20 non-target PUMAs (stratified) + 1 target PUMA
- Use ORIGINAL observations (no bootstrap resampling)
- Test on observation at position o in the target PUMA
- Repeat 1000 times to evaluate true marginal coverage

This matches the DGP true marginal design.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs

from methods.mu_methods import create_mu_method_ols_global_only
from methods.donor_hcp import (
    compute_donor_hcp_randomized_interval,
)
from methods.sample_hcp import (
    compute_sample_hcp_randomized_interval,
)
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius,
)
from scores import absolute_residual_score

# Method lists
BASELINE_METHODS = ['HCP', 'Pooling', 'Subsampling', 'Repeated']
HCP_METHODS = ['Donor-HCP', 'S-HCP']
STD_CP_METHODS = ['Std-CP']
METHODS = BASELINE_METHODS + HCP_METHODS + STD_CP_METHODS

# ==================== Helper functions ====================

def _interval_from_radius(mu_hat, T):
    return (mu_hat - T, mu_hat + T)

def _covered(interval, true_y):
    return 1.0 if interval[0] <= true_y <= interval[1] else 0.0

def _width(interval):
    return interval[1] - interval[0]

def _width_income_from_log1p_interval(interval):
    """Convert log1p interval to income width."""
    lower, upper = interval
    income_lower = np.expm1(lower)
    income_upper = np.expm1(upper)
    return income_upper - income_lower


def compute_group_share_baplus(df, group_col):
    """Compute share of BA+ for each group for stratification."""
    out = {}
    for grp in df[group_col].unique():
        grp_df = df[df[group_col] == grp]
        out[grp] = float((grp_df['educ_level'] == 'BAplus').mean())
    return out


def _build_share_baplus_strata(eligible_groups, group_share_baplus, n_strata=5):
    """Build quantile strata from the PUMA-level share of BA+."""
    shares = pd.Series({g: group_share_baplus[g] for g in eligible_groups}, dtype=float)
    bins = pd.qcut(shares, q=int(n_strata), duplicates='drop')

    strata = {}
    for b in bins.cat.categories:
        members = shares.index[bins == b].tolist()
        if members:
            strata[str(b)] = members

    if len(strata) == 0:
        strata = {"all": list(eligible_groups)}

    return strata


def stratified_sample_groups(eligible_groups, strata, selection_seed, n_groups_to_select):
    """Use stratified sampling to select non-target groups."""
    stratum_names = sorted(list(strata.keys()))
    n_strata = len(stratum_names)
    if n_strata == 0:
        raise ValueError("No strata available for stratified sampling")

    if n_groups_to_select > len(eligible_groups):
        raise ValueError(
            f"Cannot select {n_groups_to_select} groups from {len(eligible_groups)} eligible groups"
        )

    rng = np.random.default_rng(selection_seed)

    # Balanced allocation across strata
    base = n_groups_to_select // n_strata
    rem = n_groups_to_select % n_strata

    capacity = {k: len(strata[k]) for k in stratum_names}
    alloc = {k: min(base, capacity[k]) for k in stratum_names}
    assigned = sum(alloc.values())

    # Distribute leftover picks
    while assigned < n_groups_to_select:
        order = sorted(
            stratum_names,
            key=lambda k: (capacity[k] - alloc[k], rng.random()),
            reverse=True,
        )
        progressed = False
        for k in order:
            if alloc[k] < capacity[k]:
                alloc[k] += 1
                assigned += 1
                progressed = True
                if assigned >= n_groups_to_select:
                    break
        if not progressed:
            break

    selected = []
    for k in stratum_names:
        take = int(alloc[k])
        if take <= 0:
            continue
        chosen = rng.choice(np.asarray(strata[k]), size=take, replace=False).tolist()
        selected.extend(chosen)

    if len(selected) != n_groups_to_select:
        raise ValueError(
            f"Balanced allocation produced {len(selected)} groups, expected {n_groups_to_select}"
        )

    return sorted(selected)


# ==================== Single replicate ====================

def run_one_replicate(df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx):
    """
    Run one true marginal replicate:
    1. Randomly select 20 non-target PUMAs (stratified) + 1 target PUMA
    2. Use original observations (no bootstrap)
    3. Test on position o in target PUMA
    """
    rng = np.random.default_rng(config['seed'] + replicate_idx)
    alpha = config['alpha']
    alpha_sel = config.get('alpha_selection', 0.5)
    n_rep = config.get('n_repeated', 50)
    n_calib_groups = config.get('n_puma_groups', 20)

    # Select calibration groups via stratified sampling
    selection_seed = config['seed'] + replicate_idx * 1009
    calib_groups = stratified_sample_groups(
        eligible_groups=eligible_groups,
        strata=strata,
        selection_seed=selection_seed,
        n_groups_to_select=n_calib_groups,
    )

    # Select one test group uniformly from remaining groups with at least 21 observations
    # (Since we need 21 observations to test at target_index=20)
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

    # Get original observations (NO bootstrap)
    group_data = {}
    for grp in calib_groups + [test_group]:
        grp_mask = (df[group_col] == grp).values
        grp_idx = np.where(grp_mask)[0]
        group_data[grp] = {
            'X': X[grp_idx],
            'Y': df.iloc[grp_idx]['y'].values,
        }

    # Build Z_calibration for all calibration groups
    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    # Baseline methods: HCP uses its own independent split (no donor removal)
    # Both HCP and D-HCP use same seed to get same initial S_tilde, but HCP doesn't remove donor
    from methods.donor_hcp import get_hcp_train_cal_split

    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, 'tolist') else list(calib_idx)

    # DIAGNOSTIC: Print split info for first 10 replicates
    if replicate_idx < 10:
        print(f"  Rep {replicate_idx}: K_calib={K_calib}, K_train={len(train_idx)}, K_cal={len(calib_idx)}, "
              f"alpha_sel={alpha_sel}, alpha={alpha}")

    from methods.mu_methods import create_mu_method_ols_offset

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()  # Use offset method for within-group training

    model_baseline = mu_baseline['fit_global'](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
    )

    scores_list = []
    for j in calib_idx:
        Zj = Z_calibration_full[j]
        yj = np.array([z['Y'] for z in Zj])
        Uj = U_calibration_full[j]
        muj = np.array([
            mu_baseline['predict_global'](model_baseline, z['X'], Uj)
            for z in Zj
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    # DIAGNOSTIC: Print HCP computation details for first 10 replicates
    if replicate_idx < 10:
        K_scores = len(scores_list)
        weight_inf = 1.0 / (K_scores + 1)
        print(f"  Rep {replicate_idx}: len(scores_list)={K_scores}, weight_inf={weight_inf:.6f}, "
              f"infinite?={weight_inf >= alpha}")

    T_hcp = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep = compute_repeated_subsampling_interval_radius(scores_list, alpha, n_rep)

    U_test = np.zeros((1, 1))

    baseline_result = {m: {} for m in BASELINE_METHODS}
    stdcp_result = {m: {} for m in STD_CP_METHODS}
    hcp_result = {m: {} for m in HCP_METHODS}

    # IMPORTANT: Baseline methods (HCP, Pooling, etc.) should only be computed ONCE
    # since they don't use test group observations - they only depend on calibration data
    # We'll compute them at o=0 and reuse for all o values

    # FIXED target index (like DGP: always predict the SAME observation)
    # We need at least 21 observations in test group
    min_obs_needed = 21
    if len(group_data[test_group]['Y']) < min_obs_needed:
        # Not enough observations - mark baselines as NaN too
        for method in BASELINE_METHODS:
            baseline_result[method][0] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
        # Not enough observations - mark all as NaN
        for o in o_values:
            for method in HCP_METHODS:
                hcp_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
        return {
            'baseline': baseline_result,
            'hcp': hcp_result,
            'stdcp': stdcp_result,
        }

    # FIXED target: always the 20th observation (index 20, the 21st point)
    target_index = 20
    x_target = group_data[test_group]['X'][target_index]
    true_y = group_data[test_group]['Y'][target_index]

    # Baseline methods: compute at the SAME target as D-HCP (target_index=20)
    # This ensures HCP at o=0 matches D-HCP at o=0
    mu_hat_baseline = mu_baseline['predict_global'](model_baseline, x_target, U_test[0])
    for method, T in [
        ('HCP', T_hcp),
        ('Pooling', T_pool),
        ('Subsampling', T_sub),
        ('Repeated', T_rep),
    ]:
        interval = _interval_from_radius(mu_hat_baseline, T)
        baseline_result[method][0] = {
            'coverage': _covered(interval, true_y),
            'width': _width(interval),
            'width_income': _width_income_from_log1p_interval(interval)
        }

    # Build FULL Z_test with ALL observations (like DGP does)
    # D-HCP will internally use only the first o_observed observations
    Z_test_full = [
        {'X': group_data[test_group]['X'][i], 'Y': group_data[test_group]['Y'][i]}
        for i in range(len(group_data[test_group]['Y']))
    ]

    # HCP methods and Std-CP: compute for each o value
    for o in o_values:
        hcp_covered = {}
        hcp_width = {}
        hcp_width_income = {}

        # Donor-HCP (randomized) - pass full Z_test, not truncated
        try:
            dhcp_seed = config['seed'] + replicate_idx * 1009 + o
            res_dhcp = compute_donor_hcp_randomized_interval(
                U_calibration=U_calibration_full,
                Z_calibration=Z_calibration_full,
                U_test=U_test, Z_test=Z_test_full,  # FULL Z_test
                o_observed=o, alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_hcp,
                random_seed=dhcp_seed,
                test_index_target=target_index,  # FIXED target_index
            )
            int_dhcp = res_dhcp['interval']
        except Exception:
            int_dhcp = (-np.inf, np.inf)

        hcp_covered['Donor-HCP'] = _covered(int_dhcp, true_y)
        hcp_width['Donor-HCP'] = _width(int_dhcp)
        hcp_width_income['Donor-HCP'] = _width_income_from_log1p_interval(int_dhcp)

        # Sample-HCP (randomized) - pass full Z_test, not truncated
        try:
            shcp_seed = config['seed'] + replicate_idx * 1009 + o + 1
            res_shcp = compute_sample_hcp_randomized_interval(
                U_calibration=U_calibration_full,
                Z_calibration=Z_calibration_full,
                U_test=U_test, Z_test=Z_test_full,  # FULL Z_test
                o_observed=o, alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_hcp,
                random_seed=shcp_seed,
                test_index_target=target_index,  # FIXED target_index
            )
            int_shcp = res_shcp['interval']
        except Exception:
            int_shcp = (-np.inf, np.inf)

        hcp_covered['S-HCP'] = _covered(int_shcp, true_y)
        hcp_width['S-HCP'] = _width(int_shcp)
        hcp_width_income['S-HCP'] = _width_income_from_log1p_interval(int_shcp)

        # Standard CP (on test group only)
        stdcp_covered = {}
        stdcp_width = {}
        stdcp_width_income = {}

        if o > 0:
            # Use first o observations to build conformal interval
            y_cal_test = np.array([group_data[test_group]['Y'][i] for i in range(o)])
            x_cal_test = np.array([group_data[test_group]['X'][i] for i in range(o)])

            # Simple mean prediction
            mu_stdcp = np.mean(y_cal_test)
            scores_stdcp = np.abs(y_cal_test - mu_stdcp)
            q_stdcp = np.quantile(scores_stdcp, 1 - alpha, method='higher')
            int_stdcp = (mu_stdcp - q_stdcp, mu_stdcp + q_stdcp)
        else:
            int_stdcp = (-np.inf, np.inf)

        stdcp_covered['Std-CP'] = _covered(int_stdcp, true_y)
        stdcp_width['Std-CP'] = _width(int_stdcp)
        stdcp_width_income['Std-CP'] = _width_income_from_log1p_interval(int_stdcp)

        # Store results (baselines already stored at o=0 above)
        for method in HCP_METHODS:
            hcp_result[method][o] = {
                'coverage': hcp_covered[method],
                'width': hcp_width[method],
                'width_income': hcp_width_income[method]
            }

        for method in STD_CP_METHODS:
            stdcp_result[method][o] = {
                'coverage': stdcp_covered[method],
                'width': stdcp_width[method],
                'width_income': stdcp_width_income[method]
            }

    return {
        'baseline': baseline_result,
        'hcp': hcp_result,
        'stdcp': stdcp_result,
    }


# ==================== Worker functions ====================

_WORKER_SHARED = None

def _init_worker(df, X, eligible_groups, strata, group_col, o_values, config):
    global _WORKER_SHARED
    _WORKER_SHARED = (df, X, eligible_groups, strata, group_col, o_values, config)


def _run_one_replicate_worker(replicate_idx):
    df, X, eligible_groups, strata, group_col, o_values, config = _WORKER_SHARED
    result = run_one_replicate(df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx)
    return replicate_idx, result


# ==================== Main experiment ====================

def run_true_marginal_experiments(df, X, eligible_groups, strata, group_col, o_values, config):
    """Run true marginal experiments."""
    B = config['B']
    n_o = len(o_values)
    n_workers = int(config.get('n_workers', 1))

    n_m = len(METHODS)
    m_map = {m: i for i, m in enumerate(METHODS)}

    cov = np.full((B, n_m, n_o), np.nan)
    wid = np.full((B, n_m, n_o), np.nan)
    wid_income = np.full((B, n_m, n_o), np.nan)

    print(f"  Running {B} true marginal replicates with {n_workers} workers")

    def _write_result(rep_idx, result):
        if result is None:
            return

        # Baseline methods: only stored at o=0
        for method in BASELINE_METHODS:
            m_i = m_map[method]
            o_i = 0  # baselines only at o=0
            if 0 in result['baseline'][method]:
                cov[rep_idx, m_i, o_i] = result['baseline'][method][0]['coverage']
                wid[rep_idx, m_i, o_i] = result['baseline'][method][0]['width']
                wid_income[rep_idx, m_i, o_i] = result['baseline'][method][0]['width_income']

        # HCP methods: stored for each o
        for method in HCP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[rep_idx, m_i, o_i] = result['hcp'][method][o]['coverage']
                wid[rep_idx, m_i, o_i] = result['hcp'][method][o]['width']
                wid_income[rep_idx, m_i, o_i] = result['hcp'][method][o]['width_income']

        # Std-CP: stored for each o
        for method in STD_CP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[rep_idx, m_i, o_i] = result['stdcp'][method][o]['coverage']
                wid[rep_idx, m_i, o_i] = result['stdcp'][method][o]['width']
                wid_income[rep_idx, m_i, o_i] = result['stdcp'][method][o]['width_income']

    if n_workers <= 1:
        _init_worker(df, X, eligible_groups, strata, group_col, o_values, config)
        for b in range(B):
            if (b + 1) % 100 == 0 or (b + 1) <= 10:
                print(f"    Replicate {b+1}/{B}")
            rep_idx, result = _run_one_replicate_worker(b)
            _write_result(rep_idx, result)
    else:
        completed = 0
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(df, X, eligible_groups, strata, group_col, o_values, config),
        ) as pool:
            for rep_idx, result in pool.imap_unordered(_run_one_replicate_worker, range(B)):
                completed += 1
                if completed % 100 == 0 or completed <= 10 or completed == B:
                    print(f"    Replicate {completed}/{B}")
                _write_result(rep_idx, result)

    return {
        'methods': METHODS,
        'o_values': o_values,
        'coverage': cov,
        'width': wid,
        'width_income': wid_income,
    }


# ==================== Save results ====================

def save_results_to_csv(results, output_path, config):
    """Save results to CSV in long format."""
    methods = results['methods']
    o_values = results['o_values']
    cov = results['coverage']
    wid = results['width']
    wid_income = results['width_income']

    B = cov.shape[0]

    rows = []
    for b in range(B):
        for m_i, method in enumerate(methods):
            for o_i, o in enumerate(o_values):
                rows.append({
                    'replicate': b,
                    'method': method,
                    'o': o,
                    'coverage': cov[b, m_i, o_i],
                    'width': wid[b, m_i, o_i],
                    'width_income': wid_income[b, m_i, o_i],
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"  Saved detailed results to {output_path}")
    return df_out


# ==================== Main entry point ====================

if __name__ == '__main__':
    import argparse
    mp.set_start_method('fork', force=True)

    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1000, help='Number of replicates')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of workers')
    parser.add_argument('--acs_state', type=str, default='CA')
    parser.add_argument('--n_puma_groups', type=int, default=20, help='Number of non-target PUMAs')
    parser.add_argument('--min_puma_size', type=int, default=20, help='Minimum PUMA size')
    parser.add_argument('--o_values', type=str, default='0,5,10,20', help='Comma-separated o values')

    args = parser.parse_args()

    print("=" * 70)
    print("ACS TRUE MARGINAL COVERAGE EXPERIMENTS")
    print("=" * 70)

    # Load data
    print("\nLoading ACS data...")
    df = load_and_clean_acs_pums(
        str(base_dir / 'acs/data/acs_data_all50states.csv'),
        states_keep=[args.acs_state],
        age_min=25,
        age_max=54,
        yoep_min_year=2012,
        min_hours=40,
        min_income=10000.0,
    )

    df = df.dropna(subset=['puma']).copy()
    df['puma'] = df['puma'].astype(int)
    X = build_design_matrix_acs(df)

    # Get eligible groups
    counts = df.groupby('puma').size()
    counts_eligible = counts[counts >= args.min_puma_size]
    eligible_groups = counts_eligible.index.to_numpy().tolist()

    print(f"  State: {args.acs_state}")
    print(f"  Total observations: {len(df)}")
    print(f"  Total PUMAs: {len(counts)}")
    print(f"  Eligible PUMAs (size >= {args.min_puma_size}): {len(eligible_groups)}")
    print(f"  Min group size: {counts_eligible.min()}")
    print(f"  Max group size: {counts_eligible.max()}")
    print(f"  Median group size: {counts_eligible.median():.0f}")

    # Build strata
    print("\n  Computing BA+ share stratification...")
    group_share_baplus = compute_group_share_baplus(df, 'puma')
    strata = _build_share_baplus_strata(
        eligible_groups=eligible_groups,
        group_share_baplus=group_share_baplus,
        n_strata=5,
    )

    print(f"  Number of strata: {len(strata)}")
    for k in sorted(strata.keys()):
        print(f"    Stratum {k}: {len(strata[k])} PUMAs")

    # Parse o values
    o_values = sorted(set(int(v) for v in args.o_values.split(',')))
    print(f"\n  o values: {o_values}")

    # Config
    config = {
        'B': args.B,
        'seed': 456,
        'alpha': 0.15,  # Nominal coverage 85%
        'alpha_selection': 0.5,
        'n_repeated': 50,
        'n_puma_groups': args.n_puma_groups,
        'n_workers': args.n_workers,
    }

    print(f"\nRunning true marginal experiments:")
    print(f"  Replicates: {args.B}")
    print(f"  Non-target PUMAs per replicate: {args.n_puma_groups}")
    print(f"  Workers: {args.n_workers}")
    print()

    # Run experiments
    results = run_true_marginal_experiments(
        df=df,
        X=X,
        eligible_groups=eligible_groups,
        strata=strata,
        group_col='puma',
        o_values=o_values,
        config=config,
    )

    # Save results (separate folder for each alpha value)
    alpha_str = f"alpha{int(config['alpha']*100):02d}"  # e.g., "alpha10" or "alpha20"
    output_dir = base_dir / f'acs/results/true_marginal_{alpha_str}'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f'acs_true_marg_{alpha_str}_detailed.csv'
    df_results = save_results_to_csv(results, output_csv, config)

    print("\nDone!")
    print(f"Results saved to: {output_dir}")
