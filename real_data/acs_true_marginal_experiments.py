"""
ACS coverage experiments (fixed ACS snapshot, no row bootstrap).

marginal:
  Each replicate: draw n_calib_per_stratum PUMAs from each of strata 1–4 (calibration),
  then average coverage over every other eligible target PUMA (index 20). No bootstrap.

conditional:
  Same as marginal but calibration PUMAs are fixed once (group_selection_seed).

marginal_one_target:
  Each replicate: resample calibration + one random target PUMA; one coverage indicator.
  (Older Option B; not the same as conditional/marginal above.)

uniform_one_target:
  Each replicate: uniformly draw n_puma_groups calibration PUMAs from all eligible PUMAs,
  then uniformly draw one target PUMA from the remaining eligible PUMAs. No bootstrap.

Note: repeated_experiments_stratified_acs.py uses bootstrap; these true-marginal scripts do not.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs

from methods.mu_methods import create_mu_method_ols_global_only, create_mu_method_ols_offset
from methods.donor_hcp import get_hcp_train_cal_split
from methods.donor_hcp import compute_donor_hcp_randomized_interval
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

# Fixed prediction target: 21st individual (index 20). History size o uses indices 0..o-1.
TARGET_INDEX = 20
MIN_TARGET_PUMA_SIZE = TARGET_INDEX + 1  # 21

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


def compute_global_pooled_income_interval(df: pd.DataFrame, alpha: float):
    """One global prediction set: empirical income quantiles over all ACS rows."""
    incomes = df["income"].astype(float).values
    q_lo, q_hi = np.quantile(incomes, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(q_lo), float(q_hi)


def alpha_to_tag(alpha: float) -> str:
    """Stable filename tag, e.g. 0.1 -> alpha10 and 0.075 -> alpha07p5."""
    pct = f"{100.0 * float(alpha):.6g}".replace(".", "p")
    if "p" not in pct and len(pct) < 2:
        pct = pct.zfill(2)
    return f"alpha{pct}"


def _interval_to_income_bounds(interval):
    lo, hi = interval
    return (np.expm1(lo) if np.isfinite(lo) else -np.inf,
            np.expm1(hi) if np.isfinite(hi) else np.inf)


def _result_record(interval, true_y):
    """Standard per-method result dict with log and income endpoints."""
    lo_inc, hi_inc = _interval_to_income_bounds(interval)
    return {
        "coverage": _covered(interval, true_y),
        "width": _width(interval),
        "width_income": _width_income_from_log1p_interval(interval),
        "lower": interval[0],
        "upper": interval[1],
        "lower_income": lo_inc,
        "upper_income": hi_inc,
    }


def _without_within_group_training(mu_method):
    """Return a mu_method wrapper that ignores within-group history."""
    base_predict_global = mu_method["predict_global"]
    out = dict(mu_method)

    def fit_group_adjustment(model_global, u_group_vector,
                             Z_group_list, training_index_vector):
        return 0.0

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return base_predict_global(model_global, x_vector, u_group_vector)

    out["fit_group_adjustment"] = fit_group_adjustment
    out["predict_group_mu"] = predict_group_mu
    return out


def _make_mu_hcp(config):
    """Mu method for HCP-style methods, optionally disabling within-group training."""
    mu_hcp = create_mu_method_ols_offset()
    if not bool(config.get("within_group", True)):
        mu_hcp = _without_within_group_training(mu_hcp)
    return mu_hcp


def _tau_override(config):
    """Use tau=0 when within-group training is disabled."""
    return None if bool(config.get("within_group", True)) else 0


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


def sample_calibration_fixed_per_stratum(
    strata,
    group_counts,
    selection_seed,
    o_values,
    n_calib_per_stratum=4,
    n_calib_strata=4,
):
    """
    Draw a fixed calibration set: n_calib_per_stratum PUMAs from each of strata 1..n_calib_strata.
    Returns sorted calib_groups (same draw for all replicates when selection_seed is fixed).
    """
    stratum_names = sorted(list(strata.keys()))
    if len(stratum_names) < n_calib_strata:
        raise ValueError(
            f"Need at least {n_calib_strata} strata, got {len(stratum_names)}"
        )

    max_o = max(int(o) for o in o_values)
    calib_strata = stratum_names[:n_calib_strata]
    rng = np.random.default_rng(selection_seed)

    calib_groups = []
    for k in calib_strata:
        members = [
            int(g)
            for g in strata[k]
            if int(group_counts.get(g, 0)) > max_o
        ]
        if len(members) < n_calib_per_stratum:
            raise ValueError(
                f"Stratum {k}: need {n_calib_per_stratum} PUMAs with size > {max_o}, "
                f"got {len(members)}"
            )
        chosen = rng.choice(np.asarray(members), size=n_calib_per_stratum, replace=False)
        calib_groups.extend(int(g) for g in chosen)

    return sorted(calib_groups)


def _load_group_data_static(df, X, group_col, groups, truncate_n=None, rng=None):
    """Load observed rows per PUMA.

    If rng is provided, rows are permuted within each PUMA without replacement.
    This keeps the ACS snapshot fixed while making the target stream order random
    across true-marginal replicates.
    """
    group_data = {}
    for grp in groups:
        grp_idx = np.where((df[group_col] == grp).values)[0]
        if rng is not None:
            grp_idx = rng.permutation(grp_idx)
        group_data[grp] = {
            'X': X[grp_idx],
            'Y': df.iloc[grp_idx]['y'].values,
            'income': df.iloc[grp_idx]['income'].values.astype(float),
        }
        if truncate_n is not None:
            n_cap = int(truncate_n)
            for key in ('X', 'Y', 'income'):
                group_data[grp][key] = group_data[grp][key][:n_cap]
    return group_data


def _test_groups_from_calib(eligible_groups, calib_groups, group_counts,
                            min_target_size=MIN_TARGET_PUMA_SIZE):
    calib_set = set(calib_groups)
    return sorted(
        int(g) for g in eligible_groups
        if g not in calib_set and int(group_counts.get(g, 0)) >= min_target_size
    )


def strata_where_all_pumas_at_least(strata, group_counts, min_size):
    """Return sorted stratum keys whose every PUMA has at least min_size observations."""
    out = []
    for k in sorted(strata.keys()):
        if all(int(group_counts[g]) >= min_size for g in strata[k]):
            out.append(k)
    return out


def pick_target_stratum_key(strata, group_counts, target_stratum_index, min_target_size):
    """
  Pick target stratum: must have all PUMAs with size >= min_target_size.

    target_stratum_index: 1-based index among sorted strata (default 5), or -1 for
    the highest qualifying stratum.
    """
    qualifying = strata_where_all_pumas_at_least(strata, group_counts, min_target_size)
    if not qualifying:
        raise ValueError(
            f"No stratum has all PUMAs with size >= {min_target_size}. "
            f"Qualifying check failed on strata: {list(strata.keys())}"
        )
    names = sorted(strata.keys())
    if target_stratum_index == -1:
        return qualifying[-1]
    idx = int(target_stratum_index) - 1
    if idx < 0 or idx >= len(names):
        raise ValueError(f"target_stratum_index must be in 1..{len(names)}")
    key = names[idx]
    if key not in qualifying:
        raise ValueError(
            f"Stratum {target_stratum_index} ({key}) is not all >= {min_target_size}. "
            f"Qualifying strata: {qualifying}"
        )
    return key


def sample_calibration_and_target_symmetric(
    strata,
    group_counts,
    eligible_groups,
    selection_seed,
    o_values,
    n_calib_per_stratum=5,
    n_calib_strata=4,
    min_target_size=MIN_TARGET_PUMA_SIZE,
):
    """
    Option B: stratified calibration + uniform target from remaining eligible PUMAs.

    - Draw 5 PUMAs without replacement from each of strata 1..n_calib_strata (size > max o).
    - Draw 1 target PUMA uniformly from eligible PUMAs not in the calibration set
      (size >= min_target_size so TARGET_INDEX exists).
    - Target and calibration sets are disjoint by construction.

    Returns (calib_groups, test_group) or (None, None) if pools are too small.
    """
    stratum_names = sorted(list(strata.keys()))
    if len(stratum_names) < n_calib_strata:
        raise ValueError(
            f"Need at least {n_calib_strata} strata, got {len(stratum_names)}"
        )

    max_o = max(int(o) for o in o_values)
    calib_strata = stratum_names[:n_calib_strata]
    rng = np.random.default_rng(selection_seed)

    calib_groups = []
    for k in calib_strata:
        members = [
            int(g)
            for g in strata[k]
            if int(group_counts.get(g, 0)) > max_o
        ]
        if len(members) < n_calib_per_stratum:
            return None, None
        chosen = rng.choice(np.asarray(members), size=n_calib_per_stratum, replace=False)
        calib_groups.extend(int(g) for g in chosen)

    calib_set = set(calib_groups)
    target_pool = [
        int(g)
        for g in eligible_groups
        if g not in calib_set and int(group_counts.get(g, 0)) >= min_target_size
    ]
    if len(target_pool) == 0:
        return None, None

    test_group = int(rng.choice(np.asarray(target_pool, dtype=int), size=1)[0])
    if test_group in calib_set:
        raise RuntimeError("Target PUMA overlapped calibration set (should not happen).")
    return sorted(calib_groups), test_group


def sample_calibration_and_target_uniform(
    eligible_groups,
    group_counts,
    selection_seed,
    n_calib_groups=20,
    min_target_size=MIN_TARGET_PUMA_SIZE,
):
    """
    Uniformly draw calibration and target PUMAs without stratification.

    - Draw n_calib_groups calibration PUMAs uniformly without replacement.
    - Draw one target PUMA uniformly from eligible PUMAs not in calibration.
    - Target PUMA must have at least min_target_size rows.
    """
    target_eligible = [
        int(g)
        for g in eligible_groups
        if int(group_counts.get(g, 0)) >= min_target_size
    ]
    if len(target_eligible) <= n_calib_groups:
        return None, None

    rng = np.random.default_rng(selection_seed)
    calib_groups = rng.choice(
        np.asarray(target_eligible, dtype=int),
        size=int(n_calib_groups),
        replace=False,
    ).tolist()
    calib_set = set(int(g) for g in calib_groups)
    target_pool = [int(g) for g in target_eligible if int(g) not in calib_set]
    if len(target_pool) == 0:
        return None, None

    test_group = int(rng.choice(np.asarray(target_pool, dtype=int), size=1)[0])
    if test_group in calib_set:
        raise RuntimeError("Target PUMA overlapped calibration set (should not happen).")

    return sorted(int(g) for g in calib_groups), test_group


def count_possible_symmetric_draws(
    strata,
    group_counts,
    eligible_groups,
    o_values,
    n_calib_per_stratum=5,
    n_calib_strata=4,
    min_target_size=MIN_TARGET_PUMA_SIZE,
):
    """Approx. number of distinct (calib, target) pairs under Option B sampling."""
    max_o = max(int(o) for o in o_values)
    names = sorted(strata.keys())
    calib_strata = names[:n_calib_strata]

    total = 1
    for k in calib_strata:
        n = sum(1 for g in strata[k] if int(group_counts.get(g, 0)) > max_o)
        if n < n_calib_per_stratum:
            return 0
        total *= math.comb(n, n_calib_per_stratum)

    n_elig_target = sum(
        1 for g in eligible_groups if int(group_counts.get(g, 0)) >= min_target_size
    )
    n_calib = n_calib_per_stratum * n_calib_strata
    # Target pool size depends on which calib PUMAs were drawn; upper bound:
    total *= max(1, n_elig_target - n_calib)
    return total


# ==================== Single replicate ====================

def run_one_replicate(df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx):
    """
    Run one true marginal replicate (Option B):
    1. Five PUMAs from each of strata 1–4 (calibration), disjoint from target
    2. One target PUMA uniform from remaining eligible (size >= 21)
    3. Full PUMA rows; fixed target at index 20
    """
    alpha = config['alpha']
    alpha_sel = config.get('alpha_selection', 0.5)
    n_rep = config.get('n_repeated', 50)
    n_calib_per_stratum = config.get('n_calib_per_stratum', 5)
    n_puma_groups = config.get('n_puma_groups', 20)
    o_values_rep = config.get('o_values', [0, 5, 10, 15, 20])
    eligible_groups = config.get('eligible_groups', [])
    design = config.get('design', 'marginal_one_target')

    group_counts = df.groupby(group_col).size()

    selection_seed = config['seed'] + replicate_idx * 1009
    try:
        if design == 'uniform_one_target':
            calib_groups, test_group = sample_calibration_and_target_uniform(
                eligible_groups=eligible_groups,
                group_counts=group_counts,
                selection_seed=selection_seed,
                n_calib_groups=n_puma_groups,
                min_target_size=MIN_TARGET_PUMA_SIZE,
            )
        else:
            calib_groups, test_group = sample_calibration_and_target_symmetric(
                strata=strata,
                group_counts=group_counts,
                eligible_groups=eligible_groups,
                selection_seed=selection_seed,
                o_values=o_values_rep,
                n_calib_per_stratum=n_calib_per_stratum,
                n_calib_strata=4,
                min_target_size=MIN_TARGET_PUMA_SIZE,
            )
    except (ValueError, RuntimeError):
        return None

    if calib_groups is None or test_group is None:
        return None

    if test_group in set(calib_groups):
        raise RuntimeError(
            f"Replicate {replicate_idx}: target PUMA {test_group} in calibration set."
        )

    if int(group_counts[test_group]) < MIN_TARGET_PUMA_SIZE:
        return None

    # Full PUMA rows (variable group sizes); target is always index TARGET_INDEX.
    # Optional row permutation randomizes the stream order without bootstrapping.
    row_rng = (
        np.random.default_rng(config['seed'] + replicate_idx * 1009 + 811)
        if bool(config.get('permute_rows', False))
        else None
    )
    group_data = {}
    for grp in calib_groups + [test_group]:
        grp_mask = (df[group_col] == grp).values
        grp_idx = np.where(grp_mask)[0]
        if row_rng is not None:
            grp_idx = row_rng.permutation(grp_idx)
        group_data[grp] = {
            'X': X[grp_idx],
            'Y': df.iloc[grp_idx]['y'].values,
            'income': df.iloc[grp_idx]['income'].values.astype(float),
        }

    # Build Z_calibration for all calibration groups
    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    # HCP baselines: S_tilde selection without donor slot (full S_tilde for calibration)
    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel,
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, 'tolist') else list(calib_idx)

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = _make_mu_hcp(config)

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

    if len(group_data[test_group]['Y']) <= TARGET_INDEX:
        # Not enough observations - mark baselines as NaN too
        nan_rec = {
            'coverage': np.nan, 'width': np.nan, 'width_income': np.nan,
            'lower': np.nan, 'upper': np.nan, 'lower_income': np.nan, 'upper_income': np.nan,
        }
        for method in BASELINE_METHODS:
            baseline_result[method][0] = nan_rec.copy()
        for o in o_values:
            for method in HCP_METHODS:
                hcp_result[method][o] = nan_rec.copy()
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
        return {
            'baseline': baseline_result,
            'hcp': hcp_result,
            'stdcp': stdcp_result,
            'income_target': np.nan,
        }

    target_index = TARGET_INDEX
    x_target = group_data[test_group]['X'][target_index]
    true_y = group_data[test_group]['Y'][target_index]
    income_target = float(group_data[test_group]['income'][target_index])

    # Baseline methods at fixed target (index 20)
    mu_hat_baseline = mu_baseline['predict_global'](model_baseline, x_target, U_test[0])
    for method, T in [
        ('HCP', T_hcp),
        ('Pooling', T_pool),
        ('Subsampling', T_sub),
        ('Repeated', T_rep),
    ]:
        interval = _interval_from_radius(mu_hat_baseline, T)
        baseline_result[method][0] = _result_record(interval, true_y)

    Z_test_full = [
        {'X': group_data[test_group]['X'][i], 'Y': group_data[test_group]['Y'][i]}
        for i in range(len(group_data[test_group]['Y']))
    ]

    # HCP methods and Std-CP: compute for each o value
    for o in o_values:
        if len(group_data[test_group]['Y']) <= max(o, TARGET_INDEX):
            nan_rec = {
                'coverage': np.nan, 'width': np.nan, 'width_income': np.nan,
                'lower': np.nan, 'upper': np.nan, 'lower_income': np.nan, 'upper_income': np.nan,
            }
            for method in HCP_METHODS:
                hcp_result[method][o] = nan_rec.copy()
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
            continue

        # Donor-HCP with within-group correction (always computed; not forced to HCP at o=0)
        try:
            dhcp_seed = (
                config['seed']
                + (replicate_idx + 1) * 1009
                + (o + 1) * 131
                + 17
            )
            res_dhcp = compute_donor_hcp_randomized_interval(
                U_calibration=U_calibration_full,
                Z_calibration=Z_calibration_full,
                U_test=U_test,
                Z_test=Z_test_full,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_hcp,
                test_index_target=target_index,
                tau_override=_tau_override(config),
                random_seed=dhcp_seed,
            )
            int_dhcp = res_dhcp['interval']
        except Exception:
            int_dhcp = (-np.inf, np.inf)

        hcp_result['Donor-HCP'][o] = _result_record(int_dhcp, true_y)

        # Sample-HCP (randomized) - pass full Z_test, not truncated
        try:
            res_shcp = compute_sample_hcp_randomized_interval(
                U_calibration=U_calibration_full,
                Z_calibration=Z_calibration_full,
                U_test=U_test, Z_test=Z_test_full,  # FULL Z_test
                o_observed=o, alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_hcp,
                test_index_target=target_index,  # FIXED target_index
                tau_override=_tau_override(config),
            )
            int_shcp = res_shcp['interval']
        except Exception:
            int_shcp = (-np.inf, np.inf)

        hcp_result['S-HCP'][o] = _result_record(int_shcp, true_y)

        # Standard CP (on test group only)
        if o > 0:
            y_cal_test = np.array([group_data[test_group]['Y'][i] for i in range(o)])
            mu_stdcp = np.mean(y_cal_test)
            scores_stdcp = np.abs(y_cal_test - mu_stdcp)
            q_stdcp = np.quantile(scores_stdcp, 1 - alpha, method='higher')
            int_stdcp = (mu_stdcp - q_stdcp, mu_stdcp + q_stdcp)
        else:
            int_stdcp = (-np.inf, np.inf)

        stdcp_result['Std-CP'][o] = _result_record(int_stdcp, true_y)

    return {
        'baseline': baseline_result,
        'hcp': hcp_result,
        'stdcp': stdcp_result,
        'income_target': income_target,
    }


def run_one_replicate_avg_over_targets(
    df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
    fixed_calib_groups=None,
):
    """
    Average coverage over all target PUMAs disjoint from calibration.

    - fixed_calib_groups is None (marginal): resample calib PUMAs each replicate.
    - fixed_calib_groups set (conditional): same calib PUMAs every replicate.
    - Uses observed ACS rows only (no bootstrap). Target index TARGET_INDEX.
    """
    alpha = config['alpha']
    alpha_sel = config.get('alpha_selection', 0.5)
    n_rep = config.get('n_repeated', 50)
    truncate_n = config.get('truncate_n', None)
    n_calib_per = config.get('n_calib_per_stratum', 4)
    o_values_rep = config.get('o_values', o_values)

    group_counts = df.groupby(group_col).size()

    if fixed_calib_groups is not None:
        calib_groups = list(fixed_calib_groups)
    else:
        selection_seed = config['seed'] + replicate_idx * 1009
        try:
            calib_groups = sample_calibration_fixed_per_stratum(
                strata=strata,
                group_counts=group_counts,
                selection_seed=selection_seed,
                o_values=o_values_rep,
                n_calib_per_stratum=n_calib_per,
                n_calib_strata=4,
            )
        except ValueError:
            return None

    test_groups = _test_groups_from_calib(
        eligible_groups, calib_groups, group_counts,
    )
    if len(test_groups) == 0:
        return None

    all_groups = list(calib_groups) + test_groups
    row_rng = (
        np.random.default_rng(config['seed'] + replicate_idx * 1009 + 811)
        if bool(config.get('permute_rows', False))
        else None
    )
    group_data = _load_group_data_static(
        df, X, group_col, all_groups, truncate_n=truncate_n, rng=row_rng,
    )

    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel,
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, 'tolist') else list(calib_idx)

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = _make_mu_hcp(config)

    model_baseline = mu_baseline['fit_global'](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
    )

    scores_list = []
    for j in calib_idx:
        Zj = Z_calibration_full[j]
        yj = np.array([z['Y'] for z in Zj])
        muj = np.array([
            mu_baseline['predict_global'](model_baseline, z['X'], U_calibration_full[j])
            for z in Zj
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    T_hcp = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep = compute_repeated_subsampling_interval_radius(scores_list, alpha, n_rep)

    U_test = np.zeros((1, 1))
    baseline_result = {m: {} for m in BASELINE_METHODS}
    hcp_result = {m: {} for m in HCP_METHODS}
    stdcp_result = {m: {} for m in STD_CP_METHODS}
    income_vals = []

    for o in o_values:
        eligible_test = [
            grp for grp in test_groups
            if len(group_data[grp]['Y']) > max(o, TARGET_INDEX)
        ]
        if len(eligible_test) == 0:
            nan_rec = {
                'coverage': np.nan, 'width': np.nan, 'width_income': np.nan,
                'lower': np.nan, 'upper': np.nan, 'lower_income': np.nan, 'upper_income': np.nan,
            }
            for method in BASELINE_METHODS:
                baseline_result[method][0] = nan_rec.copy()
            for method in HCP_METHODS:
                hcp_result[method][o] = nan_rec.copy()
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
            continue

        base_cov = {m: [] for m in BASELINE_METHODS}
        base_wid = {m: [] for m in BASELINE_METHODS}
        base_wid_inc = {m: [] for m in BASELINE_METHODS}
        hcp_cov = {m: [] for m in HCP_METHODS}
        hcp_wid = {m: [] for m in HCP_METHODS}
        hcp_wid_inc = {m: [] for m in HCP_METHODS}
        std_cov = {m: [] for m in STD_CP_METHODS}
        std_wid = {m: [] for m in STD_CP_METHODS}
        std_wid_inc = {m: [] for m in STD_CP_METHODS}
        o_income_vals = []

        for test_group in eligible_test:
            target_index = TARGET_INDEX
            x_target = group_data[test_group]['X'][target_index]
            true_y = group_data[test_group]['Y'][target_index]
            o_income_vals.append(float(group_data[test_group]['income'][target_index]))

            mu_hat_baseline = mu_baseline['predict_global'](model_baseline, x_target, U_test[0])
            for method, T in [
                ('HCP', T_hcp),
                ('Pooling', T_pool),
                ('Subsampling', T_sub),
                ('Repeated', T_rep),
            ]:
                interval = _interval_from_radius(mu_hat_baseline, T)
                base_cov[method].append(_covered(interval, true_y))
                base_wid[method].append(_width(interval))
                base_wid_inc[method].append(_width_income_from_log1p_interval(interval))

            Z_test_full = [
                {'X': group_data[test_group]['X'][i], 'Y': group_data[test_group]['Y'][i]}
                for i in range(len(group_data[test_group]['Y']))
            ]

            try:
                dhcp_seed = (
                    config['seed']
                    + (replicate_idx + 1) * 1009
                    + (o + 1) * 131
                    + 17
                )
                res_dhcp = compute_donor_hcp_randomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test,
                    Z_test=Z_test_full,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                    test_index_target=target_index,
                    tau_override=_tau_override(config),
                    random_seed=dhcp_seed,
                )
                int_dhcp = res_dhcp['interval']
            except Exception:
                int_dhcp = (-np.inf, np.inf)

            hcp_cov['Donor-HCP'].append(_covered(int_dhcp, true_y))
            hcp_wid['Donor-HCP'].append(_width(int_dhcp))
            hcp_wid_inc['Donor-HCP'].append(_width_income_from_log1p_interval(int_dhcp))

            try:
                res_shcp = compute_sample_hcp_randomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test,
                    Z_test=Z_test_full,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                    test_index_target=target_index,
                    tau_override=_tau_override(config),
                )
                int_shcp = res_shcp['interval']
            except Exception:
                int_shcp = (-np.inf, np.inf)

            hcp_cov['S-HCP'].append(_covered(int_shcp, true_y))
            hcp_wid['S-HCP'].append(_width(int_shcp))
            hcp_wid_inc['S-HCP'].append(_width_income_from_log1p_interval(int_shcp))

            if o > 0:
                y_hist = np.array([group_data[test_group]['Y'][i] for i in range(o)])
                mu_stdcp = np.mean(y_hist)
                q_stdcp = np.quantile(np.abs(y_hist - mu_stdcp), 1 - alpha, method='higher')
                int_stdcp = (mu_stdcp - q_stdcp, mu_stdcp + q_stdcp)
            else:
                int_stdcp = (-np.inf, np.inf)

            std_cov['Std-CP'].append(_covered(int_stdcp, true_y))
            std_wid['Std-CP'].append(_width(int_stdcp))
            std_wid_inc['Std-CP'].append(_width_income_from_log1p_interval(int_stdcp))

        if o_income_vals:
            income_vals.extend(o_income_vals)

        for method in BASELINE_METHODS:
            baseline_result[method][0] = {
                'coverage': float(np.mean(base_cov[method])),
                'width': float(np.nanmean(base_wid[method])),
                'width_income': float(np.nanmean(base_wid_inc[method])),
                'lower': np.nan,
                'upper': np.nan,
                'lower_income': np.nan,
                'upper_income': np.nan,
            }

        for method in HCP_METHODS:
            hcp_result[method][o] = {
                'coverage': float(np.mean(hcp_cov[method])),
                'width': float(np.nanmean(hcp_wid[method])),
                'width_income': float(np.nanmean(hcp_wid_inc[method])),
                'lower': np.nan,
                'upper': np.nan,
                'lower_income': np.nan,
                'upper_income': np.nan,
            }

        for method in STD_CP_METHODS:
            stdcp_result[method][o] = {
                'coverage': float(np.mean(std_cov[method])),
                'width': float(np.nanmean(std_wid[method])),
                'width_income': float(np.nanmean(std_wid_inc[method])),
                'lower': np.nan,
                'upper': np.nan,
                'lower_income': np.nan,
                'upper_income': np.nan,
            }

    return {
        'baseline': baseline_result,
        'hcp': hcp_result,
        'stdcp': stdcp_result,
        'income_target': float(np.nanmean(income_vals)) if income_vals else np.nan,
    }


# ==================== Worker functions ====================

_WORKER_SHARED = None

def _init_worker(df, X, eligible_groups, strata, group_col, o_values, config,
                 calib_groups=None, test_groups=None):
    global _WORKER_SHARED
    _WORKER_SHARED = (
        df, X, list(eligible_groups), strata, group_col, o_values, config,
        calib_groups, test_groups,
    )


def _run_one_replicate_worker(replicate_idx):
    df, X, eligible_groups, strata, group_col, o_values, config, calib_groups, test_groups = (
        _WORKER_SHARED
    )
    design = config.get('design', 'marginal')
    if design == 'conditional':
        result = run_one_replicate_avg_over_targets(
            df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
            fixed_calib_groups=calib_groups,
        )
    elif design in ('marginal_one_target', 'uniform_one_target'):
        result = run_one_replicate(
            df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
        )
    else:
        # marginal: resample calib each replicate, average over all test PUMAs
        result = run_one_replicate_avg_over_targets(
            df, X, eligible_groups, strata, group_col, o_values, config, replicate_idx,
            fixed_calib_groups=None,
        )
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
    lower_log = np.full((B, n_m, n_o), np.nan)
    upper_log = np.full((B, n_m, n_o), np.nan)
    lower_income = np.full((B, n_m, n_o), np.nan)
    upper_income = np.full((B, n_m, n_o), np.nan)
    income_targets = np.full(B, np.nan)

    print(f"  Running {B} true marginal replicates with {n_workers} workers")

    def _write_result(rep_idx, result):
        if result is None:
            return

        income_targets[rep_idx] = result.get('income_target', np.nan)

        def _store(m_i, o_i, rec):
            cov[rep_idx, m_i, o_i] = rec['coverage']
            wid[rep_idx, m_i, o_i] = rec['width']
            wid_income[rep_idx, m_i, o_i] = rec['width_income']
            lower_log[rep_idx, m_i, o_i] = rec['lower']
            upper_log[rep_idx, m_i, o_i] = rec['upper']
            lower_income[rep_idx, m_i, o_i] = rec['lower_income']
            upper_income[rep_idx, m_i, o_i] = rec['upper_income']

        for method in BASELINE_METHODS:
            if 0 in result['baseline'][method]:
                _store(m_map[method], 0, result['baseline'][method][0])

        for method in HCP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                if o in result['hcp'][method]:
                    _store(m_i, o_i, result['hcp'][method][o])

        for method in STD_CP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                if o in result['stdcp'][method]:
                    _store(m_i, o_i, result['stdcp'][method][o])

    calib_groups = config.get('fixed_calib_groups')
    test_groups = config.get('fixed_test_groups')

    if n_workers <= 1:
        _init_worker(
            df, X, eligible_groups, strata, group_col, o_values, config,
            calib_groups, test_groups,
        )
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
            initargs=(
                df, X, eligible_groups, strata, group_col, o_values, config,
                calib_groups, test_groups,
            ),
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
        'lower_log': lower_log,
        'upper_log': upper_log,
        'lower_income': lower_income,
        'upper_income': upper_income,
        'income_targets': income_targets,
    }


# ==================== Save results ====================

def save_results_to_csv(results, output_path, global_pooled=None):
    """Save results to CSV in long format."""
    methods = results['methods']
    o_values = results['o_values']
    cov = results['coverage']
    wid = results['width']
    wid_income = results['width_income']
    lower_log = results['lower_log']
    upper_log = results['upper_log']
    lower_income = results['lower_income']
    upper_income = results['upper_income']
    income_targets = results['income_targets']

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
                    'lower': lower_log[b, m_i, o_i],
                    'upper': upper_log[b, m_i, o_i],
                    'lower_income': lower_income[b, m_i, o_i],
                    'upper_income': upper_income[b, m_i, o_i],
                    'income_target': income_targets[b],
                })

    if global_pooled is not None:
        L_p, U_p = global_pooled['lower'], global_pooled['upper']
        for b in range(B):
            inc = income_targets[b]
            covered = (
                float(L_p <= inc <= U_p)
                if np.isfinite(inc) and np.isfinite(L_p) and np.isfinite(U_p)
                else np.nan
            )
            for o in o_values:
                rows.append({
                    'replicate': b,
                    'method': 'Pooled-Income-Quantile',
                    'o': o,
                    'coverage': covered,
                    'width': np.nan,
                    'width_income': U_p - L_p,
                    'lower': np.nan,
                    'upper': np.nan,
                    'lower_income': L_p,
                    'upper_income': U_p,
                    'income_target': inc,
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"  Saved detailed results to {output_path}")
    return df_out


def save_summaries(df_out: pd.DataFrame, summary_dir: Path):
    """Write coverage/width summary CSVs (long + pivot)."""
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (method, o), g in df_out.groupby(['method', 'o']):
        cov_vals = g['coverage'].dropna().values
        n = len(cov_vals)
        if n == 0:
            continue
        p = float(np.mean(cov_vals))
        se = float(np.sqrt(p * (1 - p) / n))
        w_inc = g['width_income'].replace([np.inf, -np.inf], np.nan).dropna().values
        rows.append({
            'method': method,
            'o': o,
            'coverage_mean': p,
            'coverage_std': float(np.std(cov_vals, ddof=1)) if n > 1 else 0.0,
            'coverage_se': se,
            'width_income_median': float(np.median(w_inc)) if len(w_inc) else np.nan,
            'width_income_mean': float(np.mean(w_inc)) if len(w_inc) else np.nan,
            'width_income_std': float(np.std(w_inc, ddof=1)) if len(w_inc) > 1 else np.nan,
            'n': n,
        })

    df_long = pd.DataFrame(rows)
    df_long.to_csv(summary_dir / 'acs_true_marg_summary_long.csv', index=False)

    if len(df_long):
        cov_pivot = df_long.pivot(index='o', columns='method', values='coverage_mean')
        cov_pivot.to_csv(summary_dir / 'acs_true_marg_coverage_summary.csv')
        wid_pivot = df_long.pivot(index='o', columns='method', values='width_income_median')
        wid_pivot.to_csv(summary_dir / 'acs_true_marg_width_summary.csv')

    print(f"  Saved summaries to {summary_dir}")
    return df_long


def save_endpoints_analysis(df_out, global_pooled, summary_dir, o_compare=20):
    """Report fixed global pooled endpoints vs Donor-HCP replicate intervals."""
    summary_dir = Path(summary_dir)
    L_p, U_p = global_pooled['lower'], global_pooled['upper']

    lines = [
        "ACS true marginal — interval endpoints",
        "=" * 60,
        "",
        "Global pooled-income quantile (all filtered ACS rows, one interval):",
        f"  lower_income = ${L_p:,.0f}",
        f"  upper_income = ${U_p:,.0f}",
        f"  width        = ${U_p - L_p:,.0f}",
        "",
    ]

    dhcp = df_out[(df_out['method'] == 'Donor-HCP') & (df_out['o'] == o_compare)].copy()
    fin = dhcp[np.isfinite(dhcp['lower_income']) & np.isfinite(dhcp['upper_income'])]
    if len(fin):
        med_lo = fin['lower_income'].median()
        med_hi = fin['upper_income'].median()
        lines += [
            f"Donor-HCP (with local correction) at o={o_compare} — replicate intervals (income $):",
            f"  median lower = ${med_lo:,.0f}",
            f"  median upper = ${med_hi:,.0f}",
            f"  median width = ${(med_hi - med_lo):,.0f}",
            f"  replicate lower: 5%=${fin['lower_income'].quantile(0.05):,.0f}, "
            f"95%=${fin['lower_income'].quantile(0.95):,.0f}",
            f"  replicate upper: 5%=${fin['upper_income'].quantile(0.05):,.0f}, "
            f"95%=${fin['upper_income'].quantile(0.95):,.0f}",
            "",
            "Geometry vs global pooled interval (per replicate):",
        ]
        nested = ((fin['lower_income'] >= L_p) & (fin['upper_income'] <= U_p)).mean()
        overlap = (
            (fin['upper_income'] >= L_p) & (fin['lower_income'] <= U_p)
        ).mean()
        strictly_lower = (fin['upper_income'] < L_p).mean()
        strictly_higher = (fin['lower_income'] > U_p).mean()
        lines += [
            f"  nested inside pooled:     {100 * nested:.1f}%",
            f"  overlaps pooled:          {100 * overlap:.1f}%",
            f"  entirely below pooled:    {100 * strictly_lower:.1f}%",
            f"  entirely above pooled:    {100 * strictly_higher:.1f}%",
            "",
            "Interpretation: Donor-HCP intervals vary by replicate (centered at mu_hat);",
            "they are not nested in each other. Compare overlap with the fixed pooled band.",
        ]

    text = "\n".join(lines)
    out_txt = summary_dir / 'acs_true_marg_endpoints.txt'
    out_txt.write_text(text)
    print(text)
    print(f"\n  Saved endpoint report: {out_txt}")

    rows = [{
        'method': 'Pooled-Income-Quantile',
        'o': 'global',
        'lower_income': L_p,
        'upper_income': U_p,
        'width_income': U_p - L_p,
    }]
    if len(fin):
        rows.append({
            'method': 'Donor-HCP',
            'o': o_compare,
            'lower_income': med_lo,
            'upper_income': med_hi,
            'width_income': med_hi - med_lo,
            'note': 'medians over replicates',
        })
    pd.DataFrame(rows).to_csv(summary_dir / 'acs_true_marg_endpoints_summary.csv', index=False)


# ==================== Main entry point ====================

if __name__ == '__main__':
    import argparse
    mp.set_start_method('fork', force=True)

    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1000, help='Number of replicates')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of workers')
    parser.add_argument('--alpha', type=float, default=0.2, help='Miscoverage level (0.2 => 80%% coverage)')
    parser.add_argument('--acs_state', type=str, default='CA')
    parser.add_argument('--acs_csv', type=str, default=None, help='Path to ACS PUMS CSV')
    parser.add_argument('--n_puma_groups', type=int, default=20, help='Number of non-target PUMAs')
    parser.add_argument('--min_puma_size', type=int, default=21,
                        help='Minimum PUMA size (must be >= TARGET_INDEX+1 for index 20)')
    parser.add_argument('--o_values', type=str, default='0,5,10,15,20', help='Comma-separated o values')
    parser.add_argument(
        '--design',
        type=str,
        default='marginal',
        choices=['marginal', 'conditional', 'marginal_one_target', 'uniform_one_target'],
        help=(
            'marginal: resample calib PUMAs each replicate, average over all test PUMAs. '
            'conditional: fixed calib PUMAs, same averaging (no bootstrap). '
            'marginal_one_target: stratified calib + one random target PUMA per replicate. '
            'uniform_one_target: uniform calib PUMAs + one uniform target PUMA per replicate.'
        ),
    )
    parser.add_argument(
        '--n_calib_per_stratum',
        type=int,
        default=None,
        help='PUMAs per stratum for calibration (default 5 if marginal, 4 if conditional)',
    )
    parser.add_argument(
        '--group_selection_seed',
        type=int,
        default=42,
        help='Seed for fixed calibration PUMA draw (conditional design)',
    )
    parser.add_argument(
        '--truncate_n',
        type=int,
        default=None,
        help='If set, use only the first N rows per PUMA (DGP uses N=21)',
    )
    parser.add_argument(
        '--no_within_group',
        action='store_true',
        help='Disable within-group training/shrinkage in Donor-HCP and S-HCP.',
    )
    parser.add_argument(
        '--permute_rows',
        action='store_true',
        help=(
            'Randomly permute rows within every selected PUMA in each replicate. '
            'This changes the stream order without bootstrapping or replacement.'
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ACS TRUE MARGINAL COVERAGE EXPERIMENTS")
    print("=" * 70)

    acs_csv = Path(args.acs_csv) if args.acs_csv else (base_dir / 'acs/data/acs_data_all50states.csv')
    if not acs_csv.exists():
        raise FileNotFoundError(
            f"ACS data not found: {acs_csv}\n"
            "Run: python real_data/acs/download_acs_ca_pums.py\n"
            "Or pass --acs_csv /path/to/acs_data_all50states.csv"
        )

    # Load data
    print("\nLoading ACS data...")
    df = load_and_clean_acs_pums(
        str(acs_csv),
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

    group_counts = counts
    target_eligible = [
        g for g in eligible_groups if int(group_counts[g]) >= MIN_TARGET_PUMA_SIZE
    ]

    print(f"  Number of strata: {len(strata)}")
    for i, k in enumerate(sorted(strata.keys()), start=1):
        n_big = sum(1 for g in strata[k] if int(group_counts[g]) >= MIN_TARGET_PUMA_SIZE)
        role = "calibration strata" if i <= 4 else "target pool only"
        print(f"    Stratum {i} {k}: {len(strata[k])} PUMAs, {n_big} with size>={MIN_TARGET_PUMA_SIZE} ({role})")
    print(f"  Target-eligible PUMAs (size >= {MIN_TARGET_PUMA_SIZE}): {len(target_eligible)}")

    o_values = sorted(set(int(v) for v in args.o_values.split(',')))
    print(f"\n  o values: {o_values}")

    design = args.design
    n_calib_per = args.n_calib_per_stratum
    if n_calib_per is None:
        n_calib_per = 4 if design in ('conditional', 'marginal') else 5

    fixed_calib_groups = None
    fixed_test_groups = None
    if design == 'conditional':
        fixed_calib_groups = sample_calibration_fixed_per_stratum(
            strata=strata,
            group_counts=group_counts,
            selection_seed=args.group_selection_seed,
            o_values=o_values,
            n_calib_per_stratum=n_calib_per,
            n_calib_strata=4,
        )
        calib_set = set(fixed_calib_groups)
        fixed_test_groups = sorted(
            g for g in eligible_groups
            if g not in calib_set and int(group_counts[g]) >= MIN_TARGET_PUMA_SIZE
        )
        if len(fixed_test_groups) == 0:
            raise ValueError('No test PUMAs left after fixing calibration set.')

    n_possible = count_possible_symmetric_draws(
        strata,
        group_counts,
        eligible_groups,
        o_values,
        n_calib_per_stratum=n_calib_per,
    )
    if design == 'conditional':
        print(
            f"\n  Conditional: fixed {n_calib_per} calib PUMAs/stratum × 4 strata "
            f"(seed={args.group_selection_seed}), no row bootstrap, "
            f"row permutation={args.permute_rows}, "
            f"average coverage over {len(fixed_test_groups)} test PUMAs (index {TARGET_INDEX})"
        )
        print(f"  Fixed calibration PUMAs: {fixed_calib_groups}")
    elif design == 'marginal_one_target':
        print(
            f"\n  Marginal one-target: {n_calib_per} calib PUMAs/stratum from strata 1–4 "
            f"(size > max(o)={max(o_values)}), then 1 target PUMA from the remainder; "
            f"row permutation={args.permute_rows}"
        )
        print(f"  Approx. upper bound on (calib, target) pairs: {n_possible:,}; B={args.B} replicates")
    elif design == 'uniform_one_target':
        print(
            f"\n  Uniform one-target: each replicate draws {args.n_puma_groups} calibration PUMAs "
            f"uniformly from all {len(target_eligible)} target-eligible PUMAs, then 1 target "
            f"PUMA uniformly from the remainder (index {TARGET_INDEX}); no row bootstrap; "
            f"row permutation={args.permute_rows}"
        )
    else:
        print(
            f"\n  Marginal: each replicate draws {n_calib_per} calib PUMAs/stratum × 4 strata "
            f"(size > max(o)={max(o_values)}), no row bootstrap, "
            f"row permutation={args.permute_rows}, "
            f"then averages coverage over all other eligible target PUMAs (index {TARGET_INDEX})"
        )
        print(f"  Typical test PUMAs per replicate: ~{max(1, len(target_eligible) - n_calib_per * 4)}")

    # Global pooled-income interval (one prediction set for all data)
    pooled_lo, pooled_hi = compute_global_pooled_income_interval(df, args.alpha)
    global_pooled = {
        'lower': pooled_lo,
        'upper': pooled_hi,
        'width': pooled_hi - pooled_lo,
    }
    print(f"\n  Global pooled-income interval (all {len(df)} rows):")
    print(f"    lower = ${pooled_lo:,.0f},  upper = ${pooled_hi:,.0f},  width = ${global_pooled['width']:,.0f}")

    # Config
    config = {
        'B': args.B,
        'seed': 456,
        'alpha': args.alpha,
        'alpha_selection': 0.5,
        'n_repeated': 50,
        'n_puma_groups': args.n_puma_groups,
        'n_calib_per_stratum': n_calib_per,
        'o_values': o_values,
        'eligible_groups': eligible_groups,
        'n_workers': args.n_workers,
        'design': design,
        'fixed_calib_groups': fixed_calib_groups,
        'fixed_test_groups': fixed_test_groups,
        'truncate_n': args.truncate_n,
        'within_group': not args.no_within_group,
        'permute_rows': args.permute_rows,
    }

    print(f"\nRunning ACS coverage experiments (design={design}):")
    print(f"  Alpha: {args.alpha} (nominal coverage {1 - args.alpha:.0%})")
    print(f"  Donor-HCP: stock randomized interval; within-group training = {not args.no_within_group}")
    print(
        f"  Target index: {TARGET_INDEX}; calib ∩ target = ∅; no ACS row bootstrap; "
        f"row permutation = {args.permute_rows}"
    )
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
    alpha_str = alpha_to_tag(config['alpha'])  # e.g., "alpha10" or "alpha07p5"
    result_prefix = 'true_marginal_permuted' if args.permute_rows else 'true_marginal'
    output_dir = base_dir / f'acs/results/{result_prefix}_{alpha_str}'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / f'acs_true_marg_{alpha_str}_detailed.csv'
    df_results = save_results_to_csv(results, output_csv, global_pooled=global_pooled)

    summary_dir = output_dir / 'summaries'
    df_summary = save_summaries(df_results, summary_dir)
    save_endpoints_analysis(df_results, global_pooled, summary_dir, o_compare=20)

    if len(df_summary):
        print("\n  Coverage summary (mean):")
        for method in ['Donor-HCP', 'Pooled-Income-Quantile', 'HCP']:
            sub = df_summary[df_summary['method'] == method]
            if len(sub) == 0:
                continue
            print(f"    {method}:")
            for _, row in sub.sort_values('o').iterrows():
                print(
                    f"      o={row['o']}: {row['coverage_mean']:.3f} "
                    f"(SE {row['coverage_se']:.4f}), "
                    f"width_income median={row['width_income_median']:,.0f}"
                )
        dhcp20 = df_summary[(df_summary['method'] == 'Donor-HCP') & (df_summary['o'] == 20)]
        if len(dhcp20):
            w_d = float(dhcp20['width_income_median'].iloc[0])
            w_p = global_pooled['width']
            if np.isfinite(w_d) and w_p > 0:
                pct = 100 * (1 - w_d / w_p) if w_d < w_p else -100 * (w_d / w_p - 1)
                cmp_word = "narrower" if w_d < w_p else "wider"
                print(
                    f"\n  Width: D-HCP median @ o=20 = ${w_d:,.0f}; "
                    f"global pooled = ${w_p:,.0f} ({abs(pct):.1f}% {cmp_word})"
                )

    print("\nDone!")
    print(f"Results saved to: {output_dir}")
