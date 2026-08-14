"""
ACS experiment with STRATIFIED SAMPLING of non-target groups.

Instead of randomly sampling 30 PUMAs as non-target groups, this variant:
1. Computes group-level covariates (mean age, mean income, etc.) for each PUMA
2. Stratifies PUMAs into k strata based on these covariates
3. Samples one PUMA from each stratum to form non-target groups
4. Uses remaining PUMAs as target groups

This ensures heterogeneity in the non-target (calibration) group pool.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.ticker import FuncFormatter
import multiprocessing as mp
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from blood_pressure.data_processing import load_and_clean_bp_data, build_design_matrix_bp
from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs, EMERGING_STATES

from methods.mu_methods import create_mu_method_ols_global_only, create_mu_method_ols_offset
from methods.donor_hcp import (
    compute_donor_hcp_randomized_interval,
    compute_donor_hcp_derandomized_interval,
)
from methods.sample_hcp import (
    compute_sample_hcp_randomized_interval,
    compute_sample_hcp_derandomized_interval,
)
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius,
)
from scores import absolute_residual_score, weighted_quantile
from methods.donor_hcp import _ALPHA_SPLIT_TIEBREAK_SEED, _select_s_tilde_with_tie_randomization

# Helper functions for within-group local correction (copied from DGP experiments)
def _m_j(o_observed: int, n_prime: int, use_within: bool) -> int:
    """Compute number of observations to use for within-group training."""
    if not use_within:
        return 0
    return int(min(np.floor(o_observed / 2), np.floor(n_prime / 2)))


def _local_mean_y(z_group, m_j: int):
    """Compute within-group mean Y from first m_j observations."""
    if m_j <= 0:
        return None
    return float(np.mean([z_group[i]["Y"] for i in range(m_j)]))


def _local_ols_model(z_group, m_j: int):
    """Fit local OLS model on first m_j observations."""
    if m_j < 2:
        return None
    x_local = np.array([z_group[i]["X"] for i in range(m_j)], dtype=float)
    y_local = np.array([z_group[i]["Y"] for i in range(m_j)], dtype=float)
    if m_j <= x_local.shape[1] + 1:
        return None
    x_aug = np.column_stack([np.ones(m_j), x_local])
    try:
        beta, *_ = np.linalg.lstsq(x_aug, y_local, rcond=None)
        return beta
    except Exception:
        return None


def _predict_local_ols(beta, x_vec):
    """Predict using local OLS model."""
    if beta is None:
        return None
    x_aug = np.concatenate([[1.0], np.asarray(x_vec, dtype=float).ravel()])
    return float(x_aug @ beta)


def _merge_mu(mu_global: float, mu_local, m_j: int, s_size: int, is_valid: bool) -> float:
    """Merge global and local predictions via shrinkage."""
    if m_j <= 0 or mu_local is None or not is_valid:
        return float(mu_global)
    lam = float(m_j) / float(s_size + m_j)
    return float((1.0 - lam) * mu_global + lam * mu_local)


def _compute_donor_hcp_interval_with_local(
    U_calibration, Z_calibration, U_test, Z_test,
    o_observed, alpha, alpha_selection, mu_method,
    local_mode, test_index_target, random_seed=None):
    """Donor-HCP with within-group local correction (copied from DGP)."""
    K = len(Z_calibration)
    N = np.array([len(Z_calibration[j]) for j in range(K)], dtype=int)
    test_idx = K

    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    if len(Z_test) < max(o_observed + 1, test_index_target + 1):
        raise ValueError("Z_test must have at least max(o+1, target+1) observations.")

    if K == 0:
        return {"interval": (-np.inf, np.inf), "mu_hat": 0.0, "number_selected_groups": 0, "donor_group_index": None}

    use_within = local_mode != "none"

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N, o_observed=o_observed, alpha_selection=alpha_selection, include_donor_slot=True)

    if len(S_tilde) == 0:
        global_model = mu_method["fit_global"](
            U_matrix=U_calibration, Z_list=Z_calibration, group_index_vector=list(range(K)))
        m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_within=use_within)
        local_est_test = _local_mean_y(Z_test, m_test) if local_mode == "mean" else (_local_ols_model(Z_test, m_test) if local_mode == "ols" else None)
        s_size = 1
        scores = []
        if o_observed >= (m_test + 1):
            for i in range(m_test, o_observed):
                z = Z_test[i]
                mu_g = mu_method["predict_global"](model_global=global_model, x_vector=z["X"], u_vector=U_test[0, :])
                if local_mode == "mean":
                    mu_loc, is_valid = local_est_test, local_est_test is not None
                elif local_mode == "ols":
                    mu_loc = _predict_local_ols(local_est_test, z["X"]) if local_est_test is not None else None
                    is_valid = mu_loc is not None
                else:
                    mu_loc, is_valid = None, False
                mu_tilde = _merge_mu(mu_g, mu_loc, m_test, s_size, is_valid)
                scores.append(abs(float(z["Y"]) - float(mu_tilde)))
            scores.append(np.inf)
            weights = np.ones(len(scores), dtype=float) / len(scores)
            q = weighted_quantile(scores, weights, alpha)
        else:
            q = np.inf
        x_target = Z_test[test_index_target]["X"]
        mu_g_target = mu_method["predict_global"](model_global=global_model, x_vector=x_target, u_vector=U_test[0, :])
        if local_mode == "mean":
            mu_loc_target, is_valid_target = local_est_test, local_est_test is not None
        elif local_mode == "ols":
            mu_loc_target = _predict_local_ols(local_est_test, x_target) if local_est_test is not None else None
            is_valid_target = mu_loc_target is not None
        else:
            mu_loc_target, is_valid_target = None, False
        mu_center = _merge_mu(mu_g_target, mu_loc_target, m_test, s_size, is_valid_target)
        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)
        return {"interval": interval, "mu_hat": float(mu_center), "number_selected_groups": 1, "donor_group_index": None}

    donor_rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    donor = int(donor_rng.choice(S_tilde))
    N_donor = int(N[donor])
    S_cal = np.sort(np.setdiff1d(S_tilde, [donor]))
    S = np.sort(np.concatenate([S_cal, [test_idx]]))
    s_size = len(S)
    S_comp = np.setdiff1d(list(range(K + 1)), S)
    global_model = None if len(S_comp) == 0 else mu_method["fit_global"](U_matrix=U_all, Z_list=Z_all, group_index_vector=list(S_comp))

    scores, weights = [], []
    for j in S_cal:
        N_j_prime = int(N[j])
        m_j = _m_j(o_observed=o_observed, n_prime=N_j_prime, use_within=use_within)
        if N_j_prime <= m_j:
            continue
        local_est_j = _local_mean_y(Z_calibration[j], m_j) if local_mode == "mean" else (_local_ols_model(Z_calibration[j], m_j) if local_mode == "ols" else None)
        idx_tail = list(range(m_j, N_j_prime))
        for i in idx_tail:
            z = Z_calibration[j][i]
            mu_g = mu_method["predict_global"](model_global=global_model, x_vector=z["X"], u_vector=U_calibration[j, :])
            if local_mode == "mean":
                mu_loc, is_valid = local_est_j, local_est_j is not None
            elif local_mode == "ols":
                mu_loc = _predict_local_ols(local_est_j, z["X"]) if local_est_j is not None else None
                is_valid = mu_loc is not None
            else:
                mu_loc, is_valid = None, False
            mu_tilde = _merge_mu(mu_g, mu_loc, m_j, s_size, is_valid)
            scores.append(abs(float(z["Y"]) - float(mu_tilde)))
        w_j = 1.0 / (s_size * len(idx_tail))
        weights.extend([w_j] * len(idx_tail))

    m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_within=use_within)
    local_est_test = _local_mean_y(Z_test, m_test) if local_mode == "mean" else (_local_ols_model(Z_test, m_test) if local_mode == "ols" else None)
    idx_tail_test = list(range(m_test, o_observed)) if o_observed > m_test else []
    test_scores = []
    for i in idx_tail_test:
        z = Z_test[i]
        mu_g = mu_method["predict_global"](model_global=global_model, x_vector=z["X"], u_vector=U_test[0, :])
        if local_mode == "mean":
            mu_loc, is_valid = local_est_test, local_est_test is not None
        elif local_mode == "ols":
            mu_loc = _predict_local_ols(local_est_test, z["X"]) if local_est_test is not None else None
            is_valid = mu_loc is not None
        else:
            mu_loc, is_valid = None, False
        mu_tilde = _merge_mu(mu_g, mu_loc, m_test, s_size, is_valid)
        test_scores.append(abs(float(z["Y"]) - float(mu_tilde)))

    n_tail_finite = len(test_scores)
    n_inf = max(0, N_donor - o_observed)
    n_total_test = n_tail_finite + n_inf
    if n_total_test > 0:
        w_test = 1.0 / (s_size * n_total_test)
        if n_tail_finite > 0:
            scores.extend(test_scores)
            weights.extend([w_test] * n_tail_finite)
        if n_inf > 0:
            scores.extend([np.inf] * n_inf)
            weights.extend([w_test] * n_inf)

    q = np.inf if (len(scores) == 0 or all(w <= 0 for w in weights)) else weighted_quantile(scores, weights, alpha)
    x_target = Z_test[test_index_target]["X"]
    mu_g_target = mu_method["predict_global"](model_global=global_model, x_vector=x_target, u_vector=U_test[0, :])
    if local_mode == "mean":
        mu_loc_target, is_valid_target = local_est_test, local_est_test is not None
    elif local_mode == "ols":
        mu_loc_target = _predict_local_ols(local_est_test, x_target) if local_est_test is not None else None
        is_valid_target = mu_loc_target is not None
    else:
        mu_loc_target, is_valid_target = None, False
    mu_center = _merge_mu(mu_g_target, mu_loc_target, m_test, s_size, is_valid_target)
    interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)
    return {"interval": interval, "mu_hat": float(mu_center), "number_selected_groups": int(s_size), "donor_group_index": int(donor)}


# Methods that vary with o vs baselines (constant in o)
HCP_METHODS      = [
    'donor-HCP-randomized',
    'donor-HCP-derandomized',
    'sample-HCP-randomized',
    'sample-HCP-derandomized',
]
BASELINE_METHODS = ['HCP', 'Pooling', 'Subsampling', 'Repeated']
STD_CP_METHODS   = ['Std-CP']
METHODS          = HCP_METHODS + BASELINE_METHODS + STD_CP_METHODS

ACS_PLOT_HCP_METHODS = ['donor-HCP-randomized', 'sample-HCP-randomized']
ACS_PLOT_BASELINES = ['HCP', 'Std-CP']

COLORS = {
    'donor-HCP-randomized':    '#1f77b4',
    'donor-HCP-derandomized':  '#17becf',
    'sample-HCP-randomized':   '#ff7f0e',
    'sample-HCP-derandomized': '#bcbd22',
    'HCP':         '#2ca02c',
    'Pooling':     '#d62728',
    'Subsampling': '#9467bd',
    'Repeated':    '#8c564b',
    'Std-CP':      '#111111',
}

O_COLORS = {
    0: '#2166ac',
    2: '#00b4d8',
    4: '#f77f00',
    6: '#7b2cbf',
    5: '#4393c3',
    10: '#f4a582',
    15: '#00B894',
    20: '#b2182b',
    25: '#8E44AD',
    30: '#C0392B',
    35: '#2C3E50',
    40: '#7F8C8D',
}


def _get_o_color(o_val):
    try:
        key = int(o_val)
    except Exception:
        key = o_val
    return O_COLORS.get(key, '#555555')


def _interval_from_radius(center, radius):
    return (-np.inf, np.inf) if np.isinf(radius) else (center - radius, center + radius)


def _covered(interval, y):
    return int(interval[0] <= y <= interval[1])


def _width(interval):
    lo, hi = interval
    return (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else np.nan


def _width_income_from_log1p_interval(interval):
    """Map a finite interval on log1p-income scale to width on income scale."""
    lo, hi = interval
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    with np.errstate(over='ignore', invalid='ignore'):
        w = np.expm1(hi) - np.expm1(lo)
    if not np.isfinite(w) or w < 0:
        return np.nan
    return float(w)


def _nanmean_no_warning(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


def _nanstd_no_warning(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.nanstd(arr)


def _human_number_format(y, _):
    """Format numbers with human-readable suffixes (K, M, B)."""
    ay = abs(y)
    if ay >= 1e12:
        return f'{y:.1e}'
    if ay >= 1e9:
        return f'{y / 1e9:.1f}B'
    if ay >= 1e6:
        return f'{y / 1e6:.1f}M'
    if ay >= 1e3:
        return f'{y / 1e3:.0f}K'
    if ay >= 1:
        return f'{y:.0f}'
    return f'{y:.2f}'


def _split_conformal_radius(abs_residuals, alpha):
    """Finite-sample split-conformal radius from calibration residuals."""
    scores = np.asarray(abs_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(1, k), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _compute_std_cp_interval(x_hist, y_hist, x_target, alpha, rng):
    """Standard split CP within one target group using its observed history."""
    x_hist = np.asarray(x_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)

    n_hist = len(y_hist)
    n_train = n_hist // 2
    n_cal = n_hist - n_train
    if n_train < 1 or n_cal < 1:
        return (-np.inf, np.inf)

    perm = rng.permutation(n_hist)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:]

    x_train = x_hist[train_idx]
    y_train = y_hist[train_idx]
    x_cal = x_hist[cal_idx]
    y_cal = y_hist[cal_idx]

    x_train_aug = np.column_stack([np.ones(len(x_train)), x_train])
    x_cal_aug = np.column_stack([np.ones(len(x_cal)), x_cal])

    beta, *_ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
    y_cal_pred = x_cal_aug @ beta
    radius = _split_conformal_radius(np.abs(y_cal - y_cal_pred), alpha)

    x_target = np.asarray(x_target, dtype=float).reshape(1, -1)
    x_target_aug = np.column_stack([np.ones(1), x_target])
    y_hat = float((x_target_aug @ beta)[0])
    return _interval_from_radius(y_hat, radius)


def compute_group_share_baplus(df, group_col):
    """Compute PUMA-level share of BA+ for stratification."""
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
    """
    Use stratified sampling to select non-target groups.
    
    Parameters:
    -----------
    eligible_groups : list
        All eligible group IDs
    strata : dict
        stratum_name -> list of group IDs
    selection_seed : int
        Random seed for reproducibility
    n_groups_to_select : int
        Number of groups to select with balanced allocation across strata
    
    Returns:
    --------
    selected_groups : list
        Group IDs selected for non-target (calibration) set
    """
    stratum_names = sorted(list(strata.keys()))
    n_strata = len(stratum_names)
    if n_strata == 0:
        raise ValueError("No strata available for stratified sampling")

    if n_groups_to_select > len(eligible_groups):
        raise ValueError(
            f"Cannot select {n_groups_to_select} groups from {len(eligible_groups)} eligible groups"
        )

    rng = np.random.default_rng(selection_seed)

    # Balanced allocation across strata.
    base = n_groups_to_select // n_strata
    rem = n_groups_to_select % n_strata

    capacity = {k: len(strata[k]) for k in stratum_names}
    alloc = {k: min(base, capacity[k]) for k in stratum_names}
    assigned = sum(alloc.values())

    # Distribute leftover picks by remaining capacity (largest first), tie-break random.
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

    # Optional remainder encouragement: ensure at least some strata get base+1 when possible.
    if rem > 0:
        extra_candidates = [k for k in stratum_names if alloc[k] > base]
        _ = extra_candidates  # Keep deterministic behavior without additional mutation.

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


# ---------------------------------------------------------------------------
# Single bootstrap replicate
# ---------------------------------------------------------------------------

def run_one_replicate(df, X, calib_groups, test_groups, group_col,
                      o_values, config, b):
    """
    Returns dict:
        result['baseline'][method][o] = {'coverage': float, 'width': float, 'width_income': float}
        result['hcp'][method][o]      = {'coverage': float, 'width': float, 'width_income': float}
    """
    rng       = np.random.default_rng(config['seed'] + b)
    alpha     = config['alpha']
    alpha_sel = config.get('alpha_selection', 0.5)
    n_rep     = config.get('n_repeated', 50)

    # Bootstrap each group's observations for this replicate.
    all_groups = calib_groups + test_groups
    group_data = {}
    for grp in all_groups:
        grp_idx = df.index[df[group_col] == grp].to_numpy()
        n_k = len(grp_idx)
        boot_idx = rng.choice(grp_idx, size=n_k, replace=True)
        group_data[grp] = {
            'X': X[boot_idx],
            'Y': df.loc[boot_idx, 'y'].values,
        }

    # Build Z_calibration for all calibration groups
    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    # Baseline methods: use donor-style train/calib split with o=0
    from methods.donor_hcp import get_donor_style_train_cal_split

    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = get_donor_style_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, 'tolist') else list(calib_idx)

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp      = create_mu_method_ols_global_only()

    model_baseline = mu_baseline['fit_global'](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
    )

    scores_list = []
    for j in calib_idx:
        Zj  = Z_calibration_full[j]
        yj  = np.array([z['Y'] for z in Zj])
        Uj  = U_calibration_full[j]
        muj = np.array([
            mu_baseline['predict_global'](model_baseline, z['X'], Uj)
            for z in Zj
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    T_hcp  = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub  = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep  = compute_repeated_subsampling_interval_radius(scores_list, alpha, n_rep)

    U_test = np.zeros((1, 1))

    baseline_result = {m: {} for m in BASELINE_METHODS}
    stdcp_result = {m: {} for m in STD_CP_METHODS}

    # HCP methods for each o value
    hcp_result = {m: {} for m in HCP_METHODS}

    for o in o_values:
        eligible_test_groups = [
            grp for grp in test_groups
            if len(group_data[grp]['Y']) >= (o + 1)
        ]

        if len(eligible_test_groups) == 0:
            for method in BASELINE_METHODS:
                baseline_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
            for method in HCP_METHODS:
                hcp_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
            continue

        base_covered = {m: [] for m in BASELINE_METHODS}
        base_width = {m: [] for m in BASELINE_METHODS}
        base_width_income = {m: [] for m in BASELINE_METHODS}

        hcp_covered = {m: [] for m in HCP_METHODS}
        hcp_width   = {m: [] for m in HCP_METHODS}
        hcp_width_income = {m: [] for m in HCP_METHODS}

        stdcp_covered = {m: [] for m in STD_CP_METHODS}
        stdcp_width   = {m: [] for m in STD_CP_METHODS}
        stdcp_width_income = {m: [] for m in STD_CP_METHODS}

        for grp in eligible_test_groups:
            x_target = group_data[grp]['X'][o]
            true_y   = group_data[grp]['Y'][o]

            mu_hat = mu_baseline['predict_global'](model_baseline, x_target, U_test[0])
            for method, T in [
                ('HCP', T_hcp),
                ('Pooling', T_pool),
                ('Subsampling', T_sub),
                ('Repeated', T_rep),
            ]:
                interval = _interval_from_radius(mu_hat, T)
                base_covered[method].append(_covered(interval, true_y))
                base_width[method].append(_width(interval))
                base_width_income[method].append(_width_income_from_log1p_interval(interval))

            Z_test = [
                {'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
                for i in range(o)
            ]
            Z_test.append({'X': x_target, 'Y': true_y})

            # Donor-HCP randomized (with within-group mean correction)
            try:
                dhcp_seed = config['seed'] + b * 1009 + len(test_groups) * 131 + o
                res_pp = _compute_donor_hcp_interval_with_local(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                    local_mode='mean',
                    test_index_target=o,
                    random_seed=dhcp_seed,
                )
                int_pp = res_pp['interval']
            except Exception:
                int_pp = (-np.inf, np.inf)

            # Donor-HCP derandomized
            try:
                res_dd = compute_donor_hcp_derandomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                )
                int_dd = res_dd['interval']
            except Exception:
                int_dd = (-np.inf, np.inf)

            # Sample-HCP randomized
            try:
                res_hs = compute_sample_hcp_randomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                )
                int_hs = res_hs['interval']
            except Exception:
                int_hs = (-np.inf, np.inf)

            # Sample-HCP derandomized
            try:
                res_sd = compute_sample_hcp_derandomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                )
                int_sd = res_sd['interval']
            except Exception:
                int_sd = (-np.inf, np.inf)

            for method, interval in [
                ('donor-HCP-randomized', int_pp),
                ('donor-HCP-derandomized', int_dd),
                ('sample-HCP-randomized', int_hs),
                ('sample-HCP-derandomized', int_sd),
            ]:
                hcp_covered[method].append(_covered(interval, true_y))
                hcp_width[method].append(_width(interval))
                hcp_width_income[method].append(_width_income_from_log1p_interval(interval))

            stdcp_interval = _compute_std_cp_interval(
                x_hist=group_data[grp]['X'][:o],
                y_hist=group_data[grp]['Y'][:o],
                x_target=x_target,
                alpha=alpha,
                rng=rng,
            )
            stdcp_covered['Std-CP'].append(_covered(stdcp_interval, true_y))
            stdcp_width['Std-CP'].append(_width(stdcp_interval))
            stdcp_width_income['Std-CP'].append(_width_income_from_log1p_interval(stdcp_interval))

        for method in BASELINE_METHODS:
            baseline_result[method][o] = {
                'coverage': np.mean(base_covered[method]),
                'width': _nanmean_no_warning(base_width[method]),
                'width_income': _nanmean_no_warning(base_width_income[method]),
            }

        for method in HCP_METHODS:
            hcp_result[method][o] = {
                'coverage': np.mean(hcp_covered[method]),
                'width': _nanmean_no_warning(hcp_width[method]),
                'width_income': _nanmean_no_warning(hcp_width_income[method]),
            }

        for method in STD_CP_METHODS:
            stdcp_result[method][o] = {
                'coverage': np.mean(stdcp_covered[method]),
                'width': _nanmean_no_warning(stdcp_width[method]),
                'width_income': _nanmean_no_warning(stdcp_width_income[method]),
            }

    return {'baseline': baseline_result, 'hcp': hcp_result, 'stdcp': stdcp_result}


# Shared state for multiprocessing workers.
_BOOTSTRAP_SHARED = None


def _init_bootstrap_worker(df, X, calib_groups, test_groups, group_col, o_values, config):
    global _BOOTSTRAP_SHARED
    _BOOTSTRAP_SHARED = (df, X, calib_groups, test_groups, group_col, o_values, config)


def _run_one_replicate_worker(b):
    df, X, calib_groups, test_groups, group_col, o_values, config = _BOOTSTRAP_SHARED

    eligible_groups = config.get('eligible_groups', None)
    n_groups = config.get('n_puma_groups', None)
    split_seed = config.get('group_selection_seed', 42)
    redraw_split_each_rep = bool(config.get('resample_group_split_each_rep', False))

    if redraw_split_each_rep:
        if eligible_groups is None or n_groups is None:
            raise ValueError("resample_group_split_each_rep=True requires eligible_groups and n_puma_groups in config")
        rng_split = np.random.default_rng(split_seed + b)
        calib_groups_rep = rng_split.choice(
            np.asarray(eligible_groups),
            size=int(n_groups),
            replace=False,
        ).tolist()
        calib_set_rep = set(calib_groups_rep)
        test_groups_rep = [g for g in eligible_groups if g not in calib_set_rep]
        if config.get('enforce_fixed_split_counts', False):
            expected_test = int(config.get('expected_test_pumas', len(eligible_groups) - int(n_groups)))
            if len(test_groups_rep) != expected_test:
                raise ValueError(
                    f"Expected {expected_test} target PUMAs for replicate {b+1}, "
                    f"got {len(test_groups_rep)}"
                )
    else:
        calib_groups_rep = calib_groups
        test_groups_rep = test_groups

    result = run_one_replicate(
        df, X, calib_groups_rep, test_groups_rep, group_col, o_values, config, b
    )
    return b, result


# ---------------------------------------------------------------------------
# Bootstrap loop
# ---------------------------------------------------------------------------

def run_bootstrap(df, X, calib_groups, test_groups, group_col, o_values, config):
    B   = config['B']
    n_o = len(o_values)
    n_workers = int(config.get('n_workers', 1))

    n_m  = len(METHODS)
    m_map = {m: i for i, m in enumerate(METHODS)}

    cov = np.full((B, n_m, n_o), np.nan)
    wid = np.full((B, n_m, n_o), np.nan)
    wid_income = np.full((B, n_m, n_o), np.nan)

    print(f"  Running bootstrap with n_workers={n_workers}")

    def _write_result(rep_idx, result):
        if result is None:
            return

        for method in BASELINE_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[rep_idx, m_i, o_i] = result['baseline'][method][o]['coverage']
                wid[rep_idx, m_i, o_i] = result['baseline'][method][o]['width']
                wid_income[rep_idx, m_i, o_i] = result['baseline'][method][o]['width_income']

        for method in HCP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[rep_idx, m_i, o_i] = result['hcp'][method][o]['coverage']
                wid[rep_idx, m_i, o_i] = result['hcp'][method][o]['width']
                wid_income[rep_idx, m_i, o_i] = result['hcp'][method][o]['width_income']

        for method in STD_CP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[rep_idx, m_i, o_i] = result['stdcp'][method][o]['coverage']
                wid[rep_idx, m_i, o_i] = result['stdcp'][method][o]['width']
                wid_income[rep_idx, m_i, o_i] = result['stdcp'][method][o]['width_income']

    if n_workers <= 1:
        _init_bootstrap_worker(df, X, calib_groups, test_groups, group_col, o_values, config)
        for b in range(B):
            if (b + 1) % 10 == 0:
                print(f"  Replicate {b+1}/{B}")
            rep_idx, result = _run_one_replicate_worker(b)
            _write_result(rep_idx, result)
    else:
        completed = 0
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_bootstrap_worker,
            initargs=(df, X, calib_groups, test_groups, group_col, o_values, config),
        ) as pool:
            for rep_idx, result in pool.imap_unordered(_run_one_replicate_worker, range(B)):
                completed += 1
                if completed % 10 == 0 or completed == B:
                    print(f"  Replicate {completed}/{B}")
                _write_result(rep_idx, result)

    return {
        'methods': METHODS,
        'o_values': o_values,
        'coverage': cov,
        'width': wid,
        'width_income': wid_income,
    }


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _exp_metric_by_o(df, method, o, metric_col):
    """Get metric values for a specific method and o value."""
    d = df[(df["method"] == method) & (df["o"] == o)]
    vals = d.groupby("replicate", as_index=False)[metric_col].mean()[metric_col].to_numpy()
    vals = vals[np.isfinite(vals)]
    return vals


def _exp_metric_all(df, method, metric_col):
    """Get metric values for a specific method (all o values)."""
    d = df[df["method"] == method]
    vals = d.groupby("replicate", as_index=False)[metric_col].mean()[metric_col].to_numpy()
    vals = vals[np.isfinite(vals)]
    return vals


def _boxplot_style(ax, data, pos, color, width=0.58, edge_color=None, hatch=None, alpha=0.75, linestyle="-"):
    """Create a styled boxplot."""
    if edge_color is None:
        edge_color = color

    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color=edge_color, linewidth=1.2, linestyle=linestyle),
        capprops=dict(color=edge_color, linewidth=1.2, linestyle=linestyle),
        flierprops=dict(marker=".", color=edge_color, markersize=3.5, alpha=0.35),
        boxprops=dict(
            facecolor=color,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=1.2,
            linestyle=linestyle,
        ),
    )
    if hatch:
        for patch in bp["boxes"]:
            patch.set_hatch(hatch)
    return bp


def plot_dhcp_vs_hcp_stratified(df, output_dir):
    """Plot GHCP vs HCP comparison."""
    o_vals = sorted([int(o) for o in df["o"].unique() if int(o) <= 20])

    fig, axes = plt.subplots(1, 2, figsize=(21.0, 8.9))
    fig.subplots_adjust(wspace=0.32)

    for ax, metric_col, ylabel, title in [
        (axes[0], "coverage", "Coverage", "Coverage: GHCP vs HCP"),
        (axes[1], "width_income", "Width (income units)", "Width: GHCP vs HCP"),
    ]:
        pos = 0.0
        tick_positions = []
        tick_labels = []

        for o in o_vals:
            vals_d = _exp_metric_by_o(df, "donor-HCP-randomized", o, metric_col)
            if len(vals_d):
                _boxplot_style(ax, vals_d, pos, _get_o_color(o), width=0.58)
                tick_positions.append(pos)
                tick_labels.append(f"o={o}")
            pos += 1.0

        pos += 1.1
        vals_h = _exp_metric_all(df, "HCP", metric_col)
        if len(vals_h):
            _boxplot_style(ax, vals_h, pos, color="#111111", edge_color="#111111", 
                          alpha=0.30, linestyle="--", hatch="///", width=0.58)
            tick_positions.append(pos)
            tick_labels.append("HCP")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_xlabel("Observed history size o", fontsize=22)
        ax.tick_params(axis="y", labelsize=21)
        ax.tick_params(axis="x", labelsize=21)
        
        if metric_col == "coverage":
            ax.axhline(0.8, color="black", linewidth=1.4, linestyle="--", alpha=0.60)
            ax.set_ylim(0.6, 1.05)
        else:
            ax.set_ylim(bottom=50000)
            ax.yaxis.set_major_formatter(FuncFormatter(_human_number_format))
        ax.set_title(title, fontsize=28, fontweight="bold")

    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "stratified_dhcp_vs_hcp_side_by_side.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dhcp_vs_baselines_stratified(df, output_dir):
    """Plot GHCP vs all baselines comparison."""
    o_vals = sorted([int(o) for o in df["o"].unique()])

    for metric, metric_col, ylabel in [("coverage", "coverage", "Coverage"), 
                                        ("width", "width_income", "Width (income units)")]:
        fig, ax = plt.subplots(1, 1, figsize=(18.5, 8.6))

        pos = 0.0
        tick_positions = []
        tick_labels = []

        for o in o_vals:
            vals_d = _exp_metric_by_o(df, "donor-HCP-randomized", o, metric_col)
            if len(vals_d):
                _boxplot_style(ax, vals_d, pos, _get_o_color(o), width=0.58)
                tick_positions.append(pos)
                tick_labels.append(f"o={o}")
            pos += 1.0

        pos += 1.2
        baseline_colors = {
            "HCP": "#2ca02c",
            "Pooling": "#d62728",
            "Subsampling": "#9467bd",
            "Repeated": "#8c564b",
        }
        
        for label, method_name, color in [
            ("HCP", "HCP", baseline_colors["HCP"]),
            ("Pooling", "Pooling", baseline_colors["Pooling"]),
            ("Subsampling", "Subsampling", baseline_colors["Subsampling"]),
            ("Repeated", "Repeated", baseline_colors["Repeated"]),
        ]:
            vals = _exp_metric_all(df, method_name, metric_col)
            if len(vals):
                _boxplot_style(ax, vals, pos, color=color, edge_color="#222222",
                             alpha=0.30 if label == "HCP" else 0.45,
                             linestyle="--", hatch="///" if label == "HCP" else "//", width=0.58)
                tick_positions.append(pos)
                tick_labels.append(label)
            pos += 1.55

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_xlabel("Observed history size o", fontsize=22)
        ax.tick_params(axis="y", labelsize=21)
        ax.tick_params(axis="x", labelsize=21)
        
        if metric_col == "coverage":
            ax.axhline(0.8, color="black", linewidth=1.4, linestyle="--", alpha=0.60)
            ax.set_ylim(0.6, 1.05)
        else:
            ax.set_ylim(bottom=50000)
            ax.yaxis.set_major_formatter(FuncFormatter(_human_number_format))
        ax.set_title(f"{ylabel}: GHCP vs baselines", fontsize=28, fontweight="bold")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"stratified_dhcp_vs_baselines_{metric}_all_o.pdf"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def save_outputs(results, plots_dir, results_dir, tag, target_coverage, title, plot_style='dgp'):
    """Save results and generate plots."""
    plots_dir = Path(plots_dir)
    results_dir = Path(results_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = results['methods']
    o_values = results['o_values']
    cov = results['coverage']
    wid = results['width']
    wid_income = results.get('width_income', None)

    # Save raw results
    rows = []
    for b in range(cov.shape[0]):
        for m_i, method in enumerate(methods):
            for o_i, o in enumerate(o_values):
                rows.append({
                    'replicate': b,
                    'method': method,
                    'o': o,
                    'coverage': cov[b, m_i, o_i],
                    'width': wid[b, m_i, o_i],
                    'width_income': wid_income[b, m_i, o_i] if wid_income is not None else np.nan,
                })
    df_results = pd.DataFrame(rows)
    results_csv = results_dir / f'{tag}_raw_results.csv'
    df_results.to_csv(results_csv, index=False)
    print(f"  Saved: {results_csv}")

    # Generate plots
    print(f"  Generating plots for {tag}...")
    plot_dhcp_vs_hcp_stratified(df_results, plots_dir)
    plot_dhcp_vs_baselines_stratified(df_results, plots_dir)
    

def compute_o_values(m):
    """Compute o values from group size."""
    return sorted(set([
        0,
        int(np.floor(m / 4)),
        int(np.floor(m / 2)),
        int(np.floor(3 * m / 4)),
        m - 1,
    ]))


def compute_o_values_from_test_size_distribution(test_sizes):
    """Select o values from quantiles of the target-group size distribution."""
    if len(test_sizes) == 0:
        return [0]

    arr = np.asarray(test_sizes, dtype=float)
    q_sizes = np.quantile(arr, [0.0, 0.30, 0.75], method='lower')
    derived = [max(0, int(q) - 1) for q in q_sizes]
    o_values = [0] + derived
    return sorted(set(o_values))


def load_acs_with_stratified_sampling(config):
    """
    Load ACS data and use stratified sampling for non-target groups.
    
    Instead of randomly sampling 30 PUMAs, this:
    1. Computes PUMA-level share of BA+ for each PUMA
    2. Builds 5 quantile strata on that share
    3. Samples balanced counts without replacement from each stratum
    4. Uses remaining PUMAs as target groups
    """
    target_state = config.get('acs_state', 'CA')
    n_groups = config.get('n_puma_groups', 20)
    min_group_size = config.get('min_puma_size', 5)
    min_hours = config.get('min_hours', 20)
    min_income = config.get('min_income', None)
    age_min = config.get('age_min', 25)
    age_max = config.get('age_max', 54)
    selection_seed = config.get('group_selection_seed', 42)
    yoep_window_years = config.get('yoep_window_years', 2)
    yoep_min_year = config.get('yoep_min_year', None)
    top_income_q = config.get('top_income_quantile', None)
    bottom_income_q = config.get('bottom_income_quantile', None)
    n_strata = int(config.get('n_strata', 5))

    if n_groups <= 1:
        raise ValueError("n_puma_groups must be at least 2")

    # Load and clean data
    df = load_and_clean_acs_pums(
        config['data_path'],
        states_keep=[target_state],
        age_min=age_min,
        age_max=age_max,
        yoep_window_years=yoep_window_years,
        yoep_min_year=yoep_min_year,
        min_hours=min_hours,
        min_income=min_income,
        top_income_quantile=top_income_q,
        bottom_income_quantile=bottom_income_q,
    )
    if 'puma' not in df.columns:
        raise ValueError("ACS source must contain PUMA to run PUMA-group experiments")

    df = df.dropna(subset=['puma']).copy()
    try:
        df['puma'] = df['puma'].astype(int)
    except Exception:
        df['puma'] = df['puma'].astype(str)

    counts_all = df.groupby('puma').size().sort_values(ascending=True)
    counts_eligible = counts_all[counts_all >= min_group_size]
    if len(counts_eligible) <= n_groups:
        raise ValueError(
            f"Only {len(counts_eligible)} eligible PUMAs in {target_state} "
            f"with size >= {min_group_size}, but n_puma_groups={n_groups}. "
            "Need at least one additional eligible PUMA for test groups."
        )

    df = df[df['puma'].isin(counts_eligible.index)].reset_index(drop=True)
    X = build_design_matrix_acs(df)

    eligible_groups = counts_eligible.index.to_numpy().tolist()
    config['eligible_groups'] = eligible_groups

    # ===== STRATIFIED SAMPLING (BA+ SHARE) =====
    print("  Computing group-level share of BA+ for stratification...")
    group_share_baplus = compute_group_share_baplus(df, 'puma')
    strata = _build_share_baplus_strata(
        eligible_groups=eligible_groups,
        group_share_baplus=group_share_baplus,
        n_strata=n_strata,
    )

    print(f"  Using BA+ share stratification with {len(strata)} strata...")
    calib_groups = stratified_sample_groups(
        eligible_groups=eligible_groups,
        strata=strata,
        selection_seed=selection_seed,
        n_groups_to_select=n_groups,
    )
    # ===== END STRATIFIED SAMPLING =====

    calib_set = set(calib_groups)
    test_groups = [g for g in eligible_groups if g not in calib_set]

    m = min(counts_eligible[g] for g in test_groups)
    eligible_sizes = [int(v) for v in counts_eligible.values]

    if config.get('enforce_fixed_split_counts', False):
        expected_eligible = int(config.get('expected_eligible_pumas', 66))
        expected_test = int(config.get('expected_test_pumas', expected_eligible - n_groups))
        if len(eligible_groups) != expected_eligible:
            raise ValueError(
                f"Expected {expected_eligible} eligible PUMAs after filtering, got {len(eligible_groups)}. "
                "Adjust filtering settings to recover the configured split."
            )
        if len(test_groups) != expected_test:
            raise ValueError(
                f"Expected {expected_test} target PUMAs from the split, got {len(test_groups)}. "
                "Adjust filtering settings to recover the configured split."
            )

    fixed_o_values = config.get('fixed_o_values', None)
    if fixed_o_values is None:
        o_values = compute_o_values_from_test_size_distribution(eligible_sizes)
    else:
        o_values = sorted(set(int(v) for v in fixed_o_values if int(v) >= 0))

    print(f"  Target state: {target_state}")
    if yoep_min_year is None:
        print(f"  YOEP window years: {yoep_window_years}")
    else:
        print(f"  YOEP cutoff year: >= {yoep_min_year}")
    print(f"  Age filter: [{age_min}, {age_max}]")
    print(f"  Min hours filter: {min_hours}")
    print(f"  Min income filter: {min_income}")
    print(f"  Bottom income quantile keep: {bottom_income_q}")
    print(f"  Eligible PUMAs (size >= {min_group_size}): {len(counts_eligible)}")
    print(f"  STRATIFIED non-test PUMA sample (n={n_groups}, strata={len(strata)}, seed={selection_seed})")
    for k in sorted(strata.keys()):
        print(f"    stratum {k}: {len(strata[k])} eligible PUMAs")
    print(f"  test={len(test_groups)}, non-test={len(calib_groups)}")
    print(f"  Non-test PUMAs (stratified): {sorted(calib_groups)}")
    print(f"  Test PUMAs: {sorted(test_groups)}")
    print(f"  Test group sizes: {sorted(counts_eligible[g] for g in test_groups)}")
    print(f"  Test min size: {m}")
    print(f"  o values from full eligible-size distribution: {o_values}")
    return df, X, calib_groups, test_groups, 'puma', o_values


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    mp.set_start_method('fork', force=True)
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--B_acs', type=int, default=100)
    parser.add_argument('--acs_state', type=str, default='CA')
    parser.add_argument('--acs_n_groups', type=int, default=20)
    parser.add_argument('--acs_min_group_size', type=int, default=20)
    parser.add_argument('--acs_o_values', type=str, default='0,5,10,20')
    parser.add_argument('--acs_yoep_window_years', type=int, default=2)
    parser.add_argument('--acs_min_hours', type=int, default=40)
    parser.add_argument('--acs_min_income', type=float, default=10000.0)
    parser.add_argument('--acs_bottom_income_quantile', type=float, default=None)
    parser.add_argument('--acs_min_yoep', type=int, default=2012)
    parser.add_argument('--acs_age_min', type=int, default=25)
    parser.add_argument('--acs_age_max', type=int, default=54)
    parser.add_argument('--acs_no_age_filter', action='store_true')
    parser.add_argument('--acs_group_seed', type=int, default=42)
    parser.add_argument('--n_workers', type=int, default=6)
    parser.add_argument('--acs_expected_eligible_pumas', type=int, default=61)
    parser.add_argument('--acs_expected_test_pumas', type=int, default=25)
    args = parser.parse_args()

    acs_o_values = [int(v.strip()) for v in args.acs_o_values.split(',') if v.strip() != '']
    if len(acs_o_values) == 0:
        acs_o_values = [10, 15, 20, 25]

    print("=" * 70)
    print("ACS (STRATIFIED SAMPLING) — Repeated experiment")
    print("=" * 70)
    age_min = None if args.acs_no_age_filter else args.acs_age_min
    age_max = None if args.acs_no_age_filter else args.acs_age_max
    acs_config = {
        'data_path':         str(base_dir / 'acs/data/acs_data_all50states.csv'),
        'B':                 args.B_acs,
        'seed':              456,
        'alpha':             0.2,
        'alpha_selection':   0.5,
        'acs_state':         args.acs_state.upper(),
        'n_puma_groups':     args.acs_n_groups,
        'min_puma_size':     args.acs_min_group_size,
        'age_min':           age_min,
        'age_max':           age_max,
        'min_hours':         args.acs_min_hours,
        'min_income':        args.acs_min_income,
        'yoep_min_year':     args.acs_min_yoep,
        'yoep_window_years': args.acs_yoep_window_years,
        'group_selection_seed': args.acs_group_seed,
        'top_income_quantile': None,
        'bottom_income_quantile': args.acs_bottom_income_quantile,
        'fixed_o_values':    acs_o_values,
        'n_repeated':        50,
        'enforce_fixed_split_counts': False,
        'expected_eligible_pumas': args.acs_expected_eligible_pumas,
        'expected_test_pumas': args.acs_expected_test_pumas,
        'resample_group_split_each_rep': False,  # Don't resample with stratified
        'n_workers': args.n_workers,
    }
    
    df, X, calib_grps, test_grps, gcol, o_vals = load_acs_with_stratified_sampling(acs_config)
    
    print("Running bootstrap experiments...")
    res = run_bootstrap(df, X, calib_grps, test_grps, gcol, o_vals, acs_config)
    
    save_outputs(
        res,
        plots_dir       = base_dir / 'acs/NEW_PLOTS/stratified',
        results_dir     = base_dir / 'acs/results/stratified',
        tag             = 'acs_stratified',
        target_coverage = 0.8,
        title           = f"ACS ({acs_config['acs_state']} STRATIFIED PUMA groups, alpha=0.2)",
        plot_style      = 'dgp',
    )
    
    print("Done!")
