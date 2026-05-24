#!/usr/bin/env python3
"""Compare three within-group approaches: none, mean, and local OLS.

This script compares:
1. No within-group training (m_j forced to 0)
2. Within-group mean: mu_local_j = mean(Y_1, ..., Y_{m_j})
3. Within-group OLS: mu_local_j(x) = OLS fit on (X_1, Y_1), ..., (X_{m_j}, Y_{m_j})
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from methods import create_mu_method_ols_global_only
from methods.baseline_hcp import compute_hcp_interval_radius
from methods.donor_hcp import (
    _ALPHA_SPLIT_TIEBREAK_SEED,
    _select_s_tilde_with_tie_randomization,
    get_donor_style_train_cal_split,
)
from scores import absolute_residual_score, weighted_quantile
from DGP.code.joint_xy_marginal_common import draw_group_joint_xy


FIXED_N = 21
DIMENSION = 5
U_MIN = 0.0
U_MAX = 1.0
RHO = 0.5

O_VALUES = [0, 10, 20, 30, 40, 50]
TARGET_INDEX = 50
STDCP_MIN_O = 10
TEST_N = TARGET_INDEX + 1

ALPHA = 0.2
ALPHA_SELECTION = 0.5
NUMBER_GROUPS_K = 20
NUMBER_EXPERIMENTS = 50
NUMBER_TEST_GROUPS = 100

TAG = "fixedN21_K20_d5_u01_rho05_within_comparison_mean_vs_ols"

O_COLORS = {
    0: "#2166ac",
    10: "#4393c3",
    20: "#f4a582",
    30: "#b2182b",
    40: "#2C3E50",
    50: "#8E44AD",
}


def _interval_width(interval):
    lo, hi = interval
    return (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else np.nan


def _covered(interval, y):
    return int(interval[0] <= y <= interval[1])


def _safe_width_median(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.inf
    return float(np.median(finite))


def _split_conformal_radius(abs_residuals, alpha):
    scores = np.asarray(abs_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(1, k), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _compute_std_cp_interval(x_hist, y_hist, x_target, alpha):
    x_hist = np.asarray(x_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)

    n_hist = len(y_hist)
    n_train = n_hist // 2
    n_cal = n_hist - n_train
    if n_train < 1 or n_cal < 1:
        return (-np.inf, np.inf)

    perm = np.random.permutation(n_hist)
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

    if np.isfinite(radius):
        return (y_hat - radius, y_hat + radius)
    return (-np.inf, np.inf)


def _draw_u_vector():
    return np.random.uniform(low=U_MIN, high=U_MAX, size=(DIMENSION,))


def _generate_calibration_data_fixed():
    u_cal = np.vstack([_draw_u_vector() for _ in range(NUMBER_GROUPS_K)])
    z_cal = [draw_group_joint_xy(u_cal[j], FIXED_N, rho=RHO) for j in range(NUMBER_GROUPS_K)]
    return u_cal, z_cal


def _generate_test_group_fixed():
    u_test = _draw_u_vector()
    z_test = draw_group_joint_xy(u_test, TEST_N, rho=RHO)
    return u_test.reshape(1, -1), z_test


def _m_j(o_observed, n_prime, use_within):
    if not use_within:
        return 0
    return int(min(np.floor(o_observed / 2), np.floor(n_prime / 2)))


def _local_mean_y(z_group, m_j):
    """Local mean: just average Y values."""
    if m_j <= 0:
        return None
    return float(np.mean([z_group[i]['Y'] for i in range(m_j)]))


def _local_ols_model(z_group, m_j):
    """Local OLS: fit OLS on (X, Y) pairs."""
    if m_j < 2:  # Need at least 2 points for OLS
        return None

    X_local = np.array([z_group[i]['X'] for i in range(m_j)], dtype=float)
    Y_local = np.array([z_group[i]['Y'] for i in range(m_j)], dtype=float)

    # Fit OLS: Y ~ 1 + X
    X_aug = np.column_stack([np.ones(m_j), X_local])

    try:
        beta, *_ = np.linalg.lstsq(X_aug, Y_local, rcond=None)
        return beta
    except:
        return None


def _predict_local_ols(beta, x_vec):
    """Predict using local OLS model."""
    if beta is None:
        return None
    x_aug = np.concatenate([[1.0], np.asarray(x_vec, dtype=float).ravel()])
    return float(x_aug @ beta)


def _merge_mu(mu_global, mu_local, m_j, s_size, is_valid):
    """Merge global and local predictions."""
    if m_j <= 0 or mu_local is None or not is_valid:
        return float(mu_global)
    lam = float(m_j) / float(s_size + m_j)
    return float((1.0 - lam) * mu_global + lam * mu_local)


def _compute_donor_hcp_interval(
    U_calibration,
    Z_calibration,
    U_test,
    Z_test,
    o_observed,
    alpha,
    alpha_selection,
    mu_method,
    local_mode,  # 'none', 'mean', or 'ols'
    test_index_target=None,
):
    """Compute D-HCP interval with specified local mode."""
    K = len(Z_calibration)
    N = np.array([len(Z_calibration[j]) for j in range(K)], dtype=int)
    test_idx = K
    if test_index_target is None:
        test_index_target = o_observed

    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    N_test = len(Z_test)
    if N_test < max(o_observed + 1, test_index_target + 1):
        raise ValueError('Z_test must have at least max(o+1, target+1) observations.')

    if K == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': 0,
            'donor_group_index': None,
        }

    # Determine whether to use within-group training
    use_within = (local_mode != 'none')

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=True,
    )

    if len(S_tilde) == 0:
        # Fallback: use all calibration groups for global model
        global_model = mu_method['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(range(K)),
        )

        m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_within=use_within)

        # Compute local estimator for test group
        if local_mode == 'mean':
            local_est_test = _local_mean_y(Z_test, m_test)
        elif local_mode == 'ols':
            local_est_test = _local_ols_model(Z_test, m_test)
        else:
            local_est_test = None

        s_size = 1
        scores = []

        if o_observed >= (m_test + 1):
            for i in range(m_test, o_observed):
                z = Z_test[i]
                mu_g = mu_method['predict_global'](
                    model_global=global_model,
                    x_vector=z['X'],
                    u_vector=U_test[0, :],
                )

                # Get local prediction
                if local_mode == 'mean':
                    mu_loc = local_est_test
                    is_valid = (mu_loc is not None)
                elif local_mode == 'ols':
                    mu_loc = _predict_local_ols(local_est_test, z['X']) if local_est_test is not None else None
                    is_valid = (mu_loc is not None)
                else:
                    mu_loc = None
                    is_valid = False

                mu_tilde = _merge_mu(mu_g, mu_loc, m_test, s_size, is_valid)
                scores.append(abs(float(z['Y']) - float(mu_tilde)))

            scores.append(np.inf)
            weights = np.ones(len(scores), dtype=float) / len(scores)
            q = weighted_quantile(scores, weights, alpha)
        else:
            q = np.inf

        # Predict on target
        x_target = Z_test[test_index_target]['X']
        mu_g_target = mu_method['predict_global'](
            model_global=global_model,
            x_vector=x_target,
            u_vector=U_test[0, :],
        )

        if local_mode == 'mean':
            mu_loc_target = local_est_test
            is_valid_target = (mu_loc_target is not None)
        elif local_mode == 'ols':
            mu_loc_target = _predict_local_ols(local_est_test, x_target) if local_est_test is not None else None
            is_valid_target = (mu_loc_target is not None)
        else:
            mu_loc_target = None
            is_valid_target = False

        mu_center = _merge_mu(mu_g_target, mu_loc_target, m_test, s_size, is_valid_target)
        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

        return {
            'interval': interval,
            'mu_hat': float(mu_center),
            'number_selected_groups': 1,
            'donor_group_index': None,
        }

    # Select donor group
    rng = np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    donor = int(rng.choice(S_tilde))
    N_donor = int(N[donor])

    S_cal = np.sort(np.setdiff1d(S_tilde, [donor]))
    S = np.sort(np.concatenate([S_cal, [test_idx]]))
    s_size = len(S)

    S_comp = np.setdiff1d(list(range(K + 1)), S)
    if len(S_comp) == 0:
        global_model = None
    else:
        global_model = mu_method['fit_global'](
            U_matrix=U_all,
            Z_list=Z_all,
            group_index_vector=list(S_comp),
        )

    scores = []
    weights = []

    # Calibration groups
    for j in S_cal:
        N_j_prime = int(N[j])
        m_j = _m_j(o_observed=o_observed, n_prime=N_j_prime, use_within=use_within)
        if N_j_prime <= m_j:
            continue

        # Compute local estimator for this calibration group
        if local_mode == 'mean':
            local_est_j = _local_mean_y(Z_calibration[j], m_j)
        elif local_mode == 'ols':
            local_est_j = _local_ols_model(Z_calibration[j], m_j)
        else:
            local_est_j = None

        idx_tail = list(range(m_j, N_j_prime))
        for i in idx_tail:
            z = Z_calibration[j][i]
            mu_g = mu_method['predict_global'](
                model_global=global_model,
                x_vector=z['X'],
                u_vector=U_calibration[j, :],
            )

            # Get local prediction
            if local_mode == 'mean':
                mu_loc = local_est_j
                is_valid = (mu_loc is not None)
            elif local_mode == 'ols':
                mu_loc = _predict_local_ols(local_est_j, z['X']) if local_est_j is not None else None
                is_valid = (mu_loc is not None)
            else:
                mu_loc = None
                is_valid = False

            mu_tilde = _merge_mu(mu_g, mu_loc, m_j, s_size, is_valid)
            scores.append(abs(float(z['Y']) - float(mu_tilde)))

        w_j = 1.0 / (s_size * len(idx_tail))
        weights.extend([w_j] * len(idx_tail))

    # Test group
    m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_within=use_within)

    if local_mode == 'mean':
        local_est_test = _local_mean_y(Z_test, m_test)
    elif local_mode == 'ols':
        local_est_test = _local_ols_model(Z_test, m_test)
    else:
        local_est_test = None

    idx_tail_test = list(range(m_test, o_observed)) if o_observed > m_test else []
    test_scores = []
    for i in idx_tail_test:
        z = Z_test[i]
        mu_g = mu_method['predict_global'](
            model_global=global_model,
            x_vector=z['X'],
            u_vector=U_test[0, :],
        )

        if local_mode == 'mean':
            mu_loc = local_est_test
            is_valid = (mu_loc is not None)
        elif local_mode == 'ols':
            mu_loc = _predict_local_ols(local_est_test, z['X']) if local_est_test is not None else None
            is_valid = (mu_loc is not None)
        else:
            mu_loc = None
            is_valid = False

        mu_tilde = _merge_mu(mu_g, mu_loc, m_test, s_size, is_valid)
        test_scores.append(abs(float(z['Y']) - float(mu_tilde)))

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

    if len(scores) == 0 or all(w <= 0 for w in weights):
        q = np.inf
    else:
        q = weighted_quantile(scores, weights, alpha)

    # Predict on target
    x_target = Z_test[test_index_target]['X']
    mu_g_target = mu_method['predict_global'](
        model_global=global_model,
        x_vector=x_target,
        u_vector=U_test[0, :],
    )

    if local_mode == 'mean':
        mu_loc_target = local_est_test
        is_valid_target = (mu_loc_target is not None)
    elif local_mode == 'ols':
        mu_loc_target = _predict_local_ols(local_est_test, x_target) if local_est_test is not None else None
        is_valid_target = (mu_loc_target is not None)
    else:
        mu_loc_target = None
        is_valid_target = False

    mu_center = _merge_mu(mu_g_target, mu_loc_target, m_test, s_size, is_valid_target)
    interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

    return {
        'interval': interval,
        'mu_hat': float(mu_center),
        'number_selected_groups': int(s_size),
        'donor_group_index': int(donor),
    }


def _run_one_experiment(mu_baseline, mu_hcp):
    u_cal, z_cal = _generate_calibration_data_fixed()

    train_idx, calib_idx = get_donor_style_train_cal_split(
        sample_sizes=[FIXED_N] * NUMBER_GROUPS_K,
        o_observed=0,
        alpha_selection=ALPHA_SELECTION,
    )
    model_baseline = mu_baseline['fit_global'](
        U_matrix=u_cal,
        Z_list=z_cal,
        group_index_vector=train_idx,
    )

    scores_list = []
    for j in calib_idx:
        zj = z_cal[j]
        uj = u_cal[j, :]
        yj = np.array([z['Y'] for z in zj], dtype=float)
        xj = np.array([z['X'] for z in zj], dtype=float)
        mu_j = np.array(
            [
                mu_baseline['predict_global'](
                    model_global=model_baseline,
                    x_vector=xj[i],
                    u_vector=uj,
                )
                for i in range(len(xj))
            ],
            dtype=float,
        )
        scores_list.append(absolute_residual_score(yj, mu_j))

    t_hcp = compute_hcp_interval_radius(scores_list, ALPHA)

    rows = []
    for o in O_VALUES:
        cov_none, cov_mean, cov_ols, cov_stdcp, cov_hcp = [], [], [], [], []
        wid_none, wid_mean, wid_ols, wid_stdcp, wid_hcp = [], [], [], [], []
        inf_none = inf_mean = inf_ols = inf_stdcp = inf_hcp = 0

        for _ in range(NUMBER_TEST_GROUPS):
            u_test, z_test = _generate_test_group_fixed()
            x_target = z_test[TARGET_INDEX]['X']
            y_target = z_test[TARGET_INDEX]['Y']

            # No within-group training
            res_none = _compute_donor_hcp_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=ALPHA,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp,
                local_mode='none',
                test_index_target=TARGET_INDEX,
            )
            int_none = res_none['interval']
            cov_none.append(_covered(int_none, y_target))
            w = _interval_width(int_none)
            wid_none.append(w)
            if not np.isfinite(w):
                inf_none += 1

            # Within-group mean
            res_mean = _compute_donor_hcp_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=ALPHA,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp,
                local_mode='mean',
                test_index_target=TARGET_INDEX,
            )
            int_mean = res_mean['interval']
            cov_mean.append(_covered(int_mean, y_target))
            w = _interval_width(int_mean)
            wid_mean.append(w)
            if not np.isfinite(w):
                inf_mean += 1

            # Within-group OLS
            res_ols = _compute_donor_hcp_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=ALPHA,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp,
                local_mode='ols',
                test_index_target=TARGET_INDEX,
            )
            int_ols = res_ols['interval']
            cov_ols.append(_covered(int_ols, y_target))
            w = _interval_width(int_ols)
            wid_ols.append(w)
            if not np.isfinite(w):
                inf_ols += 1

            # HCP baseline
            mu_hat_hcp = mu_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=x_target,
                u_vector=u_test[0, :],
            )
            if np.isfinite(t_hcp):
                int_hcp = (mu_hat_hcp - t_hcp, mu_hat_hcp + t_hcp)
            else:
                int_hcp = (-np.inf, np.inf)
            cov_hcp.append(_covered(int_hcp, y_target))
            w = _interval_width(int_hcp)
            wid_hcp.append(w)
            if not np.isfinite(w):
                inf_hcp += 1

            # Standard CP
            if o >= STDCP_MIN_O:
                x_hist = np.array([z_test[i]['X'] for i in range(o)], dtype=float)
                y_hist = np.array([z_test[i]['Y'] for i in range(o)], dtype=float)
                int_stdcp = _compute_std_cp_interval(
                    x_hist=x_hist,
                    y_hist=y_hist,
                    x_target=x_target,
                    alpha=ALPHA,
                )
                cov_stdcp.append(_covered(int_stdcp, y_target))
                w = _interval_width(int_stdcp)
                wid_stdcp.append(w)
                if not np.isfinite(w):
                    inf_stdcp += 1

        rows.append(
            {
                'o_observed': o,
                'coverage_donor_none': float(np.mean(cov_none)),
                'coverage_donor_mean': float(np.mean(cov_mean)),
                'coverage_donor_ols': float(np.mean(cov_ols)),
                'coverage_stdcp': float(np.mean(cov_stdcp)) if len(cov_stdcp) > 0 else np.nan,
                'coverage_hcp': float(np.mean(cov_hcp)),
                'width_donor_none': _safe_width_median(wid_none),
                'width_donor_mean': _safe_width_median(wid_mean),
                'width_donor_ols': _safe_width_median(wid_ols),
                'width_stdcp': _safe_width_median(wid_stdcp) if len(wid_stdcp) > 0 else np.nan,
                'width_hcp': _safe_width_median(wid_hcp),
                'infinite_donor_none': int(inf_none),
                'infinite_donor_mean': int(inf_mean),
                'infinite_donor_ols': int(inf_ols),
                'infinite_stdcp': int(inf_stdcp) if len(cov_stdcp) > 0 else 0,
                'infinite_hcp': int(inf_hcp),
            }
        )

    return pd.DataFrame(rows)


def _run_experiments():
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_global_only()

    all_rows = []
    for e in range(NUMBER_EXPERIMENTS):
        if (e + 1) % 5 == 0:
            print(f'  Completed {e + 1}/{NUMBER_EXPERIMENTS} experiments')
        one = _run_one_experiment(mu_baseline=mu_baseline, mu_hcp=mu_hcp)
        one.insert(0, 'experiment', e + 1)
        all_rows.append(one)

    return pd.concat(all_rows, ignore_index=True)


def _summarize(results_df):
    methods = [
        ('D-HCP (none)', 'coverage_donor_none', 'width_donor_none', 'infinite_donor_none'),
        ('D-HCP (mean)', 'coverage_donor_mean', 'width_donor_mean', 'infinite_donor_mean'),
        ('D-HCP (OLS)', 'coverage_donor_ols', 'width_donor_ols', 'infinite_donor_ols'),
        ('Std-CP', 'coverage_stdcp', 'width_stdcp', 'infinite_stdcp'),
        ('HCP', 'coverage_hcp', 'width_hcp', 'infinite_hcp'),
    ]

    out = []
    total_intervals = NUMBER_EXPERIMENTS * NUMBER_TEST_GROUPS
    for o in sorted(results_df['o_observed'].unique()):
        df_o = results_df[results_df['o_observed'] == o]
        for method_name, ccol, wcol, icol in methods:
            cov = df_o[ccol].to_numpy(dtype=float)
            wid = df_o[wcol].to_numpy(dtype=float)
            inf = df_o[icol].to_numpy(dtype=float)
            out.append(
                {
                    'o_observed': int(o),
                    'Method': method_name,
                    'Coverage_Mean': float(np.nanmean(cov)),
                    'Coverage_Std': float(np.nanstd(cov, ddof=1)),
                    'Width_Median_Finite': float(np.nanmedian(wid[np.isfinite(wid)])) if np.any(np.isfinite(wid)) else np.inf,
                    'Width_Std_Finite': float(np.nanstd(wid[np.isfinite(wid)], ddof=1)) if np.any(np.isfinite(wid)) else np.nan,
                    'Infinite_Intervals': int(np.nansum(inf)),
                    'Infinite_Percentage': float(100.0 * np.nansum(inf) / total_intervals),
                    'Target_Coverage': 1.0 - ALPHA,
                    'Coverage_Difference': float(np.nanmean(cov) - (1.0 - ALPHA)),
                }
            )
    return pd.DataFrame(out)


def main():
    np.random.seed(123)

    dir_files = ROOT / 'DGP' / 'results' / 'files' / TAG
    dir_plots = ROOT / 'DGP' / 'plots' / TAG
    dir_summ = ROOT / 'DGP' / 'results' / TAG

    dir_files.mkdir(parents=True, exist_ok=True)
    dir_plots.mkdir(parents=True, exist_ok=True)
    dir_summ.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print('WITHIN-GROUP COMPARISON: None vs Mean vs Local OLS')
    print('=' * 100)
    print(f'Fixed calibration N={FIXED_N}, test draw size={TEST_N}, K={NUMBER_GROUPS_K}, d={DIMENSION}, rho={RHO}')
    print(f'o_values={O_VALUES}, target_index={TARGET_INDEX}, alpha={ALPHA}, stdcp_min_o={STDCP_MIN_O}')

    results = _run_experiments()
    raw_csv = dir_files / 'results_within_comparison.csv'
    results.to_csv(raw_csv, index=False)
    print(f'  Saved: {raw_csv}')

    summary = _summarize(results)
    summ_csv = dir_summ / 'summary_within_comparison.csv'
    summary.to_csv(summ_csv, index=False)
    print(f'  Saved: {summ_csv}')

    print('\nDone.')


if __name__ == '__main__':
    main()
