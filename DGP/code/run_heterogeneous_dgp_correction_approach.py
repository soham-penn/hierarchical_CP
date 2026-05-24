#!/usr/bin/env python3
"""Heterogeneous DGP with correction-based local training.

Instead of fitting a separate local model, we:
1. Use global model as baseline: μ_glob(U, X)
2. Compute residuals: R_ji = Y_ji - μ_glob(U_j, X_ji)
3. Fit correction: g_j(X) to predict residuals (with regularization)
4. Final prediction: μ(U, X) = μ_glob(U, X) + g_j(X)

This approach naturally shrinks toward the global model when m_j is small.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

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


FIXED_N = 20
DIMENSION = 5  # Dimension of X (not including Y)
U_MIN = 0.0
U_MAX = 1.0
RHO = 0.5

O_VALUES = [1000]  # Focus on o=1000 only
TARGET_INDEX = 1000
STDCP_MIN_O = 10
TEST_N = TARGET_INDEX + 1

ALPHA = 0.2
ALPHA_SELECTION = 0.5
NUMBER_GROUPS_K = 20
NUMBER_EXPERIMENTS = 30  # Balanced for statistics vs runtime
NUMBER_TEST_GROUPS = 50  # Balanced for statistics vs runtime

# Regularization for correction function
CORRECTION_RIDGE_ALPHA = 1.0

TAG = "fixedN20_K20_d5_heterogeneous_dgp_correction_o1000_only"


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


def _draw_u_vector():
    """Draw group covariate U ~ Uniform[0,1]^d."""
    return np.random.uniform(low=U_MIN, high=U_MAX, size=(DIMENSION,))


def x_mean(u_vec):
    """Mean of X given U: μ_X(U) = U²."""
    u = np.asarray(u_vec, dtype=float).ravel()
    return u ** 2


def x_covariance(u_vec, rho):
    """Covariance of X given U: Σ_X(U) = (1-ρ)diag(U) + ρ·1·1^T."""
    u = np.asarray(u_vec, dtype=float).ravel()
    d = len(u)
    sigma = (1.0 - rho) * np.diag(u) + rho * np.ones((d, d), dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += 1e-8 * np.eye(d)
    return sigma


def draw_group_heterogeneous(u_vec, n_obs, rho):
    """Draw n_obs observations from the heterogeneous DGP.

    For each observation:
    1. Draw X ~ N(μ_X(U), Σ_X(U))
    2. Draw Y | X ~ N(||X||, max(0.1, 1^T X 1))
    """
    mu_x = x_mean(u_vec)
    cov_x = x_covariance(u_vec, rho)

    observations = []
    for _ in range(n_obs):
        x = np.random.multivariate_normal(mean=mu_x, cov=cov_x)
        mean_y = np.linalg.norm(x)
        var_y = max(0.1, np.sum(x))
        std_y = np.sqrt(var_y)
        y = np.random.normal(loc=mean_y, scale=std_y)
        observations.append({'X': x.astype(float), 'Y': float(y)})

    return observations


def _generate_calibration_data_fixed():
    """Generate K calibration groups, each with FIXED_N observations."""
    u_cal = np.vstack([_draw_u_vector() for _ in range(NUMBER_GROUPS_K)])
    z_cal = [draw_group_heterogeneous(u_cal[j], FIXED_N, rho=RHO) for j in range(NUMBER_GROUPS_K)]
    return u_cal, z_cal


def _generate_test_group_fixed():
    """Generate one test group with TEST_N observations."""
    u_test = _draw_u_vector()
    z_test = draw_group_heterogeneous(u_test, TEST_N, rho=RHO)
    return u_test.reshape(1, -1), z_test


def _m_j(o_observed, n_prime, use_correction):
    """Number of within-group training observations."""
    if not use_correction:
        return 0
    return int(min(np.floor(o_observed / 2), np.floor(n_prime / 2)))


def _fit_correction_constant(residuals):
    """Fit constant correction: g(x) = c.

    This is just the mean of residuals.
    """
    if len(residuals) == 0:
        return None
    return {'type': 'constant', 'value': float(np.mean(residuals))}


def _fit_correction_linear_ridge(X_train, residuals, ridge_alpha):
    """Fit linear correction: g(x) = β₀ + β^T x with Ridge regularization.

    We're fitting residuals ~ 1 + X.
    """
    if len(residuals) < 2:
        return None

    X_train = np.asarray(X_train, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    # Add intercept
    X_aug = np.column_stack([np.ones(len(X_train)), X_train])

    try:
        # Ridge: beta = (X^T X + alpha*I)^{-1} X^T y
        # Don't penalize intercept
        p = X_aug.shape[1]
        penalty_matrix = ridge_alpha * np.eye(p)
        penalty_matrix[0, 0] = 0  # Don't penalize intercept

        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ residuals
        beta = np.linalg.solve(XtX + penalty_matrix, Xty)

        return {'type': 'linear_ridge', 'beta': beta}
    except:
        return None


def _fit_correction_spline(X_train, residuals, smoothing_factor=None, n_backfit_iter=3):
    """Fit additive spline correction: g(x) = intercept + sum_k g_k(x_k).

    Uses backfitting algorithm to fit additive model properly.
    Each g_k is fit to residuals after accounting for other components.
    """
    if len(residuals) < 5:  # Need at least 5 points for splines
        return None

    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    residuals = np.asarray(residuals, dtype=float)
    n_samples, n_features = X_train.shape

    intercept = np.mean(residuals)

    # Adaptive smoothing
    if smoothing_factor is None:
        # More aggressive smoothing for small samples
        res_var = np.var(residuals) if len(residuals) > 1 else 1.0
        smoothing_factor = res_var * n_samples * 0.5  # Increased smoothing

    try:
        # Initialize spline functions
        splines = [None] * n_features
        spline_preds = np.zeros((n_samples, n_features))

        # Backfitting iterations
        for iteration in range(n_backfit_iter):
            for k in range(n_features):
                x_k = X_train[:, k]

                # Check for variation
                if np.std(x_k) < 1e-10:
                    continue

                # Compute partial residuals: remove other components
                partial_resid = residuals - intercept
                for j in range(n_features):
                    if j != k:
                        partial_resid = partial_resid - spline_preds[:, j]

                # Fit spline to partial residuals
                sort_idx = np.argsort(x_k)
                x_k_sorted = x_k[sort_idx]
                r_sorted = partial_resid[sort_idx]

                # Fit with heavy smoothing
                spline = UnivariateSpline(x_k_sorted, r_sorted, k=min(3, len(x_k_sorted)-1), s=smoothing_factor)
                splines[k] = spline

                # Update predictions
                spline_preds[:, k] = np.array([float(spline(x_k[i])) for i in range(n_samples)])

        return {
            'type': 'additive_spline',
            'splines': splines,
            'intercept': float(intercept),
            'n_features': n_features
        }
    except Exception as e:
        # Fallback to constant
        return None


def _predict_correction(correction_model, x_vec):
    """Predict using fitted correction model."""
    if correction_model is None:
        return 0.0

    if correction_model['type'] == 'constant':
        return correction_model['value']
    elif correction_model['type'] == 'linear_ridge':
        x_aug = np.concatenate([[1.0], np.asarray(x_vec, dtype=float).ravel()])
        return float(x_aug @ correction_model['beta'])
    elif correction_model['type'] == 'additive_spline':
        x = np.asarray(x_vec, dtype=float).ravel()
        pred = correction_model['intercept']
        for k, spline in enumerate(correction_model['splines']):
            if spline is not None and k < len(x):
                pred += float(spline(x[k]))
        return pred
    else:
        return 0.0


def _compute_donor_hcp_interval(
    U_calibration,
    Z_calibration,
    U_test,
    Z_test,
    o_observed,
    alpha,
    alpha_selection,
    mu_method,
    correction_mode,  # 'none', 'constant', 'linear_ridge', or 'spline'
    test_index_target=None,
):
    """Compute D-HCP interval with correction-based local training."""
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
        return {'interval': (-np.inf, np.inf), 'mu_hat': 0.0, 'number_selected_groups': 0}

    use_correction = (correction_mode != 'none')

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=True,
    )

    if len(S_tilde) == 0:
        # Fallback: use all calibration groups
        global_model = mu_method['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(range(K)),
        )

        m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_correction=use_correction)

        # Fit correction for test group
        if m_test > 0 and use_correction:
            # Compute residuals from global model
            residuals = []
            X_train = []
            for i in range(m_test):
                z = Z_test[i]
                mu_glob = mu_method['predict_global'](
                    model_global=global_model,
                    x_vector=z['X'],
                    u_vector=U_test[0, :],
                )
                residuals.append(z['Y'] - mu_glob)
                X_train.append(z['X'])

            if correction_mode == 'constant':
                correction_test = _fit_correction_constant(residuals)
            elif correction_mode == 'linear_ridge':
                correction_test = _fit_correction_linear_ridge(X_train, residuals, CORRECTION_RIDGE_ALPHA)
            elif correction_mode == 'spline':
                correction_test = _fit_correction_spline(X_train, residuals)
            else:
                correction_test = None
        else:
            correction_test = None

        s_size = 1
        scores = []

        if o_observed >= (m_test + 1):
            for i in range(m_test, o_observed):
                z = Z_test[i]
                mu_glob = mu_method['predict_global'](
                    model_global=global_model,
                    x_vector=z['X'],
                    u_vector=U_test[0, :],
                )
                correction = _predict_correction(correction_test, z['X'])
                mu_tilde = mu_glob + correction
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
        correction_target = _predict_correction(correction_test, x_target)
        mu_center = mu_g_target + correction_target
        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

        return {'interval': interval, 'mu_hat': float(mu_center), 'number_selected_groups': 1}

    # Select donor
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
        m_j = _m_j(o_observed=o_observed, n_prime=N_j_prime, use_correction=use_correction)
        if N_j_prime <= m_j:
            continue

        # Fit correction for this calibration group
        if m_j > 0 and use_correction:
            residuals = []
            X_train = []
            for i in range(m_j):
                z = Z_calibration[j][i]
                mu_glob = mu_method['predict_global'](
                    model_global=global_model,
                    x_vector=z['X'],
                    u_vector=U_calibration[j, :],
                )
                residuals.append(z['Y'] - mu_glob)
                X_train.append(z['X'])

            if correction_mode == 'constant':
                correction_j = _fit_correction_constant(residuals)
            elif correction_mode == 'linear_ridge':
                correction_j = _fit_correction_linear_ridge(X_train, residuals, CORRECTION_RIDGE_ALPHA)
            elif correction_mode == 'spline':
                correction_j = _fit_correction_spline(X_train, residuals)
            else:
                correction_j = None
        else:
            correction_j = None

        idx_tail = list(range(m_j, N_j_prime))
        for i in idx_tail:
            z = Z_calibration[j][i]
            mu_glob = mu_method['predict_global'](
                model_global=global_model,
                x_vector=z['X'],
                u_vector=U_calibration[j, :],
            )
            correction = _predict_correction(correction_j, z['X'])
            mu_tilde = mu_glob + correction
            scores.append(abs(float(z['Y']) - float(mu_tilde)))

        w_j = 1.0 / (s_size * len(idx_tail))
        weights.extend([w_j] * len(idx_tail))

    # Test group
    m_test = _m_j(o_observed=o_observed, n_prime=o_observed, use_correction=use_correction)

    if m_test > 0 and use_correction:
        residuals = []
        X_train = []
        for i in range(m_test):
            z = Z_test[i]
            mu_glob = mu_method['predict_global'](
                model_global=global_model,
                x_vector=z['X'],
                u_vector=U_test[0, :],
            )
            residuals.append(z['Y'] - mu_glob)
            X_train.append(z['X'])

        if correction_mode == 'constant':
            correction_test = _fit_correction_constant(residuals)
        elif correction_mode == 'linear_ridge':
            correction_test = _fit_correction_linear_ridge(X_train, residuals, CORRECTION_RIDGE_ALPHA)
        elif correction_mode == 'spline':
            correction_test = _fit_correction_spline(X_train, residuals)
        else:
            correction_test = None
    else:
        correction_test = None

    idx_tail_test = list(range(m_test, o_observed)) if o_observed > m_test else []
    test_scores = []
    for i in idx_tail_test:
        z = Z_test[i]
        mu_glob = mu_method['predict_global'](
            model_global=global_model,
            x_vector=z['X'],
            u_vector=U_test[0, :],
        )
        correction = _predict_correction(correction_test, z['X'])
        mu_tilde = mu_glob + correction
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
    correction_target = _predict_correction(correction_test, x_target)
    mu_center = mu_g_target + correction_target
    interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

    return {'interval': interval, 'mu_hat': float(mu_center), 'number_selected_groups': int(s_size)}


def _compute_std_cp_interval(x_hist, y_hist, x_target, alpha):
    """Standard split conformal using only test-group history."""
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


def _run_one_experiment(mu_baseline, mu_hcp):
    """Run one experiment comparing different correction approaches."""
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
        cov_none, cov_const, cov_linear, cov_spline, cov_stdcp, cov_hcp = [], [], [], [], [], []
        wid_none, wid_const, wid_linear, wid_spline, wid_stdcp, wid_hcp = [], [], [], [], [], []
        inf_none = inf_const = inf_linear = inf_spline = inf_stdcp = inf_hcp = 0

        for _ in range(NUMBER_TEST_GROUPS):
            u_test, z_test = _generate_test_group_fixed()
            x_target = z_test[TARGET_INDEX]['X']
            y_target = z_test[TARGET_INDEX]['Y']

            # No correction
            res_none = _compute_donor_hcp_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=ALPHA,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp,
                correction_mode='none',
                test_index_target=TARGET_INDEX,
            )
            int_none = res_none['interval']
            cov_none.append(_covered(int_none, y_target))
            w = _interval_width(int_none)
            wid_none.append(w)
            if not np.isfinite(w):
                inf_none += 1

            # Constant correction
            res_const = _compute_donor_hcp_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=ALPHA,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp,
                correction_mode='constant',
                test_index_target=TARGET_INDEX,
            )
            int_const = res_const['interval']
            cov_const.append(_covered(int_const, y_target))
            w = _interval_width(int_const)
            wid_const.append(w)
            if not np.isfinite(w):
                inf_const += 1

            # Linear Ridge correction
            res_linear = _compute_donor_hcp_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=ALPHA,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp,
                correction_mode='linear_ridge',
                test_index_target=TARGET_INDEX,
            )
            int_linear = res_linear['interval']
            cov_linear.append(_covered(int_linear, y_target))
            w = _interval_width(int_linear)
            wid_linear.append(w)
            if not np.isfinite(w):
                inf_linear += 1

            # Spline correction
            res_spline = _compute_donor_hcp_interval(
                U_calibration=u_cal,
                Z_calibration=z_cal,
                U_test=u_test,
                Z_test=z_test,
                o_observed=o,
                alpha=ALPHA,
                alpha_selection=ALPHA_SELECTION,
                mu_method=mu_hcp,
                correction_mode='spline',
                test_index_target=TARGET_INDEX,
            )
            int_spline = res_spline['interval']
            cov_spline.append(_covered(int_spline, y_target))
            w = _interval_width(int_spline)
            wid_spline.append(w)
            if not np.isfinite(w):
                inf_spline += 1

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
                'coverage_none': float(np.mean(cov_none)),
                'coverage_constant': float(np.mean(cov_const)),
                'coverage_linear': float(np.mean(cov_linear)),
                'coverage_spline': float(np.mean(cov_spline)),
                'coverage_stdcp': float(np.mean(cov_stdcp)) if len(cov_stdcp) > 0 else np.nan,
                'coverage_hcp': float(np.mean(cov_hcp)),
                'width_none': _safe_width_median(wid_none),
                'width_constant': _safe_width_median(wid_const),
                'width_linear': _safe_width_median(wid_linear),
                'width_spline': _safe_width_median(wid_spline),
                'width_stdcp': _safe_width_median(wid_stdcp) if len(wid_stdcp) > 0 else np.nan,
                'width_hcp': _safe_width_median(wid_hcp),
                'infinite_none': int(inf_none),
                'infinite_constant': int(inf_const),
                'infinite_linear': int(inf_linear),
                'infinite_spline': int(inf_spline),
                'infinite_stdcp': int(inf_stdcp) if len(cov_stdcp) > 0 else 0,
                'infinite_hcp': int(inf_hcp),
            }
        )

    return pd.DataFrame(rows)


def _run_experiments():
    """Run all experiments."""
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_global_only()

    all_rows = []
    for e in range(NUMBER_EXPERIMENTS):
        if (e + 1) % 10 == 0:
            print(f'  Completed {e + 1}/{NUMBER_EXPERIMENTS} experiments')
        one = _run_one_experiment(mu_baseline=mu_baseline, mu_hcp=mu_hcp)
        one.insert(0, 'experiment', e + 1)
        all_rows.append(one)

    return pd.concat(all_rows, ignore_index=True)


def _summarize(results_df):
    """Summarize results across experiments."""
    methods = [
        ('D-HCP (none)', 'coverage_none', 'width_none', 'infinite_none'),
        ('D-HCP (constant corr)', 'coverage_constant', 'width_constant', 'infinite_constant'),
        ('D-HCP (linear corr)', 'coverage_linear', 'width_linear', 'infinite_linear'),
        ('D-HCP (spline corr)', 'coverage_spline', 'width_spline', 'infinite_spline'),
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
    print('HETEROGENEOUS DGP: Correction-Based Local Training')
    print('=' * 100)
    print(f'Fixed calibration N={FIXED_N}, test draw size={TEST_N}, K={NUMBER_GROUPS_K}, d={DIMENSION}, rho={RHO}')
    print(f'o_values={O_VALUES}, target_index={TARGET_INDEX}, alpha={ALPHA}, stdcp_min_o={STDCP_MIN_O}')
    print('\nDGP:')
    print('  - X ~ N(U², (1-ρ)diag(U) + ρ·1·1^T)')
    print('  - Y|X ~ N(||X||, max(0.1, sum(X)))')
    print('\nCorrection approach:')
    print('  μ(U, X) = μ_glob(U, X) + g(X)')
    print('  where g fits residuals: R = Y - μ_glob(U, X)')
    print(f'  Regularization: Ridge alpha={CORRECTION_RIDGE_ALPHA}\n')

    results = _run_experiments()
    raw_csv = dir_files / 'results_correction.csv'
    results.to_csv(raw_csv, index=False)
    print(f'\n  Saved: {raw_csv}')

    summary = _summarize(results)
    summ_csv = dir_summ / 'summary_correction.csv'
    summary.to_csv(summ_csv, index=False)
    print(f'  Saved: {summ_csv}')

    print('\nDone.')


if __name__ == '__main__':
    main()
