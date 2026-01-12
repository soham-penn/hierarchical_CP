"""
HCP++ (HCP.plus) Interval Method

This module implements the HCP++ method with generic μ-method support.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scores import weighted_quantile


def compute_hcp_plus_interval(U_calibration, Z_calibration, U_test, Z_test,
                              o_observed, alpha, alpha_selection, mu_method):
    """
    Compute HCP++ prediction interval.

    Parameters:
    -----------
    U_calibration : ndarray of shape (K, d)
        Group-level covariates for calibration groups
    Z_calibration : list of lists
        Z_calibration[j] contains observations for calibration group j
    U_test : ndarray of shape (1, d)
        Group-level covariate for test group
    Z_test : list
        Observations for test group
    o_observed : int
        Number of observed points in test group
    alpha : float
        Miscoverage level
    alpha_selection : float
        Selection level for donor groups
    mu_method : dict
        μ-estimation method object

    Returns:
    --------
    dict with keys:
        - 'interval': tuple (lower, upper)
        - 'number_selected_groups': int
        - 'donor_group_index': int or None
    """
    K = len(Z_calibration)
    N = np.array([len(Z_calibration[j]) for j in range(K)])
    test_idx = K  # 0-indexed

    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    if K == 0:
        # No calibration groups available - use standard CP on test group only
        # Split test group: first half for training, second half for calibration
        N_test = len(Z_test)
        if N_test < (o_observed + 1):
            return {
                'interval': (-np.inf, np.inf),
                'mu_hat': 0.0,
                'number_selected_groups': 0,
                'donor_group_index': None
            }

        tau = int(np.floor(o_observed / 2))
        if tau < 0:
            tau = 0
        if o_observed > 0 and tau >= o_observed:
            tau = o_observed - 1

        # Fit model on test group training data (if available)
        if tau > 0:
            train_idx = list(range(tau))
            # Create dummy U matrix for test group
            U_dummy = U_test
            Z_dummy = [Z_test]
            global_model = mu_method['fit_global'](
                U_matrix=U_dummy,
                Z_list=Z_dummy,
                group_index_vector=[0]
            )
            offset_test = mu_method['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_test[0, :],
                Z_group_list=Z_test,
                training_index_vector=train_idx
            )
        else:
            global_model = None
            offset_test = 0.0

        # Compute scores on calibration portion
        if o_observed >= (tau + 1):
            cal_idx = list(range(tau, o_observed))
            scores = []
            for i in cal_idx:
                z = Z_test[i]
                if global_model is not None:
                    mu = mu_method['predict_group_mu'](
                        model_global=global_model,
                        group_adjustment=offset_test,
                        x_vector=z['X'],
                        u_group_vector=U_test[0, :]
                    )
                else:
                    mu = 0.0
                scores.append(np.abs(z['Y'] - mu))
            weights = np.ones(len(scores)) / len(scores)
            q = weighted_quantile(scores, weights, alpha)
        else:
            q = np.inf

        # Prediction
        X_target = Z_test[o_observed]['X']
        if global_model is not None:
            mu_global_target = mu_method['predict_global'](
                model_global=global_model,
                x_vector=X_target,
                u_vector=U_test[0, :]
            )
            mu_center = mu_global_target + offset_test
        else:
            mu_center = 0.0

        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

        return {
            'interval': interval,
            'mu_hat': mu_center,
            'number_selected_groups': 0,
            'donor_group_index': None
        }

    N_test = len(Z_test)
    if N_test < (o_observed + 1):
        raise ValueError("compute_hcp_plus_interval: Z_test must have at least o+1 observations.")

    # Compute empirical CDF and selection threshold
    def Fhat_N(t):
        return np.mean(N <= t)

    p = min(1.0, Fhat_N(o_observed) + (1 - alpha_selection))

    N_sorted = np.sort(N)
    idx_V = max(0, int(np.ceil(p * K)) - 1)  # -1 for 0-indexing
    V_o = N_sorted[idx_V]

    # Select donor groups
    S_tilde = np.where((N > o_observed) & (N <= V_o))[0]
    if len(S_tilde) == 0:
        S_tilde = np.where(N > o_observed)[0]

    # Special case: no donor groups; use only test group
    if len(S_tilde) == 0:
        global_model = mu_method['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(range(K))
        )

        # τ(o) = floor(o/2), with clamping
        tau = int(np.floor(o_observed / 2))
        if tau < 0:
            tau = 0
        if o_observed > 0 and tau >= o_observed:
            tau = o_observed - 1

        if tau > 0:
            train_idx = list(range(tau))
            offset_test = mu_method['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_test[0, :],
                Z_group_list=Z_test,
                training_index_vector=train_idx
            )
        else:
            offset_test = 0.0

        if o_observed >= (tau + 1):
            cal_idx = list(range(tau, o_observed))
            scores = []
            for i in cal_idx:
                z = Z_test[i]
                mu = mu_method['predict_group_mu'](
                    model_global=global_model,
                    group_adjustment=offset_test,
                    x_vector=z['X'],
                    u_group_vector=U_test[0, :]
                )
                scores.append(np.abs(z['Y'] - mu))
            weights = np.ones(len(scores)) / len(scores)
            q = weighted_quantile(scores, weights, alpha)
        else:
            q = np.inf

        X_target = Z_test[o_observed]['X']
        mu_global_target = mu_method['predict_global'](
            model_global=global_model,
            x_vector=X_target,
            u_vector=U_test[0, :]
        )
        mu_center = mu_global_target + offset_test
        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

        return {
            'interval': interval,
            'number_selected_groups': 1,
            'donor_group_index': None
        }

    # Normal HCP++ path
    donor = np.random.choice(S_tilde)
    N_donor = N[donor]

    S_cal = np.sort(np.setdiff1d(S_tilde, [donor]))
    S = np.sort(np.concatenate([S_cal, [test_idx]]))
    S_size = len(S)

    # τ(o): ALWAYS floor(o/2), with clamping
    tau = int(np.floor(o_observed / 2))
    if tau < 0:
        tau = 0
    if o_observed > 0 and tau >= o_observed:
        tau = o_observed - 1

    # Fit global model on complement of S
    S_comp = np.setdiff1d(list(range(K + 1)), S)
    if len(S_comp) == 0:
        global_model = None
    else:
        global_model = mu_method['fit_global'](
            U_matrix=U_all,
            Z_list=Z_all,
            group_index_vector=list(S_comp)
        )

    scores = []
    weights = []

    # Calibration groups
    for j in S_cal:
        N_j = N[j]
        # Calibration uses observations from index tau onwards
        if N_j <= tau:
            continue

        if tau > 0:
            train_idx = list(range(tau))
            offset_j = mu_method['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_calibration[j, :],
                Z_group_list=Z_calibration[j],
                training_index_vector=train_idx
            )
        else:
            offset_j = 0.0

        idx_tail = list(range(tau, N_j))
        for i in idx_tail:
            z = Z_calibration[j][i]
            mu = mu_method['predict_group_mu'](
                model_global=global_model,
                group_adjustment=offset_j,
                x_vector=z['X'],
                u_group_vector=U_calibration[j, :]
            )
            scores.append(np.abs(z['Y'] - mu))

        if len(idx_tail) > 0:
            w_j = 1.0 / (S_size * len(idx_tail))
            weights.extend([w_j] * len(idx_tail))

    # Test group
    if tau > 0:
        train_idx_test = list(range(tau))
        offset_test = mu_method['fit_group_adjustment'](
            model_global=global_model,
            u_group_vector=U_test[0, :],
            Z_group_list=Z_test,
            training_index_vector=train_idx_test
        )
    else:
        offset_test = 0.0

    # Calibration uses observations from index tau to o_observed-1
    idx_tail_test = list(range(tau, o_observed)) if o_observed > tau else []

    test_scores = []
    for i in idx_tail_test:
        z = Z_test[i]
        mu = mu_method['predict_group_mu'](
            model_global=global_model,
            group_adjustment=offset_test,
            x_vector=z['X'],
            u_group_vector=U_test[0, :]
        )
        test_scores.append(np.abs(z['Y'] - mu))

    n_tail_finite = len(test_scores)
    n_inf = max(0, N_donor - o_observed)
    n_total_test = n_tail_finite + n_inf

    if n_total_test > 0:
        w_test = 1.0 / (S_size * n_total_test)
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

    X_target = Z_test[o_observed]['X']
    mu_global_target = mu_method['predict_global'](
        model_global=global_model,
        x_vector=X_target,
        u_vector=U_test[0, :]
    )
    mu_center = mu_global_target + offset_test
    interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

    return {
        'interval': interval,
        'mu_hat': mu_center,
        'number_selected_groups': S_size,
        'donor_group_index': int(donor)
    }
