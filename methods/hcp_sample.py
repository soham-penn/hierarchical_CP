"""
HCP.sample Interval Method

This module implements the HCP.sample method which uses sample-splitting
within each group for calibration.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scores import weighted_quantile


def compute_hcp_sample_interval(U_calibration, Z_calibration, U_test, Z_test,
                                o_observed, alpha, test_index_target,
                                alpha_selection, mu_method):
    """
    Compute HCP.sample prediction interval.

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
    test_index_target : int
        Index of target observation in test group (0-indexed)
    alpha_selection : float
        Selection level for donor groups
    mu_method : dict
        μ-estimation method object

    Returns:
    --------
    dict with keys:
        - 'interval': tuple (lower, upper)
        - 'number_selected_groups': int
    """
    K = len(Z_calibration)
    N = np.array([len(Z_calibration[j]) for j in range(K)])
    test_idx = K  # 0-indexed

    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    if K == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': 0
        }

    N_test = len(Z_test)
    if N_test < (o_observed + 1):
        raise ValueError("compute_hcp_sample_interval: Z_test must have at least o+1 observations.")

    # Compute empirical CDF and selection threshold
    def Fhat_N(t):
        return np.mean(N <= t)

    p = min(1.0, Fhat_N(o_observed) + (1 - alpha_selection))

    N_sorted = np.sort(N)
    idx_V = max(0, int(np.ceil(p * K)) - 1)  # -1 for 0-indexing
    V_o = N_sorted[idx_V]

    # Select groups
    S_tilde = np.where((N > o_observed) & (N <= V_o))[0]
    if len(S_tilde) == 0:
        S_tilde = np.where(N > o_observed)[0]

    # Special case: no donor groups available
    if len(S_tilde) == 0:
        # Fall back to HCP++ logic
        from .hcp_plus import compute_hcp_plus_interval
        res_pp = compute_hcp_plus_interval(
            U_calibration=U_calibration,
            Z_calibration=Z_calibration,
            U_test=U_test,
            Z_test=Z_test,
            o_observed=o_observed,
            alpha=alpha,
            alpha_selection=alpha_selection,
            mu_method=mu_method
        )
        return {
            'interval': res_pp['interval'],
            'mu_hat': res_pp.get('mu_hat', 0.0),
            'number_selected_groups': 1
        }

    S = np.sort(np.concatenate([S_tilde, [test_idx]]))
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

    n_slots = o_observed + 1 - tau
    w_slot = 1.0 / (S_size * n_slots)

    values = []
    weights = []
    offset_test = 0.0

    for j in S:
        if j < K:
            # Calibration group
            Uj = U_calibration[j, :]
            Zj = Z_calibration[j]
            Nj = N[j]
            if Nj < (o_observed + 1):
                continue

            # Sample o_observed + 1 observations without replacement
            Tj = np.random.choice(Nj, size=o_observed + 1, replace=False)

            if tau > 0:
                Tj_train = Tj[:tau]
                Tj_cal = Tj[tau:(o_observed + 1)]
                offset_j = mu_method['fit_group_adjustment'](
                    model_global=global_model,
                    u_group_vector=Uj,
                    Z_group_list=Zj,
                    training_index_vector=list(Tj_train)
                )
            else:
                Tj_train = []
                Tj_cal = Tj[0:(o_observed + 1)]
                offset_j = 0.0

            for i_idx in Tj_cal:
                z = Zj[i_idx]
                mu = mu_method['predict_group_mu'](
                    model_global=global_model,
                    group_adjustment=offset_j,
                    x_vector=z['X'],
                    u_group_vector=Uj
                )
                s = np.abs(z['Y'] - mu)
                values.append(s)
                weights.append(w_slot)

        else:
            # Test group
            Uj = U_test[0, :]
            Zj = Z_test
            Nj = len(Zj)
            if Nj < o_observed:
                raise ValueError("compute_hcp_sample_interval: test group must have at least o observed points.")

            if tau > 0:
                # Sample tau training indices from the o observed points
                Tj_train = np.random.choice(o_observed, size=tau, replace=False)
                Tj_cal = np.setdiff1d(list(range(o_observed)), Tj_train)
                offset_test = mu_method['fit_group_adjustment'](
                    model_global=global_model,
                    u_group_vector=Uj,
                    Z_group_list=Zj,
                    training_index_vector=list(Tj_train)
                )
            else:
                Tj_train = []
                Tj_cal = list(range(o_observed))
                offset_test = 0.0

            for i_idx in Tj_cal:
                z = Zj[i_idx]
                mu = mu_method['predict_group_mu'](
                    model_global=global_model,
                    group_adjustment=offset_test,
                    x_vector=z['X'],
                    u_group_vector=Uj
                )
                s = np.abs(z['Y'] - mu)
                values.append(s)
                weights.append(w_slot)

            # Add infinity
            values.append(np.inf)
            weights.append(w_slot)

    if len(values) == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': S_size
        }

    q = weighted_quantile(values, weights, alpha)

    X_target = Z_test[test_index_target]['X']
    mu_global = mu_method['predict_global'](
        model_global=global_model,
        x_vector=X_target,
        u_vector=U_test[0, :]
    )
    mu_center = mu_global + offset_test

    if np.isinf(q):
        interval = (-np.inf, np.inf)
    else:
        interval = (mu_center - q, mu_center + q)

    return {
        'interval': interval,
        'mu_hat': mu_center,
        'number_selected_groups': S_size
    }
