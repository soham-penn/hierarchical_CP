"""
Sample-HCP methods.

This module exposes two separate variants:
- randomized sample-HCP (legacy HCP.sample)
- derandomized sample-HCP (closed-form averaged subsampling measure)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scores import conformal_threshold, merge_quantile_info


_ALPHA_SPLIT_TIEBREAK_SEED = 123


def _select_conformal_q(scores, weights, alpha, quantile_mode="deterministic",
                        quantile_random_seed=None, quantile_rng=None,
                        return_quantile_info=False):
    return conformal_threshold(
        scores=scores,
        weights=weights,
        alpha=alpha,
        quantile_mode=quantile_mode,
        random_seed=quantile_random_seed,
        rng=quantile_rng,
        return_info=return_quantile_info,
    )


def _select_s_tilde_with_tie_randomization(N, o_observed, alpha_selection):
    """
    Select S_tilde with tie-aware randomization at the alpha cutoff.

    If many groups share the boundary sample size, randomize only among those
    boundary groups so that |S_tilde| matches the target count implied by the
    alpha_1 proportion. Uses a fixed seed to keep this split reproducible.
    """
    N = np.asarray(N, dtype=int)
    K = len(N)
    if K == 0:
        return np.array([], dtype=int)

    def Fhat_N(t):
        return np.mean(N <= t)

    p = min(1.0, Fhat_N(o_observed) + (1 - alpha_selection))
    N_sorted = np.sort(N)
    idx_V = max(0, int(np.ceil(p * K)) - 1)
    V_o = N_sorted[idx_V]

    eligible = np.where(N > o_observed)[0]
    if len(eligible) == 0:
        return eligible

    interior = eligible[N[eligible] < V_o]
    boundary = eligible[N[eligible] == V_o]

    n_target = int(np.ceil(p * K)) - int(np.sum(N <= o_observed))
    n_target = max(1, min(n_target, len(eligible)))

    if len(interior) >= n_target:
        order = np.argsort(N[interior], kind='stable')
        return np.sort(interior[order[:n_target]])

    n_boundary_needed = n_target - len(interior)
    if n_boundary_needed >= len(boundary):
        return np.sort(np.concatenate([interior, boundary]))

    rng = np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    keep_boundary = rng.choice(boundary, size=n_boundary_needed, replace=False)
    return np.sort(np.concatenate([interior, keep_boundary]))


def _compute_sample_hcp_randomized_interval_impl(U_calibration, Z_calibration, U_test, Z_test,
                                                 o_observed, alpha, alpha_selection, mu_method,
                                                 test_index_target=None,
                                                 tau_override=None,
                                                 quantile_mode="deterministic",
                                                 quantile_random_seed=None,
                                                 quantile_rng=None,
                                                 return_quantile_info=False):
    """Randomized sample-HCP interval implementation (formerly in hcp_sample.py)."""
    K = len(Z_calibration)
    N = np.array([len(Z_calibration[j]) for j in range(K)])
    test_idx = K

    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    if K == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': 0,
        }

    if test_index_target is None:
        test_index_target = o_observed

    N_test = len(Z_test)
    if N_test < (o_observed + 1):
        raise ValueError("compute_sample_hcp_randomized_interval: Z_test must have at least o+1 observations.")

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
    )

    if len(S_tilde) == 0:
        from .donor_hcp import compute_donor_hcp_randomized_interval
        res_dr = compute_donor_hcp_randomized_interval(
            U_calibration=U_calibration,
            Z_calibration=Z_calibration,
            U_test=U_test,
            Z_test=Z_test,
            o_observed=o_observed,
            alpha=alpha,
            alpha_selection=alpha_selection,
            mu_method=mu_method,
            test_index_target=test_index_target,
            tau_override=tau_override,
        )
        return {
            'interval': res_dr['interval'],
            'mu_hat': res_dr.get('mu_hat', 0.0),
            'number_selected_groups': 1,
        }

    S = np.sort(np.concatenate([S_tilde, [test_idx]]))
    S_size = len(S)

    if tau_override is None:
        tau = int(np.floor(o_observed / 2))
        if tau < 0:
            tau = 0
        if o_observed > 0 and tau >= o_observed:
            tau = o_observed - 1
    else:
        tau = int(tau_override)
        tau = max(0, min(tau, max(0, o_observed - 1)))

    S_comp = np.setdiff1d(list(range(K + 1)), S)
    if len(S_comp) == 0:
        global_model = None
    else:
        global_model = mu_method['fit_global'](
            U_matrix=U_all,
            Z_list=Z_all,
            group_index_vector=list(S_comp),
        )

    n_slots = o_observed + 1 - tau
    w_slot = 1.0 / (S_size * n_slots)

    values = []
    weights = []
    offset_test = 0.0

    for j in S:
        if j < K:
            Uj = U_calibration[j, :]
            Zj = Z_calibration[j]
            Nj = N[j]
            if Nj < (o_observed + 1):
                continue

            Tj = np.random.choice(Nj, size=o_observed + 1, replace=False)

            if tau > 0:
                Tj_train = Tj[:tau]
                Tj_cal = Tj[tau:(o_observed + 1)]
                offset_j = mu_method['fit_group_adjustment'](
                    model_global=global_model,
                    u_group_vector=Uj,
                    Z_group_list=Zj,
                    training_index_vector=list(Tj_train),
                )
            else:
                Tj_cal = Tj[0:(o_observed + 1)]
                offset_j = 0.0

            for i_idx in Tj_cal:
                z = Zj[i_idx]
                mu = mu_method['predict_group_mu'](
                    model_global=global_model,
                    group_adjustment=offset_j,
                    x_vector=z['X'],
                    u_group_vector=Uj,
                )
                values.append(np.abs(z['Y'] - mu))
                weights.append(w_slot)

        else:
            Uj = U_test[0, :]
            Zj = Z_test
            Nj = len(Zj)
            if Nj < o_observed:
                raise ValueError("compute_sample_hcp_randomized_interval: test group must have at least o observed points.")

            if tau > 0:
                Tj_train = np.random.choice(o_observed, size=tau, replace=False)
                Tj_cal = np.setdiff1d(list(range(o_observed)), Tj_train)
                offset_test = mu_method['fit_group_adjustment'](
                    model_global=global_model,
                    u_group_vector=Uj,
                    Z_group_list=Zj,
                    training_index_vector=list(Tj_train),
                )
            else:
                Tj_cal = list(range(o_observed))
                offset_test = 0.0

            for i_idx in Tj_cal:
                z = Zj[i_idx]
                mu = mu_method['predict_group_mu'](
                    model_global=global_model,
                    group_adjustment=offset_test,
                    x_vector=z['X'],
                    u_group_vector=Uj,
                )
                values.append(np.abs(z['Y'] - mu))
                weights.append(w_slot)

            values.append(np.inf)
            weights.append(w_slot)

    if len(values) == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': S_size,
        }

    q_info = _select_conformal_q(
        values, weights, alpha,
        quantile_mode=quantile_mode,
        quantile_random_seed=quantile_random_seed,
        quantile_rng=quantile_rng,
        return_quantile_info=True,
    )
    q = q_info["q_randomized"]

    X_target = Z_test[test_index_target]['X']
    mu_center = mu_method['predict_group_mu'](
        model_global=global_model,
        group_adjustment=offset_test,
        x_vector=X_target,
        u_group_vector=U_test[0, :],
    )

    if np.isinf(q):
        interval = (-np.inf, np.inf)
    else:
        interval = (mu_center - q, mu_center + q)

    out = {
        'interval': interval,
        'mu_hat': mu_center,
        'number_selected_groups': S_size,
    }
    if return_quantile_info or quantile_mode == "randomized":
        merge_quantile_info(out, q_info)
    return out


def compute_sample_hcp_randomized_interval(U_calibration, Z_calibration, U_test, Z_test,
                                           o_observed, alpha, alpha_selection, mu_method,
                                           test_index_target=None,
                                           tau_override=None,
                                           quantile_mode="deterministic",
                                           quantile_random_seed=None,
                                           quantile_rng=None,
                                           return_quantile_info=False):
    """Randomized sample-HCP interval."""
    return _compute_sample_hcp_randomized_interval_impl(
        U_calibration=U_calibration,
        Z_calibration=Z_calibration,
        U_test=U_test,
        Z_test=Z_test,
        o_observed=o_observed,
        alpha=alpha,
        alpha_selection=alpha_selection,
        mu_method=mu_method,
        test_index_target=test_index_target,
        tau_override=tau_override,
        quantile_mode=quantile_mode,
        quantile_random_seed=quantile_random_seed,
        quantile_rng=quantile_rng,
        return_quantile_info=return_quantile_info,
    )


def compute_sample_hcp_derandomized_interval(U_calibration, Z_calibration, U_test, Z_test,
                                             o_observed, alpha, alpha_selection, mu_method,
                                             test_index_target=None,
                                             quantile_mode="deterministic",
                                             quantile_random_seed=None,
                                             quantile_rng=None,
                                             return_quantile_info=False):
    """
    Derandomized sample-HCP using the closed-form averaged subsampling measure.

    Uses deterministic averaging and quantile level 1 - alpha/2,
    i.e., weighted_quantile called with alpha/2.
    """
    K = len(Z_calibration)
    if test_index_target is None:
        test_index_target = o_observed

    if K == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': 0,
            'infinite_weight': 1.0,
        }

    N = np.array([len(Z_calibration[j]) for j in range(K)], dtype=int)
    N_test = len(Z_test)
    if N_test < max(o_observed + 1, test_index_target + 1):
        raise ValueError(
            "compute_sample_hcp_derandomized_interval: Z_test must have at least max(o+1, target+1) observations."
        )

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
    )

    if len(S_tilde) == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': 1,
            'infinite_weight': 1.0,
        }

    m = len(S_tilde)
    test_idx = K
    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    S = np.sort(np.concatenate([S_tilde, [test_idx]]))
    S_comp = np.setdiff1d(list(range(K + 1)), S)

    if len(S_comp) == 0:
        global_model = None
    else:
        global_model = mu_method['fit_global'](
            U_matrix=U_all,
            Z_list=Z_all,
            group_index_vector=list(S_comp)
        )

    values = []
    weights = []

    for j in S_tilde:
        N_j = int(N[j])
        if N_j <= 0:
            continue
        w_j = 1.0 / ((m + 1.0) * N_j)
        for z in Z_calibration[j]:
            mu = mu_method['predict_group_mu'](
                model_global=global_model,
                group_adjustment=0.0,
                x_vector=z['X'],
                u_group_vector=U_calibration[j, :]
            )
            values.append(abs(z['Y'] - mu))
            weights.append(w_j)

    w_test = 1.0 / ((m + 1.0) * (o_observed + 1.0))
    n_obs = min(o_observed, len(Z_test))
    for i in range(n_obs):
        z = Z_test[i]
        mu = mu_method['predict_group_mu'](
            model_global=global_model,
            group_adjustment=0.0,
            x_vector=z['X'],
            u_group_vector=U_test[0, :]
        )
        values.append(abs(z['Y'] - mu))
        weights.append(w_test)

    values.append(np.inf)
    weights.append(w_test)

    q_info = None
    if len(values) == 0:
        q = np.inf
    else:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        total_w = float(np.sum(weights))
        if total_w <= 0 or np.any(weights < 0):
            q = np.inf
        else:
            weights = weights / total_w
            q_info = _select_conformal_q(
                values, weights, alpha / 2.0,
                quantile_mode=quantile_mode,
                quantile_random_seed=quantile_random_seed,
                quantile_rng=quantile_rng,
                return_quantile_info=True,
            )
            q = q_info["q_randomized"]

    X_target = Z_test[test_index_target]['X']
    mu_center = mu_method['predict_global'](
        model_global=global_model,
        x_vector=X_target,
        u_vector=U_test[0, :]
    )

    interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

    out = {
        'interval': interval,
        'mu_hat': mu_center,
        'number_selected_groups': int(m + 1),
        'infinite_weight': float(w_test),
    }
    if return_quantile_info or quantile_mode == "randomized":
        merge_quantile_info(out, q_info)
    return out
