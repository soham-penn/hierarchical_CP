"""
Donor-HCP methods.

This module exposes two separate variants:
- randomized donor-HCP (legacy HCP++)
- derandomized donor-HCP (closed-form averaged donor measure)
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


def _select_s_tilde_with_tie_randomization(
    N,
    o_observed,
    alpha_selection,
    include_donor_slot=False,
    rng=None,
):
    """
    Select S_tilde with tie-aware randomization at the alpha cutoff.

    If many groups share the boundary sample size, randomize only among those
    boundary groups so that |S_tilde| matches the target count implied by the
    alpha_1 proportion. For donor-HCP randomized/derandomized, one extra slot
    can be requested because a donor is removed from S_tilde before forming the
    calibration set. Uses a fixed seed to keep this split reproducible.
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
    if include_donor_slot:
        n_target += 1
    n_target = max(1, min(n_target, len(eligible)))

    if len(interior) >= n_target:
        order = np.argsort(N[interior], kind='stable')
        return np.sort(interior[order[:n_target]])

    n_boundary_needed = n_target - len(interior)
    if n_boundary_needed >= len(boundary):
        return np.sort(np.concatenate([interior, boundary]))

    if rng is None:
        rng = np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    keep_boundary = rng.choice(boundary, size=n_boundary_needed, replace=False)
    return np.sort(np.concatenate([interior, keep_boundary]))


def get_hcp_train_cal_split(sample_sizes, o_observed, alpha_selection):
    """
    Build train/calibration group indices for baseline HCP (no donor removal).

    Uses the same S_tilde selection logic as donor-HCP (with same seed) to ensure
    both methods independently arrive at the same initial split, but HCP does NOT
    remove a donor. This makes the methods independent while using compatible splits.

    Parameters
    ----------
    sample_sizes : array-like
        Sample sizes for each group
    o_observed : int
        Number of observed test-group observations (typically 0 for HCP)
    alpha_selection : float
        Selection parameter

    Returns
    -------
    train_idx, calib_idx : lists
        Training and calibration group indices (calib_idx = S_tilde, no donor removal)
    """
    N = np.asarray(sample_sizes, dtype=int)
    K = len(N)
    if K == 0:
        return [], []

    # Use same selection logic as donor-HCP, but WITHOUT the donor slot
    # This ensures HCP independently gets S_tilde without donor removal
    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=False,  # HCP doesn't need donor slot
    )

    if len(S_tilde) == 0:
        K0 = K // 2
        return list(range(K0)), list(range(K0, K))

    calib_idx = S_tilde.astype(int)
    train_idx = np.setdiff1d(np.arange(K, dtype=int), calib_idx).astype(int)

    return train_idx.tolist(), calib_idx.tolist()


def get_donor_style_train_cal_split(sample_sizes, o_observed, alpha_selection):
    """
    Build train/calibration group indices using donor-HCP selection logic.

    This returns the split induced by tie-aware S_tilde selection and donor
    removal for donor-HCP methods.

    Parameters
    ----------
    sample_sizes : array-like
        Sample sizes for each group
    o_observed : int
        Number of observed test-group observations
    alpha_selection : float
        Selection parameter

    Returns
    -------
    train_idx, calib_idx : lists
        Training and calibration group indices (after donor removal)
    """
    N = np.asarray(sample_sizes, dtype=int)
    K = len(N)
    if K == 0:
        return [], []

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=True,
    )

    if len(S_tilde) == 0:
        K0 = K // 2
        return list(range(K0)), list(range(K0, K))

    rng = np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    donor = int(rng.choice(S_tilde))
    calib_idx = np.setdiff1d(S_tilde, [donor]).astype(int)
    train_idx = np.setdiff1d(np.arange(K, dtype=int), calib_idx).astype(int)

    return train_idx.tolist(), calib_idx.tolist()


def _compute_donor_hcp_randomized_interval_impl(U_calibration, Z_calibration, U_test, Z_test,
                                                o_observed, alpha, alpha_selection, mu_method,
                                                test_index_target=None,
                                                tau_override=None,
                                                random_seed=None,
                                                quantile_mode="deterministic",
                                                quantile_random_seed=None,
                                                quantile_rng=None,
                                                return_quantile_info=False):
    """Randomized donor-HCP interval implementation (formerly in hcp_plus.py)."""
    K = len(Z_calibration)
    N = np.array([len(Z_calibration[j]) for j in range(K)])
    test_idx = K
    if test_index_target is None:
        test_index_target = o_observed

    U_all = np.vstack([U_calibration, U_test])
    Z_all = Z_calibration + [Z_test]

    if K == 0:
        N_test = len(Z_test)
        if N_test < (o_observed + 1):
            return {
                'interval': (-np.inf, np.inf),
                'mu_hat': 0.0,
                'number_selected_groups': 0,
                'donor_group_index': None,
            }

        if tau_override is None:
            tau = int(np.floor(o_observed / 2))
            if tau < 0:
                tau = 0
            if o_observed > 0 and tau >= o_observed:
                tau = o_observed - 1
        else:
            tau = int(tau_override)
            tau = max(0, min(tau, max(0, o_observed - 1)))

        if tau > 0:
            train_idx = list(range(tau))
            global_model = mu_method['fit_global'](
                U_matrix=U_test,
                Z_list=[Z_test],
                group_index_vector=[0],
            )
            offset_test = mu_method['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_test[0, :],
                Z_group_list=Z_test,
                training_index_vector=train_idx,
            )
        else:
            global_model = None
            offset_test = 0.0

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
                        u_group_vector=U_test[0, :],
                    )
                else:
                    mu = 0.0
                scores.append(np.abs(z['Y'] - mu))
            scores.append(np.inf)
            weights = np.ones(len(scores)) / len(scores)
            q_info = _select_conformal_q(
                scores, weights, alpha,
                quantile_mode=quantile_mode,
                quantile_random_seed=quantile_random_seed,
                quantile_rng=quantile_rng,
                return_quantile_info=True,
            )
            q = q_info["q_randomized"]
        else:
            q = np.inf
            q_info = None

        X_target = Z_test[test_index_target]['X']
        if global_model is not None:
            mu_center = mu_method['predict_group_mu'](
                model_global=global_model,
                group_adjustment=offset_test,
                x_vector=X_target,
                u_group_vector=U_test[0, :],
            )
        else:
            mu_center = 0.0

        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)
        out = {
            'interval': interval,
            'mu_hat': mu_center,
            'number_selected_groups': 0,
            'donor_group_index': None,
        }
        if return_quantile_info or quantile_mode == "randomized":
            merge_quantile_info(out, q_info)
        return out

    N_test = len(Z_test)
    if N_test < max(o_observed + 1, test_index_target + 1):
        raise ValueError("compute_donor_hcp_randomized_interval: Z_test must have at least max(o+1, target+1) observations.")

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=True,
    )

    if len(S_tilde) == 0:
        global_model = mu_method['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(range(K)),
        )

        if tau_override is None:
            tau = int(np.floor(o_observed / 2))
            if tau < 0:
                tau = 0
            if o_observed > 0 and tau >= o_observed:
                tau = o_observed - 1
        else:
            tau = int(tau_override)
            tau = max(0, min(tau, max(0, o_observed - 1)))

        if tau > 0:
            train_idx = list(range(tau))
            offset_test = mu_method['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_test[0, :],
                Z_group_list=Z_test,
                training_index_vector=train_idx,
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
                    u_group_vector=U_test[0, :],
                )
                scores.append(np.abs(z['Y'] - mu))
            scores.append(np.inf)
            weights = np.ones(len(scores)) / len(scores)
            q_info = _select_conformal_q(
                scores, weights, alpha,
                quantile_mode=quantile_mode,
                quantile_random_seed=quantile_random_seed,
                quantile_rng=quantile_rng,
                return_quantile_info=True,
            )
            q = q_info["q_randomized"]
        else:
            q = np.inf
            q_info = None

        X_target = Z_test[test_index_target]['X']
        mu_center = mu_method['predict_group_mu'](
            model_global=global_model,
            group_adjustment=offset_test,
            x_vector=X_target,
            u_group_vector=U_test[0, :],
        )
        interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)
        out = {
            'interval': interval,
            'number_selected_groups': 1,
            'donor_group_index': None,
        }
        if return_quantile_info or quantile_mode == "randomized":
            merge_quantile_info(out, q_info)
        return out

    donor_rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    donor = donor_rng.choice(S_tilde)
    N_donor = N[donor]

    S_cal = np.sort(np.setdiff1d(S_tilde, [donor]))
    S = np.sort(np.concatenate([S_cal, [test_idx]]))
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

    scores = []
    weights = []

    for j in S_cal:
        N_j = N[j]
        if N_j <= tau:
            continue

        if tau > 0:
            train_idx = list(range(tau))
            offset_j = mu_method['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_calibration[j, :],
                Z_group_list=Z_calibration[j],
                training_index_vector=train_idx,
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
                u_group_vector=U_calibration[j, :],
            )
            scores.append(np.abs(z['Y'] - mu))

        if len(idx_tail) > 0:
            w_j = 1.0 / (S_size * len(idx_tail))
            weights.extend([w_j] * len(idx_tail))

    if tau > 0:
        train_idx_test = list(range(tau))
        offset_test = mu_method['fit_group_adjustment'](
            model_global=global_model,
            u_group_vector=U_test[0, :],
            Z_group_list=Z_test,
            training_index_vector=train_idx_test,
        )
    else:
        offset_test = 0.0

    idx_tail_test = list(range(tau, o_observed)) if o_observed > tau else []
    test_scores = []
    for i in idx_tail_test:
        z = Z_test[i]
        mu = mu_method['predict_group_mu'](
            model_global=global_model,
            group_adjustment=offset_test,
            x_vector=z['X'],
            u_group_vector=U_test[0, :],
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
        q_info = None
    else:
        q_info = _select_conformal_q(
            scores, weights, alpha,
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
    interval = (-np.inf, np.inf) if np.isinf(q) else (mu_center - q, mu_center + q)

    out = {
        'interval': interval,
        'mu_hat': mu_center,
        'number_selected_groups': S_size,
        'donor_group_index': int(donor),
    }
    if return_quantile_info or quantile_mode == "randomized":
        merge_quantile_info(out, q_info)
    return out


def compute_donor_hcp_randomized_interval(U_calibration, Z_calibration, U_test, Z_test,
                                          o_observed, alpha, alpha_selection, mu_method,
                                          test_index_target=None,
                                          tau_override=None,
                                          random_seed=None,
                                          quantile_mode="deterministic",
                                          quantile_random_seed=None,
                                          quantile_rng=None,
                                          return_quantile_info=False):
    """Randomized donor-HCP interval."""
    return _compute_donor_hcp_randomized_interval_impl(
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
        random_seed=random_seed,
        quantile_mode=quantile_mode,
        quantile_random_seed=quantile_random_seed,
        quantile_rng=quantile_rng,
        return_quantile_info=return_quantile_info,
    )


def compute_donor_hcp_derandomized_interval(U_calibration, Z_calibration, U_test, Z_test,
                                            o_observed, alpha, alpha_selection, mu_method,
                                            test_index_target=None,
                                            quantile_mode="deterministic",
                                            quantile_random_seed=None,
                                            quantile_rng=None,
                                            return_quantile_info=False):
    """
    Derandomized donor-HCP using the closed-form averaged donor measure.

    Uses the deterministic donor average and quantile level 1 - alpha/2,
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
            'donor_group_index': None,
            'bar_w': 0.0,
            'infinite_weight': 1.0,
        }

    N = np.array([len(Z_calibration[j]) for j in range(K)], dtype=int)
    N_test = len(Z_test)
    if N_test < max(o_observed + 1, test_index_target + 1):
        raise ValueError(
            "compute_donor_hcp_derandomized_interval: Z_test must have at least max(o+1, target+1) observations."
        )

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
        include_donor_slot=True,
    )

    if len(S_tilde) == 0:
        return {
            'interval': (-np.inf, np.inf),
            'mu_hat': 0.0,
            'number_selected_groups': 1,
            'donor_group_index': None,
            'bar_w': 0.0,
            'infinite_weight': 1.0,
        }

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

    M = len(S_tilde)
    inv_sizes = np.array([1.0 / float(N[j]) for j in S_tilde], dtype=float)
    bar_w = float(np.sum(inv_sizes) / (M * M))

    values = []
    weights = []

    for j in S_tilde:
        N_j = int(N[j])
        if N_j <= 0:
            continue
        w_j = (M - 1.0) / (M * M * N_j)
        if w_j <= 0:
            continue
        for z in Z_calibration[j]:
            mu = mu_method['predict_group_mu'](
                model_global=global_model,
                group_adjustment=0.0,
                x_vector=z['X'],
                u_group_vector=U_calibration[j, :]
            )
            values.append(abs(z['Y'] - mu))
            weights.append(w_j)

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
        weights.append(bar_w)

    w_inf = (1.0 / M) - n_obs * bar_w
    if w_inf > 0:
        values.append(np.inf)
        weights.append(w_inf)

    q_info = None
    if len(values) == 0:
        q = np.inf
    else:
        weights = np.asarray(weights, dtype=float)
        values = np.asarray(values, dtype=float)
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
        'number_selected_groups': int(M + 1),
        'donor_group_index': None,
        'bar_w': bar_w,
        'infinite_weight': float(max(0.0, w_inf)),
    }
    if return_quantile_info or quantile_mode == "randomized":
        merge_quantile_info(out, q_info)
    return out
