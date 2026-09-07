"""
GHCP (paper Algorithm 1): restricted donor hierarchical conformal prediction.

Public routine for paper experiments: ``compute_donor_hcp_randomized_interval``
with ``quantile_mode="deterministic"`` and ``alpha_selection=0.5`` (η=0.5).

An older working name for this method was “HCP++”; that name is not used in
the paper. ``compute_donor_hcp_derandomized_interval`` is a closed-form
average over donors and is not used in the paper tables.

The empirical-Bayes merger (``mu_method["merger"] in {"bayes_re","bayes","re"}``)
is opt-in and unused in Section 3.1/3.2.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scores import conformal_threshold, merge_quantile_info
from methods.mu_methods import (
    global_weight,
    resolve_shrinkage_weight,
    estimate_re_variance_components,
    bayes_re_global_weight,
    set_w_g_override,
)
from methods.nonconformity import (
    conformity_score,
    fit_score_aux,
    get_score_type,
    interval_from_threshold,
)


def _shrinkage_weight(mu_method, global_model, group_adjustment) -> float:
    """Same w_g as mean shrinkage: N_comp^c / (N_comp^c + τ); honors w_g_override."""
    return resolve_shrinkage_weight(mu_method, global_model, group_adjustment)


_ALPHA_SPLIT_TIEBREAK_SEED = 123


def _fit_global_scale_model(U_all, Z_all, group_index_vector, mu_method, global_model,
                            alpha=0.1, tau=0):
    """Fit auxiliary score models (RF-σ or CQR quantiles)."""
    if get_score_type(mu_method) == "absolute" and bool(mu_method.get("use_standardized_score", False)):
        mu_method = dict(mu_method)
        mu_method["score_type"] = "studentized"
    return fit_score_aux(
        U_all=U_all,
        Z_all=Z_all,
        group_index_vector=group_index_vector,
        mu_method=mu_method,
        global_model=global_model,
        alpha=alpha,
        tau=int(tau),
    )


def _predict_scale(scale_model, x_vector, u_vector, w_g: float = 1.0):
    from methods.nonconformity import predict_scale
    if scale_model is None:
        return 1.0
    if isinstance(scale_model, dict):
        return predict_scale(scale_model, x_vector, u_vector, w_g=w_g)
    x = np.asarray(x_vector, dtype=float).ravel()
    u = np.asarray(u_vector, dtype=float).ravel()
    feat = np.concatenate([[1.0], x, u])
    log_s = float(feat @ scale_model)
    s = float(np.exp(log_s))
    return float(np.clip(s, 1e-6, 1e12))


def _mu_global_only(mu_method, global_model, x_vector, u_vector) -> float:
    return float(mu_method['predict_group_mu'](
        model_global=global_model,
        group_adjustment=0.0,
        x_vector=x_vector,
        u_group_vector=u_vector,
    ))


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


def _restricted_pool(N, o_observed, alpha_selection, n_sel, rng=None):
    """Pick n_sel eligible groups by size, with randomization on the cutoff tie."""
    N = np.asarray(N, dtype=int)
    K = len(N)
    if K == 0:
        return np.array([], dtype=int)

    p = min(1.0, float(np.mean(N <= o_observed) + (1 - alpha_selection)))
    V_o = np.sort(N)[max(0, int(np.ceil(p * K)) - 1)]
    eligible = np.where(N > o_observed)[0]
    if len(eligible) == 0:
        return eligible

    interior = eligible[N[eligible] < V_o]
    boundary = eligible[N[eligible] == V_o]
    exterior = eligible[N[eligible] > V_o]
    n_sel = max(1, min(int(n_sel), len(eligible)))

    if rng is None:
        rng = np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)

    chosen: list[int] = []
    if len(interior):
        order = np.argsort(N[interior], kind="stable")
        chosen.extend(interior[order].tolist())
    if len(chosen) >= n_sel:
        return np.sort(np.asarray(chosen[:n_sel], dtype=int))

    need = n_sel - len(chosen)
    if len(boundary):
        take = min(need, len(boundary))
        if take >= len(boundary):
            chosen.extend(boundary.tolist())
        else:
            pick = rng.choice(boundary, size=take, replace=False)
            chosen.extend(np.asarray(pick, dtype=int).tolist())
    if len(chosen) >= n_sel:
        return np.sort(np.asarray(chosen[:n_sel], dtype=int))

    need = n_sel - len(chosen)
    if len(exterior) and need > 0:
        order = np.argsort(N[exterior], kind="stable")
        chosen.extend(exterior[order[:need]].tolist())

    return np.sort(np.asarray(chosen[:n_sel], dtype=int))


def _select_s_tilde_with_tie_randomization(N, o_observed, alpha_selection, rng=None):
    """Restricted group pool for GHCP."""
    N = np.asarray(N, dtype=int)
    K = len(N)
    p = min(1.0, float(np.mean(N <= o_observed) + (1 - alpha_selection)))
    n_sel = int(np.ceil(p * K) - np.sum(N <= o_observed) + 1)
    return _restricted_pool(N, o_observed, alpha_selection, n_sel, rng=rng)


def get_hcp_train_cal_split(sample_sizes, o_observed, alpha_selection):
    """Train/calibration group indices for sample HCP."""
    N = np.asarray(sample_sizes, dtype=int)
    K = len(N)
    if K == 0:
        return [], []

    p = min(1.0, float(np.mean(N <= o_observed) + (1 - alpha_selection)))
    n_sel = int(np.ceil(p * K) - np.sum(N <= o_observed))
    S_tilde = _restricted_pool(N, o_observed, alpha_selection, n_sel)

    if len(S_tilde) == 0:
        K0 = K // 2
        return list(range(K0)), list(range(K0, K))

    calib_idx = S_tilde.astype(int)
    train_idx = np.setdiff1d(np.arange(K, dtype=int), calib_idx).astype(int)

    return train_idx.tolist(), calib_idx.tolist()


def get_donor_style_train_cal_split(sample_sizes, o_observed, alpha_selection):
    """Train/calibration group indices for GHCP."""
    N = np.asarray(sample_sizes, dtype=int)
    K = len(N)
    if K == 0:
        return [], []

    S_tilde = _select_s_tilde_with_tie_randomization(
        N=N,
        o_observed=o_observed,
        alpha_selection=alpha_selection,
    )

    if len(S_tilde) == 0:
        K0 = K // 2
        return list(range(K0)), list(range(K0, K))

    rng = np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    donor = int(rng.choice(S_tilde))
    calib_idx = np.setdiff1d(S_tilde, [donor]).astype(int)
    train_idx = np.setdiff1d(np.arange(K, dtype=int), S_tilde).astype(int)

    return train_idx.tolist(), calib_idx.tolist()


def _score_meta_entry(y, mu, mu_g, w_g, x, weight, *, is_inf=False, sigma=None):
    """One conformal score atom for score-swap caches."""
    return {
        "y": None if is_inf else float(y),
        "mu": None if is_inf or mu is None else float(mu),
        "mu_g": None if is_inf or mu_g is None else float(mu_g),
        "w_g": float(w_g),
        "x": None if x is None else np.asarray(x, dtype=float).ravel().copy(),
        "weight": float(weight),
        "is_inf": bool(is_inf),
        "sigma": None if sigma is None else float(sigma),
    }


def _s_comp_sigma_points(U_all, Z_all, S_comp, mu_method, global_model):
    """(x, y, mu_g) on S_comp for later RF-σ / studentized score swaps."""
    if global_model is None:
        return []
    pts = []
    for j in S_comp:
        Uj = U_all[j]
        for z in Z_all[j]:
            mu_g = _mu_global_only(mu_method, global_model, z["X"], Uj)
            pts.append({
                "x": np.asarray(z["X"], dtype=float).ravel().copy(),
                "y": float(z["Y"]),
                "mu_g": float(mu_g),
            })
    return pts


def _compute_donor_hcp_randomized_interval_impl(U_calibration, Z_calibration, U_test, Z_test,
                                                o_observed, alpha, alpha_selection, mu_method,
                                                test_index_target=None,
                                                tau_override=None,
                                                random_seed=None,
                                                quantile_mode="deterministic",
                                                quantile_random_seed=None,
                                                quantile_rng=None,
                                                return_quantile_info=False,
                                                return_intermediates=False,
                                                alphas=None):
    """Randomized donor-HCP interval implementation (formerly in hcp_plus.py).

    If ``alphas`` is a non-empty list, the global fit and scores are computed
    once and only the conformal quantile/interval is repeated per α. The return
    value then includes ``by_alpha`` mapping each α to an interval dict.
    """
    alpha_list = [float(a) for a in alphas] if alphas is not None else None
    if alpha_list is not None and len(alpha_list) == 0:
        raise ValueError("alphas must be non-empty when provided")
    if alpha_list is None:
        alpha_list_eff = [float(alpha)]
    else:
        alpha_list_eff = alpha_list
    # Use first α for any aux fits that still take a single alpha (absolute scores ignore it).
    alpha = float(alpha_list_eff[0]) if alpha_list is not None else float(alpha)

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
            scale_model = _fit_global_scale_model(
                U_all=U_test,
                Z_all=[Z_test],
                group_index_vector=[0],
                mu_method=mu_method,
                global_model=global_model,
                alpha=alpha,
                tau=tau,
            )
            offset_test = mu_method['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_test[0, :],
                Z_group_list=Z_test,
                training_index_vector=train_idx,
            )
        else:
            global_model = None
            scale_model = None
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
                mu_g = _mu_global_only(mu_method, global_model, z['X'], U_test[0, :]) if global_model is not None else mu
                w_g = _shrinkage_weight(mu_method, global_model, offset_test)
                scores.append(conformity_score(
                    z['Y'], mu, z['X'], U_test[0, :], scale_model, mu_global=mu_g, w_g=w_g,
                ))
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

        mu_g_target = _mu_global_only(mu_method, global_model, X_target, U_test[0, :]) if global_model is not None else mu_center
        w_g_t = _shrinkage_weight(mu_method, global_model, offset_test)
        interval = interval_from_threshold(
            q, mu_center, X_target, U_test[0, :], scale_model, mu_global=mu_g_target, w_g=w_g_t,
        )
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
    )
    S_comp = np.setdiff1d(np.arange(K, dtype=int), S_tilde)

    if len(S_tilde) == 0:
        if tau_override is None:
            tau = int(np.floor(o_observed / 2))
            if tau < 0:
                tau = 0
            if o_observed > 0 and tau >= o_observed:
                tau = o_observed - 1
        else:
            tau = int(tau_override)
            tau = max(0, min(tau, max(0, o_observed - 1)))

        global_model = mu_method['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(range(K)),
        )
        scale_model = _fit_global_scale_model(
            U_all=U_calibration,
            Z_all=Z_calibration,
            group_index_vector=list(range(K)),
            mu_method=mu_method,
            global_model=global_model,
            alpha=alpha,
            tau=tau,
        )
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
                mu_g = _mu_global_only(mu_method, global_model, z['X'], U_test[0, :]) if global_model is not None else mu
                w_g = _shrinkage_weight(mu_method, global_model, offset_test)
                scores.append(conformity_score(
                    z['Y'], mu, z['X'], U_test[0, :], scale_model, mu_global=mu_g, w_g=w_g,
                ))
            scores.append(np.inf)
            weights = np.ones(len(scores)) / len(scores)
            # Per-α quantiles (same pattern as the donor path). Previously only
            # alpha_list_eff[0] was evaluated and broadcast to all α.
            qs_by_alpha = {}
            for a in alpha_list_eff:
                q_info_a = _select_conformal_q(
                    scores, weights, a,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=quantile_random_seed,
                    quantile_rng=quantile_rng,
                    return_quantile_info=True,
                )
                qs_by_alpha[a] = (q_info_a["q_randomized"], q_info_a)
            q, q_info = qs_by_alpha[alpha_list_eff[0]]
        else:
            qs_by_alpha = {a: (np.inf, None) for a in alpha_list_eff}
            q, q_info = np.inf, None

        X_target = Z_test[test_index_target]['X']
        mu_center = mu_method['predict_group_mu'](
            model_global=global_model,
            group_adjustment=offset_test,
            x_vector=X_target,
            u_group_vector=U_test[0, :],
        )
        mu_g_target = _mu_global_only(mu_method, global_model, X_target, U_test[0, :]) if global_model is not None else mu_center
        w_g_t = _shrinkage_weight(mu_method, global_model, offset_test)

        def _one_out_empty(q_val, q_info_val):
            interval = interval_from_threshold(
                q_val, mu_center, X_target, U_test[0, :], scale_model,
                mu_global=mu_g_target, w_g=w_g_t,
            )
            out_a = {
                'interval': interval,
                'mu_hat': mu_center,
                'number_selected_groups': 1,
                'donor_group_index': None,
            }
            if return_quantile_info or quantile_mode == "randomized":
                merge_quantile_info(out_a, q_info_val)
            return out_a

        if alpha_list is not None:
            by_alpha = {a: _one_out_empty(qv, qi) for a, (qv, qi) in qs_by_alpha.items()}
            return {
                'by_alpha': by_alpha,
                'interval': by_alpha[alpha_list_eff[0]]['interval'],
                'mu_hat': mu_center,
                'number_selected_groups': 1,
                'donor_group_index': None,
            }
        return _one_out_empty(q, q_info)

    if tau_override is None:
        tau = int(np.floor(o_observed / 2))
        if tau < 0:
            tau = 0
        if o_observed > 0 and tau >= o_observed:
            tau = o_observed - 1
    else:
        tau = int(tau_override)
        tau = max(0, min(tau, max(0, o_observed - 1)))

    if len(S_comp) == 0:
        global_model = None
        scale_model = None
    else:
        global_model = mu_method['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
            group_index_vector=list(S_comp),
        )
        scale_model = _fit_global_scale_model(
            U_all=U_calibration,
            Z_all=Z_calibration,
            group_index_vector=list(S_comp),
            mu_method=mu_method,
            global_model=global_model,
            alpha=alpha,
            tau=tau,
        )

    donor_rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng(_ALPHA_SPLIT_TIEBREAK_SEED)
    donor = donor_rng.choice(S_tilde)
    N_donor = N[donor]

    S_cal = np.sort(np.setdiff1d(S_tilde, [donor]))
    S = np.sort(np.concatenate([S_cal, [test_idx]]))
    S_size = len(S)

    # RE merger: MoM (σ², τ_B²) from RF residuals on S_cal (held out of Strain).
    mu_work = mu_method
    merger = str(mu_method.get("merger", "") or "").lower().replace("-", "_")
    bayes_info = None
    clear_bayes_override = False
    if merger in ("bayes_re", "bayes", "re"):
        if global_model is not None and tau > 0 and len(S_cal) > 0:
            sigma2, tau_B2 = estimate_re_variance_components(
                U_matrix=U_calibration,
                Z_list=Z_calibration,
                group_index_vector=list(S_cal),
                mu_method=mu_method,
                global_model=global_model,
            )
            w_bayes = bayes_re_global_weight(sigma2, tau_B2, tau)
            bayes_info = {
                "sigma2": sigma2,
                "tau_B2": tau_B2,
                "w_g": w_bayes,
                "w_local": float(1.0 - w_bayes),
                "tau": int(tau),
                "n_cal_groups": int(len(S_cal)),
                "n_train_groups": int(len(S_comp)),
            }
        else:
            w_bayes = 1.0
            bayes_info = {
                "sigma2": None,
                "tau_B2": None,
                "w_g": 1.0,
                "w_local": 0.0,
                "tau": int(tau),
                "n_cal_groups": int(len(S_cal)),
                "n_train_groups": int(len(S_comp)),
            }
        mu_work = set_w_g_override(mu_method, w_bayes)
        clear_bayes_override = True

    scores = []
    weights = []
    scores_meta = [] if return_intermediates else None
    from methods.nonconformity import predict_scale as _predict_scale_aux

    for j in S_cal:
        N_j = N[j]
        if N_j <= tau:
            continue

        if tau > 0:
            train_idx = list(range(tau))
            offset_j = mu_work['fit_group_adjustment'](
                model_global=global_model,
                u_group_vector=U_calibration[j, :],
                Z_group_list=Z_calibration[j],
                training_index_vector=train_idx,
            )
        else:
            offset_j = 0.0

        idx_tail = list(range(tau, N_j))
        w_j = 1.0 / (S_size * len(idx_tail)) if len(idx_tail) > 0 else 0.0
        for i in idx_tail:
            z = Z_calibration[j][i]
            mu = mu_work['predict_group_mu'](
                model_global=global_model,
                group_adjustment=offset_j,
                x_vector=z['X'],
                u_group_vector=U_calibration[j, :],
            )
            mu_g = _mu_global_only(mu_work, global_model, z['X'], U_calibration[j, :])
            w_g = _shrinkage_weight(mu_work, global_model, offset_j)
            scores.append(conformity_score(
                z['Y'], mu, z['X'], U_calibration[j, :], scale_model, mu_global=mu_g, w_g=w_g,
            ))
            weights.append(w_j)
            if scores_meta is not None:
                sig = None
                if scale_model is not None:
                    sig = _predict_scale_aux(scale_model, z['X'], U_calibration[j, :], w_g=w_g)
                scores_meta.append(_score_meta_entry(
                    z['Y'], mu, mu_g, w_g, z['X'], w_j, sigma=sig,
                ))

    if tau > 0:
        train_idx_test = list(range(tau))
        offset_test = mu_work['fit_group_adjustment'](
            model_global=global_model,
            u_group_vector=U_test[0, :],
            Z_group_list=Z_test,
            training_index_vector=train_idx_test,
        )
    else:
        offset_test = 0.0

    idx_tail_test = list(range(tau, o_observed)) if o_observed > tau else []
    test_scores = []
    w_g_test = _shrinkage_weight(mu_work, global_model, offset_test)
    for i in idx_tail_test:
        z = Z_test[i]
        mu = mu_work['predict_group_mu'](
            model_global=global_model,
            group_adjustment=offset_test,
            x_vector=z['X'],
            u_group_vector=U_test[0, :],
        )
        mu_g = _mu_global_only(mu_work, global_model, z['X'], U_test[0, :])
        test_scores.append(conformity_score(
            z['Y'], mu, z['X'], U_test[0, :], scale_model, mu_global=mu_g, w_g=w_g_test,
        ))
        if scores_meta is not None:
            scores_meta.append({
                "_pending_test": True,
                "y": float(z['Y']),
                "mu": float(mu),
                "mu_g": float(mu_g),
                "w_g": float(w_g_test),
                "x": np.asarray(z['X'], dtype=float).ravel().copy(),
                "is_inf": False,
                "sigma": (
                    None if scale_model is None
                    else float(_predict_scale_aux(scale_model, z['X'], U_test[0, :], w_g=w_g_test))
                ),
            })

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
        if scores_meta is not None:
            for m in scores_meta:
                if m.pop("_pending_test", False):
                    m["weight"] = float(w_test)
            for _ in range(n_inf):
                scores_meta.append(_score_meta_entry(
                    None, None, None, w_g_test, None, w_test, is_inf=True,
                ))

    if len(scores) == 0 or all(w <= 0 for w in weights):
        qs_by_alpha = {a: (np.inf, None) for a in alpha_list_eff}
        q, q_info = np.inf, None
    else:
        qs_by_alpha = {}
        for a in alpha_list_eff:
            q_info_a = _select_conformal_q(
                scores, weights, a,
                quantile_mode=quantile_mode,
                quantile_random_seed=quantile_random_seed,
                quantile_rng=quantile_rng,
                return_quantile_info=True,
            )
            qs_by_alpha[a] = (q_info_a["q_randomized"], q_info_a)
        q, q_info = qs_by_alpha[alpha_list_eff[0]]

    X_target = Z_test[test_index_target]['X']
    mu_center = mu_work['predict_group_mu'](
        model_global=global_model,
        group_adjustment=offset_test,
        x_vector=X_target,
        u_group_vector=U_test[0, :],
    )
    mu_g_target = _mu_global_only(mu_work, global_model, X_target, U_test[0, :])

    def _one_out(q_val, q_info_val):
        interval = interval_from_threshold(
            q_val, mu_center, X_target, U_test[0, :], scale_model,
            mu_global=mu_g_target, w_g=w_g_test,
        )
        out_a = {
            'interval': interval,
            'mu_hat': mu_center,
            'number_selected_groups': S_size,
            'donor_group_index': int(donor),
        }
        if return_quantile_info or quantile_mode == "randomized":
            merge_quantile_info(out_a, q_info_val)
        return out_a

    if alpha_list is not None:
        by_alpha = {a: _one_out(qv, qi) for a, (qv, qi) in qs_by_alpha.items()}
        out = {
            'by_alpha': by_alpha,
            'interval': by_alpha[alpha_list_eff[0]]['interval'],
            'mu_hat': mu_center,
            'number_selected_groups': S_size,
            'donor_group_index': int(donor),
        }
    else:
        out = _one_out(q, q_info)

    if return_intermediates:
        sig_t = None
        if scale_model is not None:
            sig_t = float(_predict_scale_aux(scale_model, X_target, U_test[0, :], w_g=w_g_test))
        out['intermediates'] = {
            'S_tilde': np.asarray(S_tilde, dtype=int).copy(),
            'donor': int(donor),
            'S_cal': np.asarray(S_cal, dtype=int).copy(),
            'S_comp': np.asarray(S_comp, dtype=int).copy(),
            'tau': int(tau),
            'S_size': int(S_size),
            'N_donor': int(N_donor),
            'o_observed': int(o_observed),
            'test_index_target': int(test_index_target),
            'scores_meta': scores_meta or [],
            'target': {
                'mu': float(mu_center),
                'mu_g': float(mu_g_target),
                'w_g': float(w_g_test),
                'x': np.asarray(X_target, dtype=float).ravel().copy(),
                'sigma': sig_t,
            },
            's_comp_points': _s_comp_sigma_points(
                U_calibration, Z_calibration, S_comp, mu_work, global_model,
            ),
            'q': float(q) if np.isfinite(q) else np.inf,
            'bayes_re': bayes_info,
        }
    if clear_bayes_override:
        set_w_g_override(mu_method, None)
    return out


def compute_donor_hcp_randomized_interval(U_calibration, Z_calibration, U_test, Z_test,
                                          o_observed, alpha, alpha_selection, mu_method,
                                          test_index_target=None,
                                          tau_override=None,
                                          random_seed=None,
                                          quantile_mode="deterministic",
                                          quantile_random_seed=None,
                                          quantile_rng=None,
                                          return_quantile_info=False,
                                          return_intermediates=False,
                                          alphas=None):
    """Randomized donor-HCP interval.

    Pass ``alphas=[...]`` to reuse one global fit/score pass across miscoverage
    levels; results are under ``by_alpha``.
    """
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
        return_intermediates=return_intermediates,
        alphas=alphas,
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
    S_comp = np.setdiff1d(np.arange(K, dtype=int), S_tilde)

    if len(S_comp) == 0:
        global_model = None
    else:
        global_model = mu_method['fit_global'](
            U_matrix=U_calibration,
            Z_list=Z_calibration,
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
