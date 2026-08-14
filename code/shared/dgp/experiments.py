"""
Experiment Runner

This module contains functions for running experiments comparing different
hierarchical conformal prediction methods.
"""

import numpy as np
import pandas as pd
from .data_generation import generate_calibration_data, generate_test_group
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius
)
from methods.donor_hcp import (
    compute_donor_hcp_randomized_interval,
    compute_donor_hcp_derandomized_interval,
    get_hcp_train_cal_split,
)
from methods.sample_hcp import (
    compute_sample_hcp_randomized_interval,
    compute_sample_hcp_derandomized_interval,
)
from scores import absolute_residual_score, conformal_threshold, make_quantile_seed


def _safe_width_median(width_array):
    """Return median finite width; if none are finite, return +inf."""
    arr = np.asarray(width_array, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.inf
    return float(np.median(finite))


def _split_conformal_radius(abs_residuals, alpha, quantile_mode="randomized",
                            random_seed=None, rng=None):
    """Finite-sample split-conformal radius from calibration residuals."""
    scores = np.asarray(abs_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    if quantile_mode == "randomized":
        # Append inf so randomized_weighted_quantile targets exact 1-alpha coverage
        # (same augmentation as donor-HCP test-only split CP).
        scores = np.append(scores, np.inf)
        weights = np.ones(len(scores), dtype=float) / len(scores)
        return conformal_threshold(
            scores=scores,
            weights=weights,
            alpha=alpha,
            quantile_mode="randomized",
            random_seed=random_seed,
            rng=rng,
            return_info=False,
        )
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    k = max(1, k)
    return float(np.partition(scores, k - 1)[k - 1])


def _compute_std_cp_interval_global_center(
    x_hist,
    y_hist,
    x_target,
    alpha,
    *,
    mu_hat_hist,
    mu_hat_target,
    u_hist=None,
    u_target=None,
    quantile_mode="randomized",
    random_seed=None,
    rng=None,
    score_aux=None,
):
    """
    Inductive Std-CP with a frozen global predictor as the center.

    Uses all o history residuals (no local refit / half-split). With studentized
    score_aux, residuals are |Y-μ|/σ and the interval is μ ± q·σ(x_target).
    """
    from methods.nonconformity import predict_scale

    y_hist = np.asarray(y_hist, dtype=float).ravel()
    mu_hat_hist = np.asarray(mu_hat_hist, dtype=float).ravel()
    if len(y_hist) < 1 or len(mu_hat_hist) != len(y_hist):
        return (-np.inf, np.inf)

    st = (score_aux or {}).get("score_type", "absolute")
    x_hist = np.asarray(x_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)
    x_target = np.asarray(x_target, dtype=float).ravel()

    if u_hist is None:
        u_hist = np.zeros((len(y_hist), 0), dtype=float)
    else:
        u_hist = np.asarray(u_hist, dtype=float)
        if u_hist.ndim == 1:
            u_hist = np.repeat(u_hist.reshape(1, -1), len(y_hist), axis=0)
    if u_target is None:
        u_target = np.zeros(0, dtype=float)
    else:
        u_target = np.asarray(u_target, dtype=float).ravel()

    if st == "studentized" and score_aux is not None:
        sig_hist = np.array(
            [predict_scale(score_aux, x_hist[i], u_hist[i], w_g=1.0) for i in range(len(y_hist))],
            dtype=float,
        )
        scores = np.abs(y_hist - mu_hat_hist) / np.clip(sig_hist, 1e-6, 1e12)
    else:
        scores = np.abs(y_hist - mu_hat_hist)

    radius = _split_conformal_radius(
        scores,
        alpha,
        quantile_mode=quantile_mode,
        random_seed=random_seed,
        rng=rng,
    )
    if not np.isfinite(radius):
        return (-np.inf, np.inf)
    if st == "studentized" and score_aux is not None:
        s_t = float(predict_scale(score_aux, x_target, u_target, w_g=1.0))
        return (float(mu_hat_target) - radius * s_t, float(mu_hat_target) + radius * s_t)
    return (float(mu_hat_target) - radius, float(mu_hat_target) + radius)


def _compute_std_cp_interval(
    x_hist,
    y_hist,
    x_target,
    alpha,
    quantile_mode="randomized",
    random_seed=None,
    rng=None,
    *,
    ntree=50,
    nodesize=5,
    mtry=None,
    rf_random_state=123,
    score_type="absolute",
    return_models=False,
):
    """
    Standard split CP using only within-group history (local models on X).

    score_type:
      absolute     — local RF mean + |Y-μ| scores
      studentized  — local RF mean + local RF-σ on |Y-μ|, score |Y-μ|/σ
      cqr          — local quantile models, score max(q_lo-Y, Y-q_hi)

    If return_models=True, returns (interval, models_dict) with fitted local RF
    mean / scale (for reuse as GHCP local predictors).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    empty_models = {"rf": None, "rf_scale": None, "score_type": "absolute"}

    def _ret(interval, models=None):
        if return_models:
            return interval, (models if models is not None else empty_models)
        return interval

    x_hist = np.asarray(x_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)

    n_hist = len(y_hist)
    n_train = n_hist // 2
    n_cal = n_hist - n_train
    if n_train < 1 or n_cal < 1:
        return _ret((-np.inf, np.inf))

    if rng is None:
        rng = np.random.default_rng(None if random_seed is None else int(random_seed))
    perm = rng.permutation(n_hist)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:]

    x_train = x_hist[train_idx]
    y_train = y_hist[train_idx]
    x_cal = x_hist[cal_idx]
    y_cal = y_hist[cal_idx]
    x_target = np.asarray(x_target, dtype=float).reshape(1, -1)

    n_estimators = int(ntree)
    min_leaf = int(nodesize)
    min_leaf = max(1, min(min_leaf, len(y_train)))
    max_features = "sqrt" if mtry is None else mtry
    rs = int(rf_random_state)
    st = str(score_type).lower().replace("-", "_")
    if st in ("standardized", "studentised", "studentized", "std", "sigma"):
        st = "studentized"
    elif st in ("cqr", "conformalized_quantile", "quantile"):
        st = "cqr"
    else:
        st = "absolute"

    if st == "cqr":
        a = float(alpha)
        a = min(max(a, 1e-4), 0.49)
        common = dict(
            max_depth=4,
            learning_rate=0.1,
            max_iter=120,
            min_samples_leaf=max(1, min(5, len(y_train))),
            random_state=rs,
        )
        q_lo_m = HistGradientBoostingRegressor(loss="quantile", quantile=a / 2.0, **common)
        q_hi_m = HistGradientBoostingRegressor(loss="quantile", quantile=1.0 - a / 2.0, **common)
        q_lo_m.fit(x_train, y_train)
        q_hi_m.fit(x_train, y_train)
        lo_cal = q_lo_m.predict(x_cal)
        hi_cal = q_hi_m.predict(x_cal)
        scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
        radius = _split_conformal_radius(
            scores, alpha, quantile_mode=quantile_mode, random_seed=random_seed, rng=rng,
        )
        models_cqr = {"rf": None, "rf_scale": None, "score_type": "cqr", "q_lo": q_lo_m, "q_hi": q_hi_m}
        if not np.isfinite(radius):
            return _ret((-np.inf, np.inf), models_cqr)
        lo_t = float(q_lo_m.predict(x_target)[0])
        hi_t = float(q_hi_m.predict(x_target)[0])
        if hi_t < lo_t:
            lo_t, hi_t = hi_t, lo_t
        return _ret((lo_t - radius, hi_t + radius), models_cqr)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_leaf,
        max_features=max_features,
        random_state=rs,
        n_jobs=1,
    )
    rf.fit(x_train, y_train)
    y_cal_pred = rf.predict(x_cal)
    y_hat = float(rf.predict(x_target)[0])

    if st == "studentized":
        abs_tr = np.abs(y_train - rf.predict(x_train))
        rf_s = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            random_state=rs + 17,
            n_jobs=1,
        )
        rf_s.fit(x_train, abs_tr)
        s_cal = np.clip(rf_s.predict(x_cal), 1e-6, 1e12)
        scores = np.abs(y_cal - y_cal_pred) / s_cal
        radius = _split_conformal_radius(
            scores, alpha, quantile_mode=quantile_mode, random_seed=random_seed, rng=rng,
        )
        models = {"rf": rf, "rf_scale": rf_s, "score_type": "studentized"}
        if not np.isfinite(radius):
            return _ret((-np.inf, np.inf), models)
        s_t = float(np.clip(rf_s.predict(x_target)[0], 1e-6, 1e12))
        return _ret((y_hat - radius * s_t, y_hat + radius * s_t), models)

    radius = _split_conformal_radius(
        np.abs(y_cal - y_cal_pred),
        alpha,
        quantile_mode=quantile_mode,
        random_seed=random_seed,
        rng=rng,
    )
    models = {"rf": rf, "rf_scale": None, "score_type": "absolute"}
    if np.isfinite(radius):
        return _ret((y_hat - radius, y_hat + radius), models)
    return _ret((-np.inf, np.inf), models)


def run_one_experiment(number_groups_k, lambda_Poisson, dgp_specification,
                       o_values, target_index, alpha=0.1, number_subsampling_repetitions=50,
                       alpha_selection=0.1, number_test_groups=100,
                       mu_method_baseline=None, mu_method_hcp=None,
                       mu_method_hcp_no_within=None,
                       quantile_mode="deterministic",
                       quantile_base_seed=0,
                       experiment_id=0,
                       alphas=None):
    """
    Run one experiment comparing all methods across multiple o values.

    Pass ``alphas=[...]`` to evaluate several miscoverage levels on the *same*
    simulated groups / fits. GHCP randomized (with and without WGT) reuses one
    score pass per o; other methods still loop α but share the DGP draw.
    """
    alpha_list = [float(a) for a in alphas] if alphas is not None else [float(alpha)]
    if not alpha_list:
        raise ValueError("alphas must be non-empty")

    def _by_alpha(res):
        if isinstance(res, dict) and "by_alpha" in res:
            return res["by_alpha"]
        return {a: res for a in alpha_list}

    def _record(cov, wid, inf, a, o, t, lo, hi, true_target):
        cov[a][o][t] = (lo <= true_target <= hi)
        if np.isfinite(lo) and np.isfinite(hi):
            wid[a][o][t] = hi - lo
        else:
            inf[a][o] += 1

    cal = generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification
    )
    U_cal = cal['U_calibration']
    Z_cal = cal['Z_calibration']

    sample_sizes = [len(zg) for zg in Z_cal]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_selection,
    )

    model_baseline = mu_method_baseline['fit_global'](
        U_matrix=U_cal,
        Z_list=Z_cal,
        group_index_vector=train_idx
    )

    scores_list = []
    for j in calib_idx:
        Zj = Z_cal[j]
        Uj = U_cal[j, :]
        yj = np.array([z['Y'] for z in Zj])
        Xj = np.array([z['X'] for z in Zj])
        muj = np.array([
            mu_method_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=Xj[i],
                u_vector=Uj
            ) for i in range(len(Xj))
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    # Baseline radii per alpha (scores shared)
    T = {}
    for a in alpha_list:
        T[a] = {
            'hcp': compute_hcp_interval_radius(
                scores_list, a, quantile_mode=quantile_mode,
                random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "hcp"),
            ),
            'pool': compute_pooling_interval_radius(
                scores_list, a, quantile_mode=quantile_mode,
                random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "pool"),
            ),
            'sub': compute_subsampling_once_interval_radius(
                scores_list, a, quantile_mode=quantile_mode,
                random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "sub"),
            ),
            'rep': compute_repeated_subsampling_interval_radius(
                scores_list, a, number_subsampling_repetitions,
                quantile_mode=quantile_mode,
                random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "rep"),
            ),
        }

    def _empty_o():
        return {o: np.zeros(number_test_groups, dtype=bool) for o in o_values}

    def _empty_w():
        return {o: np.full(number_test_groups, np.nan) for o in o_values}

    def _empty_inf():
        return {o: 0 for o in o_values}

    cov_dr = {a: _empty_o() for a in alpha_list}
    cov_dr_no = {a: _empty_o() for a in alpha_list}
    cov_dd = {a: _empty_o() for a in alpha_list}
    cov_sr = {a: _empty_o() for a in alpha_list}
    cov_sd = {a: _empty_o() for a in alpha_list}
    cov_stdcp = {a: _empty_o() for a in alpha_list}
    wid_dr = {a: _empty_w() for a in alpha_list}
    wid_dr_no = {a: _empty_w() for a in alpha_list}
    wid_dd = {a: _empty_w() for a in alpha_list}
    wid_sr = {a: _empty_w() for a in alpha_list}
    wid_sd = {a: _empty_w() for a in alpha_list}
    wid_stdcp = {a: _empty_w() for a in alpha_list}
    inf_dr = {a: _empty_inf() for a in alpha_list}
    inf_dr_no = {a: _empty_inf() for a in alpha_list}
    inf_dd = {a: _empty_inf() for a in alpha_list}
    inf_sr = {a: _empty_inf() for a in alpha_list}
    inf_sd = {a: _empty_inf() for a in alpha_list}
    inf_stdcp = {a: _empty_inf() for a in alpha_list}

    cov_hcp = {a: np.zeros(number_test_groups, dtype=bool) for a in alpha_list}
    cov_pool = {a: np.zeros(number_test_groups, dtype=bool) for a in alpha_list}
    cov_sub = {a: np.zeros(number_test_groups, dtype=bool) for a in alpha_list}
    cov_rep = {a: np.zeros(number_test_groups, dtype=bool) for a in alpha_list}
    wid_hcp = {a: np.full(number_test_groups, np.nan) for a in alpha_list}
    wid_pool = {a: np.full(number_test_groups, np.nan) for a in alpha_list}
    wid_sub = {a: np.full(number_test_groups, np.nan) for a in alpha_list}
    wid_rep = {a: np.full(number_test_groups, np.nan) for a in alpha_list}
    inf_hcp = {a: 0 for a in alpha_list}
    inf_pool = {a: 0 for a in alpha_list}
    inf_sub = {a: 0 for a in alpha_list}
    inf_rep = {a: 0 for a in alpha_list}

    for t in range(number_test_groups):
        test = generate_test_group(
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_observed=target_index
        )
        U_test = test['U_test']
        Z_test = test['Z_test']
        N_test = test['N_test']
        if N_test < target_index + 1:
            continue

        true_target = Z_test[target_index]['Y']
        X_target = Z_test[target_index]['X']
        mu_baseline_hat = mu_method_baseline['predict_global'](
            model_global=model_baseline,
            x_vector=X_target,
            u_vector=U_test[0, :]
        )

        for a in alpha_list:
            for key, cov_arr, wid_arr, inf_arr in [
                ('hcp', cov_hcp, wid_hcp, inf_hcp),
                ('pool', cov_pool, wid_pool, inf_pool),
                ('sub', cov_sub, wid_sub, inf_sub),
                ('rep', cov_rep, wid_rep, inf_rep),
            ]:
                Trad = T[a][key]
                lo = mu_baseline_hat - Trad if np.isfinite(Trad) else -np.inf
                hi = mu_baseline_hat + Trad if np.isfinite(Trad) else np.inf
                cov_arr[a][t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_arr[a][t] = hi - lo
                else:
                    inf_arr[a] += 1

        for o in o_values:
            # GHCP ± WGT: one fit/score pass for all alphas
            res_dr = compute_donor_hcp_randomized_interval(
                U_calibration=U_cal, Z_calibration=Z_cal,
                U_test=U_test, Z_test=Z_test,
                o_observed=o, alpha=alpha_list[0],
                alpha_selection=alpha_selection, mu_method=mu_method_hcp,
                test_index_target=target_index, quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, experiment_id, target_index, o, "donor_hcp"
                ),
                alphas=alpha_list,
            )
            for a, ra in _by_alpha(res_dr).items():
                lo, hi = ra['interval']
                _record(cov_dr, wid_dr, inf_dr, a, o, t, lo, hi, true_target)

            if mu_method_hcp_no_within is not None:
                res_dr_no = compute_donor_hcp_randomized_interval(
                    U_calibration=U_cal, Z_calibration=Z_cal,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha_list[0],
                    alpha_selection=alpha_selection, mu_method=mu_method_hcp_no_within,
                    test_index_target=target_index, tau_override=0,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "donor_hcp_no_within"
                    ),
                    alphas=alpha_list,
                )
                for a, ra in _by_alpha(res_dr_no).items():
                    lo, hi = ra['interval']
                    _record(cov_dr_no, wid_dr_no, inf_dr_no, a, o, t, lo, hi, true_target)

            # Remaining methods: still α-loop (smaller share of runtime)
            for a in alpha_list:
                res_dd = compute_donor_hcp_derandomized_interval(
                    U_calibration=U_cal, Z_calibration=Z_cal,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=a, test_index_target=target_index,
                    alpha_selection=alpha_selection, mu_method=mu_method_hcp,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "donor_hcp_derand"
                    ),
                )
                lo, hi = res_dd['interval']
                _record(cov_dd, wid_dd, inf_dd, a, o, t, lo, hi, true_target)

                res_sr = compute_sample_hcp_randomized_interval(
                    U_calibration=U_cal, Z_calibration=Z_cal,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=a, test_index_target=target_index,
                    alpha_selection=alpha_selection, mu_method=mu_method_hcp,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "sample_hcp"
                    ),
                )
                lo, hi = res_sr['interval']
                _record(cov_sr, wid_sr, inf_sr, a, o, t, lo, hi, true_target)

                res_sd = compute_sample_hcp_derandomized_interval(
                    U_calibration=U_cal, Z_calibration=Z_cal,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=a, alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp, test_index_target=target_index,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "sample_hcp_derand"
                    ),
                )
                lo, hi = res_sd['interval']
                _record(cov_sd, wid_sd, inf_sd, a, o, t, lo, hi, true_target)

                x_hist = np.array([Z_test[i]['X'] for i in range(o)])
                y_hist = np.array([Z_test[i]['Y'] for i in range(o)])
                stdcp_split_seed = make_quantile_seed(
                    quantile_base_seed, experiment_id, target_index, o, "stdcp_split"
                )
                lo, hi = _compute_std_cp_interval(
                    x_hist=x_hist, y_hist=y_hist, x_target=X_target, alpha=a,
                    # Std-CP only: randomized split CP (HCP/GHCP keep quantile_mode).
                    quantile_mode="randomized",
                    random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "stdcp"
                    ),
                    rng=np.random.default_rng(stdcp_split_seed),
                )
                _record(cov_stdcp, wid_stdcp, inf_stdcp, a, o, t, lo, hi, true_target)

    rows = []
    for a in alpha_list:
        for o in o_values:
            rows.append({
                'alpha': a,
                'o_observed': o,
                'quantile_mode': quantile_mode,
                'coverage_donor_hcp_randomized': np.mean(cov_dr[a][o]),
                'coverage_donor_hcp_no_within': (
                    np.mean(cov_dr_no[a][o]) if mu_method_hcp_no_within is not None else np.nan
                ),
                'coverage_donor_hcp_derandomized': np.mean(cov_dd[a][o]),
                'coverage_sample_hcp_randomized': np.mean(cov_sr[a][o]),
                'coverage_sample_hcp_derandomized': np.mean(cov_sd[a][o]),
                'coverage_stdcp': np.mean(cov_stdcp[a][o]),
                'coverage_hcp': np.mean(cov_hcp[a]),
                'coverage_pool': np.mean(cov_pool[a]),
                'coverage_sub': np.mean(cov_sub[a]),
                'coverage_rep': np.mean(cov_rep[a]),
                'width_donor_hcp_randomized': _safe_width_median(wid_dr[a][o]),
                'width_donor_hcp_no_within': (
                    _safe_width_median(wid_dr_no[a][o]) if mu_method_hcp_no_within is not None else np.nan
                ),
                'width_donor_hcp_derandomized': _safe_width_median(wid_dd[a][o]),
                'width_sample_hcp_randomized': _safe_width_median(wid_sr[a][o]),
                'width_sample_hcp_derandomized': _safe_width_median(wid_sd[a][o]),
                'width_stdcp': _safe_width_median(wid_stdcp[a][o]),
                'width_hcp': _safe_width_median(wid_hcp[a]),
                'width_pool': _safe_width_median(wid_pool[a]),
                'width_sub': _safe_width_median(wid_sub[a]),
                'width_rep': _safe_width_median(wid_rep[a]),
                'infinite_donor_hcp_randomized': inf_dr[a][o],
                'infinite_donor_hcp_no_within': (
                    inf_dr_no[a][o] if mu_method_hcp_no_within is not None else np.nan
                ),
                'infinite_donor_hcp_derandomized': inf_dd[a][o],
                'infinite_sample_hcp_randomized': inf_sr[a][o],
                'infinite_sample_hcp_derandomized': inf_sd[a][o],
                'infinite_stdcp': inf_stdcp[a][o],
                'infinite_hcp': inf_hcp[a],
                'infinite_pool': inf_pool[a],
                'infinite_sub': inf_sub[a],
                'infinite_rep': inf_rep[a],
            })
    return pd.DataFrame(rows)


def run_experiments_outer(number_experiments, number_groups_k, lambda_Poisson,
                          dgp_specification, o_values, target_index, alpha=0.1,
                          number_subsampling_repetitions=50,
                          alpha_selection=0.1, number_test_groups=100,
                          mu_method_baseline=None, mu_method_hcp=None,
                          mu_method_hcp_no_within=None,
                          show_progress=True,
                          quantile_mode="deterministic",
                          quantile_base_seed=0,
                          alphas=None):
    """
    Run multiple experiments and return combined results.

    If ``alphas`` is provided, each replicate is evaluated at all listed
    miscoverage levels on the same simulated data (see ``run_one_experiment``).
    """
    if show_progress:
        al = alphas if alphas is not None else [alpha]
        print(
            f"Running {number_experiments} experiments sequentially "
            f"(alphas={list(al)})..."
        )

    results_list = []
    for e in range(number_experiments):
        if show_progress and (e + 1) % 5 == 0:
            print(f"  Completed {e + 1}/{number_experiments} experiments")

        res = run_one_experiment(
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_values=o_values,
            target_index=target_index,
            alpha=alpha,
            alphas=alphas,
            number_subsampling_repetitions=number_subsampling_repetitions,
            alpha_selection=alpha_selection,
            number_test_groups=number_test_groups,
            mu_method_baseline=mu_method_baseline,
            mu_method_hcp=mu_method_hcp,
            mu_method_hcp_no_within=mu_method_hcp_no_within,
            quantile_mode=quantile_mode,
            quantile_base_seed=quantile_base_seed,
            experiment_id=e + 1,
        )
        res['experiment'] = e + 1
        results_list.append(res)

    combined = pd.concat(results_list, ignore_index=True)
    cols = ['experiment'] + [c for c in combined.columns if c != 'experiment']
    return combined[cols]


# ======================================================================
# CONDITIONAL COVERAGE EXPERIMENT
# ======================================================================

def run_one_experiment_conditional(
        number_groups_k, lambda_Poisson, dgp_specification,
        o_values, target_index, u_grid, alpha,
        number_subsampling_repetitions, alpha_selection,
        number_test_groups, mu_method_baseline, mu_method_hcp,
        quantile_mode="deterministic",
        quantile_base_seed=0,
        experiment_id=0):
    """
    One U-conditional-coverage experiment.

    For each value ``u_g`` in ``u_grid`` we generate
    ``number_test_groups`` independent test groups whose group covariate
    is fixed at  U = [u_g, 0.5, …, 0.5].  Within each test group, X
    and Y are drawn from P(X, Y | U) as usual.

    This tests the group-level (U-conditional) coverage guarantee:
    for a new group with a specific U value, does the method cover
    Y_{target_index} with probability ≥ 1 − α?

    Returns
    -------
    pd.DataFrame   One row per (o_observed, u_grid_val).
    """
    d = dgp_specification['dimension']

    # ------------------------------------------------------------------
    # 1. Calibration data (once for the whole experiment)
    # ------------------------------------------------------------------
    cal   = generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification,
    )
    U_cal = cal['U_calibration']
    Z_cal = cal['Z_calibration']

    # ------------------------------------------------------------------
    # 2. Train / calib split (HCP uses independent split without donor removal)
    # ------------------------------------------------------------------
    sample_sizes = [len(zg) for zg in Z_cal]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_selection,
    )

    # ------------------------------------------------------------------
    # 3. Fit baseline model (once)
    # ------------------------------------------------------------------
    model_baseline = mu_method_baseline['fit_global'](
        U_matrix=U_cal,
        Z_list=Z_cal,
        group_index_vector=train_idx,
    )

    # ------------------------------------------------------------------
    # 4. Calibration scores → baseline radii (once)
    # ------------------------------------------------------------------
    scores_list = []
    for j in calib_idx:
        Zj  = Z_cal[j]
        Uj  = U_cal[j, :]
        yj  = np.array([z['Y'] for z in Zj])
        Xj  = np.array([z['X'] for z in Zj])
        muj = np.array([
            mu_method_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=Xj[i],
                u_vector=Uj,
            ) for i in range(len(Xj))
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    T_hcp = compute_hcp_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "hcp"),
    )
    T_pool = compute_pooling_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "pool"),
    )
    T_sub = compute_subsampling_once_interval_radius(
        scores_list, alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "sub"),
    )
    T_rep = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, number_subsampling_repetitions,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "rep"),
    )

    # ------------------------------------------------------------------
    # 5. Allocate result arrays   shape: (n_o, n_grid, n_test)
    # ------------------------------------------------------------------
    G  = len(u_grid)
    O  = len(o_values)
    NT = number_test_groups

    cov_dr  = np.zeros((O, G, NT))
    cov_dd  = np.zeros((O, G, NT))
    cov_sr  = np.zeros((O, G, NT))
    cov_sd  = np.zeros((O, G, NT))
    wid_dr  = np.full((O, G, NT), np.nan)
    wid_dd  = np.full((O, G, NT), np.nan)
    wid_sr  = np.full((O, G, NT), np.nan)
    wid_sd  = np.full((O, G, NT), np.nan)
    inf_dr  = np.zeros((O, G), dtype=int)
    inf_dd  = np.zeros((O, G), dtype=int)
    inf_sr  = np.zeros((O, G), dtype=int)
    inf_sd  = np.zeros((O, G), dtype=int)

    cov_hcp  = np.zeros((G, NT))
    cov_pool = np.zeros((G, NT))
    cov_sub  = np.zeros((G, NT))
    cov_rep  = np.zeros((G, NT))
    wid_hcp  = np.full((G, NT), np.nan)
    wid_pool = np.full((G, NT), np.nan)
    wid_sub  = np.full((G, NT), np.nan)
    wid_rep  = np.full((G, NT), np.nan)

    # Baseline radii are the same for all (u_g, t); count inf once
    inf_hcp  = (G * NT) if not np.isfinite(T_hcp)  else 0
    inf_pool = (G * NT) if not np.isfinite(T_pool) else 0
    inf_sub  = (G * NT) if not np.isfinite(T_sub)  else 0
    inf_rep  = (G * NT) if not np.isfinite(T_rep)  else 0

    # ------------------------------------------------------------------
    # 6. Main loop: for each U-grid point, generate NT test groups
    # ------------------------------------------------------------------
    for g, u_g in enumerate(u_grid):
        # Fix U = [u_g, 0.5, …, 0.5]  (entire U vector is determined)
        U_fixed = np.full(d, 0.5)
        U_fixed[0] = float(u_g)

        for t in range(NT):
            test = generate_test_group(
                lambda_Poisson=lambda_Poisson,
                dgp_specification=dgp_specification,
                o_observed=target_index,
                fixed_U=U_fixed,
            )
            U_test = test['U_test']
            Z_test = test['Z_test']
            N_test = test['N_test']

            if N_test < target_index + 1:
                continue

            true_target = Z_test[target_index]['Y']
            X_target    = Z_test[target_index]['X']
            U_t         = U_test[0, :]

            # ---- Baseline methods ----
            mu_hat = mu_method_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=X_target,
                u_vector=U_t,
            )
            for T_rad, c_arr, w_arr in [
                (T_hcp,  cov_hcp[g],  wid_hcp[g]),
                (T_pool, cov_pool[g], wid_pool[g]),
                (T_sub,  cov_sub[g],  wid_sub[g]),
                (T_rep,  cov_rep[g],  wid_rep[g]),
            ]:
                lo = mu_hat - T_rad if np.isfinite(T_rad) else -np.inf
                hi = mu_hat + T_rad if np.isfinite(T_rad) else  np.inf
                c_arr[t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    w_arr[t] = hi - lo

            # ---- donor-HCP and sample-HCP variants for each o value ----
            for oi, o in enumerate(o_values):
                res_dr = compute_donor_hcp_randomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                    test_index_target=target_index,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "donor_hcp"
                    ),
                )
                lo, hi = res_dr['interval']
                cov_dr[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_dr[oi, g, t] = hi - lo
                else:
                    inf_dr[oi, g] += 1

                res_dd = compute_donor_hcp_derandomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    test_index_target=target_index,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "donor_hcp_derand"
                    ),
                )
                lo, hi = res_dd['interval']
                cov_dd[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_dd[oi, g, t] = hi - lo
                else:
                    inf_dd[oi, g] += 1

                res_sr = compute_sample_hcp_randomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    test_index_target=target_index,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "sample_hcp"
                    ),
                )
                lo, hi = res_sr['interval']
                cov_sr[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_sr[oi, g, t] = hi - lo
                else:
                    inf_sr[oi, g] += 1

                res_sd = compute_sample_hcp_derandomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_selection,
                    mu_method=mu_method_hcp,
                    test_index_target=target_index,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment_id, target_index, o, "sample_hcp_derand"
                    ),
                )
                lo, hi = res_sd['interval']
                cov_sd[oi, g, t] = (lo <= true_target <= hi)
                if np.isfinite(lo) and np.isfinite(hi):
                    wid_sd[oi, g, t] = hi - lo
                else:
                    inf_sd[oi, g] += 1

    # ------------------------------------------------------------------
    # 7. Aggregate: one row per (o, u_g)
    # ------------------------------------------------------------------
    rows = []
    for oi, o in enumerate(o_values):
        for g, u_g in enumerate(u_grid):
            rows.append({
                'o_observed':          o,
                'quantile_mode':       quantile_mode,
                'u_grid_val':          float(u_g),
                'coverage_donor_hcp_randomized': np.mean(cov_dr[oi, g]),
                'coverage_donor_hcp_derandomized': np.mean(cov_dd[oi, g]),
                'coverage_sample_hcp_randomized': np.mean(cov_sr[oi, g]),
                'coverage_sample_hcp_derandomized': np.mean(cov_sd[oi, g]),
                'coverage_hcp':        np.mean(cov_hcp[g]),
                'coverage_pool':       np.mean(cov_pool[g]),
                'coverage_sub':        np.mean(cov_sub[g]),
                'coverage_rep':        np.mean(cov_rep[g]),
                'width_donor_hcp_randomized': np.nanmedian(wid_dr[oi, g]),
                'width_donor_hcp_derandomized': np.nanmedian(wid_dd[oi, g]),
                'width_sample_hcp_randomized': np.nanmedian(wid_sr[oi, g]),
                'width_sample_hcp_derandomized': np.nanmedian(wid_sd[oi, g]),
                'width_hcp':           np.nanmedian(wid_hcp[g]),
                'width_pool':          np.nanmedian(wid_pool[g]),
                'width_sub':           np.nanmedian(wid_sub[g]),
                'width_rep':           np.nanmedian(wid_rep[g]),
                'infinite_donor_hcp_randomized': int(inf_dr[oi, g]),
                'infinite_donor_hcp_derandomized': int(inf_dd[oi, g]),
                'infinite_sample_hcp_randomized': int(inf_sr[oi, g]),
                'infinite_sample_hcp_derandomized': int(inf_sd[oi, g]),
                'infinite_hcp':        inf_hcp,
                'infinite_pool':       inf_pool,
                'infinite_sub':        inf_sub,
                'infinite_rep':        inf_rep,
            })
    return pd.DataFrame(rows)


def run_experiments_conditional(
        number_experiments, number_groups_k, lambda_Poisson,
        dgp_specification, o_values, target_index, u_grid,
        alpha=0.1, number_subsampling_repetitions=50,
        alpha_selection=0.1, number_test_groups=50,
        mu_method_baseline=None, mu_method_hcp=None,
        show_progress=True):
    """
    Run ``number_experiments`` U-conditional-coverage experiments and combine.

    Each experiment generates an independent calibration dataset and
    evaluates coverage for test groups whose group covariate U[0] is
    fixed to each value in ``u_grid`` (with U[1:] = 0.5).

    Returns
    -------
    pd.DataFrame
        One row per (experiment, o_observed, u_grid_val).
    """
    if show_progress:
        print(f"Running {number_experiments} U-conditional experiments …")

    results_list = []
    for e in range(number_experiments):
        if show_progress and (e + 1) % 5 == 0:
            print(f"  Completed {e + 1}/{number_experiments} experiments")
        res = run_one_experiment_conditional(
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_values=o_values,
            target_index=target_index,
            u_grid=u_grid,
            alpha=alpha,
            number_subsampling_repetitions=number_subsampling_repetitions,
            alpha_selection=alpha_selection,
            number_test_groups=number_test_groups,
            mu_method_baseline=mu_method_baseline,
            mu_method_hcp=mu_method_hcp,
        )
        res['experiment'] = e + 1
        results_list.append(res)

    combined = pd.concat(results_list, ignore_index=True)
    cols = ['experiment'] + [c for c in combined.columns if c != 'experiment']
    return combined[cols]
