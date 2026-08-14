"""Run one DGP experiment with full RF capture artifacts."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from code.marginal.dgp_capture.capture_mu import (
    CaptureCallContext,
    drain_pred_log,
    wrap_mu_for_capture,
)
from code.marginal.run_true_marginal_latent_intercept_experiments import (
    without_within_group_training,
)
from code.shared.dgp.experiments import (
    _compute_std_cp_interval,
    _safe_width_median,
)
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_repeated_subsampling_interval_radius,
    compute_subsampling_once_interval_radius,
)
from methods.donor_hcp import (
    compute_donor_hcp_randomized_interval,
    get_hcp_train_cal_split,
)
from methods.sample_hcp import compute_sample_hcp_randomized_interval
from methods.mu_methods import (
    create_mu_method_random_forest_global_only,
    create_mu_method_random_forest_offset,
    create_mu_method_random_forest_local_rf_offset,
    create_mu_method_ud_squared_global_only,
    create_mu_method_ud_squared_offset,
    create_mu_method_bayes_joint_xy_global_only,
    create_mu_method_bayes_joint_xy_offset,
    set_w_g_override,
    attach_stdcp_local_predictor,
    _predict_batch,
)
from scores import absolute_residual_score, make_quantile_seed

# Import data generators from the patched module (patched by run_capture before workers start)
import code.shared.dgp.experiments as exp_mod


def _interval_covered(interval, y) -> bool:
    lo, hi = interval
    # Match code.shared.dgp.experiments: infinite bounds still count as covered.
    return bool(lo <= y <= hi)


def _interval_width(interval) -> float:
    lo, hi = interval
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.nan


def run_one_experiment_capture(
    *,
    experiment: int,
    config: dict,
    store,
) -> pd.DataFrame:
    """One DGP replicate: fit RF, store all μ_global, return plot-ready result rows."""
    t0 = time.perf_counter()
    number_groups_k = int(config["number_groups_k"])
    lambda_poisson = config.get("lambda_poisson", 21)
    dgp_specification = {
        "dimension": config["dimension"],
        "u_min": config["u_min"],
        "u_max": config["u_max"],
    }
    o_values = list(config["o_values"])
    target_index = int(config["target_index"])
    alpha = float(config["alpha"])
    alpha_sel = float(config.get("alpha_selection", 0.5))
    n_rep = int(config.get("n_repeated", 50))
    quantile_mode = str(config.get("quantile_mode", "deterministic"))
    quantile_base_seed = int(config.get("quantile_base_seed", config.get("seed", 457)))
    predictor = str(config.get("predictor", "rf")).lower()
    score_type = str(config.get("score_type", "absolute")).lower()
    within_group_mode = str(config.get("within_group_mode", "mean")).lower()
    rf_ntree = int(config.get("rf_ntree", 50))
    rf_nodesize = int(config.get("rf_nodesize", 5))
    rf_random_state = int(config.get("rf_random_state", 123))
    rf_mtry = config.get("rf_mtry", None)
    if rf_mtry is not None and str(rf_mtry).strip() not in ("", "none", "None"):
        # float fraction (e.g. 0.6) or int count; sklearn accepts both via max_features
        rf_mtry = float(rf_mtry) if "." in str(rf_mtry) else int(rf_mtry)
    else:
        rf_mtry = None

    def _attach_score_meta(mu: dict) -> dict:
        out = dict(mu)
        out["score_type"] = score_type
        out["rf_ntree"] = rf_ntree
        out["rf_nodesize"] = rf_nodesize
        out["rf_mtry"] = rf_mtry
        out["rf_random_state"] = rf_random_state
        out["within_group_mode"] = within_group_mode
        return out

    def _make_mu_pair():
        if predictor in ("ud_squared", "u_d", "ud", "u_d2", "ud2"):
            return (
                _attach_score_meta(create_mu_method_ud_squared_global_only()),
                _attach_score_meta(create_mu_method_ud_squared_offset()),
            )
        if predictor in ("bayes", "bayes_joint_xy", "oracle_bayes"):
            rho = float(config.get("rho", 0.5))
            return (
                _attach_score_meta(create_mu_method_bayes_joint_xy_global_only(rho=rho)),
                _attach_score_meta(create_mu_method_bayes_joint_xy_offset(rho=rho)),
            )
        mu_global = create_mu_method_random_forest_global_only(
            ntree=rf_ntree, nodesize=rf_nodesize, mtry=rf_mtry, random_state=rf_random_state,
        )
        if within_group_mode in ("local_rf", "rf", "local-rf"):
            mu_hcp = create_mu_method_random_forest_local_rf_offset(
                ntree=rf_ntree, nodesize=rf_nodesize, mtry=rf_mtry, random_state=rf_random_state,
            )
        else:
            mu_hcp = create_mu_method_random_forest_offset(
                ntree=rf_ntree, nodesize=rf_nodesize, mtry=rf_mtry, random_state=rf_random_state,
            )
        return (_attach_score_meta(mu_global), _attach_score_meta(mu_hcp))

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    cal = exp_mod.generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_poisson,
        dgp_specification=dgp_specification,
    )
    U_cal = cal["U_calibration"]
    Z_cal = cal["Z_calibration"]

    test = exp_mod.generate_test_group(
        lambda_Poisson=lambda_poisson,
        dgp_specification=dgp_specification,
        o_observed=target_index,
    )
    U_test = test["U_test"]
    Z_test = test["Z_test"]
    N_test = int(test["N_test"])
    if N_test < target_index + 1:
        raise RuntimeError(f"Test group too small: N={N_test}, need >= {target_index + 1}")

    capture_mu = bool(config.get("capture_mu", True))

    if capture_mu:
        store.write_experiment_design(experiment=experiment, n_calib=number_groups_k, n_test=1)
        for g in range(number_groups_k):
            store.write_observations(
                experiment=experiment,
                group_id=g,
                role="calib",
                calib_ord=g,
                U_group=U_cal[g, :],
                Z_group=Z_cal[g],
            )
        store.write_observations(
            experiment=experiment,
            group_id=number_groups_k,
            role="test",
            calib_ord=-1,
            U_group=U_test[0, :],
            Z_group=Z_test,
        )

    # ------------------------------------------------------------------
    # 2. Baseline RF (global only) on HCP train split
    # ------------------------------------------------------------------
    sample_sizes = [len(zg) for zg in Z_cal]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel,
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, "tolist") else list(calib_idx)

    mu_baseline, mu_hcp_template = _make_mu_pair()
    model_baseline = mu_baseline["fit_global"](
        U_matrix=U_cal, Z_list=Z_cal, group_index_vector=train_idx,
    )

    # Store μ_RF for ALL calib groups + test under baseline fit (capture only)
    if capture_mu:
        baseline_mu_rows = []
        for g in range(number_groups_k):
            Xg = np.array([z["X"] for z in Z_cal[g]], dtype=float)
            mu_batch = _predict_batch(model_baseline, Xg, U_cal[g, :])
            for row_idx, mu_val in enumerate(mu_batch):
                baseline_mu_rows.append({
                    "experiment": int(experiment),
                    "group_id": int(g),
                    "row_idx": int(row_idx),
                    "role": "calib",
                    "mu_global": float(mu_val),
                })
        X_test_all = np.array([z["X"] for z in Z_test], dtype=float)
        mu_test_batch = _predict_batch(model_baseline, X_test_all, U_test[0, :])
        for row_idx, mu_val in enumerate(mu_test_batch):
            baseline_mu_rows.append({
                "experiment": int(experiment),
                "group_id": int(number_groups_k),
                "row_idx": int(row_idx),
                "role": "test",
                "mu_global": float(mu_val),
            })
        store.write_baseline_fit(
            experiment=experiment,
            train_slots=train_idx,
            calib_slots=calib_idx,
            mu_global_rows=baseline_mu_rows,
        )

    scores_list = []
    for j in calib_idx:
        yj = np.array([z["Y"] for z in Z_cal[j]], dtype=float)
        Xj = np.array([z["X"] for z in Z_cal[j]], dtype=float)
        muj = _predict_batch(model_baseline, Xj, U_cal[j, :])
        scores_list.append(absolute_residual_score(yj, muj))

    T_hcp = compute_hcp_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment, target_index, 0, "hcp"),
    )
    T_pool = compute_pooling_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment, target_index, 0, "pool"),
    )
    T_sub = compute_subsampling_once_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment, target_index, 0, "sub"),
    )
    T_rep = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, n_rep, quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment, target_index, 0, "rep"),
    )

    true_target = float(Z_test[target_index]["Y"])
    X_target = Z_test[target_index]["X"]
    mu_baseline_hat = float(mu_baseline["predict_global"](model_baseline, X_target, U_test[0, :]))

    def _baseline_interval(T):
        if not np.isfinite(T):
            return (-np.inf, np.inf)
        return (mu_baseline_hat - T, mu_baseline_hat + T)

    base_intervals = {
        "hcp": _baseline_interval(T_hcp),
        "pool": _baseline_interval(T_pool),
        "sub": _baseline_interval(T_sub),
        "rep": _baseline_interval(T_rep),
    }

    # ------------------------------------------------------------------
    # 3. GHCP / S-HCP with mean within-group (capture every global fit)
    # ------------------------------------------------------------------
    mu_hcp = mu_hcp_template
    w_g_ov = config.get("w_g_override", None)
    if w_g_ov is not None and str(w_g_ov).strip() not in ("", "none", "None"):
        mu_hcp = set_w_g_override(mu_hcp, float(w_g_ov))
    _, mu_hcp_for_no_within = _make_mu_pair()
    mu_hcp_no_within = without_within_group_training(mu_hcp_for_no_within)
    include_ghcp_local_rf = bool(config.get("include_ghcp_local_rf", True))
    # GHCP-local RF: local RF within-group with w_g=0; inject Std-CP test RF when available.
    mu_hcp_local = None
    if include_ghcp_local_rf and predictor == "rf":
        mu_hcp_local = _attach_score_meta(
            create_mu_method_random_forest_local_rf_offset(
                ntree=rf_ntree, nodesize=rf_nodesize, mtry=rf_mtry, random_state=rf_random_state,
            )
        )
        mu_hcp_local = set_w_g_override(mu_hcp_local, 0.0)
    # Full group list for μ capture (calib slots 0..K-1, test slot K)
    U_all = np.vstack([U_cal, U_test])
    Z_all = list(Z_cal) + [Z_test]

    def _maybe_wrap(mu_base, store, ctx):
        if not capture_mu:
            return mu_base
        return wrap_mu_for_capture(mu_base, store=store, ctx=ctx)

    def _maybe_drain(mu_wrapped):
        if not capture_mu:
            return []
        return drain_pred_log(mu_wrapped)

    rows = []
    for o in o_values:
        # Donor-HCP (mean within-group)
        call_id_dr = f"donorhcp_o{o}"
        ctx_dr = CaptureCallContext(
            experiment=experiment, call_id=call_id_dr, hcp_method="Donor-HCP", o=int(o),
            U_all=U_all, Z_all=Z_all,
        )
        mu_dr = _maybe_wrap(mu_hcp, store, ctx_dr)
        try:
            res_dr = compute_donor_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_dr,
                test_index_target=target_index,
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, experiment, target_index, o, "donor_hcp",
                ),
            )
            int_dr = res_dr["interval"]
            mu_hat_dr = float(res_dr.get("mu_hat", np.nan))
        except Exception:
            int_dr = (-np.inf, np.inf)
            mu_hat_dr = np.nan
        pred_rows = _maybe_drain(mu_dr)
        if pred_rows:
            store.write_mu_pred(pred_rows)
        if capture_mu:
            store.write_hcp_call({
                "experiment": int(experiment),
                "call_id": call_id_dr,
                "hcp_method": "Donor-HCP",
                "o": int(o),
                "target_row_idx": int(target_index),
                "true_y": true_target,
                "mu_hat": mu_hat_dr,
                "interval_lower": float(int_dr[0]),
                "interval_upper": float(int_dr[1]),
                "covered": float(_interval_covered(int_dr, true_target)),
                "width": _interval_width(int_dr),
                "within_group_mode": "mean",
            })

        # Donor-HCP no within: same RF (same seeds → same S_comp), tau_override=0.
        # Re-run with capture so fit_id is stored; RF is refit once more here but
        # apply can use either fit. Prefer a dedicated call_id for no_within.
        call_id_nw = f"donorhcp_nowithin_o{o}"
        ctx_nw = CaptureCallContext(
            experiment=experiment, call_id=call_id_nw, hcp_method="Donor-HCP-no-within", o=int(o),
            U_all=U_all, Z_all=Z_all,
        )
        mu_nw = _maybe_wrap(mu_hcp_no_within, store, ctx_nw)
        try:
            res_nw = compute_donor_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_nw,
                test_index_target=target_index,
                tau_override=0,
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, experiment, target_index, o, "donor_hcp_no_within",
                ),
            )
            int_nw = res_nw["interval"]
            mu_hat_nw = float(res_nw.get("mu_hat", np.nan))
        except Exception:
            int_nw = (-np.inf, np.inf)
            mu_hat_nw = np.nan
        pred_rows = _maybe_drain(mu_nw)
        if pred_rows:
            store.write_mu_pred(pred_rows)
        if capture_mu:
            store.write_hcp_call({
                "experiment": int(experiment),
                "call_id": call_id_nw,
                "hcp_method": "Donor-HCP-no-within",
                "o": int(o),
                "target_row_idx": int(target_index),
                "true_y": true_target,
                "mu_hat": mu_hat_nw,
                "interval_lower": float(int_nw[0]),
                "interval_upper": float(int_nw[1]),
                "covered": float(_interval_covered(int_nw, true_target)),
                "width": _interval_width(int_nw),
                "within_group_mode": "none",
            })

        # Sample-HCP (mean within-group)
        call_id_sr = f"shcp_o{o}"
        ctx_sr = CaptureCallContext(
            experiment=experiment, call_id=call_id_sr, hcp_method="S-HCP", o=int(o),
            U_all=U_all, Z_all=Z_all,
        )
        mu_sr = _maybe_wrap(mu_hcp, store, ctx_sr)
        try:
            res_sr = compute_sample_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_sr,
                test_index_target=target_index,
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, experiment, target_index, o, "sample_hcp",
                ),
            )
            int_sr = res_sr["interval"]
            mu_hat_sr = float(res_sr.get("mu_hat", np.nan))
        except Exception:
            int_sr = (-np.inf, np.inf)
            mu_hat_sr = np.nan
        pred_rows = _maybe_drain(mu_sr)
        if pred_rows:
            store.write_mu_pred(pred_rows)
        if capture_mu:
            store.write_hcp_call({
                "experiment": int(experiment),
                "call_id": call_id_sr,
                "hcp_method": "S-HCP",
                "o": int(o),
                "target_row_idx": int(target_index),
                "true_y": true_target,
                "mu_hat": mu_hat_sr,
                "interval_lower": float(int_sr[0]),
                "interval_upper": float(int_sr[1]),
                "covered": float(_interval_covered(int_sr, true_target)),
                "width": _interval_width(int_sr),
                "within_group_mode": "mean",
            })

        # Std-CP: always local RF on target-group history (half-split), matching
        # the absolute-score protocol. Studentized fits local RF-σ on |Y-μ| too.
        # Fitted local RF(s) are reused below as GHCP-local's test-group predictor.
        stdcp_models = {"rf": None, "rf_scale": None}
        if o > 0:
            x_hist = np.array([Z_test[i]["X"] for i in range(o)])
            y_hist = np.array([Z_test[i]["Y"] for i in range(o)])
            stdcp_split_seed = make_quantile_seed(
                quantile_base_seed, experiment, target_index, o, "stdcp_split",
            )
            int_std, stdcp_models = _compute_std_cp_interval(
                x_hist=x_hist,
                y_hist=y_hist,
                x_target=X_target,
                alpha=alpha,
                quantile_mode=quantile_mode,
                random_seed=make_quantile_seed(
                    quantile_base_seed, experiment, target_index, o, "stdcp",
                ),
                rng=np.random.default_rng(stdcp_split_seed),
                ntree=rf_ntree,
                nodesize=rf_nodesize,
                mtry=rf_mtry,
                rf_random_state=rf_random_state,
                score_type=score_type,
                return_models=True,
            )
        else:
            int_std = (-np.inf, np.inf)

        # GHCP-local RF (optional): w_g=0 local RF; inject Std-CP's test RF.
        if mu_hcp_local is not None:
            call_id_loc = f"donorhcp_local_o{o}"
            ctx_loc = CaptureCallContext(
                experiment=experiment, call_id=call_id_loc, hcp_method="Donor-HCP-local", o=int(o),
                U_all=U_all, Z_all=Z_all,
            )
            mu_loc = attach_stdcp_local_predictor(
                mu_hcp_local,
                Z_test_list=Z_test,
                local_rf=stdcp_models.get("rf"),
                local_rf_scale=stdcp_models.get("rf_scale"),
            )
            mu_loc = set_w_g_override(mu_loc, 0.0)
            mu_loc = _maybe_wrap(mu_loc, store, ctx_loc)
            try:
                res_loc = compute_donor_hcp_randomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=o,
                    alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_loc,
                    test_index_target=target_index,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=make_quantile_seed(
                        quantile_base_seed, experiment, target_index, o, "donor_hcp_local",
                    ),
                )
                int_loc = res_loc["interval"]
                mu_hat_loc = float(res_loc.get("mu_hat", np.nan))
            except Exception:
                int_loc = (-np.inf, np.inf)
                mu_hat_loc = np.nan
            pred_rows = _maybe_drain(mu_loc)
            if pred_rows:
                store.write_mu_pred(pred_rows)
            if capture_mu:
                store.write_hcp_call({
                    "experiment": int(experiment),
                    "call_id": call_id_loc,
                    "hcp_method": "Donor-HCP-local",
                    "o": int(o),
                    "target_row_idx": int(target_index),
                    "true_y": true_target,
                    "mu_hat": mu_hat_loc,
                    "interval_lower": float(int_loc[0]),
                    "interval_upper": float(int_loc[1]),
                    "covered": float(_interval_covered(int_loc, true_target)),
                    "width": _interval_width(int_loc),
                    "within_group_mode": "local_rf_wg0",
                })
        else:
            int_loc = (np.nan, np.nan)

        # Plot-ready row (matches run_experiments_outer column names)
        row = {
            "experiment": int(experiment),
            "o_observed": int(o),
            "quantile_mode": quantile_mode,
            "coverage_donor_hcp_randomized": float(_interval_covered(int_dr, true_target)),
            "coverage_donor_hcp_no_within": float(_interval_covered(int_nw, true_target)),
            "coverage_donor_hcp_local": (
                float(_interval_covered(int_loc, true_target))
                if np.isfinite(int_loc[0]) or int_loc[0] == -np.inf
                else np.nan
            ),
            "coverage_donor_hcp_derandomized": np.nan,
            "coverage_sample_hcp_randomized": float(_interval_covered(int_sr, true_target)),
            "coverage_sample_hcp_derandomized": np.nan,
            "coverage_stdcp": float(_interval_covered(int_std, true_target)),
            "coverage_hcp": float(_interval_covered(base_intervals["hcp"], true_target)),
            "coverage_pool": float(_interval_covered(base_intervals["pool"], true_target)),
            "coverage_sub": float(_interval_covered(base_intervals["sub"], true_target)),
            "coverage_rep": float(_interval_covered(base_intervals["rep"], true_target)),
            "width_donor_hcp_randomized": _interval_width(int_dr),
            "width_donor_hcp_no_within": _interval_width(int_nw),
            "width_donor_hcp_local": _interval_width(int_loc) if np.isfinite(int_loc[0]) or int_loc[0] == -np.inf else np.nan,
            "width_donor_hcp_derandomized": np.nan,
            "width_sample_hcp_randomized": _interval_width(int_sr),
            "width_sample_hcp_derandomized": np.nan,
            "width_stdcp": _interval_width(int_std),
            "width_hcp": _interval_width(base_intervals["hcp"]),
            "width_pool": _interval_width(base_intervals["pool"]),
            "width_sub": _interval_width(base_intervals["sub"]),
            "width_rep": _interval_width(base_intervals["rep"]),
            "infinite_donor_hcp_randomized": int(not np.isfinite(int_dr[0])),
            "infinite_donor_hcp_no_within": int(not np.isfinite(int_nw[0])),
            "infinite_donor_hcp_local": (
                int(not np.isfinite(int_loc[0]))
                if not (isinstance(int_loc[0], float) and np.isnan(int_loc[0]))
                else np.nan
            ),
            "infinite_donor_hcp_derandomized": np.nan,
            "infinite_sample_hcp_randomized": int(not np.isfinite(int_sr[0])),
            "infinite_sample_hcp_derandomized": np.nan,
            "infinite_stdcp": int(not np.isfinite(int_std[0])),
            "infinite_hcp": int(not np.isfinite(base_intervals["hcp"][0])),
            "infinite_pool": int(not np.isfinite(base_intervals["pool"][0])),
            "infinite_sub": int(not np.isfinite(base_intervals["sub"][0])),
            "infinite_rep": int(not np.isfinite(base_intervals["rep"][0])),
            "elapsed_sec": time.perf_counter() - t0,
        }
        rows.append(row)

    return pd.DataFrame(rows)
