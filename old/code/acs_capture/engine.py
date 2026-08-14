"""Run one ACS replicate and write full CSV capture artifacts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.acs_capture.capture_mu import (
    CaptureCallContext,
    drain_pred_log,
    wrap_mu_for_capture,
)
from code.marginal.acs_capture.storage import CaptureStore
from code.marginal import run_acs_experiments as acs


def _feature_names_from_matrix(n_features: int) -> list[str]:
    return [f"f_{j}" for j in range(n_features)]


def run_one_replicate_capture(
    df,
    X,
    eligible_groups,
    strata,
    group_col,
    o_values,
    config: dict,
    replicate_idx: int,
    store: CaptureStore,
    fixed_calib_groups=None,
) -> dict | None:
    """Mirror acs.run_one_replicate_avg_over_targets but persist all intermediate data."""
    t0 = time.perf_counter()
    alpha = config["alpha"]
    alpha_sel = config.get("alpha_selection", 0.5)
    n_rep = config.get("n_repeated", 50)
    quantile_mode = config.get("quantile_mode", "deterministic")
    quantile_base_seed = config.get("quantile_base_seed", config["seed"])
    outcome_scale = str(config.get("outcome_scale", "log1p"))
    truncate_n = config.get("truncate_n", None)
    n_calib_per = config.get("n_calib_per_stratum", 4)
    o_values_rep = config.get("o_values", o_values)

    group_counts = df.groupby(group_col).size()

    if fixed_calib_groups is not None:
        calib_groups = list(fixed_calib_groups)
    else:
        selection_seed = config["seed"] + replicate_idx * 1009
        design = config.get("design", "marginal")
        try:
            if design == "marginal_uniform":
                calib_groups = acs.sample_calibration_uniform(
                    eligible_groups=eligible_groups,
                    group_counts=group_counts,
                    selection_seed=selection_seed,
                    n_calib_groups=config.get("n_puma_groups", 21),
                    o_values=o_values_rep,
                    min_calib_puma_size=config.get("min_calib_puma_size"),
                )
            else:
                calib_groups = acs.sample_calibration_fixed_per_stratum(
                    strata=strata,
                    group_counts=group_counts,
                    selection_seed=selection_seed,
                    o_values=o_values_rep,
                    n_calib_per_stratum=n_calib_per,
                    n_calib_strata=4,
                    min_calib_puma_size=config.get("min_calib_puma_size"),
                )
        except ValueError:
            return None

    test_groups = acs._test_groups_from_calib(eligible_groups, calib_groups, group_counts)
    if len(test_groups) == 0:
        return None

    store.write_replicate_design(
        replicate=replicate_idx,
        calib_groups=calib_groups,
        test_groups=test_groups,
    )

    all_groups = list(calib_groups) + test_groups
    row_rng = (
        np.random.default_rng(config["seed"] + replicate_idx * 1009 + 811)
        if bool(config.get("permute_rows", False))
        else None
    )
    group_data = acs._load_group_data_static(
        df, X, group_col, all_groups, truncate_n=truncate_n, rng=row_rng,
    )
    for puma in all_groups:
        store.write_observations(replicate=replicate_idx, puma=puma, group_data=group_data)

    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{"X": group_data[grp]["X"][i], "Y": group_data[grp]["Y"][i]}
         for i in range(len(group_data[grp]["Y"]))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = acs.get_hcp_train_cal_split(
        sample_sizes=sample_sizes, o_observed=0, alpha_selection=alpha_sel,
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, "tolist") else list(calib_idx)

    mu_baseline = acs._make_mu_baseline(config)
    model_baseline = mu_baseline["fit_global"](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
    )

    baseline_mu_rows = []
    for j in calib_idx:
        puma = int(calib_groups[j])
        for row_idx, z in enumerate(Z_calibration_full[j]):
            mu_val = mu_baseline["predict_global"](
                model_baseline, z["X"], U_calibration_full[j],
            )
            baseline_mu_rows.append({
                "replicate": int(replicate_idx),
                "puma": puma,
                "row_idx": int(row_idx),
                "calib_slot": int(j),
                "mu_global": float(mu_val),
            })

    ols_coef = None
    ols_intercept = None
    if config.get("predictor") == "ols" and model_baseline is not None:
        ols_coef = np.asarray(model_baseline.coef_, dtype=float)
        ols_intercept = float(model_baseline.intercept_)

    store.write_baseline_fit(
        replicate=replicate_idx,
        predictor=str(config.get("predictor", "rf")),
        train_slots=train_idx,
        calib_slots=calib_idx,
        calib_pumas=calib_groups,
        mu_global_rows=baseline_mu_rows,
        ols_coef=ols_coef,
        ols_intercept=ols_intercept,
    )

    scores_list = []
    for j in calib_idx:
        Zj = Z_calibration_full[j]
        yj = np.array([z["Y"] for z in Zj])
        muj = np.array([
            mu_baseline["predict_global"](model_baseline, z["X"], U_calibration_full[j])
            for z in Zj
        ])
        scores_list.append(acs.absolute_residual_score(yj, muj))

    T_hcp = acs.compute_hcp_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "hcp"),
    )
    T_pool = acs.compute_pooling_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "pool"),
    )
    T_sub = acs.compute_subsampling_once_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "sub"),
    )
    T_rep = acs.compute_repeated_subsampling_interval_radius(
        scores_list, alpha, n_rep, quantile_mode=quantile_mode,
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate_idx, 0, 0, "rep"),
    )

    U_test = np.zeros((1, 1))
    baseline_result = {m: {} for m in acs.BASELINE_METHODS}
    stdcp_result = {m: {} for m in acs.STD_CP_METHODS}
    hcp_result = {m: {} for m in acs.HCP_METHODS}
    income_vals = []
    mu_hcp_base = acs._make_mu_hcp(config)

    for o in o_values:
        eligible_test = [
            grp for grp in test_groups
            if len(group_data[grp]["Y"]) > max(o, acs.TARGET_INDEX)
        ]
        if len(eligible_test) == 0:
            nan_rec = {
                "coverage": np.nan, "width": np.nan, "width_income": np.nan,
                "lower": np.nan, "upper": np.nan, "lower_income": np.nan, "upper_income": np.nan,
            }
            for method in acs.BASELINE_METHODS:
                baseline_result[method][0] = nan_rec.copy()
            for method in acs.HCP_METHODS:
                hcp_result[method][o] = nan_rec.copy()
            for method in acs.STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
            continue

        base_cov = {m: [] for m in acs.BASELINE_METHODS}
        base_wid = {m: [] for m in acs.BASELINE_METHODS}
        base_wid_inc = {m: [] for m in acs.BASELINE_METHODS}
        hcp_cov = {m: [] for m in acs.HCP_METHODS}
        hcp_wid = {m: [] for m in acs.HCP_METHODS}
        hcp_wid_inc = {m: [] for m in acs.HCP_METHODS}
        std_cov = {m: [] for m in acs.STD_CP_METHODS}
        std_wid = {m: [] for m in acs.STD_CP_METHODS}
        std_wid_inc = {m: [] for m in acs.STD_CP_METHODS}
        o_income_vals = []

        for test_group in eligible_test:
            target_index = acs.TARGET_INDEX
            x_target = group_data[test_group]["X"][target_index]
            true_y = group_data[test_group]["Y"][target_index]
            o_income_vals.append(float(group_data[test_group]["income"][target_index]))

            mu_hat_baseline = mu_baseline["predict_global"](model_baseline, x_target, U_test[0])
            for method, T in [
                ("HCP", T_hcp), ("Pooling", T_pool), ("Subsampling", T_sub), ("Repeated", T_rep),
            ]:
                interval = acs._interval_from_radius(mu_hat_baseline, T)
                base_cov[method].append(acs._covered(interval, true_y))
                base_wid[method].append(acs._width(interval))
                base_wid_inc[method].append(
                    acs._width_income_from_interval(interval, outcome_scale=outcome_scale)
                )

            Z_test_full = [
                {"X": group_data[test_group]["X"][i], "Y": group_data[test_group]["Y"][i]}
                for i in range(len(group_data[test_group]["Y"]))
            ]

            for hcp_method, compute_fn, seed_suffix in [
                ("Donor-HCP", acs.compute_donor_hcp_randomized_interval, "donor_hcp"),
                ("S-HCP", acs.compute_sample_hcp_randomized_interval, "sample_hcp"),
            ]:
                call_id = f"{hcp_method.replace('-', '').lower()}_o{o}_t{int(test_group)}"
                ctx = CaptureCallContext(
                    replicate=replicate_idx,
                    call_id=call_id,
                    hcp_method=hcp_method,
                    o=int(o),
                    test_puma=int(test_group),
                    calib_pumas=list(calib_groups),
                    group_data=group_data,
                )
                mu_wrapped = wrap_mu_for_capture(
                    mu_hcp_base,
                    store=store,
                    ctx=ctx,
                    n_calib=K_calib,
                    test_slot=K_calib,
                )
                try:
                    if hcp_method == "Donor-HCP":
                        dhcp_seed = (
                            config["seed"] + (replicate_idx + 1) * 1009 + (o + 1) * 131 + 17
                        )
                        res = compute_fn(
                            U_calibration=U_calibration_full,
                            Z_calibration=Z_calibration_full,
                            U_test=U_test,
                            Z_test=Z_test_full,
                            o_observed=o,
                            alpha=alpha,
                            alpha_selection=alpha_sel,
                            mu_method=mu_wrapped,
                            test_index_target=target_index,
                            tau_override=acs._tau_override(config),
                            random_seed=dhcp_seed,
                            quantile_mode=quantile_mode,
                            quantile_random_seed=acs.make_quantile_seed(
                                quantile_base_seed, replicate_idx, target_index, o, seed_suffix,
                            ),
                        )
                    else:
                        res = compute_fn(
                            U_calibration=U_calibration_full,
                            Z_calibration=Z_calibration_full,
                            U_test=U_test,
                            Z_test=Z_test_full,
                            o_observed=o,
                            alpha=alpha,
                            alpha_selection=alpha_sel,
                            mu_method=mu_wrapped,
                            test_index_target=target_index,
                            tau_override=acs._tau_override(config),
                            quantile_mode=quantile_mode,
                            quantile_random_seed=acs.make_quantile_seed(
                                quantile_base_seed, replicate_idx, target_index, o, seed_suffix,
                            ),
                        )
                    interval = res["interval"]
                    mu_hat = float(res.get("mu_hat", np.nan))
                except Exception:
                    interval = (-np.inf, np.inf)
                    mu_hat = np.nan

                pred_rows = drain_pred_log(mu_wrapped)
                if pred_rows:
                    store.write_mu_pred(pred_rows)

                store.write_hcp_call({
                    "replicate": int(replicate_idx),
                    "call_id": call_id,
                    "hcp_method": hcp_method,
                    "o": int(o),
                    "test_puma": int(test_group),
                    "target_row_idx": int(target_index),
                    "true_y": float(true_y),
                    "mu_hat": mu_hat,
                    "interval_lower": float(interval[0]),
                    "interval_upper": float(interval[1]),
                    "covered": float(acs._covered(interval, true_y)),
                    "width": float(acs._width(interval)),
                    "width_income": float(
                        acs._width_income_from_interval(interval, outcome_scale=outcome_scale)
                    ),
                })

                hcp_cov[hcp_method].append(acs._covered(interval, true_y))
                hcp_wid[hcp_method].append(acs._width(interval))
                hcp_wid_inc[hcp_method].append(
                    acs._width_income_from_interval(interval, outcome_scale=outcome_scale)
                )

            if o > 0:
                stdcp_rng = np.random.default_rng(
                    acs.make_quantile_seed(
                        quantile_base_seed, replicate_idx, target_index, o, "stdcp_split",
                    )
                )
                int_stdcp = acs._compute_std_cp_interval(
                    x_hist=group_data[test_group]["X"][:o],
                    y_hist=group_data[test_group]["Y"][:o],
                    x_target=x_target,
                    alpha=alpha,
                    rng=stdcp_rng,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=acs.make_quantile_seed(
                        quantile_base_seed, replicate_idx, target_index, o, "stdcp",
                    ),
                )
            else:
                int_stdcp = (-np.inf, np.inf)

            std_cov["Std-CP"].append(acs._covered(int_stdcp, true_y))
            std_wid["Std-CP"].append(acs._width(int_stdcp))
            std_wid_inc["Std-CP"].append(
                acs._width_income_from_interval(int_stdcp, outcome_scale=outcome_scale)
            )

        if o_income_vals:
            income_vals.extend(o_income_vals)

        for method in acs.BASELINE_METHODS:
            baseline_result[method][0] = {
                "coverage": float(np.mean(base_cov[method])),
                "width": float(np.nanmean(base_wid[method])),
                "width_income": float(np.nanmean(base_wid_inc[method])),
                "lower": np.nan, "upper": np.nan,
                "lower_income": np.nan, "upper_income": np.nan,
            }
        for method in acs.HCP_METHODS:
            hcp_result[method][o] = {
                "coverage": float(np.mean(hcp_cov[method])),
                "width": float(np.nanmean(hcp_wid[method])),
                "width_income": float(np.nanmean(hcp_wid_inc[method])),
                "lower": np.nan, "upper": np.nan,
                "lower_income": np.nan, "upper_income": np.nan,
            }
        for method in acs.STD_CP_METHODS:
            stdcp_result[method][o] = {
                "coverage": float(np.mean(std_cov[method])),
                "width": float(np.nanmean(std_wid[method])),
                "width_income": float(np.nanmean(std_wid_inc[method])),
                "lower": np.nan, "upper": np.nan,
                "lower_income": np.nan, "upper_income": np.nan,
            }

    elapsed = time.perf_counter() - t0
    return {
        "baseline": baseline_result,
        "hcp": hcp_result,
        "stdcp": stdcp_result,
        "income_target": float(np.nanmean(income_vals)) if income_vals else np.nan,
        "elapsed_sec": elapsed,
    }
