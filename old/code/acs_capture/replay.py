"""Replay GHCP / S-HCP and baselines from capture CSVs."""

from __future__ import annotations

import json
from functools import partial

import numpy as np
from multiprocessing import Pool

from code.marginal import run_acs_experiments as acs
from code.marginal.acs_capture.capture_loader import (
    CaptureTables,
    build_group_data_from_observations,
    get_mu_for_replicate,
)
from code.marginal.acs_capture.storage import CaptureStore


def _slot_to_puma(slot: int, calib_groups: list[int], test_puma: int, test_slot: int) -> int:
    if slot == test_slot:
        return int(test_puma)
    return int(calib_groups[slot])


def _z_entries(group_data: dict, puma: int) -> list[dict]:
    n = len(group_data[puma]["Y"])
    return [
        {
            "X": group_data[puma]["X"][i],
            "Y": group_data[puma]["Y"][i],
            "puma": int(puma),
            "row_idx": int(i),
        }
        for i in range(n)
    ]


def _wrap_mu_lookup(
    base_mu: dict,
    *,
    tables: CaptureTables,
    replicate: int,
    call_id: str,
    global_predictor: str,
    calib_groups: list[int],
    test_puma: int,
    test_slot: int,
    store: CaptureStore | None = None,
    fit_cache: dict[tuple[int, ...], object] | None = None,
    group_adjustment_cache: dict[tuple[int, tuple[int, ...]], object] | None = None,
) -> dict:
    out = dict(base_mu)
    orig_fit_global = base_mu["fit_global"]
    orig_fit_group_adjustment = base_mu["fit_group_adjustment"]
    fit_seq = [0]
    fits = tables.global_fits

    def fit_global(U_matrix, Z_list, group_index_vector):
        group_key = tuple(int(g) for g in group_index_vector)
        if fit_cache is not None and group_key in fit_cache:
            return fit_cache[group_key]

        fit_seq[0] += 1
        fit_id = f"r{replicate}_{call_id}_f{fit_seq[0]}"
        fit_rows = fits[fits["fit_id"] == fit_id]
        fit_row = fit_rows.iloc[0] if len(fit_rows) > 0 else None
        n_comp = int(fit_row["n_comp"]) if fit_row is not None else int(len(group_index_vector))
        model = orig_fit_global(U_matrix, Z_list, group_index_vector)
        if model is not None and store is not None and global_predictor == "rf" and fit_row is not None:
            from methods.mu_methods import _predict_batch
            comp_pumas = set(json.loads(fit_row["comp_pumas_json"]))
            mu_rows = []
            for slot, zg in enumerate(Z_list):
                if not zg:
                    continue
                puma = _slot_to_puma(slot, calib_groups, test_puma, test_slot)
                if puma in comp_pumas:
                    continue
                xg = np.array([z["X"] for z in zg], dtype=float)
                mu_batch = _predict_batch(model, xg, U_matrix[slot, :])
                for row_idx, mu_val in enumerate(mu_batch):
                    mu_rows.append({
                        "fit_id": fit_id,
                        "replicate": int(replicate),
                        "call_id": call_id,
                        "puma": int(puma),
                        "row_idx": int(row_idx),
                        "mu_global": float(mu_val),
                    })
            if mu_rows:
                store.append_mu_global(mu_rows)
        if model is not None:
            model._capture_fit_id = fit_id  # type: ignore[attr-defined]
            model._capture_n_comp = n_comp  # type: ignore[attr-defined]
            model._n_comp_groups = n_comp  # type: ignore[attr-defined]
            if fit_cache is not None:
                fit_cache[group_key] = model
        return model

    out["fit_global"] = fit_global

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        if group_adjustment_cache is None:
            return orig_fit_group_adjustment(
                model_global=model_global,
                u_group_vector=u_group_vector,
                Z_group_list=Z_group_list,
                training_index_vector=training_index_vector,
            )
        if not Z_group_list:
            return orig_fit_group_adjustment(
                model_global=model_global,
                u_group_vector=u_group_vector,
                Z_group_list=Z_group_list,
                training_index_vector=training_index_vector,
            )
        puma = int(Z_group_list[0].get("puma", -1))
        train_key = tuple(int(i) for i in training_index_vector)
        key = (puma, train_key)
        if key in group_adjustment_cache:
            return group_adjustment_cache[key]
        adj = orig_fit_group_adjustment(
            model_global=model_global,
            u_group_vector=u_group_vector,
            Z_group_list=Z_group_list,
            training_index_vector=training_index_vector,
        )
        group_adjustment_cache[key] = adj
        return adj

    out["fit_group_adjustment"] = fit_group_adjustment
    return out


def _compute_baselines_for_replicate(
    replicate: int,
    *,
    tables: CaptureTables,
    o_values: list[int],
    config: dict,
    group_data: dict | None = None,
    calib_groups: list[int] | None = None,
    test_groups: list[int] | None = None,
) -> tuple[dict, dict]:
    """Recompute HCP / Pooling / Subsampling / Repeated / Std-CP for one replicate."""
    design = tables.replicate_design[tables.replicate_design["replicate"] == replicate]
    if calib_groups is None:
        calib_groups = design[design["role"] == "calib"].sort_values("calib_ord")["puma"].astype(int).tolist()
    if test_groups is None:
        test_groups = design[design["role"] == "test"]["puma"].astype(int).tolist()
    if group_data is None:
        group_data = build_group_data_from_observations(
            tables.observations, replicate, calib_groups + test_groups,
        )

    split_rows = tables.baseline_hcp_split[tables.baseline_hcp_split["replicate"] == replicate]
    if len(split_rows) == 0:
        raise KeyError(f"Missing baseline_hcp_split for replicate {replicate}")
    split_row = split_rows.iloc[0]
    train_idx = json.loads(split_row["train_slots_json"])
    calib_idx = json.loads(split_row["calib_slots_json"])

    alpha = float(config["alpha"])
    alpha_sel = float(config.get("alpha_selection", 0.5))
    n_rep = int(config.get("n_repeated", 50))
    quantile_mode = str(config.get("quantile_mode", "deterministic"))
    quantile_base_seed = int(config.get("quantile_base_seed", config.get("seed", 456)))
    outcome_scale = str(config.get("outcome_scale", "income"))
    global_predictor = str(config.get("_global_predictor", config.get("predictor", "ols")))

    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{"X": group_data[grp]["X"][i], "Y": group_data[grp]["Y"][i]}
         for i in range(len(group_data[grp]["Y"]))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    baseline_config = dict(config)
    baseline_config["predictor"] = global_predictor
    mu_baseline = acs._make_mu_baseline(baseline_config)
    model_baseline = mu_baseline["fit_global"](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
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
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate, 0, 0, "hcp"),
    )
    T_pool = acs.compute_pooling_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate, 0, 0, "pool"),
    )
    T_sub = acs.compute_subsampling_once_interval_radius(
        scores_list, alpha, quantile_mode=quantile_mode,
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate, 0, 0, "sub"),
    )
    T_rep = acs.compute_repeated_subsampling_interval_radius(
        scores_list, alpha, n_rep, quantile_mode=quantile_mode,
        random_seed=acs.make_quantile_seed(quantile_base_seed, replicate, 0, 0, "rep"),
    )

    U_test = np.zeros((1, 1))
    baseline_result = {m: {} for m in acs.BASELINE_METHODS}
    stdcp_result = {m: {} for m in acs.STD_CP_METHODS}

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
            for method in acs.STD_CP_METHODS:
                stdcp_result[method][o] = nan_rec.copy()
            continue

        base_cov = {m: [] for m in acs.BASELINE_METHODS}
        base_wid = {m: [] for m in acs.BASELINE_METHODS}
        base_wid_inc = {m: [] for m in acs.BASELINE_METHODS}
        std_cov = {m: [] for m in acs.STD_CP_METHODS}
        std_wid = {m: [] for m in acs.STD_CP_METHODS}
        std_wid_inc = {m: [] for m in acs.STD_CP_METHODS}

        for test_group in eligible_test:
            target_index = acs.TARGET_INDEX
            x_target = group_data[test_group]["X"][target_index]
            true_y = group_data[test_group]["Y"][target_index]
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

            if o > 0:
                stdcp_rng = np.random.default_rng(
                    acs.make_quantile_seed(
                        quantile_base_seed, replicate, target_index, o, "stdcp_split",
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
                        quantile_base_seed, replicate, target_index, o, "stdcp",
                    ),
                )
            else:
                int_stdcp = (-np.inf, np.inf)

            std_cov["Std-CP"].append(acs._covered(int_stdcp, true_y))
            std_wid["Std-CP"].append(acs._width(int_stdcp))
            std_wid_inc["Std-CP"].append(
                acs._width_income_from_interval(int_stdcp, outcome_scale=outcome_scale)
            )

        for method in acs.BASELINE_METHODS:
            baseline_result[method][0] = {
                "coverage": float(np.mean(base_cov[method])),
                "width": float(np.nanmean(base_wid[method])),
                "width_income": float(np.nanmean(base_wid_inc[method])),
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

    return baseline_result, stdcp_result


def _store_baseline_arrays(
    replicate: int,
    baseline_result: dict,
    stdcp_result: dict,
    *,
    o_values: list[int],
    m_map: dict,
    rep_cov: np.ndarray,
    rep_wid: np.ndarray,
    rep_winc: np.ndarray,
) -> None:
    def _store(m_i, o_i, rec):
        rep_cov[m_i, o_i] = rec["coverage"]
        rep_wid[m_i, o_i] = rec["width"]
        rep_winc[m_i, o_i] = rec["width_income"]

    for method in acs.BASELINE_METHODS:
        if 0 in baseline_result[method]:
            _store(m_map[method], 0, baseline_result[method][0])
    for method in acs.STD_CP_METHODS:
        m_i = m_map[method]
        for o_i, o in enumerate(o_values):
            if o in stdcp_result[method]:
                _store(m_i, o_i, stdcp_result[method][o])


def _replay_single_replicate(
    replicate: int,
    *,
    tables: CaptureTables,
    o_values: list[int],
    config: dict,
    hcp_template,
    m_map: dict,
    store: CaptureStore | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    design = tables.replicate_design[tables.replicate_design["replicate"] == replicate]
    calib_groups = design[design["role"] == "calib"].sort_values("calib_ord")["puma"].astype(int).tolist()
    test_groups = design[design["role"] == "test"]["puma"].astype(int).tolist()
    all_groups = sorted(set(calib_groups + test_groups))
    group_data = build_group_data_from_observations(tables.observations, replicate, all_groups)

    single_target = bool(config.get("_single_target_per_replicate", False))
    non_stratified_sampling = bool(config.get("_non_stratified_group_sampling", False))
    base_seed = int(config.get("_target_selection_seed", config.get("seed", 456)))
    target_seed = base_seed + (replicate + 1) * 7919
    rng = np.random.default_rng(target_seed)

    if non_stratified_sampling and all_groups:
        n_calib = len(calib_groups)
        min_target_n = max(max(int(o) for o in o_values), int(acs.TARGET_INDEX)) + 1
        eligible_target = [g for g in all_groups if len(group_data[g]["Y"]) >= min_target_n]
        if eligible_target:
            target_group = int(rng.choice(np.asarray(eligible_target, dtype=int)))
            remaining = [g for g in all_groups if int(g) != target_group]
            if len(remaining) >= n_calib:
                calib_groups = sorted(int(g) for g in rng.choice(np.asarray(remaining, dtype=int), size=n_calib, replace=False))
                test_groups = [target_group]
    elif single_target and test_groups:
        test_groups = [int(rng.choice(np.asarray(test_groups, dtype=int)))]

    k_calib = len(calib_groups)
    z_calibration_full = [_z_entries(group_data, g) for g in calib_groups]
    u_calibration_full = np.zeros((k_calib, 1))
    u_test = np.zeros((1, 1))
    test_slot = k_calib

    mu_base = get_mu_for_replicate(
        replicate,
        within_group_mode=str(config["within_group_mode"]),
        global_predictor=str(config.get("_global_predictor", config.get("predictor", "rf"))),
    )
    mu_base["use_standardized_score"] = bool(config.get("_use_standardized_score", False))
    global_predictor = mu_base.get("_global_predictor", "rf")

    n_o = len(o_values)
    rep_cov = np.full((len(acs.METHODS), n_o), np.nan)
    rep_wid = np.full((len(acs.METHODS), n_o), np.nan)
    rep_winc = np.full((len(acs.METHODS), n_o), np.nan)
    hcp_acc = {m: {o: {"cov": [], "wid": [], "winc": []} for o in o_values} for m in acs.HCP_METHODS}
    inc_vals = []

    alpha = float(config["alpha"])
    alpha_sel = float(config.get("alpha_selection", 0.5))
    quantile_mode = str(config.get("quantile_mode", "deterministic"))
    quantile_base_seed = int(config.get("quantile_base_seed", config.get("seed", 456)))
    outcome_scale = str(config.get("outcome_scale", "income"))

    rep_calls = hcp_template[hcp_template["replicate"] == replicate]
    reuse_global_fit = bool(config.get("_reuse_global_fit_across_targets", True))
    reuse_group_adjustment = bool(config.get("_reuse_group_adjustment_across_targets", True))
    fit_cache_by_method_o: dict[tuple[str, int], dict[tuple[int, ...], object]] = {}
    group_adjustment_cache_by_method_o: dict[tuple[str, int], dict[tuple[int, tuple[int, ...]], object]] = {}
    for o in o_values:
        for test_group in test_groups:
            if len(group_data[test_group]["Y"]) <= max(o, acs.TARGET_INDEX):
                continue
            target_index = acs.TARGET_INDEX
            true_y = float(group_data[test_group]["Y"][target_index])
            inc_vals.append(float(group_data[test_group]["income"][target_index]))
            z_test_full = _z_entries(group_data, test_group)

            for hcp_method, compute_fn, seed_suffix in [
                ("Donor-HCP", acs.compute_donor_hcp_randomized_interval, "donor_hcp"),
                ("S-HCP", acs.compute_sample_hcp_randomized_interval, "sample_hcp"),
            ]:
                sub = rep_calls[
                    (rep_calls["hcp_method"] == hcp_method)
                    & (rep_calls["o"] == o)
                    & (rep_calls["test_puma"] == test_group)
                ]
                if len(sub) == 0 and not non_stratified_sampling:
                    continue
                if len(sub) > 0:
                    call_id = str(sub.iloc[0]["call_id"])
                else:
                    call_id = f"{hcp_method}_o{o}_t{int(test_group)}"
                fit_cache = None
                if reuse_global_fit and store is None:
                    cache_key = (hcp_method, int(o))
                    fit_cache = fit_cache_by_method_o.setdefault(cache_key, {})
                group_adjustment_cache = None
                if reuse_group_adjustment:
                    cache_key = (hcp_method, int(o))
                    group_adjustment_cache = group_adjustment_cache_by_method_o.setdefault(cache_key, {})
                mu_wrapped = _wrap_mu_lookup(
                    mu_base,
                    tables=tables,
                    replicate=replicate,
                    call_id=call_id,
                    global_predictor=global_predictor,
                    calib_groups=calib_groups,
                    test_puma=int(test_group),
                    test_slot=test_slot,
                    store=store,
                    fit_cache=fit_cache,
                    group_adjustment_cache=group_adjustment_cache,
                )
                try:
                    if hcp_method == "Donor-HCP":
                        dhcp_seed = int(config["seed"]) + (replicate + 1) * 1009 + (o + 1) * 131 + 17
                        res = compute_fn(
                            U_calibration=u_calibration_full,
                            Z_calibration=z_calibration_full,
                            U_test=u_test,
                            Z_test=z_test_full,
                            o_observed=o,
                            alpha=alpha,
                            alpha_selection=alpha_sel,
                            mu_method=mu_wrapped,
                            test_index_target=target_index,
                            tau_override=acs._tau_override(config),
                            random_seed=dhcp_seed,
                            quantile_mode=quantile_mode,
                            quantile_random_seed=acs.make_quantile_seed(
                                quantile_base_seed, replicate, target_index, o, seed_suffix,
                            ),
                        )
                    else:
                        # Keep S-HCP group-selection reproducible across target PUMAs
                        # for fixed (replicate, o), so global-fit caching is valid.
                        shcp_seed = int(config["seed"]) + (replicate + 1) * 1009 + (o + 1) * 131 + 29
                        np.random.seed(shcp_seed)
                        res = compute_fn(
                            U_calibration=u_calibration_full,
                            Z_calibration=z_calibration_full,
                            U_test=u_test,
                            Z_test=z_test_full,
                            o_observed=o,
                            alpha=alpha,
                            alpha_selection=alpha_sel,
                            mu_method=mu_wrapped,
                            test_index_target=target_index,
                            tau_override=acs._tau_override(config),
                            quantile_mode=quantile_mode,
                            quantile_random_seed=acs.make_quantile_seed(
                                quantile_base_seed, replicate, target_index, o, seed_suffix,
                            ),
                        )
                    interval = res["interval"]
                except Exception:
                    interval = (-np.inf, np.inf)

                hcp_acc[hcp_method][o]["cov"].append(acs._covered(interval, true_y))
                hcp_acc[hcp_method][o]["wid"].append(acs._width(interval))
                hcp_acc[hcp_method][o]["winc"].append(
                    acs._width_income_from_interval(interval, outcome_scale=outcome_scale)
                )

    income_target = float(np.nanmean(inc_vals)) if inc_vals else np.nan
    for method in acs.HCP_METHODS:
        m_i = m_map[method]
        for o_i, o in enumerate(o_values):
            bucket = hcp_acc[method][o]["cov"]
            if not bucket:
                continue
            rep_cov[m_i, o_i] = float(np.mean(bucket))
            rep_wid[m_i, o_i] = float(np.nanmean(hcp_acc[method][o]["wid"]))
            rep_winc[m_i, o_i] = float(np.nanmean(hcp_acc[method][o]["winc"]))

    if config.get("_recompute_baselines", True):
        baseline_result, stdcp_result = _compute_baselines_for_replicate(
            replicate,
            tables=tables,
            o_values=o_values,
            config=config,
            group_data=group_data,
            calib_groups=calib_groups,
            test_groups=test_groups,
        )
        _store_baseline_arrays(
            replicate,
            baseline_result,
            stdcp_result,
            o_values=o_values,
            m_map=m_map,
            rep_cov=rep_cov,
            rep_wid=rep_wid,
            rep_winc=rep_winc,
        )

    return replicate, rep_cov, rep_wid, rep_winc, income_target


def replay_hcp_from_capture(
    *,
    tables: CaptureTables,
    o_values: list[int],
    config: dict,
    n_workers: int = 1,
    store: CaptureStore | None = None,
) -> dict:
    """Re-aggregate replicate results; re-run HCP calls using CSV-backed global mu."""
    B = tables.n_replicates
    n_o = len(o_values)
    cov = np.full((B, len(acs.METHODS), n_o), np.nan)
    wid = np.full((B, len(acs.METHODS), n_o), np.nan)
    wid_income = np.full((B, len(acs.METHODS), n_o), np.nan)
    income_targets = np.full(B, np.nan)
    m_map = {m: i for i, m in enumerate(acs.METHODS)}

    hcp_template = tables.hcp_calls

    if n_workers <= 1:
        for replicate in range(B):
            if replicate % 25 == 0:
                print(f"  Replay replicate {replicate + 1}/{B}...", flush=True)
            rep, rep_cov, rep_wid, rep_winc, inc = _replay_single_replicate(
                replicate,
                tables=tables,
                o_values=o_values,
                config=config,
                hcp_template=hcp_template,
                m_map=m_map,
                store=store,
            )
            cov[rep] = rep_cov
            wid[rep] = rep_wid
            wid_income[rep] = rep_winc
            income_targets[rep] = inc
    else:
        print(f"  Replaying {B} replicates with {n_workers} workers...", flush=True)
        if store is not None:
            print("  Appending calib/test μ_RF rows to capture mu_global.csv...", flush=True)
        worker = partial(
            _replay_single_replicate,
            tables=tables,
            o_values=o_values,
            config=config,
            hcp_template=hcp_template,
            m_map=m_map,
            store=store,
        )
        done = 0
        with Pool(processes=n_workers) as pool:
            for rep, rep_cov, rep_wid, rep_winc, inc in pool.imap_unordered(
                worker,
                range(B),
                chunksize=max(1, B // (n_workers * 4)),
            ):
                cov[rep] = rep_cov
                wid[rep] = rep_wid
                wid_income[rep] = rep_winc
                income_targets[rep] = inc
                done += 1
                if done % 25 == 0 or done == B:
                    print(f"  Replay {done}/{B} replicates done...", flush=True)

    return {
        "methods": acs.METHODS,
        "o_values": o_values,
        "coverage": cov,
        "width": wid,
        "width_income": wid_income,
        "lower_log": cov * 0 + np.nan,
        "upper_log": cov * 0 + np.nan,
        "lower_income": cov * 0 + np.nan,
        "upper_income": cov * 0 + np.nan,
        "income_targets": income_targets,
    }


def _replay_baselines_worker(
    replicate: int,
    *,
    tables: CaptureTables,
    o_values: list[int],
    config: dict,
    m_map: dict,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    n_o = len(o_values)
    rep_cov = np.full((len(acs.METHODS), n_o), np.nan)
    rep_wid = np.full((len(acs.METHODS), n_o), np.nan)
    rep_winc = np.full((len(acs.METHODS), n_o), np.nan)
    baseline_result, stdcp_result = _compute_baselines_for_replicate(
        replicate, tables=tables, o_values=o_values, config=config,
    )
    _store_baseline_arrays(
        replicate,
        baseline_result,
        stdcp_result,
        o_values=o_values,
        m_map=m_map,
        rep_cov=rep_cov,
        rep_wid=rep_wid,
        rep_winc=rep_winc,
    )
    return replicate, rep_cov, rep_wid, rep_winc


def replay_baselines_from_capture(
    *,
    tables: CaptureTables,
    o_values: list[int],
    config: dict,
    n_workers: int = 1,
) -> dict:
    """Recompute baseline / Std-CP rows only (fast OLS path)."""
    B = tables.n_replicates
    n_o = len(o_values)
    cov = np.full((B, len(acs.METHODS), n_o), np.nan)
    wid = np.full((B, len(acs.METHODS), n_o), np.nan)
    wid_income = np.full((B, len(acs.METHODS), n_o), np.nan)
    m_map = {m: i for i, m in enumerate(acs.METHODS)}

    worker = partial(
        _replay_baselines_worker,
        tables=tables,
        o_values=o_values,
        config=config,
        m_map=m_map,
    )

    if n_workers <= 1:
        for replicate in range(B):
            if replicate % 100 == 0:
                print(f"  Baseline replay replicate {replicate + 1}/{B}...", flush=True)
            rep, rep_cov, rep_wid, rep_winc = worker(replicate)
            cov[rep] = rep_cov
            wid[rep] = rep_wid
            wid_income[rep] = rep_winc
    else:
        print(f"  Replaying baselines for {B} replicates with {n_workers} workers...", flush=True)
        done = 0
        with Pool(processes=n_workers) as pool:
            for rep, rep_cov, rep_wid, rep_winc in pool.imap_unordered(
                worker,
                range(B),
                chunksize=max(1, B // (n_workers * 4)),
            ):
                cov[rep] = rep_cov
                wid[rep] = rep_wid
                wid_income[rep] = rep_winc
                done += 1
                if done % 100 == 0 or done == B:
                    print(f"  Baseline replay {done}/{B} done...", flush=True)

    return {
        "methods": acs.METHODS,
        "o_values": o_values,
        "coverage": cov,
        "width": wid,
        "width_income": wid_income,
    }
