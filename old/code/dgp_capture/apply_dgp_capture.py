#!/usr/bin/env python3
"""
Apply a within-group mode from DGP RF capture CSVs — **never refits RF**.

Uses mu_global.csv lookups only. Writes plot-ready results under
results_marginal/dgp/true_marg_latent_rf_gamma5p0_{config}_{alpha}[_{tag}]/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture.lookup_mu import (
    attach_meta_to_z,
    index_mu_global,
    make_lookup_mu_method,
)
from code.shared.dgp.experiments import _compute_std_cp_interval
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_repeated_subsampling_interval_radius,
    compute_subsampling_once_interval_radius,
)
from methods.donor_hcp import compute_donor_hcp_randomized_interval
from methods.sample_hcp import compute_sample_hcp_randomized_interval
from scores import absolute_residual_score, make_quantile_seed


def _read_config(capture_dir: Path) -> dict:
    cfg = pd.read_csv(capture_dir / "experiment_config.csv")
    out = {}
    for _, row in cfg.iterrows():
        key = row["key"]
        val = row["value"]
        try:
            out[key] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            try:
                out[key] = float(val)
            except (ValueError, TypeError):
                out[key] = val
    return out


def _load_experiment_data(obs: pd.DataFrame, experiment: int, n_calib: int):
    sub = obs[obs["experiment"] == experiment]
    x_cols = [c for c in sub.columns if c.startswith("x_")]
    u_cols = [c for c in sub.columns if c.startswith("u_")]

    Z_cal, U_cal = [], []
    for g in range(n_calib):
        gdf = sub[sub["group_id"] == g].sort_values("row_idx")
        Z_cal.append([
            {"X": row[x_cols].to_numpy(dtype=float), "Y": float(row["y"])}
            for _, row in gdf.iterrows()
        ])
        U_cal.append(gdf.iloc[0][u_cols].to_numpy(dtype=float))
    U_cal = np.asarray(U_cal, dtype=float)

    tdf = sub[sub["group_id"] == n_calib].sort_values("row_idx")
    Z_test = [
        {"X": row[x_cols].to_numpy(dtype=float), "Y": float(row["y"])}
        for _, row in tdf.iterrows()
    ]
    U_test = tdf.iloc[0][u_cols].to_numpy(dtype=float).reshape(1, -1)
    return U_cal, Z_cal, U_test, Z_test


def _interval_covered(interval, y) -> bool:
    lo, hi = interval
    # Match code.shared.dgp.experiments: infinite bounds still count as covered.
    return bool(lo <= y <= hi)


def _interval_width(interval) -> float:
    lo, hi = interval
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.nan


def _replay_experiment(
    experiment: int,
    *,
    config: dict,
    obs: pd.DataFrame,
    baseline_mu: pd.DataFrame,
    baseline_split: pd.DataFrame,
    mu_by_fit_id: dict,
    within_group_mode: str,
) -> pd.DataFrame:
    n_calib = int(config["number_groups_k"])
    o_values = [int(o) for o in config["o_values"]]
    target_index = int(config["target_index"])
    alpha = float(config["alpha"])
    alpha_sel = float(config.get("alpha_selection", 0.5))
    n_rep = int(config.get("n_repeated", 50))
    quantile_mode = str(config.get("quantile_mode", "deterministic"))
    quantile_base_seed = int(config.get("quantile_base_seed", config.get("seed", 457)))
    rf_ntree = int(config.get("rf_ntree", 50))
    rf_nodesize = int(config.get("rf_nodesize", 5))
    rf_random_state = int(config.get("rf_random_state", 123))
    rf_mtry = config.get("rf_mtry", None)
    if rf_mtry is not None and str(rf_mtry).strip() not in ("", "none", "None"):
        rf_mtry = float(rf_mtry) if "." in str(rf_mtry) else int(rf_mtry)
    else:
        rf_mtry = None

    U_cal, Z_cal, U_test, Z_test = _load_experiment_data(obs, experiment, n_calib)
    Z_cal_meta = attach_meta_to_z(Z_cal)
    Z_test_meta = attach_meta_to_z([Z_test], group_ids=[n_calib])[0]

    # Baseline from stored baseline_mu_global (no RF)
    split = baseline_split[baseline_split["experiment"] == experiment].iloc[0]
    train_idx = json.loads(split["train_slots_json"])
    calib_idx = json.loads(split["calib_slots_json"])
    bmu = baseline_mu[baseline_mu["experiment"] == experiment]
    bmu_map = {
        (int(g), int(r)): float(m)
        for g, r, m in zip(bmu["group_id"], bmu["row_idx"], bmu["mu_global"])
    }

    scores_list = []
    for j in calib_idx:
        yj = np.array([z["Y"] for z in Z_cal[j]], dtype=float)
        muj = np.array([bmu_map[(j, i)] for i in range(len(Z_cal[j]))], dtype=float)
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
    mu_baseline_hat = bmu_map[(n_calib, target_index)]

    def _base_int(T):
        if not np.isfinite(T):
            return (-np.inf, np.inf)
        return (mu_baseline_hat - T, mu_baseline_hat + T)

    base = {
        "hcp": _base_int(T_hcp),
        "pool": _base_int(T_pool),
        "sub": _base_int(T_sub),
        "rep": _base_int(T_rep),
    }

    tau_override = 0 if within_group_mode == "none" else None
    rows = []
    for o in o_values:
        # Donor-HCP
        call_id_dr = f"donorhcp_o{o}"
        mu_dr = make_lookup_mu_method(
            within_group_mode=within_group_mode,
            mu_by_fit_id=mu_by_fit_id,
            experiment=experiment,
            call_id=call_id_dr,
        )
        try:
            res_dr = compute_donor_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal_meta,
                U_test=U_test,
                Z_test=Z_test_meta,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_dr,
                test_index_target=target_index,
                tau_override=tau_override,
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, experiment, target_index, o, "donor_hcp",
                ),
            )
            int_dr = res_dr["interval"]
        except Exception:
            int_dr = (-np.inf, np.inf)

        # Donor-HCP no within (always tau=0; uses dedicated capture call_id)
        call_id_nw = f"donorhcp_nowithin_o{o}"
        mu_nw = make_lookup_mu_method(
            within_group_mode="none",
            mu_by_fit_id=mu_by_fit_id,
            experiment=experiment,
            call_id=call_id_nw,
        )
        try:
            res_nw = compute_donor_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal_meta,
                U_test=U_test,
                Z_test=Z_test_meta,
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
        except Exception:
            int_nw = (-np.inf, np.inf)

        # S-HCP
        call_id_sr = f"shcp_o{o}"
        mu_sr = make_lookup_mu_method(
            within_group_mode=within_group_mode,
            mu_by_fit_id=mu_by_fit_id,
            experiment=experiment,
            call_id=call_id_sr,
        )
        try:
            res_sr = compute_sample_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal_meta,
                U_test=U_test,
                Z_test=Z_test_meta,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_sel,
                mu_method=mu_sr,
                test_index_target=target_index,
                tau_override=tau_override,
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, experiment, target_index, o, "sample_hcp",
                ),
            )
            int_sr = res_sr["interval"]
        except Exception:
            int_sr = (-np.inf, np.inf)

        if o > 0:
            x_hist = np.array([Z_test[i]["X"] for i in range(o)])
            y_hist = np.array([Z_test[i]["Y"] for i in range(o)])
            stdcp_split_seed = make_quantile_seed(
                quantile_base_seed, experiment, target_index, o, "stdcp_split",
            )
            int_std = _compute_std_cp_interval(
                x_hist=x_hist,
                y_hist=y_hist,
                x_target=Z_test[target_index]["X"],
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
            )
        else:
            int_std = (-np.inf, np.inf)

        rows.append({
            "experiment": int(experiment),
            "o_observed": int(o),
            "quantile_mode": quantile_mode,
            "coverage_donor_hcp_randomized": float(_interval_covered(int_dr, true_target)),
            "coverage_donor_hcp_no_within": float(_interval_covered(int_nw, true_target)),
            "coverage_donor_hcp_derandomized": np.nan,
            "coverage_sample_hcp_randomized": float(_interval_covered(int_sr, true_target)),
            "coverage_sample_hcp_derandomized": np.nan,
            "coverage_stdcp": float(_interval_covered(int_std, true_target)),
            "coverage_hcp": float(_interval_covered(base["hcp"], true_target)),
            "coverage_pool": float(_interval_covered(base["pool"], true_target)),
            "coverage_sub": float(_interval_covered(base["sub"], true_target)),
            "coverage_rep": float(_interval_covered(base["rep"], true_target)),
            "width_donor_hcp_randomized": _interval_width(int_dr),
            "width_donor_hcp_no_within": _interval_width(int_nw),
            "width_donor_hcp_derandomized": np.nan,
            "width_sample_hcp_randomized": _interval_width(int_sr),
            "width_sample_hcp_derandomized": np.nan,
            "width_stdcp": _interval_width(int_std),
            "width_hcp": _interval_width(base["hcp"]),
            "width_pool": _interval_width(base["pool"]),
            "width_sub": _interval_width(base["sub"]),
            "width_rep": _interval_width(base["rep"]),
            "infinite_donor_hcp_randomized": int(not np.isfinite(int_dr[0])),
            "infinite_donor_hcp_no_within": int(not np.isfinite(int_nw[0])),
            "infinite_donor_hcp_derandomized": np.nan,
            "infinite_sample_hcp_randomized": int(not np.isfinite(int_sr[0])),
            "infinite_sample_hcp_derandomized": np.nan,
            "infinite_stdcp": int(not np.isfinite(int_std[0])),
            "infinite_hcp": int(not np.isfinite(base["hcp"][0])),
            "infinite_pool": int(not np.isfinite(base["pool"][0])),
            "infinite_sub": int(not np.isfinite(base["sub"][0])),
            "infinite_rep": int(not np.isfinite(base["rep"][0])),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Apply within-group mode from DGP RF capture.")
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument(
        "--within_group_mode",
        choices=("mean", "correction", "none"),
        required=True,
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Suffix for results dir (default: corr / none / mean).",
    )
    parser.add_argument("--max_experiments", type=int, default=None)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    config = _read_config(capture_dir)
    print(f"Loading capture from {capture_dir}", flush=True)
    obs = pd.read_csv(capture_dir / "observations.csv")
    baseline_mu = pd.read_csv(capture_dir / "baseline_mu_global.csv")
    baseline_split = pd.read_csv(capture_dir / "baseline_hcp_split.csv")
    print("Loading mu_global.csv (may be large)...", flush=True)
    mu_global = pd.read_csv(capture_dir / "mu_global.csv")
    mu_by_fit_id = index_mu_global(mu_global)
    print(f"  {len(mu_by_fit_id):,} fits indexed", flush=True)

    experiments = sorted(obs["experiment"].unique().astype(int))
    if args.max_experiments is not None:
        experiments = experiments[: int(args.max_experiments)]

    parts = []
    for i, exp_id in enumerate(experiments, start=1):
        parts.append(_replay_experiment(
            int(exp_id),
            config=config,
            obs=obs,
            baseline_mu=baseline_mu,
            baseline_split=baseline_split,
            mu_by_fit_id=mu_by_fit_id,
            within_group_mode=args.within_group_mode,
        ))
        if i <= 5 or i % 100 == 0 or i == len(experiments):
            print(f"  replayed {i}/{len(experiments)}", flush=True)

    results = pd.concat(parts, ignore_index=True)
    results["gamma"] = float(config["gamma"])

    from code.marginal.run_true_marginal_latent_intercept_experiments import (
        alpha_to_tag,
        gamma_tag,
    )
    alpha_tag = alpha_to_tag(float(config["alpha"]))
    gtag = gamma_tag(float(config["gamma"]))
    config_name = str(config.get("config_name", "unknown"))
    tag = args.output_tag
    if tag is None:
        tag = {"mean": "mean", "correction": "corr", "none": "nowithin"}[args.within_group_mode]
    suffix = "" if tag in ("", "mean") else f"_{tag}"
    results_tag = f"true_marg_latent_rf_{gtag}_{config_name}_{alpha_tag}{suffix}"
    # Default mean capture already writes without suffix; correction/none get suffix.
    if args.within_group_mode == "mean" and tag == "mean":
        results_tag = f"true_marg_latent_rf_{gtag}_{config_name}_{alpha_tag}"

    results_dir = REPO_ROOT / "results_marginal" / "dgp" / results_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"{results_tag}_raw_results_complete.csv"
    results.to_csv(out_csv, index=False)
    results.to_csv(capture_dir / f"raw_results_{args.within_group_mode}.csv", index=False)
    print(f"Wrote {out_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
