#!/usr/bin/env python3
"""Lean DGP RF experiment: compare GHCP merger rules (no Std-CP / sample / derand).

Mergers for GHCP+WGT:
  - eq4     : c=1  (paper Eq. 4)  w_g = |Strain| / (|Strain| + τ)
  - sqrt    : c=0.5 (legacy)      w_g = |Strain|^{1/2} / (|Strain|^{1/2} + τ)
  - bayes_re: empirical-Bayes RE  w_g = (σ̂²/τ) / (τ̂_B² + σ̂²/τ)
              with (σ̂², τ̂_B²) MoM from Strain residuals

Also reports GHCP no-WGT and HCP baseline.
Default: α=0.1, B=1000, fixedN21 + poisson, full o grid.
"""

from __future__ import annotations

import argparse
import functools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import code.shared.dgp.experiments as exp_mod
from code.marginal.run_true_marginal_latent_intercept_experiments import (
    BASE_SEED,
    DEFAULT_GAMMA,
    alpha_to_tag,
    build_configs,
    gamma_tag,
    offset_experiment_column,
    patch_fixed_size_generators,
    patch_poisson_generators,
    split_counts,
    without_within_group_training,
)
from code.paths import RESULTS_DGP_MARGINAL
from methods import (
    create_mu_method_random_forest_global_only,
    create_mu_method_random_forest_offset,
)
from methods.baseline_hcp import compute_hcp_interval_radius
from methods.donor_hcp import compute_donor_hcp_randomized_interval, get_hcp_train_cal_split
from scores import absolute_residual_score, make_quantile_seed

RESULTS_PREFIX = "merger_compare_rf"
RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123
N_WORKERS_DEFAULT = 12

print = functools.partial(print, flush=True)


def _width(lo, hi):
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.inf


def _run_one(
    *,
    number_groups_k,
    lambda_Poisson,
    dgp_specification,
    o_values,
    target_index,
    alpha,
    alpha_selection,
    mu_baseline,
    mu_variants,
    quantile_mode,
    quantile_base_seed,
    experiment_id,
):
    cal = exp_mod.generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification,
    )
    U_cal = cal["U_calibration"]
    Z_cal = cal["Z_calibration"]
    sample_sizes = [len(zg) for zg in Z_cal]
    train_idx, calib_idx = get_hcp_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_selection,
    )

    model_baseline = mu_baseline["fit_global"](
        U_matrix=U_cal, Z_list=Z_cal, group_index_vector=train_idx
    )
    scores_list = []
    for j in calib_idx:
        Zj = Z_cal[j]
        Uj = U_cal[j, :]
        yj = np.array([z["Y"] for z in Zj])
        Xj = np.array([z["X"] for z in Zj])
        muj = np.array(
            [
                mu_baseline["predict_global"](model_global=model_baseline, x_vector=Xj[i], u_vector=Uj)
                for i in range(len(Xj))
            ]
        )
        scores_list.append(absolute_residual_score(yj, muj))

    Trad = compute_hcp_interval_radius(
        scores_list,
        alpha,
        quantile_mode=quantile_mode,
        random_seed=make_quantile_seed(quantile_base_seed, experiment_id, target_index, 0, "hcp"),
    )

    test = exp_mod.generate_test_group(
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification,
        o_observed=target_index,
    )
    U_test = test["U_test"]
    Z_test = test["Z_test"]
    true_target = Z_test[target_index]["Y"]
    X_target = Z_test[target_index]["X"]
    mu_hat = mu_baseline["predict_global"](
        model_global=model_baseline, x_vector=X_target, u_vector=U_test[0, :]
    )
    lo_h = mu_hat - Trad if np.isfinite(Trad) else -np.inf
    hi_h = mu_hat + Trad if np.isfinite(Trad) else np.inf
    cov_hcp = bool(lo_h <= true_target <= hi_h)
    wid_hcp = _width(lo_h, hi_h)

    rows = []
    for o in o_values:
        row = {
            "experiment": experiment_id,
            "alpha": float(alpha),
            "o_observed": int(o),
            "quantile_mode": quantile_mode,
            "coverage_hcp": float(cov_hcp),
            "width_hcp": wid_hcp,
            "infinite_hcp": int(not np.isfinite(wid_hcp)),
        }
        for key, mu_m in mu_variants.items():
            res = compute_donor_hcp_randomized_interval(
                U_calibration=U_cal,
                Z_calibration=Z_cal,
                U_test=U_test,
                Z_test=Z_test,
                o_observed=o,
                alpha=alpha,
                alpha_selection=alpha_selection,
                mu_method=mu_m,
                test_index_target=target_index,
                tau_override=0 if key == "no_within" else None,
                quantile_mode=quantile_mode,
                quantile_random_seed=make_quantile_seed(
                    quantile_base_seed, experiment_id, target_index, o, f"donor_{key}"
                ),
                return_intermediates=(key == "bayes_re"),
            )
            lo, hi = res["interval"]
            w = _width(lo, hi)
            row[f"coverage_donor_hcp_{key}"] = float(lo <= true_target <= hi)
            row[f"width_donor_hcp_{key}"] = w
            row[f"infinite_donor_hcp_{key}"] = int(not np.isfinite(w))
            if key == "bayes_re":
                info = (res.get("intermediates") or {}).get("bayes_re") or {}
                row["bayes_sigma2"] = info.get("sigma2")
                row["bayes_tau_B2"] = info.get("tau_B2")
                row["bayes_w_g"] = info.get("w_g")
                row["bayes_tau"] = info.get("tau")
        rows.append(row)
    return rows


def run_chunk(worker_id, number_experiments_chunk, experiment_offset, config):
    print(f"  worker {worker_id}: start ({number_experiments_chunk} reps)", flush=True)
    np.random.seed(BASE_SEED + 1000 * worker_id)

    gamma = float(config["gamma"])
    rho = float(config["rho"])
    min_target_n = int(config["target_index"]) + 1

    if config.get("generation_mode") == "fixed":
        patch_fixed_size_generators(
            fixed_n=config["fixed_n"],
            target_n=config["target_n"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=rho,
            gamma=gamma,
        )
    elif config.get("generation_mode") == "poisson":
        patch_poisson_generators(
            lambda_poisson=config["lambda_poisson"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=rho,
            gamma=gamma,
            min_target_n=min_target_n,
            size_offset=int(config.get("size_offset", 1)),
        )
    else:
        raise ValueError(f"Unknown generation_mode: {config.get('generation_mode')}")

    mu_baseline = create_mu_method_random_forest_global_only(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE
    )
    mu_eq4 = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
    )
    mu_sqrt = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=0.5
    )
    mu_bayes = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
    )
    mu_bayes["merger"] = "bayes_re"
    mu_no = without_within_group_training(
        create_mu_method_random_forest_offset(
            ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
        )
    )
    mu_variants = {
        "eq4": mu_eq4,
        "sqrt": mu_sqrt,
        "bayes_re": mu_bayes,
        "no_within": mu_no,
    }

    all_rows = []
    for e in range(number_experiments_chunk):
        all_rows.extend(
            _run_one(
                number_groups_k=config["number_groups_k"],
                lambda_Poisson=config.get("lambda_poisson", 21),
                dgp_specification={
                    "dimension": config["dimension"],
                    "u_min": config["u_min"],
                    "u_max": config["u_max"],
                },
                o_values=config["o_values"],
                target_index=config["target_index"],
                alpha=config["alpha"],
                alpha_selection=0.5,
                mu_baseline=mu_baseline,
                mu_variants=mu_variants,
                quantile_mode=config.get("quantile_mode", "deterministic"),
                quantile_base_seed=config.get("quantile_base_seed", BASE_SEED),
                experiment_id=experiment_offset + e,
            )
        )
        if (e + 1) % 25 == 0:
            print(f"  worker {worker_id}: {e + 1}/{number_experiments_chunk}")

    results = pd.DataFrame(all_rows)
    results["worker_id"] = worker_id
    results["gamma"] = gamma
    return results


def run_experiment(config_name: str, config: dict, n_workers: int) -> pd.DataFrame:
    tag = f"{RESULTS_PREFIX}_{gamma_tag(config['gamma'])}_{config_name}_{alpha_to_tag(config['alpha'])}"
    out_dir = RESULTS_DGP_MARGINAL / tag
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"MERGER COMPARE RF: {config_name}")
    print("=" * 96)
    print(f"gamma={config['gamma']}  alpha={config['alpha']}  B={config['total_replicates']}")
    print(f"o_values={config['o_values']}")
    print(f"workers={n_workers}  (GHCP eq4 / sqrt / bayes_re / no_WGT + HCP; no Std-CP)")
    print("=" * 96)

    chunk_sizes = split_counts(config["total_replicates"], n_workers)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()
    parts = []
    if n_workers <= 1:
        for wid, (sz, off) in enumerate(zip(chunk_sizes, offsets)):
            if sz == 0:
                continue
            part = run_chunk(wid, sz, off, config)
            parts.append(part)
            path = chunk_dir / f"raw_results_chunk_worker_{wid}.csv"
            part.to_csv(path, index=False)
            print(f"[1/1] worker {wid} done ({len(part)} rows)")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for wid, (sz, off) in enumerate(zip(chunk_sizes, offsets)):
                if sz == 0:
                    continue
                futures.append(ex.submit(run_chunk, wid, sz, off, config))
            print(f"Submitted {len(futures)} worker jobs", flush=True)
            for i, fut in enumerate(as_completed(futures), start=1):
                part = fut.result()
                parts.append(part)
                wid = int(part["worker_id"].iloc[0])
                path = chunk_dir / f"raw_results_chunk_worker_{wid}.csv"
                part.to_csv(path, index=False)
                print(f"[{i}/{len(futures)}] worker {wid} done ({len(part)} rows)", flush=True)

    results = pd.concat(parts, ignore_index=True)
    results = results.sort_values(["experiment", "o_observed"]).reset_index(drop=True)
    raw_csv = out_dir / f"{tag}_raw_results_complete.csv"
    results.to_csv(raw_csv, index=False)
    print(f"Saved: {raw_csv} ({len(results)} rows)")
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--configs", type=str, default="fixedN21,poissonNmean25")
    p.add_argument("--total_replicates", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--n_workers", type=int, default=N_WORKERS_DEFAULT)
    p.add_argument(
        "--o_values",
        type=str,
        default="0,5,10,15,20,25,30,35",
        help="Comma-separated o grid",
    )
    p.add_argument("--quantile-mode", choices=["deterministic", "randomized"], default="deterministic")
    p.add_argument("--quantile-base-seed", type=int, default=BASE_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    o_values = [int(x.strip()) for x in args.o_values.split(",") if x.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    base = build_configs(
        total_replicates=args.total_replicates,
        alpha=args.alpha,
        gamma=args.gamma,
        quantile_mode=args.quantile_mode,
        quantile_base_seed=args.quantile_base_seed,
    )
    for name in config_names:
        if name not in base:
            raise ValueError(f"Unknown config {name}. Available: {sorted(base)}")
        cfg = dict(base[name])
        cfg["o_values"] = o_values
        # target must exceed max(o)
        cfg["target_index"] = max(cfg["target_index"], max(o_values))
        if cfg.get("generation_mode") == "fixed":
            cfg["target_n"] = cfg["target_index"] + 1
        run_experiment(name, cfg, n_workers=int(args.n_workers))


if __name__ == "__main__":
    main()
