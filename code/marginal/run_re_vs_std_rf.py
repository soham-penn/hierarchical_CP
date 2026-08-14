#!/usr/bin/env python3
"""GHCP: RE variance weights vs standard merger w_g = |S| / (|S| + τ).

Train RF on Strain = S_comp. Estimate (σ̂², τ̂_B²) by classical MoM on RF
residuals on S_cal (held out). No LOGO.

Methods
-------
rf_re
    w_g = (σ̂²/τ) / (τ̂_B² + σ̂²/τ)   [local weight = 1 - w_g]
rf_std
    w_g = |S_comp| / (|S_comp| + τ)  [local weight = τ / (|S_comp| + τ)]
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
    patch_fixed_size_generators,
    patch_poisson_generators,
    split_counts,
)
from code.paths import RESULTS_DGP_MARGINAL
from methods import create_mu_method_random_forest_offset
from methods.donor_hcp import compute_donor_hcp_randomized_interval
from methods.mu_methods import global_weight
from scores import make_quantile_seed

RESULTS_PREFIX = "re_vs_std_rf"
RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123
N_WORKERS_DEFAULT = 6

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
    mu_rf_re,
    mu_rf_std,
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
    test = exp_mod.generate_test_group(
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification,
        o_observed=target_index,
    )
    U_test = test["U_test"]
    Z_test = test["Z_test"]
    true_target = Z_test[target_index]["Y"]

    rows = []
    for o in o_values:
        row = {
            "experiment": experiment_id,
            "alpha": float(alpha),
            "o_observed": int(o),
            "quantile_mode": quantile_mode,
        }
        tau = int(np.floor(o / 2)) if o > 0 else 0

        # --- RE weights ---
        res_re = compute_donor_hcp_randomized_interval(
            U_calibration=U_cal,
            Z_calibration=Z_cal,
            U_test=U_test,
            Z_test=Z_test,
            o_observed=o,
            alpha=alpha,
            alpha_selection=alpha_selection,
            mu_method=mu_rf_re,
            test_index_target=target_index,
            quantile_mode=quantile_mode,
            quantile_random_seed=make_quantile_seed(
                quantile_base_seed, experiment_id, target_index, o, "donor_rf_re"
            ),
            return_intermediates=True,
        )
        lo, hi = res_re["interval"]
        w = _width(lo, hi)
        row["coverage_rf_re"] = float(lo <= true_target <= hi)
        row["width_rf_re"] = w
        row["infinite_rf_re"] = int(not np.isfinite(w))
        info = (res_re.get("intermediates") or {}).get("bayes_re") or {}
        row["re_sigma2"] = info.get("sigma2")
        row["re_tau_B2"] = info.get("tau_B2")
        row["re_w_g"] = info.get("w_g")
        row["re_w_local"] = info.get("w_local")
        row["n_train"] = info.get("n_train_groups")
        row["n_cal"] = info.get("n_cal_groups")

        # --- standard c=1: w_g = |S|/(|S|+τ), w_local = τ/(|S|+τ) ---
        res_std = compute_donor_hcp_randomized_interval(
            U_calibration=U_cal,
            Z_calibration=Z_cal,
            U_test=U_test,
            Z_test=Z_test,
            o_observed=o,
            alpha=alpha,
            alpha_selection=alpha_selection,
            mu_method=mu_rf_std,
            test_index_target=target_index,
            quantile_mode=quantile_mode,
            quantile_random_seed=make_quantile_seed(
                quantile_base_seed, experiment_id, target_index, o, "donor_rf_std"
            ),
            return_intermediates=True,
        )
        lo, hi = res_std["interval"]
        w = _width(lo, hi)
        row["coverage_rf_std"] = float(lo <= true_target <= hi)
        row["width_rf_std"] = w
        row["infinite_rf_std"] = int(not np.isfinite(w))
        n_train = int((res_std.get("intermediates") or {}).get("S_comp", np.array([])).size)
        if n_train <= 0 and row.get("n_train") is not None:
            n_train = int(row["n_train"])
        w_g_std = float(global_weight(n_train, tau, 1.0)) if tau > 0 else 1.0
        row["std_w_g"] = w_g_std
        row["std_w_local"] = float(1.0 - w_g_std)
        row["std_n_train"] = n_train
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

    mu_rf_re = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
    )
    mu_rf_re["merger"] = "bayes_re"
    mu_rf_std = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
    )

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
                mu_rf_re=mu_rf_re,
                mu_rf_std=mu_rf_std,
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
    print(f"RE vs STD (c=1): {config_name}")
    print("=" * 96)
    print(f"gamma={config['gamma']}  alpha={config['alpha']}  B={config['total_replicates']}")
    print(f"o_values={config['o_values']}")
    print("rf_re: MoM on S_cal RF residuals | rf_std: w_g=|S|/(|S|+τ)")
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
            part.to_csv(chunk_dir / f"raw_results_chunk_worker_{wid}.csv", index=False)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(run_chunk, wid, sz, off, config)
                for wid, (sz, off) in enumerate(zip(chunk_sizes, offsets))
                if sz > 0
            ]
            print(f"Submitted {len(futures)} worker jobs", flush=True)
            for i, fut in enumerate(as_completed(futures), start=1):
                part = fut.result()
                parts.append(part)
                wid = int(part["worker_id"].iloc[0])
                part.to_csv(chunk_dir / f"raw_results_chunk_worker_{wid}.csv", index=False)
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
    p.add_argument("--configs", type=str, default="fixedN21")
    p.add_argument("--total_replicates", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--n_workers", type=int, default=N_WORKERS_DEFAULT)
    p.add_argument("--o_values", type=str, default="0,5,10,15,20")
    p.add_argument("--quantile-mode", choices=["deterministic", "randomized"], default="deterministic")
    p.add_argument("--quantile-base-seed", type=int, default=BASE_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    o_values = [int(x.strip()) for x in args.o_values.split(",") if x.strip()]
    base = build_configs(
        total_replicates=args.total_replicates,
        alpha=args.alpha,
        gamma=args.gamma,
        quantile_mode=args.quantile_mode,
        quantile_base_seed=args.quantile_base_seed,
    )
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        cfg = dict(base[name])
        cfg["o_values"] = o_values
        cfg["target_index"] = max(cfg["target_index"], max(o_values))
        if cfg.get("generation_mode") == "fixed":
            cfg["target_n"] = cfg["target_index"] + 1
        run_experiment(name, cfg, n_workers=int(args.n_workers))


if __name__ == "__main__":
    main()
