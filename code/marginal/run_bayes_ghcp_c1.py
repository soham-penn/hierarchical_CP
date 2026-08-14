#!/usr/bin/env python3
"""GHCP with Bayes E[Y|X,U] center, Eq.(4) weights c=1 (for oracle-vs-RF table)."""

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
    split_counts,
)
from code.paths import RESULTS_DGP_MARGINAL
from methods.donor_hcp import compute_donor_hcp_randomized_interval
from methods.mu_methods import create_mu_method_bayes_joint_xy_offset
from scores import make_quantile_seed

RESULTS_PREFIX = "true_marg_latent_bayes_c1"
N_WORKERS_DEFAULT = 6
print = functools.partial(print, flush=True)


def _width(lo, hi):
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.inf


def _run_one(*, number_groups_k, lambda_Poisson, dgp_specification, o_values,
             target_index, alpha, mu_bayes, quantile_mode, quantile_base_seed,
             experiment_id):
    cal = exp_mod.generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification,
    )
    test = exp_mod.generate_test_group(
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification,
        o_observed=target_index,
    )
    true_target = test["Z_test"][target_index]["Y"]
    rows = []
    for o in o_values:
        res = compute_donor_hcp_randomized_interval(
            U_calibration=cal["U_calibration"],
            Z_calibration=cal["Z_calibration"],
            U_test=test["U_test"],
            Z_test=test["Z_test"],
            o_observed=o,
            alpha=alpha,
            alpha_selection=0.5,
            mu_method=mu_bayes,
            test_index_target=target_index,
            quantile_mode=quantile_mode,
            quantile_random_seed=make_quantile_seed(
                quantile_base_seed, experiment_id, target_index, o, "donor_bayes_c1"
            ),
        )
        lo, hi = res["interval"]
        w = _width(lo, hi)
        rows.append({
            "experiment": experiment_id,
            "alpha": float(alpha),
            "o_observed": int(o),
            "coverage_bayes": float(lo <= true_target <= hi),
            "width_bayes": w,
            "infinite_bayes": int(not np.isfinite(w)),
        })
    return rows


def run_chunk(worker_id, number_experiments_chunk, experiment_offset, config):
    print(f"  worker {worker_id}: start ({number_experiments_chunk} reps)", flush=True)
    np.random.seed(BASE_SEED + 1000 * worker_id)
    gamma = float(config["gamma"])
    rho = float(config["rho"])
    patch_fixed_size_generators(
        fixed_n=config["fixed_n"],
        target_n=config["target_n"],
        dimension=config["dimension"],
        u_min=config["u_min"],
        u_max=config["u_max"],
        rho=rho,
        gamma=gamma,
    )
    mu_bayes = create_mu_method_bayes_joint_xy_offset(rho=rho, c=1.0)
    all_rows = []
    for e in range(number_experiments_chunk):
        all_rows.extend(_run_one(
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
            mu_bayes=mu_bayes,
            quantile_mode=config.get("quantile_mode", "deterministic"),
            quantile_base_seed=config.get("quantile_base_seed", BASE_SEED),
            experiment_id=experiment_offset + e,
        ))
        if (e + 1) % 50 == 0:
            print(f"  worker {worker_id}: {e + 1}/{number_experiments_chunk}")
    out = pd.DataFrame(all_rows)
    out["worker_id"] = worker_id
    out["gamma"] = gamma
    return out


def run_experiment(config_name, config, n_workers):
    tag = f"{RESULTS_PREFIX}_{gamma_tag(config['gamma'])}_{config_name}_{alpha_to_tag(config['alpha'])}"
    out_dir = RESULTS_DGP_MARGINAL / tag
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"Bayes GHCP c=1: {config_name} gamma={config['gamma']} B={config['total_replicates']}")
    print("=" * 80)
    sizes = split_counts(config["total_replicates"], n_workers)
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    parts = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(run_chunk, wid, sz, off, config)
                for wid, (sz, off) in enumerate(zip(sizes, offsets)) if sz > 0]
        for i, fut in enumerate(as_completed(futs), 1):
            part = fut.result()
            parts.append(part)
            wid = int(part["worker_id"].iloc[0])
            part.to_csv(chunk_dir / f"raw_results_chunk_worker_{wid}.csv", index=False)
            print(f"[{i}/{len(futs)}] worker {wid} done")
    results = pd.concat(parts, ignore_index=True).sort_values(
        ["experiment", "o_observed"]).reset_index(drop=True)
    raw = out_dir / f"{tag}_raw_results_complete.csv"
    results.to_csv(raw, index=False)
    print(f"Saved: {raw} ({len(results)} rows)")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--total_replicates", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--n_workers", type=int, default=N_WORKERS_DEFAULT)
    p.add_argument("--o_values", type=str, default="0,5,10,15,20")
    args = p.parse_args()
    o_values = [int(x) for x in args.o_values.split(",") if x.strip()]
    cfg = dict(build_configs(
        total_replicates=args.total_replicates,
        alpha=args.alpha,
        gamma=args.gamma,
    )["fixedN21"])
    cfg["o_values"] = o_values
    cfg["target_index"] = max(cfg["target_index"], max(o_values))
    cfg["target_n"] = cfg["target_index"] + 1
    run_experiment("fixedN21", cfg, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
