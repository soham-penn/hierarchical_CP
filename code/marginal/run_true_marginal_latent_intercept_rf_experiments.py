#!/usr/bin/env python3
"""True-marginal latent-intercept DGP (gamma) with Random Forest global predictors."""

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

from code.shared.dgp.experiments import run_experiments_outer
from code.shared.plot_engine import PAPER_ALPHA_GRID_STR
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
from methods import (
    create_mu_method_random_forest_global_only,
    create_mu_method_random_forest_offset,
)

RESULTS_PREFIX = "true_marg_latent_rf"
RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123
N_WORKERS_DEFAULT = 18

print = functools.partial(print, flush=True)


def run_chunk(worker_id, number_experiments_chunk, experiment_offset, config):
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
        )
    else:
        raise ValueError(f"Unknown generation_mode: {config.get('generation_mode')}")

    mu_baseline = create_mu_method_random_forest_global_only(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE
    )
    mu_hcp = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE
    )
    mu_hcp_no_within = without_within_group_training(
        create_mu_method_random_forest_offset(
            ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE
        )
    )

    results = run_experiments_outer(
        number_experiments=number_experiments_chunk,
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
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=1,
        mu_method_baseline=mu_baseline,
        mu_method_hcp=mu_hcp,
        mu_method_hcp_no_within=mu_hcp_no_within,
        show_progress=True,
        quantile_mode=config.get("quantile_mode", "deterministic"),
        quantile_base_seed=config.get("quantile_base_seed", BASE_SEED),
    )

    results = offset_experiment_column(results, experiment_offset)
    results["worker_id"] = worker_id
    results["gamma"] = gamma
    return results


def run_experiment(config_name: str, config: dict, n_workers: int) -> pd.DataFrame:
    tag = f"{RESULTS_PREFIX}_{gamma_tag(config['gamma'])}_{config_name}"
    out_dir = PROJECT_ROOT / "results_marginal" / "dgp" / tag
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"TRUE MARGINAL LATENT RF: {config_name}")
    print("=" * 96)
    print(f"gamma: {config['gamma']}  |  RF ntree={RF_NTREE}, nodesize={RF_NODESIZE}")
    print(f"Generation mode: {config.get('generation_mode')}")
    print(f"Workers: {n_workers}, replicates: {config['total_replicates']}, alpha: {config['alpha']}")
    print("=" * 96)

    chunk_sizes = split_counts(config["total_replicates"], n_workers)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()
    results_parts = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        for worker_id, (chunk_size, offset) in enumerate(zip(chunk_sizes, offsets)):
            if chunk_size == 0:
                continue
            futures.append(ex.submit(run_chunk, worker_id, chunk_size, offset, config))

        for i, fut in enumerate(as_completed(futures), start=1):
            part = fut.result()
            results_parts.append(part)
            worker_id = int(part["worker_id"].iloc[0])
            chunk_path = chunk_dir / f"raw_results_chunk_worker_{worker_id}.csv"
            part.to_csv(chunk_path, index=False)
            print(f"[{i}/{len(futures)}] worker {worker_id} done ({len(part)} rows)")

    results = pd.concat(results_parts, ignore_index=True)
    sort_cols = [c for c in ["experiment", "experiment_id", "o_observed", "worker_id"] if c in results.columns]
    if sort_cols:
        results = results.sort_values(sort_cols).reset_index(drop=True)

    raw_csv = out_dir / f"{tag}_raw_results_complete.csv"
    results.to_csv(raw_csv, index=False)
    print(f"Saved: {raw_csv} ({len(results)} rows)")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="True-marginal latent-intercept DGP with RF global predictors."
    )
    parser.add_argument("--alphas", type=str, default=PAPER_ALPHA_GRID_STR)
    parser.add_argument("--configs", type=str, default="fixedN21,poissonNmean21")
    parser.add_argument("--total_replicates", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--n_workers", type=int, default=N_WORKERS_DEFAULT)
    parser.add_argument(
        "--quantile-mode",
        choices=["deterministic", "randomized"],
        default="deterministic",
    )
    parser.add_argument("--quantile-base-seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    for alpha in alphas:
        configs = build_configs(
            total_replicates=args.total_replicates,
            alpha=alpha,
            gamma=args.gamma,
            quantile_mode=args.quantile_mode,
            quantile_base_seed=args.quantile_base_seed,
        )
        alpha_tag = alpha_to_tag(alpha)
        for name in config_names:
            run_experiment(f"{name}_{alpha_tag}", configs[name], args.n_workers)


if __name__ == "__main__":
    main()
