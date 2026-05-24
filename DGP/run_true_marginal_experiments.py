#!/usr/bin/env python3
"""Run TRUE MARGINAL coverage DGP experiments with parallel processing.

Key difference from previous experiments:
- OLD: Fix calibration data, generate 100 test groups -> calibration-conditional coverage
- NEW: For each replicate, regenerate ALL data (calibration + test) -> true marginal coverage

This evaluates marginal coverage properly by regenerating the entire dataset for each replicate.

Settings:
- 1000 total replicates (instead of 50 × 100 = 5000)
- Each replicate: new calibration data + ONE test group
- Results saved with "true_marg_" prefix
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import DGP.code.experiments as exp_mod
from DGP.code.experiments import run_experiments_outer
from DGP.code.joint_xy_marginal_common import draw_group_joint_xy
from methods import create_mu_method_ols_global_only, create_mu_method_ols_offset


# Configuration
N_WORKERS = 5
BASE_SEED = 456  # Different seed from previous experiments


def split_counts(total, n_parts):
    """Split total count into n_parts as evenly as possible."""
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def offset_experiment_column(df, offset):
    """Offset experiment ID column by given amount."""
    out = df.copy()
    for col in ["experiment", "experiment_id", "exp_id"]:
        if col in out.columns:
            out[col] = out[col] + offset
            break
    return out


def run_chunk(
    worker_id,
    number_experiments_chunk,
    experiment_offset,
    config,
):
    """Run a chunk of experiments for one worker.

    Args:
        worker_id: Worker ID for seeding
        number_experiments_chunk: Number of experiments for this worker
        experiment_offset: Offset for experiment IDs
        config: Dict with experiment configuration
    """
    np.random.seed(BASE_SEED + 1000 * worker_id)

    # Patch generators if using fixed N
    if config.get("generation_mode") == "fixed":
        patch_fixed_size_generators(
            fixed_n=config["fixed_n"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=config["rho"],
        )
    elif config.get("generation_mode") == "poisson":
        patch_poisson_generators(
            lambda_poisson=config["lambda_poisson"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=config["rho"],
        )

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    # KEY CHANGE: number_test_groups = 1 (true marginal)
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
        number_test_groups=1,  # TRUE MARGINAL: 1 test group per experiment
        mu_method_baseline=mu_baseline,
        mu_method_hcp=mu_hcp,
        show_progress=False,
    )

    results = offset_experiment_column(results, experiment_offset)
    results["worker_id"] = worker_id
    return results


def patch_fixed_size_generators(fixed_n, dimension, u_min, u_max, rho):
    """Patch experiment module with fixed group size generators."""

    def generate_calibration_data_fixed(number_groups, lambda_Poisson, dgp_specification):
        u_cal = np.random.uniform(
            low=u_min, high=u_max, size=(number_groups, dimension)
        )
        n_vec = np.full(number_groups, fixed_n, dtype=int)
        z_cal = []
        for j in range(number_groups):
            z_cal.append(draw_group_joint_xy(u_cal[j, :], int(n_vec[j]), rho=rho))
        return {
            "U_calibration": u_cal,
            "Z_calibration": z_cal,
            "sample_size_vector": n_vec,
        }

    def generate_test_group_fixed(lambda_Poisson, dgp_specification, o_observed, fixed_U=None):
        if fixed_n <= o_observed:
            raise ValueError(f"Fixed group size N={fixed_n} is not greater than o_observed={o_observed}.")

        if fixed_U is not None:
            u_test = np.asarray(fixed_U, dtype=float).reshape(1, -1)
        else:
            u_test = np.random.uniform(low=u_min, high=u_max, size=(1, dimension))

        z_test = draw_group_joint_xy(u_test[0, :], fixed_n, rho=rho)
        return {
            "U_test": u_test,
            "Z_test": z_test,
            "N_test": fixed_n,
        }

    exp_mod.generate_calibration_data = generate_calibration_data_fixed
    exp_mod.generate_test_group = generate_test_group_fixed


def patch_poisson_generators(lambda_poisson, dimension, u_min, u_max, rho):
    """Patch experiment module with Poisson group size generators."""

    def generate_calibration_data_poisson(number_groups, lambda_Poisson, dgp_specification):
        u_cal = np.random.uniform(
            low=u_min, high=u_max, size=(number_groups, dimension)
        )
        n_vec = np.random.poisson(lam=lambda_poisson, size=number_groups)
        n_vec = np.maximum(n_vec, 1)  # At least 1 observation per group
        z_cal = []
        for j in range(number_groups):
            z_cal.append(draw_group_joint_xy(u_cal[j, :], int(n_vec[j]), rho=rho))
        return {
            "U_calibration": u_cal,
            "Z_calibration": z_cal,
            "sample_size_vector": n_vec,
        }

    def generate_test_group_poisson(lambda_Poisson, dgp_specification, o_observed, fixed_U=None):
        if fixed_U is not None:
            u_test = np.asarray(fixed_U, dtype=float).reshape(1, -1)
        else:
            u_test = np.random.uniform(low=u_min, high=u_max, size=(1, dimension))

        n_test = np.random.poisson(lam=lambda_poisson)
        n_test = max(1, n_test)

        # Make sure we have enough observations
        while n_test <= o_observed:
            n_test = np.random.poisson(lam=lambda_poisson)
            n_test = max(1, n_test)

        z_test = draw_group_joint_xy(u_test[0, :], n_test, rho=rho)
        return {
            "U_test": u_test,
            "Z_test": z_test,
            "N_test": n_test,
        }

    exp_mod.generate_calibration_data = generate_calibration_data_poisson
    exp_mod.generate_test_group = generate_test_group_poisson


def run_true_marginal_experiment(config_name, config):
    """Run true marginal coverage experiment with given configuration.

    Args:
        config_name: Name for output files (e.g., "fixedN21", "poissonNmean21")
        config: Dict with experiment configuration
    """
    tag = f"true_marg_{config_name}"
    out_dir = PROJECT_ROOT / "NEW_RESULTS" / tag
    chunk_dir = out_dir / "chunks"

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"TRUE MARGINAL COVERAGE EXPERIMENT: {config_name}")
    print("=" * 96)
    print(f"Generation mode: {config.get('generation_mode')}")
    if config.get('generation_mode') == 'fixed':
        print(f"Fixed N: {config.get('fixed_n')}")
    else:
        print(f"Poisson lambda: {config.get('lambda_poisson')}")
    print(f"K (calibration groups): {config['number_groups_k']}")
    print(f"Dimension: {config['dimension']}")
    print(f"U ~ Uniform[{config['u_min']}, {config['u_max']}]^d")
    print(f"rho: {config['rho']}")
    print(f"Workers: {N_WORKERS}")
    print(f"Total replicates: {config['total_replicates']}")
    print(f"o_values: {config['o_values']}")
    print(f"target_index: {config['target_index']}")
    print(f"alpha: {config['alpha']}")
    print("=" * 96)

    chunk_sizes = split_counts(config["total_replicates"], N_WORKERS)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()

    futures = []
    results_parts = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for worker_id, (chunk_size, offset) in enumerate(zip(chunk_sizes, offsets)):
            if chunk_size == 0:
                continue
            futures.append(
                ex.submit(run_chunk, worker_id, chunk_size, offset, config)
            )

        for i, fut in enumerate(as_completed(futures), start=1):
            part = fut.result()
            results_parts.append(part)

            worker_id = int(part["worker_id"].iloc[0]) if "worker_id" in part.columns else i - 1
            chunk_path = chunk_dir / f"raw_results_chunk_worker_{worker_id}.csv"
            part.to_csv(chunk_path, index=False)

            print(f"[{i}/{len(futures)}] worker chunk completed with {len(part)} rows -> {chunk_path.name}")

    # Combine results
    results = pd.concat(results_parts, ignore_index=True)

    # Sort and save
    sort_cols = [c for c in ["experiment", "experiment_id", "o_observed", "worker_id"] if c in results.columns]
    if sort_cols:
        results = results.sort_values(sort_cols).reset_index(drop=True)

    raw_csv = out_dir / f"{tag}_raw_results_complete.csv"
    results.to_csv(raw_csv, index=False)
    print(f"\nSaved complete results: {raw_csv}")
    print(f"Total rows: {len(results)}")

    return results


def main():
    """Run true marginal experiments for both fixed N and Poisson N configurations."""

    # Configuration 1: Fixed N = 21
    config_fixedN21 = {
        "generation_mode": "fixed",
        "fixed_n": 21,
        "total_replicates": 1000,
        "number_groups_k": 20,
        "dimension": 5,
        "u_min": 1.0,
        "u_max": 5.0,
        "rho": 0.5,
        "o_values": [0, 5, 10, 15, 20],
        "target_index": 20,
        "alpha": 0.1,
    }

    # Configuration 2: Poisson with mean 21
    config_poissonNmean21 = {
        "generation_mode": "poisson",
        "lambda_poisson": 21,
        "total_replicates": 1000,
        "number_groups_k": 20,
        "dimension": 5,
        "u_min": 1.0,
        "u_max": 5.0,
        "rho": 0.5,
        "o_values": [0, 5, 10, 15, 20],
        "target_index": 20,
        "alpha": 0.1,
    }

    # Run both configurations
    print("\n\n")
    print("╔" + "═" * 94 + "╗")
    print("║" + " " * 30 + "CONFIGURATION 1: Fixed N=21" + " " * 37 + "║")
    print("╚" + "═" * 94 + "╝")
    results_fixed = run_true_marginal_experiment("fixedN21", config_fixedN21)

    print("\n\n")
    print("╔" + "═" * 94 + "╗")
    print("║" + " " * 27 + "CONFIGURATION 2: Poisson N (mean=21)" + " " * 32 + "║")
    print("╚" + "═" * 94 + "╝")
    results_poisson = run_true_marginal_experiment("poissonNmean21", config_poissonNmean21)

    print("\n\n" + "=" * 96)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 96)
    print(f"Results saved to: {PROJECT_ROOT / 'NEW_RESULTS'}")
    print("  - true_marg_fixedN21/")
    print("  - true_marg_poissonNmean21/")
    print("=" * 96)


if __name__ == "__main__":
    main()
