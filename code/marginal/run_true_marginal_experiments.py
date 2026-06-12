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
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import code.shared.dgp.experiments as exp_mod
from code.shared.dgp.experiments import run_experiments_outer
from code.shared.dgp.joint_xy_marginal_common import draw_group_joint_xy
from methods import create_mu_method_ols_global_only, create_mu_method_ols_offset


# Configuration
N_WORKERS = 5
BASE_SEED = 456  # Different seed from previous experiments


def alpha_to_tag(alpha):
    """Stable filename tag, e.g. 0.1 -> alpha10 and 0.075 -> alpha7p5."""
    pct = f"{100.0 * float(alpha):.6g}".replace(".", "p")
    if "p" not in pct and len(pct) < 2:
        pct = pct.zfill(2)
    return f"alpha{pct}"


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


def without_within_group_training(mu_method):
    """Return a mu_method wrapper that ignores within-group history."""
    base_predict_global = mu_method["predict_global"]
    out = dict(mu_method)

    def fit_group_adjustment(model_global, u_group_vector,
                             Z_group_list, training_index_vector):
        return 0.0

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return base_predict_global(model_global, x_vector, u_group_vector)

    out["fit_group_adjustment"] = fit_group_adjustment
    out["predict_group_mu"] = predict_group_mu
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
            target_n=config["target_n"],
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
    mu_hcp_no_within = without_within_group_training(create_mu_method_ols_offset())

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
        mu_method_hcp_no_within=mu_hcp_no_within,
        show_progress=False,
        quantile_mode=config.get("quantile_mode", "deterministic"),
        quantile_base_seed=config.get("quantile_base_seed", BASE_SEED),
    )

    results = offset_experiment_column(results, experiment_offset)
    results["worker_id"] = worker_id
    return results


def patch_fixed_size_generators(fixed_n, target_n, dimension, u_min, u_max, rho):
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
        if target_n <= o_observed:
            raise ValueError(f"Fixed target group size N={target_n} is not greater than o_observed={o_observed}.")

        if fixed_U is not None:
            u_test = np.asarray(fixed_U, dtype=float).reshape(1, -1)
        else:
            u_test = np.random.uniform(low=u_min, high=u_max, size=(1, dimension))

        z_test = draw_group_joint_xy(u_test[0, :], target_n, rho=rho)
        return {
            "U_test": u_test,
            "Z_test": z_test,
            "N_test": target_n,
        }

    exp_mod.generate_calibration_data = generate_calibration_data_fixed
    exp_mod.generate_test_group = generate_test_group_fixed


def patch_poisson_generators(lambda_poisson, dimension, u_min, u_max, rho):
    """Patch experiment module with Poisson group size generators."""

    def generate_calibration_data_poisson(number_groups, lambda_Poisson, dgp_specification):
        u_cal = np.random.uniform(
            low=u_min, high=u_max, size=(number_groups, dimension)
        )
        n_vec = 1 + np.random.poisson(lam=lambda_poisson, size=number_groups)
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

        n_test = 1 + np.random.poisson(lam=lambda_poisson)

        # Make sure we have enough observations
        while n_test <= o_observed:
            n_test = 1 + np.random.poisson(lam=lambda_poisson)

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
    out_dir = PROJECT_ROOT / "results_marginal" / "dgp" / tag
    chunk_dir = out_dir / "chunks"

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"TRUE MARGINAL COVERAGE EXPERIMENT: {config_name}")
    print("=" * 96)
    print(f"Generation mode: {config.get('generation_mode')}")
    if config.get('generation_mode') == 'fixed':
        print(f"Fixed N: {config.get('fixed_n')}")
        print(f"Fixed target N: {config.get('target_n')}")
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
    print(f"quantile_mode: {config.get('quantile_mode', 'deterministic')}")
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


def build_configs(total_replicates=1000, alpha=0.1, quantile_mode="deterministic",
                  quantile_base_seed=BASE_SEED):
    """Build true-marginal DGP configs for a given alpha."""
    o_values = [0, 5, 10, 15, 20, 25, 30, 35]
    target_index = 35

    config_fixedN21 = {
        "generation_mode": "fixed",
        "fixed_n": 21,
        "target_n": target_index + 1,
        "total_replicates": int(total_replicates),
        "number_groups_k": 20,
        "dimension": 5,
        "u_min": 1.0,
        "u_max": 5.0,
        "rho": 0.5,
        "o_values": o_values,
        "target_index": target_index,
        "alpha": float(alpha),
        "quantile_mode": quantile_mode,
        "quantile_base_seed": int(quantile_base_seed),
    }

    config_poissonNmean21 = {
        "generation_mode": "poisson",
        "lambda_poisson": 20,
        "total_replicates": int(total_replicates),
        "number_groups_k": 20,
        "dimension": 5,
        "u_min": 1.0,
        "u_max": 5.0,
        "rho": 0.5,
        "o_values": o_values,
        "target_index": target_index,
        "alpha": float(alpha),
        "quantile_mode": quantile_mode,
        "quantile_base_seed": int(quantile_base_seed),
    }

    return {
        "fixedN21": config_fixedN21,
        "poissonNmean21": config_poissonNmean21,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DGP true-marginal experiments for one or more alpha values."
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.1",
        help="Comma-separated alpha values, e.g. 0.2,0.175,0.15.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="fixedN21,poissonNmean21",
        help="Comma-separated configs: fixedN21,poissonNmean21.",
    )
    parser.add_argument("--total_replicates", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--quantile-mode",
        choices=["deterministic", "randomized"],
        default="deterministic",
        help="Conformal threshold selection: deterministic (default) or randomized.",
    )
    parser.add_argument(
        "--quantile-base-seed",
        type=int,
        default=BASE_SEED,
        help="Base seed for reproducible randomized conformal quantiles.",
    )
    return parser.parse_args()


def main():
    """Run true marginal experiments for requested DGP configurations."""
    global N_WORKERS
    args = parse_args()
    N_WORKERS = int(args.n_workers)

    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    all_configs = set(build_configs().keys())
    unknown = sorted(set(config_names) - all_configs)
    if unknown:
        raise ValueError(f"Unknown configs {unknown}. Available: {sorted(all_configs)}")

    for alpha in alphas:
        configs = build_configs(
            total_replicates=args.total_replicates,
            alpha=alpha,
            quantile_mode=args.quantile_mode,
            quantile_base_seed=args.quantile_base_seed,
        )
        alpha_tag = alpha_to_tag(alpha)
        for name in config_names:
            print("\n\n")
            print("╔" + "═" * 94 + "╗")
            title = f"CONFIGURATION: {name}, {alpha_tag}"
            print("║" + title.center(94) + "║")
            print("╚" + "═" * 94 + "╝")
            run_true_marginal_experiment(f"{name}_{alpha_tag}", configs[name])

    print("\n\n" + "=" * 96)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 96)
    print(f"Results saved to: {PROJECT_ROOT / 'results_marginal' / 'dgp'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
