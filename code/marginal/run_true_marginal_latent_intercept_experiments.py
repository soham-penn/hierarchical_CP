#!/usr/bin/env python3
"""True-marginal DGP experiments with latent intercept B_j ~ N(0, gamma^2).

Each replicate regenerates K historical groups (each with its own B_j) and one
target group (with its own B_j). One test observation per replicate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import code.shared.dgp.experiments as exp_mod
from code.shared.dgp.experiments import run_experiments_outer
from code.shared.plot_engine import PAPER_ALPHA_GRID_STR
from methods import create_mu_method_ols_global_only, create_mu_method_ols_offset

N_WORKERS = 5
BASE_SEED = 457
DEFAULT_GAMMA = 5.0


def alpha_to_tag(alpha: float) -> str:
    pct = f"{100.0 * float(alpha):.6g}".replace(".", "p")
    if "p" not in pct and len(pct) < 2:
        pct = pct.zfill(2)
    return f"alpha{pct}"


def gamma_tag(gamma: float) -> str:
    return f"gamma{str(float(gamma)).replace('.', 'p')}"


def draw_group_latent_intercept(u_vec, n_obs, b_j, rho):
    """Draw group data with response shift B_j on the last coordinate.

    Each observation dict includes ``B`` (the group latent intercept) so oracle
    predictors that know \(B\) can recover \(E[Y\\mid X,U,B]\).
    """
    u = np.asarray(u_vec, dtype=float).ravel()
    d = len(u)
    mean = u ** 2
    mean = mean.copy()
    mean[-1] += float(b_j)
    sigma = (1.0 - rho) * np.diag(u) + rho * np.ones((d, d), dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += 1e-8 * np.eye(d)
    z = np.random.multivariate_normal(mean=mean, cov=sigma, size=int(n_obs))
    b = float(b_j)
    return [
        {"X": z[i, :-1].astype(float), "Y": float(z[i, -1]), "B": b}
        for i in range(int(n_obs))
    ]


def split_counts(total: int, n_workers: int) -> list[int]:
    base, rem = divmod(int(total), int(n_workers))
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def offset_experiment_column(df, offset):
    out = df.copy()
    for col in ("experiment", "experiment_id"):
        if col in out.columns:
            out[col] = out[col] + offset
    return out


def without_within_group_training(mu_method):
    base_predict_global = mu_method["predict_global"]
    out = dict(mu_method)

    def fit_group_adjustment(model_global, u_group_vector, Z_group_list, training_index_vector):
        return 0.0

    def predict_group_mu(model_global, group_adjustment, x_vector, u_group_vector):
        return base_predict_global(model_global, x_vector, u_group_vector)

    out["fit_group_adjustment"] = fit_group_adjustment
    out["predict_group_mu"] = predict_group_mu
    return out


def patch_fixed_size_generators(fixed_n, target_n, dimension, u_min, u_max, rho, gamma):
    def generate_calibration_data_fixed(number_groups, lambda_Poisson, dgp_specification):
        u_cal = np.random.uniform(low=u_min, high=u_max, size=(number_groups, dimension))
        n_vec = np.full(number_groups, fixed_n, dtype=int)
        z_cal = []
        for j in range(number_groups):
            b_j = np.random.normal(loc=0.0, scale=gamma)
            z_cal.append(draw_group_latent_intercept(u_cal[j, :], int(n_vec[j]), b_j, rho))
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
        b_test = np.random.normal(loc=0.0, scale=gamma)
        z_test = draw_group_latent_intercept(u_test[0, :], target_n, b_test, rho)
        return {"U_test": u_test, "Z_test": z_test, "N_test": target_n}

    exp_mod.generate_calibration_data = generate_calibration_data_fixed
    exp_mod.generate_test_group = generate_test_group_fixed


def patch_poisson_generators(
    lambda_poisson,
    dimension,
    u_min,
    u_max,
    rho,
    gamma,
    min_target_n,
    size_offset=1,
):
    """Patch DGP generators to use N = size_offset + Poisson(lambda_poisson).

    ``size_offset=1`` is the legacy 1+Poi(λ) law (mean λ+1).
    ``size_offset=0`` is pure Poi(λ); draws of 0 are rejected (resample).
    """
    size_offset = int(size_offset)
    if size_offset < 0:
        raise ValueError(f"size_offset must be >= 0, got {size_offset}")

    def _draw_n(size=None):
        if size is None:
            while True:
                n = size_offset + int(np.random.poisson(lam=lambda_poisson))
                if n >= 1:
                    return n
        out = np.empty(int(size), dtype=int)
        for i in range(int(size)):
            out[i] = _draw_n()
        return out

    def generate_calibration_data_poisson(number_groups, lambda_Poisson, dgp_specification):
        u_cal = np.random.uniform(low=u_min, high=u_max, size=(number_groups, dimension))
        n_vec = _draw_n(size=number_groups)
        z_cal = []
        for j in range(number_groups):
            b_j = np.random.normal(loc=0.0, scale=gamma)
            z_cal.append(draw_group_latent_intercept(u_cal[j, :], int(n_vec[j]), b_j, rho))
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
        n_test = _draw_n()
        while n_test < min_target_n or n_test <= o_observed:
            n_test = _draw_n()
        b_test = np.random.normal(loc=0.0, scale=gamma)
        z_test = draw_group_latent_intercept(u_test[0, :], int(n_test), b_test, rho)
        return {"U_test": u_test, "Z_test": z_test, "N_test": int(n_test)}

    exp_mod.generate_calibration_data = generate_calibration_data_poisson
    exp_mod.generate_test_group = generate_test_group_poisson


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
            size_offset=int(config.get("size_offset", 1)),
        )
    else:
        raise ValueError(f"Unknown generation_mode: {config.get('generation_mode')}")

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()
    mu_hcp_no_within = without_within_group_training(create_mu_method_ols_offset())

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
        show_progress=False,
        quantile_mode=config.get("quantile_mode", "deterministic"),
        quantile_base_seed=config.get("quantile_base_seed", BASE_SEED),
    )

    results = offset_experiment_column(results, experiment_offset)
    results["worker_id"] = worker_id
    results["gamma"] = gamma
    return results


def run_experiment(config_name, config):
    tag = f"true_marg_latent_{gamma_tag(config['gamma'])}_{config_name}"
    out_dir = PROJECT_ROOT / "results_marginal" / "dgp" / tag
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"TRUE MARGINAL LATENT INTERCEPT: {config_name}")
    print("=" * 96)
    print(f"gamma: {config['gamma']}  (B_j ~ N(0, {float(config['gamma']) ** 2:g}))")
    print(f"Generation mode: {config.get('generation_mode')}")
    print(f"K: {config['number_groups_k']}, replicates: {config['total_replicates']}")
    print(f"o_values: {config['o_values']}, target_index: {config['target_index']}")
    print(f"alpha: {config['alpha']}, quantile_mode: {config.get('quantile_mode', 'deterministic')}")
    print("=" * 96)

    chunk_sizes = split_counts(config["total_replicates"], N_WORKERS)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()

    results_parts = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
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
            print(f"[{i}/{len(futures)}] worker {worker_id} done ({len(part)} rows) -> {chunk_path.name}")

    results = pd.concat(results_parts, ignore_index=True)
    sort_cols = [c for c in ["experiment", "experiment_id", "o_observed", "worker_id"] if c in results.columns]
    if sort_cols:
        results = results.sort_values(sort_cols).reset_index(drop=True)

    raw_csv = out_dir / f"{tag}_raw_results_complete.csv"
    results.to_csv(raw_csv, index=False)
    print(f"Saved: {raw_csv} ({len(results)} rows)")
    return results


def build_configs(
    total_replicates=1000,
    alpha=0.1,
    gamma=DEFAULT_GAMMA,
    quantile_mode="deterministic",
    quantile_base_seed=BASE_SEED,
):
    o_values = [0, 5, 10, 15, 20, 25, 30, 35]
    target_index = 35
    common = {
        "total_replicates": int(total_replicates),
        "number_groups_k": 20,
        "dimension": 5,
        "u_min": 1.0,
        "u_max": 5.0,
        "rho": 0.5,
        "o_values": o_values,
        "target_index": target_index,
        "alpha": float(alpha),
        "gamma": float(gamma),
        "quantile_mode": quantile_mode,
        "quantile_base_seed": int(quantile_base_seed),
    }
    return {
        "fixedN21": {
            **common,
            "generation_mode": "fixed",
            "fixed_n": 21,
            "target_n": target_index + 1,
        },
        "poissonNmean21": {
            **common,
            "generation_mode": "poisson",
            "lambda_poisson": 20,
            "size_offset": 1,  # N = 1 + Poi(20), mean 21
        },
        "poissonNmean25": {
            **common,
            "generation_mode": "poisson",
            "lambda_poisson": 25,
            "size_offset": 0,  # N = Poi(25)
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="True-marginal latent-intercept DGP experiments (gamma configurable)."
    )
    parser.add_argument("--alphas", type=str, default=PAPER_ALPHA_GRID_STR)
    parser.add_argument("--configs", type=str, default="fixedN21,poissonNmean25")
    parser.add_argument("--total_replicates", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--n_workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--quantile-mode",
        choices=["deterministic", "randomized"],
        default="deterministic",
    )
    parser.add_argument("--quantile-base-seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def main():
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
            gamma=args.gamma,
            quantile_mode=args.quantile_mode,
            quantile_base_seed=args.quantile_base_seed,
        )
        alpha_tag = alpha_to_tag(alpha)
        for name in config_names:
            run_experiment(f"{name}_{alpha_tag}", configs[name])


if __name__ == "__main__":
    main()
