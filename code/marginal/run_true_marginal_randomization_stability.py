#!/usr/bin/env python3
"""Randomization-stability runs for DGP true-marginal D-HCP experiments."""

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
from code.marginal.run_true_marginal_experiments import (
    BASE_SEED,
    alpha_to_tag,
    build_configs,
    patch_fixed_size_generators,
    patch_poisson_generators,
)
from methods import create_mu_method_ols_offset
from methods.donor_hcp import compute_donor_hcp_randomized_interval


def _width(interval: tuple[float, float]) -> float:
    lo, hi = interval
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.nan


def _patch_generators(config: dict) -> None:
    if config["generation_mode"] == "fixed":
        patch_fixed_size_generators(
            fixed_n=config["fixed_n"],
            target_n=config["target_n"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=config["rho"],
        )
    elif config["generation_mode"] == "poisson":
        patch_poisson_generators(
            lambda_poisson=config["lambda_poisson"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=config["rho"],
        )
    else:
        raise ValueError(f"Unknown generation mode: {config['generation_mode']}")


def run_one_dataset(args: tuple[str, dict, float, int, int, int]) -> dict:
    config_name, config, alpha, o, dataset_id, n_reruns = args
    seed = BASE_SEED + 50_000_000 + int(round(alpha * 1_000_000)) + 10_000 * int(o) + int(dataset_id)
    np.random.seed(seed)
    _patch_generators(config)

    cal = exp_mod.generate_calibration_data(
        number_groups=config["number_groups_k"],
        lambda_Poisson=config.get("lambda_poisson", 20),
        dgp_specification={
            "dimension": config["dimension"],
            "u_min": config["u_min"],
            "u_max": config["u_max"],
        },
    )
    test = exp_mod.generate_test_group(
        lambda_Poisson=config.get("lambda_poisson", 20),
        dgp_specification={
            "dimension": config["dimension"],
            "u_min": config["u_min"],
            "u_max": config["u_max"],
        },
        o_observed=config["target_index"],
    )

    mu_hcp = create_mu_method_ols_offset()
    true_y = test["Z_test"][config["target_index"]]["Y"]
    widths: list[float] = []
    uppers: list[float] = []
    coverages: list[int] = []

    for r in range(n_reruns):
        res = compute_donor_hcp_randomized_interval(
            U_calibration=cal["U_calibration"],
            Z_calibration=cal["Z_calibration"],
            U_test=test["U_test"],
            Z_test=test["Z_test"],
            o_observed=o,
            alpha=alpha,
            alpha_selection=0.5,
            mu_method=mu_hcp,
            test_index_target=config["target_index"],
            random_seed=seed + 1_000 + r,
        )
        interval = res["interval"]
        w = _width(interval)
        if np.isfinite(w):
            widths.append(w)
            uppers.append(float(interval[1]))
        coverages.append(int(interval[0] <= true_y <= interval[1]))

    mean_width = float(np.mean(widths)) if widths else np.nan
    sd_upper = float(np.std(uppers, ddof=1)) if len(uppers) > 1 else 0.0
    sd_width = float(np.std(widths, ddof=1)) if len(widths) > 1 else 0.0

    return {
        "dataset": config_name,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "o": int(o),
        "dataset_id": int(dataset_id),
        "mean_width": mean_width,
        "sd_upper": sd_upper,
        "sd_width": sd_width,
        "relative_instability": sd_upper / mean_width if mean_width and mean_width > 1e-12 else np.nan,
        "coverage_mean": float(np.mean(coverages)) if coverages else np.nan,
        "n_finite": len(widths),
        "n_reruns": int(n_reruns),
        "base_seed": BASE_SEED,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DGP true-marginal randomization stability.")
    parser.add_argument("--alphas", default="0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25")
    parser.add_argument("--configs", default="fixedN21,poissonNmean21")
    parser.add_argument("--n_datasets", type=int, default=50)
    parser.add_argument("--n_reruns", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs0 = build_configs()

    for config_name in config_names:
        if config_name not in configs0:
            raise ValueError(f"Unknown config {config_name}")

    for config_name in config_names:
        rows = []
        for alpha in alphas:
            configs = build_configs(alpha=alpha)
            config = configs[config_name]
            jobs = [
                (config_name, config, alpha, o, dataset_id, args.n_reruns)
                for o in config["o_values"]
                for dataset_id in range(args.n_datasets)
            ]
            print(f"Running stability: {config_name}, {alpha_to_tag(alpha)}, {len(jobs)} datasets")
            with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
                futures = [ex.submit(run_one_dataset, job) for job in jobs]
                for i, fut in enumerate(as_completed(futures), start=1):
                    rows.append(fut.result())
                    if i % 50 == 0 or i == len(futures):
                        print(f"  completed {i}/{len(futures)}")

        df = pd.DataFrame(rows)
        out_metrics = PROJECT_ROOT / "results_marginal" / "dgp" / f"true_marg_{config_name}_randomization_stability_dataset_metrics.csv"
        out_summary = PROJECT_ROOT / "results_marginal" / "dgp" / f"true_marg_{config_name}_randomization_stability_summary.csv"
        df.to_csv(out_metrics, index=False)
        summary = (
            df.groupby(["dataset", "alpha", "nominal_coverage", "o"], as_index=False)
            .agg(
                mean_width_mean=("mean_width", "mean"),
                sd_upper_mean=("sd_upper", "mean"),
                sd_width_mean=("sd_width", "mean"),
                relative_instability_mean=("relative_instability", "mean"),
                coverage_mean=("coverage_mean", "mean"),
                n_finite_total=("n_finite", "sum"),
                n_datasets=("dataset_id", "count"),
                n_reruns=("n_reruns", "first"),
                base_seed=("base_seed", "first"),
            )
        )
        summary.to_csv(out_summary, index=False)
        print(f"Saved {out_metrics}")
        print(f"Saved {out_summary}")


if __name__ == "__main__":
    main()
