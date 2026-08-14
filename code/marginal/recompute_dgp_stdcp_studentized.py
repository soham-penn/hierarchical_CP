#!/usr/bin/env python3
"""DGP Std-CP studentized vs absolute, same seeds / splits as the paper suite.

The paper DGP Std-CP used absolute |Y-μ|. This regenerates the same test groups
(8 workers, BASE_SEED+1000*worker_id) and evaluates both score types.

α=0.05 is a standalone draw (original suite). α∈{0.10,0.15,0.20} share one draw.
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
from scores import make_quantile_seed

print = functools.partial(print, flush=True)
N_WORKERS_DEFAULT = 8
O_VALUES = [0, 5, 10, 15, 20, 25, 30, 35]
SCORE_TYPES = ("absolute", "studentized")


def _width(lo, hi):
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.inf


def run_chunk(worker_id, n_chunk, experiment_offset, config, alphas):
    print(f"  worker {worker_id}: start ({n_chunk} reps)", flush=True)
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
    else:
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

    q_seed = int(config.get("quantile_base_seed", BASE_SEED))
    target_index = int(config["target_index"])
    rows = []
    for e in range(n_chunk):
        experiment_id = e + 1  # local id used in original quantile seeds
        cal = exp_mod.generate_calibration_data(
            number_groups=config["number_groups_k"],
            lambda_Poisson=config.get("lambda_poisson", 21),
            dgp_specification={
                "dimension": config["dimension"],
                "u_min": config["u_min"],
                "u_max": config["u_max"],
            },
        )
        del cal  # consume np.random exactly as in run_one_experiment
        test = exp_mod.generate_test_group(
            lambda_Poisson=config.get("lambda_poisson", 21),
            dgp_specification={
                "dimension": config["dimension"],
                "u_min": config["u_min"],
                "u_max": config["u_max"],
            },
            o_observed=target_index,
        )
        Z_test = test["Z_test"]
        true_target = Z_test[target_index]["Y"]
        X_target = Z_test[target_index]["X"]
        for o in O_VALUES:
            x_hist = np.array([Z_test[i]["X"] for i in range(o)])
            y_hist = np.array([Z_test[i]["Y"] for i in range(o)])
            split_seed = make_quantile_seed(
                q_seed, experiment_id, target_index, o, "stdcp_split"
            )
            q_rseed = make_quantile_seed(
                q_seed, experiment_id, target_index, o, "stdcp"
            )
            for score_type in SCORE_TYPES:
                for a in alphas:
                    lo, hi = exp_mod._compute_std_cp_interval(
                        x_hist=x_hist,
                        y_hist=y_hist,
                        x_target=X_target,
                        alpha=a,
                        quantile_mode="randomized",
                        random_seed=q_rseed,
                        rng=np.random.default_rng(split_seed),
                        score_type=score_type,
                    )
                    w = _width(lo, hi)
                    rows.append(
                        {
                            "experiment": experiment_offset + e + 1,
                            "alpha": float(a),
                            "o_observed": int(o),
                            "score_type": score_type,
                            "coverage": float(lo <= true_target <= hi),
                            "width": w,
                            "infinite": int(not np.isfinite(w)),
                            "worker_id": worker_id,
                            "gamma": gamma,
                        }
                    )
        if (e + 1) % 50 == 0:
            print(f"  worker {worker_id}: {e + 1}/{n_chunk}")
    return pd.DataFrame(rows)


def run_design(name: str, config: dict, alphas: list[float], n_workers: int, tag_suffix: str):
    tag = f"stdcp_score_compare_{gamma_tag(config['gamma'])}_{name}_{tag_suffix}"
    out_dir = RESULTS_DGP_MARGINAL / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"DGP Std-CP score compare: {name} alphas={alphas} B={config['total_replicates']}")
    print("=" * 80)
    sizes = split_counts(config["total_replicates"], n_workers)
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    parts = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(run_chunk, wid, sz, off, config, alphas)
            for wid, (sz, off) in enumerate(zip(sizes, offsets))
            if sz > 0
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            part = fut.result()
            parts.append(part)
            print(f"[{i}/{len(futs)}] worker done ({len(part)} rows)")
    results = pd.concat(parts, ignore_index=True)
    results = results.sort_values(
        ["alpha", "experiment", "o_observed", "score_type"]
    ).reset_index(drop=True)
    raw = out_dir / f"{tag}_raw_results_complete.csv"
    results.to_csv(raw, index=False)
    print(f"Saved: {raw} ({len(results)} rows)")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", default="fixedN21,poissonNmean25")
    p.add_argument("--total_replicates", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--n_workers", type=int, default=N_WORKERS_DEFAULT)
    args = p.parse_args()

    base = build_configs(
        total_replicates=args.total_replicates,
        alpha=0.1,
        gamma=args.gamma,
        # HCP/GHCP unused here; Std-CP path hardcodes randomized below.
        quantile_mode="deterministic",
        quantile_base_seed=BASE_SEED,
    )
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for name in names:
        cfg = dict(base[name])
        # Match original suite: α=0.05 standalone, then shared 0.10/0.15/0.20.
        run_design(name, cfg, [0.05], args.n_workers, "alpha05")
        run_design(name, cfg, [0.10, 0.15, 0.20], args.n_workers, "alpha10_15_20")


if __name__ == "__main__":
    main()
