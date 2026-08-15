#!/usr/bin/env python3
"""Effect of merger weights when the global predictor is useful (γ=0).

Paper Poi(25) Gaussian DGP with:
  - U_{j,1:d-1} ~ Unif(1,5), U_{j,d} ~ Unif(1,2)  (only last coord narrowed)
  - B_j = 0 (gamma = 0)
  - o ∈ {0,5,10,15,20}
  - λ_local ∈ {1/7, 3/7, 4/7, 6/7}

For each replicate and weight, evaluate restricted GHCP (η=0.5) under both
the RF offset merger and the Bayes E[Y|X,U] offset merger.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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
    alpha_to_tag,
    build_configs,
    draw_group_latent_intercept,
    gamma_tag,
    split_counts,
)
from code.paths import RESULTS_DGP_MARGINAL
from methods.donor_hcp import compute_donor_hcp_randomized_interval
from methods.mu_methods import (
    create_mu_method_bayes_joint_xy_offset,
    create_mu_method_random_forest_offset,
    set_w_g_override,
)
from scores import make_quantile_seed

RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123
RESULTS_PREFIX = "effect_of_weights_ud_narrow"
O_VALUES = [0, 5, 10, 15, 20]
# λ_local ∈ {1/7, 3/7, 4/7, 6/7} ⇔ w_g ∈ {6/7, 4/7, 3/7, 1/7}.
WEIGHT_GRID_UD = tuple(1.0 - k / 7.0 for k in (1, 3, 4, 6))
U_D_MAX = 2.0
PREDICTORS = ("rf", "bayes")

print = functools.partial(print, flush=True)


def _width(lo, hi) -> float:
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.inf


def _draw_u(n_groups: int, dimension: int, u_min: float, u_max: float, u_d_max: float):
    """U_{1:d-1} ~ Unif(u_min,u_max); U_d ~ Unif(u_min, u_d_max)."""
    u = np.random.uniform(low=u_min, high=u_max, size=(int(n_groups), int(dimension)))
    u[:, -1] = np.random.uniform(low=u_min, high=u_d_max, size=int(n_groups))
    return u


def patch_poisson_ud_narrow(
    *,
    lambda_poisson,
    dimension,
    u_min,
    u_max,
    u_d_max,
    rho,
    gamma,
    min_target_n,
    size_offset=0,
):
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
        u_cal = _draw_u(number_groups, dimension, u_min, u_max, u_d_max)
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
            u_test = _draw_u(1, dimension, u_min, u_max, u_d_max)
        n_test = _draw_n()
        while n_test < min_target_n or n_test <= o_observed:
            n_test = _draw_n()
        b_test = np.random.normal(loc=0.0, scale=gamma)
        z_test = draw_group_latent_intercept(u_test[0, :], int(n_test), b_test, rho)
        return {"U_test": u_test, "Z_test": z_test, "N_test": int(n_test)}

    exp_mod.generate_calibration_data = generate_calibration_data_poisson
    exp_mod.generate_test_group = generate_test_group_poisson


def _run_one(
    *,
    number_groups_k,
    lambda_Poisson,
    dgp_specification,
    o_values,
    target_index,
    alpha,
    alpha_selection,
    mu_by_predictor,
    weights,
    quantile_mode,
    quantile_base_seed,
    experiment_id,
):
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
    U_cal = cal["U_calibration"]
    Z_cal = cal["Z_calibration"]
    U_test = test["U_test"]
    Z_test = test["Z_test"]
    true_target = Z_test[target_index]["Y"]

    rows = []
    for o in o_values:
        for pred_name, mu_hcp in mu_by_predictor.items():
            dhcp_seed = (
                experiment_id * 1009
                + (int(o) + 1) * 131
                + 17
                + (0 if pred_name == "rf" else 101)
            )
            q_seed = make_quantile_seed(
                quantile_base_seed,
                experiment_id,
                target_index,
                o,
                f"donor_hcp_wgrid_{pred_name}",
            )
            for w_g in weights:
                mu_w = set_w_g_override(mu_hcp, float(w_g))
                res = compute_donor_hcp_randomized_interval(
                    U_calibration=U_cal,
                    Z_calibration=Z_cal,
                    U_test=U_test,
                    Z_test=Z_test,
                    o_observed=int(o),
                    alpha=alpha,
                    alpha_selection=alpha_selection,
                    mu_method=mu_w,
                    test_index_target=target_index,
                    random_seed=dhcp_seed,
                    quantile_mode=quantile_mode,
                    quantile_random_seed=q_seed,
                )
                lo, hi = res["interval"]
                w = _width(lo, hi)
                rows.append(
                    {
                        "experiment": experiment_id,
                        "alpha": float(alpha),
                        "o_observed": int(o),
                        "predictor": pred_name,
                        "w_g": float(w_g),
                        "w_local": float(1.0 - w_g),
                        "coverage": float(lo <= true_target <= hi),
                        "width": w,
                        "infinite": int(not np.isfinite(w)),
                    }
                )
            set_w_g_override(mu_hcp, None)
    return rows


def run_chunk(worker_id, number_experiments_chunk, experiment_offset, config):
    print(f"  chunk {worker_id}: start ({number_experiments_chunk} reps)", flush=True)
    np.random.seed(BASE_SEED + 1000 * worker_id)

    gamma = float(config["gamma"])
    rho = float(config["rho"])
    min_target_n = int(config["target_index"]) + 1
    patch_poisson_ud_narrow(
        lambda_poisson=config["lambda_poisson"],
        dimension=config["dimension"],
        u_min=config["u_min"],
        u_max=config["u_max"],
        u_d_max=float(config["u_d_max"]),
        rho=rho,
        gamma=gamma,
        min_target_n=min_target_n,
        size_offset=int(config.get("size_offset", 0)),
    )

    mu_by_predictor = {
        "rf": create_mu_method_random_forest_offset(
            ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
        ),
        "bayes": create_mu_method_bayes_joint_xy_offset(rho=rho, c=1.0),
    }
    weights = tuple(config["weights"])

    all_rows = []
    for e in range(number_experiments_chunk):
        all_rows.extend(
            _run_one(
                number_groups_k=config["number_groups_k"],
                lambda_Poisson=config["lambda_poisson"],
                dgp_specification={
                    "dimension": config["dimension"],
                    "u_min": config["u_min"],
                    "u_max": config["u_max"],
                },
                o_values=config["o_values"],
                target_index=config["target_index"],
                alpha=config["alpha"],
                alpha_selection=0.5,
                mu_by_predictor=mu_by_predictor,
                weights=weights,
                quantile_mode=config.get("quantile_mode", "deterministic"),
                quantile_base_seed=config.get("quantile_base_seed", BASE_SEED),
                experiment_id=experiment_offset + e,
            )
        )
        if (e + 1) % 10 == 0:
            print(f"  chunk {worker_id}: {e + 1}/{number_experiments_chunk}")

    results = pd.DataFrame(all_rows)
    results["worker_id"] = worker_id
    results["gamma"] = gamma
    return results


def run_experiment(config: dict, n_workers: int, chunk_size: int) -> pd.DataFrame:
    tag = (
        f"{RESULTS_PREFIX}_{gamma_tag(config['gamma'])}_"
        f"poissonNmean25_{alpha_to_tag(config['alpha'])}"
    )
    out_dir = RESULTS_DGP_MARGINAL / tag
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = max(int(n_workers), int(np.ceil(config["total_replicates"] / chunk_size)))
    chunk_sizes = split_counts(config["total_replicates"], n_chunks)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()

    print("=" * 96)
    print("EFFECT OF WEIGHTS (U_d narrow, γ=0)  |  Poi(25) RF + Bayes")
    print("=" * 96)
    print(f"gamma={config['gamma']}  alpha={config['alpha']}  B={config['total_replicates']}")
    print(f"U_d ~ Unif({config['u_min']},{config['u_d_max']}); else Unif({config['u_min']},{config['u_max']})")
    print(f"o_values={config['o_values']}  target_index={config['target_index']}")
    print(f"w_g grid = {list(config['weights'])}")
    print(f"predictors={list(PREDICTORS)}  chunks={n_chunks}  workers={n_workers}")
    print("=" * 96)

    parts = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        for wid, (sz, off) in enumerate(zip(chunk_sizes, offsets)):
            if sz == 0:
                continue
            futures.append(ex.submit(run_chunk, wid, sz, off, config))
        for i, fut in enumerate(as_completed(futures), start=1):
            part = fut.result()
            parts.append(part)
            wid = int(part["worker_id"].iloc[0])
            path = chunk_dir / f"raw_results_chunk_worker_{wid}.csv"
            part.to_csv(path, index=False)
            print(f"[{i}/{len(futures)}] chunk {wid} done ({len(part)} rows)")

    results = pd.concat(parts, ignore_index=True)
    results = results.sort_values(
        ["experiment", "predictor", "o_observed", "w_g"]
    ).reset_index(drop=True)
    raw_csv = out_dir / f"{tag}_raw_results_complete.csv"
    results.to_csv(raw_csv, index=False)
    print(f"Saved: {raw_csv} ({len(results)} rows)")

    manifest = {
        "suite": "effect_of_weights_ud_narrow",
        "generation_mode": "poisson",
        "lambda_poisson": config["lambda_poisson"],
        "size_offset": config.get("size_offset", 0),
        "gamma": float(config["gamma"]),
        "alpha": float(config["alpha"]),
        "total_replicates": int(config["total_replicates"]),
        "o_values": list(config["o_values"]),
        "target_index": int(config["target_index"]),
        "u_min": float(config["u_min"]),
        "u_max": float(config["u_max"]),
        "u_d_max": float(config["u_d_max"]),
        "weights_w_g": list(config["weights"]),
        "predictors": list(PREDICTORS),
        "quantile_mode": config.get("quantile_mode", "deterministic"),
        "base_seed": BASE_SEED,
        "chunk_seed_formula": "np.random.seed(BASE_SEED + 1000 * chunk_id)",
        "n_chunks": int(n_chunks),
        "chunk_sizes": list(map(int, chunk_sizes)),
        "rf": {
            "ntree": RF_NTREE,
            "nodesize": RF_NODESIZE,
            "random_state": RF_RANDOM_STATE,
        },
        "rows": int(len(results)),
        "saved": str(raw_csv),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n_workers", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--u-d-max", type=float, default=U_D_MAX)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_configs(
        total_replicates=args.B,
        alpha=args.alpha,
        gamma=args.gamma,
        quantile_mode="deterministic",
        quantile_base_seed=BASE_SEED,
    )
    cfg = dict(configs["poissonNmean25"])
    cfg["o_values"] = list(O_VALUES)
    cfg["weights"] = list(WEIGHT_GRID_UD)
    cfg["total_replicates"] = int(args.B)
    cfg["alpha"] = float(args.alpha)
    cfg["gamma"] = float(args.gamma)
    cfg["u_d_max"] = float(args.u_d_max)
    run_experiment(cfg, n_workers=int(args.n_workers), chunk_size=int(args.chunk_size))


if __name__ == "__main__":
    main()
