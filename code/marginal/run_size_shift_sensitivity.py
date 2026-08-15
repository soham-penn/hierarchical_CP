#!/usr/bin/env python3
"""GHCP coverage under a size-coupled latent intercept (paper App. D.3).

Reference-group sizes N_j = M_j ~ Poi(25). The latent intercept is

    B_j = γ (√(1-ξ²) ε_j + ξ (M_j − 25)/5),

with M_{K+1} latent for the test group (used only to form B_{K+1}). The
observed test stream follows the Poisson GHCP protocol: length
    target_index+1 (36 when target_index=35); history is the first o rows.
|ξ|<1; ξ=0 recovers the paper Poisson DGP. Each replicate shares
(U, M, ε, X) across ξ. Negative ξ is the same coupling with opposite sign.
Default ξ grid (paper 2×2): {−0.75, 0, 0.5, 0.75}. A default run overwrites
the trial CSV; pass --merge-existing-xi to keep other ξ.
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

from code.marginal.run_true_marginal_latent_intercept_experiments import (
    BASE_SEED,
    DEFAULT_GAMMA,
    alpha_to_tag,
    build_configs,
    draw_group_latent_intercept,
    gamma_tag,
    size_coupled_intercept,
    split_counts,
)
from code.paths import RESULTS_DGP_MARGINAL
from methods.donor_hcp import compute_donor_hcp_randomized_interval
from methods.mu_methods import create_mu_method_random_forest_offset
from scores import make_quantile_seed

XI_GRID = (-0.75, 0.0, 0.5, 0.75)
RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123
RESULTS_PREFIX = "size_shift"
O_VALUES = [0, 5, 10, 15, 20]

print = functools.partial(print, flush=True)


def _draw_poi(lambda_poisson, size_offset=0):
    while True:
        n = int(size_offset) + int(np.random.poisson(lam=lambda_poisson))
        if n >= 1:
            return n


def _shift_y(z_base, b):
    b = float(b)
    return [
        {"X": np.asarray(z["X"], dtype=float).copy(), "Y": float(z["Y"]) + b, "B": b}
        for z in z_base
    ]


def _width(lo, hi) -> float:
    if np.isfinite(lo) and np.isfinite(hi):
        return float(hi - lo)
    return np.inf


def _draw_replicate(config):
    k = int(config["number_groups_k"])
    d = int(config["dimension"])
    rho = float(config["rho"])
    lam = float(config["lambda_poisson"])
    mean_m = float(lam)
    min_target_n = int(config["target_index"]) + 1
    size_offset = int(config.get("size_offset", 0))

    u_cal = np.random.uniform(
        low=config["u_min"], high=config["u_max"], size=(k, d)
    )
    m_cal = np.array(
        [_draw_poi(lam, size_offset) for _ in range(k)], dtype=int
    )
    eps_cal = np.random.normal(size=k)
    z_cal_base = [
        draw_group_latent_intercept(u_cal[j], int(m_cal[j]), 0.0, rho)
        for j in range(k)
    ]

    u_test = np.random.uniform(
        low=config["u_min"], high=config["u_max"], size=(1, d)
    )
    # M_test drives B_test only; observed stream length is target_index+1
    # (Poisson GHCP protocol: target at index 35, history = first o rows).
    m_test = _draw_poi(lam, size_offset)
    eps_test = float(np.random.normal())
    n_test = min_target_n
    z_test_base = draw_group_latent_intercept(u_test[0], int(n_test), 0.0, rho)

    return {
        "U_cal": u_cal,
        "m_cal": m_cal,
        "eps_cal": eps_cal,
        "z_cal_base": z_cal_base,
        "U_test": u_test,
        "m_test": m_test,
        "eps_test": eps_test,
        "z_test_base": z_test_base,
        "mean_m": mean_m,
    }


def _run_one(config, mu_hcp, experiment_id):
    draw = _draw_replicate(config)
    gamma = float(config["gamma"])
    alpha = float(config["alpha"])
    target_index = int(config["target_index"])
    quantile_mode = config.get("quantile_mode", "deterministic")
    quantile_base_seed = int(config.get("quantile_base_seed", BASE_SEED))
    rows = []

    for xi in config["xi_grid"]:
        xi = float(xi)
        b_cal = np.array(
            [
                size_coupled_intercept(
                    gamma, xi, draw["m_cal"][j], draw["eps_cal"][j], draw["mean_m"]
                )
                for j in range(len(draw["m_cal"]))
            ],
            dtype=float,
        )
        z_cal = [
            _shift_y(draw["z_cal_base"][j], b_cal[j])
            for j in range(len(draw["m_cal"]))
        ]
        b_test = size_coupled_intercept(
            gamma, xi, draw["m_test"], draw["eps_test"], draw["mean_m"]
        )
        z_test = _shift_y(draw["z_test_base"], b_test)
        true_target = z_test[target_index]["Y"]
        if np.std(b_cal) > 0 and np.std(draw["m_cal"]) > 0:
            corr_bn = float(np.corrcoef(b_cal, draw["m_cal"])[0, 1])
        else:
            corr_bn = np.nan

        for o in config["o_values"]:
            dhcp_seed = experiment_id * 1009 + (int(o) + 1) * 131 + 17
            q_seed = make_quantile_seed(
                quantile_base_seed, experiment_id, target_index, o, "donor_hcp"
            )
            res = compute_donor_hcp_randomized_interval(
                U_calibration=draw["U_cal"],
                Z_calibration=z_cal,
                U_test=draw["U_test"],
                Z_test=z_test,
                o_observed=int(o),
                alpha=alpha,
                alpha_selection=0.5,
                mu_method=mu_hcp,
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
                    "alpha": alpha,
                    "xi": xi,
                    "o_observed": int(o),
                    "coverage": float(lo <= true_target <= hi),
                    "width": w,
                    "infinite": int(not np.isfinite(w)),
                    "corr_B_N": corr_bn,
                    "b_test": float(b_test),
                    "m_test": int(draw["m_test"]),
                }
            )
    return rows


def _xi_in_grid(xi, grid) -> bool:
    return any(np.isclose(float(xi), float(v)) for v in grid)


def _merge_existing_xi(raw_csv: Path, results: pd.DataFrame, xi_grid) -> pd.DataFrame:
    """Keep previously saved ξ cells that were not recomputed in this run.

    Draws of (U, M, ε, X) do not depend on the ξ loop (GHCP uses isolated RNGs),
    so a negative-ξ follow-up can be merged onto the original ξ≥0 file.
    """
    if not raw_csv.exists() or results.empty:
        return results
    prev = pd.read_csv(raw_csv)
    keep = ~prev["xi"].map(lambda x: _xi_in_grid(x, xi_grid))
    kept = prev.loc[keep]
    if len(kept) == 0:
        return results
    print(f"Merging {len(kept)} existing rows (other ξ) with {len(results)} new rows")
    return pd.concat([kept, results], ignore_index=True)


def run_chunk(worker_id, number_experiments_chunk, experiment_offset, config):
    print(f"  chunk {worker_id}: start ({number_experiments_chunk} reps)", flush=True)
    np.random.seed(BASE_SEED + 1000 * worker_id)

    mu_hcp = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
    )
    all_rows = []
    for e in range(number_experiments_chunk):
        all_rows.extend(_run_one(config, mu_hcp, experiment_offset + e))
        if (e + 1) % 10 == 0:
            print(f"  chunk {worker_id}: {e + 1}/{number_experiments_chunk}")

    results = pd.DataFrame(all_rows)
    results["worker_id"] = worker_id
    results["gamma"] = float(config["gamma"])
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

    n_chunks = int(np.ceil(config["total_replicates"] / max(int(chunk_size), 1)))
    chunk_sizes = split_counts(config["total_replicates"], n_chunks)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()

    print("=" * 96)
    print("SIZE-SHIFT SENSITIVITY  |  Poi(25) latent intercept RF")
    print("=" * 96)
    print(f"gamma={config['gamma']}  alpha={config['alpha']}  B={config['total_replicates']}")
    print(f"xi grid = {list(config['xi_grid'])}")
    print(f"o_values={config['o_values']}  chunks={n_chunks}  workers={n_workers}")
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
    raw_csv = out_dir / f"{tag}_raw_results_complete.csv"
    if bool(config.get("merge_existing_xi", False)):
        results = _merge_existing_xi(raw_csv, results, config["xi_grid"])
    elif raw_csv.exists():
        print(f"Replacing {raw_csv} (pass --merge-existing-xi to keep other ξ)")
    results = results.sort_values(
        ["experiment", "xi", "o_observed"]
    ).reset_index(drop=True)
    results.to_csv(raw_csv, index=False)
    print(f"Saved: {raw_csv} ({len(results)} rows)")

    manifest = {
        "suite": "size_shift",
        "generation_mode": "poisson",
        "lambda_poisson": config["lambda_poisson"],
        "size_offset": config.get("size_offset", 0),
        "gamma": float(config["gamma"]),
        "alpha": float(config["alpha"]),
        "total_replicates": int(config["total_replicates"]),
        "o_values": list(config["o_values"]),
        "target_index": int(config["target_index"]),
        "xi_grid": [float(x) for x in config["xi_grid"]],
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
    p.add_argument("--n_workers", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument(
        "--xi",
        type=str,
        default="",
        help="Comma-separated ξ grid (default: −0.75, 0, 0.5, 0.75).",
    )
    p.add_argument(
        "--merge-existing-xi",
        action="store_true",
        help="Keep previously saved ξ that are not in this run (incremental).",
    )
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
    if str(args.xi).strip():
        cfg["xi_grid"] = [float(x) for x in str(args.xi).split(",") if str(x).strip()]
    else:
        cfg["xi_grid"] = list(XI_GRID)
    cfg["total_replicates"] = int(args.B)
    cfg["alpha"] = float(args.alpha)
    cfg["merge_existing_xi"] = bool(args.merge_existing_xi)
    run_experiment(cfg, n_workers=int(args.n_workers), chunk_size=int(args.chunk_size))


if __name__ == "__main__":
    main()
