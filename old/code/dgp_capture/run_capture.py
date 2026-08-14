#!/usr/bin/env python3
"""
DGP RF capture: fit RF once per HCP call, store full μ_global CSVs.

Within-group mode for the capture run is **mean** (sample-mean shrinkage).
Use apply_dgp_capture.py later for correction / none without refitting RF.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture.engine import run_one_experiment_capture
from code.marginal.dgp_capture.storage import DgpCaptureStore, NullDgpCaptureStore
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
from code.shared.plot_engine import PAPER_ALPHA_GRID_STR

print = functools.partial(print, flush=True)

RF_NTREE = 50
RF_NODESIZE = 5
RF_MTRY = None  # None -> sqrt(p_total); float -> feature fraction; int -> feature count
RF_RANDOM_STATE = 123
# "absolute" (default), "studentized" (RF-σ), or "cqr"
SCORE_TYPE = "absolute"
# "mean" (sample-mean shrink) or "local_rf" (within-group RF on X)
WITHIN_GROUP_MODE = "mean"
# "rf" (default) or "ud_squared" (oracle-structure mu = u_d^2)
PREDICTOR = "rf"
# If False, skip heavy μ-capture CSVs (results CSV only). Studentized sets this False.
CAPTURE_MU = True
# Optional force on GHCP mean/local blend: None = default w_g; 0.0 = pure local half-mean.
W_G_OVERRIDE = None
# If False, skip extra Donor-HCP-local (RF) method.
INCLUDE_GHCP_LOCAL_RF = True
PLOTS_ROOT = REPO_ROOT / "plots_marginal" / "dgp_true_marginal_rf"
RESULTS_PREFIX = "true_marg_latent_rf"
SUITE_NAME = "dgp_true_marginal_rf"


CAPTURE_TABLES = (
    "observations",
    "experiment_design",
    "baseline_hcp_split",
    "baseline_mu_global",
    "global_fits",
    "mu_global",
    "mu_predictions",
    "hcp_calls",
)


def _merge_capture_chunks(capture_dir: Path, B: int) -> None:
    """Merge per-experiment CSVs into top-level capture tables (no concurrent writes)."""
    exp_root = capture_dir / "experiments"
    for table in CAPTURE_TABLES:
        parts = []
        for exp_id in range(1, B + 1):
            path = exp_root / f"e{exp_id}" / f"{table}.csv"
            if path.exists():
                parts.append(pd.read_csv(path))
        if not parts:
            continue
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(capture_dir / f"{table}.csv", index=False)
        print(f"  merged {table}.csv ({len(out):,} rows)")


def _patch_generators(config: dict) -> None:
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
        )


def _worker_chunk(worker_id: int, chunk_size: int, experiment_offset: int, config: dict, capture_dir: str):
    """Match OLS/RF latent runners: one seed per worker, sequential draws in chunk."""
    np.random.seed(BASE_SEED + 1000 * worker_id)
    _patch_generators(config)

    d = int(config["dimension"])
    feature_x = [f"x_{j}" for j in range(d - 1)]
    feature_u = [f"u_{j}" for j in range(d)]
    capture_mu = bool(config.get("capture_mu", True))
    log_every = max(1, min(10, chunk_size // 10 or 1))

    parts = []
    t_chunk = time.perf_counter()
    for i in range(chunk_size):
        experiment = experiment_offset + i + 1
        t_exp = time.perf_counter()
        if capture_mu:
            exp_dir = Path(capture_dir) / "experiments" / f"e{int(experiment)}"
            store = DgpCaptureStore(exp_dir)
            store.set_feature_names(feature_x, feature_u)
        else:
            store = NullDgpCaptureStore()
        parts.append(
            run_one_experiment_capture(
                experiment=int(experiment),
                config=config,
                store=store,
            )
        )
        if (i + 1) % log_every == 0 or (i + 1) == chunk_size:
            elapsed = time.perf_counter() - t_chunk
            last = time.perf_counter() - t_exp
            rate = (i + 1) / max(elapsed, 1e-9)
            eta_min = (chunk_size - i - 1) / max(rate, 1e-9) / 60.0
            print(
                f"  worker {worker_id}: {i + 1}/{chunk_size} exps "
                f"(last={last:.1f}s, {rate * 60:.1f}/min, ETA {eta_min:.1f} min)"
            )
            # Spawned workers often don't inherit nohup stdout; mirror to a file.
            try:
                prog = Path(capture_dir) / "worker_progress.log"
                with prog.open("a") as fh:
                    fh.write(
                        f"worker {worker_id}: {i + 1}/{chunk_size} exps "
                        f"(last={last:.1f}s, {rate * 60:.1f}/min, ETA {eta_min:.1f} min)\n"
                    )
            except OSError:
                pass

    df = pd.concat(parts, ignore_index=True)
    df["worker_id"] = worker_id
    return df


def run_one_config(config_name: str, config: dict, n_workers: int) -> Path:
    gamma = float(config["gamma"])
    alpha = float(config["alpha"])
    alpha_tag = alpha_to_tag(alpha)
    gtag = gamma_tag(gamma)

    capture_dir = PLOTS_ROOT / "capture_data" / f"{config_name}_{alpha_tag}"
    results_tag = f"{RESULTS_PREFIX}_{gtag}_{config_name}_{alpha_tag}"
    results_dir = REPO_ROOT / "results_marginal" / "dgp" / results_tag
    capture_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    _patch_generators(config)

    d = int(config["dimension"])
    store0 = DgpCaptureStore(capture_dir)
    store0.set_feature_names(
        [f"x_{j}" for j in range(d - 1)],
        [f"u_{j}" for j in range(d)],
    )
    capture_config = {
        **config,
        "predictor": PREDICTOR,
        "score_type": SCORE_TYPE,
        "within_group_mode": WITHIN_GROUP_MODE,
        "rf_ntree": RF_NTREE,
        "rf_nodesize": RF_NODESIZE,
        "rf_mtry": RF_MTRY if RF_MTRY is not None else "",
        "rf_random_state": RF_RANDOM_STATE,
        "seed": BASE_SEED,
        "alpha_selection": 0.5,
        "n_repeated": 50,
        "suite": SUITE_NAME,
        "config_name": config_name,
        "capture_mu": bool(CAPTURE_MU),
        "w_g_override": "" if W_G_OVERRIDE is None else float(W_G_OVERRIDE),
        "include_ghcp_local_rf": bool(INCLUDE_GHCP_LOCAL_RF),
    }
    # JSON-serialize lists
    capture_config["o_values"] = list(config["o_values"])
    store0.write_config(capture_config)

    B = int(config["total_replicates"])
    print("=" * 96)
    print(f"DGP CAPTURE ({PREDICTOR}): {config_name}  alpha={alpha}  gamma={gamma}")
    print("=" * 96)
    print(f"  within_group_mode={WITHIN_GROUP_MODE}")
    print(f"  score_type={SCORE_TYPE}")
    print(f"  capture_mu={bool(CAPTURE_MU)}")
    print(f"  w_g_override={W_G_OVERRIDE}")
    print(f"  include_ghcp_local_rf={bool(INCLUDE_GHCP_LOCAL_RF)}")
    if PREDICTOR == "rf":
        print(f"  RF ntree={RF_NTREE}, nodesize={RF_NODESIZE}, mtry={RF_MTRY if RF_MTRY is not None else 'sqrt(p)'}")
    elif PREDICTOR in ("bayes", "bayes_joint_xy", "oracle_bayes"):
        print(f"  predictor={PREDICTOR}  (global mu = E[Y|X,U] analytical Bayes)")
    else:
        print(f"  predictor={PREDICTOR}  (global mu = u_d^2)")
    print(f"  replicates={B}, workers={n_workers}")
    print(f"  RNG seed per worker: {BASE_SEED} + 1000 * worker_id (matches OLS latent)")
    print(f"  capture -> {capture_dir}")
    print(f"  results -> {results_dir}")
    print("=" * 96)

    chunk_sizes = split_counts(B, n_workers)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()

    parts = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for worker_id, (chunk_size, offset) in enumerate(zip(chunk_sizes, offsets)):
            if chunk_size == 0:
                continue
            fut = ex.submit(
                _worker_chunk, worker_id, chunk_size, offset, capture_config, str(capture_dir),
            )
            futures[fut] = worker_id

        done = 0
        n_chunks = len(futures)
        for fut in as_completed(futures):
            worker_id = futures[fut]
            part = fut.result()
            parts.append(part)
            done += 1
            elapsed = time.perf_counter() - t0
            print(f"  [{done}/{n_chunks}] worker {worker_id} done ({len(part)} rows, "
                  f"{elapsed / 60:.1f} min elapsed)")

    results = pd.concat(parts, ignore_index=True)
    sort_cols = [c for c in ["experiment", "o_observed", "worker_id"] if c in results.columns]
    if sort_cols:
        results = results.sort_values(sort_cols).reset_index(drop=True)
    results["gamma"] = gamma

    if CAPTURE_MU:
        print("Merging per-experiment capture CSVs...")
        _merge_capture_chunks(capture_dir, B)
    else:
        print("Skipping capture CSV merge (capture_mu=False; results-only mode)")

    raw_csv = results_dir / f"{results_tag}_raw_results_complete.csv"
    results.to_csv(raw_csv, index=False)
    results.to_csv(capture_dir / "raw_results_mean.csv", index=False)
    print(f"Saved results: {raw_csv} ({len(results)} rows)")
    print(f"Capture artifacts: {capture_dir}")
    return results_dir


def parse_args():
    p = argparse.ArgumentParser(description="DGP RF capture (mean within-group).")
    p.add_argument("--alphas", type=str, default=PAPER_ALPHA_GRID_STR)
    p.add_argument("--configs", type=str, default="fixedN21,poissonNmean21")
    p.add_argument("--total_replicates", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--quantile-mode", choices=("deterministic", "randomized"), default="deterministic")
    p.add_argument("--quantile-base-seed", type=int, default=BASE_SEED)
    return p.parse_args()


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
        for name in config_names:
            run_one_config(name, configs[name], args.n_workers)


if __name__ == "__main__":
    main()
