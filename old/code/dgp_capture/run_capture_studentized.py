#!/usr/bin/env python3
"""
DGP RF capture with studentized scores → plots_marginal/dgp_studentized.

GHCP uses dual RF-σ (global + within), blended with the same w_g as the mean.
Std-CP uses local RF μ + local RF-σ on the target half-split (same protocol as
absolute Std-CP, with studentized scores).

Usage:
  .venv/bin/python code/marginal/dgp_capture/run_capture_studentized.py
  .venv/bin/python code/marginal/dgp_capture/run_capture_studentized.py \\
      --alphas 0.05,0.1,0.2 --configs fixedN21,poissonNmean21 --total_replicates 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture import run_capture as rc
from code.marginal.run_true_marginal_latent_intercept_experiments import (
    BASE_SEED,
    DEFAULT_GAMMA,
    build_configs,
)
from code.shared.plot_engine import PAPER_ALPHA_GRID_STR


def _configure() -> None:
    rc.SCORE_TYPE = "studentized"
    rc.WITHIN_GROUP_MODE = "mean"
    rc.PREDICTOR = "rf"
    rc.RF_NTREE = 50
    rc.RF_NODESIZE = 5
    rc.RF_MTRY = None
    # Results-only: skip heavy μ-capture CSV I/O (was crashing mid-run when dirs
    # were wiped, and dominated runtime vs RF fits). Plots only need raw_results.
    rc.CAPTURE_MU = False
    rc.PLOTS_ROOT = REPO_ROOT / "plots_marginal" / "dgp_studentized"
    rc.RESULTS_PREFIX = "true_marg_latent_rf_studentized"
    rc.SUITE_NAME = "dgp_studentized"


def parse_args():
    p = argparse.ArgumentParser(description="DGP RF studentized capture → dgp_studentized.")
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
    _configure()
    print(f"Studentized DGP suite → {rc.PLOTS_ROOT}", flush=True)
    print(f"  score_type={rc.SCORE_TYPE}  (GHCP: dual σ blended with mean w_g)", flush=True)
    print(f"  Std-CP: local RF μ + local RF-σ (half-split history)", flush=True)
    print(f"  GHCP-local: w_g=0 local RF; reuses Std-CP test-group RF when available", flush=True)
    print(f"  capture_mu={rc.CAPTURE_MU}  (False = results CSV only, no μ capture)", flush=True)
    print(f"  results prefix={rc.RESULTS_PREFIX}", flush=True)

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
            rc.run_one_config(name, configs[name], args.n_workers)


if __name__ == "__main__":
    main()
