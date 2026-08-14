#!/usr/bin/env python3
"""
Strong-RF DGP capture with alternate nonconformity scores.

Writes to separate plot roots so residual-score strong-RF results are untouched:
  plots_marginal/dgp_true_marginal_strong_rf_studentized/
  plots_marginal/dgp_true_marginal_strong_rf_cqr/

Usage:
  python code/marginal/dgp_capture/run_capture_strong_rf_scores.py --score studentized
  python code/marginal/dgp_capture/run_capture_strong_rf_scores.py --score cqr
  python code/marginal/dgp_capture/run_capture_strong_rf_scores.py --score both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture import run_capture as rc
from code.marginal.run_true_marginal_latent_intercept_experiments import build_configs

GAMMA = 5.0
# Focused grid for score comparison (full residual suite already exists).
ALPHAS = [0.10]
CONFIG_NAMES = ["fixedN21"]
TOTAL_REPLICATES = 200
N_WORKERS = 8


def _configure(score_type: str) -> None:
    tag = score_type
    rc.PLOTS_ROOT = REPO_ROOT / "plots_marginal" / f"dgp_true_marginal_strong_rf_{tag}"
    rc.RESULTS_PREFIX = f"true_marg_latent_strong_rf_{tag}"
    rc.SUITE_NAME = f"dgp_true_marginal_strong_rf_{tag}"
    rc.RF_NTREE = 100
    rc.RF_NODESIZE = 5
    rc.RF_MTRY = 0.6
    rc.SCORE_TYPE = score_type


def _run_one(score_type: str) -> None:
    _configure(score_type)
    print(f"\n>>> Strong RF + score_type={score_type} -> {rc.PLOTS_ROOT}\n", flush=True)
    for alpha in ALPHAS:
        configs = build_configs(
            total_replicates=TOTAL_REPLICATES,
            alpha=alpha,
            gamma=GAMMA,
            quantile_mode="deterministic",
            quantile_base_seed=rc.BASE_SEED,
        )
        for name in CONFIG_NAMES:
            rc.run_one_config(name, configs[name], N_WORKERS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--score",
        choices=["studentized", "cqr", "both"],
        default="both",
    )
    p.add_argument("--alphas", nargs="+", type=float, default=None)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--replicates", type=int, default=None)
    args = p.parse_args()

    global ALPHAS, N_WORKERS, TOTAL_REPLICATES
    if args.alphas is not None:
        ALPHAS = list(args.alphas)
    if args.n_workers is not None:
        N_WORKERS = int(args.n_workers)
    if args.replicates is not None:
        TOTAL_REPLICATES = int(args.replicates)

    scores = ["studentized", "cqr"] if args.score == "both" else [args.score]
    for st in scores:
        _run_one(st)


if __name__ == "__main__":
    main()
