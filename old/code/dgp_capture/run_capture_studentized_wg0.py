#!/usr/bin/env python3
"""
Studentized GHCP with w_g=0 (pure local half-mean) + local Std-CP.

Equal-size (fixedN21), alpha=0.1 by default. Main GHCP column uses sample-mean
within-group with global weight forced to 0; Std-CP unchanged.
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


def _configure() -> None:
    rc.SCORE_TYPE = "studentized"
    rc.WITHIN_GROUP_MODE = "mean"
    rc.PREDICTOR = "rf"
    rc.RF_NTREE = 50
    rc.RF_NODESIZE = 5
    rc.RF_MTRY = None
    rc.CAPTURE_MU = False
    rc.W_G_OVERRIDE = 0.0  # pure local half-mean
    rc.INCLUDE_GHCP_LOCAL_RF = False  # skip extra RF-local method (speed)
    rc.PLOTS_ROOT = REPO_ROOT / "plots_marginal" / "dgp_studentized_wg0"
    rc.RESULTS_PREFIX = "true_marg_latent_rf_studentized_wg0"
    rc.SUITE_NAME = "dgp_studentized_wg0"


def parse_args():
    p = argparse.ArgumentParser(description="Studentized GHCP w_g=0 + Std-CP (fixedN21).")
    p.add_argument("--alphas", type=str, default="0.1")
    p.add_argument("--configs", type=str, default="fixedN21")
    p.add_argument("--total_replicates", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--quantile-mode", choices=("deterministic", "randomized"), default="deterministic")
    p.add_argument("--quantile-base-seed", type=int, default=BASE_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    _configure()
    print(f"Studentized GHCP w_g=0 → {rc.PLOTS_ROOT}", flush=True)
    print(f"  score_type={rc.SCORE_TYPE}  w_g_override={rc.W_G_OVERRIDE}", flush=True)
    print(f"  Std-CP: local RF (half-split); GHCP: local half-mean only", flush=True)

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
