#!/usr/bin/env python3
"""
Strong RF + local-RF within-group correction (replaces sample-mean shrink).

Default: gamma=0, alpha=0.2, absolute residual score, B=200.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--replicates", type=int, default=200)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--score", default="absolute", choices=["absolute", "studentized"])
    args = p.parse_args()

    gtag = str(float(args.gamma)).replace(".", "p")
    rc.PLOTS_ROOT = (
        REPO_ROOT
        / "plots_marginal"
        / f"dgp_true_marginal_strong_rf_localrf_gamma{gtag}"
    )
    rc.RESULTS_PREFIX = "true_marg_latent_strong_rf_localrf"
    rc.SUITE_NAME = f"dgp_true_marginal_strong_rf_localrf_gamma{gtag}"
    rc.RF_NTREE = 100
    rc.RF_NODESIZE = 5
    rc.RF_MTRY = 0.6
    rc.SCORE_TYPE = args.score
    rc.WITHIN_GROUP_MODE = "local_rf"

    print(
        f"\n>>> Strong RF + local_rf within  gamma={args.gamma}  alpha={args.alpha}  "
        f"score={args.score}  B={args.replicates}\n  -> {rc.PLOTS_ROOT}\n",
        flush=True,
    )
    configs = build_configs(
        total_replicates=args.replicates,
        alpha=args.alpha,
        gamma=args.gamma,
        quantile_mode="deterministic",
        quantile_base_seed=rc.BASE_SEED,
    )
    rc.run_one_config("fixedN21", configs["fixedN21"], args.n_workers)


if __name__ == "__main__":
    main()
