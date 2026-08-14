#!/usr/bin/env python3
"""Strong-RF + studentized score at gamma=0, alpha=0.2 (fixedN21)."""

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
    p.add_argument("--score", default="studentized")
    args = p.parse_args()

    gtag = str(float(args.gamma)).replace(".", "p")
    atag = f"alpha{int(round(100 * args.alpha))}" if abs(100 * args.alpha - round(100 * args.alpha)) < 1e-9 else None
    score = str(args.score)

    rc.PLOTS_ROOT = (
        REPO_ROOT
        / "plots_marginal"
        / f"dgp_true_marginal_strong_rf_{score}_gamma{gtag}"
    )
    rc.RESULTS_PREFIX = f"true_marg_latent_strong_rf_{score}"
    rc.SUITE_NAME = f"dgp_true_marginal_strong_rf_{score}_gamma{gtag}"
    rc.RF_NTREE = 100
    rc.RF_NODESIZE = 5
    rc.RF_MTRY = 0.6
    rc.SCORE_TYPE = score

    print(
        f"\n>>> Strong RF + {score}  gamma={args.gamma}  alpha={args.alpha}  "
        f"B={args.replicates} -> {rc.PLOTS_ROOT}\n",
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
