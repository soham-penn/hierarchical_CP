#!/usr/bin/env python3
"""
One-off wrapper: run the DGP capture pipeline with a STRONGER global RF, writing
outputs to plots_marginal/dgp_true_marginal_strong_rf (and a distinct results
prefix) so it never touches the default weak-RF gamma=5 run.

Stronger RF config (chosen by benchmarking test MSE on this DGP at the actual
HCP training size, 19 groups x 21 obs, averaged over 8 seeds):

    ntree=100, nodesize=5, mtry=0.6  (max_features = 60% of features per split)

Rationale:
  * The default RF (ntree=50, nodesize=5, mtry=sqrt(p)=3/9) under-uses the
    informative U features (response is ~ u_d^2), so many splits miss u_d.
  * Raising mtry to 0.6 lets each split see enough features to find u_d, which
    is the dominant MSE lever here. At gamma=5 pushing mtry all the way to 1.0
    over-fits each training group's latent shift, so 0.6 is the sweet spot.
  * ntree=100 captures essentially all of the ensemble-averaging gain; 500 trees
    were no better (and 5x slower).

Benchmarked test MSE (train 19x21):
    default(50,5,sqrt): gamma=5 -> 58.3,  gamma=0 -> 19.0
    strong (100,5,0.6): gamma=5 -> 52.9,  gamma=0 ->  9.4

Mirrors run_capture.py exactly, only redirecting output roots + RF params.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture import run_capture as rc
from code.marginal.run_true_marginal_latent_intercept_experiments import build_configs

# Redirect output roots so this never touches the default weak-RF run.
rc.PLOTS_ROOT = REPO_ROOT / "plots_marginal" / "dgp_true_marginal_strong_rf"
rc.RESULTS_PREFIX = "true_marg_latent_strong_rf"
rc.SUITE_NAME = "dgp_true_marginal_strong_rf"

# Stronger global RF.
rc.RF_NTREE = 100
rc.RF_NODESIZE = 5
rc.RF_MTRY = 0.6

GAMMA = 5.0
ALPHAS = [0.05, 0.075, 0.10, 0.125, 0.15]
CONFIG_NAMES = ["fixedN21", "poissonNmean21"]
TOTAL_REPLICATES = 1000
N_WORKERS = 8


def main():
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


if __name__ == "__main__":
    main()
