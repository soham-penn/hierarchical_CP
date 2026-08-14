#!/usr/bin/env python3
"""
DGP capture with near-Bayes global predictor mu(x, u) = u_d^2.

Under the joint-XY latent-intercept DGP, E[Y | U, B=0] = u_d^2, and X adds
almost no MSE reduction, so this is essentially the Bayes-optimal *global*
predictor (still irreducible gamma^2 from the unobserved group intercept).

Writes to plots_marginal/dgp_true_marginal_u_d and results prefix
true_marg_latent_ud — never touches the RF runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture import run_capture as rc
from code.marginal.run_true_marginal_latent_intercept_experiments import build_configs

rc.PLOTS_ROOT = REPO_ROOT / "plots_marginal" / "dgp_true_marginal_u_d"
rc.RESULTS_PREFIX = "true_marg_latent_ud"
rc.SUITE_NAME = "dgp_true_marginal_u_d"
rc.PREDICTOR = "ud_squared"

GAMMA = 5.0
# Match the RF / strong-RF paper alphas used for width/coverage panels.
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
