#!/usr/bin/env python3
"""
DGP capture with the analytical near-Bayes global predictor

    mu*(x, u) = E[Y | X, U] = u_d^2 + Sigma_yx(u) Sigma_xx(u)^{-1} (x - mu_X(u))

under the known joint-XY latent-intercept DGP (rho known; B_j marginalized).
This is the MSE-optimal global predictor; Bayes risk ≈ 26.70 at gamma=5.

Writes to plots_marginal/dgp_true_marginal_bayes and results prefix
true_marg_latent_bayes — never touches the RF / u_d runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture import run_capture as rc
from code.marginal.run_true_marginal_latent_intercept_experiments import build_configs

rc.PLOTS_ROOT = REPO_ROOT / "plots_marginal" / "dgp_true_marginal_bayes"
rc.RESULTS_PREFIX = "true_marg_latent_bayes"
rc.SUITE_NAME = "dgp_true_marginal_bayes"
rc.PREDICTOR = "bayes"

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
