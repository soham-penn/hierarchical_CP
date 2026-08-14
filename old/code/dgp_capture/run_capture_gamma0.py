#!/usr/bin/env python3
"""
One-off wrapper: run the DGP RF capture pipeline at gamma=0, writing outputs
to plots_marginal/dgp_true_marginal_rf_gamma0 instead of the default
plots_marginal/dgp_true_marginal_rf (which holds the gamma=5 results and
must not be overwritten).

Mirrors run_capture.py exactly, only redirecting PLOTS_ROOT.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.dgp_capture import run_capture as rc
from code.marginal.run_true_marginal_latent_intercept_experiments import build_configs

# Redirect output root so this never touches the gamma=5 capture data.
rc.PLOTS_ROOT = REPO_ROOT / "plots_marginal" / "dgp_true_marginal_rf_gamma0"

GAMMA = 0.0
ALPHAS = [0.05, 0.075, 0.10, 0.125]
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
