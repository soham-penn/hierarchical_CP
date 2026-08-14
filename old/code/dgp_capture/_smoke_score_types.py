#!/usr/bin/env python3
"""Smoke test for strong-RF alternate scores."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from code.marginal.dgp_capture import run_capture as rc
from code.marginal.run_true_marginal_latent_intercept_experiments import build_configs

rc.RF_NTREE = 100
rc.RF_NODESIZE = 5
rc.RF_MTRY = 0.6


def main():
    for score in ["studentized", "cqr"]:
        rc.SCORE_TYPE = score
        rc.PLOTS_ROOT = REPO / "plots_marginal" / f"_smoke_strong_rf_{score}"
        rc.RESULTS_PREFIX = f"_smoke_strong_rf_{score}"
        rc.SUITE_NAME = f"_smoke_strong_rf_{score}"
        cfg = build_configs(
            total_replicates=2,
            alpha=0.1,
            gamma=5.0,
            quantile_mode="deterministic",
            quantile_base_seed=rc.BASE_SEED,
        )["fixedN21"]
        print("running", score, flush=True)
        rc.run_one_config("fixedN21", cfg, n_workers=1)
        raw = list((rc.PLOTS_ROOT / "capture_data").rglob("raw_results_mean.csv"))[0]
        df = pd.read_csv(raw)
        print(score, "rows", len(df), flush=True)
        print("  GHCP width med", df.groupby("o_observed")["width_donor_hcp_randomized"].median().to_dict(), flush=True)
        print("  StdCP width med", df.groupby("o_observed")["width_stdcp"].median().to_dict(), flush=True)
        print("  GHCP cov", df.groupby("o_observed")["coverage_donor_hcp_randomized"].mean().to_dict(), flush=True)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
