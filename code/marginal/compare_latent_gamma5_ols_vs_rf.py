#!/usr/bin/env python3
"""Compare OLS vs RF true-marginal latent-intercept (gamma=5) fixedN21 results."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_pm_path = REPO_ROOT / "code" / "shared" / "plot_engine.py"
_spec = importlib.util.spec_from_file_location("plot_engine", _pm_path)
pm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pm)

GAMMA = 5.0
ALPHA = 0.10
DATASET = "fixedN21"
OLS_PATH = (
    REPO_ROOT
    / "results_marginal"
    / "dgp"
    / f"true_marg_latent_gamma5p0_{DATASET}_alpha10"
    / f"true_marg_latent_gamma5p0_{DATASET}_alpha10_raw_results_complete.csv"
)
RF_PATH = (
    REPO_ROOT
    / "results_marginal"
    / "dgp"
    / f"true_marg_latent_rf_gamma5p0_{DATASET}_alpha10"
    / f"true_marg_latent_rf_gamma5p0_{DATASET}_alpha10_raw_results_complete.csv"
)


def _load_trials_from_path(path: Path) -> pd.DataFrame:
    alpha = pm._alpha_from_path(path)
    raw = pd.read_csv(path)
    rows = []
    for key, method in pm.METHOD_KEYS.items():
        cov_col = f"coverage_{key}"
        width_col = f"width_{key}"
        if cov_col not in raw.columns or width_col not in raw.columns:
            continue
        part = raw[["experiment", "o_observed", cov_col, width_col]].copy()
        part.columns = ["experiment", "o", "coverage", "width"]
        part["method"] = method
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    df["dataset"] = DATASET
    df["alpha"] = alpha
    df["nominal_coverage"] = 1.0 - alpha
    return df[df["o"].isin(pm.O_VALUES)].copy()


def summarize(path: Path, label: str) -> pd.DataFrame:
    trials = _load_trials_from_path(path)
    return pm.summarize_trials(trials).assign(predictor=label)


def main() -> None:
    if not OLS_PATH.exists():
        raise FileNotFoundError(f"Missing OLS results: {OLS_PATH}")
    if not RF_PATH.exists():
        raise FileNotFoundError(f"Missing RF results: {RF_PATH}")

    ols = summarize(OLS_PATH, "OLS")
    rf = summarize(RF_PATH, "RF")
    methods = ["D-HCP", "D-HCP no within", "HCP"]

    print(f"gamma={GAMMA}, {DATASET}, alpha={ALPHA}\n")
    print(f"{'method':<18} {'o':>3}  {'OLS width':>10} {'RF width':>10} {'ratio':>7}  {'OLS cov':>8} {'RF cov':>8}")
    print("-" * 78)
    for method in methods:
        for o in pm.O_VALUES:
            ols_row = ols[(ols.method == method) & (ols.o == o) & (ols.alpha == ALPHA)]
            rf_row = rf[(rf.method == method) & (rf.o == o) & (rf.alpha == ALPHA)]
            if ols_row.empty or rf_row.empty:
                continue
            ow = float(ols_row.width_mean.iloc[0])
            rw = float(rf_row.width_mean.iloc[0])
            oc = float(ols_row.coverage_mean.iloc[0])
            rc = float(rf_row.coverage_mean.iloc[0])
            ratio = rw / ow if ow > 0 else float("nan")
            print(f"{method:<18} {o:>3}  {ow:10.2f} {rw:10.2f} {ratio:7.2f}  {oc:8.3f} {rc:8.3f}")
        print()


if __name__ == "__main__":
    main()
