#!/usr/bin/env python3
"""Paper plotting for calibration-conditional experiments (fixed calibration, many test groups)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.paths import PLOTS_CONDITIONAL, RESULTS_DGP_CONDITIONAL

_pm_path = REPO_ROOT / "code" / "shared" / "plot_engine.py"
_spec = importlib.util.spec_from_file_location("plot_engine", _pm_path)
pm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pm)

GAMMA = 5.0
RESULTS_BY_DATASET = {
    "fixedN21": RESULTS_DGP_CONDITIONAL / "latent_gamma5_fixedN21",
    "poissonNmean21": RESULTS_DGP_CONDITIONAL / "latent_gamma5_poissonNmean21",
}
METHOD_MAP = {
    "Donor-HCP-within": "D-HCP",
    "Donor-HCP-no-within": "D-HCP no within",
    "HCP": "HCP",
    "Std-CP": "Std-CP",
}
PAPER_ROOT = PLOTS_CONDITIONAL / "dgp_latent_intercept_gamma5"
FIG_DIR = PAPER_ROOT / "figures"
SUMMARY_DIR = PAPER_ROOT / "summaries"


def _ensure_gamma_column(df: pd.DataFrame) -> pd.DataFrame:
    if "gamma" not in df.columns and "tau_B" in df.columns:
        df = df.rename(columns={"tau_B": "gamma"})
    return df


def _raw_path(dataset: str) -> Path:
    path = RESULTS_BY_DATASET[dataset] / f"{dataset}_raw_results_complete.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw results: {path}")
    return path


def load_trials(dataset: str, o_values: list[int] | None = None) -> pd.DataFrame:
    raw = _ensure_gamma_column(pd.read_csv(_raw_path(dataset)))
    raw = raw[np.isclose(raw["gamma"], GAMMA)].copy()
    raw = raw[raw["method"].isin(METHOD_MAP)].copy()
    if "target_group" not in raw.columns:
        raise ValueError(f"Expected target_group column in {dataset} raw results")

    rows = []
    for method_key, method_label in METHOD_MAP.items():
        sub = raw[raw["method"] == method_key].copy()
        if sub.empty:
            continue
        part = sub[["experiment", "target_group", "o", "alpha", "coverage", "width"]].copy()
        part["experiment"] = part["experiment"].astype(int) * 10_000 + part["target_group"].astype(int)
        part = part.drop(columns=["target_group"])
        part.columns = ["experiment", "o", "alpha", "coverage", "width"]
        part["method"] = method_label
        rows.append(part)

    if not rows:
        raise ValueError(f"No usable trial rows for dataset={dataset}, gamma={GAMMA}")

    df = pd.concat(rows, ignore_index=True)
    df["dataset"] = dataset
    df["nominal_coverage"] = 1.0 - df["alpha"]
    df["o"] = df["o"].astype(int)
    if o_values is not None:
        df = df[df["o"].isin(o_values)].copy()
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    return df.sort_values(["alpha", "method", "o", "experiment"]).reset_index(drop=True)


def load_stability_summary(dataset: str) -> pd.DataFrame:
    path = RESULTS_BY_DATASET[dataset] / f"{dataset}_randomization_stability_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing stability summary: {path}")
    df = _ensure_gamma_column(pd.read_csv(path))
    df = df[np.isclose(df["gamma"], GAMMA)].copy()
    df = df[df["o"].isin(pm.O_VALUES)].copy()
    return df.sort_values(["alpha", "o"]).reset_index(drop=True)


def _reset_output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def build_latent_gamma5() -> list[Path]:
    pm.PAPER_ROOT = PAPER_ROOT
    pm.FIG_DIR = FIG_DIR
    pm.SUMMARY_DIR = SUMMARY_DIR
    pm.load_trials = load_trials
    pm.load_stability_summary = load_stability_summary
    pm.reset_output_dirs = _reset_output_dirs
    pm.STABILITY_ALPHAS = [0.10, 0.05]
    pm.FIXED_ALPHA_PANEL_VALUES = [0.05, 0.10]
    _reset_output_dirs()

    all_o = sorted(set(pm.O_VALUES_UPTO35))
    poisson_trials = load_trials(pm.POISSON_DATASET, all_o)
    poisson_summary = pm.summarize_trials(poisson_trials)
    poisson_stability = load_stability_summary(pm.POISSON_DATASET)
    fixed_trials = load_trials(pm.FIXED_DATASET, all_o)
    fixed_summary = pm.summarize_trials(fixed_trials)
    gamma_label = rf"$\gamma={GAMMA:g}$"

    outputs: list[Path] = []
    outputs.extend(pm.write_summary_files(pm.POISSON_DATASET, poisson_trials, poisson_summary, poisson_stability))
    outputs.extend(pm.write_summary_files(pm.FIXED_DATASET, fixed_trials, fixed_summary))
    outputs += [
        pm.plot_coverage_lines_width_boxplots(poisson_trials, poisson_summary, pm.O_VALUES,
            "poisson_1_dhcp_coverage_lines_width_boxplots_by_alpha.pdf", rf"Poisson, {gamma_label}"),
        pm.plot_coverage_width_line_bands(poisson_summary, pm.O_VALUES,
            "poisson_2_dhcp_coverage_width_line_bands_by_alpha.pdf", rf"Poisson, {gamma_label}"),
        pm.plot_randomization_stability(poisson_stability),
        pm.plot_with_vs_no_within_alpha10(poisson_trials, poisson_summary, rf"Poisson, {gamma_label}", "poisson_4"),
        pm.plot_coverage_lines_width_boxplots(
            fixed_trials[fixed_trials["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf",
            rf"Fixed $N_k=21$, {gamma_label}"),
        pm.plot_coverage_width_line_bands(
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf",
            rf"Fixed $N_k=21$, {gamma_label}"),
        pm.plot_with_vs_no_within_alpha10(fixed_trials, fixed_summary, rf"Fixed $N_k=21$, {gamma_label}", "fixedN21_3"),
    ]
    for alpha in pm.FIXED_ALPHA_PANEL_VALUES:
        outputs.append(pm.plot_alpha_o_axis_panel(fixed_trials, fixed_summary, alpha, rf"Fixed $N_k=21$, {gamma_label}", "fixedN21"))
        outputs.append(pm.plot_alpha_o_axis_panel(poisson_trials, poisson_summary, alpha, rf"Poisson, {gamma_label}", "poisson"))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibration-conditional paper plots.")
    parser.add_argument("--suite", choices=["latent_gamma5"], default="latent_gamma5")
    args = parser.parse_args()
    if args.suite == "latent_gamma5":
        outputs = build_latent_gamma5()
        print(f"Wrote plots to {PAPER_ROOT}")
        for path in outputs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
