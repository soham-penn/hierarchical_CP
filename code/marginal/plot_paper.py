#!/usr/bin/env python3
"""Unified paper plotting for true-marginal experiments.

Suites:
  dgp_ols            - standard joint-XY DGP, OLS global predictor
  dgp_rf             - latent intercept gamma=5, RF global predictor
  dgp_latent_gamma5  - latent intercept gamma=5, OLS global predictor
  acs                - ACS true-marginal (permuted rows)
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.paths import PLOTS_MARGINAL, RESULTS_ACS_MARGINAL, RESULTS_DGP_MARGINAL

_pm_path = REPO_ROOT / "code" / "shared" / "plot_engine.py"
_spec = importlib.util.spec_from_file_location("plot_engine", _pm_path)
pm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pm)


def _alpha_from_glob(path: Path) -> float:
    match = re.search(r"_alpha([0-9p.]+)/", str(path))
    if not match:
        raise ValueError(f"Cannot parse alpha from {path}")
    return float(match.group(1).replace("p", ".")) / 100.0


def _raw_files_prefix(prefix: str, dataset: str) -> list[Path]:
    by_alpha: dict[float, Path] = {}
    root = RESULTS_DGP_MARGINAL
    for path in sorted(root.glob(f"{prefix}_{dataset}_alpha*/*_raw_results_complete.csv")):
        alpha = _alpha_from_glob(path)
        if alpha not in by_alpha:
            by_alpha[alpha] = path
    files = [by_alpha[a] for a in sorted(by_alpha)]
    if not files:
        raise FileNotFoundError(f"No results for {prefix}_{dataset}_alpha* under {root}")
    return files


def _patch_dgp(prefix: str, plot_subdir: str, title_suffix: str = "") -> None:
    paper_root = PLOTS_MARGINAL / plot_subdir
    pm.PAPER_ROOT = paper_root
    pm.FIG_DIR = paper_root / "figures"
    pm.SUMMARY_DIR = paper_root / "summaries"
    pm.RESULTS_ROOT = RESULTS_DGP_MARGINAL
    pm._raw_files = lambda dataset: _raw_files_prefix(prefix, dataset)
    pm.reset_output_dirs = lambda: (
        pm.FIG_DIR.mkdir(parents=True, exist_ok=True),
        pm.SUMMARY_DIR.mkdir(parents=True, exist_ok=True),
    )


def _build_dgp_ols(include_stability: bool = True, include_baselines: bool = True) -> list[Path]:
    _patch_dgp("true_marg", "dgp_true_marginal")
    pm.reset_output_dirs()
    all_o = sorted(set(pm.O_VALUES_UPTO35))
    poisson_trials = pm.load_trials(pm.POISSON_DATASET, all_o)
    poisson_summary = pm.summarize_trials(poisson_trials)
    poisson_stability = pm.load_stability_summary(pm.POISSON_DATASET) if include_stability else None
    fixed_trials = pm.load_trials(pm.FIXED_DATASET, all_o)
    fixed_summary = pm.summarize_trials(fixed_trials)
    baseline_trials_poisson = baseline_summary_poisson = None
    baseline_trials_fixed = baseline_summary_fixed = None
    if include_baselines:
        baseline_trials_poisson = pm.load_trials(pm.POISSON_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS)
        baseline_trials_fixed = pm.load_trials(pm.FIXED_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS)
        baseline_summary_poisson = pm.summarize_trials(baseline_trials_poisson)
        baseline_summary_fixed = pm.summarize_trials(baseline_trials_fixed)

    outputs: list[Path] = []
    outputs.extend(pm.write_summary_files(pm.POISSON_DATASET, poisson_trials, poisson_summary, poisson_stability))
    outputs.extend(pm.write_summary_files(pm.FIXED_DATASET, fixed_trials, fixed_summary))
    label_p, label_f = "Poisson", r"Fixed $N_k=21$"
    outputs += [
        pm.plot_coverage_lines_width_boxplots(poisson_trials, poisson_summary, pm.O_VALUES,
            "poisson_1_dhcp_coverage_lines_width_boxplots_by_alpha.pdf", label_p),
        pm.plot_coverage_width_line_bands(poisson_summary, pm.O_VALUES,
            "poisson_2_dhcp_coverage_width_line_bands_by_alpha.pdf", label_p),
        pm.plot_with_vs_no_within_alpha10(poisson_trials, poisson_summary, label_p, "poisson_4" if include_stability else "poisson_3"),
        pm.plot_coverage_lines_width_boxplots(
            fixed_trials[fixed_trials["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf", label_f),
        pm.plot_coverage_width_line_bands(
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf", label_f),
        pm.plot_with_vs_no_within_alpha10(fixed_trials, fixed_summary, label_f, "fixedN21_3"),
    ]
    if include_stability and poisson_stability is not None:
        outputs.append(pm.plot_randomization_stability(poisson_stability))
    for alpha in pm.FIXED_ALPHA_PANEL_VALUES:
        outputs.append(pm.plot_alpha_o_axis_panel(fixed_trials, fixed_summary, alpha, label_f, "fixedN21"))
        outputs.append(pm.plot_alpha_o_axis_panel(poisson_trials, poisson_summary, alpha, label_p, "poisson"))
        if include_baselines and baseline_trials_fixed is not None:
            outputs.extend(pm.plot_all_baselines_by_o(
                baseline_trials_fixed, baseline_summary_fixed, alpha, pm.O_VALUES, "fixedN21", label_f))
            outputs.extend(pm.plot_all_baselines_by_o(
                baseline_trials_poisson, baseline_summary_poisson, alpha, pm.O_VALUES, "poisson", label_p))
    outputs.append(pm.plot_alpha_o_axis_panel(
        fixed_trials, fixed_summary, 0.10, label_f, "fixedN21",
        o_values=pm.O_VALUES_UPTO35, out_suffix="_upto35"))
    if include_baselines:
        outputs.append(pm.plot_alpha_o_axis_panel(
            fixed_trials, fixed_summary, 0.10, label_f, "fixedN21",
            o_values=pm.O_VALUES_UPTO35, out_suffix="_with_stdcp", include_stdcp=True))
    return outputs


def _build_dgp_rf() -> list[Path]:
    _patch_dgp("true_marg_latent_rf_gamma5p0", "dgp_true_marginal_rf")
    pm.reset_output_dirs()
    rf_label = "RF"
    all_o = sorted(set(pm.O_VALUES_UPTO35))
    poisson_trials = pm.load_trials(pm.POISSON_DATASET, all_o)
    poisson_summary = pm.summarize_trials(poisson_trials)
    fixed_trials = pm.load_trials(pm.FIXED_DATASET, all_o)
    fixed_summary = pm.summarize_trials(fixed_trials)
    baseline_trials_poisson = pm.load_trials(pm.POISSON_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS)
    baseline_trials_fixed = pm.load_trials(pm.FIXED_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS)
    baseline_summary_poisson = pm.summarize_trials(baseline_trials_poisson)
    baseline_summary_fixed = pm.summarize_trials(baseline_trials_fixed)
    outputs: list[Path] = []
    outputs.extend(pm.write_summary_files(pm.POISSON_DATASET, poisson_trials, poisson_summary))
    outputs.extend(pm.write_summary_files(pm.FIXED_DATASET, fixed_trials, fixed_summary))
    outputs += [
        pm.plot_coverage_lines_width_boxplots(poisson_trials, poisson_summary, pm.O_VALUES,
            "poisson_1_dhcp_coverage_lines_width_boxplots_by_alpha.pdf", f"Poisson ({rf_label})"),
        pm.plot_coverage_width_line_bands(poisson_summary, pm.O_VALUES,
            "poisson_2_dhcp_coverage_width_line_bands_by_alpha.pdf", f"Poisson ({rf_label})"),
        pm.plot_with_vs_no_within_alpha10(poisson_trials, poisson_summary, f"Poisson ({rf_label})", "poisson_3"),
        pm.plot_coverage_lines_width_boxplots(
            fixed_trials[fixed_trials["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf",
            rf"Fixed $N_k=21$ ({rf_label})"),
        pm.plot_coverage_width_line_bands(
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf",
            rf"Fixed $N_k=21$ ({rf_label})"),
        pm.plot_with_vs_no_within_alpha10(fixed_trials, fixed_summary, rf"Fixed $N_k=21$ ({rf_label})", "fixedN21_3"),
    ]
    for alpha in pm.FIXED_ALPHA_PANEL_VALUES:
        outputs.append(pm.plot_alpha_o_axis_panel(fixed_trials, fixed_summary, alpha, rf"Fixed $N_k=21$ ({rf_label})", "fixedN21"))
        outputs.append(pm.plot_alpha_o_axis_panel(poisson_trials, poisson_summary, alpha, f"Poisson ({rf_label})", "poisson"))
        outputs.extend(pm.plot_all_baselines_by_o(
            baseline_trials_fixed, baseline_summary_fixed, alpha, pm.O_VALUES, "fixedN21", rf"Fixed $N_k=21$ ({rf_label})"))
        outputs.extend(pm.plot_all_baselines_by_o(
            baseline_trials_poisson, baseline_summary_poisson, alpha, pm.O_VALUES, "poisson", f"Poisson ({rf_label})"))
    outputs.append(pm.plot_alpha_o_axis_panel(
        fixed_trials, fixed_summary, 0.10, rf"Fixed $N_k=21$ ({rf_label})", "fixedN21",
        o_values=pm.O_VALUES_UPTO35, out_suffix="_upto35"))
    return outputs


def _build_dgp_latent_gamma5() -> list[Path]:
    prefix = "true_marg_latent_gamma5p0"
    _patch_dgp(prefix, "dgp_true_marginal_latent_gamma5")
    pm.reset_output_dirs()
    all_o = sorted(set(pm.O_VALUES_UPTO35))
    poisson_trials = pm.load_trials(pm.POISSON_DATASET, all_o)
    poisson_summary = pm.summarize_trials(poisson_trials)
    fixed_trials = pm.load_trials(pm.FIXED_DATASET, all_o)
    fixed_summary = pm.summarize_trials(fixed_trials)
    baseline_trials_poisson = pm.load_trials(pm.POISSON_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS)
    baseline_trials_fixed = pm.load_trials(pm.FIXED_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS)
    baseline_summary_poisson = pm.summarize_trials(baseline_trials_poisson)
    baseline_summary_fixed = pm.summarize_trials(baseline_trials_fixed)
    label_p = "Poisson"
    label_f = r"Fixed $N_k=21$"
    outputs: list[Path] = []
    outputs.extend(pm.write_summary_files(pm.POISSON_DATASET, poisson_trials, poisson_summary))
    outputs.extend(pm.write_summary_files(pm.FIXED_DATASET, fixed_trials, fixed_summary))
    outputs += [
        pm.plot_coverage_lines_width_boxplots(poisson_trials, poisson_summary, pm.O_VALUES,
            "poisson_1_dhcp_coverage_lines_width_boxplots_by_alpha.pdf", label_p),
        pm.plot_coverage_width_line_bands(poisson_summary, pm.O_VALUES,
            "poisson_2_dhcp_coverage_width_line_bands_by_alpha.pdf", label_p),
        pm.plot_with_vs_no_within_alpha10(poisson_trials, poisson_summary, label_p, "poisson_3"),
        pm.plot_coverage_lines_width_boxplots(
            fixed_trials[fixed_trials["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf", label_f),
        pm.plot_coverage_width_line_bands(
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf", label_f),
        pm.plot_with_vs_no_within_alpha10(fixed_trials, fixed_summary, label_f, "fixedN21_3"),
    ]
    for alpha in pm.FIXED_ALPHA_PANEL_VALUES:
        outputs.append(pm.plot_alpha_o_axis_panel(fixed_trials, fixed_summary, alpha, label_f, "fixedN21"))
        outputs.append(pm.plot_alpha_o_axis_panel(poisson_trials, poisson_summary, alpha, label_p, "poisson"))
        outputs.extend(pm.plot_all_baselines_by_o(
            baseline_trials_fixed, baseline_summary_fixed, alpha, pm.O_VALUES, "fixedN21", label_f))
        outputs.extend(pm.plot_all_baselines_by_o(
            baseline_trials_poisson, baseline_summary_poisson, alpha, pm.O_VALUES, "poisson", label_p))
    outputs.append(pm.plot_alpha_o_axis_panel(
        fixed_trials, fixed_summary, 0.10, label_f, "fixedN21",
        o_values=pm.O_VALUES_UPTO35, out_suffix="_upto35"))
    return outputs


def _build_acs() -> list[Path]:
    acs_plot = REPO_ROOT / "real_data" / "acs" / "plot_dgp_style_paper_plots.py"
    spec = importlib.util.spec_from_file_location("acs_plot", acs_plot)
    acs = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    acs.RESULTS_ROOT = RESULTS_ACS_MARGINAL
    acs.PAPER_ROOT = PLOTS_MARGINAL / "acs"
    acs.FIG_DIR = acs.PAPER_ROOT / "figures"
    acs.SUMMARY_DIR = acs.PAPER_ROOT / "summaries"
    spec.loader.exec_module(acs)
    acs.RESULTS_ROOT = RESULTS_ACS_MARGINAL
    acs.PAPER_ROOT = PLOTS_MARGINAL / "acs"
    acs.FIG_DIR = acs.PAPER_ROOT / "figures"
    acs.SUMMARY_DIR = acs.PAPER_ROOT / "summaries"
    acs.main()
    return list(acs.FIG_DIR.glob("*.pdf"))


SUITES: dict[str, Callable[[], list[Path]]] = {
    "dgp_ols": lambda: _build_dgp_ols(include_stability=True, include_baselines=True),
    "dgp_rf": _build_dgp_rf,
    "dgp_latent_gamma5": _build_dgp_latent_gamma5,
    "acs": _build_acs,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate true-marginal paper plots.")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="dgp_ols",
        help="Which experiment suite to plot.",
    )
    parser.add_argument("--all", action="store_true", help="Run all marginal suites.")
    args = parser.parse_args()
    suites = list(SUITES) if args.all else [args.suite]
    for name in suites:
        print(f"\n=== Suite: {name} ===")
        outputs = SUITES[name]()
        for path in outputs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
