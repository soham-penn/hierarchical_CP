#!/usr/bin/env python3
"""Unified paper plotting for true-marginal experiments.

Main paper Simulations suite (RF, absolute score, gamma=5):
  dgp_rf             - plots_marginal/dgp_true_marginal_rf
                       results: true_marg_latent_rf_gamma5p0_*

Other suites regenerate under plots_marginal/<subdir>/ (archived copies live
in plots_marginal/old/ and results_marginal/dgp/old/).

  dgp_ols / dgp_latent_gamma5 / dgp_rf_gamma0 / dgp_rf_strong / ...
  dgp_studentized / dgp_studentized_wg0 / dgp_ud / dgp_bayes
  acs / acs_xgboost
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


def _append_stdcp_o_panels(
    outputs: list[Path],
    fixed_trials: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    poisson_trials: pd.DataFrame,
    poisson_summary: pd.DataFrame,
    *,
    alphas: tuple[float, ...] = (0.05, 0.10),
    fixed_label: str = r"Fixed $N_k=21$",
    poisson_label: str = "Poisson",
) -> None:
    """GHCP vs Std-CP vs HCP coverage/width panels (o up to 35)."""
    for alpha in alphas:
        outputs.append(
            pm.plot_alpha_o_axis_panel(
                fixed_trials,
                fixed_summary,
                alpha,
                fixed_label,
                "fixedN21",
                o_values=pm.O_VALUES_UPTO35,
                out_suffix="_with_stdcp",
                include_stdcp=True,
            )
        )
        outputs.append(
            pm.plot_alpha_o_axis_panel(
                poisson_trials,
                poisson_summary,
                alpha,
                poisson_label,
                "poisson",
                o_values=pm.O_VALUES_UPTO35,
                out_suffix="_with_stdcp",
                include_stdcp=True,
            )
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
    _append_stdcp_o_panels(outputs, fixed_trials, fixed_summary, poisson_trials, poisson_summary,
                           fixed_label=label_f, poisson_label=label_p)
    return outputs


def _build_dgp_rf_family(
    *,
    prefix: str,
    plot_subdir: str,
    rf_label: str,
) -> list[Path]:
    _patch_dgp(prefix, plot_subdir)
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
    _append_stdcp_o_panels(
        outputs,
        fixed_trials,
        fixed_summary,
        poisson_trials,
        poisson_summary,
        alphas=(0.05, 0.10),
        fixed_label=rf"Fixed $N_k=21$ ({rf_label})",
        poisson_label=f"Poisson ({rf_label})",
    )
    return outputs


def _build_dgp_rf() -> list[Path]:
    outputs = _build_dgp_rf_family(
        prefix="true_marg_latent_rf_gamma5p0",
        plot_subdir="dgp_true_marginal_rf",
        rf_label="RF",
    )
    # Paper Simulation tables (mean width) from the same summaries.
    from code.marginal.export_paper_tables_mean_width import main as _export_tables

    _export_tables()
    tables = PLOTS_MARGINAL / "dgp_true_marginal_rf" / "tables" / "simulations_mean_width_tables.tex"
    if tables.exists():
        outputs.append(tables)
    return outputs


def _build_dgp_rf_gamma0() -> list[Path]:
    return _build_dgp_rf_family(
        prefix="true_marg_latent_rf_gamma0p0",
        plot_subdir="dgp_true_marginal_rf_gamma0",
        rf_label=r"RF, $\gamma=0$",
    )


def _build_dgp_rf_strong() -> list[Path]:
    return _build_dgp_rf_family(
        prefix="true_marg_latent_strong_rf_gamma5p0",
        plot_subdir="dgp_true_marginal_strong_rf",
        rf_label=r"Strong RF",
    )


def _build_dgp_studentized() -> list[Path]:
    return _build_dgp_rf_family(
        prefix="true_marg_latent_rf_studentized_gamma5p0",
        plot_subdir="dgp_studentized",
        rf_label=r"RF, studentized",
    )


def _build_dgp_studentized_wg0() -> list[Path]:
    """Fixed-N α=0.1 only: GHCP with w_g=0 (local half-mean) + Std-CP."""
    _patch_dgp("true_marg_latent_rf_studentized_wg0_gamma5p0", "dgp_studentized_wg0")
    pm.reset_output_dirs()
    all_o = sorted(set(pm.O_VALUES_UPTO35))
    label = rf"Fixed $N_k=21$ (studentized, $w_g=0$)"
    fixed_trials = pm.load_trials(pm.FIXED_DATASET, all_o)
    fixed_summary = pm.summarize_trials(fixed_trials)
    baseline_trials = pm.load_trials(pm.FIXED_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS)
    baseline_summary = pm.summarize_trials(baseline_trials)
    outputs: list[Path] = []
    outputs.extend(pm.write_summary_files(pm.FIXED_DATASET, fixed_trials, fixed_summary))
    outputs += [
        pm.plot_coverage_lines_width_boxplots(
            fixed_trials[fixed_trials["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf",
            label,
        ),
        pm.plot_coverage_width_line_bands(
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf",
            label,
        ),
        pm.plot_with_vs_no_within_alpha10(fixed_trials, fixed_summary, label, "fixedN21_3"),
        pm.plot_alpha_o_axis_panel(fixed_trials, fixed_summary, 0.10, label, "fixedN21"),
        pm.plot_alpha_o_axis_panel(
            fixed_trials, fixed_summary, 0.10, label, "fixedN21",
            o_values=pm.O_VALUES_UPTO35, out_suffix="_upto35",
        ),
        pm.plot_alpha_o_axis_panel(
            fixed_trials, fixed_summary, 0.10, label, "fixedN21",
            o_values=pm.O_VALUES_UPTO35, out_suffix="_with_stdcp", include_stdcp=True,
        ),
    ]
    outputs.extend(pm.plot_all_baselines_by_o(
        baseline_trials, baseline_summary, 0.10, pm.O_VALUES, "fixedN21", label,
    ))
    return outputs


def _build_dgp_ud() -> list[Path]:
    return _build_dgp_rf_family(
        prefix="true_marg_latent_ud_gamma5p0",
        plot_subdir="dgp_true_marginal_u_d",
        rf_label=r"$u_d^2$",
    )


def _build_dgp_bayes() -> list[Path]:
    return _build_dgp_rf_family(
        prefix="true_marg_latent_bayes_gamma5p0",
        plot_subdir="dgp_true_marginal_bayes",
        rf_label=r"Bayes $E[Y\mid X,U]$",
    )


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
    hcp_trials_poisson = pm.load_trials(pm.POISSON_DATASET, pm.O_VALUES, method_keys=pm.DHCP_SHCP_HCP_KEYS)
    hcp_trials_fixed = pm.load_trials(pm.FIXED_DATASET, pm.O_VALUES, method_keys=pm.DHCP_SHCP_HCP_KEYS)
    hcp_summary_poisson = pm.summarize_trials(hcp_trials_poisson)
    hcp_summary_fixed = pm.summarize_trials(hcp_trials_fixed)
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
        outputs.extend(pm.plot_dhcp_shcp_hcp_by_o(
            hcp_trials_fixed, hcp_summary_fixed, alpha, pm.FIXED_NOMINAL_O_VALUES, "fixedN21", label_f))
        outputs.extend(pm.plot_dhcp_shcp_hcp_by_o(
            hcp_trials_poisson, hcp_summary_poisson, alpha, pm.FIXED_NOMINAL_O_VALUES, "poisson", label_p))
    outputs.append(pm.plot_alpha_o_axis_panel(
        fixed_trials, fixed_summary, 0.10, label_f, "fixedN21",
        o_values=pm.O_VALUES_UPTO35, out_suffix="_upto35"))
    _append_stdcp_o_panels(outputs, fixed_trials, fixed_summary, poisson_trials, poisson_summary,
                           fixed_label=label_f, poisson_label=label_p)
    return outputs


def _build_acs() -> list[Path]:
    acs_plot = REPO_ROOT / "real_data" / "acs" / "plot_dgp_style_paper_plots.py"
    spec = importlib.util.spec_from_file_location("acs_plot", acs_plot)
    acs = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(acs)
    acs.RESULTS_ROOT = RESULTS_ACS_MARGINAL
    acs.DEFAULT_SUITE = acs.AcsPlotSuite(
        paper_root=PLOTS_MARGINAL / "acs",
        result_glob=f"{acs.PERMUTED_PREFIX}*",
        alpha_panel_values=tuple(acs.ALPHA_PANEL_VALUES),
    )
    return acs.run_suite(acs.DEFAULT_SUITE)


def _build_acs_xgboost() -> list[Path]:
    acs_plot = REPO_ROOT / "real_data" / "acs" / "plot_dgp_style_paper_plots.py"
    spec = importlib.util.spec_from_file_location("acs_xgb_plot", acs_plot)
    acs = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(acs)
    acs.RESULTS_ROOT = RESULTS_ACS_MARGINAL
    acs.DING_XGB_SUITE = acs.AcsPlotSuite(
        paper_root=PLOTS_MARGINAL / "acs" / "xgboost",
        result_glob=f"{acs.PERMUTED_DING_XGB_PREFIX}*",
        alpha_panel_values=(0.20,),
        title_prefix="ACS (Ding XGBoost)",
        include_nominal_panels=False,
        include_upto35_panels=False,
    )
    return acs.run_suite(acs.DING_XGB_SUITE)


SUITES: dict[str, Callable[[], list[Path]]] = {
    "dgp_ols": lambda: _build_dgp_ols(include_stability=True, include_baselines=True),
    "dgp_rf": _build_dgp_rf,
    "dgp_rf_gamma0": _build_dgp_rf_gamma0,
    "dgp_rf_strong": _build_dgp_rf_strong,
    "dgp_studentized": _build_dgp_studentized,
    "dgp_studentized_wg0": _build_dgp_studentized_wg0,
    "dgp_ud": _build_dgp_ud,
    "dgp_bayes": _build_dgp_bayes,
    "dgp_latent_gamma5": _build_dgp_latent_gamma5,
    "acs": _build_acs,
    "acs_xgboost": _build_acs_xgboost,
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
