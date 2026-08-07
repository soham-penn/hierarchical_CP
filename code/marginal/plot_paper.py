#!/usr/bin/env python3
"""Paper plotting for true-marginal experiments.

Canonical Simulations suite (paper):
  dgp_rf  — latent intercept γ=5, RF global μ, absolute residual score
            plots → plots_marginal/dgp_true_marginal_rf/
            raw   → results_marginal/dgp/true_marg_latent_rf_gamma5p0_*

ACS plotting wrappers remain available; the active real-data suite is `plots_marginal/acs/min21/`.
Alternate DGP suites (OLS, γ=0, studentized, capture, …) live under old/code/.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Callable

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


def _patch_dgp(prefix: str, plot_subdir: str) -> None:
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


def _build_dgp_rf() -> list[Path]:
    """Regenerate Simulations figures, summaries, and mean-width tables."""
    prefix = "true_marg_latent_rf_gamma5p0"
    plot_subdir = "dgp_true_marginal_rf"
    rf_label = "RF"
    _patch_dgp(prefix, plot_subdir)
    pm.reset_output_dirs()
    all_o = sorted(set(pm.O_VALUES_UPTO35))
    poisson_trials = pm.load_trials(pm.POISSON_DATASET, all_o)
    poisson_summary = pm.summarize_trials(poisson_trials)
    fixed_trials = pm.load_trials(pm.FIXED_DATASET, all_o)
    fixed_summary = pm.summarize_trials(fixed_trials)
    baseline_trials_poisson = pm.load_trials(
        pm.POISSON_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS
    )
    baseline_trials_fixed = pm.load_trials(
        pm.FIXED_DATASET, pm.O_VALUES, method_keys=pm.BASELINE_METHOD_KEYS
    )
    baseline_summary_poisson = pm.summarize_trials(baseline_trials_poisson)
    baseline_summary_fixed = pm.summarize_trials(baseline_trials_fixed)
    outputs: list[Path] = []
    outputs.extend(pm.write_summary_files(pm.POISSON_DATASET, poisson_trials, poisson_summary))
    outputs.extend(pm.write_summary_files(pm.FIXED_DATASET, fixed_trials, fixed_summary))
    outputs += [
        pm.plot_coverage_lines_width_boxplots(
            poisson_trials,
            poisson_summary,
            pm.O_VALUES,
            "poisson_1_dhcp_coverage_lines_width_boxplots_by_alpha.pdf",
            f"Poisson ({rf_label})",
        ),
        pm.plot_coverage_width_line_bands(
            poisson_summary,
            pm.O_VALUES,
            "poisson_2_dhcp_coverage_width_line_bands_by_alpha.pdf",
            f"Poisson ({rf_label})",
        ),
        pm.plot_with_vs_no_within_alpha10(
            poisson_trials, poisson_summary, f"Poisson ({rf_label})", "poisson_3"
        ),
        pm.plot_coverage_lines_width_boxplots(
            fixed_trials[fixed_trials["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf",
            rf"Fixed $N_k=21$ ({rf_label})",
        ),
        pm.plot_coverage_width_line_bands(
            fixed_summary[fixed_summary["o"].isin(pm.FIXED_NOMINAL_O_VALUES)],
            pm.FIXED_NOMINAL_O_VALUES,
            "fixedN21_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf",
            rf"Fixed $N_k=21$ ({rf_label})",
        ),
        pm.plot_with_vs_no_within_alpha10(
            fixed_trials, fixed_summary, rf"Fixed $N_k=21$ ({rf_label})", "fixedN21_3"
        ),
    ]
    for alpha in pm.FIXED_ALPHA_PANEL_VALUES:
        outputs.append(
            pm.plot_alpha_o_axis_panel(
                fixed_trials, fixed_summary, alpha, rf"Fixed $N_k=21$ ({rf_label})", "fixedN21"
            )
        )
        outputs.append(
            pm.plot_alpha_o_axis_panel(
                poisson_trials, poisson_summary, alpha, f"Poisson ({rf_label})", "poisson"
            )
        )
        outputs.extend(
            pm.plot_all_baselines_by_o(
                baseline_trials_fixed,
                baseline_summary_fixed,
                alpha,
                pm.O_VALUES,
                "fixedN21",
                rf"Fixed $N_k=21$ ({rf_label})",
            )
        )
        outputs.extend(
            pm.plot_all_baselines_by_o(
                baseline_trials_poisson,
                baseline_summary_poisson,
                alpha,
                pm.O_VALUES,
                "poisson",
                f"Poisson ({rf_label})",
            )
        )
    outputs.append(
        pm.plot_alpha_o_axis_panel(
            fixed_trials,
            fixed_summary,
            0.10,
            rf"Fixed $N_k=21$ ({rf_label})",
            "fixedN21",
            o_values=pm.O_VALUES_UPTO35,
            out_suffix="_upto35",
        )
    )
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

    from code.marginal.export_paper_tables_mean_width import main as _export_tables

    _export_tables()
    tables = PLOTS_MARGINAL / "dgp_true_marginal_rf" / "tables" / "simulations_mean_width_tables.tex"
    if tables.exists():
        outputs.append(tables)
    return outputs


def _build_acs() -> list[Path]:
    acs_plot = REPO_ROOT / "real_data" / "acs" / "plot_dgp_style_paper_plots.py"
    spec = importlib.util.spec_from_file_location("acs_plot", acs_plot)
    acs = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(acs)
    return acs.run_suite(acs.RF_YOEP_FB_MIN21_SUITE)


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
    "dgp_rf": _build_dgp_rf,
    "acs": _build_acs,
    "acs_xgboost": _build_acs_xgboost,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate true-marginal paper plots.")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="dgp_rf",
        help="Which experiment suite to plot (default: dgp_rf).",
    )
    parser.add_argument("--all", action="store_true", help="Run all active suites.")
    args = parser.parse_args()
    suites = list(SUITES) if args.all else [args.suite]
    for name in suites:
        print(f"\n=== Suite: {name} ===")
        outputs = SUITES[name]()
        for path in outputs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
