#!/usr/bin/env python3
"""Create DGP-style ACS true-marginal paper plots."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


SCRIPT_PATH = Path(__file__).resolve()
REAL_DATA_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = REAL_DATA_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.paths import PLOTS_MARGINAL, RESULTS_ACS_MARGINAL  # noqa: E402

_pm_path = REPO_ROOT / "code" / "shared" / "plot_engine.py"
_pm_spec = importlib.util.spec_from_file_location("plot_true_marginal_alpha_grid", _pm_path)
pm = importlib.util.module_from_spec(_pm_spec)
assert _pm_spec.loader is not None
_pm_spec.loader.exec_module(pm)
RESULTS_ROOT = RESULTS_ACS_MARGINAL
PAPER_ROOT = PLOTS_MARGINAL / "acs"

PERMUTED_PREFIX = "true_marginal_permuted_alpha"
PERMUTED_OLS_PREFIX = "true_marginal_permuted_ols_trim2_corr_alpha"
PERMUTED_OLS_YOEP2000_PREFIX = "true_marginal_permuted_ols_notrim_corr_yoep2000_alpha"
PERMUTED_OLS_YOEP2000_TRIM2_PREFIX = "true_marginal_permuted_ols_trim2_corr_yoep2000_alpha"
PERMUTED_OLS_LEGACY_INCOME_PREFIX = "true_marginal_permuted_income_alpha"
PERMUTED_DING_XGB_PREFIX = "true_marginal_permuted_ding_xgb_income_alpha"
PERMUTED_DING_XGB_NO_WITHIN_PREFIX = "true_marginal_permuted_ding_xgb_no_within_income_alpha"
PERMUTED_RF_PREFIX = "true_marginal_permuted_rf_income_notrim_corr_yoep2000_alpha"
PERMUTED_RF_TRIM2_CORR_PREFIX = "true_marginal_permuted_rf_income_trim2_corr_alpha"
PERMUTED_RF_TRIM2_MEAN_PREFIX = "true_marginal_permuted_rf_income_trim2_alpha"
PERMUTED_RF_LEGACY_INCOME_PREFIX = "true_marginal_permuted_rf_income_alpha"
PERMUTED_RF_2012_MEAN_PREFIX = "true_marginal_permuted_rf_income_notrim_alpha"
PERMUTED_RF_2012_RESIDUAL_PREFIX = "true_marginal_permuted_rf_income_notrim_corr_alpha"
PERMUTED_OLS_2012_MEAN_PREFIX = "true_marginal_permuted_ols_notrim_alpha"
PERMUTED_OLS_2000_MEAN_PREFIX = "true_marginal_permuted_ols_notrim_yoep2000_alpha"
PERMUTED_OLS_2012_CORRECTION_PREFIX = "true_marginal_permuted_ols_notrim_corr_alpha"
PERMUTED_RF_NO_WITHIN_PREFIX = "true_marginal_permuted_rf_no_within_income_alpha"
PERMUTED_T35_PREFIX = "true_marginal_permuted_t35_alpha"
RESULT_GLOB = f"{PERMUTED_PREFIX}*"
O_VALUES = [0, 5, 10, 15, 20]
O_VALUES_UPTO35 = [0, 5, 10, 15, 20, 25, 30, 35]
NOMINAL_O_VALUES = [0, 10, 20]
ALPHA_PANEL_VALUES = [0.05, 0.10, 0.20]
UPTO35_ALPHA_VALUES = [0.10, 0.20]
TITLE_PREFIX = "ACS"
ALL_BASELINES_TITLE_PREFIX = "ACS"
ALL_BASELINES_SHORT_TITLES = False
ALL_BASELINES_FONT_SCALE = 1.0
ALL_BASELINES_STACKED_HEIGHT = 20.0
ALL_BASELINES_PANEL_HEIGHT = 10.5
FILE_PREFIX = "acs"
NOMINAL_COVERAGE_MAX = pm.NOMINAL_COVERAGE_MAX

O_COLORS = pm.O_COLORS
O_MARKERS = pm.O_MARKERS
METHOD_COLORS = pm.METHOD_COLORS
METHOD_LINESTYLES = pm.METHOD_LINESTYLES
METHOD_MARKERS = pm.METHOD_MARKERS
BASELINE_COLORS = pm.BASELINE_COLORS
BASELINE_MARKERS = pm.BASELINE_MARKERS
BASELINE_LINESTYLES = pm.BASELINE_LINESTYLES
NOMINAL_COLOR = pm.NOMINAL_COLOR
INK_COLOR = pm.INK_COLOR
ALPHA_BASELINE_COMPARISON = 0.20
# (method, o, x-axis label) for fixed-o baseline comparison panels.
BASELINE_COMPARISON_SPECS: list[tuple[str, int, str]] = [
    ("GHCP", 20, "GHCP\n($o=20$)"),
    ("HCP", 0, "HCP"),
    ("Pooling", 0, "Pooling"),
    ("Subsampling", 0, "Subsampling"),
    ("Repeated", 0, "Repeated\nSubsampling"),
]
METHOD_COMPARISON_SPACING = 3.8
METHOD_COMPARISON_BOX_WIDTH = 0.68
WIDTH_AXIS_PADDING = 1.04

FONT_TICK = pm.FONT_TICK
FONT_LABEL = pm.FONT_LABEL
FONT_TITLE = pm.FONT_TITLE
FONT_LEGEND = pm.FONT_LEGEND
FONT_METHODS_XLABEL = pm.FONT_METHODS_XLABEL
COVERAGE_YMARGIN_BELOW_NOMINAL = pm.COVERAGE_YMARGIN_BELOW_NOMINAL
PLOT_BORDER_COLOR = pm.PLOT_BORDER_COLOR
PLOT_BORDER_WIDTH = pm.PLOT_BORDER_WIDTH
LINEWIDTH = pm.LINEWIDTH
MARKERSIZE = pm.MARKERSIZE
CAPSIZE = pm.CAPSIZE
SE_VISUAL_MULTIPLIER = pm.SE_VISUAL_MULTIPLIER
BOX_WIDTH_NOMINAL = pm.BOX_WIDTH_NOMINAL
BOX_WIDTH_O_AXIS = pm.BOX_WIDTH_O_AXIS
NOMINAL_BOX_GROUP_SPACING = pm.NOMINAL_BOX_GROUP_SPACING
WIDTH_AXIS_QUANTILE = pm.WIDTH_AXIS_QUANTILE

# Paper o-axis: reference layout is o=0,5,10,15,20 on a step-5 grid, fig width 24.
O_AXIS_REF_N = 5
O_AXIS_REF_STEP = 5
O_AXIS_REF_FIG_WIDTH = 24.0
O_AXIS_REF_FIG_HEIGHT = 8.8
O_AXIS_HCP_OFFSET = 6
O_AXIS_XPAD = 0.75


def _standard_o_axis(o_values: list[int]) -> tuple[dict[int, float], float, tuple[float, float], float]:
    """Same step-5 grid as o=0,5,10,15,20; scale fig width by n_o / 5."""
    o_values = [int(o) for o in o_values]
    n = len(o_values)
    ref_grid = [i * O_AXIS_REF_STEP for i in range(O_AXIS_REF_N)]
    if o_values == ref_grid:
        xpos = {o: float(o) for o in o_values}
    else:
        xpos = {o: float(i * O_AXIS_REF_STEP) for i, o in enumerate(o_values)}
    max_x = float((n - 1) * O_AXIS_REF_STEP)
    hcp_pos = max_x + O_AXIS_HCP_OFFSET
    cov_xlim = (-O_AXIS_XPAD, max_x + O_AXIS_XPAD)
    fig_width = O_AXIS_REF_FIG_WIDTH * n / O_AXIS_REF_N
    return xpos, hcp_pos, cov_xlim, fig_width

X_LABEL_NOMINAL = pm.X_LABEL_NOMINAL
X_LABEL_TARGET_O = pm.X_LABEL_TARGET_O
Y_LABEL_WIDTH = pm.Y_LABEL_WIDTH


def _alpha_from_dir(path: Path) -> float | None:
    match = re.search(r"alpha([0-9p.]+)$", path.name)
    if not match:
        return None
    return float(match.group(1).replace("p", ".")) / 100.0


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.3f}".rstrip("0").rstrip(".")


def _plot_tag(alpha: float) -> str:
    return _alpha_label(alpha).replace(".", "p")


def _nominal_label(nominal: float) -> str:
    return f"{nominal:.3f}".rstrip("0").rstrip(".")


def _currency_formatter(x: float, _pos: int) -> str:
    if not np.isfinite(x):
        return ""
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    return f"${x / 1_000:.0f}K"


@dataclass(frozen=True)
class AcsPlotSuite:
    """Output location and result glob for one ACS predictor run."""

    paper_root: Path
    result_glob: str
    alpha_panel_values: tuple[float, ...]
    results_root: Path | None = None
    title_prefix: str = "ACS"
    file_prefix: str = "acs"
    include_nominal_panels: bool = True
    include_upto35_panels: bool = True
    width_col: str = "width_income"
    width_ylabel: str = Y_LABEL_WIDTH
    o_values: tuple[int, ...] = (0, 5, 10, 15, 20)
    all_baselines_title_prefix: str | None = None
    all_baselines_short_titles: bool = False
    all_baselines_font_scale: float = 1.0
    all_baselines_stacked_height: float = 20.0
    all_baselines_panel_height: float = 10.5
    extra_width_axis_min: float | None = None
    trials_csv: Path | None = None
    skip_all_baselines: bool = False
    coverage_width_out_suffix: str = ""
    width_currency_format: bool = True


DEFAULT_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT,
    result_glob=f"{PERMUTED_PREFIX}*",
    alpha_panel_values=tuple(ALPHA_PANEL_VALUES),
)

OLS_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols",
    results_root=PAPER_ROOT / "ols" / "results",
    result_glob=f"{PERMUTED_OLS_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, income scale, trim top 2%, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

OLS_YOEP2000_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols_yoep2000",
    results_root=PAPER_ROOT / "ols_yoep2000" / "results",
    result_glob=f"{PERMUTED_OLS_YOEP2000_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, income scale, YOEP≥2000, n≥31, no income trim, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

OLS_YOEP2000_TRIM2_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols_yoep2000",
    results_root=PAPER_ROOT / "ols_yoep2000" / "results_trim2_backup",
    result_glob=f"{PERMUTED_OLS_YOEP2000_TRIM2_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, income scale, YOEP≥2000, n≥31, trim top 2%, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

OLS_YOEP2000_NOTRIM_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols_yoep2000_notrim",
    results_root=PAPER_ROOT / "ols_yoep2000_notrim" / "results",
    result_glob=f"{PERMUTED_OLS_YOEP2000_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, income scale, YOEP≥2000, n≥21, no income trim, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width_income",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 5, 10, 15, 20),
)

OLS_INCOME_LEGACY_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols" / "legacy_mean",
    results_root=PAPER_ROOT / "ols" / "legacy_mean" / "results",
    result_glob=f"{PERMUTED_OLS_LEGACY_INCOME_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, income scale, mean shrinkage)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

DING_XGB_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "xgboost",
    result_glob=f"{PERMUTED_DING_XGB_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Ding XGBoost, income scale)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

DING_XGB_NO_WITHIN_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "xgboost" / "no_within",
    result_glob=f"{PERMUTED_DING_XGB_NO_WITHIN_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Ding XGBoost, no within-group, income scale)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

RF_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf",
    results_root=PAPER_ROOT / "rf" / "results",
    result_glob=f"{PERMUTED_RF_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, income scale, YOEP≥2000, n≥31, no income trim, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

RF_TRIM2_CORR_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf",
    results_root=PAPER_ROOT / "rf" / "results_trim2_backup",
    result_glob=f"{PERMUTED_RF_TRIM2_CORR_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, income scale, trim top 2%, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

RF_TRIM2_MEAN_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf" / "trim2_mean",
    results_root=PAPER_ROOT / "rf" / "trim2_mean" / "results",
    result_glob=f"{PERMUTED_RF_TRIM2_MEAN_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, income scale, trim top 2%, mean shrinkage)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

RF_LEGACY_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf" / "legacy_mean",
    results_root=PAPER_ROOT / "rf" / "legacy_mean" / "results",
    result_glob=f"{PERMUTED_RF_LEGACY_INCOME_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, income scale, mean shrinkage)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

RF_2012_MEAN_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf_2012_mean",
    results_root=PAPER_ROOT / "rf_2012_mean" / "results",
    result_glob=f"{PERMUTED_RF_2012_MEAN_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, YOEP≥2012, no income filter, mean shrinkage)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

OLS_2012_MEAN_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols_2012_mean",
    results_root=PAPER_ROOT / "ols_2012_mean" / "results",
    result_glob=f"{PERMUTED_OLS_2012_MEAN_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, YOEP≥2012, no income filter, mean shrinkage)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

OLS_2000_MEAN_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols_2000_mean",
    results_root=PAPER_ROOT / "ols_2000_mean" / "results",
    result_glob=f"{PERMUTED_OLS_2000_MEAN_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, YOEP≥2000, no income filter, mean shrinkage)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

OLD_LOG_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "old",
    result_glob="",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (legacy log1p CP, YOEP≥2012, $10K floor)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    skip_all_baselines=True,
    width_col="width",
    width_ylabel="Prediction Set Width (log1p)",
    width_currency_format=False,
    coverage_width_out_suffix="_logwidth",
    trials_csv=PAPER_ROOT / "old" / "summaries" / "acs_trials_long.csv",
)

OLS_2012_CORRECTION_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "ols_2012_correction",
    results_root=PAPER_ROOT / "ols_2012_correction" / "results",
    result_glob=f"{PERMUTED_OLS_2012_CORRECTION_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (OLS, YOEP≥2012, no income filter, residual correction)",
    all_baselines_title_prefix="ACS",
    all_baselines_short_titles=True,
    all_baselines_font_scale=1.08,
    all_baselines_stacked_height=25.0,
    all_baselines_panel_height=13.0,
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
    extra_width_axis_min=50_000.0,
)

RF_2012_CORRECTION_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf_2012_correction",
    results_root=PAPER_ROOT / "rf_2012_correction" / "results",
    result_glob=f"{PERMUTED_RF_2012_RESIDUAL_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, YOEP≥2012, no income filter, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
    extra_width_axis_min=50_000.0,
)

RF_2012_RESIDUAL_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf_2012_residual",
    results_root=PAPER_ROOT / "rf_2012_residual" / "results",
    result_glob=f"{PERMUTED_RF_2012_RESIDUAL_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, YOEP≥2012, no income filter, residual correction)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

RF_NO_WITHIN_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "rf" / "no_within",
    result_glob=f"{PERMUTED_RF_NO_WITHIN_PREFIX}*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (Random Forest, no within-group, income scale)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
)

RF_2000_MEAN_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "non-stratified" / "rf_2000_mean" / "rf_2000_mean",
    results_root=PAPER_ROOT / "non-stratified" / "rf_2000_mean" / "rf_2000_mean" / "results",
    result_glob="true_marginal_permuted_rf_income_notrim_t31_yoep2000_alpha*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (RF)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

RF_2000_MEAN_TRIM_TOP_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "non-stratified" / "rf_2000_mean_trim_top" / "rf_2000_mean_trim_top",
    results_root=PAPER_ROOT / "non-stratified" / "rf_2000_mean_trim_top" / "rf_2000_mean_trim_top" / "results",
    result_glob="true_marginal_permuted_rf_income_trim2_t31_yoep2000_alpha*",
    alpha_panel_values=(0.20,),
    title_prefix="ACS (RF)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

STUDENTIZED_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "studentized",
    results_root=PAPER_ROOT / "studentized" / "results",
    result_glob="true_marginal_rf_income_notrim_t31_yoep2000_studentized_alpha*",
    alpha_panel_values=(0.10, 0.20),
    title_prefix="ACS (RF, studentized |Y−μ|/σ, no row permute)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width_income",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

RF_NO_PERMUTE_MIN21_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "no_permute_min21",
    results_root=PAPER_ROOT / "no_permute_min21" / "results",
    result_glob="true_marginal_rf_income_notrim_yoep2000_alpha*",
    alpha_panel_values=(0.10, 0.20),
    title_prefix="ACS (RF, no row permute, PUMA size ≥21, target idx 20)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width_income",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 5, 10, 15, 20),
)

RF_NO_PERMUTE_MIN31_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "no_permute_min31",
    results_root=PAPER_ROOT / "no_permute_min31" / "results",
    result_glob="true_marginal_rf_income_notrim_t30_yoep2000_alpha*",
    alpha_panel_values=(0.10, 0.20),
    title_prefix="ACS (RF, no row permute, PUMA size ≥31, target idx 30, |S̃|=11)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width_income",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

RF_YOEP_FB_MIN31_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "yoep_fb_extended" / "min31",
    results_root=PAPER_ROOT / "yoep_fb_extended" / "min31" / "results",
    result_glob="true_marginal_rf_income_notrim_t30_yoep2000*alpha*",
    alpha_panel_values=(0.10, 0.20),
    title_prefix="ACS (RF, YOEP+FB only, size ≥31, target idx 30)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width_income",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 10, 20, 30),
)

RF_YOEP_FB_MIN21_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT,
    results_root=PAPER_ROOT / "results",
    result_glob="true_marginal_permuted_rf_income_notrim*yoep2000*alpha*",
    alpha_panel_values=(0.05, 0.10, 0.15, 0.20),
    title_prefix="ACS (RF, age 25–54, hours ≥40, YOEP≥2000, permute, size ≥21)",
    all_baselines_title_prefix="ACS (RF)",
    all_baselines_short_titles=True,
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width_income",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 5, 10, 15, 20),
)

RF_NEW_SAMPLING_SUITE = AcsPlotSuite(
    paper_root=PAPER_ROOT / "new_sampling",
    results_root=PAPER_ROOT / "new_sampling" / "results",
    result_glob="true_marginal_rf_income_notrim_yoep2010*alpha*",
    alpha_panel_values=(0.10, 0.20),
    title_prefix="ACS (RF, YOEP≥2010, mixed-size calib, target idx 20)",
    include_nominal_panels=False,
    include_upto35_panels=False,
    width_col="width_income",
    width_ylabel="Prediction Set Width ($)",
    o_values=(0, 5, 10, 15, 20),
)


def configure_suite(suite: AcsPlotSuite) -> None:
    """Point module-level output paths and glob at one predictor's results."""
    global PAPER_ROOT, FIG_DIR, SUMMARY_DIR, RESULT_GLOB, RESULTS_ROOT
    global ALPHA_PANEL_VALUES, TITLE_PREFIX, FILE_PREFIX
    global WIDTH_COL, WIDTH_YLABEL, O_VALUES
    global ALL_BASELINES_TITLE_PREFIX, ALL_BASELINES_SHORT_TITLES, ALL_BASELINES_FONT_SCALE
    global ALL_BASELINES_STACKED_HEIGHT, ALL_BASELINES_PANEL_HEIGHT
    PAPER_ROOT = suite.paper_root
    FIG_DIR = PAPER_ROOT / "figures"
    SUMMARY_DIR = PAPER_ROOT / "summaries"
    RESULT_GLOB = suite.result_glob
    RESULTS_ROOT = suite.results_root if suite.results_root is not None else (
        RESULTS_ACS_MARGINAL
    )
    ALPHA_PANEL_VALUES = list(suite.alpha_panel_values)
    TITLE_PREFIX = suite.title_prefix
    FILE_PREFIX = suite.file_prefix
    WIDTH_COL = suite.width_col
    WIDTH_YLABEL = suite.width_ylabel
    O_VALUES = list(suite.o_values)
    ALL_BASELINES_TITLE_PREFIX = (
        suite.all_baselines_title_prefix
        if suite.all_baselines_title_prefix is not None
        else suite.title_prefix
    )
    ALL_BASELINES_SHORT_TITLES = suite.all_baselines_short_titles
    ALL_BASELINES_FONT_SCALE = suite.all_baselines_font_scale
    ALL_BASELINES_STACKED_HEIGHT = suite.all_baselines_stacked_height
    ALL_BASELINES_PANEL_HEIGHT = suite.all_baselines_panel_height


FIG_DIR = PAPER_ROOT / "figures"
SUMMARY_DIR = PAPER_ROOT / "summaries"
WIDTH_COL = "width_income"
WIDTH_YLABEL = Y_LABEL_WIDTH


def reset_output_dirs() -> None:
    # Never rmtree PAPER_ROOT: the paper suite writes figures next to results/.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def _detailed_files(*, result_glob: str | None = None) -> list[tuple[float, Path]]:
    """Load ACS detailed CSVs; prefer permuted runs when both exist for an alpha.

    Row permutation within each PUMA is required for the fixed target index design:
    without it, indices 0..o-1 are not exchangeable with the target at index 20, and
    coverage collapses as o grows.
    """
    if result_glob is None:
        result_glob = RESULT_GLOB
    by_alpha: dict[float, Path] = {}
    for result_dir in sorted(RESULTS_ROOT.glob("true_marginal_alpha*")):
        if "permuted" in result_dir.name:
            continue
        alpha = _alpha_from_dir(result_dir)
        if alpha is None:
            continue
        details = sorted(result_dir.glob("acs_true_marg_alpha*_detailed.csv"))
        if details and alpha not in by_alpha:
            by_alpha[alpha] = details[0]
    for result_dir in sorted(RESULTS_ROOT.glob(result_glob)):
        alpha = _alpha_from_dir(result_dir)
        if alpha is None:
            continue
        details = sorted(result_dir.glob("acs_true_marg_alpha*_detailed.csv"))
        if details:
            by_alpha[alpha] = details[0]
    if not by_alpha:
        raise FileNotFoundError(
            f"No ACS detailed files found under {RESULTS_ROOT} "
            f"(expected {result_glob})"
        )
    return [(alpha, by_alpha[alpha]) for alpha in sorted(by_alpha)]


def load_trials_from_csv(path: Path, *, width_col: str | None = None) -> pd.DataFrame:
    """Load replicate-level trials already saved under a suite summaries folder."""
    col = width_col or WIDTH_COL
    df = pd.read_csv(path)
    if "plot_width" not in df.columns:
        df["plot_width"] = pd.to_numeric(df[col], errors="coerce")
    else:
        df["plot_width"] = pd.to_numeric(df[col], errors="coerce")
    df["o"] = df["o"].astype(int)
    if "method" in df.columns:
        df["method"] = df["method"].replace({"Donor-HCP": "GHCP", "D-HCP": "GHCP", "D-HCP no within": "GHCP no within"})
    for c in ("coverage", "width", "width_income"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "nominal_coverage" not in df.columns and "alpha" in df.columns:
        df["nominal_coverage"] = 1.0 - df["alpha"]
    return df.sort_values(["alpha", "method", "o", "replicate"]).reset_index(drop=True)


def load_trials(*, result_glob: str | None = None) -> pd.DataFrame:
    pieces = []
    for alpha, path in _detailed_files(result_glob=result_glob):
        df = pd.read_csv(path)
        df = df[["replicate", "method", "o", "coverage", "width", "width_income"]].copy()
        df["alpha"] = alpha
        df["nominal_coverage"] = 1.0 - alpha
        df["o"] = df["o"].astype(int)
        df["method"] = df["method"].replace({"Donor-HCP": "GHCP", "D-HCP": "GHCP", "D-HCP no within": "GHCP no within"})
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
        df["width"] = pd.to_numeric(df["width"], errors="coerce")
        df["width_income"] = pd.to_numeric(df["width_income"], errors="coerce")
        df["plot_width"] = pd.to_numeric(df[WIDTH_COL], errors="coerce")
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True).sort_values(["alpha", "method", "o", "replicate"])


def _summary_for_group(group: pd.DataFrame) -> pd.Series:
    cov = group["coverage"].dropna().to_numpy(dtype=float)
    width_all = group["plot_width"].to_numpy(dtype=float)
    finite_width = group["plot_width"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    p_hat = float(np.mean(cov)) if len(cov) else np.nan
    cov_std = float(np.std(cov, ddof=1)) if len(cov) > 1 else np.nan
    width_std = float(np.std(finite_width, ddof=1)) if len(finite_width) > 1 else np.nan
    n_width_total = int(np.sum(~pd.isna(width_all)))
    n_width_infinite = int(np.sum(np.isinf(width_all)))
    return pd.Series(
        {
            "nominal_coverage": float(group["nominal_coverage"].iloc[0]),
            "coverage_mean": p_hat,
            "coverage_std": cov_std,
            "coverage_se": pm.coverage_standard_error(cov),
            "coverage_n": int(len(cov)),
            "width_mean": float(np.mean(finite_width)) if len(finite_width) else np.nan,
            "width_std": width_std,
            "width_se": width_std / np.sqrt(len(finite_width)) if len(finite_width) > 1 else np.nan,
            "width_median": float(np.median(finite_width)) if len(finite_width) else np.nan,
            "width_q25": float(np.quantile(finite_width, 0.25)) if len(finite_width) else np.nan,
            "width_q75": float(np.quantile(finite_width, 0.75)) if len(finite_width) else np.nan,
            "width_min": float(np.min(finite_width)) if len(finite_width) else np.nan,
            "width_max": float(np.max(finite_width)) if len(finite_width) else np.nan,
            "width_n_total": n_width_total,
            "width_n_finite": int(len(finite_width)),
            "width_n_infinite": n_width_infinite,
            "width_finite_rate": float(len(finite_width) / n_width_total) if n_width_total else np.nan,
            "width_infinite_rate": float(n_width_infinite / n_width_total) if n_width_total else np.nan,
        }
    )


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["alpha", "method", "o"]
    for keys, group in trials.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(_summary_for_group(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["alpha", "method", "o"]).reset_index(drop=True)


def _coverage_ylim_lower(alpha: float) -> float:
    return max(0.0, 1.0 - float(alpha) - COVERAGE_YMARGIN_BELOW_NOMINAL)


def _coverage_ylim_lower_from_nominals(nominal_values: list[float]) -> float:
    if not nominal_values:
        return 0.0
    return max(0.0, min(float(v) for v in nominal_values) - COVERAGE_YMARGIN_BELOW_NOMINAL)


def _apply_plot_border(ax: plt.Axes) -> None:
    pm._apply_plot_border(ax)


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    pm._style_axis(ax, xlabel, ylabel)


def _set_nominal_ticks(ax: plt.Axes, nominal_values: list[float]) -> None:
    ax.set_xticks(nominal_values)
    ax.set_xticklabels([_nominal_label(nominal) for nominal in nominal_values], rotation=35, ha="right")


def _plot_coverage_line_with_band(ax, x, y, se, color, **kwargs):
    pm._plot_coverage_line_with_band(ax, x, y, se, color, **kwargs)


def _error_band(ax: plt.Axes, x: np.ndarray, y: np.ndarray, se: np.ndarray, color: str, alpha: float = 0.16) -> None:
    pm._error_band(ax, x, y, se, color, alpha=alpha)


def _finite_width_array(series: pd.Series) -> np.ndarray:
    return series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _all_finite_width_array(series: pd.Series) -> np.ndarray:
    return pm._all_finite_width_array(series)


def _managed_width_upper(
    values: list[float] | np.ndarray,
    quantile: float = WIDTH_AXIS_QUANTILE,
    *,
    include_all: bool = False,
) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if include_all or quantile >= 1.0:
        upper = float(np.max(finite))
    else:
        upper = float(np.quantile(finite, quantile))
    return upper * WIDTH_AXIS_PADDING


def _boxplot_whisker_upper(values: list[float] | np.ndarray, whis: float = 1.5) -> float | None:
    """Upper whisker cap for matplotlib's default 1.5*IQR boxplots."""
    return pm._boxplot_whisker_upper(values, whis=whis)


def _width_axis_upper_from_box_groups(
    box_groups: list[np.ndarray],
    *,
    padding: float = WIDTH_AXIS_PADDING,
) -> float | None:
    """Y-axis cap from visible boxplot whiskers (showfliers=False), not raw outliers."""
    return pm._width_axis_upper_from_box_groups(box_groups, padding=padding)


def _filter_nominal(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["nominal_coverage"] <= NOMINAL_COVERAGE_MAX + 1e-12].copy()


def _dedupe_hcp(summary: pd.DataFrame) -> pd.DataFrame:
    hcp = summary[summary["method"] == "HCP"].copy()
    return hcp[hcp["o"] == 0].copy()


def plot_nominal_coverage_width(trials: pd.DataFrame, summary: pd.DataFrame) -> Path:
    trials = _filter_nominal(trials)
    summary = _filter_nominal(summary)
    nominal_values = sorted(summary["nominal_coverage"].dropna().unique())
    fig, axes = plt.subplots(2, 1, figsize=(26.0, 18.5))
    ax_cov, ax_width = axes

    dhcp_summary = summary[summary["method"] == "GHCP"]
    for o in NOMINAL_O_VALUES:
        sub = dhcp_summary[dhcp_summary["o"] == o].sort_values("nominal_coverage")
        x = sub["nominal_coverage"].to_numpy(dtype=float)
        y = sub["coverage_mean"].to_numpy(dtype=float)
        se = sub["coverage_se"].fillna(0.0).to_numpy(dtype=float)
        _plot_coverage_line_with_band(
            ax_cov, x, y, se, O_COLORS[o],
            label=f"GHCP, o={o}",
            marker=O_MARKERS.get(o, "o"),
            linestyle="-",
            alpha_band=0.12,
        )

    hcp = _dedupe_hcp(summary).sort_values("nominal_coverage")
    x_hcp = hcp["nominal_coverage"].to_numpy(dtype=float)
    y_hcp = hcp["coverage_mean"].to_numpy(dtype=float)
    se_hcp = hcp["coverage_se"].fillna(0.0).to_numpy(dtype=float)
    _plot_coverage_line_with_band(
        ax_cov, x_hcp, y_hcp, se_hcp, METHOD_COLORS["HCP"],
        label="HCP",
        marker=METHOD_MARKERS["HCP"],
        linestyle=METHOD_LINESTYLES["HCP"],
        alpha_band=0.10,
    )
    ax_cov.plot(nominal_values, nominal_values, color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal")
    _style_axis(ax_cov, X_LABEL_NOMINAL, "Empirical coverage")
    _set_nominal_ticks(ax_cov, nominal_values)
    ax_cov.set_ylim(_coverage_ylim_lower_from_nominals(nominal_values), 1.02)
    ax_cov.set_title(f"{TITLE_PREFIX} Coverage", fontsize=FONT_TITLE, pad=12)

    positions = np.arange(len(nominal_values), dtype=float) * NOMINAL_BOX_GROUP_SPACING
    n_box_series = len(NOMINAL_O_VALUES) + 1
    offsets = (np.arange(n_box_series, dtype=float) - (n_box_series - 1) / 2.0) * BOX_WIDTH_NOMINAL * 2.05
    width_box_groups: list[np.ndarray] = []
    for nominal_idx, nominal in enumerate(nominal_values):
        for o_idx, o in enumerate(NOMINAL_O_VALUES):
            vals = _finite_width_array(
                trials[
                    (trials["method"] == "GHCP")
                    & (trials["o"] == o)
                    & np.isclose(trials["nominal_coverage"], nominal)
                ]["plot_width"]
            )
            if len(vals):
                width_box_groups.append(vals)
                x_pos = positions[nominal_idx] + offsets[o_idx]
                ax_width.boxplot(
                    [vals],
                    positions=[x_pos],
                    widths=BOX_WIDTH_NOMINAL,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=pm._accessible_medianprops(),
                    whiskerprops=pm._accessible_whiskerprops(INK_COLOR),
                    capprops=pm._accessible_whiskerprops(INK_COLOR),
                    boxprops=pm._accessible_boxprops(O_COLORS[o], alpha=0.82),
                )
                ax_width.scatter(x_pos, float(np.median(vals)), color=INK_COLOR, marker="D", s=34, zorder=4)

        vals_hcp = _finite_width_array(
            trials[(trials["method"] == "HCP") & (trials["o"] == 0) & np.isclose(trials["nominal_coverage"], nominal)][
                "plot_width"
            ]
        )
        if len(vals_hcp):
            width_box_groups.append(vals_hcp)
            ax_width.boxplot(
                [vals_hcp],
                positions=[positions[nominal_idx] + offsets[-1]],
                widths=BOX_WIDTH_NOMINAL,
                patch_artist=True,
                showfliers=False,
                medianprops=pm._accessible_medianprops(),
                whiskerprops=pm._accessible_whiskerprops(METHOD_COLORS["HCP"]),
                capprops=pm._accessible_whiskerprops(METHOD_COLORS["HCP"]),
                boxprops=pm._accessible_boxprops(METHOD_COLORS["HCP"], hatch="///", alpha=0.52),
            )

    ax_width.set_xticks(positions)
    ax_width.set_xticklabels([_nominal_label(nominal) for nominal in nominal_values], rotation=35, ha="right")
    ax_width.set_xlim(positions[0] + offsets[0] - 0.5, positions[-1] + offsets[-1] + 0.5)
    width_upper = _width_axis_upper_from_box_groups(width_box_groups)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    ax_width.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    _style_axis(ax_width, X_LABEL_NOMINAL, WIDTH_YLABEL)
    ax_width.set_title(f"{TITLE_PREFIX} Prediction Set Width", fontsize=FONT_TITLE, pad=12)

    handles = [
        Line2D([0], [0], color=O_COLORS[o], marker=O_MARKERS.get(o, "o"), linestyle="-",
               linewidth=LINEWIDTH, label=f"GHCP, o={o}")
        for o in NOMINAL_O_VALUES
    ]
    handles.append(Line2D([0], [0], color=METHOD_COLORS["HCP"], marker=METHOD_MARKERS["HCP"],
                          linestyle=METHOD_LINESTYLES["HCP"], linewidth=LINEWIDTH, label="HCP"))
    handles.append(Line2D([0], [0], color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), **pm._legend_kwargs())
    fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.2)
    out = FIG_DIR / f"{FILE_PREFIX}_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_nominal_line_bands(summary: pd.DataFrame) -> Path:
    summary = _filter_nominal(summary)
    nominal_values = sorted(summary["nominal_coverage"].dropna().unique())
    fig, axes = plt.subplots(2, 1, figsize=(26.0, 18.0))

    for ax, metric, se_col, ylabel, title in [
        (axes[0], "coverage_mean", "coverage_se", "Empirical coverage", f"{TITLE_PREFIX} Coverage with SE Bands"),
        (axes[1], "width_mean", "width_se", WIDTH_YLABEL, f"{TITLE_PREFIX} Prediction Set Width with SE Bands"),
    ]:
        for o in NOMINAL_O_VALUES:
            sub = summary[(summary["method"] == "GHCP") & (summary["o"] == o)].sort_values("nominal_coverage")
            x = sub["nominal_coverage"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            se = sub[se_col].fillna(0.0).to_numpy(dtype=float)
            _error_band(ax, x, y, se, O_COLORS[o], alpha=0.12)
            ax.plot(
                x, y,
                marker=O_MARKERS.get(o, "o"),
                linestyle="-",
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                color=O_COLORS[o],
                markeredgecolor=INK_COLOR,
                markeredgewidth=0.6,
                label=f"GHCP, o={o}",
            )

        hcp = _dedupe_hcp(summary).sort_values("nominal_coverage")
        x = hcp["nominal_coverage"].to_numpy(dtype=float)
        y = hcp[metric].to_numpy(dtype=float)
        se = hcp[se_col].fillna(0.0).to_numpy(dtype=float)
        _error_band(ax, x, y, se, METHOD_COLORS["HCP"], alpha=0.10)
        ax.plot(
            x, y,
            marker=METHOD_MARKERS["HCP"],
            linestyle=METHOD_LINESTYLES["HCP"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            color=METHOD_COLORS["HCP"],
            markeredgecolor=INK_COLOR,
            markeredgewidth=0.6,
            label="HCP",
        )

        if metric == "coverage_mean":
            ax.plot(nominal_values, nominal_values, color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal")
            ax.set_ylim(_coverage_ylim_lower_from_nominals(nominal_values), 1.02)
        else:
            width_upper = _managed_width_upper(summary[metric].to_numpy(dtype=float))
            if width_upper is not None:
                ax.set_ylim(0.0, width_upper)
            ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
        _style_axis(ax, X_LABEL_NOMINAL, ylabel)
        _set_nominal_ticks(ax, nominal_values)
        ax.set_title(title, fontsize=FONT_TITLE, pad=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), **pm._legend_kwargs())
    fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.2)
    out = FIG_DIR / f"{FILE_PREFIX}_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_alpha_o_axis_panel(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    alpha: float,
    o_values: list[int] | None = None,
    out_suffix: str = "",
    width_ymin: float | None = None,
    width_currency_format: bool = True,
    include_stdcp: bool = False,
) -> Path:
    if o_values is None:
        o_values = O_VALUES
    xpos, hcp_pos, cov_xlim, fig_width = _standard_o_axis(o_values)
    d_trials = trials[np.isclose(trials["alpha"], alpha)].copy()
    d_summary = summary[np.isclose(summary["alpha"], alpha)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, O_AXIS_REF_FIG_HEIGHT))
    ax_cov, ax_width = axes

    dhcp = d_summary[(d_summary["method"] == "GHCP") & (d_summary["o"].isin(o_values))].sort_values("o")
    stdcp = d_summary[(d_summary["method"] == "Std-CP") & (d_summary["o"].isin(o_values))].sort_values("o")
    hcp = d_summary[(d_summary["method"] == "HCP") & (d_summary["o"] == 0)]
    if dhcp.empty or hcp.empty:
        raise ValueError(f"Missing GHCP or HCP rows for alpha={alpha}, o_values={o_values}")
    _plot_coverage_line_with_band(
        ax_cov,
        np.asarray([xpos[int(o)] for o in dhcp["o"]], dtype=float),
        dhcp["coverage_mean"].to_numpy(dtype=float),
        dhcp["coverage_se"].fillna(0.0).to_numpy(dtype=float),
        METHOD_COLORS["GHCP"],
        label="GHCP",
        marker=METHOD_MARKERS["GHCP"],
        linestyle=METHOD_LINESTYLES["GHCP"],
        alpha_band=0.10,
    )
    if include_stdcp and not stdcp.empty:
        # Skip o with no finite widths (e.g. o=0); keep o with any finite (e.g. o=5 at α=0.2).
        std_o_keep = []
        for o in o_values:
            vals = _finite_width_array(
                d_trials[(d_trials["method"] == "Std-CP") & (d_trials["o"] == int(o))]["plot_width"]
            )
            if len(vals):
                std_o_keep.append(int(o))
        stdcp_plot = stdcp[stdcp["o"].isin(std_o_keep)].sort_values("o")
        if not stdcp_plot.empty:
            _plot_coverage_line_with_band(
                ax_cov,
                np.asarray([xpos[int(o)] for o in stdcp_plot["o"]], dtype=float),
                stdcp_plot["coverage_mean"].to_numpy(dtype=float),
                stdcp_plot["coverage_se"].fillna(0.0).to_numpy(dtype=float),
                METHOD_COLORS["Std-CP"],
                label="Std-CP",
                marker=METHOD_MARKERS["Std-CP"],
                linestyle=METHOD_LINESTYLES["Std-CP"],
                alpha_band=0.10,
            )
    if not hcp.empty:
        hcp_cov = float(hcp["coverage_mean"].iloc[0])
        hcp_cov_se = float(hcp["coverage_se"].fillna(0.0).iloc[0])
        x_hcp = np.asarray([xpos[int(o)] for o in o_values], dtype=float)
        y_hcp = np.repeat(hcp_cov, len(o_values))
        se_hcp = np.repeat(hcp_cov_se, len(o_values))
        _plot_coverage_line_with_band(
            ax_cov, x_hcp, y_hcp, se_hcp, METHOD_COLORS["HCP"],
            label="HCP",
            marker=METHOD_MARKERS["HCP"],
            linestyle=METHOD_LINESTYLES["HCP"],
            alpha_band=0.10,
        )
    ax_cov.axhline(1.0 - alpha, color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal")
    ax_cov.set_xlim(*cov_xlim)
    ax_cov.set_ylim(_coverage_ylim_lower(alpha), 1.02)
    ax_cov.set_xticks([xpos[int(o)] for o in o_values])
    ax_cov.set_xticklabels([str(o) for o in o_values])
    _style_axis(ax_cov, X_LABEL_TARGET_O, "Empirical coverage")
    ax_cov.set_title("Coverage", fontsize=FONT_TITLE, pad=12)

    width_box_groups: list[np.ndarray] = []
    # Keep GHCP/Std-CP boxes visually separate (centers ±dx; widths < 2*dx).
    dhcp_dx = -0.62 if include_stdcp else 0.0
    stdcp_dx = 0.62
    box_w = BOX_WIDTH_O_AXIS * (0.55 if include_stdcp else 1.0)
    for o in o_values:
        vals = _finite_width_array(d_trials[(d_trials["method"] == "GHCP") & (d_trials["o"] == o)]["plot_width"])
        if len(vals):
            width_box_groups.append(vals)
            ax_width.boxplot(
                [vals],
                positions=[xpos[int(o)] + dhcp_dx],
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                medianprops=pm._accessible_medianprops(),
                whiskerprops=pm._accessible_whiskerprops(METHOD_COLORS["GHCP"]),
                capprops=pm._accessible_whiskerprops(METHOD_COLORS["GHCP"]),
                boxprops=pm._accessible_boxprops(METHOD_COLORS["GHCP"], alpha=0.68),
            )
        if include_stdcp:
            vals_std = _finite_width_array(
                d_trials[(d_trials["method"] == "Std-CP") & (d_trials["o"] == o)]["plot_width"]
            )
            if len(vals_std):
                width_box_groups.append(vals_std)
                ax_width.boxplot(
                    [vals_std],
                    positions=[xpos[int(o)] + stdcp_dx],
                    widths=box_w,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=pm._accessible_medianprops(),
                    whiskerprops=pm._accessible_whiskerprops(METHOD_COLORS["Std-CP"]),
                    capprops=pm._accessible_whiskerprops(METHOD_COLORS["Std-CP"]),
                    boxprops=pm._accessible_boxprops(METHOD_COLORS["Std-CP"], alpha=0.55),
                )
    vals_hcp = _finite_width_array(d_trials[(d_trials["method"] == "HCP") & (d_trials["o"] == 0)]["plot_width"])
    if len(vals_hcp):
        width_box_groups.append(vals_hcp)
        ax_width.boxplot(
            [vals_hcp],
            positions=[hcp_pos],
            widths=BOX_WIDTH_O_AXIS,
            patch_artist=True,
            showfliers=False,
            medianprops=pm._accessible_medianprops(),
            whiskerprops=pm._accessible_whiskerprops(METHOD_COLORS["HCP"]),
            capprops=pm._accessible_whiskerprops(METHOD_COLORS["HCP"]),
            boxprops=pm._accessible_boxprops(METHOD_COLORS["HCP"], hatch="///", alpha=0.50),
        )
    ax_width.set_xticks([*(xpos[int(o)] for o in o_values), hcp_pos])
    ax_width.set_xticklabels([*(str(o) for o in o_values), "HCP"])
    width_upper = _width_axis_upper_from_box_groups(width_box_groups)
    if width_upper is not None:
        ymin = 0.0 if width_ymin is None else float(width_ymin)
        ax_width.set_ylim(ymin, width_upper)
    if width_currency_format:
        ax_width.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    _style_axis(ax_width, X_LABEL_TARGET_O, WIDTH_YLABEL)
    ax_width.set_title(WIDTH_YLABEL, fontsize=FONT_TITLE, pad=12)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["GHCP"], marker=METHOD_MARKERS["GHCP"],
               linestyle=METHOD_LINESTYLES["GHCP"], linewidth=LINEWIDTH, label="GHCP"),
    ]
    if include_stdcp:
        handles.append(
            Line2D([0], [0], color=METHOD_COLORS["Std-CP"], marker=METHOD_MARKERS["Std-CP"],
                   linestyle=METHOD_LINESTYLES["Std-CP"], linewidth=LINEWIDTH, label="Std-CP")
        )
    handles.extend([
        Line2D([0], [0], color=METHOD_COLORS["HCP"], marker=METHOD_MARKERS["HCP"],
               linestyle=METHOD_LINESTYLES["HCP"], linewidth=LINEWIDTH, label="HCP"),
        Line2D([0], [0], color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal"),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), **pm._legend_kwargs())
    fig.tight_layout(rect=[0, 0.15, 1, 1], w_pad=3.0)
    out = FIG_DIR / f"{FILE_PREFIX}_alpha{_plot_tag(alpha)}_coverage_width_by_o{out_suffix}.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out



def plot_all_baselines_by_o(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    alpha: float,
    o_values: list[int] | None = None,
) -> list[Path]:
    o_values = O_VALUES if o_values is None else o_values
    saved_fig_dir = pm.FIG_DIR
    pm.FIG_DIR = FIG_DIR
    try:
        return pm.plot_all_baselines_by_o(
            trials,
            summary,
            alpha,
            o_values,
            file_prefix=FILE_PREFIX,
            title_prefix=ALL_BASELINES_TITLE_PREFIX,
            width_col="plot_width",
            y_formatter=FuncFormatter(_currency_formatter),
            short_titles=ALL_BASELINES_SHORT_TITLES,
            font_scale=ALL_BASELINES_FONT_SCALE,
            stacked_fig_height=ALL_BASELINES_STACKED_HEIGHT,
            panel_fig_height=ALL_BASELINES_PANEL_HEIGHT,
        )
    finally:
        pm.FIG_DIR = saved_fig_dir


def write_summaries(trials: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    paths = [
        SUMMARY_DIR / f"{FILE_PREFIX}_trials_long.csv",
        SUMMARY_DIR / f"{FILE_PREFIX}_summary_by_alpha_o_method.csv",
        SUMMARY_DIR / f"{FILE_PREFIX}_coverage_table.csv",
        SUMMARY_DIR / f"{FILE_PREFIX}_width_table.csv",
    ]
    trials.to_csv(paths[0], index=False)
    summary.to_csv(paths[1], index=False)
    summary[["method", "o", "alpha", "nominal_coverage", "coverage_mean", "coverage_se", "coverage_std", "coverage_n"]].to_csv(paths[2], index=False)
    summary[
        [
            "method",
            "o",
            "alpha",
            "nominal_coverage",
            "width_mean",
            "width_se",
            "width_std",
            "width_median",
            "width_q25",
            "width_q75",
            "width_min",
            "width_max",
            "width_n_total",
            "width_n_finite",
            "width_n_infinite",
            "width_finite_rate",
            "width_infinite_rate",
        ]
    ].to_csv(paths[3], index=False)
    return paths


def run_suite(suite: AcsPlotSuite = DEFAULT_SUITE) -> list[Path]:
    configure_suite(suite)
    ensure_output_dirs()
    if suite.trials_csv is not None:
        print(f"Loading saved trials from {suite.trials_csv}...")
        trials = load_trials_from_csv(suite.trials_csv, width_col=suite.width_col)
    else:
        print(f"Loading ACS results from {RESULTS_ROOT} ({RESULT_GLOB})...")
        trials = load_trials()
    summary = summarize_trials(trials)
    summary_paths = write_summaries(trials, summary) if suite.trials_csv is None else []
    outputs: list[Path] = []
    if suite.include_nominal_panels and len(ALPHA_PANEL_VALUES) >= 2:
        outputs.extend([
            plot_nominal_coverage_width(trials, summary),
            plot_nominal_line_bands(summary),
        ])
    has_stdcp = "Std-CP" in set(trials["method"].astype(str))
    for alpha in ALPHA_PANEL_VALUES:
        out_suffix = suite.coverage_width_out_suffix
        outputs.append(
            plot_alpha_o_axis_panel(
                trials,
                summary,
                alpha,
                o_values=O_VALUES,
                out_suffix=out_suffix,
                width_ymin=suite.extra_width_axis_min,
                width_currency_format=suite.width_currency_format,
            )
        )
        if has_stdcp:
            outputs.append(
                plot_alpha_o_axis_panel(
                    trials,
                    summary,
                    alpha,
                    o_values=O_VALUES,
                    out_suffix=f"{out_suffix}_with_stdcp",
                    width_ymin=suite.extra_width_axis_min,
                    width_currency_format=suite.width_currency_format,
                    include_stdcp=True,
                )
            )
        if suite.extra_width_axis_min is not None:
            outputs.append(
                plot_alpha_o_axis_panel(
                    trials,
                    summary,
                    alpha,
                    o_values=O_VALUES,
                    out_suffix=f"{out_suffix}_width_from50k",
                    width_ymin=suite.extra_width_axis_min,
                    width_currency_format=suite.width_currency_format,
                )
            )
        if not suite.skip_all_baselines:
            outputs.extend(plot_all_baselines_by_o(trials, summary, alpha, o_values=O_VALUES))

    if suite.include_upto35_panels:
        try:
            trials_upto35 = load_trials(result_glob=f"{PERMUTED_T35_PREFIX}*")
            summary_upto35 = summarize_trials(trials_upto35)
            for alpha in UPTO35_ALPHA_VALUES:
                outputs.append(
                    plot_alpha_o_axis_panel(
                        trials_upto35,
                        summary_upto35,
                        alpha,
                        o_values=O_VALUES_UPTO35,
                        out_suffix="_upto35",
                    )
                )
            print("Loaded target-index-35 ACS results for o up to 35 plots.")
        except FileNotFoundError as exc:
            print(f"Skipping o-up-to-35 ACS panels ({exc}).")

    print("Saved summaries:")
    for path in summary_paths:
        print(f"  {path}")
    print("Saved plots:")
    for path in outputs:
        print(f"  {path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ACS true-marginal paper plots.")
    parser.add_argument(
        "--predictor",
        "--suite",
        dest="predictor",
        choices=("ols", "ols_income", "ols_yoep2000", "ols_yoep2000_notrim", "ols_legacy", "ols_2012_mean", "ols_2000_mean", "ols_2012_correction", "old_log", "ding_xgb", "ding_xgb_no_within", "rf", "rf_2012_mean", "rf_2012_correction", "rf_2012_residual", "rf_trim2_mean", "rf_legacy", "rf_no_within", "rf_2000_mean", "rf_2000_mean_trim_top", "studentized", "rf_no_permute_min21", "rf_no_permute_min31", "rf_yoep_fb_min31", "rf_yoep_fb_min21", "rf_new_sampling"),
        default="ols",
        help="Result set including rf_new_sampling.",
    )
    args = parser.parse_args()
    if args.predictor == "ols_income":
        suite = OLS_SUITE
    elif args.predictor == "ols_yoep2000":
        suite = OLS_YOEP2000_SUITE
    elif args.predictor == "ols_yoep2000_notrim":
        suite = OLS_YOEP2000_NOTRIM_SUITE
    elif args.predictor == "ols_legacy":
        suite = OLS_INCOME_LEGACY_SUITE
    elif args.predictor == "ols_2012_mean":
        suite = OLS_2012_MEAN_SUITE
    elif args.predictor == "ols_2000_mean":
        suite = OLS_2000_MEAN_SUITE
    elif args.predictor == "ols_2012_correction":
        suite = OLS_2012_CORRECTION_SUITE
    elif args.predictor == "old_log":
        suite = OLD_LOG_SUITE
    elif args.predictor == "ding_xgb":
        suite = DING_XGB_SUITE
    elif args.predictor == "ding_xgb_no_within":
        suite = DING_XGB_NO_WITHIN_SUITE
    elif args.predictor == "rf":
        suite = RF_SUITE
    elif args.predictor == "rf_2012_mean":
        suite = RF_2012_MEAN_SUITE
    elif args.predictor == "rf_2012_correction":
        suite = RF_2012_CORRECTION_SUITE
    elif args.predictor == "rf_2012_residual":
        suite = RF_2012_RESIDUAL_SUITE
    elif args.predictor == "rf_trim2_mean":
        suite = RF_TRIM2_MEAN_SUITE
    elif args.predictor == "rf_legacy":
        suite = RF_LEGACY_SUITE
    elif args.predictor == "rf_no_within":
        suite = RF_NO_WITHIN_SUITE
    elif args.predictor == "rf_2000_mean":
        suite = RF_2000_MEAN_SUITE
    elif args.predictor == "rf_2000_mean_trim_top":
        suite = RF_2000_MEAN_TRIM_TOP_SUITE
    elif args.predictor == "studentized":
        suite = STUDENTIZED_SUITE
    elif args.predictor == "rf_no_permute_min21":
        suite = RF_NO_PERMUTE_MIN21_SUITE
    elif args.predictor == "rf_no_permute_min31":
        suite = RF_NO_PERMUTE_MIN31_SUITE
    elif args.predictor == "rf_yoep_fb_min31":
        suite = RF_YOEP_FB_MIN31_SUITE
    elif args.predictor == "rf_yoep_fb_min21":
        suite = RF_YOEP_FB_MIN21_SUITE
    elif args.predictor == "rf_new_sampling":
        suite = RF_NEW_SAMPLING_SUITE
    else:
        suite = DEFAULT_SUITE
    run_suite(suite)


if __name__ == "__main__":
    main()
