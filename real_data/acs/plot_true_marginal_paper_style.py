#!/usr/bin/env python3
"""Create paper-ready plots for ACS TRUE MARGINAL coverage experiments.

This script creates publication-quality figures for the ACS true marginal experiments,
matching the style of both:
- DGP true marginal plots (error bars for coverage)
- ACS stratified plots (consistent color scheme and typography)

Input:
    real_data/acs/results/true_marginal/acs_true_marg_detailed.csv

Output:
    real_data/acs/NEW_PLOTS/true_marginal/paper_style/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent.parent
RESULTS_FILE = BASE_DIR / "acs/results/true_marginal/acs_true_marg_detailed.csv"
OUTPUT_DIR = BASE_DIR / "acs/NEW_PLOTS/true_marginal/paper_style"
SUMMARY_DIR = BASE_DIR / "acs/results/true_marginal/summaries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Palette and typography (matching DGP and ACS paper style)
# -----------------------------------------------------------------------------
O_COLORS = {
    0: "#0072B2",
    5: "#00A1D5",
    10: "#00B894",
    15: "#F39C12",
    20: "#E74C3C",
    25: "#8E44AD",
    30: "#C0392B",
    35: "#2C3E50",
    40: "#7F8C8D",
}

BASELINE_GRAYS = {
    "HCP": "#333333",
    "Pooling": "#666666",
    "Subsampling": "#999999",
    "Repeated": "#bbbbbb",
}

METHOD_LABELS = {
    "Donor-HCP": "D-HCP",
    "S-HCP": "S-HCP",
    "HCP": "HCP",
    "Pooling": "Pooling",
    "Subsampling": "Subsampling",
    "Repeated": "Repeated",
    "Std-CP": "Std-CP",
}

FONT_TICK = 24
FONT_LABEL = 24
FONT_TITLE = 26
FONT_LEGEND = 26

ALPHA = 0.20  # Target coverage level


# -----------------------------------------------------------------------------
# Legend handler for vertical color stacks
# -----------------------------------------------------------------------------
class _VerticalColorHandle:
    """Legend proxy handle for a vertical stack of colors."""

    def __init__(self, colors: List[str], label: str):
        self.colors = colors
        self._label = label

    def get_label(self) -> str:
        return self._label


class _VerticalColorHandleHandler(HandlerBase):
    """Draw a single legend key as stacked vertical color bars."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        n = max(1, len(orig_handle.colors))
        bar_h = height / float(n)
        artists = []
        for i, color in enumerate(orig_handle.colors):
            y0 = ydescent + i * bar_h
            rect = mpatches.Rectangle(
                (xdescent, y0),
                width,
                bar_h,
                facecolor=color,
                edgecolor="none",
                transform=trans,
            )
            artists.append(rect)

        border = mpatches.Rectangle(
            (xdescent, ydescent),
            width,
            height,
            fill=False,
            edgecolor="#222222",
            linewidth=0.8,
            transform=trans,
        )
        artists.append(border)
        return artists


# -----------------------------------------------------------------------------
# Coverage statistics computation (binomial standard error)
# -----------------------------------------------------------------------------
def _compute_coverage_stats(df: pd.DataFrame, method: str, o: int) -> Dict:
    """Compute coverage mean and binomial standard error for error bars."""
    d = df[(df["method"] == method) & (df["o"] == o)]
    cov_vals = d["coverage"].dropna().values  # Drop NaN values before computing mean
    n = len(cov_vals)

    if n == 0:
        return {"mean": np.nan, "se": np.nan, "n": 0}

    p = np.mean(cov_vals)
    # Binomial SE: sqrt(p(1-p)/n)
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0

    return {"mean": p, "se": se, "n": n}


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------
def _boxplot_style(
    ax,
    data,
    pos,
    color,
    edge_color=None,
    hatch=None,
    alpha_val=0.75,
    linestyle="-",
):
    """Create a styled boxplot matching DGP paper style."""
    if edge_color is None:
        edge_color = color

    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=0.58,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color=edge_color, linewidth=1.2, linestyle=linestyle),
        capprops=dict(color=edge_color, linewidth=1.2, linestyle=linestyle),
        boxprops=dict(
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.5,
            alpha=alpha_val,
            linestyle=linestyle,
            hatch=hatch,
        ),
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor=color,
            markeredgecolor=edge_color,
            alpha=0.5,
        ),
        zorder=3,
    )
    return bp


def _get_metric_data(df: pd.DataFrame, method: str, o: int, metric_col: str):
    """Get metric values for a specific method and o value."""
    d = df[(df["method"] == method) & (df["o"] == o)]
    vals = d[metric_col].values
    vals = vals[np.isfinite(vals)]
    return vals


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------
def _plot_coverage_and_width(
    df: pd.DataFrame,
    out_file: Path,
    o_vals: List[int],
    method_specs: List[tuple],
    title_cov: str,
    title_width: str,
    layout: str = "side_by_side",
    metric_key: str = "width_income",
):
    """
    Create side-by-side or stacked coverage + width plots.

    For coverage: use error bars (mean ± binomial SE)
    For width: use boxplots

    Args:
        method_specs: List of (method_name, color_or_dict) where color_or_dict is either:
            - A single color string for single o value
            - A dict mapping o -> color for multiple o values
    """
    if layout == "side_by_side":
        fig, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(18, 6))
    else:  # stacked
        fig, (ax_cov, ax_width) = plt.subplots(2, 1, figsize=(12, 12))

    fig.patch.set_alpha(0.0)

    n_methods = len(method_specs)
    n_o = len(o_vals)
    group_width = n_o * 0.7
    group_gap = 0.5
    total_width_per_method = group_width + group_gap

    # === Coverage panel (error bars) ===
    for m_idx, (method, color_spec) in enumerate(method_specs):
        base_x = m_idx * total_width_per_method

        for o_idx, o in enumerate(o_vals):
            if isinstance(color_spec, dict):
                color = color_spec[o]
            else:
                color = color_spec

            pos = base_x + o_idx * 0.7

            # Compute coverage statistics
            stats = _compute_coverage_stats(df, method, o)

            if not np.isnan(stats["mean"]):
                # Plot error bar
                ax_cov.errorbar(
                    pos,
                    stats["mean"],
                    yerr=stats["se"],
                    fmt='o',
                    color=color,
                    markersize=8,
                    capsize=5,
                    capthick=1.5,
                    elinewidth=1.5,
                    markeredgewidth=1.5,
                    markeredgecolor=color,
                    zorder=3,
                )

    # Target coverage line
    ax_cov.axhline(1 - ALPHA, color="red", linestyle="--", linewidth=1.5, zorder=2, alpha=0.7)

    ax_cov.set_ylim(0.65, 1.02)
    ax_cov.set_ylabel("Coverage", fontsize=FONT_LABEL, fontweight="bold")
    ax_cov.set_title(title_cov, fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax_cov.tick_params(axis="both", labelsize=FONT_TICK)
    ax_cov.grid(True, alpha=0.2, zorder=1)
    ax_cov.set_facecolor("white")

    # Set x-axis ticks for coverage
    method_centers = [m_idx * total_width_per_method + group_width / 2 - 0.35 for m_idx in range(n_methods)]
    ax_cov.set_xticks(method_centers)
    ax_cov.set_xticklabels([METHOD_LABELS.get(m, m) for m, _ in method_specs], fontsize=FONT_TICK)

    # === Width panel (boxplots) ===
    for m_idx, (method, color_spec) in enumerate(method_specs):
        base_x = m_idx * total_width_per_method

        for o_idx, o in enumerate(o_vals):
            if isinstance(color_spec, dict):
                color = color_spec[o]
            else:
                color = color_spec

            pos = base_x + o_idx * 0.7

            # Get width data
            vals = _get_metric_data(df, method, o, metric_key)

            if len(vals) > 0:
                _boxplot_style(ax_width, vals, pos, color=color)

    ax_width.set_ylabel("Interval Width (Income)", fontsize=FONT_LABEL, fontweight="bold")
    ax_width.set_title(title_width, fontsize=FONT_TITLE, fontweight="bold", pad=15)
    ax_width.tick_params(axis="both", labelsize=FONT_TICK)
    ax_width.grid(True, alpha=0.2, zorder=1)
    ax_width.set_facecolor("white")

    # Set x-axis ticks for width
    ax_width.set_xticks(method_centers)
    ax_width.set_xticklabels([METHOD_LABELS.get(m, m) for m, _ in method_specs], fontsize=FONT_TICK)

    # === Legend ===
    handles = []
    labels = []

    # Add o value colors
    for o in o_vals:
        handles.append(mpatches.Patch(color=O_COLORS[o], label=f"$o = {o}$"))
        labels.append(f"$o = {o}$")

    if layout == "side_by_side":
        ax_width.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=FONT_LEGEND,
            frameon=True,
            fancybox=False,
            edgecolor="#222222",
            framealpha=0.95,
        )
    else:
        ax_cov.legend(
            handles,
            labels,
            loc="lower left",
            fontsize=FONT_LEGEND,
            frameon=True,
            fancybox=False,
            edgecolor="#222222",
            framealpha=0.95,
        )

    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved: {out_file.name}")


# -----------------------------------------------------------------------------
# Generate plots
# -----------------------------------------------------------------------------
def generate_all_plots(df: pd.DataFrame):
    """Generate all paper-ready plots."""

    # Get unique o values
    o_vals_all = sorted(df["o"].unique())
    o_vals_upto20 = [o for o in o_vals_all if o <= 20]

    print("\nGenerating plots...")

    # Plot 1: D-HCP vs HCP (side-by-side, o ≤ 20)
    method_specs_1 = [
        ("Donor-HCP", {o: O_COLORS[o] for o in o_vals_upto20}),
        ("HCP", BASELINE_GRAYS["HCP"]),
    ]
    _plot_coverage_and_width(
        df,
        OUTPUT_DIR / "acs_true_marg_1_dhcp_vs_hcp_upto20.pdf",
        o_vals_upto20,
        method_specs_1,
        title_cov="Coverage: D-HCP vs HCP (True Marginal)",
        title_width="Interval Width: D-HCP vs HCP (True Marginal)",
        layout="side_by_side",
    )

    # Plot 2: D-HCP vs HCP (stacked, all o)
    method_specs_2 = [
        ("Donor-HCP", {o: O_COLORS[o] for o in o_vals_all}),
        ("HCP", BASELINE_GRAYS["HCP"]),
    ]
    _plot_coverage_and_width(
        df,
        OUTPUT_DIR / "acs_true_marg_2_dhcp_vs_hcp_all_o.pdf",
        o_vals_all,
        method_specs_2,
        title_cov="Coverage: D-HCP vs HCP, All $o$ (True Marginal)",
        title_width="Interval Width: D-HCP vs HCP, All $o$ (True Marginal)",
        layout="stacked",
    )

    # Plot 3: D-HCP vs all baselines (stacked, all o)
    method_specs_3 = [
        ("Donor-HCP", {o: O_COLORS[o] for o in o_vals_all}),
        ("HCP", BASELINE_GRAYS["HCP"]),
        ("Pooling", BASELINE_GRAYS["Pooling"]),
        ("Subsampling", BASELINE_GRAYS["Subsampling"]),
        ("Repeated", BASELINE_GRAYS["Repeated"]),
    ]
    _plot_coverage_and_width(
        df,
        OUTPUT_DIR / "acs_true_marg_3_dhcp_vs_all_baselines.pdf",
        o_vals_all,
        method_specs_3,
        title_cov="Coverage: D-HCP vs Baselines (True Marginal)",
        title_width="Interval Width: D-HCP vs Baselines (True Marginal)",
        layout="stacked",
    )

    # Plot 4: D-HCP vs S-HCP (side-by-side, o ≤ 20)
    method_specs_4 = [
        ("Donor-HCP", {o: O_COLORS[o] for o in o_vals_upto20}),
        ("S-HCP", {o: O_COLORS[o] for o in o_vals_upto20}),
        ("HCP", BASELINE_GRAYS["HCP"]),
    ]
    _plot_coverage_and_width(
        df,
        OUTPUT_DIR / "acs_true_marg_4_dhcp_vs_shcp_with_hcp_upto20.pdf",
        o_vals_upto20,
        method_specs_4,
        title_cov="Coverage: D-HCP vs S-HCP vs HCP (True Marginal)",
        title_width="Interval Width: D-HCP vs S-HCP vs HCP (True Marginal)",
        layout="side_by_side",
    )

    print("\nAll plots generated successfully!")


# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------
def generate_summary_tables(df: pd.DataFrame):
    """Generate summary statistics CSV files."""

    print("\nGenerating summary tables...")

    # Coverage summary (filter out NaN rows from baselines at o>0)
    df_valid = df[df["coverage"].notna()].copy()
    coverage_summary = df_valid.groupby(["o", "method"])["coverage"].agg(["mean", "std", "count"]).reset_index()
    coverage_summary.columns = ["o", "method", "coverage_mean", "coverage_std", "n"]
    coverage_summary["coverage_se"] = coverage_summary.apply(
        lambda row: np.sqrt(row["coverage_mean"] * (1 - row["coverage_mean"]) / row["n"]) if row["n"] > 0 else np.nan, axis=1
    )

    # Pivot to wide format
    coverage_wide = coverage_summary.pivot(index="o", columns="method", values="coverage_mean")
    coverage_wide.to_csv(SUMMARY_DIR / "acs_true_marg_coverage_summary.csv")
    print(f"  Saved: {SUMMARY_DIR.name}/acs_true_marg_coverage_summary.csv")

    # Width summary (filter out NaN rows from baselines at o>0)
    width_summary = df_valid.groupby(["o", "method"])["width_income"].agg(["median", "mean", "std"]).reset_index()
    width_summary.columns = ["o", "method", "width_median", "width_mean", "width_std"]

    # Pivot to wide format
    width_wide = width_summary.pivot(index="o", columns="method", values="width_median")
    width_wide.to_csv(SUMMARY_DIR / "acs_true_marg_width_summary.csv")
    print(f"  Saved: {SUMMARY_DIR.name}/acs_true_marg_width_summary.csv")

    # Long format summary (filter out NaN rows from baselines at o>0)
    summary_long = df_valid.groupby(["method", "o"]).agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        width_income_median=("width_income", "median"),
        width_income_mean=("width_income", "mean"),
        width_income_std=("width_income", "std"),
        n=("coverage", "count"),
    ).reset_index()

    summary_long["coverage_se"] = summary_long.apply(
        lambda row: np.sqrt(row["coverage_mean"] * (1 - row["coverage_mean"]) / row["n"]) if row["n"] > 0 else np.nan, axis=1
    )

    summary_long.to_csv(SUMMARY_DIR / "acs_true_marg_summary_long.csv", index=False)
    print(f"  Saved: {SUMMARY_DIR.name}/acs_true_marg_summary_long.csv")

    print("\nSummary statistics:")
    print(summary_long.to_string(index=False))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("ACS TRUE MARGINAL COVERAGE - Paper Plots")
    print("=" * 70)

    if not RESULTS_FILE.exists():
        print(f"\nError: Results file not found: {RESULTS_FILE}")
        print("Please run acs_true_marginal_experiments.py first.")
        exit(1)

    print(f"\nLoading results from: {RESULTS_FILE}")
    df = pd.read_csv(RESULTS_FILE)

    print(f"  Total rows: {len(df)}")
    print(f"  Methods: {sorted(df['method'].unique())}")
    print(f"  o values: {sorted(df['o'].unique())}")
    print(f"  Replicates: {df['replicate'].nunique()}")

    # Generate plots
    generate_all_plots(df)

    # Generate summary tables
    generate_summary_tables(df)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Summary directory: {SUMMARY_DIR}")
    print("\nDone!")
