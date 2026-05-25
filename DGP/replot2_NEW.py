#!/usr/bin/env python3
"""
Create the requested D-HCP plots from the raw results file.

Plots produced:
1. Side-by-side coverage/width: D-HCP vs HCP for o <= 20
2. Coverage only: D-HCP vs HCP + Pooling + Subsampling + Repeated for all o
3. Width only:    D-HCP vs HCP + Pooling + Subsampling + Repeated for all o

Width plots are truncated at y = 15.

Input:
    <repo>/NEW_RESULTS_2/
    fixedN21_K20_d5_u15_rho05_joint_xy_marginal_parallel5/
    raw_results_effect_of_o_joint_xy_marginal.csv

Output:
    <repo>/NEW_PLOTS_2/
    fixedN21_K20_d5_u15_rho05_joint_xy_marginal_parallel5/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea

# ---------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = (
    BASE_DIR
    / "NEW_RESULTS_2"
    / "fixedN21_K20_d5_u15_rho05_joint_xy_marginal_parallel5"
    / "raw_results_effect_of_o_joint_xy_marginal.csv"
)

OUT_PLOTS = (
    BASE_DIR
    / "NEW_PLOTS_2"
    / "fixedN21_K20_d5_u15_rho05_joint_xy_marginal_parallel5"
)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

ALPHA = 0.20

# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
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

BASELINE_COLORS = {
    "HCP": "#111111",
    "Pooling": "#666666",
    "Subsampling": "#999999",
    "Repeated": "#bbbbbb",
}

TITLE_FS = 32
LABEL_FS = 34
TICK_FS = 28
LEGEND_FS = 30

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _find_raw_file() -> Path:
    print("Checking raw file:", RAW_FILE)
    print("Exists:", RAW_FILE.exists())
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Could not find file: {RAW_FILE}")
    return RAW_FILE


def _boxplot_style(
    ax,
    data,
    pos,
    color,
    width=0.58,
    edge_color=None,
    hatch=None,
    alpha=0.75,
    linestyle="-",
):
    if edge_color is None:
        edge_color = color

    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color=edge_color, linewidth=1.2, linestyle=linestyle),
        capprops=dict(color=edge_color, linewidth=1.2, linestyle=linestyle),
        flierprops=dict(marker=".", color=edge_color, markersize=3.5, alpha=0.35),
        boxprops=dict(
            facecolor=color,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=1.2,
            linestyle=linestyle,
        ),
    )
    if hatch:
        for patch in bp["boxes"]:
            patch.set_hatch(hatch)
    return bp


def _style_axis(ax, ylabel: str, title: str, metric: str):
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_xlabel("Observed history size o", fontsize=LABEL_FS - 2)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    if metric == "coverage":
        ax.axhline(1.0 - ALPHA, color="black", linewidth=1.4, linestyle="--", alpha=0.60)
        ax.set_ylim(0.6, 1.05)
    else:
        ax.set_ylim(0, 15)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")


def _get_o_column(df: pd.DataFrame) -> str:
    if "test_sample_size_o" in df.columns:
        return "test_sample_size_o"
    if "o_observed" in df.columns:
        return "o_observed"
    raise ValueError("Could not find o column.")


def _exp_metric_by_o(df: pd.DataFrame, o_col: str, value_col: str, o: int, metric: str) -> np.ndarray:
    d = df[df[o_col] == o]
    vals = d.groupby("experiment", as_index=False)[value_col].mean()[value_col].to_numpy()
    if metric == "width":
        vals = vals[np.isfinite(vals)]
    return vals


def _exp_metric_all(df: pd.DataFrame, value_col: str, metric: str) -> np.ndarray:
    vals = df.groupby("experiment", as_index=False)[value_col].mean()[value_col].to_numpy()
    if metric == "width":
        vals = vals[np.isfinite(vals)]
    return vals


def _make_color_cluster(o_vals):
    width = 16 + 8 * len(o_vals)
    height = 34
    da = DrawingArea(width, height, clip=False)
    x0 = 6
    dx = 7
    y0, y1 = 5, 29
    for k, o in enumerate(o_vals):
        x = x0 + k * dx
        line = mlines.Line2D(
            [x, x],
            [y0, y1],
            color=O_COLORS[o],
            linewidth=4.5,
            solid_capstyle="butt",
        )
        da.add_artist(line)
    return da


def _make_patch_box(facecolor, edgecolor, hatch=None, alpha=1.0):
    da = DrawingArea(30, 24, clip=False)
    rect = mpatches.Rectangle(
        (2, 3),
        24,
        18,
        facecolor=facecolor,
        edgecolor=edgecolor,
        hatch=hatch,
        linewidth=1.2,
        alpha=alpha,
    )
    da.add_artist(rect)
    return da


def _add_custom_bottom_legend(fig, o_vals, include_hcp=True, include_pool=False, include_sub=False, include_rep=False):
    text_props = dict(size=LEGEND_FS)

    cluster = _make_color_cluster(o_vals)
    text_dhcp = TextArea("D-HCP", textprops=text_props)

    children = [
        cluster,
        DrawingArea(10, 1, clip=False),
        text_dhcp,
    ]

    def add_group(facecolor, edgecolor, label, hatch=None, alpha=1.0):
        children.extend(
            [
                DrawingArea(32, 1, clip=False),
                _make_patch_box(facecolor=facecolor, edgecolor=edgecolor, hatch=hatch, alpha=alpha),
                DrawingArea(10, 1, clip=False),
                TextArea(label, textprops=text_props),
            ]
        )

    if include_hcp:
        add_group("#111111", "#111111", "HCP", hatch="///", alpha=0.30)
    if include_pool:
        add_group(BASELINE_COLORS["Pooling"], "#222222", "Pooling", hatch="//", alpha=0.45)
    if include_sub:
        add_group(BASELINE_COLORS["Subsampling"], "#222222", "Subsampling", hatch="//", alpha=0.45)
    if include_rep:
        add_group(BASELINE_COLORS["Repeated"], "#222222", "Repeated", hatch="//", alpha=0.45)

    legend_box = HPacker(children=children, align="center", pad=0, sep=0)

    anchored = AnchoredOffsetbox(
        loc="lower center",
        child=legend_box,
        frameon=False,
        pad=0.0,
        borderpad=0.0,
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
    )
    fig.add_artist(anchored)


# ---------------------------------------------------------------------
# Plot 1: D-HCP vs HCP, side-by-side coverage/width, o <= 20
# ---------------------------------------------------------------------
def plot_dhcp_vs_hcp_side_by_side(df: pd.DataFrame):
    o_col = _get_o_column(df)
    o_vals = sorted(int(o) for o in df[o_col].unique() if int(o) <= 20)

    fig, axes = plt.subplots(1, 2, figsize=(21.0, 8.9))
    fig.subplots_adjust(wspace=0.32)

    for ax, metric, ylabel, dcol, hcol, title in [
        (
            axes[0],
            "coverage",
            "Coverage",
            "coverage_donor_hcp_randomized",
            "coverage_hcp",
            "Coverage: D-HCP vs HCP",
        ),
        (
            axes[1],
            "width",
            "Width",
            "width_donor_hcp_randomized",
            "width_hcp",
            "Width: D-HCP vs HCP",
        ),
    ]:
        pos = 0.0
        tick_positions = []
        tick_labels = []

        for o in o_vals:
            vals_d = _exp_metric_by_o(df, o_col, dcol, o, metric)
            if len(vals_d):
                _boxplot_style(ax, vals_d, pos, O_COLORS[o], width=0.58)
                tick_positions.append(pos)
                tick_labels.append(f"o={o}")
            pos += 1.0

        pos += 1.1
        vals_h = _exp_metric_all(df, hcol, metric)
        if len(vals_h):
            _boxplot_style(
                ax,
                vals_h,
                pos,
                color="#111111",
                edge_color="#111111",
                alpha=0.30,
                linestyle="--",
                hatch="///",
                width=0.58,
            )
            tick_positions.append(pos)
            tick_labels.append("HCP")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=18, ha="right")
        _style_axis(ax, ylabel, title, metric)

    _add_custom_bottom_legend(fig, o_vals, include_hcp=True)

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    out = OUT_PLOTS / "dhcp_vs_hcp_side_by_side_o_upto20.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ---------------------------------------------------------------------
# Plot 2a: Coverage, D-HCP vs all baselines, all o
# ---------------------------------------------------------------------
def plot_dhcp_vs_all_baselines_coverage(df: pd.DataFrame):
    o_col = _get_o_column(df)
    o_vals = sorted(int(o) for o in df[o_col].unique())

    fig, ax = plt.subplots(1, 1, figsize=(18.5, 8.6))

    pos = 0.0
    tick_positions = []
    tick_labels = []

    for o in o_vals:
        vals_d = _exp_metric_by_o(df, o_col, "coverage_donor_hcp_randomized", o, "coverage")
        if len(vals_d):
            _boxplot_style(ax, vals_d, pos, O_COLORS[o], width=0.58)
            tick_positions.append(pos)
            tick_labels.append(f"o={o}")
        pos += 1.0

    pos += 1.2
    for label, col, color in [
        ("HCP", "coverage_hcp", BASELINE_COLORS["HCP"]),
        ("Pooling", "coverage_pool", BASELINE_COLORS["Pooling"]),
        ("Subsampling", "coverage_sub", BASELINE_COLORS["Subsampling"]),
        ("Repeated", "coverage_rep", BASELINE_COLORS["Repeated"]),
    ]:
        vals = _exp_metric_all(df, col, "coverage")
        if len(vals):
            _boxplot_style(
                ax,
                vals,
                pos,
                color=color,
                edge_color="#222222",
                alpha=0.45 if label != "HCP" else 0.30,
                linestyle="--",
                hatch="//" if label != "HCP" else "///",
                width=0.58,
            )
            tick_positions.append(pos)
            tick_labels.append(label)
        pos += 1.55

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=18, ha="right")
    _style_axis(ax, "Coverage", "Coverage: D-HCP vs baselines", "coverage")

    _add_custom_bottom_legend(
        fig,
        o_vals,
        include_hcp=True,
        include_pool=True,
        include_sub=True,
        include_rep=True,
    )

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    out = OUT_PLOTS / "dhcp_vs_all_baselines_coverage_all_o.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ---------------------------------------------------------------------
# Plot 2b: Width, D-HCP vs all baselines, all o
# ---------------------------------------------------------------------
def plot_dhcp_vs_all_baselines_width(df: pd.DataFrame):
    o_col = _get_o_column(df)
    o_vals = sorted(int(o) for o in df[o_col].unique())

    fig, ax = plt.subplots(1, 1, figsize=(18.5, 8.6))

    pos = 0.0
    tick_positions = []
    tick_labels = []

    for o in o_vals:
        vals_d = _exp_metric_by_o(df, o_col, "width_donor_hcp_randomized", o, "width")
        if len(vals_d):
            _boxplot_style(ax, vals_d, pos, O_COLORS[o], width=0.58)
            tick_positions.append(pos)
            tick_labels.append(f"o={o}")
        pos += 1.0

    pos += 1.2
    for label, col, color in [
        ("HCP", "width_hcp", BASELINE_COLORS["HCP"]),
        ("Pooling", "width_pool", BASELINE_COLORS["Pooling"]),
        ("Subsampling", "width_sub", BASELINE_COLORS["Subsampling"]),
        ("Repeated", "width_rep", BASELINE_COLORS["Repeated"]),
    ]:
        vals = _exp_metric_all(df, col, "width")
        if len(vals):
            _boxplot_style(
                ax,
                vals,
                pos,
                color=color,
                edge_color="#222222",
                alpha=0.45 if label != "HCP" else 0.30,
                linestyle="--",
                hatch="//" if label != "HCP" else "///",
                width=0.58,
            )
            tick_positions.append(pos)
            tick_labels.append(label)
        pos += 1.55

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=18, ha="right")
    _style_axis(ax, "Width", "Width: D-HCP vs baselines", "width")

    _add_custom_bottom_legend(
        fig,
        o_vals,
        include_hcp=True,
        include_pool=True,
        include_sub=True,
        include_rep=True,
    )

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    out = OUT_PLOTS / "dhcp_vs_all_baselines_width_all_o.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    raw_path = _find_raw_file()
    df = pd.read_csv(raw_path)

    if "alpha" in df.columns:
        df = df[df["alpha"] == ALPHA].copy()

    plot_dhcp_vs_hcp_side_by_side(df)
    plot_dhcp_vs_all_baselines_coverage(df)
    plot_dhcp_vs_all_baselines_width(df)


if __name__ == "__main__":
    main()