#!/usr/bin/env python3
"""
Create ACS real-data plots in the same style as the DGP plots,
but use raw-income width on the width panels.

Plots produced:
1. Side-by-side coverage/width: D-HCP vs HCP for o <= 20
2. Coverage only: D-HCP vs HCP + Pooling + Subsampling + Repeated for all o
3. Width only:    D-HCP vs HCP + Pooling + Subsampling + Repeated for all o

Input:
    /Users/soham/UPenn/Claude/hier_current/real_data/acs/percentile_bottom98_claude/results/acs_new_detailed_corrected.csv

Output folder:
    /Users/soham/UPenn/Claude/hier_current/real_data/acs/NEW_PLOTS
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
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------
BASE_DIR = Path("/Users/soham/UPenn/Claude/hier_current/real_data/acs")
RAW_FILE = (
    BASE_DIR
    / "percentile_bottom98_claude"
    / "results"
    / "acs_new_detailed_corrected.csv"
)
OUT_PLOTS = BASE_DIR / "NEW_PLOTS"
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
WIDTH_YMIN = 50000

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _find_raw_file() -> Path:
    print("Checking raw file:", RAW_FILE)
    print("Exists:", RAW_FILE.exists())
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Could not find file: {RAW_FILE}")
    return RAW_FILE


def _human_number_format(y, _):
    ay = abs(y)
    if ay >= 1e12:
        return f"{y:.1e}"
    if ay >= 1e9:
        return f"{y / 1e9:.1f}B"
    if ay >= 1e6:
        return f"{y / 1e6:.1f}M"
    if ay >= 1e3:
        return f"{y / 1e3:.0f}K"
    if ay >= 1:
        return f"{y:.0f}"
    return f"{y:.2f}"


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
        ax.set_ylim(bottom=WIDTH_YMIN)
        ax.yaxis.set_major_formatter(FuncFormatter(_human_number_format))
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")


def _exp_metric_by_o(df: pd.DataFrame, method: str, o: int, metric_col: str) -> np.ndarray:
    d = df[(df["method"] == method) & (df["o"] == o)]
    vals = d.groupby("replicate", as_index=False)[metric_col].mean()[metric_col].to_numpy()
    vals = vals[np.isfinite(vals)]
    return vals


def _exp_metric_all(df: pd.DataFrame, method: str, metric_col: str) -> np.ndarray:
    d = df[df["method"] == method]
    vals = d.groupby("replicate", as_index=False)[metric_col].mean()[metric_col].to_numpy()
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


def _add_custom_bottom_legend(
    fig,
    o_vals,
    include_hcp=True,
    include_pool=False,
    include_sub=False,
    include_rep=False,
):
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
    o_vals = sorted(int(o) for o in df["o"].unique() if int(o) <= 20)

    fig, axes = plt.subplots(1, 2, figsize=(21.0, 8.9))
    fig.subplots_adjust(wspace=0.32)

    panel_specs = [
        (axes[0], "coverage", "Coverage", "Coverage: D-HCP vs HCP", "coverage"),
        (axes[1], "width_income", "Width (income units)", "Width: D-HCP vs HCP", "width"),
    ]

    for ax, metric_col, ylabel, title, metric_kind in panel_specs:
        pos = 0.0
        tick_positions = []
        tick_labels = []

        for o in o_vals:
            vals_d = _exp_metric_by_o(df, "donor-HCP-randomized", o, metric_col)
            if len(vals_d):
                _boxplot_style(ax, vals_d, pos, O_COLORS[o], width=0.58)
                tick_positions.append(pos)
                tick_labels.append(f"o={o}")
            pos += 1.0

        pos += 1.1
        vals_h = _exp_metric_all(df, "HCP", metric_col)
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
        _style_axis(ax, ylabel, title, metric_kind)

    _add_custom_bottom_legend(fig, o_vals, include_hcp=True)

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    out = OUT_PLOTS / "acs_dhcp_vs_hcp_side_by_side_o_upto20.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ---------------------------------------------------------------------
# Plot 2: Coverage, D-HCP vs all baselines, all o
# ---------------------------------------------------------------------
def plot_dhcp_vs_all_baselines_coverage(df: pd.DataFrame):
    o_vals = sorted(int(o) for o in df["o"].unique())

    fig, ax = plt.subplots(1, 1, figsize=(18.5, 8.6))

    pos = 0.0
    tick_positions = []
    tick_labels = []

    for o in o_vals:
        vals_d = _exp_metric_by_o(df, "donor-HCP-randomized", o, "coverage")
        if len(vals_d):
            _boxplot_style(ax, vals_d, pos, O_COLORS[o], width=0.58)
            tick_positions.append(pos)
            tick_labels.append(f"o={o}")
        pos += 1.0

    pos += 1.2
    for label, method_name, color in [
        ("HCP", "HCP", BASELINE_COLORS["HCP"]),
        ("Pooling", "Pooling", BASELINE_COLORS["Pooling"]),
        ("Subsampling", "Subsampling", BASELINE_COLORS["Subsampling"]),
        ("Repeated", "Repeated", BASELINE_COLORS["Repeated"]),
    ]:
        vals = _exp_metric_all(df, method_name, "coverage")
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
    out = OUT_PLOTS / "acs_dhcp_vs_all_baselines_coverage_all_o.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ---------------------------------------------------------------------
# Plot 3: Width, D-HCP vs all baselines, all o
# ---------------------------------------------------------------------
def plot_dhcp_vs_all_baselines_width(df: pd.DataFrame):
    o_vals = sorted(int(o) for o in df["o"].unique())

    fig, ax = plt.subplots(1, 1, figsize=(18.5, 8.6))

    pos = 0.0
    tick_positions = []
    tick_labels = []

    for o in o_vals:
        vals_d = _exp_metric_by_o(df, "donor-HCP-randomized", o, "width_income")
        if len(vals_d):
            _boxplot_style(ax, vals_d, pos, O_COLORS[o], width=0.58)
            tick_positions.append(pos)
            tick_labels.append(f"o={o}")
        pos += 1.0

    pos += 1.2
    for label, method_name, color in [
        ("HCP", "HCP", BASELINE_COLORS["HCP"]),
        ("Pooling", "Pooling", BASELINE_COLORS["Pooling"]),
        ("Subsampling", "Subsampling", BASELINE_COLORS["Subsampling"]),
        ("Repeated", "Repeated", BASELINE_COLORS["Repeated"]),
    ]:
        vals = _exp_metric_all(df, method_name, "width_income")
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
    _style_axis(ax, "Width (income units)", "Width: D-HCP vs baselines", "width")

    _add_custom_bottom_legend(
        fig,
        o_vals,
        include_hcp=True,
        include_pool=True,
        include_sub=True,
        include_rep=True,
    )

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    out = OUT_PLOTS / "acs_dhcp_vs_all_baselines_width_all_o.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    raw_path = _find_raw_file()
    df = pd.read_csv(raw_path)

    methods_keep = {
        "donor-HCP-randomized",
        "HCP",
        "Pooling",
        "Subsampling",
        "Repeated",
    }
    df = df[df["method"].isin(methods_keep)].copy()

    if "width_income" not in df.columns:
        raise ValueError("The input file does not contain a 'width_income' column.")

    plot_dhcp_vs_hcp_side_by_side(df)
    plot_dhcp_vs_all_baselines_coverage(df)
    plot_dhcp_vs_all_baselines_width(df)


if __name__ == "__main__":
    main()