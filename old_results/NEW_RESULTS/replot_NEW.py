#!/usr/bin/env python3
"""
Make ONLY the 'within vs without training' plot from the raw results file.

- Reads:  NEW_RESULTS/raw_results_complete_tauB0_tauB6.csv
- Uses:   alpha = 0.20
- Plots:  Donor-HCP without training vs Donor-HCP with training, for o in {0,10,20,30},
          plus a single HCP baseline boxplot at the end.
- Saves:  NEW_PLOTS_2/tauB_<tau>_4_dhcp_with_vs_without_o_0_10_20_30.pdf

This version uses a manually packed bottom legend so:
- the vertical color marks are close together
- the text sits close to the marks
- the spacing between legend groups is controlled
- the shaded boxes are larger
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
# Path handling
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()

if SCRIPT_PATH.parent.name == "NEW_RESULTS":
    BASE_DIR = SCRIPT_PATH.parent.parent
else:
    BASE_DIR = SCRIPT_PATH.parent

IN_DIR = BASE_DIR / "NEW_RESULTS"
OUT_PLOTS = BASE_DIR / "NEW_PLOTS_2"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

RAW_FILE = IN_DIR / "raw_results_complete_tauB0_tauB6.csv"

ALPHA = 0.20

# ---------------------------------------------------------------------
# Style palette
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

# Large sizes for full-width LaTeX figure use
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


def _exp_metric(df: pd.DataFrame, method: str, o: int, metric: str) -> np.ndarray:
    d = df[(df["method"] == method) & (df["o"] == o)]
    vals = d.groupby("experiment", as_index=False)[metric].mean()[metric].to_numpy()
    if metric == "width":
        vals = vals[np.isfinite(vals)]
    return vals


def _exp_metric_o_independent(df: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    d = df[df["method"] == method]
    vals = d.groupby("experiment", as_index=False)[metric].mean()[metric].to_numpy()
    if metric == "width":
        vals = vals[np.isfinite(vals)]
    return vals


def _style_axis(ax, ylabel: str, title: str, alpha: float, metric: str):
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_xlabel("Observed history size o", fontsize=LABEL_FS - 2)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    if metric == "coverage":
        ax.axhline(1.0 - alpha, color="black", linewidth=1.4, linestyle="--", alpha=0.60)
        ax.set_ylim(0.6, 1.05)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")


def _make_color_cluster(o_vals):
    # Close-together vertical color marks with full text-height feel
    width = 18 + 10 * len(o_vals)
    height = 34
    da = DrawingArea(width, height, clip=False)
    x0 = 6
    dx = 9  # close spacing between marks
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
    # Larger legend box
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


def _add_custom_bottom_legend(fig, o_vals):
    text_props = dict(size=LEGEND_FS)

    cluster = _make_color_cluster(o_vals)
    text1 = TextArea("D-HCP with training", textprops=text_props)

    patch_no = _make_patch_box(
        facecolor="#f4a582",
        edgecolor="#333333",
        hatch="///",
        alpha=0.65,
    )
    text2 = TextArea("D-HCP without training", textprops=text_props)

    patch_hcp = _make_patch_box(
        facecolor="#111111",
        edgecolor="#111111",
        hatch="///",
        alpha=0.30,
    )
    text3 = TextArea("HCP", textprops=text_props)

    # Spacers: small after cluster, then consistent larger gaps between groups
    spacer_small = DrawingArea(12, 1, clip=False)
    spacer_group = DrawingArea(36, 1, clip=False)

    legend_box = HPacker(
        children=[
            cluster,
            spacer_small,
            text1,
            spacer_group,
            patch_no,
            spacer_small,
            text2,
            spacer_group,
            patch_hcp,
            spacer_small,
            text3,
        ],
        align="center",
        pad=0,
        sep=0,
    )

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
# Plot only: within vs without
# ---------------------------------------------------------------------
def plot_within_vs_without(df: pd.DataFrame, tau_b: float):
    d = df[(df["tau_B"] == tau_b) & (df["alpha"] == ALPHA)].copy()
    o_vals = [o for o in [0, 10, 20, 30] if o in sorted(d["o"].unique())]

    fig, axes = plt.subplots(1, 2, figsize=(21.0, 8.9))
    fig.subplots_adjust(wspace=0.32)

    for ax, metric, ylabel in [
        (axes[0], "coverage", "Coverage"),
        (axes[1], "width", "Width"),
    ]:
        group_gap = 1.9
        no_off, with_off = 0.0, 0.72
        tick_positions = []
        tick_labels = []

        # no-within first, then with-within
        for gi, o in enumerate(o_vals):
            base = gi * group_gap

            vals_no = _exp_metric(d, "Donor-HCP-no-within", o, metric)
            vals_with = _exp_metric(d, "Donor-HCP-within", o, metric)

            if len(vals_no):
                _boxplot_style(
                    ax,
                    vals_no,
                    base + no_off,
                    color="#f4a582",
                    edge_color="#333333",
                    alpha=0.65,
                    hatch="///",
                    width=0.58,
                )

            if len(vals_with):
                _boxplot_style(
                    ax,
                    vals_with,
                    base + with_off,
                    color=O_COLORS[o],
                    width=0.58,
                )

            tick_positions.append(base + 0.36)
            tick_labels.append(f"o={o}")

        # single HCP baseline at the end
        vals_h = _exp_metric_o_independent(d, "HCP", metric)
        hcp_pos = len(o_vals) * group_gap + 1.0
        if len(vals_h):
            _boxplot_style(
                ax,
                vals_h,
                hcp_pos,
                color="#111111",
                edge_color="#111111",
                alpha=0.30,
                linestyle="--",
                hatch="///",
                width=0.58,
            )
            tick_positions.append(hcp_pos)
            tick_labels.append("HCP")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=18, ha="right")
        _style_axis(ax, ylabel, f"{ylabel}: D-HCP with vs without training", ALPHA, metric)

    _add_custom_bottom_legend(fig, o_vals)

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    out = OUT_PLOTS / f"tauB_{str(tau_b).replace('.', 'p')}_4_dhcp_with_vs_without_o_0_10_20_30.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    raw_path = _find_raw_file()
    df = pd.read_csv(raw_path)

    df = df[df["alpha"] == ALPHA].copy()
    present_taus = sorted([float(x) for x in df["tau_B"].dropna().unique()])

    if not present_taus:
        raise ValueError("No tau values found for alpha=0.2 in the raw file.")

    print("tau_B values found in raw file:", present_taus)

    for tau_b in present_taus:
        plot_within_vs_without(df, tau_b=tau_b)


if __name__ == "__main__":
    main()