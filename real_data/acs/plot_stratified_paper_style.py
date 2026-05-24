#!/usr/bin/env python3
"""Create paper-ready transparent boxplots for stratified ACS results.

This script mirrors the DGP paper plotting style from paper_plots_from_new_results.py
to create consistent, publication-quality figures for the stratified ACS experiment.

Input:
    real_data/acs/results/stratified/acs_stratified_raw_results.csv

Output:
    real_data/acs/NEW_PLOTS/stratified/paper_style/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
RESULTS_FILE = BASE_DIR / "acs/results/stratified/acs_stratified_raw_results.csv"
OUTPUT_DIR = BASE_DIR / "acs/NEW_PLOTS/stratified/paper_style"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Palette and typography (matching DGP paper style)
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
    "donor-HCP-randomized": "D-HCP",
    "donor-HCP-derandomized": "D-HCP (derandomized)",
    "sample-HCP-randomized": "S-HCP",
    "sample-HCP-derandomized": "S-HCP (derandomized)",
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
        flierprops=dict(marker=".", color=edge_color, markersize=3.5, alpha=0.35),
        boxprops=dict(
            facecolor=color,
            alpha=alpha_val,
            edgecolor=edge_color,
            linewidth=1.0,
            linestyle=linestyle,
        ),
    )

    if hatch:
        for patch in bp["boxes"]:
            patch.set_hatch(hatch)

    return bp


def _exp_metric(df: pd.DataFrame, method: str, o: int, metric: str) -> np.ndarray:
    """Get metric values for a specific method and o value."""
    d = df[(df["method"] == method) & (df["o"] == o)]
    vals = d.groupby("replicate", as_index=False)[metric].mean()[metric].to_numpy()

    if metric in ["width", "width_income"]:
        vals = vals[np.isfinite(vals)]

    return vals


def _exp_metric_o_independent(df: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    """Get metric values for a specific method (all o values)."""
    d = df[df["method"] == method]
    vals = d.groupby("replicate", as_index=False)[metric].mean()[metric].to_numpy()

    if metric in ["width", "width_income"]:
        vals = vals[np.isfinite(vals)]

    return vals


def _human_number_format(y, _):
    """Format numbers with human-readable suffixes (K, M, B)."""
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


def _set_dynamic_width_ylim(ax, values: Sequence[float]) -> None:
    """Set y-axis limits for width plots based on data."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)

    if finite.size == 0:
        return

    ymax = float(np.max(finite))

    if ymax <= 0:
        upper = 1.0
    else:
        # Minimal data-respecting upper limit
        upper = ymax + max(0.05 * ymax, 0.10)

    ax.set_ylim(0.0, upper)


def _apply_common_axis_style(
    ax,
    alpha: float,
    metric: str,
    title: str,
    width_values: Optional[Sequence[float]] = None,
) -> None:
    """Apply common axis styling."""
    ax.grid(axis="y", alpha=0.25)

    ax.set_xlabel("Observations in the target group", fontsize=FONT_LABEL)
    ax.set_ylabel("Coverage" if metric == "coverage" else "Width (income units)", fontsize=FONT_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    if metric == "coverage":
        ax.axhline(1 - alpha, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
        ax.set_ylim(0.55, 1.05)
    else:
        # For income width, set minimum at 50K
        if width_values:
            _set_dynamic_width_ylim(ax, width_values)
        ax.set_ylim(bottom=50000)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(_human_number_format))

    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")


def _append_once(items: List[str], item: str) -> None:
    """Append item to list if not already present."""
    if item not in items:
        items.append(item)


def _add_bottom_legend(fig, displayed_methods: List[str], o_vals_for_legend: List[int]) -> None:
    """Add bottom legend with vertical color stacks for D-HCP."""
    handles = []
    handler_map = {_VerticalColorHandle: _VerticalColorHandleHandler()}

    if "donor-HCP-randomized" in displayed_methods:
        colors = [O_COLORS.get(int(o), "#0072B2") for o in o_vals_for_legend]
        handles.append(_VerticalColorHandle(colors=colors, label="D-HCP across o"))

    for method in ["HCP", "Pooling", "Subsampling", "Repeated"]:
        if method in displayed_methods:
            handles.append(
                mpatches.Patch(
                    facecolor=BASELINE_GRAYS[method],
                    edgecolor="#222222",
                    hatch="//",
                    alpha=0.45,
                    label=METHOD_LABELS[method],
                )
            )

    if not handles:
        return

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, -0.01),
        handler_map=handler_map,
    )


def _make_metric_figure(layout: str):
    """Create figure with specified layout."""
    if layout == "side_by_side":
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    elif layout == "stacked":
        fig, axes = plt.subplots(2, 1, figsize=(13, 12))
    else:
        raise ValueError("layout must be 'side_by_side' or 'stacked'")

    fig.patch.set_alpha(0.0)

    for ax in np.ravel(axes):
        ax.set_facecolor("none")

    return fig, np.ravel(axes)


def _save_metric_figure(fig, out_file: Path, layout: str) -> None:
    """Save figure with appropriate spacing."""
    if layout == "side_by_side":
        fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=1.5)
    else:
        fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.0)

    fig.savefig(out_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _baseline_style(method: str) -> Dict:
    """Get styling parameters for baseline methods."""
    return {
        "color": BASELINE_GRAYS[method],
        "edge_color": "#222222",
        "hatch": "//",
        "alpha_val": 0.45,
        "linestyle": "--",
        "step": 1.25,
    }


# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------
def _plot_box_family(
    df: pd.DataFrame,
    out_file: Path,
    o_vals: List[int],
    method_specs: List[Dict],
    baseline_methods: List[str],
    title_cov: str,
    title_width: str,
    layout: str,
    metric_key: str = "width_income",
) -> None:
    """Create boxplot family (coverage + width) for specified methods."""
    fig, axes = _make_metric_figure(layout)
    displayed_methods: List[str] = []

    for ax, metric, title in [
        (axes[0], "coverage", title_cov),
        (axes[1], metric_key, title_width),
    ]:
        pos = 0.0
        tick_pos: List[float] = []
        tick_lab: List[str] = []
        width_values: List[float] = []

        for o in o_vals:
            group_positions: List[float] = []

            for spec in method_specs:
                method = spec["method"]
                vals = _exp_metric(df, method, int(o), metric)

                if len(vals):
                    color = spec["color"](int(o)) if callable(spec["color"]) else spec["color"]

                    _boxplot_style(
                        ax,
                        vals,
                        pos,
                        color=color,
                        edge_color=spec.get("edge_color"),
                        hatch=spec.get("hatch"),
                        alpha_val=spec.get("alpha_val", spec.get("alpha", 0.75)),
                        linestyle=spec.get("linestyle", "-"),
                    )

                    group_positions.append(pos)
                    _append_once(displayed_methods, method)

                    if metric == metric_key:
                        width_values.extend(vals.tolist())

                pos += spec.get("step", 0.75)

            if group_positions:
                tick_pos.append(float(np.mean(group_positions)))
                tick_lab.append(f"o={int(o)}")

            pos += 0.2

        if baseline_methods:
            pos += 0.60

            for method in baseline_methods:
                vals = _exp_metric_o_independent(df, method, metric)

                if len(vals):
                    style = _baseline_style(method)

                    _boxplot_style(
                        ax,
                        vals,
                        pos,
                        color=style["color"],
                        edge_color=style["edge_color"],
                        hatch=style["hatch"],
                        alpha_val=style["alpha_val"],
                        linestyle=style["linestyle"],
                    )

                    tick_pos.append(pos)
                    tick_lab.append(method)
                    _append_once(displayed_methods, method)

                    if metric == metric_key:
                        width_values.extend(vals.tolist())

                pos += style["step"]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=18, ha="right", fontsize=FONT_TICK)

        _apply_common_axis_style(
            ax,
            ALPHA,
            metric,
            title,
            width_values=width_values if metric == metric_key else None,
        )

    _add_bottom_legend(
        fig,
        displayed_methods=displayed_methods,
        o_vals_for_legend=o_vals,
    )

    _save_metric_figure(fig, out_file=out_file, layout=layout)


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main() -> None:
    """Generate all paper-quality plots for stratified ACS results."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    print(f"Loading results from: {RESULTS_FILE}")
    df = pd.read_csv(RESULTS_FILE)

    # Convert to numeric
    df["o"] = pd.to_numeric(df["o"], errors="coerce")
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["width_income"] = pd.to_numeric(df["width_income"], errors="coerce")
    df = df.dropna(subset=["o"])

    o_vals_all = sorted(int(x) for x in df["o"].dropna().unique())
    o_vals_upto20 = [o for o in o_vals_all if o <= 20]

    print(f"Found o values: {o_vals_all}")
    print(f"O values <= 20: {o_vals_upto20}")

    # Method specification for D-HCP
    dhcp_spec = {
        "method": "donor-HCP-randomized",
        "color": lambda o: O_COLORS.get(int(o), "#0072B2"),
        "step": 0.85,
    }

    # 1. Side-by-side: D-HCP vs HCP, o <= 20
    if o_vals_upto20:
        out_file = OUTPUT_DIR / "stratified_dhcp_vs_hcp_upto20_transparent.pdf"
        print(f"\nGenerating: {out_file.name}")

        _plot_box_family(
            df,
            out_file,
            o_vals=o_vals_upto20,
            method_specs=[dhcp_spec],
            baseline_methods=["HCP"],
            title_cov="Coverage: D-HCP vs HCP",
            title_width="Width: D-HCP vs HCP",
            layout="side_by_side",
        )

        print(f"Saved: {out_file}")

    # 2. Stacked: D-HCP vs HCP, all o
    out_file = OUTPUT_DIR / "stratified_dhcp_vs_hcp_all_o_transparent.pdf"
    print(f"\nGenerating: {out_file.name}")

    _plot_box_family(
        df,
        out_file,
        o_vals=o_vals_all,
        method_specs=[dhcp_spec],
        baseline_methods=["HCP"],
        title_cov="Coverage: D-HCP vs HCP",
        title_width="Width: D-HCP vs HCP",
        layout="stacked",
    )

    print(f"Saved: {out_file}")

    # 3. Stacked: D-HCP vs all baselines, all o
    out_file = OUTPUT_DIR / "stratified_dhcp_vs_all_baselines_transparent.pdf"
    print(f"\nGenerating: {out_file.name}")

    _plot_box_family(
        df,
        out_file,
        o_vals=o_vals_all,
        method_specs=[dhcp_spec],
        baseline_methods=["HCP", "Pooling", "Subsampling", "Repeated"],
        title_cov="Coverage: D-HCP vs baselines",
        title_width="Width: D-HCP vs baselines",
        layout="stacked",
    )

    print(f"Saved: {out_file}")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
