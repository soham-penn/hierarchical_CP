#!/usr/bin/env python3
"""Create paper-ready transparent boxplots for TRUE MARGINAL coverage experiments.

This script mirrors paper_plots_from_new_results.py but works with the true marginal
coverage data (true_marg_fixedN21, true_marg_poissonNmean21).

Key differences in the data:
- 1,000 experiments (not 50)
- 1 test group per experiment (not 100)
- Each experiment regenerates all data (true marginal coverage)
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
NEW_RESULTS_DIR = BASE_DIR / "NEW_RESULTS"

PAPER_PLOTS_DIR = BASE_DIR / "paper_plots"
PAPER_FIG_DIR = PAPER_PLOTS_DIR / "figures"
PAPER_SUMMARY_DIR = PAPER_PLOTS_DIR / "summaries"

PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Palette and typography
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
    "Donor-HCP-within": "D-HCP",
    "Donor-HCP-no-within": "D-HCP without within-training",
    "S-HCP-within": "S-HCP",
    "HCP": "HCP",
    "Pooling": "Pooling",
    "Subsampling": "Subsampling",
    "Repeated": "Repeated",
}

# Map column names from raw results to method names
METHOD_NAME_MAP = {
    "donor_hcp_randomized": "Donor-HCP-within",
    "donor_hcp_derandomized": "Donor-HCP-derandomized",
    "sample_hcp_randomized": "S-HCP-within",
    "sample_hcp_derandomized": "S-HCP-derandomized",
    "hcp": "HCP",
    "pool": "Pooling",
    "sub": "Subsampling",
    "rep": "Repeated",
    "stdcp": "Std-CP",
}

FONT_TICK = 24
FONT_LABEL = 24
FONT_TITLE = 26
FONT_LEGEND = 26


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


def _safe_tag(s: str) -> str:
    return s.replace(".", "p").replace("-", "m")


def _source_name(path: Path) -> str:
    """Extract source name from path."""
    stem = path.stem
    # Remove _raw_results_complete suffix
    if stem.endswith("_raw_results_complete"):
        base = stem[: -len("_raw_results_complete")]
    else:
        base = stem

    # Remove true_marg_ prefix for cleaner naming
    if base.startswith("true_marg_"):
        base = base[len("true_marg_"):]

    return base if base else "overall"


def _discover_raw_files() -> List[Path]:
    """Find true_marg raw results files."""
    files = sorted(NEW_RESULTS_DIR.glob("true_marg_*/*_raw_results_complete.csv"))

    if not files:
        raise FileNotFoundError("No true_marg_*_raw_results_complete.csv files found in NEW_RESULTS")

    return files


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
    vals = d.groupby("experiment", as_index=False)[metric].mean()[metric].to_numpy()

    if metric == "width":
        vals = vals[np.isfinite(vals)]

    return vals


def _exp_metric_o_independent(df: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    """Get metric values for a specific method (all o values)."""
    d = df[df["method"] == method]
    vals = d.groupby("experiment", as_index=False)[metric].mean()[metric].to_numpy()

    if metric == "width":
        vals = vals[np.isfinite(vals)]

    return vals


def _compute_coverage_stats(df: pd.DataFrame, method: str, o: int) -> Dict:
    """Compute coverage mean and binomial standard error for error bars."""
    d = df[(df["method"] == method) & (df["o"] == o)]
    cov_vals = d["coverage"].values
    n = len(cov_vals)

    if n == 0:
        return {"mean": np.nan, "se": np.nan, "n": 0}

    p = np.mean(cov_vals)
    # Binomial SE: sqrt(p(1-p)/n)
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0

    return {"mean": p, "se": se, "n": n}


def _compute_coverage_stats_o_independent(df: pd.DataFrame, method: str) -> Dict:
    """Compute coverage stats across all o values."""
    d = df[df["method"] == method]
    cov_vals = d["coverage"].values
    n = len(cov_vals)

    if n == 0:
        return {"mean": np.nan, "se": np.nan, "n": 0}

    p = np.mean(cov_vals)
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0

    return {"mean": p, "se": se, "n": n}


def _set_dynamic_width_ylim(ax, values: Sequence[float]) -> None:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)

    if finite.size == 0:
        return

    ymax = float(np.max(finite))

    if ymax <= 0:
        upper = 1.0
    else:
        upper = ymax + max(0.05 * ymax, 0.10)

    ax.set_ylim(0.0, upper)


def _apply_common_axis_style(
    ax,
    alpha: float,
    metric: str,
    title: str,
    width_values: Optional[Sequence[float]] = None,
) -> None:
    ax.grid(axis="y", alpha=0.25)

    ax.set_xlabel("Observations in the target group", fontsize=FONT_LABEL)
    ax.set_ylabel("Coverage" if metric == "coverage" else "Width", fontsize=FONT_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    if metric == "coverage":
        ax.axhline(1 - alpha, color="black", linewidth=1.2, linestyle="--", alpha=0.55)
        ax.set_ylim(0.55, 1.05)
    else:
        _set_dynamic_width_ylim(ax, width_values or [])

    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")


def _append_once(items: List[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _add_bottom_legend(fig, displayed_methods: List[str], o_vals_for_legend: List[int]) -> None:
    handles = []
    handler_map = {_VerticalColorHandle: _VerticalColorHandleHandler()}

    if "Donor-HCP-within" in displayed_methods:
        colors = [O_COLORS.get(int(o), "#0072B2") for o in o_vals_for_legend]
        handles.append(_VerticalColorHandle(colors=colors, label="D-HCP across o"))

    if "S-HCP-within" in displayed_methods:
        handles.append(
            mpatches.Patch(
                facecolor="#7F8C8D",
                edgecolor="#333333",
                hatch="xx",
                alpha=0.60,
                label=METHOD_LABELS["S-HCP-within"],
            )
        )

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
    if layout == "side_by_side":
        fig.tight_layout(rect=[0, 0.10, 1, 1], w_pad=1.5)
    else:
        fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.0)

    fig.savefig(out_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _baseline_style(method: str) -> Dict:
    return {
        "color": BASELINE_GRAYS[method],
        "edge_color": "#222222",
        "hatch": "//",
        "alpha_val": 0.45,
        "linestyle": "--",
        "step": 1.25,
    }


def _plot_box_family(
    df: pd.DataFrame,
    alpha: float,
    out_file: Path,
    o_vals: List[int],
    method_specs: List[Dict],
    baseline_methods: List[str],
    title_cov: str,
    title_width: str,
    layout: str,
) -> None:
    """Create plots: Coverage (error bars) + Width (boxplots)."""
    fig, axes = _make_metric_figure(layout)
    displayed_methods: List[str] = []

    for ax_idx, (ax, metric, title) in enumerate([
        (axes[0], "coverage", title_cov),
        (axes[1], "width", title_width),
    ]):
        pos = 0.0
        tick_pos: List[float] = []
        tick_lab: List[str] = []
        width_values: List[float] = []

        if metric == "coverage":
            # Use error bars for coverage
            for o in o_vals:
                for spec in method_specs:
                    method = spec["method"]
                    stats = _compute_coverage_stats(df, method, int(o))

                    if not np.isnan(stats["mean"]):
                        color = spec["color"](int(o)) if callable(spec["color"]) else spec["color"]

                        # Plot error bar
                        ax.errorbar(
                            pos, stats["mean"], yerr=stats["se"],
                            fmt='o',
                            color=color,
                            markersize=8,
                            capsize=5,
                            capthick=1.5,
                            elinewidth=1.5,
                            markeredgewidth=1.0,
                            markeredgecolor='black',
                        )

                        tick_pos.append(pos)
                        tick_lab.append(f"o={int(o)}")
                        _append_once(displayed_methods, method)

                    pos += spec.get("step", 0.75)

                pos += 0.2

            if baseline_methods:
                pos += 0.60

                for method in baseline_methods:
                    stats = _compute_coverage_stats_o_independent(df, method)

                    if not np.isnan(stats["mean"]):
                        style = _baseline_style(method)

                        # Plot error bar
                        ax.errorbar(
                            pos, stats["mean"], yerr=stats["se"],
                            fmt='s',  # Square marker for baselines
                            color=style["color"],
                            markersize=8,
                            capsize=5,
                            capthick=1.5,
                            elinewidth=1.5,
                            markeredgewidth=1.0,
                            markeredgecolor='#222222',
                        )

                        tick_pos.append(pos)
                        tick_lab.append(method)
                        _append_once(displayed_methods, method)

                    pos += style["step"]

        else:
            # Use boxplots for width
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
                        width_values.extend(vals.tolist())

                    pos += style["step"]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=18, ha="right", fontsize=FONT_TICK)

        _apply_common_axis_style(
            ax,
            alpha,
            metric,
            title,
            width_values=width_values if metric == "width" else None,
        )

    _add_bottom_legend(
        fig,
        displayed_methods=displayed_methods,
        o_vals_for_legend=o_vals,
    )

    _save_metric_figure(fig, out_file=out_file, layout=layout)


def _export_summary_tables(df: pd.DataFrame, out_prefix: Path) -> None:
    g = (
        df.groupby(["alpha", "o", "method"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            width_mean=("width", "mean"),
            width_median=("width", "median"),
            width_std=("width", "std"),
        )
    )

    cov = g.pivot_table(
        index=["alpha", "o"],
        columns="method",
        values="coverage_mean",
        aggfunc="mean",
    )

    wid = g.pivot_table(
        index=["alpha", "o"],
        columns="method",
        values="width_mean",
        aggfunc="mean",
    )

    cov.to_csv(out_prefix.with_name(out_prefix.name + "_coverage_summary.csv"))
    wid.to_csv(out_prefix.with_name(out_prefix.name + "_width_summary.csv"))
    g.to_csv(out_prefix.with_name(out_prefix.name + "_summary_long.csv"), index=False)


def reshape_raw_to_long(df_raw: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Reshape raw results from wide to long format for plotting.

    Input format (wide):
        experiment, o_observed, coverage_donor_hcp_randomized, width_donor_hcp_randomized, ...

    Output format (long):
        experiment, o, method, coverage, width, alpha
    """
    # Identify coverage and width columns
    cov_cols = [c for c in df_raw.columns if c.startswith("coverage_")]
    wid_cols = [c for c in df_raw.columns if c.startswith("width_")]

    rows = []
    for _, row in df_raw.iterrows():
        exp_id = row["experiment"]
        o_val = row["o_observed"]

        for cov_col in cov_cols:
            # Extract method name from column
            method_key = cov_col.replace("coverage_", "")
            method_name = METHOD_NAME_MAP.get(method_key, method_key)

            wid_col = f"width_{method_key}"

            if wid_col in df_raw.columns:
                rows.append({
                    "experiment": exp_id,
                    "o": o_val,
                    "method": method_name,
                    "coverage": row[cov_col],
                    "width": row[wid_col],
                    "alpha": alpha,
                })

    return pd.DataFrame(rows)


def main() -> None:
    raw_files = _discover_raw_files()

    print("Found true_marg raw files:")
    for f in raw_files:
        print(" -", f.name)

    for csv_path in raw_files:
        source = _source_name(csv_path)
        print(f"\nProcessing: {source}")

        d_raw = pd.read_csv(csv_path)

        # Infer alpha from filename or use default
        alpha = 0.1  # Default for current experiments

        # Reshape to long format
        d = reshape_raw_to_long(d_raw, alpha)

        # Convert to numeric
        d["o"] = pd.to_numeric(d["o"], errors="coerce")
        d["alpha"] = pd.to_numeric(d["alpha"], errors="coerce")
        d["width"] = pd.to_numeric(d["width"], errors="coerce")
        d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce")
        d = d.dropna(subset=["o", "alpha"])

        o_vals_all = sorted(int(x) for x in d["o"].dropna().unique())
        o_vals_upto20 = [o for o in o_vals_all if o <= 20]

        # Use true_marg prefix for outputs
        tag = f"true_marg_{source}_alpha{_safe_tag(str(alpha))}"
        out_prefix = PAPER_SUMMARY_DIR / tag

        dhcp_spec = {
            "method": "Donor-HCP-within",
            "color": lambda o: O_COLORS.get(int(o), "#0072B2"),
            "step": 0.85,
        }

        shcp_spec = {
            "method": "S-HCP-within",
            "color": "#7F8C8D",
            "hatch": "xx",
            "alpha": 0.60,
            "edge_color": "#333333",
            "step": 0.85,
        }

        # 1. Side-by-side: D-HCP vs HCP, o <= 20
        if o_vals_upto20:
            out_file = PAPER_FIG_DIR / f"{tag}_1_dhcp_vs_hcp_upto20_transparent.pdf"

            _plot_box_family(
                d,
                float(alpha),
                out_file,
                o_vals=o_vals_upto20,
                method_specs=[dhcp_spec],
                baseline_methods=["HCP"],
                title_cov="Coverage: D-HCP vs HCP",
                title_width="Width: D-HCP vs HCP",
                layout="side_by_side",
            )

            print(f"Saved plot: {out_file}")

        # 2. D-HCP vs HCP over all o, stacked vertically
        out_file = PAPER_FIG_DIR / f"{tag}_2_dhcp_vs_hcp_all_o_transparent.pdf"

        _plot_box_family(
            d,
            float(alpha),
            out_file,
            o_vals=o_vals_all,
            method_specs=[dhcp_spec],
            baseline_methods=["HCP"],
            title_cov="Coverage: D-HCP vs HCP",
            title_width="Width: D-HCP vs HCP",
            layout="stacked",
        )

        print(f"Saved plot: {out_file}")

        # 3. D-HCP vs all baselines, stacked vertically
        out_file = PAPER_FIG_DIR / f"{tag}_3_dhcp_vs_all_baselines_transparent.pdf"

        _plot_box_family(
            d,
            float(alpha),
            out_file,
            o_vals=o_vals_all,
            method_specs=[dhcp_spec],
            baseline_methods=["HCP", "Pooling", "Subsampling", "Repeated"],
            title_cov="Coverage: D-HCP vs baselines",
            title_width="Width: D-HCP vs baselines",
            layout="stacked",
        )

        print(f"Saved plot: {out_file}")

        # 4. D-HCP vs S-HCP plus HCP, stacked vertically
        if o_vals_upto20:
            out_file = PAPER_FIG_DIR / f"{tag}_4_dhcp_vs_shcp_with_hcp_upto20_transparent.pdf"

            _plot_box_family(
                d,
                float(alpha),
                out_file,
                o_vals=o_vals_upto20,
                method_specs=[dhcp_spec, shcp_spec],
                baseline_methods=["HCP"],
                title_cov="Coverage: D-HCP vs S-HCP",
                title_width="Width: D-HCP vs S-HCP",
                layout="stacked",
            )

            print(f"Saved plot: {out_file}")

        _export_summary_tables(d, out_prefix)

        print(
            f"Saved summaries: "
            f"{out_prefix}_coverage_summary.csv, "
            f"{out_prefix}_width_summary.csv"
        )

    print("\nDone. Outputs are in:", PAPER_PLOTS_DIR)


if __name__ == "__main__":
    main()
