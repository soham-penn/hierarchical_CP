#!/usr/bin/env python3
"""Create transparent paper-ready boxplots from NEW_RESULTS raw files.

This script mirrors the plotting families from run_latent_intercept_experiment.py,
but writes paper-ready transparent figures to paper_plots/figures and summaries to
paper_plots/summaries.
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
    stem = path.stem
    suffix = "_raw_results_complete"

    if stem.endswith(suffix):
        base = stem[: -len(suffix)]
    elif stem == "raw_results_complete_tauB0_tauB6":
        base = "overall"
    else:
        base = stem

    return base if base else "overall"


def _stability_source_name(path: Path) -> str:
    stem = path.stem
    suffix = "_randomization_stability_summary"

    if stem.endswith(suffix):
        base = stem[: -len(suffix)]
    else:
        base = stem

    return base if base else "overall"


def _discover_raw_files() -> List[Path]:
    files = sorted(NEW_RESULTS_DIR.glob("*raw_results_complete*.csv"))

    # Remove aggregate/overall plots.
    files = [f for f in files if _source_name(f) != "overall"]

    if not files:
        raise FileNotFoundError("No non-overall *raw_results_complete*.csv files found in NEW_RESULTS")

    return files


def _discover_stability_summary_files() -> Dict[str, Path]:
    files = sorted(NEW_RESULTS_DIR.glob("*randomization_stability_summary*.csv"))
    return {
        _stability_source_name(f): f
        for f in files
        if _stability_source_name(f) != "overall"
    }


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


def _set_dynamic_width_ylim(ax, values: Sequence[float]) -> None:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)

    if finite.size == 0:
        return

    ymax = float(np.max(finite))

    if ymax <= 0:
        upper = 1.0
    else:
        # Minimal data-respecting upper limit. No fixed cap at 17.5.
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

    if "Donor-HCP-no-within" in displayed_methods:
        # If showing both with and without, use vertical color stack for no-within too
        if "Donor-HCP-within" in displayed_methods:
            colors = ["#f4a582" for _ in o_vals_for_legend]
            handles.append(_VerticalColorHandle(colors=colors, label="D-HCP without within-training"))
        else:
            # Otherwise use a simple patch
            handles.append(
                mpatches.Patch(
                    facecolor="#f4a582",
                    edgecolor="#333333",
                    hatch="///",
                    alpha=0.60,
                    label=METHOD_LABELS["Donor-HCP-no-within"],
                )
            )

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
    # More spacing prevents the width-panel y-label from overlapping coverage.
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
    tau_b: float,
    alpha: float,
    out_file: Path,
    o_vals: List[int],
    method_specs: List[Dict],
    baseline_methods: List[str],
    title_cov: str,
    title_width: str,
    layout: str,
) -> None:
    d = df[(df["tau_B"] == tau_b) & (df["alpha"] == alpha)].copy()

    if d.empty:
        return

    fig, axes = _make_metric_figure(layout)
    displayed_methods: List[str] = []

    for ax, metric, title in [
        (axes[0], "coverage", title_cov),
        (axes[1], "width", title_width),
    ]:
        pos = 0.0
        tick_pos: List[float] = []
        tick_lab: List[str] = []
        width_values: List[float] = []

        for o in o_vals:
            group_positions: List[float] = []

            for spec in method_specs:
                method = spec["method"]
                vals = _exp_metric(d, method, int(o), metric)

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

                    if metric == "width":
                        width_values.extend(vals.tolist())

                pos += spec.get("step", 0.75)

            if group_positions:
                tick_pos.append(float(np.mean(group_positions)))
                tick_lab.append(f"o={int(o)}")

            pos += 0.2

        if baseline_methods:
            pos += 0.60

            for method in baseline_methods:
                vals = _exp_metric_o_independent(d, method, metric)

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

                    if metric == "width":
                        width_values.extend(vals.tolist())

                pos += style["step"]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=18, ha="right", fontsize=FONT_TICK)

        _apply_common_axis_style(
            ax,
            alpha,
            metric,
            title,
            width_values=width_values,
        )

    _add_bottom_legend(
        fig,
        displayed_methods=displayed_methods,
        o_vals_for_legend=o_vals,
    )

    _save_metric_figure(fig, out_file=out_file, layout=layout)


def _plot_randomization_stability(
    stability_df: pd.DataFrame,
    tau_b: float,
    alpha: float,
    out_file: Path,
) -> bool:
    required = {"tau_B", "alpha", "o", "relative_instability_mean"}

    if not required.issubset(stability_df.columns):
        missing = sorted(required.difference(stability_df.columns))
        print(f"Skipping stability plot {out_file.name}: missing columns {missing}")
        return False

    stab = stability_df[
        (pd.to_numeric(stability_df["tau_B"], errors="coerce") == tau_b)
        & (pd.to_numeric(stability_df["alpha"], errors="coerce") == alpha)
    ].copy()

    if stab.empty:
        return False

    stab["o"] = pd.to_numeric(stab["o"], errors="coerce")
    stab["relative_instability_mean"] = pd.to_numeric(
        stab["relative_instability_mean"],
        errors="coerce",
    )

    stab = stab.dropna(subset=["o", "relative_instability_mean"]).sort_values("o")

    if stab.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.plot(
        stab["o"],
        stab["relative_instability_mean"],
        marker="o",
        linewidth=2.0,
        color="#0072B2",
    )

    ax.set_xlabel("Observations in the target group", fontsize=FONT_LABEL)
    ax.set_ylabel("Relative instability: SD(upper) / mean width", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(alpha=0.25)
    ax.set_title("Randomization stability (D-HCP)", fontsize=FONT_TITLE, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return True


def _export_summary_tables(df: pd.DataFrame, out_prefix: Path) -> None:
    g = (
        df.groupby(["tau_B", "alpha", "o", "method"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            width_mean=("width", "mean"),
            width_median=("width", "median"),
            width_std=("width", "std"),
        )
    )

    cov = g.pivot_table(
        index=["tau_B", "alpha", "o"],
        columns="method",
        values="coverage_mean",
        aggfunc="mean",
    )

    wid = g.pivot_table(
        index=["tau_B", "alpha", "o"],
        columns="method",
        values="width_mean",
        aggfunc="mean",
    )

    cov.to_csv(out_prefix.with_name(out_prefix.name + "_coverage_summary.csv"))
    wid.to_csv(out_prefix.with_name(out_prefix.name + "_width_summary.csv"))
    g.to_csv(out_prefix.with_name(out_prefix.name + "_summary_long.csv"), index=False)


def _load_stability_for_source(
    source: str,
    stability_files: Dict[str, Path],
) -> Optional[pd.DataFrame]:
    if source in stability_files:
        return pd.read_csv(stability_files[source])

    if len(stability_files) == 1:
        return pd.read_csv(next(iter(stability_files.values())))

    return None


def main() -> None:
    raw_files = _discover_raw_files()
    stability_files = _discover_stability_summary_files()

    print("Found raw files:")
    for f in raw_files:
        print(" -", f.name)

    if stability_files:
        print("Found randomization-stability summaries:")
        for source, f in stability_files.items():
            print(f" - {source}: {f.name}")
    else:
        print("No randomization-stability summary files found; stability plots will be skipped.")

    for csv_path in raw_files:
        source = _source_name(csv_path)
        d = pd.read_csv(csv_path)

        required_cols = {"tau_B", "alpha", "o", "method", "coverage", "width", "experiment"}
        missing = required_cols.difference(set(d.columns))

        if missing:
            print(f"Skipping {csv_path.name}: missing columns {sorted(missing)}")
            continue

        d = d.copy()
        d["o"] = pd.to_numeric(d["o"], errors="coerce")
        d["tau_B"] = pd.to_numeric(d["tau_B"], errors="coerce")
        d["alpha"] = pd.to_numeric(d["alpha"], errors="coerce")
        d["width"] = pd.to_numeric(d["width"], errors="coerce")
        d["coverage"] = pd.to_numeric(d["coverage"], errors="coerce")
        d = d.dropna(subset=["o", "tau_B", "alpha"])

        stability_df = _load_stability_for_source(source, stability_files)

        taus = sorted(d["tau_B"].unique())
        alphas = sorted(d["alpha"].unique())

        for tau in taus:
            for alpha in alphas:
                ds = d[(d["tau_B"] == tau) & (d["alpha"] == alpha)].copy()

                if ds.empty:
                    continue

                o_vals_all = sorted(int(x) for x in ds["o"].dropna().unique())
                o_vals_upto20 = [o for o in o_vals_all if o <= 20]

                tag = f"{source}_tauB{_safe_tag(str(tau))}_alpha{_safe_tag(str(alpha))}"
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

                no_within_spec = {
                    "method": "Donor-HCP-no-within",
                    "color": "#f4a582",
                    "hatch": "///",
                    "alpha": 0.60,
                    "edge_color": "#333333",
                    "step": 0.85,
                }

                # 1. Side-by-side only here: D-HCP vs HCP, o <= 20.
                if o_vals_upto20:
                    out_file = PAPER_FIG_DIR / f"{tag}_1_dhcp_vs_hcp_upto20_transparent.pdf"

                    _plot_box_family(
                        ds,
                        float(tau),
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

                # 2. D-HCP vs HCP over all o, stacked vertically.
                out_file = PAPER_FIG_DIR / f"{tag}_2_dhcp_vs_hcp_all_o_transparent.pdf"

                _plot_box_family(
                    ds,
                    float(tau),
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

                # 3. D-HCP vs all baselines, stacked vertically.
                out_file = PAPER_FIG_DIR / f"{tag}_3_dhcp_vs_all_baselines_transparent.pdf"

                _plot_box_family(
                    ds,
                    float(tau),
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

                # 4. D-HCP vs S-HCP plus HCP, stacked vertically.
                if o_vals_upto20:
                    out_file = PAPER_FIG_DIR / f"{tag}_4_dhcp_vs_shcp_with_hcp_upto20_transparent.pdf"

                    _plot_box_family(
                        ds,
                        float(tau),
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

                # 5a. Within-group training vs no-within training, side-by-side for o <= 20.
                if o_vals_upto20:
                    out_file = PAPER_FIG_DIR / f"{tag}_5a_dhcp_with_vs_no_within_hcp_upto20_transparent.pdf"

                    _plot_box_family(
                        ds,
                        float(tau),
                        float(alpha),
                        out_file,
                        o_vals=o_vals_upto20,
                        method_specs=[dhcp_spec, no_within_spec],
                        baseline_methods=["HCP"],
                        title_cov="Coverage: D-HCP: WGT vs no-WGT",
                        title_width="Width: D-HCP: WGT vs no-WGT",
                        layout="side_by_side",
                    )

                    print(f"Saved plot: {out_file}")

                # 5b. Within-group training vs no-within training, stacked vertically.
                out_file = PAPER_FIG_DIR / f"{tag}_5b_dhcp_with_vs_no_within_hcp_all_o_transparent.pdf"

                _plot_box_family(
                    ds,
                    float(tau),
                    float(alpha),
                    out_file,
                    o_vals=o_vals_all,
                    method_specs=[dhcp_spec, no_within_spec],
                    baseline_methods=["HCP"],
                    title_cov="Coverage: D-HCP with vs without within-training",
                    title_width="Width: D-HCP with vs without within-training",
                    layout="stacked",
                )

                print(f"Saved plot: {out_file}")

                # 6. Randomization stability.
                if stability_df is not None:
                    out_file = PAPER_FIG_DIR / f"{tag}_6_randomization_stability_transparent.pdf"

                    if _plot_randomization_stability(stability_df, float(tau), float(alpha), out_file):
                        print(f"Saved plot: {out_file}")
                    else:
                        print(f"Skipped stability plot for tau_B={tau}, alpha={alpha}: no matching rows.")
                else:
                    print(f"Skipped stability plot for {source}: no matching stability summary file.")

                _export_summary_tables(ds, out_prefix)

                print(
                    f"Saved summaries: "
                    f"{out_prefix}_coverage_summary.csv, "
                    f"{out_prefix}_width_summary.csv"
                )

    print("Done. Outputs are in:", PAPER_PLOTS_DIR)


if __name__ == "__main__":
    main()