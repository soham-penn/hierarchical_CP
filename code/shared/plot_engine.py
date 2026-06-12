#!/usr/bin/env python3
"""Create the reduced Poisson DGP plot set for the paper."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Accessible defaults: high-contrast text, colorblind-safe series styling below.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "axes.labelcolor": "#1a1a1a",
        "axes.edgecolor": "#333333",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "text.color": "#1a1a1a",
        "legend.edgecolor": "#333333",
    }
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RESULTS_ROOT = REPO_ROOT / "results_marginal" / "dgp"
PAPER_PARENT = REPO_ROOT / "plots_marginal"
PAPER_ROOT = PAPER_PARENT / "dgp_true_marginal"
FIG_DIR = PAPER_ROOT / "figures"
SUMMARY_DIR = PAPER_ROOT / "summaries"

POISSON_DATASET = "poissonNmean21"
FIXED_DATASET = "fixedN21"
O_VALUES = [0, 5, 10, 15, 20]
O_VALUES_UPTO35 = [0, 5, 10, 15, 20, 25, 30, 35]
FIXED_NOMINAL_O_VALUES = [0, 10, 20]
FIXED_ALPHA_PANEL_VALUES = [0.05, 0.10]
ALPHA_FOR_WITHIN_COMPARISON = 0.1
NOMINAL_COVERAGE_MAX = 0.90
# Full miscoverage grid used for coverage_lines_width_boxplots (matches dgp_true_marginal results).
PAPER_ALPHA_GRID = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]
PAPER_ALPHA_GRID_STR = ",".join(str(a) for a in PAPER_ALPHA_GRID)
STABILITY_ALPHAS = [0.20, 0.15, 0.10]

# Okabe–Ito colorblind-safe palette (+ wine instead of yellow for contrast on white).
O_COLORS = {
    0: "#0072B2",
    5: "#E69F00",
    10: "#009E73",
    15: "#D55E00",
    20: "#CC79A7",
    25: "#56B4E9",
    30: "#882255",
    35: "#000000",
}
O_MARKERS = {
    0: "o",
    5: "s",
    10: "^",
    15: "D",
    20: "v",
    25: "P",
    30: "X",
    35: "*",
}

METHOD_COLORS = {
    "D-HCP": "#0072B2",
    "D-HCP no within": "#CC79A7",
    "HCP": "#D55E00",
    "Std-CP": "#009E73",
}
METHOD_LINESTYLES = {
    "D-HCP": "-",
    "D-HCP no within": (0, (4, 1.5, 1.5, 1.5)),
    "HCP": "--",
    "Std-CP": "-.",
}
METHOD_MARKERS = {
    "D-HCP": "o",
    "D-HCP no within": "v",
    "HCP": "s",
    "Std-CP": "^",
}
NOMINAL_COLOR = "#1a1a1a"
INK_COLOR = "#1a1a1a"

ALPHA_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#882255",
    "#000000",
    "#4D4D4D",
]
STABILITY_COLORS = {
    0.20: "#0072B2",
    0.15: "#E69F00",
    0.10: "#009E73",
}

METHOD_KEYS = {
    "donor_hcp_randomized": "D-HCP",
    "donor_hcp_no_within": "D-HCP no within",
    "hcp": "HCP",
    "stdcp": "Std-CP",
}

BASELINE_METHOD_KEYS = {
    "donor_hcp_randomized": "D-HCP",
    "hcp": "HCP",
    "pool": "Pooling",
    "sub": "Subsampling",
    "rep": "Repeated",
}

BASELINE_COLORS = {
    "D-HCP": "#0072B2",
    "HCP": "#D55E00",
    "Pooling": "#666666",
    "Subsampling": "#999999",
    "Repeated": "#bbbbbb",
}
BASELINE_MARKERS = {
    "D-HCP": "o",
    "HCP": "s",
    "Pooling": "^",
    "Subsampling": "D",
    "Repeated": "v",
}
BASELINE_LINESTYLES = {
    "D-HCP": "-",
    "HCP": "--",
    "Pooling": "-.",
    "Subsampling": (0, (3, 1, 1, 1)),
    "Repeated": ":",
}

# Paper-plot font controls. Adjust these four values if the exported figures
# need to be scaled for a different paper layout.
FONT_TICK = 30
FONT_LABEL = 34
FONT_TITLE = 36
FONT_LEGEND = 34
FONT_GROUP_AXIS = 34
FONT_METHODS_XLABEL = 42
PLOT_BORDER_COLOR = "#666666"
PLOT_BORDER_WIDTH = 1.1
BOX_EDGE_COLOR = INK_COLOR
BOX_MEDIAN_COLOR = INK_COLOR
TICK_SLANT_DEG = 35
COVERAGE_YMARGIN_BELOW_NOMINAL = 0.10

LINEWIDTH = 3.8
MARKERSIZE = 10.0
CAPSIZE = 6
X_LABEL_NOMINAL = r"Nominal coverage, $1-\alpha$"
X_LABEL_TARGET_O = r"Target group size, $o$"
Y_LABEL_WIDTH = "Prediction Set Width"
# Width boxplot axes use this quantile instead of the raw maximum so a few very
# wide finite intervals do not compress the boxes. Set to 1.0 to show all widths.
WIDTH_AXIS_QUANTILE = 0.99
WIDTH_AXIS_PADDING = 1.04
BOX_WIDTH_NOMINAL = 0.30
BOX_WIDTH_O_AXIS = 1.9
NOMINAL_BOX_GROUP_SPACING = 2.4
SE_VISUAL_MULTIPLIER = 1.0


def _alpha_from_path(path: Path) -> float:
    match = re.search(r"_alpha([0-9p.]+)$", path.parent.name)
    if not match:
        raise ValueError(f"Cannot infer alpha from {path}")
    return float(match.group(1).replace("p", ".")) / 100.0


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.3f}".rstrip("0").rstrip(".")


def _nominal_label(nominal: float) -> str:
    return f"{nominal:.3f}".rstrip("0").rstrip(".")


def _plot_tag(alpha: float) -> str:
    return _alpha_label(alpha).replace(".", "p")


def _raw_files(dataset: str) -> list[Path]:
    """Load raw result CSVs from results_marginal/dgp."""
    search_roots = [REPO_ROOT / "results_marginal" / "dgp"]
    by_alpha: dict[float, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(f"true_marg_{dataset}_alpha*/*_raw_results_complete.csv")):
            alpha = _alpha_from_path(path)
            if alpha not in by_alpha:
                by_alpha[alpha] = path
    files = [by_alpha[a] for a in sorted(by_alpha)]
    if not files:
        raise FileNotFoundError(
            f"No alpha-grid raw files found for {dataset} under results_marginal/dgp"
        )
    return files


def reset_output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_trials(
    dataset: str,
    o_values: list[int] | None = None,
    method_keys: dict[str, str] | None = None,
) -> pd.DataFrame:
    keys = METHOD_KEYS if method_keys is None else method_keys
    pieces: list[pd.DataFrame] = []
    for path in _raw_files(dataset):
        alpha = _alpha_from_path(path)
        raw = pd.read_csv(path)
        rows = []
        for key, method in keys.items():
            cov_col = f"coverage_{key}"
            width_col = f"width_{key}"
            if cov_col not in raw.columns or width_col not in raw.columns:
                continue
            method_rows = raw[["experiment", "o_observed", cov_col, width_col]].copy()
            method_rows.columns = ["experiment", "o", "coverage", "width"]
            method_rows["method"] = method
            rows.append(method_rows)
        if not rows:
            continue
        df_alpha = pd.concat(rows, ignore_index=True)
        df_alpha["dataset"] = dataset
        df_alpha["alpha"] = alpha
        df_alpha["nominal_coverage"] = 1.0 - alpha
        pieces.append(df_alpha)

    if not pieces:
        raise ValueError("No usable Poisson trial rows were loaded.")

    df = pd.concat(pieces, ignore_index=True)
    df["o"] = df["o"].astype(int)
    if o_values is not None:
        df = df[df["o"].isin(o_values)].copy()
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    return df.sort_values(["alpha", "method", "o", "experiment"]).reset_index(drop=True)


def _summary_for_group(group: pd.DataFrame) -> pd.Series:
    cov = group["coverage"].dropna().to_numpy(dtype=float)
    width_all = group["width"].to_numpy(dtype=float)
    finite_width = group["width"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    p_hat = float(np.mean(cov)) if len(cov) else np.nan
    cov_std = float(np.std(cov, ddof=1)) if len(cov) > 1 else np.nan
    cov_se = float(np.sqrt(p_hat * (1.0 - p_hat) / len(cov))) if len(cov) else np.nan
    width_std = float(np.std(finite_width, ddof=1)) if len(finite_width) > 1 else np.nan
    width_se = width_std / np.sqrt(len(finite_width)) if len(finite_width) > 1 else np.nan
    n_width_total = int(np.sum(~pd.isna(width_all)))
    n_width_infinite = int(np.sum(np.isinf(width_all)))
    return pd.Series(
        {
            "nominal_coverage": float(group["nominal_coverage"].iloc[0]),
            "coverage_mean": p_hat,
            "coverage_std": cov_std,
            "coverage_se": cov_se,
            "coverage_n": int(len(cov)),
            "width_mean": float(np.mean(finite_width)) if len(finite_width) else np.nan,
            "width_std": width_std,
            "width_se": width_se,
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


def summarize_trials(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "alpha", "method", "o"]
    for keys, group in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(_summary_for_group(group).to_dict())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(["alpha", "method", "o"]).reset_index(drop=True)


def _dedupe_hcp(summary: pd.DataFrame) -> pd.DataFrame:
    hcp = summary[summary["method"] == "HCP"].copy()
    if hcp.empty:
        return hcp
    return hcp[hcp["o"] == O_VALUES[0]].copy()


def _coverage_ylim_lower(alpha: float) -> float:
    return max(0.0, 1.0 - float(alpha) - COVERAGE_YMARGIN_BELOW_NOMINAL)


def _coverage_ylim_lower_from_nominals(nominal_values: list[float]) -> float:
    if not nominal_values:
        return 0.0
    return max(0.0, min(float(v) for v in nominal_values) - COVERAGE_YMARGIN_BELOW_NOMINAL)


def _apply_plot_border(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PLOT_BORDER_WIDTH)
        spine.set_edgecolor(PLOT_BORDER_COLOR)


def _legend_kwargs(**overrides) -> dict:
    """Framed legend defaults for readability on slides and print."""
    kw = {
        "frameon": True,
        "fancybox": False,
        "edgecolor": "#333333",
        "facecolor": "white",
        "framealpha": 0.96,
        "fontsize": FONT_LEGEND,
    }
    kw.update(overrides)
    return kw


def _method_style(method: str) -> dict:
    return {
        "color": METHOD_COLORS[method],
        "linestyle": METHOD_LINESTYLES.get(method, "-"),
        "marker": METHOD_MARKERS.get(method, "o"),
    }


def _baseline_style(method: str) -> dict:
    return {
        "color": BASELINE_COLORS[method],
        "linestyle": BASELINE_LINESTYLES.get(method, "-"),
        "marker": BASELINE_MARKERS.get(method, "o"),
    }


def _accessible_boxprops(facecolor: str, *, hatch: str | None = None, alpha: float = 0.72) -> dict:
    props = {
        "facecolor": facecolor,
        "edgecolor": BOX_EDGE_COLOR,
        "alpha": alpha,
        "linewidth": 1.6,
    }
    if hatch is not None:
        props["hatch"] = hatch
    return props


def _accessible_medianprops() -> dict:
    return {"color": BOX_MEDIAN_COLOR, "linewidth": 2.2}


def _accessible_whiskerprops(color: str) -> dict:
    return {"color": color, "linewidth": 1.5}


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, color=INK_COLOR)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, color=INK_COLOR)
    ax.tick_params(axis="both", labelsize=FONT_TICK, colors=INK_COLOR)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.38, color="#888888")
    _apply_plot_border(ax)


def _set_nominal_ticks(ax: plt.Axes, nominal_values: list[float]) -> None:
    ax.set_xticks(nominal_values)
    ax.set_xticklabels([_nominal_label(nominal) for nominal in nominal_values], rotation=35, ha="right")


def _error_band(ax: plt.Axes, x: np.ndarray, y: np.ndarray, se: np.ndarray, color: str, alpha: float = 0.16) -> None:
    finite = np.isfinite(y) & np.isfinite(se)
    if np.any(finite):
        band = SE_VISUAL_MULTIPLIER * se[finite]
        ax.fill_between(x[finite], y[finite] - band, y[finite] + band, color=color, alpha=alpha, linewidth=0)


def _finite_width_array(series: pd.Series) -> np.ndarray:
    return series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _managed_width_upper(values: list[float] | np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if WIDTH_AXIS_QUANTILE >= 1.0:
        upper = float(np.max(finite))
    else:
        upper = float(np.quantile(finite, WIDTH_AXIS_QUANTILE))
    return upper * WIDTH_AXIS_PADDING


def _boxplot_whisker_upper(values: list[float] | np.ndarray, whis: float = 1.5) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    q1, q3 = np.quantile(finite, [0.25, 0.75])
    iqr = q3 - q1
    return float(min(np.max(finite), q3 + whis * iqr))


def _width_axis_upper_from_box_groups(
    box_groups: list[np.ndarray],
    *,
    padding: float = WIDTH_AXIS_PADDING,
) -> float | None:
    uppers = [_boxplot_whisker_upper(group) for group in box_groups if len(group)]
    uppers = [u for u in uppers if u is not None]
    if not uppers:
        return None
    return max(uppers) * padding


def _filter_alpha_grid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["nominal_coverage"] <= NOMINAL_COVERAGE_MAX + 1e-12].copy()


def plot_coverage_lines_width_boxplots(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    o_values: list[int],
    out_name: str,
    title_prefix: str,
) -> Path:
    df = _filter_alpha_grid(df)
    summary = _filter_alpha_grid(summary)
    nominal_values = sorted(summary["nominal_coverage"].dropna().unique())
    fig, axes = plt.subplots(2, 1, figsize=(26.0, 18.5))

    ax_cov, ax_width = axes
    dhcp_summary = summary[summary["method"] == "D-HCP"]
    for o in o_values:
        sub = dhcp_summary[dhcp_summary["o"] == o].sort_values("nominal_coverage")
        x = sub["nominal_coverage"].to_numpy(dtype=float)
        y = sub["coverage_mean"].to_numpy(dtype=float)
        se = sub["coverage_se"].fillna(0.0).to_numpy(dtype=float)
        _error_band(ax_cov, x, y, se, O_COLORS[o], alpha=0.12)
        ax_cov.errorbar(
            x,
            y,
            yerr=SE_VISUAL_MULTIPLIER * se,
            linestyle="-",
            marker=O_MARKERS.get(o, "o"),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            color=O_COLORS[o],
            markeredgecolor=INK_COLOR,
            markeredgewidth=0.6,
            label=f"D-HCP, o={o}",
        )

    hcp_summary = _dedupe_hcp(summary).sort_values("nominal_coverage")
    x_hcp = hcp_summary["nominal_coverage"].to_numpy(dtype=float)
    y_hcp = hcp_summary["coverage_mean"].to_numpy(dtype=float)
    se_hcp = hcp_summary["coverage_se"].fillna(0.0).to_numpy(dtype=float)
    _error_band(ax_cov, x_hcp, y_hcp, se_hcp, METHOD_COLORS["HCP"], alpha=0.12)
    ax_cov.errorbar(
        x_hcp,
        y_hcp,
        yerr=SE_VISUAL_MULTIPLIER * se_hcp,
        linestyle=METHOD_LINESTYLES["HCP"],
        marker=METHOD_MARKERS["HCP"],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        capsize=CAPSIZE,
        color=METHOD_COLORS["HCP"],
        markeredgecolor=INK_COLOR,
        markeredgewidth=0.6,
        label="HCP",
    )
    ax_cov.plot(
        nominal_values,
        nominal_values,
        color=NOMINAL_COLOR,
        linestyle=(0, (5, 2)),
        linewidth=2.4,
        label="Nominal",
        zorder=1,
    )
    _style_axis(ax_cov, X_LABEL_NOMINAL, "Empirical coverage")
    _set_nominal_ticks(ax_cov, nominal_values)
    ax_cov.set_ylim(_coverage_ylim_lower_from_nominals(nominal_values), 1.02)
    ax_cov.set_title(rf"{title_prefix}: coverage over $1-\alpha$", fontsize=FONT_TITLE, pad=12)

    positions = np.arange(len(nominal_values), dtype=float) * NOMINAL_BOX_GROUP_SPACING
    n_box_series = len(o_values) + 1
    offset_step = BOX_WIDTH_NOMINAL * 2.05
    offsets = (np.arange(n_box_series, dtype=float) - (n_box_series - 1) / 2.0) * offset_step
    width_box_groups: list[np.ndarray] = []
    for nominal_idx, nominal in enumerate(nominal_values):
        for o_idx, o in enumerate(o_values):
            vals = _finite_width_array(df[
                (df["method"] == "D-HCP")
                & np.isclose(df["nominal_coverage"], nominal)
                & (df["o"] == o)
            ]["width"])
            if len(vals):
                width_box_groups.append(vals)
                ax_width.boxplot(
                    [vals],
                    positions=[positions[nominal_idx] + offsets[o_idx]],
                    widths=BOX_WIDTH_NOMINAL,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=_accessible_medianprops(),
                    whiskerprops=_accessible_whiskerprops(INK_COLOR),
                    capprops=_accessible_whiskerprops(INK_COLOR),
                    boxprops=_accessible_boxprops(O_COLORS[o], alpha=0.82),
                )
                ax_width.scatter(
                    positions[nominal_idx] + offsets[o_idx],
                    float(np.median(vals)),
                    color=INK_COLOR,
                    marker=O_MARKERS.get(o, "D"),
                    s=34,
                    zorder=4,
                )

        hcp_vals = _finite_width_array(df[
            (df["method"] == "HCP")
            & np.isclose(df["nominal_coverage"], nominal)
            & (df["o"] == O_VALUES[0])
        ]["width"])
        if len(hcp_vals):
            width_box_groups.append(hcp_vals)
            ax_width.boxplot(
                [hcp_vals],
                positions=[positions[nominal_idx] + offsets[-1]],
                widths=BOX_WIDTH_NOMINAL,
                patch_artist=True,
                showfliers=False,
                medianprops=_accessible_medianprops(),
                whiskerprops=_accessible_whiskerprops(METHOD_COLORS["HCP"]),
                capprops=_accessible_whiskerprops(METHOD_COLORS["HCP"]),
                boxprops=_accessible_boxprops(METHOD_COLORS["HCP"], hatch="///", alpha=0.52),
            )

    ax_width.set_xticks(positions)
    ax_width.set_xticklabels([_nominal_label(nominal) for nominal in nominal_values], rotation=35, ha="right")
    ax_width.set_xlim(positions[0] + offsets[0] - 0.5, positions[-1] + offsets[-1] + 0.5)
    _style_axis(ax_width, X_LABEL_NOMINAL, Y_LABEL_WIDTH)
    width_upper = _width_axis_upper_from_box_groups(width_box_groups)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    ax_width.set_title(rf"{title_prefix}: widths over $1-\alpha$", fontsize=FONT_TITLE, pad=12)

    handles = [
        Line2D([0], [0], color=O_COLORS[o], marker=O_MARKERS.get(o, "o"), linestyle="-",
               linewidth=LINEWIDTH, label=f"D-HCP, o={o}")
        for o in o_values
    ]
    handles.append(Line2D([0], [0], color=METHOD_COLORS["HCP"], marker=METHOD_MARKERS["HCP"],
                          linestyle=METHOD_LINESTYLES["HCP"], linewidth=LINEWIDTH, label="HCP"))
    handles.append(Line2D([0], [0], color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), **_legend_kwargs())
    fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.2)
    out = FIG_DIR / out_name
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_coverage_width_line_bands(
    summary: pd.DataFrame,
    o_values: list[int],
    out_name: str,
    title_prefix: str,
) -> Path:
    summary = _filter_alpha_grid(summary)
    nominal_values = sorted(summary["nominal_coverage"].dropna().unique())
    fig, axes = plt.subplots(2, 1, figsize=(26.0, 18.0))

    for ax, metric, se_col, ylabel, title in [
        (axes[0], "coverage_mean", "coverage_se", "Empirical coverage", "Coverage with standard-error bands"),
        (axes[1], "width_mean", "width_se", "Mean interval width", "Width with standard-error bands"),
    ]:
        dhcp = summary[summary["method"] == "D-HCP"]
        for o in o_values:
            sub = dhcp[dhcp["o"] == o].sort_values("nominal_coverage")
            x = sub["nominal_coverage"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            se = sub[se_col].fillna(0.0).to_numpy(dtype=float)
            _error_band(ax, x, y, se, O_COLORS[o])
            ax.plot(x, y, marker="o", linestyle="-", linewidth=LINEWIDTH, markersize=MARKERSIZE, color=O_COLORS[o], label=f"D-HCP, o={o}")

        hcp = _dedupe_hcp(summary).sort_values("nominal_coverage")
        x = hcp["nominal_coverage"].to_numpy(dtype=float)
        y = hcp[metric].to_numpy(dtype=float)
        se = hcp[se_col].fillna(0.0).to_numpy(dtype=float)
        _error_band(ax, x, y, se, METHOD_COLORS["HCP"], alpha=0.12)
        ax.plot(x, y, marker="s", linestyle="-", linewidth=LINEWIDTH, markersize=MARKERSIZE, color=METHOD_COLORS["HCP"], label="HCP")

        if metric == "coverage_mean":
            ax.plot(nominal_values, nominal_values, color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal")
            ax.set_ylim(_coverage_ylim_lower_from_nominals(nominal_values), 1.02)
        else:
            finite_y = summary[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            width_upper = _managed_width_upper(finite_y)
            if width_upper is not None:
                ax.set_ylim(0.0, width_upper)
        _style_axis(ax, X_LABEL_NOMINAL, ylabel)
        _set_nominal_ticks(ax, nominal_values)
        ax.set_title(f"{title_prefix}: {title}", fontsize=FONT_TITLE, pad=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), **_legend_kwargs())
    fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.2)
    out = FIG_DIR / out_name
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_alpha_o_axis_panel(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    alpha: float,
    dataset_label: str,
    file_prefix: str,
    o_values: list[int] = O_VALUES,
    out_suffix: str = "",
    include_stdcp: bool = False,
) -> Path:
    """For one alpha, plot coverage over o and width boxplots over o."""
    d_alpha = df[np.isclose(df["alpha"], alpha)].copy()
    s_alpha = summary[np.isclose(summary["alpha"], alpha)].copy()
    if d_alpha.empty or s_alpha.empty:
        raise ValueError(f"No fixed-N rows found for alpha={alpha}")

    fig, axes = plt.subplots(1, 2, figsize=(24.0, 8.8))
    ax_cov, ax_width = axes

    dhcp = s_alpha[(s_alpha["method"] == "D-HCP") & (s_alpha["o"].isin(o_values))].sort_values("o")
    stdcp = s_alpha[(s_alpha["method"] == "Std-CP") & (s_alpha["o"].isin(o_values))].sort_values("o")
    hcp = s_alpha[(s_alpha["method"] == "HCP") & (s_alpha["o"] == O_VALUES[0])]
    if dhcp.empty or hcp.empty:
        raise ValueError(f"Missing D-HCP or HCP rows for fixed-N alpha={alpha}")

    _error_band(
        ax_cov,
        dhcp["o"].to_numpy(dtype=float),
        dhcp["coverage_mean"].to_numpy(dtype=float),
        dhcp["coverage_se"].fillna(0.0).to_numpy(dtype=float),
        METHOD_COLORS["D-HCP"],
        alpha=0.10,
    )
    dhcp_style = _method_style("D-HCP")
    ax_cov.errorbar(
        dhcp["o"],
        dhcp["coverage_mean"],
        yerr=SE_VISUAL_MULTIPLIER * dhcp["coverage_se"],
        color=dhcp_style["color"],
        marker=dhcp_style["marker"],
        linestyle=dhcp_style["linestyle"],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        capsize=CAPSIZE,
        markeredgecolor=INK_COLOR,
        markeredgewidth=0.6,
        label="D-HCP",
    )
    if include_stdcp and not stdcp.empty:
        stdcp_style = _method_style("Std-CP")
        _error_band(
            ax_cov,
            stdcp["o"].to_numpy(dtype=float),
            stdcp["coverage_mean"].to_numpy(dtype=float),
            stdcp["coverage_se"].fillna(0.0).to_numpy(dtype=float),
            stdcp_style["color"],
            alpha=0.10,
        )
        ax_cov.errorbar(
            stdcp["o"],
            stdcp["coverage_mean"],
            yerr=SE_VISUAL_MULTIPLIER * stdcp["coverage_se"],
            color=stdcp_style["color"],
            marker=stdcp_style["marker"],
            linestyle=stdcp_style["linestyle"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            markeredgecolor=INK_COLOR,
            markeredgewidth=0.6,
            label="Std-CP",
        )
    hcp_cov = float(hcp["coverage_mean"].iloc[0])
    hcp_cov_se = float(hcp["coverage_se"].fillna(0.0).iloc[0])
    hcp_x = np.asarray(o_values, dtype=float)
    hcp_y = np.repeat(hcp_cov, len(o_values))
    hcp_band = np.repeat(SE_VISUAL_MULTIPLIER * hcp_cov_se, len(o_values))
    hcp_style = _method_style("HCP")
    ax_cov.plot(
        hcp_x,
        hcp_y,
        color=hcp_style["color"],
        linestyle=hcp_style["linestyle"],
        linewidth=LINEWIDTH,
        label="HCP",
    )
    ax_cov.fill_between(hcp_x, hcp_y - hcp_band, hcp_y + hcp_band, color=hcp_style["color"], alpha=0.10, linewidth=0)
    ax_cov.axhline(1.0 - alpha, color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal")
    ax_cov.set_ylim(_coverage_ylim_lower(alpha), 1.02)
    ax_cov.set_xlim(min(o_values) - 0.75, max(o_values) + 0.75)
    ax_cov.set_xticks(o_values)
    _style_axis(ax_cov, X_LABEL_TARGET_O, "Empirical coverage")
    ax_cov.set_title("Coverage", fontsize=FONT_TITLE, pad=12)

    width_box_groups: list[np.ndarray] = []
    dhcp_offset = -0.75 if include_stdcp else 0.0
    stdcp_offset = 0.75
    box_width = BOX_WIDTH_O_AXIS * (0.62 if include_stdcp else 1.0)
    for o in o_values:
        vals = _finite_width_array(d_alpha[(d_alpha["method"] == "D-HCP") & (d_alpha["o"] == o)]["width"])
        if len(vals):
            width_box_groups.append(vals)
            ax_width.boxplot(
                [vals],
                positions=[o + dhcp_offset],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops=_accessible_medianprops(),
                whiskerprops=_accessible_whiskerprops(METHOD_COLORS["D-HCP"]),
                capprops=_accessible_whiskerprops(METHOD_COLORS["D-HCP"]),
                boxprops=_accessible_boxprops(METHOD_COLORS["D-HCP"], alpha=0.68),
            )
        if include_stdcp:
            vals_std = _finite_width_array(d_alpha[(d_alpha["method"] == "Std-CP") & (d_alpha["o"] == o)]["width"])
            if len(vals_std):
                width_box_groups.append(vals_std)
                ax_width.boxplot(
                    [vals_std],
                    positions=[o + stdcp_offset],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=_accessible_medianprops(),
                    whiskerprops=_accessible_whiskerprops(METHOD_COLORS["Std-CP"]),
                    capprops=_accessible_whiskerprops(METHOD_COLORS["Std-CP"]),
                    boxprops=_accessible_boxprops(METHOD_COLORS["Std-CP"], alpha=0.55),
                )

    hcp_vals = _finite_width_array(d_alpha[(d_alpha["method"] == "HCP") & (d_alpha["o"] == O_VALUES[0])]["width"])
    hcp_width_pos = max(o_values) + 6
    if len(hcp_vals):
        width_box_groups.append(hcp_vals)
        ax_width.boxplot(
            [hcp_vals],
            positions=[hcp_width_pos],
            widths=BOX_WIDTH_O_AXIS,
            patch_artist=True,
            showfliers=False,
            medianprops=_accessible_medianprops(),
            whiskerprops=_accessible_whiskerprops(METHOD_COLORS["HCP"]),
            capprops=_accessible_whiskerprops(METHOD_COLORS["HCP"]),
            boxprops=_accessible_boxprops(METHOD_COLORS["HCP"], hatch="///", alpha=0.50),
        )

    ax_width.set_xticks([*o_values, hcp_width_pos])
    ax_width.set_xticklabels([*(str(o) for o in o_values), "HCP"])
    width_upper = _width_axis_upper_from_box_groups(width_box_groups)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    _style_axis(ax_width, X_LABEL_TARGET_O, Y_LABEL_WIDTH)
    ax_width.set_title("Prediction Set Width", fontsize=FONT_TITLE, pad=12)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["D-HCP"], marker=METHOD_MARKERS["D-HCP"],
               linestyle=METHOD_LINESTYLES["D-HCP"], linewidth=LINEWIDTH, label="D-HCP"),
        Line2D([0], [0], color=METHOD_COLORS["HCP"], marker=METHOD_MARKERS["HCP"],
               linestyle=METHOD_LINESTYLES["HCP"], linewidth=LINEWIDTH, label="HCP"),
        Line2D([0], [0], color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal"),
    ]
    if include_stdcp:
        handles.insert(1, Line2D([0], [0], color=METHOD_COLORS["Std-CP"], marker=METHOD_MARKERS["Std-CP"],
                                  linestyle=METHOD_LINESTYLES["Std-CP"], linewidth=LINEWIDTH, label="Std-CP"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), **_legend_kwargs())
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    out = FIG_DIR / f"{file_prefix}_alpha{_plot_tag(alpha)}_coverage_width_by_o{out_suffix}.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


BASELINE_STATIC_METHODS = ["HCP", "Pooling", "Subsampling", "Repeated"]
DHCP_INNER_SPACING = 1.25
BASELINE_CLUSTER_GAP = 1.25
BASELINE_INNER_SPACING = 1.48
BASELINE_STATIC_BOX_WIDTH = 0.52
BASELINE_GROUP_LABEL_Y = -0.30
BASELINE_STATIC_O = 0
DHCP_BOX_WIDTH = 0.72
COVERAGE_HBAR_HALF_WIDTH = 0.32
COVERAGE_CAPSIZE = 10
COVERAGE_CAPTHICK = 3.2
X_LABEL_METHOD = "Methods"


def _baseline_tick_label(method: str) -> str:
    if method == "Repeated":
        return "Repeated\nSubsampling"
    return method


def _all_baselines_axis_layout(o_values: list[int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compact D-HCP positions by o; baselines in a spaced cluster to the right."""
    dhcp_positions = np.arange(len(o_values), dtype=float) * DHCP_INNER_SPACING
    cluster_start = float(dhcp_positions[-1]) + BASELINE_CLUSTER_GAP
    baseline_positions = cluster_start + np.arange(len(BASELINE_STATIC_METHODS), dtype=float) * BASELINE_INNER_SPACING
    tick_labels = [f"o={o}" for o in o_values] + [_baseline_tick_label(m) for m in BASELINE_STATIC_METHODS]
    return dhcp_positions, baseline_positions, tick_labels


def _dhcp_position_map(o_values: list[int]) -> dict[int, float]:
    dhcp_positions, _, _ = _all_baselines_axis_layout(o_values)
    return {int(o): float(p) for o, p in zip(o_values, dhcp_positions, strict=True)}


def _style_all_baselines_xaxis(
    ax: plt.Axes,
    o_values: list[int],
    *,
    show_group_labels: bool = True,
) -> None:
    """Slanted o=/baseline ticks; optional D-HCP group label and Methods xlabel."""
    dhcp_positions, baseline_positions, tick_labels = _all_baselines_axis_layout(o_values)
    margin_left = 0.55
    margin_right = 0.55
    ax.set_xticks(np.concatenate([dhcp_positions, baseline_positions]))
    ax.set_xlim(float(dhcp_positions[0]) - margin_left, float(baseline_positions[-1]) + margin_right)
    if show_group_labels:
        ax.set_xticklabels(tick_labels, rotation=TICK_SLANT_DEG, ha="right")
        ax.tick_params(axis="x", pad=6)
        group_trans = ax.get_xaxis_transform()
        dhcp_center = 0.5 * (float(dhcp_positions[0]) + float(dhcp_positions[-1]))
        ax.text(
            dhcp_center,
            BASELINE_GROUP_LABEL_Y,
            "D-HCP",
            transform=group_trans,
            ha="center",
            va="top",
            fontsize=FONT_GROUP_AXIS,
            rotation=0,
            clip_on=False,
        )
        ax.set_xlabel(X_LABEL_METHOD, fontsize=FONT_METHODS_XLABEL, labelpad=58)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", pad=4, labelbottom=False)
        ax.set_xlabel("")


def _plot_coverage_mean_with_thick_hbar(
    ax: plt.Axes,
    x: float | np.ndarray,
    y: float | np.ndarray,
    se: float | np.ndarray,
    color: str,
    *,
    connect: bool = False,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    """Plot coverage mean with thick horizontal bar and visible error caps."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    se_arr = np.atleast_1d(np.asarray(se, dtype=float))
    for xi, yi, sei in zip(x_arr, y_arr, se_arr, strict=True):
        ax.plot(
            [xi - COVERAGE_HBAR_HALF_WIDTH, xi + COVERAGE_HBAR_HALF_WIDTH],
            [yi, yi],
            color=color,
            linewidth=LINEWIDTH * 1.35,
            solid_capstyle="round",
            zorder=2,
        )
    _error_band(ax, x_arr, y_arr, se_arr, color, alpha=0.14)
    ax.errorbar(
        x_arr,
        y_arr,
        yerr=SE_VISUAL_MULTIPLIER * se_arr,
        color=color,
        marker=marker,
        linestyle=linestyle if connect and len(x_arr) > 1 else "none",
        linewidth=LINEWIDTH * 1.1 if connect else 0,
        markersize=MARKERSIZE,
        markerfacecolor=color,
        markeredgecolor=INK_COLOR,
        markeredgewidth=0.7,
        capsize=COVERAGE_CAPSIZE,
        capthick=COVERAGE_CAPTHICK,
        elinewidth=LINEWIDTH * 0.95,
        zorder=3,
    )


def _draw_all_baselines_coverage_ax(
    ax: plt.Axes,
    summary: pd.DataFrame,
    alpha: float,
    o_values: list[int],
) -> None:
    """Draw mean coverage with SE on an existing axes."""
    s = summary[
        np.isclose(summary["alpha"], alpha)
        & summary["o"].isin(o_values + [BASELINE_STATIC_O])
    ].copy()
    if s.empty:
        raise ValueError(f"No summary rows for alpha={alpha}")

    _, baseline_positions, _ = _all_baselines_axis_layout(o_values)
    o_to_pos = _dhcp_position_map(o_values)
    dhcp_color = BASELINE_COLORS["D-HCP"]

    dhcp = s[s["method"] == "D-HCP"].sort_values("o")
    if not dhcp.empty:
        x = np.array([o_to_pos[int(o)] for o in dhcp["o"]], dtype=float)
        y = dhcp["coverage_mean"].to_numpy(dtype=float)
        se = dhcp["coverage_se"].fillna(0.0).to_numpy(dtype=float)
        ds = _baseline_style("D-HCP")
        _plot_coverage_mean_with_thick_hbar(
            ax, x, y, se, ds["color"], connect=True, marker=ds["marker"], linestyle=ds["linestyle"],
        )

    for pos, method in zip(baseline_positions, BASELINE_STATIC_METHODS, strict=True):
        sub = s[(s["method"] == method) & (s["o"] == BASELINE_STATIC_O)]
        if sub.empty:
            continue
        y = float(sub["coverage_mean"].iloc[0])
        se = float(sub["coverage_se"].fillna(0.0).iloc[0])
        bs = _baseline_style(method)
        _plot_coverage_mean_with_thick_hbar(
            ax, pos, y, se, bs["color"], connect=False, marker=bs["marker"], linestyle=bs["linestyle"],
        )

    ax.axhline(1.0 - alpha, color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal")
    ax.set_ylim(_coverage_ylim_lower(alpha), 1.02)
    ax.set_ylabel("Empirical coverage", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.30)
    _apply_plot_border(ax)


def _draw_all_baselines_width_ax(
    ax: plt.Axes,
    df: pd.DataFrame,
    alpha: float,
    o_values: list[int],
    *,
    width_col: str = "width",
) -> None:
    """Draw width boxplots on an existing axes; returns nothing."""
    d = df[np.isclose(df["alpha"], alpha) & df["o"].isin(o_values + [BASELINE_STATIC_O])].copy()
    if d.empty:
        raise ValueError(f"No rows for alpha={alpha}")

    _, baseline_positions, _ = _all_baselines_axis_layout(o_values)
    o_to_pos = _dhcp_position_map(o_values)
    width_box_groups: list[np.ndarray] = []
    dhcp_color = BASELINE_COLORS["D-HCP"]

    for o in o_values:
        vals = _finite_width_array(d[(d["method"] == "D-HCP") & (d["o"] == o)][width_col])
        if len(vals) == 0:
            continue
        width_box_groups.append(vals)
        ax.boxplot(
            [vals],
            positions=[o_to_pos[o]],
            widths=DHCP_BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            medianprops=_accessible_medianprops(),
            whiskerprops=_accessible_whiskerprops(dhcp_color),
            capprops=_accessible_whiskerprops(dhcp_color),
            boxprops=_accessible_boxprops(dhcp_color, alpha=0.72),
        )

    for pos, method in zip(baseline_positions, BASELINE_STATIC_METHODS, strict=True):
        vals = _finite_width_array(
            d[(d["method"] == method) & (d["o"] == BASELINE_STATIC_O)][width_col]
        )
        if len(vals) == 0:
            continue
        width_box_groups.append(vals)
        color = BASELINE_COLORS[method]
        hatch = "///" if method == "HCP" else None
        ax.boxplot(
            [vals],
            positions=[pos],
            widths=BASELINE_STATIC_BOX_WIDTH,
            patch_artist=True,
            showfliers=False,
            medianprops=_accessible_medianprops(),
            whiskerprops=_accessible_whiskerprops(color),
            capprops=_accessible_whiskerprops(color),
            boxprops=_accessible_boxprops(color, hatch=hatch, alpha=0.50 if method == "HCP" else 0.62),
        )

    width_upper = _width_axis_upper_from_box_groups(width_box_groups)
    if width_upper is not None:
        ax.set_ylim(0.0, width_upper)
    ax.set_ylabel(Y_LABEL_WIDTH, fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.30)
    _apply_plot_border(ax)


def plot_all_baselines_coverage_by_o(
    summary: pd.DataFrame,
    alpha: float,
    o_values: list[int],
    out_name: str,
    title_prefix: str,
) -> Path:
    """Mean coverage with SE: D-HCP at each o; other baselines plotted once."""
    fig, ax = plt.subplots(figsize=(28.0, 10.5))
    _draw_all_baselines_coverage_ax(ax, summary, alpha, o_values)
    _style_all_baselines_xaxis(ax, o_values)
    ax.set_title(f"{title_prefix}: Coverage (all methods)", fontsize=FONT_TITLE, pad=14)
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIG_DIR / out_name
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_all_baselines_width_by_o(
    df: pd.DataFrame,
    alpha: float,
    o_values: list[int],
    out_name: str,
    title_prefix: str,
    *,
    width_col: str = "width",
    y_formatter=None,
) -> Path:
    """Width boxplots: D-HCP at each o; o-invariant baselines plotted once."""
    fig, ax = plt.subplots(figsize=(28.0, 10.5))
    _draw_all_baselines_width_ax(ax, df, alpha, o_values, width_col=width_col)
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(y_formatter)
    _style_all_baselines_xaxis(ax, o_values)
    ax.set_title(f"{title_prefix}: Prediction set width (all methods)", fontsize=FONT_TITLE, pad=14)
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIG_DIR / out_name
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_all_baselines_stacked_by_o(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    alpha: float,
    o_values: list[int],
    out_name: str,
    title_prefix: str,
    *,
    width_col: str = "width",
    y_formatter=None,
) -> Path:
    """Coverage (top) and width (bottom) all-baselines panels in one figure."""
    fig, (ax_cov, ax_width) = plt.subplots(2, 1, figsize=(28.0, 20.0), sharex=True)
    _draw_all_baselines_coverage_ax(ax_cov, summary, alpha, o_values)
    _style_all_baselines_xaxis(ax_cov, o_values, show_group_labels=False)
    ax_cov.set_title(f"{title_prefix}: Coverage (all methods)", fontsize=FONT_TITLE, pad=14)

    _draw_all_baselines_width_ax(ax_width, df, alpha, o_values, width_col=width_col)
    if y_formatter is not None:
        ax_width.yaxis.set_major_formatter(y_formatter)
    _style_all_baselines_xaxis(ax_width, o_values)
    ax_width.set_title(f"{title_prefix}: Prediction set width (all methods)", fontsize=FONT_TITLE, pad=14)

    fig.subplots_adjust(bottom=0.16, hspace=0.28)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = FIG_DIR / out_name
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_all_baselines_by_o(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    alpha: float,
    o_values: list[int],
    file_prefix: str,
    title_prefix: str,
    *,
    width_col: str = "width",
    y_formatter=None,
) -> list[Path]:
    tag = _plot_tag(alpha)
    return [
        plot_all_baselines_coverage_by_o(
            summary,
            alpha,
            o_values,
            f"{file_prefix}_alpha{tag}_all_baselines_coverage_by_o.pdf",
            title_prefix,
        ),
        plot_all_baselines_width_by_o(
            df,
            alpha,
            o_values,
            f"{file_prefix}_alpha{tag}_all_baselines_width_by_o.pdf",
            title_prefix,
            width_col=width_col,
            y_formatter=y_formatter,
        ),
        plot_all_baselines_stacked_by_o(
            df,
            summary,
            alpha,
            o_values,
            f"{file_prefix}_alpha{tag}_all_baselines_coverage_width_by_o.pdf",
            title_prefix,
            width_col=width_col,
            y_formatter=y_formatter,
        ),
    ]


def load_stability_summary(dataset: str) -> pd.DataFrame:
    candidates = [
        REPO_ROOT / "results_marginal" / "dgp" / f"true_marg_{dataset}_randomization_stability_summary.csv",
        REPO_ROOT / "results_marginal" / "dgp" / f"true_marg_{dataset}_randomization_stability_summary.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing randomization stability summary for {dataset} under results_marginal/dgp"
        )
    df = pd.read_csv(path)
    df = df[df["o"].isin(O_VALUES)].copy()
    return df.sort_values(["alpha", "o"]).reset_index(drop=True)


def write_summary_files(dataset: str, trials: pd.DataFrame, summary: pd.DataFrame, stability: pd.DataFrame | None = None) -> list[Path]:
    """Write table-ready summaries for coverage, width, and stability."""
    paths: list[Path] = []

    trial_path = SUMMARY_DIR / f"{dataset}_trials_long.csv"
    summary_path = SUMMARY_DIR / f"{dataset}_summary_by_alpha_o_method.csv"
    coverage_path = SUMMARY_DIR / f"{dataset}_coverage_table.csv"
    width_path = SUMMARY_DIR / f"{dataset}_width_table.csv"
    stability_path = SUMMARY_DIR / f"{dataset}_randomization_stability_summary.csv"

    trials.to_csv(trial_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary[
        [
            "dataset",
            "method",
            "o",
            "alpha",
            "nominal_coverage",
            "coverage_mean",
            "coverage_se",
            "coverage_std",
            "coverage_n",
        ]
    ].sort_values(["method", "o", "nominal_coverage"]).to_csv(coverage_path, index=False)
    summary[
        [
            "dataset",
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
    ].sort_values(["method", "o", "nominal_coverage"]).to_csv(width_path, index=False)
    paths.extend([trial_path, summary_path, coverage_path, width_path])
    if stability is not None:
        stability.sort_values(["alpha", "o"]).to_csv(stability_path, index=False)
        paths.append(stability_path)
    return paths


def plot_randomization_stability(stability: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    alphas = [alpha for alpha in STABILITY_ALPHAS if np.any(np.isclose(stability["alpha"], alpha))]
    for idx, alpha in enumerate(alphas):
        sub = stability[np.isclose(stability["alpha"], alpha)].sort_values("o")
        color = STABILITY_COLORS.get(round(float(alpha), 2), ALPHA_COLORS[idx % len(ALPHA_COLORS)])
        ax.plot(
            sub["o"],
            sub["relative_instability_mean"],
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            color=color,
            label=rf"$\alpha$={_alpha_label(float(alpha))}",
        )
    _style_axis(ax, X_LABEL_TARGET_O, "Relative instability: SD(upper) / mean width")
    ax.set_xticks(O_VALUES)
    ax.set_title("D-HCP randomization stability", fontsize=FONT_TITLE, pad=12)
    ax.legend(ncol=len(alphas), **_legend_kwargs())
    fig.tight_layout()
    out = FIG_DIR / "poisson_3_randomization_stability_by_alpha.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_with_vs_no_within_alpha10(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    dataset_label: str,
    file_prefix: str,
) -> Path:
    d = summary[np.isclose(summary["alpha"], ALPHA_FOR_WITHIN_COMPARISON)].copy()
    d_trials = df[np.isclose(df["alpha"], ALPHA_FOR_WITHIN_COMPARISON)].copy()
    if d.empty:
        raise ValueError(f"No rows found for alpha={ALPHA_FOR_WITHIN_COMPARISON}")

    fig, axes = plt.subplots(1, 2, figsize=(24.0, 8.8))
    fig.suptitle(
        "D-HCP WGT vs no-WGT",
        fontsize=FONT_TITLE,
        y=1.02,
    )
    methods = ["D-HCP", "D-HCP no within", "HCP"]
    labels = {
        "D-HCP": "D-HCP WGT",
        "D-HCP no within": "D-HCP no-WGT",
        "HCP": "HCP",
    }

    ax_cov, ax_width = axes
    for method in methods:
        sub = d[d["method"] == method].copy()
        ms = _method_style(method)
        if method == "HCP":
            sub = sub[sub["o"] == O_VALUES[0]].copy()
            if sub.empty:
                continue
            y_val = float(sub["coverage_mean"].iloc[0])
            se_val = float(sub["coverage_se"].fillna(0.0).iloc[0])
            x = np.asarray(O_VALUES, dtype=float)
            y = np.repeat(y_val, len(O_VALUES))
            se = np.repeat(se_val, len(O_VALUES))
        else:
            sub = sub[sub["o"].isin(O_VALUES)].sort_values("o")
            x = sub["o"].to_numpy(dtype=float)
            y = sub["coverage_mean"].to_numpy(dtype=float)
            se = sub["coverage_se"].fillna(0.0).to_numpy(dtype=float)
        _error_band(ax_cov, x, y, se, ms["color"])
        ax_cov.errorbar(
            x,
            y,
            yerr=SE_VISUAL_MULTIPLIER * se,
            color=ms["color"],
            marker=ms["marker"],
            linestyle=ms["linestyle"],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            markeredgecolor=INK_COLOR,
            markeredgewidth=0.6,
            label=labels[method],
        )

    nominal = 1.0 - ALPHA_FOR_WITHIN_COMPARISON
    ax_cov.axhline(nominal, color=NOMINAL_COLOR, linestyle=(0, (5, 2)), linewidth=2.4, label="Nominal")
    cov_rows = d[d["method"].isin(methods)]
    ymin = float((cov_rows["coverage_mean"] - 2.0 * cov_rows["coverage_se"].fillna(0.0)).min())
    ymax = float((cov_rows["coverage_mean"] + 2.0 * cov_rows["coverage_se"].fillna(0.0)).max())
    ax_cov.set_ylim(_coverage_ylim_lower(ALPHA_FOR_WITHIN_COMPARISON), min(1.02, ymax + 0.02))
    ax_cov.set_xticks(O_VALUES)
    _style_axis(ax_cov, X_LABEL_TARGET_O, "Empirical coverage")
    ax_cov.set_title("Coverage", fontsize=FONT_TITLE, pad=12)

    width_box_groups: list[np.ndarray] = []
    offsets = {"D-HCP no within": -0.85, "D-HCP": 0.85}
    for o in O_VALUES:
        for method in ["D-HCP no within", "D-HCP"]:
            vals = _finite_width_array(d_trials[(d_trials["method"] == method) & (d_trials["o"] == o)]["width"])
            if len(vals):
                width_box_groups.append(vals)
                color = METHOD_COLORS[method]
                hatch = "///" if method == "D-HCP no within" else None
                ax_width.boxplot(
                    [vals],
                    positions=[o + offsets[method]],
                    widths=1.35,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=_accessible_medianprops(),
                    whiskerprops=_accessible_whiskerprops(color),
                    capprops=_accessible_whiskerprops(color),
                    boxprops=_accessible_boxprops(color, hatch=hatch, alpha=0.68),
                )

    hcp_width_pos = max(O_VALUES) + 6
    hcp_vals = _finite_width_array(d_trials[(d_trials["method"] == "HCP") & (d_trials["o"] == O_VALUES[0])]["width"])
    if len(hcp_vals):
        width_box_groups.append(hcp_vals)
        ax_width.boxplot(
            [hcp_vals],
            positions=[hcp_width_pos],
            widths=BOX_WIDTH_O_AXIS,
            patch_artist=True,
            showfliers=False,
            medianprops=_accessible_medianprops(),
            whiskerprops=_accessible_whiskerprops(METHOD_COLORS["HCP"]),
            capprops=_accessible_whiskerprops(METHOD_COLORS["HCP"]),
            boxprops=_accessible_boxprops(METHOD_COLORS["HCP"], hatch="///", alpha=0.50),
        )
    ax_width.set_xticks([*O_VALUES, hcp_width_pos])
    ax_width.set_xticklabels([*(str(o) for o in O_VALUES), "HCP"])
    width_upper = _width_axis_upper_from_box_groups(width_box_groups)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    _style_axis(ax_width, X_LABEL_TARGET_O, Y_LABEL_WIDTH)
    ax_width.set_title("Prediction Set Width", fontsize=FONT_TITLE, pad=12)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(handles), **_legend_kwargs())
    fig.tight_layout(rect=[0, 0.15, 1, 1.0], w_pad=3.0)
    out = FIG_DIR / f"{file_prefix}_alpha{_plot_tag(ALPHA_FOR_WITHIN_COMPARISON)}_within_vs_no_within.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    reset_output_dirs()
    print("Loading true-marginal alpha-grid results...")
    all_plot_o_values = sorted(set(O_VALUES_UPTO35))
    poisson_trials = load_trials(POISSON_DATASET, all_plot_o_values)
    poisson_summary = summarize_trials(poisson_trials)
    poisson_stability = load_stability_summary(POISSON_DATASET)

    fixed_trials = load_trials(FIXED_DATASET, all_plot_o_values)
    fixed_summary = summarize_trials(fixed_trials)

    baseline_trials_poisson = load_trials(
        POISSON_DATASET, O_VALUES, method_keys=BASELINE_METHOD_KEYS
    )
    baseline_trials_fixed = load_trials(
        FIXED_DATASET, O_VALUES, method_keys=BASELINE_METHOD_KEYS
    )
    baseline_summary_poisson = summarize_trials(baseline_trials_poisson)
    baseline_summary_fixed = summarize_trials(baseline_trials_fixed)

    summary_paths: list[Path] = []
    summary_paths.extend(write_summary_files(POISSON_DATASET, poisson_trials, poisson_summary, poisson_stability))
    summary_paths.extend(write_summary_files(FIXED_DATASET, fixed_trials, fixed_summary))

    outputs = [
        plot_coverage_lines_width_boxplots(
            poisson_trials,
            poisson_summary,
            O_VALUES,
            "poisson_1_dhcp_coverage_lines_width_boxplots_by_alpha.pdf",
            "Poisson",
        ),
        plot_coverage_width_line_bands(
            poisson_summary,
            O_VALUES,
            "poisson_2_dhcp_coverage_width_line_bands_by_alpha.pdf",
            "Poisson",
        ),
        plot_randomization_stability(poisson_stability),
        plot_with_vs_no_within_alpha10(
            poisson_trials,
            poisson_summary,
            "Poisson",
            "poisson_4",
        ),
        plot_coverage_lines_width_boxplots(
            fixed_trials[fixed_trials["o"].isin(FIXED_NOMINAL_O_VALUES)].copy(),
            fixed_summary[fixed_summary["o"].isin(FIXED_NOMINAL_O_VALUES)].copy(),
            FIXED_NOMINAL_O_VALUES,
            "fixedN21_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf",
            r"Fixed $N_k=21$",
        ),
        plot_coverage_width_line_bands(
            fixed_summary[fixed_summary["o"].isin(FIXED_NOMINAL_O_VALUES)].copy(),
            FIXED_NOMINAL_O_VALUES,
            "fixedN21_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf",
            r"Fixed $N_k=21$",
        ),
        plot_with_vs_no_within_alpha10(
            fixed_trials,
            fixed_summary,
            r"Fixed $N_k=21$",
            "fixedN21_3",
        ),
    ]
    for alpha in FIXED_ALPHA_PANEL_VALUES:
        outputs.append(plot_alpha_o_axis_panel(fixed_trials, fixed_summary, alpha, r"Fixed $N_k=21$", "fixedN21"))
        outputs.append(plot_alpha_o_axis_panel(poisson_trials, poisson_summary, alpha, "Poisson", "poisson"))
        outputs.extend(
            plot_all_baselines_by_o(
                baseline_trials_fixed,
                baseline_summary_fixed,
                alpha,
                O_VALUES,
                "fixedN21",
                r"Fixed $N_k=21$",
            )
        )
        outputs.extend(
            plot_all_baselines_by_o(
                baseline_trials_poisson,
                baseline_summary_poisson,
                alpha,
                O_VALUES,
                "poisson",
                "Poisson",
            )
        )
    outputs.append(
        plot_alpha_o_axis_panel(
            fixed_trials,
            fixed_summary,
            0.10,
            r"Fixed $N_k=21$",
            "fixedN21",
            o_values=O_VALUES_UPTO35,
            out_suffix="_upto35",
        )
    )
    outputs.append(
        plot_alpha_o_axis_panel(
            fixed_trials,
            fixed_summary,
            0.10,
            r"Fixed $N_k=21$",
            "fixedN21",
            o_values=O_VALUES_UPTO35,
            out_suffix="_with_stdcp",
            include_stdcp=True,
        )
    )

    print("Saved summaries:")
    for path in summary_paths:
        print(f"  {path}")
    print("Saved plots:")
    for out in outputs:
        print(f"  {out}")


if __name__ == "__main__":
    main()
