#!/usr/bin/env python3
"""Create the reduced Poisson DGP plot set for the paper."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
RESULTS_ROOT = REPO_ROOT / "NEW_RESULTS"
if not RESULTS_ROOT.exists():
    RESULTS_ROOT = REPO_ROOT / "old_results" / "NEW_RESULTS"
PAPER_PARENT = REPO_ROOT / "paper_plots"
PAPER_ROOT = PAPER_PARENT / "dgp_true_marginal"
FIG_DIR = PAPER_ROOT / "figures"
SUMMARY_DIR = PAPER_ROOT / "summaries"

POISSON_DATASET = "poissonNmean21"
FIXED_DATASET = "fixedN21"
O_VALUES = [0, 5, 10, 15, 20]
O_VALUES_UPTO30 = [0, 5, 10, 15, 20, 25, 30]
FIXED_NOMINAL_O_VALUES = [0, 10, 20]
FIXED_ALPHA_PANEL_VALUES = [0.05, 0.10]
ALPHA_FOR_WITHIN_COMPARISON = 0.1
NOMINAL_COVERAGE_MAX = 0.90
STABILITY_ALPHAS = [0.20, 0.15, 0.10]

O_COLORS = {
    0: "#0072B2",
    5: "#00A1D5",
    10: "#00B894",
    15: "#F39C12",
    20: "#E74C3C",
    25: "#6C5CE7",
    30: "#2C3E50",
}

METHOD_COLORS = {
    "D-HCP": "#005A8C",
    "D-HCP no within": "#f4a582",
    "HCP": "#8E44AD",
    "Std-CP": "#D55E00",
}
NOMINAL_COLOR = "#111111"

ALPHA_COLORS = [
    "#0072B2",
    "#56B4E9",
    "#00A1D5",
    "#00B894",
    "#009E73",
    "#F39C12",
    "#E69F00",
    "#E74C3C",
    "#8E44AD",
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

# Paper-plot font controls. Adjust these four values if the exported figures
# need to be scaled for a different paper layout.
FONT_TICK = 30
FONT_LABEL = 34
FONT_TITLE = 36
FONT_LEGEND = 34

LINEWIDTH = 3.8
MARKERSIZE = 10.0
CAPSIZE = 6
X_LABEL_NOMINAL = r"Nominal coverage, $1-\alpha$"
X_LABEL_TARGET_O = r"Target group size, $o$"
Y_LABEL_WIDTH = "Prediction Set Width"
# Width boxplot axes use this quantile instead of the raw maximum so a few very
# wide finite intervals do not compress the boxes. Set to 1.0 to show all widths.
WIDTH_AXIS_QUANTILE = 0.99
BOX_WIDTH_NOMINAL = 0.30
BOX_WIDTH_O_AXIS = 1.9
NOMINAL_BOX_GROUP_SPACING = 2.4
SE_VISUAL_MULTIPLIER = 2.0


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
    files = sorted((RESULTS_ROOT).glob(f"true_marg_{dataset}_alpha*/*_raw_results_complete.csv"))
    if not files:
        raise FileNotFoundError(f"No alpha-grid raw files found for {dataset} under {RESULTS_ROOT}")
    return files


def reset_output_dirs() -> None:
    if PAPER_ROOT.exists():
        shutil.rmtree(PAPER_ROOT)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_trials(dataset: str, o_values: list[int] | None = None) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in _raw_files(dataset):
        alpha = _alpha_from_path(path)
        raw = pd.read_csv(path)
        rows = []
        for key, method in METHOD_KEYS.items():
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


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.30)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


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
    return upper * 1.03


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
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            color=O_COLORS[o],
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
        linestyle="-",
        marker="s",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        capsize=CAPSIZE,
        color=METHOD_COLORS["HCP"],
        label="HCP",
    )
    ax_cov.plot(nominal_values, nominal_values, color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal")
    _style_axis(ax_cov, X_LABEL_NOMINAL, "Empirical coverage")
    _set_nominal_ticks(ax_cov, nominal_values)
    y_min = min(0.80, float(summary["coverage_mean"].min()) - 0.03)
    ax_cov.set_ylim(max(0.0, y_min), 1.02)
    ax_cov.set_title(rf"{title_prefix}: coverage over $1-\alpha$", fontsize=FONT_TITLE, pad=12)

    positions = np.arange(len(nominal_values), dtype=float) * NOMINAL_BOX_GROUP_SPACING
    n_box_series = len(o_values) + 1
    offset_step = BOX_WIDTH_NOMINAL * 2.05
    offsets = (np.arange(n_box_series, dtype=float) - (n_box_series - 1) / 2.0) * offset_step
    width_values_for_axis: list[float] = []
    for nominal_idx, nominal in enumerate(nominal_values):
        for o_idx, o in enumerate(o_values):
            vals = _finite_width_array(df[
                (df["method"] == "D-HCP")
                & np.isclose(df["nominal_coverage"], nominal)
                & (df["o"] == o)
            ]["width"])
            if len(vals):
                width_values_for_axis.extend(vals.tolist())
                ax_width.boxplot(
                    [vals],
                    positions=[positions[nominal_idx] + offsets[o_idx]],
                    widths=BOX_WIDTH_NOMINAL,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#111111", "linewidth": 2.0},
                    whiskerprops={"color": "#111111", "linewidth": 1.5},
                    capprops={"color": "#111111", "linewidth": 1.5},
                    boxprops={"facecolor": O_COLORS[o], "edgecolor": "#111111", "alpha": 0.82, "linewidth": 1.8},
                )
                ax_width.scatter(
                    positions[nominal_idx] + offsets[o_idx],
                    float(np.median(vals)),
                    color="#111111",
                    marker="D",
                    s=34,
                    zorder=4,
                )

        hcp_vals = _finite_width_array(df[
            (df["method"] == "HCP")
            & np.isclose(df["nominal_coverage"], nominal)
            & (df["o"] == O_VALUES[0])
        ]["width"])
        if len(hcp_vals):
            width_values_for_axis.extend(hcp_vals.tolist())
            ax_width.boxplot(
                [hcp_vals],
                positions=[positions[nominal_idx] + offsets[-1]],
                widths=BOX_WIDTH_NOMINAL,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.6},
                whiskerprops={"color": "#111111", "linewidth": 1.2},
                capprops={"color": "#111111", "linewidth": 1.2},
                boxprops={"facecolor": METHOD_COLORS["HCP"], "edgecolor": "#111111", "alpha": 0.52, "linewidth": 1.4, "hatch": "///"},
            )

    ax_width.set_xticks(positions)
    ax_width.set_xticklabels([_nominal_label(nominal) for nominal in nominal_values], rotation=35, ha="right")
    ax_width.set_xlim(positions[0] + offsets[0] - 0.5, positions[-1] + offsets[-1] + 0.5)
    _style_axis(ax_width, X_LABEL_NOMINAL, Y_LABEL_WIDTH)
    width_upper = _managed_width_upper(width_values_for_axis)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    ax_width.set_title(rf"{title_prefix}: widths over $1-\alpha$", fontsize=FONT_TITLE, pad=12)

    handles = [Line2D([0], [0], color=O_COLORS[o], marker="o", linestyle="-", linewidth=LINEWIDTH, label=f"D-HCP, o={o}") for o in o_values]
    handles.append(Line2D([0], [0], color=METHOD_COLORS["HCP"], marker="s", linestyle="-", linewidth=LINEWIDTH, label="HCP"))
    handles.append(Line2D([0], [0], color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT_LEGEND)
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
            ax.set_ylim(max(0.0, min(0.80, float(summary["coverage_mean"].min()) - 0.03)), 1.02)
        else:
            finite_y = summary[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            width_upper = _managed_width_upper(finite_y)
            if width_upper is not None:
                ax.set_ylim(0.0, width_upper)
        _style_axis(ax, X_LABEL_NOMINAL, ylabel)
        _set_nominal_ticks(ax, nominal_values)
        ax.set_title(f"{title_prefix}: {title}", fontsize=FONT_TITLE, pad=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT_LEGEND)
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
    ax_cov.errorbar(
        dhcp["o"],
        dhcp["coverage_mean"],
        yerr=SE_VISUAL_MULTIPLIER * dhcp["coverage_se"],
        color=METHOD_COLORS["D-HCP"],
        marker="o",
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        capsize=CAPSIZE,
        label="D-HCP",
    )
    if include_stdcp and not stdcp.empty:
        _error_band(
            ax_cov,
            stdcp["o"].to_numpy(dtype=float),
            stdcp["coverage_mean"].to_numpy(dtype=float),
            stdcp["coverage_se"].fillna(0.0).to_numpy(dtype=float),
            METHOD_COLORS["Std-CP"],
            alpha=0.10,
        )
        ax_cov.errorbar(
            stdcp["o"],
            stdcp["coverage_mean"],
            yerr=SE_VISUAL_MULTIPLIER * stdcp["coverage_se"],
            color=METHOD_COLORS["Std-CP"],
            marker="^",
            linestyle="-",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            label="Std-CP",
        )
    hcp_cov = float(hcp["coverage_mean"].iloc[0])
    hcp_cov_se = float(hcp["coverage_se"].fillna(0.0).iloc[0])
    hcp_x = np.asarray(o_values, dtype=float)
    hcp_y = np.repeat(hcp_cov, len(o_values))
    hcp_band = np.repeat(SE_VISUAL_MULTIPLIER * hcp_cov_se, len(o_values))
    ax_cov.plot(hcp_x, hcp_y, color=METHOD_COLORS["HCP"], linestyle="-", linewidth=LINEWIDTH, label="HCP")
    ax_cov.fill_between(hcp_x, hcp_y - hcp_band, hcp_y + hcp_band, color=METHOD_COLORS["HCP"], alpha=0.10, linewidth=0)
    ax_cov.axhline(1.0 - alpha, color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal")
    cov_candidates = [
        float((dhcp["coverage_mean"] - 2.0 * dhcp["coverage_se"].fillna(0.0)).min()),
        hcp_cov - 2.0 * hcp_cov_se,
        1.0 - alpha,
    ]
    if include_stdcp and not stdcp.empty:
        cov_candidates.append(float((stdcp["coverage_mean"] - 2.0 * stdcp["coverage_se"].fillna(0.0)).min()))
    cov_low = min(cov_candidates)
    y_lower = 0.70 if np.isclose(alpha, 0.10) else max(0.0, cov_low - 0.02)
    ax_cov.set_ylim(y_lower, 1.02)
    ax_cov.set_xlim(min(o_values) - 0.75, max(o_values) + 0.75)
    ax_cov.set_xticks(o_values)
    _style_axis(ax_cov, X_LABEL_TARGET_O, "Empirical coverage")
    ax_cov.set_title("Coverage", fontsize=FONT_TITLE, pad=12)

    width_values_for_axis: list[float] = []
    dhcp_offset = -0.75 if include_stdcp else 0.0
    stdcp_offset = 0.75
    box_width = BOX_WIDTH_O_AXIS * (0.62 if include_stdcp else 1.0)
    for o in o_values:
        vals = _finite_width_array(d_alpha[(d_alpha["method"] == "D-HCP") & (d_alpha["o"] == o)]["width"])
        if len(vals):
            width_values_for_axis.extend(vals.tolist())
            ax_width.boxplot(
                [vals],
                positions=[o + dhcp_offset],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.4},
                whiskerprops={"color": METHOD_COLORS["D-HCP"], "linewidth": 1.1},
                capprops={"color": METHOD_COLORS["D-HCP"], "linewidth": 1.1},
                boxprops={"facecolor": METHOD_COLORS["D-HCP"], "edgecolor": METHOD_COLORS["D-HCP"], "alpha": 0.68, "linewidth": 1.0},
            )
        if include_stdcp:
            vals_std = _finite_width_array(d_alpha[(d_alpha["method"] == "Std-CP") & (d_alpha["o"] == o)]["width"])
            if len(vals_std):
                width_values_for_axis.extend(vals_std.tolist())
                ax_width.boxplot(
                    [vals_std],
                    positions=[o + stdcp_offset],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#111111", "linewidth": 1.4},
                    whiskerprops={"color": METHOD_COLORS["Std-CP"], "linewidth": 1.1},
                    capprops={"color": METHOD_COLORS["Std-CP"], "linewidth": 1.1},
                    boxprops={"facecolor": METHOD_COLORS["Std-CP"], "edgecolor": METHOD_COLORS["Std-CP"], "alpha": 0.52, "linewidth": 1.0},
                )

    hcp_vals = _finite_width_array(d_alpha[(d_alpha["method"] == "HCP") & (d_alpha["o"] == O_VALUES[0])]["width"])
    hcp_width_pos = max(o_values) + 6
    if len(hcp_vals):
        width_values_for_axis.extend(hcp_vals.tolist())
        ax_width.boxplot(
            [hcp_vals],
            positions=[hcp_width_pos],
            widths=BOX_WIDTH_O_AXIS,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": METHOD_COLORS["HCP"], "linewidth": 1.1},
            capprops={"color": METHOD_COLORS["HCP"], "linewidth": 1.1},
            boxprops={"facecolor": METHOD_COLORS["HCP"], "edgecolor": METHOD_COLORS["HCP"], "alpha": 0.45, "linewidth": 1.0, "hatch": "///"},
        )

    ax_width.set_xticks([*o_values, hcp_width_pos])
    ax_width.set_xticklabels([*(str(o) for o in o_values), "HCP"])
    width_upper = _managed_width_upper(width_values_for_axis)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    _style_axis(ax_width, X_LABEL_TARGET_O, Y_LABEL_WIDTH)
    ax_width.set_title("Prediction Set Width", fontsize=FONT_TITLE, pad=12)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["D-HCP"], marker="o", linestyle="-", linewidth=LINEWIDTH, label="D-HCP"),
        Line2D([0], [0], color=METHOD_COLORS["HCP"], linestyle="-", linewidth=LINEWIDTH, label="HCP"),
        Line2D([0], [0], color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal"),
    ]
    if include_stdcp:
        handles.insert(1, Line2D([0], [0], color=METHOD_COLORS["Std-CP"], marker="^", linestyle="-", linewidth=LINEWIDTH, label="Std-CP"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    out = FIG_DIR / f"{file_prefix}_alpha{_plot_tag(alpha)}_coverage_width_by_o{out_suffix}.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def load_stability_summary(dataset: str) -> pd.DataFrame:
    path = RESULTS_ROOT / f"true_marg_{dataset}_randomization_stability_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing randomization stability summary: {path}")
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
    ax.legend(frameon=False, fontsize=FONT_LEGEND, ncol=len(alphas))
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

    fig, axes = plt.subplots(1, 2, figsize=(28.0, 10.5))
    fig.suptitle(
        "D-HCP WGT vs no-WGT",
        fontsize=FONT_TITLE + 2,
        y=0.98,
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
        if method == "HCP":
            sub = sub[sub["o"] == O_VALUES[0]].copy()
            if sub.empty:
                continue
            y_val = float(sub["coverage_mean"].iloc[0])
            se_val = float(sub["coverage_se"].fillna(0.0).iloc[0])
            x = np.asarray(O_VALUES, dtype=float)
            y = np.repeat(y_val, len(O_VALUES))
            se = np.repeat(se_val, len(O_VALUES))
            marker = "s"
        else:
            sub = sub[sub["o"].isin(O_VALUES)].sort_values("o")
            x = sub["o"].to_numpy(dtype=float)
            y = sub["coverage_mean"].to_numpy(dtype=float)
            se = sub["coverage_se"].fillna(0.0).to_numpy(dtype=float)
            marker = "o"
        color = METHOD_COLORS[method]
        _error_band(ax_cov, x, y, se, color)
        ax_cov.errorbar(
            x,
            y,
            yerr=SE_VISUAL_MULTIPLIER * se,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            label=labels[method],
        )

    nominal = 1.0 - ALPHA_FOR_WITHIN_COMPARISON
    ax_cov.axhline(nominal, color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal")
    cov_rows = d[d["method"].isin(methods)]
    ymin = float((cov_rows["coverage_mean"] - 2.0 * cov_rows["coverage_se"].fillna(0.0)).min())
    ymax = float((cov_rows["coverage_mean"] + 2.0 * cov_rows["coverage_se"].fillna(0.0)).max())
    ax_cov.set_ylim(0.70, min(1.02, ymax + 0.02))
    ax_cov.set_xticks(O_VALUES)
    _style_axis(ax_cov, X_LABEL_TARGET_O, "Empirical coverage")
    ax_cov.set_title("Coverage", fontsize=FONT_TITLE, pad=14)

    width_values_for_axis: list[float] = []
    offsets = {"D-HCP no within": -0.85, "D-HCP": 0.85}
    for o in O_VALUES:
        for method in ["D-HCP no within", "D-HCP"]:
            vals = _finite_width_array(d_trials[(d_trials["method"] == method) & (d_trials["o"] == o)]["width"])
            if len(vals):
                width_values_for_axis.extend(vals.tolist())
                color = METHOD_COLORS[method]
                hatch = "///" if method == "D-HCP no within" else None
                ax_width.boxplot(
                    [vals],
                    positions=[o + offsets[method]],
                    widths=1.35,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#111111", "linewidth": 1.4},
                    whiskerprops={"color": color, "linewidth": 1.1},
                    capprops={"color": color, "linewidth": 1.1},
                    boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.68, "linewidth": 1.0, "hatch": hatch},
                )

    hcp_width_pos = max(O_VALUES) + 6
    hcp_vals = _finite_width_array(d_trials[(d_trials["method"] == "HCP") & (d_trials["o"] == O_VALUES[0])]["width"])
    if len(hcp_vals):
        width_values_for_axis.extend(hcp_vals.tolist())
        ax_width.boxplot(
            [hcp_vals],
            positions=[hcp_width_pos],
            widths=BOX_WIDTH_O_AXIS,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": METHOD_COLORS["HCP"], "linewidth": 1.1},
            capprops={"color": METHOD_COLORS["HCP"], "linewidth": 1.1},
            boxprops={"facecolor": METHOD_COLORS["HCP"], "edgecolor": METHOD_COLORS["HCP"], "alpha": 0.45, "linewidth": 1.0, "hatch": "///"},
        )
    ax_width.set_xticks([*O_VALUES, hcp_width_pos])
    ax_width.set_xticklabels([*(str(o) for o in O_VALUES), "HCP"])
    width_upper = _managed_width_upper(width_values_for_axis)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    _style_axis(ax_width, X_LABEL_TARGET_O, Y_LABEL_WIDTH)
    ax_width.set_title("Prediction Set Width", fontsize=FONT_TITLE, pad=14)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout(rect=[0, 0.15, 1, 0.92], w_pad=3.0)
    out = FIG_DIR / f"{file_prefix}_alpha{_plot_tag(ALPHA_FOR_WITHIN_COMPARISON)}_within_vs_no_within.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    reset_output_dirs()
    print("Loading true-marginal alpha-grid results...")
    all_plot_o_values = sorted(set(O_VALUES_UPTO30))
    poisson_trials = load_trials(POISSON_DATASET, all_plot_o_values)
    poisson_summary = summarize_trials(poisson_trials)
    poisson_stability = load_stability_summary(POISSON_DATASET)

    fixed_trials = load_trials(FIXED_DATASET, all_plot_o_values)
    fixed_summary = summarize_trials(fixed_trials)

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
    outputs.append(
        plot_alpha_o_axis_panel(
            fixed_trials,
            fixed_summary,
            0.10,
            r"Fixed $N_k=21$",
            "fixedN21",
            o_values=O_VALUES_UPTO30,
            out_suffix="_upto30",
        )
    )
    outputs.append(
        plot_alpha_o_axis_panel(
            fixed_trials,
            fixed_summary,
            0.10,
            r"Fixed $N_k=21$",
            "fixedN21",
            o_values=O_VALUES,
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
