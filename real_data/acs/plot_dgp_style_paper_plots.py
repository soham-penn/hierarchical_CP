#!/usr/bin/env python3
"""Create DGP-style ACS true-marginal paper plots."""

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
from matplotlib.ticker import FuncFormatter


SCRIPT_PATH = Path(__file__).resolve()
REAL_DATA_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = REAL_DATA_DIR.parent
RESULTS_ROOT = REAL_DATA_DIR / "acs" / "results"
PAPER_ROOT = REPO_ROOT / "paper_plots" / "acs"
FIG_DIR = PAPER_ROOT / "figures"
SUMMARY_DIR = PAPER_ROOT / "summaries"

PERMUTED_PREFIX = "true_marginal_permuted_alpha"
O_VALUES = [0, 5, 10, 15, 20]
NOMINAL_O_VALUES = [0, 10, 20]
ALPHA_PANEL_VALUES = [0.05, 0.10, 0.20]
NOMINAL_COVERAGE_MAX = 0.90

O_COLORS = {
    0: "#0072B2",
    5: "#00A1D5",
    10: "#00B894",
    15: "#F39C12",
    20: "#E74C3C",
}
METHOD_COLORS = {
    "D-HCP": "#005A8C",
    "HCP": "#8E44AD",
}
NOMINAL_COLOR = "#111111"

# Paper-plot font controls.
FONT_TICK = 30
FONT_LABEL = 34
FONT_TITLE = 36
FONT_LEGEND = 30
LINEWIDTH = 3.8
MARKERSIZE = 10.0
CAPSIZE = 6
SE_VISUAL_MULTIPLIER = 2.0
BOX_WIDTH_NOMINAL = 0.30
BOX_WIDTH_O_AXIS = 1.9
NOMINAL_BOX_GROUP_SPACING = 2.4
WIDTH_AXIS_QUANTILE = 0.99

X_LABEL_NOMINAL = r"Nominal coverage, $1-\alpha$"
X_LABEL_TARGET_O = r"Target group size, $o$"
Y_LABEL_WIDTH = "Prediction Set Width"


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


def reset_output_dirs() -> None:
    if PAPER_ROOT.exists():
        shutil.rmtree(PAPER_ROOT)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def _detailed_files() -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    for result_dir in sorted(RESULTS_ROOT.glob(f"{PERMUTED_PREFIX}*")):
        alpha = _alpha_from_dir(result_dir)
        if alpha is None:
            continue
        details = sorted(result_dir.glob("acs_true_marg_alpha*_detailed.csv"))
        if details:
            out.append((alpha, details[0]))
    if not out:
        raise FileNotFoundError(f"No permuted ACS detailed files found under {RESULTS_ROOT}")
    return out


def load_trials() -> pd.DataFrame:
    pieces = []
    for alpha, path in _detailed_files():
        df = pd.read_csv(path)
        df = df[["replicate", "method", "o", "coverage", "width_income"]].copy()
        df["alpha"] = alpha
        df["nominal_coverage"] = 1.0 - alpha
        df["o"] = df["o"].astype(int)
        df["method"] = df["method"].replace({"Donor-HCP": "D-HCP"})
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
        df["width_income"] = pd.to_numeric(df["width_income"], errors="coerce")
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True).sort_values(["alpha", "method", "o", "replicate"])


def _summary_for_group(group: pd.DataFrame) -> pd.Series:
    cov = group["coverage"].dropna().to_numpy(dtype=float)
    width_all = group["width_income"].to_numpy(dtype=float)
    finite_width = group["width_income"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
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
            "coverage_se": float(np.sqrt(p_hat * (1.0 - p_hat) / len(cov))) if len(cov) else np.nan,
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


def _managed_width_upper(values: list[float] | np.ndarray, quantile: float = WIDTH_AXIS_QUANTILE) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    upper = float(np.quantile(finite, quantile)) if quantile < 1.0 else float(np.max(finite))
    return upper * 1.03


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

    dhcp_summary = summary[summary["method"] == "D-HCP"]
    for o in NOMINAL_O_VALUES:
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

    hcp = _dedupe_hcp(summary).sort_values("nominal_coverage")
    x_hcp = hcp["nominal_coverage"].to_numpy(dtype=float)
    y_hcp = hcp["coverage_mean"].to_numpy(dtype=float)
    se_hcp = hcp["coverage_se"].fillna(0.0).to_numpy(dtype=float)
    _error_band(ax_cov, x_hcp, y_hcp, se_hcp, METHOD_COLORS["HCP"], alpha=0.10)
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
    y_min = min(0.70, float(summary["coverage_mean"].min()) - 0.04)
    ax_cov.set_ylim(max(0.0, y_min), 1.02)
    ax_cov.set_title("ACS Coverage", fontsize=FONT_TITLE, pad=12)

    positions = np.arange(len(nominal_values), dtype=float) * NOMINAL_BOX_GROUP_SPACING
    n_box_series = len(NOMINAL_O_VALUES) + 1
    offsets = (np.arange(n_box_series, dtype=float) - (n_box_series - 1) / 2.0) * BOX_WIDTH_NOMINAL * 2.05
    width_values_for_axis: list[float] = []
    for nominal_idx, nominal in enumerate(nominal_values):
        for o_idx, o in enumerate(NOMINAL_O_VALUES):
            vals = _finite_width_array(
                trials[
                    (trials["method"] == "D-HCP")
                    & (trials["o"] == o)
                    & np.isclose(trials["nominal_coverage"], nominal)
                ]["width_income"]
            )
            if len(vals):
                width_values_for_axis.extend(vals.tolist())
                x_pos = positions[nominal_idx] + offsets[o_idx]
                ax_width.boxplot(
                    [vals],
                    positions=[x_pos],
                    widths=BOX_WIDTH_NOMINAL,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#111111", "linewidth": 2.0},
                    whiskerprops={"color": "#111111", "linewidth": 1.5},
                    capprops={"color": "#111111", "linewidth": 1.5},
                    boxprops={"facecolor": O_COLORS[o], "edgecolor": "#111111", "alpha": 0.82, "linewidth": 1.8},
                )
                ax_width.scatter(x_pos, float(np.median(vals)), color="#111111", marker="D", s=34, zorder=4)

        vals_hcp = _finite_width_array(
            trials[(trials["method"] == "HCP") & (trials["o"] == 0) & np.isclose(trials["nominal_coverage"], nominal)][
                "width_income"
            ]
        )
        if len(vals_hcp):
            width_values_for_axis.extend(vals_hcp.tolist())
            ax_width.boxplot(
                [vals_hcp],
                positions=[positions[nominal_idx] + offsets[-1]],
                widths=BOX_WIDTH_NOMINAL,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 2.0},
                whiskerprops={"color": "#111111", "linewidth": 1.5},
                capprops={"color": "#111111", "linewidth": 1.5},
                boxprops={"facecolor": METHOD_COLORS["HCP"], "edgecolor": "#111111", "alpha": 0.52, "linewidth": 1.8, "hatch": "///"},
            )

    ax_width.set_xticks(positions)
    ax_width.set_xticklabels([_nominal_label(nominal) for nominal in nominal_values], rotation=35, ha="right")
    ax_width.set_xlim(positions[0] + offsets[0] - 0.5, positions[-1] + offsets[-1] + 0.5)
    width_upper = _managed_width_upper(width_values_for_axis, quantile=1.0)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    ax_width.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    _style_axis(ax_width, X_LABEL_NOMINAL, Y_LABEL_WIDTH)
    ax_width.set_title("ACS Prediction Set Width", fontsize=FONT_TITLE, pad=12)

    handles = [Line2D([0], [0], color=O_COLORS[o], marker="o", linestyle="-", linewidth=LINEWIDTH, label=f"D-HCP, o={o}") for o in NOMINAL_O_VALUES]
    handles.append(Line2D([0], [0], color=METHOD_COLORS["HCP"], marker="s", linestyle="-", linewidth=LINEWIDTH, label="HCP"))
    handles.append(Line2D([0], [0], color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.2)
    out = FIG_DIR / "acs_1_dhcp_coverage_lines_width_boxplots_by_nominal_o0_10_20.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_nominal_line_bands(summary: pd.DataFrame) -> Path:
    summary = _filter_nominal(summary)
    nominal_values = sorted(summary["nominal_coverage"].dropna().unique())
    fig, axes = plt.subplots(2, 1, figsize=(26.0, 18.0))

    for ax, metric, se_col, ylabel, title in [
        (axes[0], "coverage_mean", "coverage_se", "Empirical coverage", "ACS Coverage with SE Bands"),
        (axes[1], "width_mean", "width_se", Y_LABEL_WIDTH, "ACS Prediction Set Width with SE Bands"),
    ]:
        for o in NOMINAL_O_VALUES:
            sub = summary[(summary["method"] == "D-HCP") & (summary["o"] == o)].sort_values("nominal_coverage")
            x = sub["nominal_coverage"].to_numpy(dtype=float)
            y = sub[metric].to_numpy(dtype=float)
            se = sub[se_col].fillna(0.0).to_numpy(dtype=float)
            _error_band(ax, x, y, se, O_COLORS[o], alpha=0.12)
            ax.plot(x, y, marker="o", linestyle="-", linewidth=LINEWIDTH, markersize=MARKERSIZE, color=O_COLORS[o], label=f"D-HCP, o={o}")

        hcp = _dedupe_hcp(summary).sort_values("nominal_coverage")
        x = hcp["nominal_coverage"].to_numpy(dtype=float)
        y = hcp[metric].to_numpy(dtype=float)
        se = hcp[se_col].fillna(0.0).to_numpy(dtype=float)
        _error_band(ax, x, y, se, METHOD_COLORS["HCP"], alpha=0.10)
        ax.plot(x, y, marker="s", linestyle="-", linewidth=LINEWIDTH, markersize=MARKERSIZE, color=METHOD_COLORS["HCP"], label="HCP")

        if metric == "coverage_mean":
            ax.plot(nominal_values, nominal_values, color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal")
            ax.set_ylim(max(0.0, min(0.70, float(summary["coverage_mean"].min()) - 0.04)), 1.02)
        else:
            width_upper = _managed_width_upper(summary[metric].to_numpy(dtype=float))
            if width_upper is not None:
                ax.set_ylim(0.0, width_upper)
            ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
        _style_axis(ax, X_LABEL_NOMINAL, ylabel)
        _set_nominal_ticks(ax, nominal_values)
        ax.set_title(title, fontsize=FONT_TITLE, pad=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout(rect=[0, 0.10, 1, 1], h_pad=3.2)
    out = FIG_DIR / "acs_2_dhcp_coverage_width_line_bands_by_nominal_o0_10_20.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def plot_alpha_o_axis_panel(trials: pd.DataFrame, summary: pd.DataFrame, alpha: float) -> Path:
    d_trials = trials[np.isclose(trials["alpha"], alpha)].copy()
    d_summary = summary[np.isclose(summary["alpha"], alpha)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(24.0, 8.8))
    ax_cov, ax_width = axes

    dhcp = d_summary[(d_summary["method"] == "D-HCP") & (d_summary["o"].isin(O_VALUES))].sort_values("o")
    hcp = d_summary[(d_summary["method"] == "HCP") & (d_summary["o"] == 0)]
    _error_band(ax_cov, dhcp["o"].to_numpy(dtype=float), dhcp["coverage_mean"].to_numpy(dtype=float), dhcp["coverage_se"].fillna(0.0).to_numpy(dtype=float), METHOD_COLORS["D-HCP"], alpha=0.10)
    ax_cov.errorbar(
        dhcp["o"],
        dhcp["coverage_mean"],
        yerr=SE_VISUAL_MULTIPLIER * dhcp["coverage_se"].fillna(0.0),
        color=METHOD_COLORS["D-HCP"],
        marker="o",
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        capsize=CAPSIZE,
        label="D-HCP",
    )
    if not hcp.empty:
        hcp_cov = float(hcp["coverage_mean"].iloc[0])
        hcp_cov_se = float(hcp["coverage_se"].fillna(0.0).iloc[0])
        x_hcp = np.asarray(O_VALUES, dtype=float)
        y_hcp = np.repeat(hcp_cov, len(O_VALUES))
        band = np.repeat(SE_VISUAL_MULTIPLIER * hcp_cov_se, len(O_VALUES))
        ax_cov.plot(x_hcp, y_hcp, color=METHOD_COLORS["HCP"], linestyle="-", linewidth=LINEWIDTH, label="HCP")
        ax_cov.fill_between(x_hcp, y_hcp - band, y_hcp + band, color=METHOD_COLORS["HCP"], alpha=0.10, linewidth=0)
    ax_cov.axhline(1.0 - alpha, color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal")
    ax_cov.set_xlim(min(O_VALUES) - 0.75, max(O_VALUES) + 0.75)
    ax_cov.set_ylim(0.70, 1.02)
    ax_cov.set_xticks(O_VALUES)
    _style_axis(ax_cov, X_LABEL_TARGET_O, "Empirical coverage")
    ax_cov.set_title("Coverage", fontsize=FONT_TITLE, pad=12)

    width_values_for_axis: list[float] = []
    for o in O_VALUES:
        vals = _finite_width_array(d_trials[(d_trials["method"] == "D-HCP") & (d_trials["o"] == o)]["width_income"])
        if len(vals):
            width_values_for_axis.extend(vals.tolist())
            ax_width.boxplot(
                [vals],
                positions=[o],
                widths=BOX_WIDTH_O_AXIS,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                medianprops={"color": "#111111", "linewidth": 1.4},
                whiskerprops={"color": METHOD_COLORS["D-HCP"], "linewidth": 1.1},
                capprops={"color": METHOD_COLORS["D-HCP"], "linewidth": 1.1},
                boxprops={"facecolor": METHOD_COLORS["D-HCP"], "edgecolor": METHOD_COLORS["D-HCP"], "alpha": 0.68, "linewidth": 1.0},
            )
    hcp_pos = max(O_VALUES) + 6
    vals_hcp = _finite_width_array(d_trials[(d_trials["method"] == "HCP") & (d_trials["o"] == 0)]["width_income"])
    if len(vals_hcp):
        width_values_for_axis.extend(vals_hcp.tolist())
        ax_width.boxplot(
            [vals_hcp],
            positions=[hcp_pos],
            widths=BOX_WIDTH_O_AXIS,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": METHOD_COLORS["HCP"], "linewidth": 1.1},
            capprops={"color": METHOD_COLORS["HCP"], "linewidth": 1.1},
            boxprops={"facecolor": METHOD_COLORS["HCP"], "edgecolor": METHOD_COLORS["HCP"], "alpha": 0.45, "linewidth": 1.0, "hatch": "///"},
        )
    ax_width.set_xticks([*O_VALUES, hcp_pos])
    ax_width.set_xticklabels([*(str(o) for o in O_VALUES), "HCP"])
    width_upper = _managed_width_upper(width_values_for_axis, quantile=1.0)
    if width_upper is not None:
        ax_width.set_ylim(0.0, width_upper)
    ax_width.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    _style_axis(ax_width, X_LABEL_TARGET_O, Y_LABEL_WIDTH)
    ax_width.set_title(Y_LABEL_WIDTH, fontsize=FONT_TITLE, pad=12)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["D-HCP"], marker="o", linestyle="-", linewidth=LINEWIDTH, label="D-HCP"),
        Line2D([0], [0], color=METHOD_COLORS["HCP"], linestyle="-", linewidth=LINEWIDTH, label="HCP"),
        Line2D([0], [0], color=NOMINAL_COLOR, linestyle="--", linewidth=2.0, label="Nominal"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout(rect=[0, 0.15, 1, 1], w_pad=3.0)
    out = FIG_DIR / f"acs_alpha{_plot_tag(alpha)}_coverage_width_by_o.pdf"
    fig.savefig(out, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def write_summaries(trials: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    paths = [
        SUMMARY_DIR / "acs_trials_long.csv",
        SUMMARY_DIR / "acs_summary_by_alpha_o_method.csv",
        SUMMARY_DIR / "acs_coverage_table.csv",
        SUMMARY_DIR / "acs_width_table.csv",
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


def main() -> None:
    reset_output_dirs()
    print("Loading ACS true marginal detailed results...")
    trials = load_trials()
    summary = summarize_trials(trials)
    summary_paths = write_summaries(trials, summary)
    outputs = [
        plot_nominal_coverage_width(trials, summary),
        plot_nominal_line_bands(summary),
    ]
    for alpha in ALPHA_PANEL_VALUES:
        outputs.append(plot_alpha_o_axis_panel(trials, summary, alpha))

    print("Saved summaries:")
    for path in summary_paths:
        print(f"  {path}")
    print("Saved plots:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
