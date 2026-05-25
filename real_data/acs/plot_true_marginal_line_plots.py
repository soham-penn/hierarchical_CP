#!/usr/bin/env python3
"""Create ACS true marginal line plots across nominal coverage levels.

The true marginal experiment produces one Bernoulli coverage indicator per
trial. This script therefore uses the stored marginal coverage standard error
when available, and otherwise falls back to the Bernoulli standard error.

By default, it prefers the final per-replicate row-permuted result folders:
``real_data/acs/results/true_marginal_permuted_alphaXX``. If those are absent,
it falls back to the older fixed-row ``true_marginal_alphaXX`` summaries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_PATH = Path(__file__).resolve()
REAL_DATA_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = REAL_DATA_DIR.parent
RESULTS_ROOT = REAL_DATA_DIR / "acs" / "results"
PERMUTED_PREFIX = "true_marginal_permuted_alpha"
FIXED_ROW_PREFIX = "true_marginal_alpha"
PAPER_ROOT = REPO_ROOT / "paper_plots" / "acs_true_marginal"
FIG_DIR = PAPER_ROOT / "figures"
SUMMARY_DIR = PAPER_ROOT / "summaries"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


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

METHOD_COLORS = {
    "D-HCP": "#0072B2",
    "HCP": "#333333",
    "S-HCP": "#00B894",
    "Pooling": "#666666",
    "Pooled": "#999999",
}

METHOD_LABELS = {
    "Donor-HCP": "D-HCP",
    "donor-HCP-randomized": "D-HCP",
    "Donor-HCP-within": "D-HCP",
    "HCP": "HCP",
    "S-HCP": "S-HCP",
    "sample-HCP-randomized": "S-HCP",
    "S-HCP-within": "S-HCP",
    "Pooling": "Pooling",
    "Pooled-Income-Quantile": "Pooled",
}

FONT_TICK = 20
FONT_LABEL = 22
FONT_TITLE = 24
FONT_LEGEND = 18


def _alpha_from_dir(path: Path) -> float | None:
    match = re.search(r"alpha([0-9p.]+)$", path.name)
    if not match:
        return None
    return float(match.group(1).replace("p", ".")) / 100.0


def _result_dirs() -> list[Path]:
    permuted = sorted(RESULTS_ROOT.glob(f"{PERMUTED_PREFIX}*"))
    if permuted:
        return permuted
    return sorted(RESULTS_ROOT.glob(f"{FIXED_ROW_PREFIX}*"))


def _using_permuted_dirs() -> bool:
    return any(path.name.startswith("true_marginal_permuted") for path in _result_dirs())


def _summary_files() -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    for result_dir in _result_dirs():
        alpha = _alpha_from_dir(result_dir)
        summary = result_dir / "summaries" / "acs_true_marg_summary_long.csv"
        if alpha is not None and summary.exists():
            out.append((alpha, summary))
    return out


def _detailed_files() -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    for result_dir in _result_dirs():
        alpha = _alpha_from_dir(result_dir)
        if alpha is None:
            continue
        details = sorted(result_dir.glob("acs_true_marg_alpha*_detailed.csv"))
        if details:
            out.append((alpha, details[0]))
    return out


def _choose_width_column(df: pd.DataFrame) -> str:
    for col in ("width_income_median", "width_median", "median_width"):
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a median-width column. Expected one of "
        "width_income_median, width_median, median_width."
    )


def _coverage_se(row: pd.Series) -> float:
    if "coverage_se" in row and pd.notna(row["coverage_se"]):
        return float(row["coverage_se"])

    p_hat = float(row["coverage_mean"])
    n_col = "n_trials" if "n_trials" in row else "n"
    n_trials = float(row[n_col]) if n_col in row and pd.notna(row[n_col]) else np.nan
    if not np.isfinite(n_trials) or n_trials <= 0:
        return np.nan
    return float(np.sqrt(p_hat * (1.0 - p_hat) / n_trials))


def _load_from_summaries() -> pd.DataFrame:
    pieces = []
    for alpha, path in _summary_files():
        df = pd.read_csv(path)
        df["alpha"] = alpha
        df["nominal_coverage"] = 1.0 - alpha
        pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    combined = pd.concat(pieces, ignore_index=True)
    combined["coverage_se_plot"] = combined.apply(_coverage_se, axis=1)
    return combined


def _load_from_details() -> pd.DataFrame:
    pieces = []
    for alpha, path in _detailed_files():
        df = pd.read_csv(path)
        if not {"method", "o", "coverage"}.issubset(df.columns):
            continue
        width_col = "width_income" if "width_income" in df.columns else "width"
        grouped = (
            df[df["coverage"].notna()]
            .groupby(["method", "o"], as_index=False)
            .agg(
                coverage_mean=("coverage", "mean"),
                n=("coverage", "count"),
                width_income_median=(width_col, "median"),
                width_income_mean=(width_col, "mean"),
                width_income_std=(width_col, "std"),
            )
        )
        grouped["coverage_se_plot"] = np.sqrt(
            grouped["coverage_mean"] * (1.0 - grouped["coverage_mean"]) / grouped["n"]
        )
        grouped["alpha"] = alpha
        grouped["nominal_coverage"] = 1.0 - alpha
        pieces.append(grouped)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def load_true_marginal_summary() -> pd.DataFrame:
    """Load alpha-specific true marginal summaries, with raw-file fallback."""
    df = _load_from_summaries()
    if df.empty:
        df = _load_from_details()
    if df.empty:
        raise FileNotFoundError(
            f"No ACS true marginal summary or detailed files found under {RESULTS_ROOT}"
        )

    width_col = _choose_width_column(df)
    if width_col != "width_income_median":
        df["width_income_median"] = df[width_col]

    df["o"] = df["o"].astype(int)
    df["source"] = "permuted_rows" if _using_permuted_dirs() else "fixed_rows"
    df = df.sort_values(["nominal_coverage", "method", "o"]).reset_index(drop=True)
    return df


def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.8, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _currency_formatter(x: float, _pos: int) -> str:
    if not np.isfinite(x):
        return ""
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    return f"${x / 1_000:.0f}K"


def _available_o_values(df: pd.DataFrame, method: str) -> list[int]:
    return sorted(df.loc[df["method"] == method, "o"].dropna().astype(int).unique())


def load_width_trials() -> pd.DataFrame:
    """Load replicate-level width data for boxplots from detailed CSVs."""
    pieces = []
    for alpha, path in _detailed_files():
        df = pd.read_csv(path)
        if not {"method", "o", "width_income"}.issubset(df.columns):
            continue
        df = df[["method", "o", "width_income"]].copy()
        df["alpha"] = alpha
        df["nominal_coverage"] = 1.0 - alpha
        df["o"] = df["o"].astype(int)
        df["width_income"] = pd.to_numeric(df["width_income"], errors="coerce")
        df = df[np.isfinite(df["width_income"])]
        pieces.append(df)

    if not pieces:
        raise FileNotFoundError(
            "No detailed ACS true marginal files with replicate-level widths found."
        )
    return pd.concat(pieces, ignore_index=True)


def plot_dhcp_coverage_by_o(df: pd.DataFrame) -> Path:
    method = "Donor-HCP"
    d = df[df["method"] == method].copy()
    if d.empty:
        raise ValueError(f"Could not find {method} rows for the D-HCP coverage plot.")

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for o in _available_o_values(d, method):
        sub = d[d["o"] == o].sort_values("nominal_coverage")
        ax.errorbar(
            sub["nominal_coverage"],
            sub["coverage_mean"],
            yerr=sub["coverage_se_plot"],
            color=O_COLORS.get(o, "#444444"),
            marker="o",
            markersize=7,
            linewidth=2.7,
            capsize=4,
            label=f"o={o}",
        )

    min_nom = float(d["nominal_coverage"].min())
    max_nom = float(d["nominal_coverage"].max())
    ax.plot([min_nom, max_nom], [min_nom, max_nom], "--", color="#777777", linewidth=1.7)
    ax.set_xlabel("Nominal coverage, 1 - alpha", fontsize=FONT_LABEL)
    ax.set_ylabel("Empirical marginal coverage", fontsize=FONT_LABEL)
    ax.set_title("ACS true marginal coverage", fontsize=FONT_TITLE, pad=14)
    ax.set_xlim(min_nom - 0.015, max_nom + 0.015)
    y_min = min(min_nom - 0.04, float((d["coverage_mean"] - d["coverage_se_plot"]).min()) - 0.02)
    y_max = max(max_nom + 0.04, float((d["coverage_mean"] + d["coverage_se_plot"]).max()) + 0.02)
    ax.set_ylim(max(0.0, y_min), min(1.02, y_max))
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND, title="History size", title_fontsize=FONT_LEGEND)

    out = FIG_DIR / "acs_true_marginal_coverage_vs_nominal_by_o.pdf"
    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_width_boxplots_by_o(df: pd.DataFrame) -> Path:
    """Plot replicate-level width distributions by nominal coverage and o."""
    trials = load_width_trials()
    nominal_values = sorted(df["nominal_coverage"].dropna().unique())
    o_values = _available_o_values(df, "Donor-HCP")
    series = [("HCP", None)] + [("Donor-HCP", o) for o in o_values]
    offsets = np.linspace(-0.34, 0.34, len(series))

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    legend_handles = []
    legend_labels = []

    for s_idx, (method, o) in enumerate(series):
        data = []
        positions = []
        color = "#555555" if method == "HCP" else O_COLORS.get(int(o), "#444444")
        label = "HCP" if method == "HCP" else f"o={int(o)}"

        for x_idx, nominal in enumerate(nominal_values):
            if method == "HCP":
                sub = trials[
                    (trials["method"] == "HCP")
                    & (np.isclose(trials["nominal_coverage"], nominal))
                ]
            else:
                sub = trials[
                    (trials["method"] == "Donor-HCP")
                    & (trials["o"] == int(o))
                    & (np.isclose(trials["nominal_coverage"], nominal))
                ]
            vals = sub["width_income"].dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(x_idx + offsets[s_idx])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.095,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": "#333333", "linewidth": 0.9},
            capprops={"color": "#333333", "linewidth": 0.9},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("#222222")
            patch.set_alpha(0.72 if method == "Donor-HCP" else 0.35)
            if method == "HCP":
                patch.set_hatch("///")

        legend_handles.append(bp["boxes"][0])
        legend_labels.append(label)

    ax.set_xticks(range(len(nominal_values)))
    ax.set_xticklabels([f"{v:.3f}".rstrip("0").rstrip(".") for v in nominal_values])
    ax.set_xlabel("Nominal coverage, 1 - alpha", fontsize=FONT_LABEL)
    ax.set_ylabel("Interval width", fontsize=FONT_LABEL)
    ax.set_title("ACS true marginal width distributions", fontsize=FONT_TITLE, pad=14)
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    _style_axis(ax)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(False, axis="x")
    ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        fontsize=FONT_LEGEND,
        ncol=3,
        loc="upper left",
    )

    out = FIG_DIR / "acs_true_marginal_width_vs_nominal_by_o.pdf"
    out_box = FIG_DIR / "acs_true_marginal_width_boxplots_vs_nominal_by_o.pdf"
    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")
    fig.savefig(out_box, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out


def _pick_method_rows(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Use o=20 when available for history-dependent methods, else o=0."""
    method_rows = df[df["method"] == method].copy()
    if method_rows.empty:
        return method_rows

    available_o = set(method_rows["o"].astype(int))
    target_o = 20 if method in {"Donor-HCP", "S-HCP"} and 20 in available_o else 0
    return method_rows[method_rows["o"] == target_o].copy()


def plot_key_methods_coverage(df: pd.DataFrame) -> Path | None:
    method_order = ["Donor-HCP", "HCP", "S-HCP", "Pooling", "Pooled-Income-Quantile"]
    pieces = [_pick_method_rows(df, method) for method in method_order]
    d = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    if d.empty or d["method"].nunique() < 2:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for method in method_order:
        sub = d[d["method"] == method].sort_values("nominal_coverage")
        if sub.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        o_value = int(sub["o"].iloc[0])
        label = f"{label} (o={o_value})" if method in {"Donor-HCP", "S-HCP"} else label
        ax.errorbar(
            sub["nominal_coverage"],
            sub["coverage_mean"],
            yerr=sub["coverage_se_plot"],
            color=METHOD_COLORS.get(METHOD_LABELS.get(method, method), "#444444"),
            marker="o",
            markersize=7,
            linewidth=2.7,
            capsize=4,
            label=label,
        )

    min_nom = float(d["nominal_coverage"].min())
    max_nom = float(d["nominal_coverage"].max())
    ax.plot([min_nom, max_nom], [min_nom, max_nom], "--", color="#777777", linewidth=1.7)
    ax.set_xlabel("Nominal coverage, 1 - alpha", fontsize=FONT_LABEL)
    ax.set_ylabel("Empirical marginal coverage", fontsize=FONT_LABEL)
    ax.set_title("ACS true marginal coverage by method", fontsize=FONT_TITLE, pad=14)
    ax.set_xlim(min_nom - 0.015, max_nom + 0.015)
    ax.set_ylim(0.68, 1.02)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=FONT_LEGEND)

    out = FIG_DIR / "acs_true_marginal_coverage_vs_nominal_methods.pdf"
    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out


def write_combined_summary(df: pd.DataFrame) -> Path:
    out = SUMMARY_DIR / "acs_true_marginal_by_alpha_summary.csv"
    df.to_csv(out, index=False)
    return out


def main() -> None:
    print("Loading ACS true marginal summaries...")
    df = load_true_marginal_summary()
    print(
        f"  Loaded {len(df)} rows from {df['alpha'].nunique()} alpha levels: "
        f"{sorted(df['alpha'].unique())}"
    )
    print(f"  Source: {df['source'].iloc[0]}")
    print(f"  Methods: {sorted(df['method'].unique())}")

    summary = write_combined_summary(df)
    outputs: list[Path | None] = [
        plot_dhcp_coverage_by_o(df),
        plot_width_boxplots_by_o(df),
        plot_key_methods_coverage(df),
    ]

    print("\nSaved combined summary:")
    print(f"  {summary}")
    print("\nSaved plots:")
    for path in outputs:
        if path is not None:
            print(f"  {path}")


if __name__ == "__main__":
    main()
