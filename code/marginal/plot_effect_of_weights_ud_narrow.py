#!/usr/bin/env python3
"""2×2 coverage/width panels for U_d-narrow γ=0 (RF + Bayes over o)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.marginal.run_effect_of_weights_ud_narrow import O_VALUES, RESULTS_PREFIX
from code.marginal.run_true_marginal_latent_intercept_experiments import (
    alpha_to_tag,
    gamma_tag,
)
from code.paths import PLOTS_MARGINAL, RESULTS_DGP_MARGINAL

_pm_path = REPO_ROOT / "code" / "shared" / "plot_engine.py"
_spec = importlib.util.spec_from_file_location("plot_engine", _pm_path)
pm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(pm)

OUT_DIR = PLOTS_MARGINAL / "effect_of_weights" / "ud_narrow_gamma0"
FIGSIZE = (18.0, 14.0)
FONTS = pm._stacked_panel_fonts()
# Axis titles: smaller than stacked DGP figures (46) so they fit the 2×2.
AXIS_LABEL_SIZE = 40
# ud_narrow only: 2×2 over λ_local ∈ {1/7, 3/7, 4/7, 6/7}.
LAMBDA_LOCAL_PANELS = (1 / 7, 3 / 7, 4 / 7, 6 / 7)
PREDICTORS = ("rf", "bayes")
PRED_COLORS = {
    "rf": pm.METHOD_COLORS["GHCP"],
    "bayes": "#C45C26",
}
PRED_MARKERS = {"rf": "o", "bayes": "s"}
PRED_LINESTYLES = {"rf": "-", "bayes": "--"}
PRED_LABELS = {"rf": "RF", "bayes": r"Bayes $E[Y\mid X,U]$"}
# Twin boxplot offset around each o.
BOX_HALF_OFFSET = 1.05
BOX_WIDTH = 1.7


def _load(alpha: float, gamma: float) -> pd.DataFrame:
    tag = f"{RESULTS_PREFIX}_{gamma_tag(gamma)}_poissonNmean25_{alpha_to_tag(alpha)}"
    path = RESULTS_DGP_MARGINAL / tag / f"{tag}_raw_results_complete.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    return pd.read_csv(path)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pred, w_g, o), g in df.groupby(["predictor", "w_g", "o_observed"]):
        cov = g["coverage"].to_numpy(dtype=float)
        width = g["width"].to_numpy(dtype=float)
        finite = width[np.isfinite(width)]
        n = len(g)
        p = float(np.mean(cov))
        rows.append(
            {
                "predictor": str(pred),
                "w_g": float(w_g),
                "w_local": float(1.0 - w_g),
                "o": int(o),
                "n": n,
                "coverage_mean": p,
                "coverage_se": float(np.sqrt(p * (1.0 - p) / n)) if n else np.nan,
                "width_mean": float(np.mean(finite)) if len(finite) else np.nan,
                "width_se": (
                    float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                    if len(finite) > 1
                    else np.nan
                ),
                "finite_rate": float(len(finite) / n) if n else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["predictor", "w_local", "o"]).reset_index(drop=True)


def _panel_lambdas(summary: pd.DataFrame) -> list[float]:
    avail = sorted(summary["w_local"].unique())
    chosen = []
    for lam in LAMBDA_LOCAL_PANELS:
        match = [a for a in avail if np.isclose(a, lam)]
        if not match:
            raise ValueError(f"missing λ_local={lam} in results; have {avail}")
        chosen.append(float(match[0]))
    return chosen


def _panel_title(lam: float) -> str:
    for k in range(1, 7):
        if np.isclose(lam, k / 7):
            return rf"$\lambda_{{\mathrm{{local}}}}={k}/7$"
    return rf"$\lambda_{{\mathrm{{local}}}}={lam:.3f}$"


def _new_grid():
    fig, axes = plt.subplots(
        2, 2, figsize=FIGSIZE, sharex=True, sharey=True, constrained_layout=True
    )
    return fig, np.atleast_2d(axes)


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), transparent=True, bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _legend_handles():
    return [
        plt.Line2D(
            [0],
            [0],
            color=PRED_COLORS[p],
            marker=PRED_MARKERS[p],
            linestyle=PRED_LINESTYLES[p],
            linewidth=2.4,
            markersize=9,
            label=PRED_LABELS[p],
        )
        for p in PREDICTORS
    ]


def _set_o_ticks(ax) -> None:
    ax.set_xticks(O_VALUES)
    ax.set_xticklabels([str(o) for o in O_VALUES])


def _add_fig_legend(fig) -> None:
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=FONTS["tick"],
    )


def _plot_coverage(summary: pd.DataFrame, *, alpha: float, out_stem: Path) -> None:
    lambs = _panel_lambdas(summary)
    fig, axes = _new_grid()
    ylabel = "Empirical coverage"
    for i, lam in enumerate(lambs):
        ax = axes[i // 2, i % 2]
        for pred in PREDICTORS:
            g = summary[
                (summary["predictor"] == pred) & np.isclose(summary["w_local"], lam)
            ].sort_values("o")
            if g.empty:
                continue
            pm._plot_coverage_line_with_band(
                ax,
                g["o"].to_numpy(dtype=float),
                g["coverage_mean"].to_numpy(dtype=float),
                g["coverage_se"].fillna(0.0).to_numpy(dtype=float),
                PRED_COLORS[pred],
                marker=PRED_MARKERS[pred],
                linestyle=PRED_LINESTYLES[pred],
                alpha_band=0.10,
            )
        ax.axhline(
            1.0 - alpha,
            color=pm.NOMINAL_COLOR,
            linestyle=(0, (5, 2)),
            linewidth=2.4,
        )
        ax.set_ylim(pm._coverage_ylim_lower(alpha), 1.02)
        ax.set_xlim(min(O_VALUES) - 1.2, max(O_VALUES) + 1.2)
        ax.set_title(_panel_title(lam), fontsize=FONTS["title"], color=pm.INK_COLOR, pad=16)
        xlabel = pm.X_LABEL_TARGET_O if i // 2 == 1 else ""
        ylab = ylabel if i % 2 == 0 else ""
        pm._style_axis(ax, xlabel, ylab, tick=FONTS["tick"], label=AXIS_LABEL_SIZE)
        _set_o_ticks(ax)
    _add_fig_legend(fig)
    _save(fig, out_stem)


def _plot_width_boxplots(df: pd.DataFrame, summary: pd.DataFrame, *, out_stem: Path) -> None:
    lambs = _panel_lambdas(summary)
    fig, axes = _new_grid()
    all_groups: list[np.ndarray] = []
    for i, lam in enumerate(lambs):
        ax = axes[i // 2, i % 2]
        w_g = 1.0 - float(lam)
        for o in O_VALUES:
            for pred, sign in (("rf", -1.0), ("bayes", 1.0)):
                sub = df[
                    (df["predictor"] == pred)
                    & np.isclose(df["w_g"], w_g)
                    & (df["o_observed"] == int(o))
                ]
                vals = pm._finite_width_array(sub["width"])
                if not len(vals):
                    continue
                all_groups.append(vals)
                color = PRED_COLORS[pred]
                ax.boxplot(
                    [vals],
                    positions=[float(o) + sign * BOX_HALF_OFFSET],
                    widths=BOX_WIDTH,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=pm._accessible_medianprops(),
                    whiskerprops=pm._accessible_whiskerprops(color),
                    capprops=pm._accessible_whiskerprops(color),
                    boxprops=pm._accessible_boxprops(color, alpha=0.68),
                )
        ax.set_xlim(min(O_VALUES) - 3.2, max(O_VALUES) + 3.2)
        ax.set_title(_panel_title(lam), fontsize=FONTS["title"], color=pm.INK_COLOR, pad=16)
        xlabel = pm.X_LABEL_TARGET_O if i // 2 == 1 else ""
        ylab = pm.Y_LABEL_WIDTH if i % 2 == 0 else ""
        pm._style_axis(ax, xlabel, ylab, tick=FONTS["tick"], label=AXIS_LABEL_SIZE)
        _set_o_ticks(ax)

    width_upper = pm._width_axis_upper_from_box_groups(all_groups)
    if width_upper is not None:
        for ax in axes.ravel():
            ax.set_ylim(0.0, width_upper)
    _add_fig_legend(fig)
    _save(fig, out_stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = _load(float(args.alpha), float(args.gamma))
    summary = _summarize(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "summary_by_predictor_weight.csv", index=False)
    _plot_coverage(
        summary,
        alpha=float(args.alpha),
        out_stem=OUT_DIR / "coverage_by_weight",
    )
    _plot_width_boxplots(df, summary, out_stem=OUT_DIR / "width_by_weight")
    print(f"Wrote {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
