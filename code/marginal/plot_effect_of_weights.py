#!/usr/bin/env python3
"""2×3 GHCP coverage and width panels across the λ_local merger-weight grid."""

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

from code.marginal.run_effect_of_weights import O_VALUES, RESULTS_PREFIX
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

OUT_DIR_ROOT = PLOTS_MARGINAL / "effect_of_weights"
GHCP_COLOR = pm.METHOD_COLORS["GHCP"]
FIGSIZE = (26.0, 16.5)
FONTS = pm._stacked_panel_fonts()
# Full design grid: λ_local = k/7 for k=1..6.
LAMBDA_LOCAL_PANELS = tuple(k / 7.0 for k in range(1, 7))


def _out_dir(gamma: float) -> Path:
    return OUT_DIR_ROOT / gamma_tag(gamma)


def _load(alpha: float, gamma: float) -> pd.DataFrame:
    tag = f"{RESULTS_PREFIX}_{gamma_tag(gamma)}_poissonNmean25_{alpha_to_tag(alpha)}"
    path = RESULTS_DGP_MARGINAL / tag / f"{tag}_raw_results_complete.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing weight-grid results: {path}")
    df = pd.read_csv(path)
    df = df[df["o_observed"].isin(O_VALUES)].copy()
    df["w_local"] = 1.0 - df["w_g"].astype(float)
    return df


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (w_local, o), g in df.groupby(["w_local", "o_observed"]):
        cov = g["coverage"].to_numpy(dtype=float)
        width = g["width"].to_numpy(dtype=float)
        finite = width[np.isfinite(width)]
        n = len(g)
        p = float(np.mean(cov))
        rows.append(
            {
                "w_local": float(w_local),
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
    return pd.DataFrame(rows).sort_values(["w_local", "o"]).reset_index(drop=True)


def _monotonicity_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Check whether mean finite width is nonincreasing in o, for o>0."""
    rows = []
    for w_local, g in summary.groupby("w_local"):
        g = g.sort_values("o")
        g_pos = g[g["o"] > 0]
        widths = g_pos["width_mean"].to_numpy(dtype=float)
        os_ = g_pos["o"].to_numpy(dtype=int)
        diffs = np.diff(widths)
        ok = bool(np.all(diffs <= 1e-12))
        rows.append(
            {
                "w_local": float(w_local),
                "width_nonincreasing_o_gt0": ok,
                "max_upward_step": float(np.max(diffs)) if len(diffs) else np.nan,
                "widths": ", ".join(
                    f"o={o}:{w:.4g}" for o, w in zip(os_, widths) if np.isfinite(w)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("w_local").reset_index(drop=True)


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
        2, 3, figsize=FIGSIZE, sharex=True, sharey=True, constrained_layout=True
    )
    return fig, np.atleast_2d(axes)


def _save(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), transparent=True, bbox_inches="tight", dpi=300)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_coverage(summary: pd.DataFrame, *, alpha: float, out_stem: Path) -> None:
    lambs = _panel_lambdas(summary)
    fig, axes = _new_grid()
    ylabel = "Empirical coverage"
    for i, lam in enumerate(lambs):
        ax = axes[i // 3, i % 3]
        g = summary[np.isclose(summary["w_local"], lam)].sort_values("o")
        pm._plot_coverage_line_with_band(
            ax,
            g["o"].to_numpy(dtype=float),
            g["coverage_mean"].to_numpy(dtype=float),
            g["coverage_se"].fillna(0.0).to_numpy(dtype=float),
            GHCP_COLOR,
            marker=pm.METHOD_MARKERS["GHCP"],
            linestyle=pm.METHOD_LINESTYLES["GHCP"],
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
        ax.set_xticks(O_VALUES)
        ax.set_title(_panel_title(lam), fontsize=FONTS["title"], color=pm.INK_COLOR, pad=16)
        xlabel = pm.X_LABEL_TARGET_O if i // 3 == 1 else ""
        ylab = ylabel if i % 3 == 0 else ""
        pm._style_axis(ax, xlabel, ylab, tick=FONTS["tick"], label=FONTS["label"])
    _save(fig, out_stem)


def _plot_width_boxplots(df: pd.DataFrame, summary: pd.DataFrame, *, out_stem: Path) -> None:
    lambs = _panel_lambdas(summary)
    fig, axes = _new_grid()
    all_groups: list[np.ndarray] = []
    for i, lam in enumerate(lambs):
        ax = axes[i // 3, i % 3]
        for o in O_VALUES:
            sub = df[np.isclose(df["w_local"], lam) & (df["o_observed"] == int(o))]
            vals = pm._finite_width_array(sub["width"])
            if not len(vals):
                continue
            all_groups.append(vals)
            ax.boxplot(
                [vals],
                positions=[int(o)],
                widths=pm.BOX_WIDTH_O_AXIS,
                patch_artist=True,
                showfliers=False,
                medianprops=pm._accessible_medianprops(),
                whiskerprops=pm._accessible_whiskerprops(GHCP_COLOR),
                capprops=pm._accessible_whiskerprops(GHCP_COLOR),
                boxprops=pm._accessible_boxprops(GHCP_COLOR, alpha=0.68),
            )
        ax.set_xticks(O_VALUES)
        ax.set_xlim(min(O_VALUES) - 2.4, max(O_VALUES) + 2.4)
        ax.set_title(_panel_title(lam), fontsize=FONTS["title"], color=pm.INK_COLOR, pad=16)
        xlabel = pm.X_LABEL_TARGET_O if i // 3 == 1 else ""
        ylab = pm.Y_LABEL_WIDTH if i % 3 == 0 else ""
        pm._style_axis(ax, xlabel, ylab, tick=FONTS["tick"], label=FONTS["label"])

    width_upper = pm._width_axis_upper_from_box_groups(all_groups)
    if width_upper is not None:
        for ax in axes.ravel():
            ax.set_ylim(0.0, width_upper)
    _save(fig, out_stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = _load(args.alpha, args.gamma)
    summary = _summarize(df)
    mono = _monotonicity_report(summary)

    out_dir = _out_dir(float(args.gamma))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary_by_weight_o.csv", index=False)
    mono.to_csv(out_dir / "width_monotonicity.csv", index=False)

    _plot_coverage(
        summary,
        alpha=float(args.alpha),
        out_stem=out_dir / "coverage_by_weight",
    )
    _plot_width_boxplots(df, summary, out_stem=out_dir / "width_by_weight")

    lines = [
        f"# Effect of GHCP merger weights (Poi(25), $\\gamma={float(args.gamma):g}$)",
        "",
        "Local merger weight $\\lambda_{\\mathrm{local}}\\in\\{k/7\\}_{k=1}^{6}$ "
        "(2×3 panels). Same DGP as the paper Poisson design except $\\gamma$.",
        "",
        "Figures use the Simulations plot style. Width panels are boxplots of "
        "finite interval length (no outliers).",
        "",
        "## Width nonincreasing in $o$ (excluding $o=0$, where $\\tau=0$)",
        "",
        "| $\\lambda_{local}$ | nonincreasing? | max upward step |",
        "|---:|:---:|---:|",
    ]
    for _, row in mono.iterrows():
        lines.append(
            f"| {row['w_local']:.3f} | "
            f"{'yes' if row['width_nonincreasing_o_gt0'] else 'no'} | "
            f"{row['max_upward_step']:.4g} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")
    print(mono.to_string(index=False))


if __name__ == "__main__":
    main()
