#!/usr/bin/env python3
"""GHCP coverage and width panels across size–intercept correlation ξ."""

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

from code.marginal.run_size_shift_sensitivity import O_VALUES, RESULTS_PREFIX, XI_GRID
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

OUT_DIR = PLOTS_MARGINAL / "size_shift"
GHCP_COLOR = pm.METHOD_COLORS["GHCP"]
FONTS = pm._stacked_panel_fonts()
# Paper 2×2 figures — same grid a fresh run_size_shift_sensitivity.py writes.
PANEL_XI = XI_GRID
NCOLS = 2
FIGSIZE = (18.0, 16.5)


def _load(alpha: float, gamma: float) -> pd.DataFrame:
    tag = f"{RESULTS_PREFIX}_{gamma_tag(gamma)}_poissonNmean25_{alpha_to_tag(alpha)}"
    path = RESULTS_DGP_MARGINAL / tag / f"{tag}_raw_results_complete.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing size-shift results: {path}")
    df = pd.read_csv(path)
    df = df[df["o_observed"].isin(O_VALUES)].copy()
    df = df[df["xi"].apply(lambda x: any(np.isclose(x, p) for p in PANEL_XI))].copy()
    if df.empty:
        raise ValueError(f"No rows with ξ in {PANEL_XI} in {path}")
    return df


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (xi, o), g in df.groupby(["xi", "o_observed"]):
        cov = g["coverage"].to_numpy(dtype=float)
        width = g["width"].to_numpy(dtype=float)
        finite = width[np.isfinite(width)]
        n = len(g)
        p = float(np.mean(cov))
        corr = g["corr_B_N"].to_numpy(dtype=float)
        rows.append(
            {
                "xi": float(xi),
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
                "corr_B_N_mean": float(np.nanmean(corr)),
            }
        )
    return pd.DataFrame(rows).sort_values(["xi", "o"]).reset_index(drop=True)


def _panel_xi(summary: pd.DataFrame) -> list[float]:
    avail = [float(x) for x in summary["xi"].unique()]
    chosen = []
    for want in PANEL_XI:
        match = [a for a in avail if np.isclose(a, want)]
        if not match:
            raise ValueError(f"missing ξ={want} in results; have {sorted(avail)}")
        chosen.append(float(match[0]))
    return chosen


def _panel_title(xi: float) -> str:
    return rf"$\xi={xi:g}$"


def _new_grid(n_panels: int):
    nrows = int(np.ceil(n_panels / NCOLS))
    ncols = min(NCOLS, n_panels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    return fig, np.atleast_2d(axes), nrows, ncols


def _save(fig, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), transparent=True, bbox_inches="tight", dpi=300)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_coverage(summary: pd.DataFrame, *, alpha: float, out_stem: Path) -> None:
    xis = _panel_xi(summary)
    fig, axes, nrows, ncols = _new_grid(len(xis))
    ylabel = "Empirical coverage"
    for i, xi in enumerate(xis):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        g = summary[np.isclose(summary["xi"], xi)].sort_values("o")
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
        ax.set_title(_panel_title(xi), fontsize=FONTS["title"], color=pm.INK_COLOR, pad=16)
        xlabel = pm.X_LABEL_TARGET_O if r == nrows - 1 else ""
        ylab = ylabel if c == 0 else ""
        pm._style_axis(ax, xlabel, ylab, tick=FONTS["tick"], label=FONTS["label"])
    _save(fig, out_stem)


def _plot_width_boxplots(df: pd.DataFrame, summary: pd.DataFrame, *, out_stem: Path) -> None:
    xis = _panel_xi(summary)
    fig, axes, nrows, ncols = _new_grid(len(xis))
    all_groups: list[np.ndarray] = []
    for i, xi in enumerate(xis):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        for o in O_VALUES:
            sub = df[np.isclose(df["xi"], xi) & (df["o_observed"] == int(o))]
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
        ax.set_title(_panel_title(xi), fontsize=FONTS["title"], color=pm.INK_COLOR, pad=16)
        xlabel = pm.X_LABEL_TARGET_O if r == nrows - 1 else ""
        ylab = pm.Y_LABEL_WIDTH if c == 0 else ""
        pm._style_axis(ax, xlabel, ylab, tick=FONTS["tick"], label=FONTS["label"])

    width_upper = pm._width_axis_upper_from_box_groups(all_groups)
    if width_upper is not None:
        for ax in axes.ravel():
            ax.set_ylim(0.0, width_upper)
    _save(fig, out_stem)


def _fmt_cov(mean: float, se: float) -> str:
    return f"{mean:.3f} ({se:.3f})"


def _fmt_width(mean: float, se: float) -> str:
    return f"{mean:.3f} ({se:.3f})"


def _xi_label(xi: float) -> str:
    if np.isclose(xi, 0.0):
        return r"$\xi=0$"
    return rf"$\xi={xi:g}$"


def _write_table(summary: pd.DataFrame, *, alpha: float, out_path: Path) -> None:
    xis = _panel_xi(summary)
    o_cols = " & ".join(str(o) for o in O_VALUES)
    cov_lines = []
    width_lines = []
    for xi in xis:
        g = summary[np.isclose(summary["xi"], xi)].sort_values("o")
        cov_cells = " & ".join(
            _fmt_cov(float(r.coverage_mean), float(r.coverage_se)) for r in g.itertuples()
        )
        width_cells = " & ".join(
            _fmt_width(float(r.width_mean), float(r.width_se)) for r in g.itertuples()
        )
        cov_lines.append(f"{_xi_label(xi)}\n& {cov_cells} \\\\")
        width_lines.append(f"{_xi_label(xi)}\n& {width_cells} \\\\")
    nom = 1.0 - float(alpha)
    tex = f"""% Size-shift GHCP table (App. D.3). Source: summary_by_xi_o.csv
% ξ ∈ {{{", ".join(str(x) for x in PANEL_XI)}}}.
\\begin{{table}}
\\centering
\\caption{{\\footnotesize Empirical coverage and mean interval width for GHCP
under size--intercept coupling $\\xi$, Poisson DGP of Sec.\\,3.1 except $B_j$,
$\\gamma=5$, $\\alpha={alpha:g}$. Entries are means across $B=1000$ trials,
with standard errors in parentheses. All intervals were finite.}}
\\label{{tab:size-shift}}
\\small
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{l{"c" * len(O_VALUES)}}}
\\toprule
& \\multicolumn{{{len(O_VALUES)}}}{{c}}{{Test-group sample size $o$}} \\\\
\\cmidrule(lr){{2-{len(O_VALUES)+1}}}
$\\xi$ & {o_cols} \\\\
\\midrule
\\multicolumn{{{len(O_VALUES)+1}}}{{l}}{{\\textit{{Empirical coverage}} ($1-\\alpha={nom:.2f}$)}} \\\\[2pt]
{chr(10).join(cov_lines)}
\\addlinespace[4pt]
\\multicolumn{{{len(O_VALUES)+1}}}{{l}}{{\\textit{{Mean interval width}}}} \\\\[2pt]
{chr(10).join(width_lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out_path.write_text(tex)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = _load(args.alpha, args.gamma)
    summary = _summarize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "summary_by_xi_o.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_by_xi_o_panels.csv", index=False)

    _plot_coverage(
        summary,
        alpha=float(args.alpha),
        out_stem=OUT_DIR / "coverage_by_xi",
    )
    _plot_width_boxplots(df, summary, out_stem=OUT_DIR / "width_by_xi")
    _write_table(summary, alpha=float(args.alpha), out_path=OUT_DIR / "size_shift_table.tex")
    print(f"Wrote {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
