#!/usr/bin/env python3
"""
Export paper Simulation tables (mean width) from dgp_true_marginal_rf summaries.

Source experiment (exact match to previous median-width paper numbers):
  paper-results/dgp_true_marginal_rf/summaries/*_summary_by_alpha_o_method.csv
  B=1000 true-marginal RF replicates, absolute score, latent intercept γ=5.

Width entries use width_mean; SEs are the existing width_se (finite-width sample SE).
Re-run after regenerating summaries via:
  .venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from code.paths import PLOTS_MARGINAL  # noqa: E402

DEFAULT_SUMMARY_DIR = PLOTS_MARGINAL / "dgp_true_marginal_rf" / "summaries"
DEFAULT_OUT_DIR = PLOTS_MARGINAL / "dgp_true_marginal_rf" / "tables"
POISSON_DATASET = os.environ.get("HCP_POISSON_DATASET", "poissonNmean25")

O_PAPER = [0, 5, 10, 15, 20]


def _fmt_cov(mean: float, se: float) -> str:
    if not np.isfinite(mean):
        return "---"
    if se is None or not np.isfinite(se) or se == 0.0:
        return f"{mean:.3f} (0.0)"
    return f"{mean:.3f} ({se:.3f})"


def _fmt_width(mean: float, se: float, *, n_finite: float, n_total: float) -> str:
    if n_total > 0 and n_finite == 0:
        return r"$+\infty$"
    if not np.isfinite(mean):
        return r"$+\infty$"
    if se is None or not np.isfinite(se):
        return f"{mean:.3f}"
    return f"{mean:.3f} ({se:.3f})"


def _row(summary: pd.DataFrame, method: str, alpha: float, o: int) -> pd.Series:
    sub = summary[
        (summary["method"] == method)
        & np.isclose(summary["alpha"], alpha)
        & (summary["o"] == o)
    ]
    if sub.empty:
        raise KeyError(f"Missing row method={method} alpha={alpha} o={o}")
    return sub.iloc[0]


def _pct_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def build_ghcp_hcp_table(
    summary: pd.DataFrame,
    *,
    alpha: float,
    label: str,
    caption: str,
) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{5}{c}{GHCP} & HCP \\",
        r"\cmidrule(lr){2-6}",
        r"Value of $o$ & 0 & 5 & 10 & 15 & 20 & -- \\",
        r"\midrule",
    ]
    cov_cells = []
    wid_cells = []
    for o in O_PAPER:
        r = _row(summary, "GHCP", alpha, o)
        cov_cells.append(_fmt_cov(float(r["coverage_mean"]), float(r["coverage_se"])))
        wid_cells.append(
            _fmt_width(
                float(r["width_mean"]) if pd.notna(r["width_mean"]) else np.nan,
                float(r["width_se"]) if pd.notna(r["width_se"]) else np.nan,
                n_finite=float(r["width_n_finite"]),
                n_total=float(r["width_n_total"]),
            )
        )
    h = _row(summary, "HCP", alpha, 0)
    cov_cells.append(_fmt_cov(float(h["coverage_mean"]), float(h["coverage_se"])))
    wid_cells.append(
        _fmt_width(
            float(h["width_mean"]) if pd.notna(h["width_mean"]) else np.nan,
            float(h["width_se"]) if pd.notna(h["width_se"]) else np.nan,
            n_finite=float(h["width_n_finite"]),
            n_total=float(h["width_n_total"]),
        )
    )
    nom = 1.0 - alpha
    lines.append(
        rf"Coverage ($1-\alpha={nom:.2f}$)"
        + "".join(f"\n& {c}" for c in cov_cells)
        + r" \\"
    )
    lines.append(
        r"Mean Width"
        + "".join(f"\n& {c}" for c in wid_cells)
        + r" \\"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_wgt_table(summary: pd.DataFrame, *, alpha: float = 0.1) -> str:
    caption = (
        r"Empirical coverage and mean width for GHCP with and without WGT "
        r"(within-group training) across test-group size $o$, together with HCP as a "
        r"baseline, for fixed $N_j=21$, $\gamma=5$, and $\alpha=0.1$, using a random "
        r"forest global predictor. Entries are means across "
        r"$1000$ simulation repetitions, with standard errors in parentheses."
    )
    methods = [
        ("GHCP no within", "GHCP (no WGT)"),
        ("GHCP", "GHCP (WGT)"),
        ("HCP", "HCP"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:sim-with-vs-without-training}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Metric & Method & 0 & 5 & 10 & 15 & 20 \\",
        r"\midrule",
    ]

    for i, (method_key, method_label) in enumerate(methods):
        cells = [
            _fmt_cov(
                float(_row(summary, method_key, alpha, o)["coverage_mean"]),
                float(_row(summary, method_key, alpha, o)["coverage_se"]),
            )
            for o in O_PAPER
        ]
        prefix = "Coverage\n& " if i == 0 else "& "
        lines.append(prefix + method_label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\addlinespace")

    for i, (method_key, method_label) in enumerate(methods):
        cells = []
        for o in O_PAPER:
            r = _row(summary, method_key, alpha, o)
            cells.append(
                _fmt_width(
                    float(r["width_mean"]) if pd.notna(r["width_mean"]) else np.nan,
                    float(r["width_se"]) if pd.notna(r["width_se"]) else np.nan,
                    n_finite=float(r["width_n_finite"]),
                    n_total=float(r["width_n_total"]),
                )
            )
        if i == 0:
            lines.append(r"Mean" + "\n& " + method_label + " & " + " & ".join(cells) + r" \\")
        elif i == 1:
            lines.append(r"Width & " + method_label + "  & " + " & ".join(cells) + r" \\")
        else:
            lines.append(r"& " + method_label + "            & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def reduction_notes(fixed: pd.DataFrame, poisson: pd.DataFrame) -> str:
    g0 = _row(fixed, "GHCP", 0.1, 0)
    g20 = _row(fixed, "GHCP", 0.1, 20)
    nw20 = _row(fixed, "GHCP no within", 0.1, 20)
    w_red = _pct_reduction(float(g0["width_mean"]), float(g20["width_mean"]))
    se_red = _pct_reduction(float(g0["width_se"]), float(g20["width_se"]))
    wgt_red = _pct_reduction(float(nw20["width_mean"]), float(g20["width_mean"]))

    pg0 = _row(poisson, "GHCP", 0.1, 0)
    pg20 = _row(poisson, "GHCP", 0.1, 20)
    pw_red = _pct_reduction(float(pg0["width_mean"]), float(pg20["width_mean"]))

    return "\n".join(
        [
            "% Auto-computed from width_mean / width_se in fixedN21 & poisson summaries",
            f"% Fixed N21 GHCP mean-width reduction o=0→20: {w_red:.2f}%  "
            f"(means {float(g0['width_mean']):.6f} → {float(g20['width_mean']):.6f})",
            f"% Fixed N21 GHCP width SE reduction o=0→20: {se_red:.2f}%  "
            f"(SE {float(g0['width_se']):.6f} → {float(g20['width_se']):.6f})",
            f"% Fixed N21 WGT vs no-WGT mean-width reduction at o=20: {wgt_red:.2f}%  "
            f"({float(nw20['width_mean']):.6f} → {float(g20['width_mean']):.6f})",
            f"% Poisson GHCP mean-width reduction o=0→20: {pw_red:.2f}%  "
            f"(means {float(pg0['width_mean']):.6f} → {float(pg20['width_mean']):.6f})",
            f"% Suggested prose (fixed): At $o=20$, the mean width is reduced by "
            f"{w_red:.0f}\\%, and the standard error of the width is reduced by "
            f"{se_red:.0f}\\%, relative to $o=0$.",
            f"% Suggested prose (WGT): For $o=20$, WGT yields a {wgt_red:.0f}\\% reduction "
            f"in mean width relative to no-WGT.",
            "",
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary_dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args, _unknown = p.parse_known_args()

    fixed = pd.read_csv(args.summary_dir / "fixedN21_summary_by_alpha_o_method.csv")
    poisson = pd.read_csv(
        args.summary_dir / f"{POISSON_DATASET}_summary_by_alpha_o_method.csv"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if POISSON_DATASET == "poissonNmean21":
        poi_law = r"$N_j\overset{\mathrm{i.i.d.}}{\sim}1+\mathrm{Poi}(20)$"
    else:
        poi_law = r"$N_j\overset{\mathrm{i.i.d.}}{\sim}\mathrm{Poi}(25)$"

    cap005 = (
        r"Empirical coverage and mean interval width for GHCP across test-group size $o$, "
        r"together with HCP, for fixed $N_j=21$, $\gamma=5$, and $\alpha=0.05$, using a "
        r"random forest global predictor. Entries are means across "
        r"$1000$ simulation repetitions, with standard errors in parentheses. "
        r"Width entries equal to $+\infty$ indicate that all reported intervals were trivial."
    )
    cap01 = (
        r"Empirical coverage and mean interval width for GHCP across test-group size $o$, "
        r"together with HCP, for fixed $N_j=21$, $\gamma=5$, and $\alpha=0.1$, using a "
        r"random forest global predictor. Entries are means across "
        r"$1000$ simulation repetitions, with standard errors in parentheses."
    )
    cap_poi = (
        r"Empirical coverage and mean interval width for GHCP across test-group size $o$, "
        rf"together with HCP, for {poi_law}, "
        r"$\gamma=5$, and $\alpha=0.1$, using a random forest global predictor. Entries are "
        r"means across $1000$ simulation repetitions, with "
        r"standard errors in parentheses."
    )

    parts = [
        "% Source: paper-results/dgp_true_marginal_rf/summaries/",
        "",
        reduction_notes(fixed, poisson),
        build_ghcp_hcp_table(
            fixed, alpha=0.05, label="tab:dhcp-hcp-alpha005", caption=cap005,
        ),
        build_ghcp_hcp_table(
            fixed, alpha=0.1, label="tab:dhcp-hcp-alpha01", caption=cap01,
        ),
        build_ghcp_hcp_table(
            poisson, alpha=0.1, label="tab:dhcp-hcp-poisson-alpha01", caption=cap_poi,
        ),
        build_wgt_table(fixed, alpha=0.1),
    ]
    out = args.out_dir / "simulations_mean_width_tables.tex"
    out.write_text("\n".join(parts))
    print(f"Wrote {out}")

    # Also write a short prose helper with exact % for the paper body
    notes = args.out_dir / "simulations_mean_width_prose_notes.txt"
    notes.write_text(reduction_notes(fixed, poisson))
    print(f"Wrote {notes}")


if __name__ == "__main__":
    main()
