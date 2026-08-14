#!/usr/bin/env python3
"""
Export appendix baseline-comparison tables (GHCP + Dunn et al. baselines).

Reads raw DGP result CSVs (same source as plot_engine baseline panels):
  results_marginal/dgp/true_marg_latent_rf_gamma5p0_{fixedN21,poissonNmean25}_alpha10/
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
from code.paths import PLOTS_MARGINAL, RESULTS_DGP_MARGINAL  # noqa: E402

DEFAULT_OUT_DIR = PLOTS_MARGINAL / "dgp_true_marginal_rf" / "tables"
POISSON_DATASET = os.environ.get("HCP_POISSON_DATASET", "poissonNmean25")

RAW_FILES = {
    "fixedN21": RESULTS_DGP_MARGINAL
    / "true_marg_latent_rf_gamma5p0_fixedN21_alpha10"
    / "true_marg_latent_rf_gamma5p0_fixedN21_alpha10_raw_results_complete.csv",
    POISSON_DATASET: RESULTS_DGP_MARGINAL
    / f"true_marg_latent_rf_gamma5p0_{POISSON_DATASET}_alpha10"
    / f"true_marg_latent_rf_gamma5p0_{POISSON_DATASET}_alpha10_raw_results_complete.csv",
}

O_PAPER = [0, 5, 10, 15, 20]

BASELINE_ROWS = [
    ("hcp", "HCP"),
    ("pool", "Pooling CDF"),
    ("sub", "Subsampling Once"),
    ("rep", "Repeated Subsampling"),
]


def _se(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _cov_se(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return 0.0
    p = float(np.mean(x))
    return float(np.sqrt(p * (1.0 - p) / len(x)))


def _fmt_cov(mean: float, se: float) -> str:
    if not np.isfinite(mean):
        return "---"
    if se == 0.0:
        return f"{mean:.2f} (0.0)"
    return f"{mean:.2f} ({se:.3f})"


def _fmt_width(mean: float, se: float) -> str:
    if not np.isfinite(mean):
        return r"$+\infty$"
    if se == 0.0:
        return f"{mean:.2f}"
    return f"{mean:.2f} ({se:.3f})"


def _stats(raw: pd.DataFrame, key: str, o: int) -> tuple[float, float, float, float]:
    sub = raw[raw["o_observed"] == o]
    cov = sub[f"coverage_{key}"].to_numpy(dtype=float)
    width = (
        sub[f"width_{key}"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    return float(np.mean(cov)), _cov_se(cov), float(np.mean(width)), _se(width)


def build_table(
    raw: pd.DataFrame,
    *,
    label: str,
    caption: str,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Coverage & Width \\",
        r"\midrule",
    ]
    for o in O_PAPER:
        c_mean, c_se, w_mean, w_se = _stats(raw, "donor_hcp_randomized", o)
        lines.append(
            rf"GHCP, $o={o}$  & {_fmt_cov(c_mean, c_se)} & {_fmt_width(w_mean, w_se)} \\"
        )
    for key, label_name in BASELINE_ROWS:
        c_mean, c_se, w_mean, w_se = _stats(raw, key, 0)
        lines.append(
            rf"{label_name} & {_fmt_cov(c_mean, c_se)} & {_fmt_width(w_mean, w_se)} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args, _unknown = p.parse_known_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(RAW_FILES["fixedN21"])
    poisson = pd.read_csv(RAW_FILES[POISSON_DATASET])

    cap_fixed = (
        r"Empirical coverage and mean interval width for GHCP across initial "
        r"test-group sample sizes $o$, together with the baseline procedures, "
        r"under the fixed-size simulation setting of Figure~\ref{fig:N21-cov-width-20}. "
        r"Values are averaged over $B=1000$ repetitions, with standard errors "
        r"reported in parentheses."
    )
    cap_poi = (
        r"Empirical coverage and mean interval width for GHCP across initial "
        r"test-group sample sizes $o$, together with the baseline procedures, "
        r"under the Poisson group-size simulation setting of "
        r"Figure~\ref{fig:poi-cov-width-20}. Values are averaged over "
        r"$B=1000$ repetitions, with standard errors reported in parentheses."
    )

    parts = [
        "% Source: results_marginal/dgp/true_marg_latent_rf_gamma5p0_*_alpha10 raw CSVs",
        "",
        build_table(
            fixed,
            label="tab:dhcp-hcp-alpha01-extra",
            caption=cap_fixed,
        ),
        build_table(
            poisson,
            label="tab:dhcp-hcp-poisson-alpha01-extra",
            caption=cap_poi,
        ),
    ]
    out = args.out_dir / "simulations_added_baselines_tables.tex"
    out.write_text("\n".join(parts))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
