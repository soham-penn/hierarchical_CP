#!/usr/bin/env python3
"""Compare Std-CP absolute vs studentized (DGP + ACS), write tables and plots."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.paths import PLOTS_MARGINAL, RESULTS_DGP_MARGINAL

FIG_DIR = PLOTS_MARGINAL / "dgp" / "stdcp_score_compare" / "figures"
SUM_DIR = PLOTS_MARGINAL / "dgp" / "stdcp_score_compare" / "summaries"
O_FOCUS = [0, 5, 10, 15, 20]


def _finite(s):
    x = np.asarray(s, float)
    return x[np.isfinite(x)]


def _summ_width_cov(df, o_col, w_col, c_col, alpha_col="alpha"):
    rows = []
    for alpha, ga in df.groupby(alpha_col):
        for o, g in ga.groupby(o_col):
            w = _finite(g[w_col])
            n_tot = len(g)
            n_fin = len(w)
            rows.append(
                {
                    "alpha": float(alpha),
                    "o": int(o),
                    "n": n_tot,
                    "n_finite": n_fin,
                    "finite_rate": n_fin / n_tot if n_tot else np.nan,
                    "cov": float(g[c_col].mean()),
                    "cov_se": float(np.sqrt(g[c_col].mean() * (1 - g[c_col].mean()) / n_tot))
                    if n_tot
                    else np.nan,
                    "width_mean": float(w.mean()) if n_fin else np.nan,
                    "width_se": float(w.std(ddof=1) / np.sqrt(n_fin)) if n_fin > 1 else np.nan,
                    "width_median": float(np.median(w)) if n_fin else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _load_dgp(design: str) -> pd.DataFrame:
    parts = []
    for suffix in ("alpha05", "alpha10_15_20"):
        tag = f"stdcp_score_compare_gamma5p0_{design}_{suffix}"
        p = RESULTS_DGP_MARGINAL / tag / f"{tag}_raw_results_complete.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _load_acs_stud(alpha: float) -> pd.DataFrame:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    p = (
        PLOTS_MARGINAL
        / "acs"
        / "stdcp_studentized"
        / tag
        / f"stdcp_studentized_{tag}_detailed.csv"
    )
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["alpha"] = alpha
    df["score_type"] = "studentized"
    return df


def _load_acs_abs(alpha: float) -> pd.DataFrame:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    p = (
        PLOTS_MARGINAL
        / "acs"
        / "stdcp_absolute"
        / tag
        / f"stdcp_absolute_{tag}_detailed.csv"
    )
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["alpha"] = alpha
    df["score_type"] = "absolute"
    return df


def _verify_dgp_absolute(design: str, new_abs: pd.DataFrame) -> None:
    paper = RESULTS_DGP_MARGINAL / f"true_marg_latent_rf_gamma5p0_{design}_alpha10" / (
        f"true_marg_latent_rf_gamma5p0_{design}_alpha10_raw_results_complete.csv"
    )
    if not paper.exists() or new_abs.empty:
        print(f"  skip verify {design}: missing files")
        return
    old = pd.read_csv(paper)
    old = old[np.isclose(old.alpha, 0.1)]
    merged = new_abs.merge(
        old[["experiment", "o_observed", "width_stdcp"]],
        on=["experiment", "o_observed"],
        how="inner",
    )
    if merged.empty:
        print(f"  verify {design} α=0.1: no merge keys")
        return
    a = merged["width"].to_numpy(float)
    b = merged["width_stdcp"].to_numpy(float)
    both_fin = np.isfinite(a) & np.isfinite(b)
    both_inf = (~np.isfinite(a)) & (~np.isfinite(b))
    n = len(merged)
    n_match = int(np.sum(both_inf | (both_fin & np.isclose(a, b, rtol=1e-8, atol=1e-8))))
    print(
        f"  verify {design} α=0.1 absolute vs paper Std-CP: "
        f"{n_match}/{n} exact matches "
        f"(finite {int(both_fin.sum())}, inf {int(both_inf.sum())})"
    )
    if both_fin.sum():
        rel = np.abs(a[both_fin] - b[both_fin]) / np.maximum(b[both_fin], 1e-12)
        print(f"    max |Δ| among finite: {np.max(np.abs(a[both_fin]-b[both_fin])):.3e}")


def _plot_compare(summary: pd.DataFrame, title: str, out_pdf: Path, width_scale=1.0):
    alphas = sorted(summary.alpha.unique())
    fig, axes = plt.subplots(len(alphas), 2, figsize=(9.2, 2.8 * max(len(alphas), 1)), squeeze=False)
    for i, a in enumerate(alphas):
        s = summary[summary.alpha == a]
        abs_ = s[s.score_type == "absolute"].sort_values("o")
        stu = s[s.score_type == "studentized"].sort_values("o")
        ax = axes[i, 0]
        ax.plot(np.asarray(abs_["o"]), np.asarray(abs_["cov"], dtype=float),
                "o-", label="absolute |Y−μ|", color="#0072B2")
        ax.plot(np.asarray(stu["o"]), np.asarray(stu["cov"], dtype=float),
                "s--", label="studentized |Y−μ|/σ", color="#D55E00")
        ax.axhline(1.0 - a, color="gray", ls=":", lw=1)
        ax.set_ylabel("coverage")
        ax.set_title(rf"{title}, $\alpha={a:g}$")
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=8)
        ax = axes[i, 1]
        ax.plot(abs_["o"], abs_["width_mean"] / width_scale, "o-", label="absolute", color="#0072B2")
        ax.plot(stu["o"], stu["width_mean"] / width_scale, "s--", label="studentized", color="#D55E00")
        ax.set_ylabel("mean finite width" + (" / $10^3$" if width_scale > 1 else ""))
        ax.set_xlabel("o")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUM_DIR.mkdir(parents=True, exist_ok=True)

    dgp_rows = []
    for design in ("fixedN21", "poissonNmean25"):
        df = _load_dgp(design)
        if df.empty:
            print(f"DGP {design}: not ready")
            continue
        abs_df = df[df.score_type == "absolute"].copy()
        _verify_dgp_absolute(design, abs_df[np.isclose(abs_df.alpha, 0.1)])
        parts = []
        for st, g in df.groupby("score_type"):
            s = _summ_width_cov(g, "o_observed", "width", "coverage")
            s["score_type"] = st
            s["design"] = design
            parts.append(s)
        d_sum = pd.concat(parts, ignore_index=True)
        dgp_rows.append(d_sum)
        print(f"\n-- {design} --")
        for a, ga in d_sum[d_sum.o.isin(O_FOCUS)].groupby("alpha"):
            piv = ga.pivot(index="o", columns="score_type", values=["cov", "width_mean", "finite_rate"])
            print(f"α={a:g}")
            print(piv.round(3).to_string())
        _plot_compare(d_sum[d_sum.o.isin(O_FOCUS)], design, FIG_DIR / f"dgp_{design}_stdcp_abs_vs_stud.pdf")
    if dgp_rows:
        dgp_sum = pd.concat(dgp_rows, ignore_index=True)
        dgp_sum = dgp_sum[dgp_sum.o.isin(O_FOCUS)]
        dgp_sum.to_csv(SUM_DIR / "dgp_stdcp_score_compare.csv", index=False)

    acs_rows = []
    for a in (0.05, 0.10, 0.15, 0.20):
        stud = _load_acs_stud(a)
        abso = _load_acs_abs(a)
        if stud.empty or abso.empty:
            print(f"ACS α={a}: missing abs={abso.empty} stud={stud.empty}")
            continue
        for st, g in (("studentized", stud), ("absolute", abso)):
            s = _summ_width_cov(g, "o", "width", "coverage")
            s["score_type"] = st
            acs_rows.append(s)
    if acs_rows:
        acs_sum = pd.concat(acs_rows, ignore_index=True)
        acs_sum.to_csv(SUM_DIR / "acs_stdcp_score_compare.csv", index=False)
        print("\n=== ACS Std-CP absolute vs studentized (income-scale width) ===")
        for a, ga in acs_sum.groupby("alpha"):
            piv = ga.pivot(index="o", columns="score_type", values=["cov", "width_mean", "finite_rate"])
            print(f"α={a:g}")
            print(piv.round(3).to_string())
        _plot_compare(
            acs_sum,
            "ACS",
            FIG_DIR / "acs_stdcp_abs_vs_stud.pdf",
            width_scale=1000.0,
        )
        print(f"\nWrote {SUM_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
