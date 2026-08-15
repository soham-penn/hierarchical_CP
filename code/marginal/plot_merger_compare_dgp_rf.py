#!/usr/bin/env python3
"""Plot GHCP merger comparison (eq4 / sqrt / bayes_re) for α=0.1 DGP RF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.paths import PLOTS_MARGINAL, RESULTS_DGP_MARGINAL

METHODS = [
    ("donor_hcp_bayes_re", "GHCP Bayes-RE", "#1b9e77", "-"),
    ("donor_hcp_sqrt", "GHCP sqrt (c=0.5)", "#d95f02", "--"),
    ("donor_hcp_eq4", "GHCP Eq.(4) (c=1)", "#7570b3", ":"),
    ("donor_hcp_no_within", "GHCP no WGT", "#e7298a", "-."),
    ("hcp", "HCP", "#666666", "-"),
]


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for o, g in raw.groupby("o_observed"):
        for key, label, *_ in METHODS:
            cov = g[f"coverage_{key}"].to_numpy(dtype=float)
            wid = g[f"width_{key}"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            row = {
                "o": int(o),
                "method_key": key,
                "method": label,
                "coverage_mean": float(np.mean(cov)),
                "coverage_se": float(np.sqrt(np.mean(cov) * (1 - np.mean(cov)) / len(cov))) if len(cov) else np.nan,
                "width_median": float(np.median(wid)) if len(wid) else np.nan,
                "width_mean": float(np.mean(wid)) if len(wid) else np.nan,
                "width_q25": float(np.quantile(wid, 0.25)) if len(wid) else np.nan,
                "width_q75": float(np.quantile(wid, 0.75)) if len(wid) else np.nan,
                "n": int(len(cov)),
            }
            if key == "donor_hcp_bayes_re" and "bayes_w_g" in g.columns:
                wg = g["bayes_w_g"].dropna().to_numpy(dtype=float)
                row["bayes_w_g_mean"] = float(np.mean(wg)) if len(wg) else np.nan
                row["bayes_w_g_median"] = float(np.median(wg)) if len(wg) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_design(raw: pd.DataFrame, title: str, out_pdf: Path, alpha: float):
    s = _summary(raw)
    o_vals = sorted(s["o"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    for key, label, color, ls in METHODS:
        sub = s[s["method_key"] == key].sort_values("o")
        if sub.empty:
            continue
        y = sub["coverage_mean"].to_numpy()
        se = sub["coverage_se"].to_numpy()
        x = sub["o"].to_numpy()
        ax.plot(x, y, color=color, linestyle=ls, marker="o", label=label, linewidth=2)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.12)
    ax.axhline(1.0 - alpha, color="black", linewidth=1.2, linestyle="--", label=rf"$1-\alpha={1-alpha:.2f}$")
    ax.set_xlabel(r"within-group history $o$")
    ax.set_ylabel("empirical coverage")
    ax.set_xticks(o_vals)
    ax.set_ylim(0.75, 1.01)
    ax.grid(True, alpha=0.3)
    ax.set_title("Coverage")

    ax = axes[1]
    for key, label, color, ls in METHODS:
        sub = s[s["method_key"] == key].sort_values("o")
        if sub.empty:
            continue
        # HCP is flat in o — plot as horizontal reference
        if key == "hcp":
            med = float(sub["width_median"].iloc[0])
            ax.axhline(med, color=color, linestyle=ls, linewidth=2, label=f"{label} (med={med:.2f})")
            continue
        ax.plot(
            sub["o"], sub["width_median"],
            color=color, linestyle=ls, marker="o", label=label, linewidth=2,
        )
    ax.set_xlabel(r"within-group history $o$")
    ax.set_ylabel("median interval width")
    ax.set_xticks(o_vals)
    ax.grid(True, alpha=0.3)
    ax.set_title("Width (median)")
    ax.legend(loc="best", fontsize=8, frameon=True)

    axes[0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle(title, fontsize=13)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return s


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=5.0)
    p.add_argument("--configs", type=str, default="fixedN21,poissonNmean25")
    return p.parse_args()


def main():
    args = parse_args()
    alpha_tag = f"alpha{100 * args.alpha:.6g}".replace(".", "p")
    if "p" not in alpha_tag and len(alpha_tag.replace("alpha", "")) < 2:
        alpha_tag = f"alpha{int(100 * args.alpha):02d}"
    # match alpha_to_tag: 0.1 -> alpha10
    pct = f"{100.0 * float(args.alpha):.6g}".replace(".", "p")
    alpha_tag = f"alpha{pct}"
    gtag = f"gamma{str(float(args.gamma)).replace('.', 'p')}"
    fig_dir = PLOTS_MARGINAL / "dgp" / "figures" / "merger_compare"
    sum_dir = PLOTS_MARGINAL / "dgp" / "summaries" / "merger_compare"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        tag = f"merger_compare_rf_{gtag}_{name}_{alpha_tag}"
        raw_path = RESULTS_DGP_MARGINAL / tag / f"{tag}_raw_results_complete.csv"
        if not raw_path.exists():
            print(f"Missing: {raw_path}")
            continue
        raw = pd.read_csv(raw_path)
        title = f"Merger compare · {name} · α={args.alpha:g} · γ={args.gamma:g}"
        out_pdf = fig_dir / f"{name}_{alpha_tag}_coverage_width_by_o.pdf"
        s = plot_design(raw, title, out_pdf, alpha=args.alpha)
        s.to_csv(sum_dir / f"{name}_{alpha_tag}_summary.csv", index=False)
        # compact print
        print(f"\n=== {name} ===")
        pivot = s.pivot(index="o", columns="method", values="width_median")
        print("median width:")
        print(pivot.to_string(float_format=lambda x: f"{x:7.3f}"))
        pivot_c = s.pivot(index="o", columns="method", values="coverage_mean")
        print("coverage:")
        print(pivot_c.to_string(float_format=lambda x: f"{x:7.3f}"))
        if "bayes_w_g_mean" in s.columns:
            bw = s[s["method_key"] == "donor_hcp_bayes_re"][["o", "bayes_w_g_mean", "bayes_w_g_median"]]
            print("Bayes w_g:")
            print(bw.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
        print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
