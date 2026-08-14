#!/usr/bin/env python3
"""
Compare GHCP vs Std-CP under residual / studentized / CQR scores (strong RF, α=0.1).

Produces coverage + width boxplots side by side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = REPO / "plots_marginal" / "dgp_true_marginal_strong_rf" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

O_VALUES = [0, 5, 10, 15, 20, 25, 30, 35]
ALPHA = 0.1

SCORE_SOURCES = {
    "residual |Y−μ|": {
        "raw": REPO
        / "results_marginal"
        / "dgp"
        / "true_marg_latent_strong_rf_gamma5p0_fixedN21_alpha10"
        / "true_marg_latent_strong_rf_gamma5p0_fixedN21_alpha10_raw_results_complete.csv",
        # fallback: existing trials long already has GHCP; Std-CP from local-RF recompute
        "trials": REPO
        / "plots_marginal"
        / "dgp_true_marginal_strong_rf"
        / "summaries"
        / "fixedN21_trials_long.csv",
        "stdcp_rf": REPO
        / "plots_marginal"
        / "dgp_true_marginal_strong_rf"
        / "summaries"
        / "stdcp_recomputed_local_rf.csv",
    },
    "studentized |Y−μ|/σ": {
        "raw": REPO
        / "results_marginal"
        / "dgp"
        / "true_marg_latent_strong_rf_studentized_gamma5p0_fixedN21_alpha10"
        / "true_marg_latent_strong_rf_studentized_gamma5p0_fixedN21_alpha10_raw_results_complete.csv",
    },
    "CQR": {
        "raw": REPO
        / "results_marginal"
        / "dgp"
        / "true_marg_latent_strong_rf_cqr_gamma5p0_fixedN21_alpha10"
        / "true_marg_latent_strong_rf_cqr_gamma5p0_fixedN21_alpha10_raw_results_complete.csv",
    },
}

COLORS = {"GHCP": "#0072B2", "Std-CP": "#009E73"}


def _from_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    mapping = [
        ("GHCP", "coverage_donor_hcp_randomized", "width_donor_hcp_randomized"),
        ("Std-CP", "coverage_stdcp", "width_stdcp"),
    ]
    for method, ccol, wcol in mapping:
        for _, r in df.iterrows():
            rows.append({
                "method": method,
                "o": int(r["o_observed"]),
                "coverage": float(r[ccol]),
                "width": float(r[wcol]) if pd.notna(r[wcol]) else np.nan,
            })
    return pd.DataFrame(rows)


def _from_residual_fallback(cfg: dict) -> pd.DataFrame:
    t = pd.read_csv(cfg["trials"])
    t = t[(np.isclose(t["alpha"], ALPHA)) & (t["method"].isin(["GHCP", "Std-CP"]))]
    out = t[["method", "o", "coverage", "width"]].copy()
    # Prefer local-RF Std-CP recompute if present
    std_path = cfg.get("stdcp_rf")
    if std_path and Path(std_path).exists():
        std = pd.read_csv(std_path)
        std = std[np.isclose(std["alpha"], ALPHA)]
        out = out[out["method"] != "Std-CP"]
        std = std.rename(columns={})
        std["method"] = "Std-CP"
        out = pd.concat([out, std[["method", "o", "coverage", "width"]]], ignore_index=True)
    return out


def load_score(name: str, cfg: dict) -> pd.DataFrame | None:
    raw = cfg.get("raw")
    if raw and Path(raw).exists():
        print(f"  {name}: loaded raw {raw.name}")
        return _from_raw(Path(raw))
    if name.startswith("residual") and cfg.get("trials") and Path(cfg["trials"]).exists():
        print(f"  {name}: fallback trials + stdcp recompute")
        return _from_residual_fallback(cfg)
    print(f"  {name}: MISSING ({raw})")
    return None


def _box_pair(ax, data: pd.DataFrame, metric: str, title: str, logy: bool = False):
    for o in O_VALUES:
        for dx, method in [(-0.2, "GHCP"), (0.2, "Std-CP")]:
            g = data[(data["method"] == method) & (data["o"] == o)]
            if metric == "coverage":
                vals = g["coverage"].to_numpy(float)
                vals = vals[np.isfinite(vals)]
            else:
                vals = g["width"].to_numpy(float)
                vals = vals[np.isfinite(vals)]
            if not len(vals):
                if metric == "width":
                    ax.text(
                        o + dx, 0.02, "∞",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8,
                        color=COLORS[method], fontweight="bold",
                    )
                continue
            ax.boxplot(
                [vals],
                positions=[o + dx],
                widths=0.35,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", markersize=2.0, alpha=0.25, color=COLORS[method]),
                medianprops=dict(color="k", lw=1.1),
                boxprops=dict(facecolor=COLORS[method], alpha=0.55, edgecolor=COLORS[method]),
                whiskerprops=dict(color=COLORS[method]),
                capprops=dict(color=COLORS[method]),
            )
    ax.set_xticks(O_VALUES)
    ax.set_xlabel("o")
    ax.set_title(title)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    if metric == "coverage":
        ax.axhline(1 - ALPHA, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.7, 1.02)
    else:
        ax.set_ylabel("Width")
        if logy:
            ax.set_yscale("log")
    ax.legend(
        handles=[Patch(facecolor=COLORS[m], alpha=0.55, label=m) for m in ("GHCP", "Std-CP")],
        frameon=False, fontsize=8, loc="best",
    )


def main():
    loaded = {}
    print("Loading score variants:")
    for name, cfg in SCORE_SOURCES.items():
        df = load_score(name, cfg)
        if df is not None:
            loaded[name] = df

    if not loaded:
        raise SystemExit("No score results available yet.")

    names = list(loaded.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 8.2), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for j, name in enumerate(names):
        _box_pair(axes[0, j], loaded[name], "coverage", f"{name}\ncoverage (α={ALPHA})")
        _box_pair(axes[1, j], loaded[name], "width", f"{name}\nwidth", logy=True)

    fig.suptitle(
        "Strong RF · fixed N=21 · GHCP vs Std-CP under alternate scores",
        fontsize=12, fontweight="bold",
    )
    pdf = OUT / "strong_rf_score_types_ghcp_vs_stdcp_boxplots.pdf"
    png = OUT / "strong_rf_score_types_ghcp_vs_stdcp_boxplots.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print("Wrote", pdf)

    # Numeric summary
    print("\n=== median width / mean coverage (finite widths for med) ===")
    for name, df in loaded.items():
        print(f"\n{name}")
        print(f'{"o":>3}  {"GHCP cov":>8} {"GHCP med":>9}  {"Std cov":>8} {"Std med":>9} {"Std fin":>8}')
        for o in O_VALUES:
            line = [f"{o:3d}"]
            for method in ("GHCP", "Std-CP"):
                g = df[(df.method == method) & (df.o == o)]
                cov = g.coverage.mean()
                w = g.width.to_numpy(float)
                fin = np.isfinite(w)
                med = float(np.nanmedian(w[fin])) if fin.any() else float("nan")
                if method == "GHCP":
                    line.append(f"{cov:8.3f} {med:9.2f}")
                else:
                    line.append(f"{cov:8.3f} {med:9.2f} {fin.mean():8.0%}")
            print("  ".join(line))


if __name__ == "__main__":
    main()
