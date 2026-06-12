#!/usr/bin/env python3
"""Diagnose ACS over-coverage: infinite intervals, quantile selection, inf mass."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scores import conformal_threshold, weighted_quantile
from methods.baseline_hcp import compute_hcp_interval_radius


def _is_infinite_interval(row) -> bool:
    lo, hi = row.get("lower"), row.get("upper")
    if not np.isfinite(lo) or not np.isfinite(hi):
        return True
    if lo <= -1e10 or hi >= 1e10:
        return True
    return False


def analyze_detailed_csv(path: Path, alpha_nominal: float = 0.8) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["o"].isin([0, 5, 10, 15, 20])].copy()
    df["infinite_interval"] = df.apply(_is_infinite_interval, axis=1)
    df["log_width"] = df["upper"] - df["lower"]

    rows = []
    for (method, o), g in df.groupby(["method", "o"], dropna=False):
        if g["coverage"].isna().all():
            continue
        rows.append(
            {
                "method": method,
                "o": int(o),
                "n": int(g["coverage"].notna().sum()),
                "coverage_mean": float(g["coverage"].mean()),
                "infinite_rate": float(g["infinite_interval"].mean()),
                "finite_coverage": float(g.loc[~g["infinite_interval"], "coverage"].mean())
                if (~g["infinite_interval"]).any()
                else np.nan,
                "log_width_median": float(g.loc[~g["infinite_interval"], "log_width"].median())
                if (~g["infinite_interval"]).any()
                else np.nan,
                "width_income_median": float(g.loc[~g["infinite_interval"], "width_income"].median())
                if (~g["infinite_interval"]).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def hcp_quantile_diagnostics(scores_list, alpha=0.2, quantile_modes=("deterministic", "randomized")):
    """Rebuild HCP weighted scores + inf atom; report CDF jump and chosen q."""
    K = len(scores_list)
    all_scores, all_weights = [], []
    for scores in scores_list:
        n = len(scores)
        if n > 0:
            all_scores.extend(scores)
            all_weights.extend([1.0 / ((K + 1) * n)] * n)
    w_inf = 1.0 / (K + 1)
    all_scores.append(np.inf)
    all_weights.append(w_inf)

    scores = np.asarray(all_scores, dtype=float)
    weights = np.asarray(all_weights, dtype=float)
    beta = 1.0 - alpha

    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]
    cdf = np.cumsum(w)

    # mass at +inf
    finite_mask = np.isfinite(s)
    mass_finite = float(np.sum(w[finite_mask]))
    mass_inf = float(np.sum(w[~finite_mask]))
    max_finite = float(np.max(s[finite_mask])) if np.any(finite_mask) else np.nan

    # jump containing beta
    j = int(np.searchsorted(cdf, beta, side="left"))
    j = min(j, len(s) - 1)
    a = float(cdf[j - 1]) if j > 0 else 0.0
    b = float(cdf[j])
    q_upper = float(s[j])
    q_lower = float(s[j - 1]) if j > 0 else 0.0

    out = {
        "K_calib_groups_used": K,
        "n_scores_total": len(scores),
        "mass_finite": mass_finite,
        "mass_inf": mass_inf,
        "max_finite_score": max_finite,
        "beta": beta,
        "cdf_before_upper": a,
        "cdf_at_upper": b,
        "q_lower": q_lower,
        "q_upper": q_upper,
        "beta_in_jump": bool((a < beta - 1e-12) and (b > beta + 1e-12)),
    }

    for mode in quantile_modes:
        if mode == "deterministic":
            out[f"q_{mode}"] = weighted_quantile(scores, weights, alpha)
        else:
            info = conformal_threshold(
                scores, weights, alpha=alpha, quantile_mode="randomized",
                random_seed=42, return_info=True,
            )
            out[f"q_{mode}"] = info["q_randomized"]
            out[f"randomized_draw"] = info["randomized"]
            out[f"gamma_upper"] = info["gamma_upper"]

    return out


def main():
    path = ROOT / "real_data/acs/results/true_marginal_permuted_alpha20/acs_true_marg_alpha20_detailed.csv"
    if not path.exists():
        print(f"Missing {path}")
        return

    print("=" * 72)
    print("ACS permuted alpha=0.2 — interval-level diagnostics")
    print("=" * 72)
    tab = analyze_detailed_csv(path)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nNote: HCP/Pooling/Subsampling/Repeated rows exist only at o=0 in this design.")
    print("      D-HCP/S-HCP are recomputed at each o.\n")

    # HCP structural example using median calibration scores from one replicate is hard offline;
    # use typical K=20 calib groups with ~27 points each from experiment design.
    print("=" * 72)
    print("HCP structural quantile (illustrative K=20 groups, n=27 each, synthetic scores)")
    print("=" * 72)
    rng = np.random.default_rng(0)
    K = 20
    n = 27
    # typical log-residual scale ~0.5-2
    scores_list = [rng.exponential(0.8, size=n) for _ in range(K)]
    diag = hcp_quantile_diagnostics(scores_list, alpha=0.2)
    for k, v in diag.items():
        print(f"  {k}: {v}")

    print("\nSame with heavier tails (more conservatism risk):")
    scores_list = [rng.exponential(1.2, size=n) for _ in range(K)]
    diag2 = hcp_quantile_diagnostics(scores_list, alpha=0.2)
    print(f"  mass_inf={diag2['mass_inf']:.4f}, max_finite={diag2['max_finite_score']:.3f}")
    print(f"  q_det={diag2['q_deterministic']:.3f}, q_rand={diag2['q_randomized']:.3f}, jump={diag2['beta_in_jump']}")

    # Compare randomized vs det over many seeds
    scores_list = [rng.exponential(0.9, size=n) for _ in range(K)]
    K = 20
    all_scores, all_weights = [], []
    for scores in scores_list:
        all_scores.extend(scores)
        all_weights.extend([1.0 / ((K + 1) * len(scores))] * len(scores))
    all_scores.append(np.inf)
    all_weights.append(1.0 / (K + 1))
    det = weighted_quantile(all_scores, all_weights, 0.2)
    rands = [
        conformal_threshold(all_scores, all_weights, alpha=0.2, quantile_mode="randomized", random_seed=i)
        for i in range(5000)
    ]
    print(f"\nOver 5000 randomized draws: det q={det:.3f}, E[q_rand]={np.mean(rands):.3f}, P(q<det)={np.mean(np.array(rands)<det):.3f}")


if __name__ == "__main__":
    main()
