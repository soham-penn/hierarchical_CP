#!/usr/bin/env python3
"""
Score-level diagnostic: why is GHCP wider than Std-CP?

Uses frozen global μ from capture (same score definition for both methods).
For each (experiment, o):

  1. Build full GHCP weighted score pool (cal groups + target tail + ∞ padding),
     tagging each finite score as target vs other.
  2. Build Std-CP scores = |Y-μ| on the same target history rows (global-center
     inductive CP; no local refit) so scores are comparable.
  3. Report:
     - similarity of target GHCP scores vs Std-CP scores (same indices)
     - Spearman rank agreement (ordering)
     - where target scores sit in the sorted GHCP pool (percentiles)
     - whether the GHCP (1-α) knot comes from target or another group

Usage:
  python code/marginal/dgp_capture/diagnose_ghcp_vs_stdcp_scores.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from code.shared.dgp.experiments import _split_conformal_radius
from scores import weighted_quantile

TARGET_INDEX = 35
O_FOCUS = [10, 15, 20, 25, 30, 35]


def _load_groups(obs_e: pd.DataFrame, n_calib: int):
    x_cols = [c for c in obs_e.columns if c.startswith("x_")]
    u_cols = [c for c in obs_e.columns if c.startswith("u_")]
    groups = {}
    for g in range(n_calib + 1):
        gdf = obs_e[obs_e["group_id"] == g].sort_values("row_idx")
        if gdf.empty:
            continue
        groups[g] = {
            "y": gdf["y"].to_numpy(dtype=float),
            "X": gdf[x_cols].to_numpy(dtype=float),
        }
    return groups


def _one_experiment(args):
    experiment, alpha, obs_e, fits_e, mu_by_fit, n_calib = args
    test_idx = n_calib
    groups = _load_groups(obs_e, n_calib)
    score_rows = []
    summary_rows = []

    efits = fits_e[(fits_e.experiment == experiment) & (fits_e.hcp_method == "Donor-HCP")]
    for _, fr in efits.iterrows():
        o = int(fr.o)
        if o not in O_FOCUS and o != 0:
            # still allow all o in O_FOCUS only for speed; include 5 too
            pass
        fit_id = fr.fit_id
        s_comp = [int(x) for x in json.loads(fr.comp_slots_json)]
        all_idx = list(range(n_calib + 1))
        S = [g for g in all_idx if g not in set(s_comp)]
        S_cal = [g for g in S if g != test_idx]
        S_size = len(S)
        if S_size < 1:
            continue

        mu_map = mu_by_fit.get(fit_id, {})
        tau = int(np.floor(o / 2))
        if o > 0 and tau >= o:
            tau = o - 1
        tau = max(0, tau)

        # ---- GHCP pool ----
        pool_scores, pool_weights, pool_src, pool_gid, pool_row = [], [], [], [], []
        for j in S_cal:
            if j not in groups:
                continue
            Nj = len(groups[j]["y"])
            if Nj <= tau:
                continue
            idx_tail = list(range(tau, Nj))
            for i in idx_tail:
                mu = mu_map.get((j, i))
                if mu is None:
                    continue
                sc = abs(float(groups[j]["y"][i]) - float(mu))
                pool_scores.append(sc)
                pool_src.append("other")
                pool_gid.append(j)
                pool_row.append(i)
            if idx_tail:
                wj = 1.0 / (S_size * len(idx_tail))
                pool_weights.extend([wj] * len(idx_tail))

        N_donor = int(np.median([len(groups[g]["y"]) for g in range(n_calib) if g in groups]))
        test = groups[test_idx]
        idx_tail_test = list(range(tau, o)) if o > tau else []
        n_added_target = 0
        for i in idx_tail_test:
            mu = mu_map.get((test_idx, i))
            if mu is None:
                continue
            sc = abs(float(test["y"][i]) - float(mu))
            pool_scores.append(sc)
            pool_src.append("target")
            pool_gid.append(test_idx)
            pool_row.append(i)
            n_added_target += 1
        n_inf = max(0, N_donor - o)
        n_total_test = n_added_target + n_inf
        if n_total_test > 0:
            w_test = 1.0 / (S_size * n_total_test)
            # weights for target finite
            pool_weights.extend([w_test] * n_added_target)
            if n_inf:
                pool_scores.extend([np.inf] * n_inf)
                pool_weights.extend([w_test] * n_inf)
                pool_src.extend(["inf"] * n_inf)
                pool_gid.extend([-1] * n_inf)
                pool_row.extend([-1] * n_inf)

        if not pool_scores or sum(pool_weights) <= 0:
            continue
        w = np.asarray(pool_weights, float)
        w = w / w.sum()
        scores_arr = np.asarray(pool_scores, float)
        q_ghcp = float(weighted_quantile(scores_arr, w, alpha))

        # which finite score realizes q? (first in sorted order reaching cumweight >= 1-α)
        finite_mask = np.isfinite(scores_arr)
        order = np.argsort(scores_arr)
        cum = np.cumsum(w[order])
        knot_pos = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
        if knot_pos >= len(order):
            knot_src = "inf"
            knot_score = np.inf
        else:
            knot_i = int(order[knot_pos])
            knot_src = pool_src[knot_i]
            knot_score = float(scores_arr[knot_i])

        # ---- Std-CP scores on SAME target rows 0..o-1 under SAME μ ----
        std_by_row = {}
        if o >= 1:
            for i in range(o):
                mu = mu_map.get((test_idx, i))
                if mu is None:
                    continue
                std_by_row[i] = abs(float(test["y"][i]) - float(mu))
            r_std = _split_conformal_radius(list(std_by_row.values()), alpha)
        else:
            r_std = np.inf

        # GHCP target scores on overlapping indices (τ:o used in pool; also compare on 0:o)
        ghcp_target_by_row = {}
        for i, src, gid, sc in zip(pool_row, pool_src, pool_gid, pool_scores):
            if src == "target" and gid == test_idx and np.isfinite(sc):
                ghcp_target_by_row[int(i)] = float(sc)

        # paired comparison on shared row indices
        shared = sorted(set(ghcp_target_by_row) & set(std_by_row))
        if shared:
            g_vals = np.array([ghcp_target_by_row[i] for i in shared], float)
            s_vals = np.array([std_by_row[i] for i in shared], float)
            # under frozen μ these should be identical on shared rows (τ:o)
            mae = float(np.mean(np.abs(g_vals - s_vals)))
            spearman = float(pd.Series(g_vals).corr(pd.Series(s_vals), method="spearman")) if len(shared) > 1 else np.nan
            # ordering vs Std-CP on ALL std rows: compare rank of τ:o subset
        else:
            mae, spearman = np.nan, np.nan
            g_vals = np.array([])
            s_vals = np.array([])

        # Also compare full Std-CP vector (0:o) ranks vs GHCP target ranks on τ:o
        # Position of each target score among all finite GHCP pool scores
        finite_other = scores_arr[(np.asarray(pool_src) == "other") & finite_mask]
        finite_all = scores_arr[finite_mask]
        target_finite = scores_arr[(np.asarray(pool_src) == "target") & np.isfinite(scores_arr)]

        if len(target_finite) and len(finite_all):
            # percentile of each target score in the full finite pool
            pcts = np.array([
                float(np.mean(finite_all <= t)) for t in target_finite
            ])
            mean_pct = float(pcts.mean())
            # fraction of target scores below the GHCP q
            frac_target_below_q = float(np.mean(target_finite <= q_ghcp)) if np.isfinite(q_ghcp) else np.nan
        else:
            mean_pct = np.nan
            frac_target_below_q = np.nan

        # Std-CP order statistic index
        std_vals = np.array(list(std_by_row.values()), float)
        if len(std_vals):
            k = int(np.ceil((len(std_vals) + 1) * (1 - alpha)))
            std_finite = k <= len(std_vals)
        else:
            std_finite = False

        summary_rows.append({
            "experiment": experiment,
            "o": o,
            "q_ghcp": q_ghcp,
            "r_std": r_std if np.isfinite(r_std) else np.nan,
            "knot_src": knot_src,
            "knot_score": knot_score if np.isfinite(knot_score) else np.nan,
            "n_target_scores": int(len(target_finite)),
            "n_other_scores": int(len(finite_other)),
            "n_shared": int(len(shared)),
            "mae_shared": mae,
            "spearman_shared": spearman,
            "mean_target_pct_in_pool": mean_pct,
            "frac_target_below_q": frac_target_below_q,
            "med_target": float(np.median(target_finite)) if len(target_finite) else np.nan,
            "med_other": float(np.median(finite_other)) if len(finite_other) else np.nan,
            "med_std": float(np.median(std_vals)) if len(std_vals) else np.nan,
            "std_smaller": float(np.isfinite(r_std) and np.isfinite(q_ghcp) and r_std < q_ghcp),
            "both_finite": float(np.isfinite(r_std) and np.isfinite(q_ghcp)),
        })

        # store per-score rows for focused o
        if o in O_FOCUS:
            for sc, src, gid, row_i, wt in zip(pool_scores, pool_src, pool_gid, pool_row, pool_weights):
                if src == "inf":
                    continue
                score_rows.append({
                    "experiment": experiment,
                    "o": o,
                    "source": src,
                    "group_id": gid,
                    "row_idx": row_i,
                    "score": float(sc) if np.isfinite(sc) else np.nan,
                    "weight": float(wt),
                    "method_pool": "ghcp",
                })
            for row_i, sc in std_by_row.items():
                score_rows.append({
                    "experiment": experiment,
                    "o": o,
                    "source": "stdcp",
                    "group_id": test_idx,
                    "row_idx": row_i,
                    "score": float(sc),
                    "weight": np.nan,
                    "method_pool": "stdcp",
                })

    return summary_rows, score_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--capture",
        type=Path,
        default=REPO
        / "plots_marginal"
        / "dgp_true_marginal_strong_rf_localrf_gamma0p0"
        / "capture_data"
        / "fixedN21_alpha20",
    )
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--max-experiments", type=int, default=None)
    args = p.parse_args()

    obs = pd.read_csv(args.capture / "observations.csv")
    fits = pd.read_csv(args.capture / "global_fits.csv")
    mu = pd.read_csv(args.capture / "mu_global.csv")

    mu_by_fit = {}
    for fit_id, g in mu.groupby("fit_id"):
        mu_by_fit[fit_id] = {
            (int(r.group_id), int(r.row_idx)): float(r.mu_global)
            for r in g.itertuples()
        }

    n_calib = int(obs[obs.role == "calib"]["group_id"].max()) + 1
    exps = sorted(obs.experiment.unique())
    if args.max_experiments:
        exps = exps[: args.max_experiments]

    print(f"Diagnosing {len(exps)} experiments from {args.capture}", flush=True)
    tasks = []
    for e in exps:
        e = int(e)
        tasks.append((
            e,
            float(args.alpha),
            obs[obs.experiment == e].copy(),
            fits[fits.experiment == e].copy(),
            {fid: mp for fid, mp in mu_by_fit.items() if fid.startswith(f"e{e}_")},
            n_calib,
        ))

    summaries, scores = [], []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [ex.submit(_one_experiment, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            s, sc = fut.result()
            summaries.extend(s)
            scores.extend(sc)
            if i % 50 == 0 or i == len(futs):
                print(f"  done {i}/{len(futs)}", flush=True)

    out_dir = args.capture.parent.parent / "summaries"
    fig_dir = args.capture.parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summ = pd.DataFrame(summaries)
    scor = pd.DataFrame(scores)
    summ_path = out_dir / "ghcp_vs_stdcp_score_diagnostic_summary.csv"
    scor_path = out_dir / "ghcp_vs_stdcp_score_diagnostic_scores.csv"
    summ.to_csv(summ_path, index=False)
    scor.to_csv(scor_path, index=False)
    print("Wrote", summ_path)
    print("Wrote", scor_path)

    # ---- tables ----
    print("\n=== Check 1–2: GHCP target vs Std-CP scores (same frozen μ; shared rows τ:o) ===")
    print(f"{'o':>3}  {'MAE':>8} {'Spearman':>9} {'med_tgt':>8} {'med_std':>8} {'med_other':>9}")
    for o in sorted(summ.o.unique()):
        g = summ[summ.o == o]
        print(f"{o:3d}  {g.mae_shared.mean():8.2e} {g.spearman_shared.mean():9.3f} "
              f"{g.med_target.mean():8.3f} {g.med_std.mean():8.3f} {g.med_other.mean():9.3f}")

    print("\n=== Check 3: where do target scores sit in the GHCP pool? ===")
    print(f"{'o':>3}  {'mean_pct':>8} {'frac≤q':>8} {'P(knot=target)':>14} {'P(knot=other)':>14} {'P(r_std<q)':>10}")
    for o in sorted(summ.o.unique()):
        g = summ[summ.o == o]
        p_tgt = (g.knot_src == "target").mean()
        p_oth = (g.knot_src == "other").mean()
        p_inf = (g.knot_src == "inf").mean()
        both = g[g.both_finite > 0]
        print(f"{o:3d}  {g.mean_target_pct_in_pool.mean():8.3f} {g.frac_target_below_q.mean():8.3f} "
              f"{p_tgt:14.3f} {p_oth:14.3f} {both.std_smaller.mean() if len(both) else np.nan:10.3f}")

    # ---- figures ----
    o_plot = [o for o in [10, 20, 30, 35] if o in summ.o.unique()]
    fig, axes = plt.subplots(2, len(o_plot), figsize=(3.4 * len(o_plot), 7.2), constrained_layout=True)
    if len(o_plot) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, o in enumerate(o_plot):
        ax = axes[0, j]
        g = scor[(scor.o == o) & (scor.method_pool == "ghcp")]
        tgt = g[g.source == "target"]["score"].dropna().to_numpy()
        oth = g[g.source == "other"]["score"].dropna().to_numpy()
        # subsample others for viz
        rng = np.random.default_rng(0)
        if len(oth) > 5000:
            oth = rng.choice(oth, size=5000, replace=False)
        bins = np.linspace(0, np.nanpercentile(np.concatenate([tgt, oth]), 99), 40)
        ax.hist(oth, bins=bins, density=True, alpha=0.45, color="#E69F00", label="GHCP other")
        ax.hist(tgt, bins=bins, density=True, alpha=0.55, color="#0072B2", label="GHCP target")
        # mark mean q and mean r
        s = summ[summ.o == o]
        ax.axvline(s.q_ghcp.median(), color="#0072B2", ls="--", lw=1.5, label="med q_GHCP")
        ax.axvline(s.r_std.median(), color="#009E73", ls="-.", lw=1.5, label="med r_Std")
        ax.set_title(f"o={o}: score densities")
        ax.set_xlabel("score |Y−μ|")
        if j == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=7, frameon=False)

        ax = axes[1, j]
        # ECDFs of target percentile in pool
        pct = summ[summ.o == o]["mean_target_pct_in_pool"].dropna()
        ax.hist(pct, bins=20, range=(0, 1), color="#0072B2", alpha=0.7)
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_title(f"o={o}: mean target pct in pool")
        ax.set_xlabel("mean percentile of target scores")
        if j == 0:
            ax.set_ylabel("replicates")

    fig.suptitle(
        r"Frozen-$\mu$ score diagnostic · $\gamma=0$ localRF capture · $\alpha=0.2$"
        "\nAre target residuals small, so GHCP's knot comes from another group?",
        fontsize=11, fontweight="bold",
    )
    pdf = fig_dir / "ghcp_vs_stdcp_score_diagnostic.pdf"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "ghcp_vs_stdcp_score_diagnostic.png", dpi=160, bbox_inches="tight")
    print("Wrote", pdf)

    # knot source bar chart
    fig2, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    os_ = sorted(summ.o.unique())
    p_t = [(summ[summ.o == o].knot_src == "target").mean() for o in os_]
    p_o = [(summ[summ.o == o].knot_src == "other").mean() for o in os_]
    p_i = [(summ[summ.o == o].knot_src == "inf").mean() for o in os_]
    x = np.arange(len(os_))
    ax.bar(x, p_t, label="knot from target", color="#0072B2")
    ax.bar(x, p_o, bottom=p_t, label="knot from other group", color="#E69F00")
    ax.bar(x, p_i, bottom=np.array(p_t) + np.array(p_o), label="knot = ∞", color="#999999")
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in os_])
    ax.set_xlabel("o")
    ax.set_ylabel("fraction of replicates")
    ax.set_ylim(0, 1.05)
    ax.set_title("Which score realizes the GHCP (1−α) quantile?")
    ax.legend(frameon=False, fontsize=8)
    pdf2 = fig_dir / "ghcp_quantile_knot_source.pdf"
    fig2.savefig(pdf2, dpi=300, bbox_inches="tight")
    print("Wrote", pdf2)


if __name__ == "__main__":
    main()
