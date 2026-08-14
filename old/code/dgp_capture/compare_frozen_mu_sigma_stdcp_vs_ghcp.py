#!/usr/bin/env python3
"""
Apples-to-apples: freeze the SAME global μ (and RF-σ) from S_comp, then compare

  Std-CP (global center): split-CP radius from ONLY target-group history scores
  GHCP:                   weighted hierarchical quantile of the SAME score definition

Both intervals use the same center μ(x*) and scale σ(x*).
Only the threshold construction differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from code.shared.dgp.experiments import _split_conformal_radius
from scores import weighted_quantile

O_VALUES = [0, 5, 10, 15, 20, 25, 30, 35]
TARGET_INDEX = 35
EPS = 1e-6


def _load_groups(obs: pd.DataFrame, experiment: int, n_calib: int):
    sub = obs[obs["experiment"] == experiment]
    x_cols = [c for c in sub.columns if c.startswith("x_")]
    u_cols = [c for c in sub.columns if c.startswith("u_")]
    groups = {}
    for g in range(n_calib + 1):
        gdf = sub[sub["group_id"] == g].sort_values("row_idx")
        if gdf.empty:
            continue
        groups[g] = {
            "X": gdf[x_cols].to_numpy(dtype=float),
            "U": gdf.iloc[0][u_cols].to_numpy(dtype=float),
            "y": gdf["y"].to_numpy(dtype=float),
        }
    return groups


def _fit_sigma_rf(groups, s_comp, mu_map, ntree=100, nodesize=5, mtry=0.6, rs=123):
    feats, targets = [], []
    for g in s_comp:
        if g not in groups:
            continue
        X = groups[g]["X"]
        U = groups[g]["U"]
        y = groups[g]["y"]
        for i in range(len(y)):
            mu = mu_map.get((g, i))
            if mu is None or not np.isfinite(mu):
                continue
            feats.append(np.concatenate([X[i], U]))
            targets.append(abs(float(y[i]) - float(mu)))
    if len(targets) < 20:
        return None
    Xmat = np.asarray(feats, float)
    yvec = np.asarray(targets, float)
    rf = RandomForestRegressor(
        n_estimators=int(ntree),
        max_features=mtry,
        min_samples_leaf=max(1, min(int(nodesize), len(yvec))),
        random_state=int(rs) + 17,
        n_jobs=1,
    )
    rf.fit(Xmat, yvec)
    return rf


def _sigma(rf, x, u):
    if rf is None:
        return 1.0
    feat = np.concatenate([np.asarray(x, float).ravel(), np.asarray(u, float).ravel()]).reshape(1, -1)
    return float(np.clip(rf.predict(feat)[0], EPS, 1e12))


def _score(y, mu, s):
    return abs(float(y) - float(mu)) / max(float(s), EPS)


def _one_experiment(args):
    experiment, alpha, obs_e, fits_e, mu_by_fit, score_type, n_calib = args
    test_idx = n_calib
    groups = _load_groups(obs_e, experiment, n_calib)
    rows = []

    efits = fits_e[(fits_e.experiment == experiment) & (fits_e.hcp_method == "Donor-HCP")]
    for _, fr in efits.iterrows():
        o = int(fr.o)
        fit_id = fr.fit_id
        s_comp = [int(x) for x in json.loads(fr.comp_slots_json)]
        all_idx = list(range(n_calib + 1))
        S = [g for g in all_idx if g not in set(s_comp)]
        S_cal = [g for g in S if g != test_idx]
        S_size = len(S)
        if S_size < 1:
            continue

        mu_map = mu_by_fit.get(fit_id, {})
        sigma_rf = _fit_sigma_rf(groups, s_comp, mu_map) if score_type == "studentized" else None

        tau = int(np.floor(o / 2))
        if o > 0 and tau >= o:
            tau = o - 1
        tau = max(0, tau)

        scores, weights = [], []
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
                s = _sigma(sigma_rf, groups[j]["X"][i], groups[j]["U"])
                scores.append(_score(groups[j]["y"][i], mu, s))
            if idx_tail:
                wj = 1.0 / (S_size * len(idx_tail))
                weights.extend([wj] * len(idx_tail))

        N_donor = int(np.median([len(groups[g]["y"]) for g in range(n_calib) if g in groups]))
        test = groups[test_idx]
        idx_tail_test = list(range(tau, o)) if o > tau else []
        test_scores = []
        for i in idx_tail_test:
            mu = mu_map.get((test_idx, i))
            if mu is None:
                continue
            s = _sigma(sigma_rf, test["X"][i], test["U"])
            test_scores.append(_score(test["y"][i], mu, s))
        n_tail = len(test_scores)
        n_inf = max(0, N_donor - o)
        n_total_test = n_tail + n_inf
        if n_total_test > 0:
            w_test = 1.0 / (S_size * n_total_test)
            if n_tail:
                scores.extend(test_scores)
                weights.extend([w_test] * n_tail)
            if n_inf:
                scores.extend([np.inf] * n_inf)
                weights.extend([w_test] * n_inf)

        if scores and sum(weights) > 0:
            w = np.asarray(weights, float)
            w = w / w.sum()
            q_ghcp = float(weighted_quantile(np.asarray(scores, float), w, alpha))
        else:
            q_ghcp = np.inf

        if o < 1:
            r_std = np.inf
        else:
            hist_scores = []
            for i in range(o):
                mu = mu_map.get((test_idx, i))
                if mu is None:
                    continue
                s = _sigma(sigma_rf, test["X"][i], test["U"])
                hist_scores.append(_score(test["y"][i], mu, s))
            r_std = _split_conformal_radius(hist_scores, alpha)

        mu_t = mu_map.get((test_idx, TARGET_INDEX))
        if mu_t is None:
            continue
        s_t = _sigma(sigma_rf, test["X"][TARGET_INDEX], test["U"])
        y_true = float(test["y"][TARGET_INDEX])

        def interval(q):
            if not np.isfinite(q):
                return (-np.inf, np.inf)
            return (float(mu_t) - q * s_t, float(mu_t) + q * s_t)

        lo_g, hi_g = interval(q_ghcp)
        lo_s, hi_s = interval(r_std)

        rows.append({
            "experiment": experiment,
            "o": o,
            "alpha": alpha,
            "q_ghcp": q_ghcp,
            "r_std": r_std,
            "s_target": s_t,
            "mu_target": float(mu_t),
            "width_ghcp": (hi_g - lo_g) if np.isfinite(lo_g) and np.isfinite(hi_g) else np.nan,
            "width_std": (hi_s - lo_s) if np.isfinite(lo_s) and np.isfinite(hi_s) else np.nan,
            "cov_ghcp": float(lo_g <= y_true <= hi_g),
            "cov_std": float(lo_s <= y_true <= hi_s),
            "std_smaller": float(
                np.isfinite(r_std) and np.isfinite(q_ghcp) and (r_std < q_ghcp)
            ),
            "both_finite": float(np.isfinite(r_std) and np.isfinite(q_ghcp)),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--capture",
        type=Path,
        default=REPO
        / "plots_marginal"
        / "dgp_true_marginal_strong_rf_studentized_gamma0p0"
        / "capture_data"
        / "fixedN21_alpha20",
    )
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--score-type", choices=["absolute", "studentized"], default="studentized")
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
    print(f"Replay {len(exps)} experiments, score={args.score_type}, alpha={args.alpha}", flush=True)

    tasks = []
    for e in exps:
        e = int(e)
        tasks.append((
            e,
            float(args.alpha),
            obs[obs.experiment == e].copy(),
            fits[fits.experiment == e].copy(),
            {fid: mp for fid, mp in mu_by_fit.items() if fid.startswith(f"e{e}_")},
            args.score_type,
            n_calib,
        ))

    rows = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [ex.submit(_one_experiment, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % 50 == 0 or i == len(futs):
                print(f"  done {i}/{len(futs)}", flush=True)

    out = pd.DataFrame(rows)
    out_dir = args.capture.parent.parent / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"frozen_mu_sigma_{args.score_type}_alpha{int(round(100 * args.alpha))}_compare.csv"
    out.to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    print(f"\n{'o':>3}  {'P(r_std<q)':>10} {'med r/q':>8} {'GHCP medW':>10} {'Std medW':>10} "
          f"{'both fin':>8} {'GHCP cov':>8} {'Std cov':>8}")
    for o in O_VALUES:
        g = out[out.o == o]
        both = g[g.both_finite > 0]
        if both.empty:
            print(f"{o:3d}  {'n/a':>10} {'n/a':>8} {np.nanmedian(g.width_ghcp):10.2f} "
                  f"{np.nanmedian(g.width_std):10.2f} {0:8.0%} {g.cov_ghcp.mean():8.3f} {g.cov_std.mean():8.3f}")
            continue
        ratio = (both.r_std / both.q_ghcp).replace([np.inf, -np.inf], np.nan)
        print(f"{o:3d}  {both.std_smaller.mean():10.2f} {ratio.median():8.2f} "
              f"{np.nanmedian(g.width_ghcp.to_numpy(float)):10.2f} "
              f"{np.nanmedian(g.width_std.to_numpy(float)):10.2f} "
              f"{g.both_finite.mean():8.0%} {g.cov_ghcp.mean():8.3f} {g.cov_std.mean():8.3f}")


if __name__ == "__main__":
    main()
