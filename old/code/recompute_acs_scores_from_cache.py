"""
Recompute GHCP / Std-CP intervals from ACS v2 replicate caches by swapping score types.

Caches store frozen μ (and split indices / S_comp points) so absolute ↔ studentized
can be compared without refitting global RF-μ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scores import conformal_threshold


def _fit_local_sigma(x_train, abs_train, x_cal, x_target, *, ntree=500, nodesize=5, rs=0):
    from sklearn.ensemble import RandomForestRegressor

    x_train = np.asarray(x_train, dtype=float)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    abs_train = np.asarray(abs_train, dtype=float)
    min_leaf = max(1, min(int(nodesize), len(abs_train)))
    rf_s = RandomForestRegressor(
        n_estimators=int(ntree),
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        random_state=int(rs) + 17,
        n_jobs=1,
    )
    rf_s.fit(x_train, abs_train)
    x_cal = np.asarray(x_cal, dtype=float)
    if x_cal.ndim == 1:
        x_cal = x_cal.reshape(-1, 1)
    xt = np.asarray(x_target, dtype=float).reshape(1, -1)
    sigma_cal = np.maximum(rf_s.predict(x_cal), 1e-8)
    sigma_t = float(max(rf_s.predict(xt)[0], 1e-8))
    return sigma_cal, sigma_t


def replay_stdcp(inter: dict, *, score_type: str, alpha: float,
                 quantile_mode: str = "deterministic",
                 quantile_random_seed: int | None = None, rng=None) -> tuple[float, float]:
    st = str(score_type).lower()
    if inter is None or inter.get("center_mode") == "global":
        # Global-center path: need mu_hist; studentized needs stored/global σ (not in local cache)
        if inter is None:
            return (-np.inf, np.inf)
        y = np.asarray(inter["y_hist"], dtype=float)
        mu = np.asarray(inter["mu_hist"], dtype=float)
        mu_t = float(inter["mu_target"])
        if st == "studentized":
            raise ValueError("Global-center studentized replay needs score_aux; re-run with local center.")
        scores = np.abs(y - mu)
        if len(scores) == 0:
            return (-np.inf, np.inf)
        q = conformal_threshold(
            scores=scores, weights=None, alpha=alpha,
            quantile_mode=quantile_mode, random_seed=quantile_random_seed, rng=rng,
        )
        if not np.isfinite(q):
            return (-np.inf, np.inf)
        return (mu_t - float(q), mu_t + float(q))

    cal_idx = np.asarray(inter["cal_idx"], dtype=int)
    train_idx = np.asarray(inter["train_idx"], dtype=int)
    y_hist = np.asarray(inter["y_hist"], dtype=float)
    mu_cal = np.asarray(inter["mu_cal"], dtype=float)
    mu_t = float(inter["mu_target"])
    y_cal = y_hist[cal_idx]
    abs_cal = np.abs(y_cal - mu_cal)

    if st == "studentized":
        sigma_cal = inter.get("sigma_cal")
        sigma_t = inter.get("sigma_target")
        if sigma_cal is None or sigma_t is None:
            x_hist = np.asarray(inter["x_hist"], dtype=float)
            sigma_cal, sigma_t = _fit_local_sigma(
                x_hist[train_idx],
                inter["abs_train"],
                x_hist[cal_idx],
                inter["x_target"],
            )
        scores = abs_cal / np.maximum(np.asarray(sigma_cal, dtype=float), 1e-8)
        q = conformal_threshold(
            scores=scores, weights=None, alpha=alpha,
            quantile_mode=quantile_mode, random_seed=quantile_random_seed, rng=rng,
        )
        if not np.isfinite(q):
            return (-np.inf, np.inf)
        return (mu_t - float(q) * float(sigma_t), mu_t + float(q) * float(sigma_t))

    q = conformal_threshold(
        scores=abs_cal, weights=None, alpha=alpha,
        quantile_mode=quantile_mode, random_seed=quantile_random_seed, rng=rng,
    )
    if not np.isfinite(q):
        return (-np.inf, np.inf)
    return (mu_t - float(q), mu_t + float(q))


def replay_ghcp(inter: dict, *, score_type: str, alpha: float,
                quantile_mode: str = "deterministic",
                quantile_random_seed: int | None = None, rng=None) -> tuple[float, float]:
    if inter is None:
        return (-np.inf, np.inf)
    st = str(score_type).lower()
    meta = inter["scores_meta"]
    target = inter["target"]
    mu_t = float(target["mu"])

    if st == "studentized":
        # Prefer stored σ; else fit RF-σ on S_comp |y - mu_g|
        need_fit = any(m.get("sigma") is None for m in meta if not m.get("is_inf")) or target.get("sigma") is None
        sigma_by_x = None
        if need_fit:
            pts = inter.get("s_comp_points") or []
            if len(pts) < 2:
                raise ValueError("Cannot build studentized GHCP: missing s_comp_points for σ.")
            X = np.vstack([np.asarray(p["x"], dtype=float).ravel() for p in pts])
            abs_r = np.array([abs(float(p["y"]) - float(p["mu_g"])) for p in pts], dtype=float)
            from sklearn.ensemble import RandomForestRegressor
            rf_s = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                random_state=0, n_jobs=1,
            )
            rf_s.fit(X, abs_r)

            def _sig(x, w_g=1.0):
                return float(max(rf_s.predict(np.asarray(x, dtype=float).reshape(1, -1))[0], 1e-8))

            sigma_by_x = _sig

        values, weights = [], []
        for m in meta:
            w = float(m["weight"])
            if m.get("is_inf"):
                values.append(np.inf)
                weights.append(w)
                continue
            sig = m.get("sigma")
            if sig is None:
                sig = sigma_by_x(m["x"], m.get("w_g", 1.0))
            values.append(abs(float(m["y"]) - float(m["mu"])) / max(float(sig), 1e-8))
            weights.append(w)
        q = conformal_threshold(
            scores=values, weights=weights, alpha=alpha,
            quantile_mode=quantile_mode, random_seed=quantile_random_seed, rng=rng,
        )
        if not np.isfinite(q):
            return (-np.inf, np.inf)
        sig_t = target.get("sigma")
        if sig_t is None:
            sig_t = sigma_by_x(target["x"], target.get("w_g", 1.0))
        return (mu_t - float(q) * float(sig_t), mu_t + float(q) * float(sig_t))

    values, weights = [], []
    for m in meta:
        w = float(m["weight"])
        if m.get("is_inf"):
            values.append(np.inf)
            weights.append(w)
            continue
        values.append(abs(float(m["y"]) - float(m["mu"])))
        weights.append(w)
    q = conformal_threshold(
        scores=values, weights=weights, alpha=alpha,
        quantile_mode=quantile_mode, random_seed=quantile_random_seed, rng=rng,
    )
    if not np.isfinite(q):
        return (-np.inf, np.inf)
    return (mu_t - float(q), mu_t + float(q))


def _covered(interval, y):
    lo, hi = interval
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    return float(lo <= float(y) <= hi)


def _width(interval):
    lo, hi = interval
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    return float(hi - lo)


def summarize_cache_dir(
    cache_dir: Path,
    *,
    ghcp_score: str,
    stdcp_score: str,
    B: int | None = None,
) -> dict:
    import pickle

    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    alpha = float(cfg["alpha"])
    o_values = list(cfg["o_values"])
    quantile_mode = cfg.get("quantile_mode", "deterministic")
    if B is None:
        B = int(manifest["B"])

    rows = []
    for b in range(B):
        p = cache_dir / f"rep_{b:05d}.pkl"
        if not p.exists():
            continue
        with p.open("rb") as f:
            cache = pickle.load(f)
        if int(cache.get("version", 1)) < 2 or "by_o" not in cache:
            raise ValueError(f"{p} is not a v2 score-swap cache (missing by_o).")
        y_true = float(cache["true_y"])
        for o, o_inter in cache["by_o"].items():
            o = int(o)
            dhcp_i = o_inter.get("donor_hcp")
            std_i = o_inter.get("stdcp")
            int_g = replay_ghcp(dhcp_i, score_type=ghcp_score, alpha=alpha, quantile_mode=quantile_mode)
            int_s = (
                (-np.inf, np.inf) if o == 0
                else replay_stdcp(std_i, score_type=stdcp_score, alpha=alpha, quantile_mode=quantile_mode)
            )
            rows.append({
                "replicate": b, "o": o,
                "ghcp_covered": _covered(int_g, y_true),
                "ghcp_width": _width(int_g),
                "stdcp_covered": _covered(int_s, y_true),
                "stdcp_width": _width(int_s),
            })
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("o")
        .agg(
            ghcp_coverage=("ghcp_covered", "mean"),
            ghcp_width_mean=("ghcp_width", "mean"),
            ghcp_width_median=("ghcp_width", "median"),
            stdcp_coverage=("stdcp_covered", "mean"),
            stdcp_width_mean=("stdcp_width", "mean"),
            stdcp_width_median=("stdcp_width", "median"),
            n=("replicate", "count"),
        )
        .reset_index()
    )
    return {"detail": df, "summary": summary, "ghcp_score": ghcp_score, "stdcp_score": stdcp_score}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--ghcp_score", choices=("absolute", "studentized"), default="absolute")
    p.add_argument("--stdcp_score", choices=("absolute", "studentized"), default="studentized")
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()
    out = summarize_cache_dir(
        Path(args.cache_dir),
        ghcp_score=args.ghcp_score,
        stdcp_score=args.stdcp_score,
        B=args.B,
    )
    print(out["summary"].to_string(index=False))
    if args.out_csv:
        out["summary"].to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
