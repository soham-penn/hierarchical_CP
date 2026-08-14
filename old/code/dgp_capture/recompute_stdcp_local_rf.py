#!/usr/bin/env python3
"""Recompute DGP Std-CP with local RF from capture observations (no GHCP re-run)."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.shared.dgp.experiments import _compute_std_cp_interval
from scores import make_quantile_seed

O_VALUES = [0, 5, 10, 15, 20, 25, 30, 35]
TARGET_INDEX = 35
QUANTILE_BASE_SEED = 457


def _x_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("x_")]


def _one_experiment(args):
    experiment, alpha, test_df, rf_kw = args
    xcols = _x_cols(test_df)
    test_df = test_df.sort_values("row_idx")
    X = test_df[xcols].to_numpy(dtype=float)
    y = test_df["y"].to_numpy(dtype=float)
    y_true = float(y[TARGET_INDEX])
    x_target = X[TARGET_INDEX]
    rows = []
    for o in O_VALUES:
        if o < 2:
            lo, hi = (-np.inf, np.inf)
        else:
            split_seed = make_quantile_seed(
                QUANTILE_BASE_SEED, experiment, TARGET_INDEX, o, "stdcp_split"
            )
            lo, hi = _compute_std_cp_interval(
                x_hist=X[:o],
                y_hist=y[:o],
                x_target=x_target,
                alpha=alpha,
                quantile_mode="deterministic",
                random_seed=make_quantile_seed(
                    QUANTILE_BASE_SEED, experiment, TARGET_INDEX, o, "stdcp"
                ),
                rng=np.random.default_rng(split_seed),
                **rf_kw,
            )
        covered = bool(lo <= y_true <= hi)
        width = float(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
        rows.append({
            "experiment": experiment,
            "alpha": alpha,
            "o": o,
            "coverage": float(covered),
            "width": width,
            "finite": int(np.isfinite(width)),
        })
    return rows


def recompute(alpha: float, capture_dir: Path, rf_kw: dict, n_workers: int) -> pd.DataFrame:
    obs = pd.read_csv(capture_dir / "observations.csv")
    test = obs[obs["role"] == "test"]
    tasks = [
        (int(exp), float(alpha), g, rf_kw)
        for exp, g in test.groupby("experiment")
    ]
    print(f"alpha={alpha}: {len(tasks)} experiments, workers={n_workers}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_one_experiment, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % 200 == 0 or i == len(futs):
                print(f"  done {i}/{len(futs)}", flush=True)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--plots-root",
        type=Path,
        default=REPO_ROOT / "plots_marginal" / "dgp_true_marginal_strong_rf",
    )
    p.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.05])
    p.add_argument("--ntree", type=int, default=100)
    p.add_argument("--nodesize", type=int, default=5)
    p.add_argument("--mtry", type=float, default=0.6)
    p.add_argument("--rf-random-state", type=int, default=123)
    p.add_argument("--n-workers", type=int, default=8)
    args = p.parse_args()

    alpha_tag = {
        0.1: "alpha10",
        0.05: "alpha05",
        0.075: "alpha07p5",
        0.125: "alpha12p5",
        0.15: "alpha15",
    }
    rf_kw = dict(
        ntree=args.ntree,
        nodesize=args.nodesize,
        mtry=args.mtry,
        rf_random_state=args.rf_random_state,
    )

    parts = []
    for alpha in args.alphas:
        tag = alpha_tag[float(alpha)]
        capture_dir = args.plots_root / "capture_data" / f"fixedN21_{tag}"
        parts.append(recompute(alpha, capture_dir, rf_kw, args.n_workers))

    std = pd.concat(parts, ignore_index=True)
    out_csv = args.plots_root / "summaries" / "stdcp_recomputed_local_rf.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    std.to_csv(out_csv, index=False)
    print("Wrote", out_csv, flush=True)

    for alpha in args.alphas:
        print(f"\nalpha={alpha}")
        print(f'{"o":>3}  {"cov":>7} {"finite":>8} {"med":>8} {"iqr":>8}')
        for o in O_VALUES:
            g = std[(np.isclose(std.alpha, alpha)) & (std.o == o)]
            w = g.width.to_numpy(float)
            fin = np.isfinite(w)
            med = float(np.nanmedian(w[fin])) if fin.any() else float("nan")
            iqr = (
                float(np.nanpercentile(w[fin], 75) - np.nanpercentile(w[fin], 25))
                if fin.any() else float("nan")
            )
            print(f"{o:3d}  {g.coverage.mean():7.3f} {fin.mean():8.1%} {med:8.2f} {iqr:8.2f}")


if __name__ == "__main__":
    main()
