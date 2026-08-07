#!/usr/bin/env python3
"""
Std-CP only, studentized scores, same group draws as YOEP+FB min21 suite.

Reuses selection seeds from run_acs_experiments (seed + replicate * 1009).
Does not re-run GHCP/HCP. Writes parallel detailed CSVs + a comparison summary
of width spread vs absolute Std-CP from the finished suite.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "real_data"))

from acs.data_processing import build_design_matrix_acs, load_and_clean_acs_pums
from code.marginal import run_acs_experiments as acs
from scores import make_quantile_seed

SUITE_ROOT = REPO_ROOT / "plots_marginal" / "acs" / "yoep_fb_extended" / "min21"
OUT_ROOT = SUITE_ROOT / "stdcp_studentized"
SEED = 456
QUANTILE_BASE_SEED = 456
O_VALUES = [0, 5, 10, 15, 20]
N_CALIB = 20
TARGET_INDEX = 20
B_DEFAULT = 1000
YOEP_MIN = 2000
OUTCOME_SCALE = "income"

# Shared cohort loaded once in workers via initializer
_DF = None
_X = None
_ELIGIBLE = None
_GROUP_COUNTS = None


def _init_worker(df, X, eligible, group_counts):
    global _DF, _X, _ELIGIBLE, _GROUP_COUNTS
    _DF = df
    _X = X
    _ELIGIBLE = eligible
    _GROUP_COUNTS = group_counts
    acs._set_target_index(TARGET_INDEX)


def _load_cohort():
    acs._set_target_index(TARGET_INDEX)
    acs_csv = REPO_ROOT / "real_data" / "acs" / "data" / "acs_data_all50states.csv"
    df = load_and_clean_acs_pums(
        str(acs_csv),
        states_keep=["CA"],
        age_min=None,
        age_max=None,
        yoep_min_year=YOEP_MIN,
        min_hours=None,
        min_income=None,
        drop_top_income_fraction=None,
        y_transform=acs._outcome_transform(OUTCOME_SCALE),
    )
    df = df.dropna(subset=["puma"]).copy()
    df["puma"] = df["puma"].astype(int)
    # Match suite: include COW, exclude YOEP/entry_recency
    X = build_design_matrix_acs(df, exclude_entry_recency=True, exclude_cow=False)
    counts = df.groupby("puma").size()
    eligible = counts[counts >= TARGET_INDEX + 1].index.to_numpy().tolist()
    return df, X, eligible, counts


def _stdcp_one_replicate(args):
    """Studentized Std-CP for one replicate; same PUMA draw as original suite."""
    replicate_idx, alpha = args
    df, X, eligible, group_counts = _DF, _X, _ELIGIBLE, _GROUP_COUNTS

    selection_seed = SEED + replicate_idx * 1009
    calib_groups, test_group = acs.sample_calibration_and_target_uniform(
        eligible_groups=eligible,
        group_counts=group_counts,
        selection_seed=selection_seed,
        n_calib_groups=N_CALIB,
        min_target_size=acs.MIN_TARGET_PUMA_SIZE,
    )
    if calib_groups is None or test_group is None:
        return None

    # Match --no_permute_rows: fixed ACS row order within PUMA
    grp_mask = (df["puma"] == test_group).values
    grp_idx = np.where(grp_mask)[0]
    Xg = X[grp_idx]
    Yg = df.iloc[grp_idx]["y"].to_numpy(dtype=float)
    income = df.iloc[grp_idx]["income"].to_numpy(dtype=float) if "income" in df.columns else Yg

    if len(Yg) <= TARGET_INDEX:
        return None

    x_target = Xg[TARGET_INDEX]
    true_y = float(Yg[TARGET_INDEX])
    income_target = float(income[TARGET_INDEX])
    rows = []
    for o in O_VALUES:
        if len(Yg) <= max(o, TARGET_INDEX) or o <= 0:
            rec = acs._result_record((-np.inf, np.inf), true_y, outcome_scale=OUTCOME_SCALE)
        else:
            stdcp_rng = np.random.default_rng(
                make_quantile_seed(
                    QUANTILE_BASE_SEED, replicate_idx, TARGET_INDEX, o, "stdcp_split"
                )
            )
            interval = acs._compute_std_cp_interval(
                x_hist=Xg[:o],
                y_hist=Yg[:o],
                x_target=x_target,
                alpha=alpha,
                rng=stdcp_rng,
                quantile_mode="deterministic",
                quantile_random_seed=make_quantile_seed(
                    QUANTILE_BASE_SEED, replicate_idx, TARGET_INDEX, o, "stdcp"
                ),
                score_type="studentized",
            )
            rec = acs._result_record(interval, true_y, outcome_scale=OUTCOME_SCALE)
        rows.append(
            {
                "replicate": replicate_idx,
                "method": "Std-CP",
                "o": o,
                "income_target": income_target,
                "score_type": "studentized",
                **rec,
            }
        )
    return rows


def _summarize(detailed: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for o, g in detailed.groupby("o"):
        cov = g["coverage"].dropna()
        w = g["width"].replace([np.inf, -np.inf], np.nan)
        w_fin = w.dropna()
        wi = g["width_income"].replace([np.inf, -np.inf], np.nan) if "width_income" in g else w
        wi_fin = wi.dropna()
        n_inf = int(np.isinf(g["width"].to_numpy(dtype=float)).sum()) if len(g) else 0
        rows.append(
            {
                "alpha": alpha,
                "method": "Std-CP-studentized",
                "o": int(o),
                "coverage_mean": float(cov.mean()) if len(cov) else np.nan,
                "coverage_se": float(cov.std(ddof=1) / np.sqrt(len(cov))) if len(cov) > 1 else np.nan,
                "width_mean": float(w_fin.mean()) if len(w_fin) else np.nan,
                "width_se": float(w_fin.std(ddof=1) / np.sqrt(len(w_fin))) if len(w_fin) > 1 else np.nan,
                "width_std": float(w_fin.std(ddof=1)) if len(w_fin) > 1 else np.nan,
                "width_median": float(w_fin.median()) if len(w_fin) else np.nan,
                "width_q25": float(w_fin.quantile(0.25)) if len(w_fin) else np.nan,
                "width_q75": float(w_fin.quantile(0.75)) if len(w_fin) else np.nan,
                "width_iqr": float(w_fin.quantile(0.75) - w_fin.quantile(0.25)) if len(w_fin) else np.nan,
                "width_income_mean": float(wi_fin.mean()) if len(wi_fin) else np.nan,
                "width_income_std": float(wi_fin.std(ddof=1)) if len(wi_fin) > 1 else np.nan,
                "width_income_median": float(wi_fin.median()) if len(wi_fin) else np.nan,
                "width_income_iqr": (
                    float(wi_fin.quantile(0.75) - wi_fin.quantile(0.25)) if len(wi_fin) else np.nan
                ),
                "width_n_finite": int(len(w_fin)),
                "width_n_infinite": n_inf,
            }
        )
    return pd.DataFrame(rows).sort_values("o")


def _absolute_summary_from_suite(alpha: float) -> pd.DataFrame | None:
    """Build absolute Std-CP summary (with IQR/std) from the finished suite detailed CSV."""
    tag = f"alpha{int(round(alpha * 100)):02d}"
    candidates = [
        SUITE_ROOT
        / "results"
        / f"true_marginal_rf_income_notrim_yoep{YOEP_MIN}_{tag}"
        / f"acs_true_marg_{tag}_detailed.csv",
        REPO_ROOT
        / "plots_marginal"
        / "acs"
        / "rf"
        / "results"
        / f"true_marginal_rf_income_notrim_yoep{YOEP_MIN}_{tag}"
        / f"acs_true_marg_{tag}_detailed.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    d = pd.read_csv(path)
    d = d[d["method"] == "Std-CP"].copy()
    if d.empty:
        return None
    d["score_type"] = "absolute"
    return _summarize(d, alpha).assign(method="Std-CP-absolute")


def run_alpha(alpha: float, B: int, n_workers: int, df, X, eligible, group_counts) -> Path:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Std-CP studentized {tag} B={B} workers={n_workers} ===", flush=True)

    rows = []
    worker_args = [(r, alpha) for r in range(B)]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(df, X, eligible, group_counts),
    ) as ex:
        futs = {ex.submit(_stdcp_one_replicate, a): a[0] for a in worker_args}
        done = 0
        for fut in as_completed(futs):
            part = fut.result()
            if part is None:
                raise RuntimeError(f"replicate {futs[fut]} failed")
            rows.extend(part)
            done += 1
            if done % 100 == 0 or done == B:
                print(f"  {tag}: {done}/{B}", flush=True)

    detailed = pd.DataFrame(rows).sort_values(["replicate", "o"]).reset_index(drop=True)
    detailed_path = out_dir / f"stdcp_studentized_{tag}_detailed.csv"
    detailed.to_csv(detailed_path, index=False)
    summary = _summarize(detailed, alpha)
    summary_path = out_dir / f"stdcp_studentized_{tag}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detailed_path}")
    print(f"Wrote {summary_path}")
    return summary_path


def compare_and_write(alphas: list[float]) -> Path:
    parts = []
    for alpha in alphas:
        tag = f"alpha{int(round(alpha * 100)):02d}"
        stud = pd.read_csv(OUT_ROOT / tag / f"stdcp_studentized_{tag}_summary.csv")
        abs_s = _absolute_summary_from_suite(alpha)
        parts.append(stud)
        if abs_s is not None:
            # Align columns of interest
            keep = [
                c
                for c in [
                    "alpha",
                    "method",
                    "o",
                    "coverage_mean",
                    "coverage_se",
                    "width_mean",
                    "width_se",
                    "width_std",
                    "width_median",
                    "width_q25",
                    "width_q75",
                    "width_iqr",
                    "width_n_finite",
                    "width_n_infinite",
                ]
                if c in abs_s.columns
            ]
            parts.append(abs_s[keep])
    out = pd.concat(parts, ignore_index=True)
    out_path = OUT_ROOT / "stdcp_absolute_vs_studentized_summary.csv"
    out.to_csv(out_path, index=False)

    # Compact comparison print
    print("\n=== Width spread: absolute vs studentized Std-CP ===")
    for alpha in alphas:
        print(f"\nα={alpha}")
        sub = out[np.isclose(out["alpha"], alpha)]
        for o in O_VALUES:
            rows = sub[sub["o"] == o]
            if rows.empty:
                continue
            print(f"  o={o}:")
            for _, r in rows.iterrows():
                med = r.get("width_median", np.nan)
                iqr = r.get("width_iqr", np.nan)
                std = r.get("width_std", np.nan)
                n_fin = r.get("width_n_finite", np.nan)
                cov = r.get("coverage_mean", np.nan)
                print(
                    f"    {r['method']}: cov={cov:.3f}  "
                    f"median={med:.0f}  IQR={iqr:.0f}  std={std:.0f}  "
                    f"n_finite={n_fin}"
                )
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", default="0.1,0.2")
    p.add_argument("--B", type=int, default=B_DEFAULT)
    p.add_argument("--n_workers", type=int, default=6)
    p.add_argument("--compare_only", action="store_true")
    args = p.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        compare_and_write(alphas)
        return

    print("Loading YOEP+FB cohort (same filters as min21 suite)...")
    df, X, eligible, group_counts = _load_cohort()
    print(f"  n={len(df)}, eligible PUMAs={len(eligible)}")

    for alpha in alphas:
        run_alpha(alpha, args.B, args.n_workers, df, X, eligible, group_counts)

    compare_and_write(alphas)


if __name__ == "__main__":
    main()
