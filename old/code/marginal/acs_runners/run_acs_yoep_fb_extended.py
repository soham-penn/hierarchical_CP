#!/usr/bin/env python3
"""
ACS YOEP + foreign-born only cohort (no age / hours filters).

1) Build extended person-level CSV (CA, foreign-born, YOEP cutoff).
2) Run GHCP suites:
   - PUMA size ≥ 31, target index 30, o ∈ {0,10,20,30}
   - PUMA size ≥ 21, target index 20, o ∈ {0,5,10,15,20}
3) Optionally plot both suites.

X predictors: age, age², hours, married, female, education, English, COW
(exclude YOEP / entry_recency, which define the cohort filter).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "real_data"))

from acs.data_processing import build_design_matrix_acs, load_and_clean_acs_pums  # noqa: E402

SUITE_ROOT = REPO_ROOT / "plots_marginal" / "acs" / "yoep_fb_extended"
DATA_DIR = SUITE_ROOT / "data"
DEFAULT_ACS_CSV = REPO_ROOT / "real_data" / "acs" / "data" / "acs_data_all50states.csv"


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Person-level design matrix as a DataFrame (YOEP/entry_recency excluded)."""
    educ_dummies = pd.get_dummies(df["educ_level"], prefix="educ", drop_first=True)
    english_dummies = pd.get_dummies(df["english"], prefix="eng", drop_first=True)
    cow_dummies = pd.get_dummies(df["cow"], prefix="cow", drop_first=True)
    continuous = df[["age", "age_sq", "hours", "married", "female"]].copy()
    X = pd.concat([continuous, educ_dummies, english_dummies, cow_dummies], axis=1)
    X.columns = [str(c) for c in X.columns]
    return X


def build_and_save_extended_csv(
    *,
    acs_csv: Path,
    yoep_min_year: int,
    state: str,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_acs_pums(
        str(acs_csv),
        states_keep=[state],
        age_min=None,
        age_max=None,
        yoep_min_year=yoep_min_year,
        min_hours=None,
        min_income=None,
        drop_top_income_fraction=None,
        y_transform=lambda x: np.asarray(x, dtype=float),  # raw income in y
    )
    df = df.dropna(subset=["puma"]).copy()
    df["puma"] = df["puma"].astype(int)

    X = _feature_frame(df)
    # Sanity: same construction as experiment design matrix with cow on, yoep off
    X_np = build_design_matrix_acs(df, exclude_entry_recency=True, exclude_cow=False)
    if X.shape != X_np.shape:
        raise RuntimeError(f"Design mismatch: frame {X.shape} vs ndarray {X_np.shape}")

    out = df.copy()
    for col in X.columns:
        out[f"x_{col}"] = X[col].to_numpy()

    out_path = DATA_DIR / f"acs_{state.lower()}_yoep{yoep_min_year}_fb_extended.csv"
    out.to_csv(out_path, index=False)

    meta = {
        "n_rows": int(len(out)),
        "n_pumas": int(out["puma"].nunique()),
        "yoep_min_year": int(yoep_min_year),
        "state": state,
        "filters": "foreign-born + YOEP only (no age / hours filter)",
        "x_features": list(X.columns),
        "puma_size_min": int(out.groupby("puma").size().min()),
        "puma_size_median": float(out.groupby("puma").size().median()),
        "puma_size_max": int(out.groupby("puma").size().max()),
        "n_puma_ge21": int((out.groupby("puma").size() >= 21).sum()),
        "n_puma_ge31": int((out.groupby("puma").size() >= 31).sum()),
    }
    meta_path = DATA_DIR / f"acs_{state.lower()}_yoep{yoep_min_year}_fb_extended_meta.json"
    import json

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path} ({meta['n_rows']} rows, {meta['n_pumas']} PUMAs)")
    print(f"  size≥21: {meta['n_puma_ge21']}  size≥31: {meta['n_puma_ge31']}")
    print(f"  X features ({len(meta['x_features'])}): {meta['x_features']}")
    return out_path


def _run_one(
    *,
    alpha: float,
    min_puma_size: int,
    target_index: int,
    o_values: str,
    dest_subdir: str,
    B: int,
    n_workers: int,
    yoep_min_year: int,
) -> Path:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    dest = SUITE_ROOT / dest_subdir
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results").mkdir(exist_ok=True)
    (dest / "logs").mkdir(exist_ok=True)

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-u",
        str(REPO_ROOT / "code" / "marginal" / "run_acs_experiments.py"),
        "--predictor",
        "rf",
        "--outcome_scale",
        "income",
        "--within_group_mode",
        "mean",
        "--min_income",
        "0",
        "--drop_top_income_pct",
        "0",
        "--yoep_min_year",
        str(yoep_min_year),
        "--no_age_filter",
        "--no_hours_filter",
        "--no-exclude_cow",  # include COW in X
        "--exclude_entry_recency",  # drop YOEP-derived predictor
        "--design",
        "uniform_one_target",
        "--no_permute_rows",
        "--min_puma_size",
        str(min_puma_size),
        "--target_index",
        str(target_index),
        "--o_values",
        o_values,
        "--alpha",
        str(alpha),
        "--B",
        str(B),
        "--n_workers",
        str(n_workers),
        # Absolute residual |Y-μ| for GHCP and Std-CP (local RF Std-CP)
        "--score_type",
        "absolute",
        "--stdcp_score_type",
        "absolute",
    ]
    log = dest / "logs" / f"run_{tag}.log"
    print(f"\n=== {dest_subdir} {tag} ===", flush=True)
    print(" ".join(cmd), flush=True)
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed ({proc.returncode}); see {log}")

    # Default write path under plots_marginal/acs/rf/results/
    src_name = f"true_marginal_rf_income_notrim_t{target_index}_yoep{yoep_min_year}_{tag}"
    if target_index == 20:
        src_name = f"true_marginal_rf_income_notrim_yoep{yoep_min_year}_{tag}"
    src = REPO_ROOT / "plots_marginal" / "acs" / "rf" / "results" / src_name
    if not src.is_dir():
        # fallback: newest matching (exclude studentized tags)
        cand = sorted(
            (
                p
                for p in (REPO_ROOT / "plots_marginal" / "acs" / "rf" / "results").glob(
                    f"*{tag}*"
                )
                if "stdcpstud" not in p.name and "studentized" not in p.name
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        src = cand[0] if cand else src
    if not src.is_dir():
        raise FileNotFoundError(f"Expected results at {src}")
    out = dest / "results" / src.name
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(src), str(out))
    print(f"Moved {src} -> {out}", flush=True)
    return out


def plot_suite(dest_subdir: str, predictor_flag: str) -> None:
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "real_data" / "acs" / "plot_dgp_style_paper_plots.py"),
        "--predictor",
        predictor_flag,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acs_csv", type=str, default=str(DEFAULT_ACS_CSV))
    p.add_argument("--yoep_min_year", type=int, default=2000)
    p.add_argument("--acs_state", type=str, default="CA")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--n_workers", type=int, default=6)
    p.add_argument("--alphas", type=str, default="0.1,0.2")
    p.add_argument("--data_only", action="store_true", help="Only build/save extended CSV")
    p.add_argument("--skip_data", action="store_true", help="Skip CSV build")
    p.add_argument("--skip_min31", action="store_true")
    p.add_argument("--skip_min21", action="store_true")
    p.add_argument("--plot", action="store_true", help="Generate paper-style plots after runs")
    args = p.parse_args()

    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.skip_data:
        build_and_save_extended_csv(
            acs_csv=Path(args.acs_csv),
            yoep_min_year=args.yoep_min_year,
            state=args.acs_state,
        )
    if args.data_only:
        return

    alphas = [float(x) for x in args.alphas.split(",")]
    for alpha in alphas:
        if not args.skip_min31:
            _run_one(
                alpha=alpha,
                min_puma_size=31,
                target_index=30,
                o_values="0,10,20,30",
                dest_subdir="min31",
                B=args.B,
                n_workers=args.n_workers,
                yoep_min_year=args.yoep_min_year,
            )
        if not args.skip_min21:
            _run_one(
                alpha=alpha,
                min_puma_size=21,
                target_index=20,
                o_values="0,5,10,15,20",
                dest_subdir="min21",
                B=args.B,
                n_workers=args.n_workers,
                yoep_min_year=args.yoep_min_year,
            )

    if args.plot:
        # Plot suites registered in plot_dgp_style_paper_plots.py
        if not args.skip_min31:
            plot_suite("min31", "rf_yoep_fb_min31")
        if not args.skip_min21:
            plot_suite("min21", "rf_yoep_fb_min21")


if __name__ == "__main__":
    main()
