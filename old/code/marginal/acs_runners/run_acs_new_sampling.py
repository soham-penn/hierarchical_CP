#!/usr/bin/env python3
"""
ACS new sampling plan → plots_marginal/acs/new_sampling/

Cohort: CA, YOEP >= x (default 2010), no age/hours filters.
  Chosen so PUMA sizes are roughly 10–30 with majority > 20 and ≥30 PUMAs with N≥21.

Sampling (each of B replicates):
  1. Draw 20 reference PUMAs uniformly without replacement from *all* PUMAs (any size).
  2. Draw 1 test PUMA from the unselected PUMAs with size ≥ 21.
  3. For o ∈ {0,5,10,15,20}, predict the 21st observation (target index 20);
     record coverage and width (GHCP / baselines via run_acs_experiments).

Can wait for another runner PID to exit before starting.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "real_data"))

from acs.data_processing import build_design_matrix_acs, load_and_clean_acs_pums  # noqa: E402

SUITE_ROOT = REPO_ROOT / "plots_marginal" / "acs" / "new_sampling"
DATA_DIR = SUITE_ROOT / "data"
DEFAULT_ACS = REPO_ROOT / "real_data" / "acs" / "data" / "acs_data_all50states.csv"
# YOEP>=2010: median N≈23, majority >20, ~149 PUMAs with N≥21 (see cohort scan).
DEFAULT_YOEP = 2010


def build_cohort_csv(acs_csv: Path, yoep_min_year: int, state: str = "CA") -> tuple[Path, dict]:
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
        y_transform=lambda x: np.asarray(x, dtype=float),
    )
    df = df.dropna(subset=["puma"]).copy()
    df["puma"] = df["puma"].astype(int)
    X = build_design_matrix_acs(df, exclude_entry_recency=True, exclude_cow=False)
    # attach design cols
    # rebuild named frame
    educ = pd.get_dummies(df["educ_level"], prefix="educ", drop_first=True)
    eng = pd.get_dummies(df["english"], prefix="eng", drop_first=True)
    cow = pd.get_dummies(df["cow"], prefix="cow", drop_first=True)
    cont = df[["age", "age_sq", "hours", "married", "female"]]
    Xdf = pd.concat([cont, educ, eng, cow], axis=1)
    assert Xdf.shape == X.shape

    out = df.copy()
    for c in Xdf.columns:
        out[f"x_{c}"] = Xdf[c].to_numpy()

    N = out.groupby("puma").size()
    meta = {
        "yoep_min_year": yoep_min_year,
        "state": state,
        "filters": "YOEP only (no age/hours); foreign-born implied by YOEP",
        "n_rows": int(len(out)),
        "n_pumas": int(len(N)),
        "N_min": int(N.min()),
        "N_q25": float(N.quantile(0.25)),
        "N_median": float(N.median()),
        "N_q75": float(N.quantile(0.75)),
        "N_max": int(N.max()),
        "N_mean": float(N.mean()),
        "frac_N_in_10_30": float(((N >= 10) & (N <= 30)).mean()),
        "frac_N_gt_20": float((N > 20).mean()),
        "n_puma_ge21": int((N >= 21).sum()),
        "x_features": [str(c) for c in Xdf.columns],
    }
    csv_path = DATA_DIR / f"acs_{state.lower()}_yoep{yoep_min_year}_new_sampling.csv"
    meta_path = DATA_DIR / f"acs_{state.lower()}_yoep{yoep_min_year}_new_sampling_meta.json"
    sizes_path = DATA_DIR / "puma_sizes.csv"
    out.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    N.rename("N").reset_index().to_csv(sizes_path, index=False)
    print(json.dumps(meta, indent=2))
    print(f"Wrote {csv_path}")
    return csv_path, meta


def wait_for_pid(pid: int, poll_s: float = 30.0) -> None:
    print(f"Waiting for PID {pid} to finish...", flush=True)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"PID {pid} exited.", flush=True)
            return
        except PermissionError:
            # process exists
            pass
        time.sleep(poll_s)


def run_alphas(
    *,
    yoep_min_year: int,
    B: int,
    n_workers: int,
    alphas: list[float],
) -> None:
    results_root = SUITE_ROOT / "results"
    logs = SUITE_ROOT / "logs"
    results_root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    for alpha in alphas:
        tag = f"alpha{int(round(alpha * 100)):02d}"
        cmd = [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-u",
            str(REPO_ROOT / "code" / "marginal" / "run_acs_experiments.py"),
            "--predictor", "rf",
            "--outcome_scale", "income",
            "--within_group_mode", "mean",
            "--min_income", "0",
            "--drop_top_income_pct", "0",
            "--yoep_min_year", str(yoep_min_year),
            "--no_age_filter",
            "--no_hours_filter",
            "--no-exclude_cow",
            "--exclude_entry_recency",
            "--design", "mixed_size_one_target",
            "--no_permute_rows",
            "--min_puma_size", "1",  # calib pool = all PUMAs
            "--target_index", "20",  # 21st observation
            "--o_values", "0,5,10,15,20",
            "--alpha", str(alpha),
            "--B", str(B),
            "--n_workers", str(n_workers),
            "--score_type", "absolute",
            "--stdcp_score_type", "studentized",
            "--cache_dir", str(SUITE_ROOT / "caches" / tag),
        ]
        log = logs / f"run_{tag}.log"
        print(f"=== new_sampling {tag} ===", flush=True)
        print(" ".join(cmd), flush=True)
        with log.open("w") as f:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
        if rc.returncode != 0:
            raise RuntimeError(f"Failed {tag}; see {log}")

        # Result dir: GHCP absolute + Std-CP studentized → _stdcpstud_
        src = (
            REPO_ROOT
            / "plots_marginal"
            / "acs"
            / "rf"
            / "results"
            / f"true_marginal_rf_income_notrim_yoep{yoep_min_year}_stdcpstud_{tag}"
        )
        if not src.is_dir():
            cands = sorted(
                (REPO_ROOT / "plots_marginal" / "acs" / "rf" / "results").glob(f"*{tag}*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            src = cands[0] if cands else src
        dest = results_root / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        cache_src = SUITE_ROOT / "caches" / tag
        if cache_src.is_dir():
            cache_dst = dest / "caches"
            if cache_dst.exists():
                shutil.rmtree(cache_dst)
            shutil.move(str(cache_src), str(cache_dst))
            print(f"Moved caches -> {cache_dst}", flush=True)
        print(f"Moved {src} -> {dest}", flush=True)


def plot_results() -> None:
    # Register suite inline via env-less call: write a tiny plot invoke using existing machinery
    plot_py = REPO_ROOT / "real_data" / "acs" / "plot_dgp_style_paper_plots.py"
    # Ensure suite exists; if not registered, call a local plot helper
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(plot_py),
        "--predictor",
        "rf_new_sampling",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--acs_csv", type=str, default=str(DEFAULT_ACS))
    p.add_argument("--yoep_min_year", type=int, default=DEFAULT_YOEP)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--n_workers", type=int, default=6)
    p.add_argument("--alphas", type=str, default="0.1,0.2")
    p.add_argument("--wait_pid", type=int, default=None, help="Wait for this PID before starting")
    p.add_argument("--wait_pid_file", type=str, default=None)
    p.add_argument("--data_only", action="store_true")
    p.add_argument("--skip_data", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    (SUITE_ROOT / "figures").mkdir(exist_ok=True)
    (SUITE_ROOT / "summaries").mkdir(exist_ok=True)

    wait_pid = args.wait_pid
    if args.wait_pid_file:
        wait_pid = int(Path(args.wait_pid_file).read_text().strip())
    if wait_pid is not None:
        wait_for_pid(wait_pid)

    if not args.skip_data:
        _, meta = build_cohort_csv(Path(args.acs_csv), args.yoep_min_year)
        if meta["n_puma_ge21"] < 30:
            raise RuntimeError(
                f"Need ≥30 PUMAs with N≥21; got {meta['n_puma_ge21']} for YOEP>={args.yoep_min_year}"
            )
    if args.data_only:
        return

    alphas = [float(x) for x in args.alphas.split(",")]
    run_alphas(
        yoep_min_year=args.yoep_min_year,
        B=args.B,
        n_workers=args.n_workers,
        alphas=alphas,
    )
    if args.plot:
        plot_results()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
