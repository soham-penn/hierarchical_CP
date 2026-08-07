#!/usr/bin/env python3
"""
YOEP+FB min31 with row permutation, Std-CP skipped.

Same cohort/filters/seeds as yoep_fb_extended/min31, but:
  - permute_rows=True (default)
  - skip_stdcp=True (GHCP/HCP/baselines only)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = REPO_ROOT / "plots_marginal" / "acs" / "yoep_fb_extended" / "min31_permute"
SEED = 456
QUANTILE_BASE_SEED = 456


def _manifest(*, alphas: list[float], B: int, n_workers: int) -> dict:
    return {
        "suite": "yoep_fb_extended/min31_permute",
        "description": (
            "CA foreign-born YOEP>=2000, hours>=40 cohort filter, hours excluded from X; "
            "no age filter; uniform_one_target; row permutation ON; Std-CP skipped; COW excluded"
        ),
        "seed": SEED,
        "quantile_base_seed": QUANTILE_BASE_SEED,
        "quantile_mode": "deterministic",
        "permute_rows": True,
        "skip_stdcp": True,
        "design": "uniform_one_target",
        "min_puma_size": 31,
        "target_index": 30,
        "o_values": [0, 10, 20, 30],
        "n_puma_groups": 20,
        "B": B,
        "alphas": alphas,
        "n_workers": n_workers,
        "predictor": "rf",
        "within_group_mode": "mean",
        "score_type": "absolute",
        "yoep_min_year": 2000,
        "no_age_filter": True,
        "no_hours_filter": False,
        "min_hours": 40,
        "exclude_hours_from_predictors": True,
        "exclude_entry_recency": True,
        "include_cow": False,
        "selection_seed_formula": "seed + replicate_idx * 1009",
        "row_permutation_seed_formula": "seed + replicate_idx * 1009 + 811",
        "stdcp_split_seed_formula": (
            "make_quantile_seed(quantile_base_seed, replicate_idx, target_index, o, 'stdcp_split')"
        ),
        "stdcp_quantile_seed_formula": (
            "make_quantile_seed(quantile_base_seed, replicate_idx, target_index, o, 'stdcp')"
        ),
    }


def _run_one(*, alpha: float, B: int, n_workers: int) -> Path:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    (SUITE_ROOT / "results").mkdir(exist_ok=True)
    (SUITE_ROOT / "logs").mkdir(exist_ok=True)

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-u",
        str(REPO_ROOT / "code" / "marginal" / "run_acs_experiments.py"),
        "--predictor", "rf",
        "--outcome_scale", "income",
        "--within_group_mode", "mean",
        "--min_income", "0",
        "--drop_top_income_pct", "0",
        "--yoep_min_year", "2000",
        "--no_age_filter",
        "--exclude_hours",
        "--exclude_entry_recency",
        "--design", "uniform_one_target",
        "--skip_stdcp",
        "--min_puma_size", "31",
        "--target_index", "30",
        "--o_values", "0,10,20,30",
        "--alpha", str(alpha),
        "--B", str(B),
        "--n_workers", str(n_workers),
        "--score_type", "absolute",
        "--quantile-base-seed", str(QUANTILE_BASE_SEED),
    ]
    log = SUITE_ROOT / "logs" / f"run_{tag}.log"
    print(f"\n=== min31_permute {tag} ===", flush=True)
    print(" ".join(cmd), flush=True)
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed ({proc.returncode}); see {log}")

    src = (
        REPO_ROOT
        / "plots_marginal"
        / "acs"
        / "rf"
        / "results"
        / f"true_marginal_permuted_rf_income_notrim_t30_yoep2000_{tag}"
    )
    if not src.is_dir():
        raise FileNotFoundError(f"Expected results at {src}")
    dest_results = SUITE_ROOT / "results" / src.name
    if dest_results.exists():
        import shutil
        shutil.rmtree(dest_results)
    src.rename(dest_results)
    print(f"Moved {dest_results}", flush=True)
    return dest_results


def _plot(*, alphas: list[float]) -> None:
    from dataclasses import replace

    sys.path.insert(0, str(REPO_ROOT))
    from real_data.acs import plot_dgp_style_paper_plots as acs

    _orig_load = acs.load_trials

    def load_trials_no_stdcp(*args, **kwargs):
        trials = _orig_load(*args, **kwargs)
        return trials[trials["method"].astype(str) != "Std-CP"].copy()

    acs.load_trials = load_trials_no_stdcp
    suite = replace(
        acs.RF_YOEP_FB_MIN31_SUITE,
        paper_root=SUITE_ROOT,
        results_root=SUITE_ROOT / "results",
        result_glob="true_marginal_permuted_rf_income_notrim_t30_yoep2000*alpha*",
        title_prefix="ACS (RF, YOEP+FB, permute rows, size ≥31)",
        all_baselines_title_prefix="ACS (RF)",
        all_baselines_short_titles=True,
        alpha_panel_values=tuple(alphas),
    )
    acs.run_suite(suite)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", default="0.2,0.1", help="Run order preserved (default: 0.2 then 0.1)")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--n_workers", type=int, default=7)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    (SUITE_ROOT / "logs").mkdir(exist_ok=True)
    man_path = SUITE_ROOT / "seeds_manifest.json"
    man_path.write_text(json.dumps(_manifest(alphas=alphas, B=args.B, n_workers=args.n_workers), indent=2))
    print(f"Wrote {man_path}", flush=True)

    for alpha in alphas:
        _run_one(alpha=alpha, B=args.B, n_workers=args.n_workers)

    if args.plot:
        _plot(alphas=alphas)


if __name__ == "__main__":
    main()
