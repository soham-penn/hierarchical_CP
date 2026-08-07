#!/usr/bin/env python3
"""
YOEP+FB min21 with row permutation.

Standard cohort filters (age 25–54, hours ≥ 40), min21 / target idx 20,
o ∈ {0,5,10,15,20}, row permutation ON.

Full suite (B=1000 default):
  - α=0.2 and α=0.1, GHCP/HCP/baselines (use --skip_stdcp for fast runs)
  - With Std-CP: studentized at all o (α=0.2) or o=20 only (α=0.1)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = REPO_ROOT / "plots_marginal" / "acs" / "min21"
SEED = 456
QUANTILE_BASE_SEED = 456
RF_RESULTS = REPO_ROOT / "plots_marginal" / "acs" / "rf" / "results"


def _manifest(*, alphas: list[float], B: int, n_workers: int, full_stdcp: bool) -> dict:
    return {
        "suite": "acs/min21",
        "description": (
            "CA foreign-born YOEP>=2000, age 25–54, hours>=40; "
            "uniform_one_target; row permutation ON; studentized Std-CP (RF local)"
        ),
        "seed": SEED,
        "quantile_base_seed": QUANTILE_BASE_SEED,
        "quantile_mode": "deterministic",
        "permute_rows": True,
        "design": "uniform_one_target",
        "min_puma_size": 21,
        "target_index": 20,
        "o_values": [0, 5, 10, 15, 20],
        "n_puma_groups": 20,
        "B": B,
        "alphas": alphas,
        "n_workers": n_workers,
        "predictor": "rf",
        "within_group_mode": "mean",
        "score_type": "absolute",
        "stdcp_score_type": "studentized",
        "stdcp_center": "local",
        "yoep_min_year": 2000,
        "no_age_filter": False,
        "no_hours_filter": False,
        "age_range": [25, 54],
        "min_hours": 40,
        "exclude_entry_recency": True,
        "include_cow": False,
        "alpha20_stdcp_o": [0, 5, 10, 15, 20],
        "alpha10_stdcp_o": [20] if full_stdcp else None,
        "selection_seed_formula": "seed + replicate_idx * 1009",
        "row_permutation_seed_formula": "seed + replicate_idx * 1009 + 811",
    }


def _run_one(
    *,
    alpha: float,
    B: int,
    n_workers: int,
    skip_stdcp: bool = False,
    stdcp_score_type: str | None = "studentized",
    stdcp_o_values: list[int] | None = None,
) -> Path:
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
        "--exclude_entry_recency",
        "--design", "uniform_one_target",
        "--min_puma_size", "21",
        "--target_index", "20",
        "--o_values", "0,5,10,15,20",
        "--alpha", str(alpha),
        "--B", str(B),
        "--n_workers", str(n_workers),
        "--score_type", "absolute",
        "--quantile-base-seed", str(QUANTILE_BASE_SEED),
    ]
    if skip_stdcp:
        cmd.append("--skip_stdcp")
    elif stdcp_score_type:
        cmd.extend(["--stdcp_score_type", stdcp_score_type])
    if stdcp_o_values is not None and not skip_stdcp:
        cmd.extend(["--stdcp_o_values", ",".join(str(o) for o in stdcp_o_values)])

    log = SUITE_ROOT / "logs" / f"run_{tag}.log"
    print(f"\n=== min21_permute {tag} ===", flush=True)
    print(" ".join(cmd), flush=True)
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed ({proc.returncode}); see {log}")

    matches = sorted(
        RF_RESULTS.glob(f"true_marginal_permuted_rf_income_notrim*yoep2000_{tag}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No results dir matching *yoep2000_{tag} under {RF_RESULTS}")
    src = matches[0]
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

    suite = replace(
        acs.RF_YOEP_FB_MIN21_SUITE,
        paper_root=SUITE_ROOT,
        results_root=SUITE_ROOT / "results",
        result_glob="true_marginal_permuted_rf_income_notrim*yoep2000*alpha*",
        title_prefix="ACS (RF, age/hrs filtered, permute rows, size ≥21)",
        all_baselines_title_prefix="ACS (RF)",
        all_baselines_short_titles=True,
        alpha_panel_values=tuple(alphas),
    )
    acs.run_suite(suite)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", default="0.2,0.1")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--n_workers", type=int, default=7)
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Alias for --skip_stdcp (legacy name)",
    )
    p.add_argument(
        "--skip_stdcp",
        action="store_true",
        help="Skip studentized Std-CP (GHCP/HCP/SHCP only; much faster)",
    )
    args = p.parse_args()
    skip_stdcp = args.skip_stdcp or args.smoke
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    (SUITE_ROOT / "logs").mkdir(exist_ok=True)
    man_path = SUITE_ROOT / "seeds_manifest.json"
    man_path.write_text(
        json.dumps(
            _manifest(
                alphas=alphas,
                B=args.B,
                n_workers=args.n_workers,
                full_stdcp=not skip_stdcp,
            ),
            indent=2,
        )
    )
    print(f"Wrote {man_path}", flush=True)

    for alpha in alphas:
        if skip_stdcp:
            _run_one(
                alpha=alpha,
                B=args.B,
                n_workers=args.n_workers,
                skip_stdcp=True,
                stdcp_score_type=None,
            )
            continue
        if abs(alpha - 0.2) < 1e-9:
            _run_one(
                alpha=alpha,
                B=args.B,
                n_workers=args.n_workers,
                stdcp_score_type="studentized",
                stdcp_o_values=None,
            )
        elif abs(alpha - 0.1) < 1e-9:
            _run_one(
                alpha=alpha,
                B=args.B,
                n_workers=args.n_workers,
                stdcp_score_type="studentized",
                stdcp_o_values=[20],
            )
        else:
            _run_one(
                alpha=alpha,
                B=args.B,
                n_workers=args.n_workers,
                stdcp_score_type="studentized",
            )

    if args.plot:
        _plot(alphas=alphas)


if __name__ == "__main__":
    main()
