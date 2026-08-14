#!/usr/bin/env python3
"""Recompute OLS baselines from capture data and patch an existing apply output suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from code.marginal import run_acs_experiments as acs
from code.marginal.acs_capture.capture_loader import load_capture_tables
from code.marginal.acs_capture.replay import replay_baselines_from_capture

PLOTS_ACS_ROOT = REPO_ROOT / "plots_marginal" / "acs"


def _patch_detailed_with_baselines(
    df: pd.DataFrame,
    baseline_results: dict,
    o_values: list[int],
) -> pd.DataFrame:
    """Replace baseline / Std-CP rows in a detailed results CSV."""
    m_map = {m: i for i, m in enumerate(baseline_results["methods"])}
    cov = baseline_results["coverage"]
    wid = baseline_results["width"]
    wid_inc = baseline_results["width_income"]
    out = df.copy()

    for b in range(cov.shape[0]):
        for method in acs.BASELINE_METHODS:
            m_i = m_map[method]
            mask = (out["replicate"] == b) & (out["method"] == method) & (out["o"] == o_values[0])
            if not mask.any():
                continue
            out.loc[mask, "coverage"] = cov[b, m_i, 0]
            out.loc[mask, "width"] = wid[b, m_i, 0]
            out.loc[mask, "width_income"] = wid_inc[b, m_i, 0]

        for method in acs.STD_CP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                mask = (out["replicate"] == b) & (out["method"] == method) & (out["o"] == o)
                if not mask.any():
                    continue
                out.loc[mask, "coverage"] = cov[b, m_i, o_i]
                out.loc[mask, "width"] = wid[b, m_i, o_i]
                out.loc[mask, "width_income"] = wid_inc[b, m_i, o_i]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch OLS baseline rows into an apply output suite.")
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--output_suite", type=str, required=True)
    parser.add_argument("--global_predictor", choices=("ols", "rf"), default="ols")
    parser.add_argument("--within_group_mode", choices=("mean", "correction"), default="correction")
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--max_replicates", type=int, default=None)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    tables = load_capture_tables(capture_dir, load_mu_global=False)
    config = dict(tables.config)
    o_values = [int(x) for x in config["o_values"]]
    alpha = float(config["alpha"])
    if args.max_replicates is not None:
        tables.n_replicates = min(int(args.max_replicates), tables.n_replicates)

    replay_config = dict(config)
    replay_config["within_group_mode"] = args.within_group_mode
    replay_config["_global_predictor"] = args.global_predictor
    replay_config["_recompute_baselines"] = True

    print(f"Recomputing baselines with global_predictor={args.global_predictor}...")
    baseline_results = replay_baselines_from_capture(
        tables=tables,
        o_values=o_values,
        config=replay_config,
        n_workers=args.n_workers,
    )

    alpha_str = acs.alpha_to_tag(alpha)
    suffix = "_corr" if args.within_group_mode == "correction" else ""
    if args.global_predictor == "ols":
        run_tag = f"true_marginal_permuted_ols_notrim{suffix}_{alpha_str}"
    else:
        run_tag = f"true_marginal_permuted_rf_income_notrim{suffix}_{alpha_str}"

    out_dir = PLOTS_ACS_ROOT / args.output_suite / "results" / run_tag
    detailed_path = out_dir / f"acs_true_marg_{alpha_str}_detailed.csv"
    if not detailed_path.exists():
        raise FileNotFoundError(f"Missing detailed results: {detailed_path}")

    df = pd.read_csv(detailed_path)
    df_patched = _patch_detailed_with_baselines(df, baseline_results, o_values)
    df_patched.to_csv(detailed_path, index=False)
    print(f"Patched {detailed_path}")

    summary_dir = out_dir / "summaries"
    acs.save_summaries(df_patched, summary_dir)
    global_pooled = json.loads((capture_dir / "global_pooled.json").read_text())
    acs.save_endpoints_analysis(df_patched, global_pooled, summary_dir, o_compare=20)

    hcp = df_patched[(df_patched.method == "HCP") & (df_patched.o == 0)]
    print(
        f"HCP o=0 after patch: coverage mean={hcp.coverage.mean():.4f}, "
        f"median width={hcp.width.median():.0f}"
    )


if __name__ == "__main__":
    main()
