#!/usr/bin/env python3
"""
Apply a new within-group correction (or global OLS) from capture CSV artifacts.

Requires capture_data/ produced by run_capture.py. Does NOT refit RF when
--global_predictor rf (default): uses mu_global.csv from the capture run.
With --global_predictor ols, fits OLS from observations.csv per global_fits row (fast).
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from code.marginal import run_acs_experiments as acs
from code.marginal.acs_capture.capture_loader import load_capture_tables
from code.marginal.acs_capture.storage import CaptureStore


PLOTS_ACS_ROOT = REPO_ROOT / "plots_marginal" / "acs"


def _inject_reference_baselines(df_results: pd.DataFrame, alpha_str: str) -> pd.DataFrame:
    """Fill baseline/Std-CP rows from the reference rf_2012_mean detailed output."""
    ref_path = (
        PLOTS_ACS_ROOT
        / "rf_2012_mean"
        / "results"
        / f"true_marginal_permuted_rf_income_notrim_{alpha_str}"
        / f"acs_true_marg_{alpha_str}_detailed.csv"
    )
    if not ref_path.exists():
        return df_results

    ref = pd.read_csv(ref_path)
    baseline_methods = set(acs.BASELINE_METHODS + acs.STD_CP_METHODS)
    key_cols = ["replicate", "method", "o"]
    value_cols = [
        "coverage", "width", "width_income",
        "lower", "upper", "lower_income", "upper_income", "income_target",
    ]

    ref_sub = ref[ref["method"].isin(baseline_methods)][key_cols + value_cols].copy()
    out = df_results.copy()
    mask = out["method"].isin(baseline_methods)
    out_sub = out.loc[mask, key_cols].merge(ref_sub, on=key_cols, how="left")
    for col in value_cols:
        out.loc[mask, col] = out_sub[col].to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply within-group mode from ACS capture CSVs.")
    parser.add_argument("--capture_dir", type=str, required=True)
    parser.add_argument("--within_group_mode", choices=("mean", "correction"), required=True)
    parser.add_argument("--global_predictor", choices=("rf", "ols"), default="rf",
                        help="rf: use stored mu_global.csv; ols: refit OLS per fit from observations.")
    parser.add_argument("--output_suite", type=str, default="rf_2012_correction")
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--max_replicates", type=int, default=None,
                        help="Optional cap for testing (default: all replicates in capture).")
    parser.add_argument("--store_mu_global", action=argparse.BooleanOptionalAction, default=True,
                        help="Append calib/test μ_RF rows to capture_dir/mu_global.csv during replay (default: on).")
    parser.add_argument(
        "--reuse_global_fit_across_targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse global fit when training indices are unchanged across target PUMAs "
            "(default: on; much faster for conditional ACS runs)."
        ),
    )
    parser.add_argument(
        "--reuse_group_adjustment_across_targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse within-group adjustment fits for calibration groups when "
            "(replicate, method, o) are fixed (default: on)."
        ),
    )
    parser.add_argument(
        "--single_target_per_replicate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, draw one target PUMA per replicate (from captured test groups) "
            "instead of averaging over all captured test PUMAs."
        ),
    )
    parser.add_argument(
        "--target_selection_seed",
        type=int,
        default=None,
        help="Base seed for single-target-per-replicate target-PUMA draws.",
    )
    parser.add_argument(
        "--non_stratified_group_sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Per replicate, randomly sample calibration groups and one target group "
            "from all available groups (non-stratified)."
        ),
    )
    parser.add_argument(
        "--use_standardized_score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic: use standardized residual scores |Y-mu|/s(X,U) where s is "
            "a global residual-scale model fit on global-training groups."
        ),
    )
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    tables = load_capture_tables(capture_dir, load_mu_global=False)
    config = tables.config
    o_values = [int(x) for x in config["o_values"]]
    alpha = float(config["alpha"])

    print(f"Loaded capture from {capture_dir}")
    print(f"  Applying within_group_mode={args.within_group_mode}, global_predictor={args.global_predictor}")
    print(f"  Replicates: {tables.n_replicates}, o_values: {o_values}")

    if args.max_replicates is not None:
        tables.n_replicates = min(int(args.max_replicates), tables.n_replicates)
        print(f"  Capped to {tables.n_replicates} replicates (--max_replicates)", flush=True)

    replay_config = dict(config)
    replay_config["within_group_mode"] = args.within_group_mode
    replay_config["_global_predictor"] = args.global_predictor
    replay_config["_recompute_baselines"] = (args.global_predictor == "ols") or bool(args.single_target_per_replicate)
    replay_config["_reuse_global_fit_across_targets"] = bool(args.reuse_global_fit_across_targets)
    replay_config["_reuse_group_adjustment_across_targets"] = bool(args.reuse_group_adjustment_across_targets)
    replay_config["_single_target_per_replicate"] = bool(args.single_target_per_replicate)
    replay_config["_target_selection_seed"] = (
        int(args.target_selection_seed)
        if args.target_selection_seed is not None
        else int(config.get("seed", 456))
    )
    replay_config["_non_stratified_group_sampling"] = bool(args.non_stratified_group_sampling)
    replay_config["_use_standardized_score"] = bool(args.use_standardized_score)

    store = None
    if args.store_mu_global and args.global_predictor == "rf":
        manager = Manager()
        store = CaptureStore(capture_dir, lock=manager.Lock())
        print(f"  Will append missing μ_RF rows to {capture_dir / 'mu_global.csv'}", flush=True)

    from code.marginal.acs_capture.replay import replay_hcp_from_capture

    results = replay_hcp_from_capture(
        tables=tables,
        o_values=o_values,
        config=replay_config,
        n_workers=args.n_workers,
        store=store,
    )

    alpha_str = acs.alpha_to_tag(alpha)
    suffix = "_corr" if args.within_group_mode == "correction" else ""
    if args.global_predictor == "ols":
        run_tag = f"true_marginal_permuted_ols_notrim{suffix}_{alpha_str}"
    else:
        run_tag = f"true_marginal_permuted_rf_income_notrim{suffix}_{alpha_str}"

    out_dir = PLOTS_ACS_ROOT / args.output_suite / "results" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    global_pooled = json.loads((capture_dir / "global_pooled.json").read_text())
    df_results = acs.save_results_to_csv(
        results,
        out_dir / f"acs_true_marg_{alpha_str}_detailed.csv",
        global_pooled=global_pooled,
    )
    if args.global_predictor != "ols" and not bool(args.single_target_per_replicate):
        df_results = _inject_reference_baselines(df_results, alpha_str=alpha_str)
    df_results.to_csv(out_dir / f"acs_true_marg_{alpha_str}_detailed.csv", index=False)
    summary_dir = out_dir / "summaries"
    acs.save_summaries(df_results, summary_dir)
    acs.save_endpoints_analysis(df_results, global_pooled, summary_dir, o_compare=20)
    print("Wrote results to", out_dir)


if __name__ == "__main__":
    main()
