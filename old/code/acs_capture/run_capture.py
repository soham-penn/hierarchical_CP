#!/usr/bin/env python3
"""
ACS capture experiment: run once (expensive global RF), store full CSV artifacts.

Use apply_acs_capture.py afterward to apply different within-group corrections
or swap global OLS without re-running RF training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_DATA_DIR = REPO_ROOT / "real_data"
ACS_DIR = REAL_DATA_DIR / "acs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REAL_DATA_DIR))

from acs.data_processing import build_design_matrix_acs, load_and_clean_acs_pums
from code.marginal import run_acs_experiments as acs
from code.marginal.acs_capture.engine import run_one_replicate_capture, _feature_names_from_matrix
from code.marginal.acs_capture.storage import CaptureStore

PLOTS_ACS_ROOT = REPO_ROOT / "plots_marginal" / "acs"


def _worker_run(replicate_idx: int) -> tuple[int, dict | None, float]:
  shared = _WORKER_SHARED
  t0 = time.perf_counter()
  result = run_one_replicate_capture(
      shared["df"], shared["X"], shared["eligible_groups"], shared["strata"],
      shared["group_col"], shared["o_values"], shared["config"], replicate_idx,
      shared["store"],
  )
  return replicate_idx, result, time.perf_counter() - t0


_WORKER_SHARED = None


def _init_worker(df, X, eligible_groups, strata, group_col, o_values, config, store) -> None:
  global _WORKER_SHARED
  _WORKER_SHARED = {
      "df": df, "X": X, "eligible_groups": eligible_groups, "strata": strata,
      "group_col": group_col, "o_values": o_values, "config": config, "store": store,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="ACS capture experiment (full CSV artifacts).")
  parser.add_argument("--suite", type=str, default="rf_2012_mean",
                      help="Output under plots_marginal/acs/{suite}/capture_data/")
  parser.add_argument("--predictor", choices=("rf", "ols"), default="rf")
  parser.add_argument("--within_group_mode", choices=("mean", "correction"), default="mean",
                      help="Within-group mode for the capture run (RF global is always stored).")
  parser.add_argument("--B", type=int, default=1000)
  parser.add_argument("--n_workers", type=int, default=6)
  parser.add_argument("--alpha", type=float, default=0.2)
  parser.add_argument("--min_income", type=float, default=0.0,
                      help="0 disables income floor.")
  parser.add_argument("--drop_top_income_pct", type=float, default=0.0)
  parser.add_argument("--yoep_min_year", type=int, default=2012)
  parser.add_argument("--min_puma_size", type=int, default=21)
  parser.add_argument("--o_values", type=str, default="0,5,10,15,20")
  parser.add_argument(
      "--sampling_strategy",
      choices=("stratified", "random"),
      default="stratified",
      help="Calibration-group sampling: stratified by BA+ share (default) or random uniform.",
  )
  parser.add_argument("--acs_csv", type=str, default=None)
  args = parser.parse_args()

  capture_root = PLOTS_ACS_ROOT / args.suite / "capture_data"
  acs_csv = Path(args.acs_csv) if args.acs_csv else (ACS_DIR / "data/acs_data_all50states.csv")
  min_income = None if args.min_income <= 0 else float(args.min_income)

  df = load_and_clean_acs_pums(
      str(acs_csv),
      states_keep=["CA"],
      age_min=25, age_max=54,
      yoep_min_year=args.yoep_min_year,
      min_hours=40,
      min_income=min_income,
      drop_top_income_fraction=args.drop_top_income_pct if args.drop_top_income_pct > 0 else None,
      y_transform=lambda x: np.asarray(x, dtype=float),
  )
  df = df.dropna(subset=["puma"]).copy()
  df["puma"] = df["puma"].astype(int)
  X = build_design_matrix_acs(df, exclude_entry_recency=True, exclude_cow=True)

  counts = df.groupby("puma").size()
  eligible_groups = counts[counts >= args.min_puma_size].index.to_numpy().tolist()
  group_share = acs.compute_group_share_baplus(df, "puma")
  strata = acs._build_share_baplus_strata(eligible_groups, group_share, n_strata=5)

  o_values = sorted(int(v) for v in args.o_values.split(","))
  acs._set_target_index(20)

  n_feat = X.shape[1]

  config = {
      "B": args.B,
      "seed": 456,
      "alpha": args.alpha,
      "alpha_selection": 0.5,
      "quantile_mode": "deterministic",
      "quantile_base_seed": 456,
      "n_repeated": 50,
      "n_puma_groups": 20,
      "n_calib_per_stratum": 4,
      "o_values": o_values,
      "eligible_groups": eligible_groups,
      "design": "marginal_uniform" if args.sampling_strategy == "random" else "marginal",
      "within_group": True,
      "permute_rows": True,
      "min_calib_puma_size": args.min_puma_size,
      "predictor": args.predictor,
      "outcome_scale": "income",
      "within_group_mode": args.within_group_mode,
      "drop_top_income_fraction": float(args.drop_top_income_pct),
      "yoep_min_year": int(args.yoep_min_year),
      "min_income": min_income,
      "suite": args.suite,
      "sampling_strategy": args.sampling_strategy,
  }

  mp.set_start_method("fork", force=True)
  manager = mp.Manager()
  csv_lock = manager.Lock()
  store = CaptureStore(capture_root, lock=csv_lock)
  store.set_feature_names(_feature_names_from_matrix(n_feat))
  store.write_config(config)

  pooled_lo, pooled_hi = acs.compute_global_pooled_income_interval(df, args.alpha)
  with (capture_root / "global_pooled.json").open("w") as f:
    json.dump({"lower": pooled_lo, "upper": pooled_hi, "width": pooled_hi - pooled_lo}, f)

  print("=" * 70)
  print("ACS CAPTURE EXPERIMENT")
  print("=" * 70)
  print(f"  Capture root: {capture_root}")
  print(f"  Predictor: {args.predictor}  within_group_mode: {args.within_group_mode}")
  print(f"  Sampling: {args.sampling_strategy}")
  print(f"  Cohort: YOEP>={args.yoep_min_year}, min_income={min_income}, trim={args.drop_top_income_pct}")
  print(f"  Observations: {len(df)}, eligible PUMAs: {len(eligible_groups)}")
  print(f"  Replicates: {args.B}, workers: {args.n_workers}")
  print()
  print("  Why RF is slow: each replicate evaluates ~{0} test PUMAs × {1} o values × "
        "2 HCP methods; each call fits a 50-tree RF on the complement group set.".format(
            "~30-45", len(o_values)))
  print("  This script stores observations + every mu_global(fit_id,puma,row) so later")
  print("  apply_acs_capture.py can swap within-group correction or global OLS without refitting RF.")
  print()

  # Aggregate results (same layout as run_acs_experiments)
  n_o = len(o_values)
  cov = np.full((args.B, len(acs.METHODS), n_o), np.nan)
  m_map = {m: i for i, m in enumerate(acs.METHODS)}
  income_targets = np.full(args.B, np.nan)
  wid = np.full((args.B, len(acs.METHODS), n_o), np.nan)
  wid_income = np.full((args.B, len(acs.METHODS), n_o), np.nan)
  timings = []

  mp.set_start_method("fork", force=True)
  ctx = mp.get_context("fork")
  with ctx.Pool(
      processes=args.n_workers,
      initializer=_init_worker,
      initargs=(df, X, eligible_groups, strata, "puma", o_values, config, store),
  ) as pool:
    for rep_idx, result, elapsed in pool.imap_unordered(_worker_run, range(args.B)):
      timings.append(elapsed)
      if result is None:
        continue
      income_targets[rep_idx] = result.get("income_target", np.nan)

      def _write_rec(m_i, o_i, rec):
        cov[rep_idx, m_i, o_i] = rec["coverage"]
        wid[rep_idx, m_i, o_i] = rec["width"]
        wid_income[rep_idx, m_i, o_i] = rec["width_income"]

      for method in acs.BASELINE_METHODS:
        if 0 in result["baseline"][method]:
          _write_rec(m_map[method], 0, result["baseline"][method][0])
      for method in acs.HCP_METHODS:
        m_i = m_map[method]
        for o_i, o in enumerate(o_values):
          if o in result["hcp"][method]:
            _write_rec(m_i, o_i, result["hcp"][method][o])
      for method in acs.STD_CP_METHODS:
        m_i = m_map[method]
        for o_i, o in enumerate(o_values):
          if o in result["stdcp"][method]:
            _write_rec(m_i, o_i, result["stdcp"][method][o])

      done = len(timings)
      if done % 50 == 0 or done <= 5 or done == args.B:
        med = float(np.median(timings))
        eta_h = med * (args.B - done) / 3600
        print(f"    Replicate {done}/{args.B}  (median {med:.1f}s/rep, ETA ~{eta_h:.1f}h)")

  results = {
      "methods": acs.METHODS,
      "o_values": o_values,
      "coverage": cov,
      "width": wid,
      "width_income": wid_income,
      "lower_log": wid * 0 + np.nan,
      "upper_log": wid * 0 + np.nan,
      "lower_income": wid * 0 + np.nan,
      "upper_income": wid * 0 + np.nan,
      "income_targets": income_targets,
  }

  alpha_str = acs.alpha_to_tag(args.alpha)
  if args.predictor == "rf":
    run_tag = f"true_marginal_permuted_rf_income_notrim_{alpha_str}"
  else:
    run_tag = f"true_marginal_permuted_ols_notrim_{alpha_str}"
  out_dir = PLOTS_ACS_ROOT / args.suite / "results" / run_tag
  out_dir.mkdir(parents=True, exist_ok=True)
  global_pooled = json.loads((capture_root / "global_pooled.json").read_text())
  df_results = acs.save_results_to_csv(
      results, out_dir / f"acs_true_marg_{alpha_str}_detailed.csv",
      global_pooled=global_pooled,
  )
  summary_dir = out_dir / "summaries"
  acs.save_summaries(df_results, summary_dir)
  acs.save_endpoints_analysis(df_results, global_pooled, summary_dir, o_compare=20)

  print("\nDone. Capture CSVs in:", capture_root)
  print("Summary results in:", out_dir)
  print("Next:")
  print(f"  .venv/bin/python code/marginal/acs_capture/apply_acs_capture.py \\")
  print(f"    --capture_dir {capture_root} --within_group_mode correction --output_suite rf_2012_correction")


if __name__ == "__main__":
  main()
