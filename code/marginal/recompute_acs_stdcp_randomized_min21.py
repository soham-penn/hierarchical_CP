#!/usr/bin/env python3
"""Recompute ACS min21 Std-CP with randomized conformal quantiles (same seeds).

Uses studentized local RF scores (paper setting). At small o, deterministic
quantiles are always infinite; randomization can yield finite radii with
positive probability while targeting exact marginal coverage 1-α.

Patches Std-CP rows into paper-results/acs/min21 results and regenerates
paper plots. Does not re-run GHCP/HCP.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from code.paths import PLOTS_MARGINAL
from scores import make_quantile_seed

SUITE_ROOT = PLOTS_MARGINAL / "acs" / "min21"
OUT_ROOT = SUITE_ROOT / "stdcp_studentized_randomized"
SEED = 456
QUANTILE_BASE_SEED = 456
O_VALUES = [0, 5, 10, 15, 20]
N_CALIB = 20
TARGET_INDEX = 20
B_DEFAULT = 1000
YOEP_MIN = 2000
OUTCOME_SCALE = "income"
RESULT_GLOB = "true_marginal_permuted_rf_income_notrim*yoep2000_alpha*"
SCORE_TYPE = "studentized"
QUANTILE_MODE = "randomized"

_DF = None
_X = None
_ELIGIBLE = None
_GROUP_COUNTS = None


def _init_worker(df, X, eligible, group_counts):
    global _DF, _X, _ELIGIBLE, _GROUP_COUNTS
    _DF, _X, _ELIGIBLE, _GROUP_COUNTS = df, X, eligible, group_counts
    acs._set_target_index(TARGET_INDEX)


def _load_cohort():
    acs._set_target_index(TARGET_INDEX)
    acs_csv = REPO_ROOT / "real_data" / "acs" / "data" / "acs_data_all50states.csv"
    df = load_and_clean_acs_pums(
        str(acs_csv),
        states_keep=["CA"],
        age_min=25,
        age_max=54,
        yoep_min_year=YOEP_MIN,
        min_hours=40,
        min_income=None,
        drop_top_income_fraction=None,
        y_transform=acs._outcome_transform(OUTCOME_SCALE),
    )
    df = df.dropna(subset=["puma"]).copy()
    df["puma"] = df["puma"].astype(int)
    X = build_design_matrix_acs(df, exclude_entry_recency=True, exclude_cow=True)
    counts = df.groupby("puma").size()
    eligible = counts[counts >= TARGET_INDEX + 1].index.to_numpy().tolist()
    return df, X, eligible, counts


def _stdcp_one_replicate(args):
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

    row_rng = np.random.default_rng(SEED + replicate_idx * 1009 + 811)
    group_data = {}
    for grp in list(calib_groups) + [test_group]:
        grp_idx = np.where((df["puma"] == grp).values)[0]
        grp_idx = row_rng.permutation(grp_idx)
        group_data[grp] = {
            "X": X[grp_idx],
            "Y": df.iloc[grp_idx]["y"].to_numpy(dtype=float),
            "income": (
                df.iloc[grp_idx]["income"].to_numpy(dtype=float)
                if "income" in df.columns
                else df.iloc[grp_idx]["y"].to_numpy(dtype=float)
            ),
        }
    Xg = group_data[test_group]["X"]
    Yg = group_data[test_group]["Y"]
    income = group_data[test_group]["income"]
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
                quantile_mode=QUANTILE_MODE,
                quantile_random_seed=make_quantile_seed(
                    QUANTILE_BASE_SEED, replicate_idx, TARGET_INDEX, o, "stdcp"
                ),
                score_type=SCORE_TYPE,
            )
            rec = acs._result_record(interval, true_y, outcome_scale=OUTCOME_SCALE)
        rows.append(
            {
                "replicate": replicate_idx,
                "method": "Std-CP",
                "o": o,
                "income_target": income_target,
                "score_type": SCORE_TYPE,
                "quantile_mode": QUANTILE_MODE,
                **rec,
            }
        )
    return rows


def _result_dir_for_alpha(alpha: float) -> Path:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    matches = sorted(
        (SUITE_ROOT / "results").glob(
            f"true_marginal_permuted_rf_income_notrim*yoep2000_{tag}"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No results dir for {tag} under {SUITE_ROOT / 'results'}")
    return matches[0]


def run_alpha(alpha: float, B: int, n_workers: int) -> Path:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Std-CP randomized studentized {tag} B={B} workers={n_workers} ===", flush=True)

    rows = []
    worker_args = [(r, alpha) for r in range(B)]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(_DF, _X, _ELIGIBLE, _GROUP_COUNTS),
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
    detailed_path = out_dir / f"stdcp_studentized_randomized_{tag}_detailed.csv"
    detailed.to_csv(detailed_path, index=False)
    print(f"Wrote {detailed_path}", flush=True)

    # quick summary
    for o, g in detailed.groupby("o"):
        w = g["width"].astype(float)
        fin = np.isfinite(w)
        cov = float(g["coverage"].mean())
        se = float(np.sqrt(cov * (1 - cov) / len(g)))
        print(
            f"  o={o}: cov={cov:.3f}±{se:.3f}  finite={int(fin.sum())}/{len(g)}  "
            f"mean_finite_w={w[fin].mean() if fin.any() else float('nan'):.1f}",
            flush=True,
        )
    return detailed_path


def patch_suite_results(alpha: float, stdcp_detailed: pd.DataFrame) -> Path:
    tag = f"alpha{int(round(alpha * 100)):02d}"
    result_dir = _result_dir_for_alpha(alpha)
    detailed_path = result_dir / f"acs_true_marg_{tag}_detailed.csv"
    df = pd.read_csv(detailed_path)
    keep = df[df["method"] != "Std-CP"].copy()
    std = stdcp_detailed.copy()
    for c in df.columns:
        if c not in std.columns:
            std[c] = np.nan
    std = std[df.columns]
    out = pd.concat([keep, std], ignore_index=True)
    out = out.sort_values(["replicate", "method", "o"]).reset_index(drop=True)
    backup = detailed_path.with_suffix(".csv.bak_before_rand_stdcp")
    if not backup.exists():
        shutil.copy2(detailed_path, backup)
    out.to_csv(detailed_path, index=False)
    acs.save_summaries(out, result_dir / "summaries")
    print(f"Patched {detailed_path}", flush=True)
    w = std["width"].to_numpy(dtype=float)
    print(
        f"  Std-CP finite widths: {int(np.isfinite(w).sum())}/{len(std)}",
        flush=True,
    )
    return detailed_path


def _plot() -> None:
    from dataclasses import replace
    from real_data.acs import plot_dgp_style_paper_plots as acs_plot

    suite = replace(
        acs_plot.RF_YOEP_FB_MIN21_SUITE,
        paper_root=SUITE_ROOT,
        results_root=SUITE_ROOT / "results",
        result_glob=RESULT_GLOB,
        title_prefix="ACS (RF, age/hrs filtered, permute rows, size ≥21)",
        all_baselines_title_prefix="ACS (RF)",
        all_baselines_short_titles=True,
        alpha_panel_values=(0.2, 0.1),
    )
    acs_plot.run_suite(suite)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alphas", default="0.05,0.1,0.15,0.2")
    p.add_argument("--B", type=int, default=B_DEFAULT)
    p.add_argument("--n_workers", type=int, default=7)
    p.add_argument("--skip_plot", action="store_true")
    p.add_argument("--skip_patch", action="store_true")
    args = p.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    man = {
        "suite": "acs/min21/stdcp_studentized_randomized",
        "seed": SEED,
        "quantile_base_seed": QUANTILE_BASE_SEED,
        "stdcp_score_type": SCORE_TYPE,
        "quantile_mode": QUANTILE_MODE,
        "stdcp_center": "local",
        "half_split": "random half of first o target-PUMA rows (train), other half (cal)",
        "selection_seed_formula": "seed + replicate_idx * 1009",
        "row_permutation_seed_formula": "seed + replicate_idx * 1009 + 811",
        "stdcp_split_seed": "make_quantile_seed(..., 'stdcp_split')",
        "B": args.B,
        "alphas": alphas,
        "o_values": O_VALUES,
        "permute_rows": True,
    }
    (OUT_ROOT / "seeds_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"Wrote {OUT_ROOT / 'seeds_manifest.json'}", flush=True)

    global _DF, _X, _ELIGIBLE, _GROUP_COUNTS
    print("Loading paper cohort...")
    _DF, _X, _ELIGIBLE, _GROUP_COUNTS = _load_cohort()
    print(f"  n={len(_DF)}, eligible PUMAs={len(_ELIGIBLE)}", flush=True)

    for alpha in alphas:
        path = run_alpha(alpha, args.B, args.n_workers)
        detailed = pd.read_csv(path)
        if not args.skip_patch:
            patch_suite_results(alpha, detailed)

    if not args.skip_plot:
        print("\nRegenerating paper plots...", flush=True)
        _plot()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
