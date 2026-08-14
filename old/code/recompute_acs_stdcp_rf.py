#!/usr/bin/env python3
"""
Recompute Std-CP only for rf_2000_mean using within-group RF (half train / half cal),
keeping all other method rows unchanged. Same seeds / sampling as the original run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "real_data"))

from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs
from code.marginal import run_acs_experiments as acs

PAPER_ROOT = (
    REPO_ROOT
    / "plots_marginal"
    / "acs"
    / "non-stratified"
    / "rf_2000_mean"
    / "rf_2000_mean"
)
DETAILED = (
    PAPER_ROOT
    / "results"
    / "true_marginal_permuted_rf_income_notrim_t31_yoep2000_alpha20"
    / "acs_true_marg_alpha20_detailed.csv"
)

ALPHA = 0.2
SEED = 456
QUANTILE_BASE_SEED = 456
O_VALUES = [0, 10, 20, 30]
N_CALIB = 20
TARGET_INDEX = 31
B = 1000
OUTCOME_SCALE = "income"


def _load_cohort():
    acs._set_target_index(TARGET_INDEX)
    acs_csv = REPO_ROOT / "real_data" / "acs" / "data" / "acs_data_all50states.csv"
    df = load_and_clean_acs_pums(
        str(acs_csv),
        states_keep=["CA"],
        age_min=25,
        age_max=54,
        yoep_min_year=2000,
        min_hours=40,
        min_income=None,
        drop_top_income_fraction=None,
        y_transform=acs._outcome_transform(OUTCOME_SCALE),
    )
    df = df.dropna(subset=["puma"]).copy()
    df["puma"] = df["puma"].astype(int)
    X = build_design_matrix_acs(df, exclude_entry_recency=True, exclude_cow=True)
    counts = df.groupby("puma").size()
    min_puma = TARGET_INDEX + 1  # 32
    eligible = counts[counts >= min_puma].index.to_numpy().tolist()
    return df, X, eligible, counts


def _stdcp_for_replicate(df, X, eligible, group_counts, replicate_idx):
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
    grp_mask = (df["puma"] == test_group).values
    grp_idx = np.where(grp_mask)[0]
    grp_idx = row_rng.permutation(grp_idx)
    Xg = X[grp_idx]
    Yg = df.iloc[grp_idx]["y"].values.astype(float)

    if len(Yg) <= TARGET_INDEX:
        return None

    x_target = Xg[TARGET_INDEX]
    true_y = float(Yg[TARGET_INDEX])
    rows = []
    for o in O_VALUES:
        if len(Yg) <= max(o, TARGET_INDEX):
            rec = {
                "coverage": np.nan,
                "width": np.nan,
                "width_income": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "lower_income": np.nan,
                "upper_income": np.nan,
            }
        elif o <= 0:
            rec = acs._result_record((-np.inf, np.inf), true_y, outcome_scale=OUTCOME_SCALE)
        else:
            stdcp_rng = np.random.default_rng(
                acs.make_quantile_seed(
                    QUANTILE_BASE_SEED, replicate_idx, TARGET_INDEX, o, "stdcp_split"
                )
            )
            interval = acs._compute_std_cp_interval(
                x_hist=Xg[:o],
                y_hist=Yg[:o],
                x_target=x_target,
                alpha=ALPHA,
                rng=stdcp_rng,
                quantile_mode="deterministic",
                quantile_random_seed=acs.make_quantile_seed(
                    QUANTILE_BASE_SEED, replicate_idx, TARGET_INDEX, o, "stdcp"
                ),
            )
            rec = acs._result_record(interval, true_y, outcome_scale=OUTCOME_SCALE)
        rows.append({"replicate": replicate_idx, "method": "Std-CP", "o": o, **rec})
    return rows


def _rebuild_summaries(detailed: pd.DataFrame) -> None:
    """Rebuild paper summaries under PAPER_ROOT/summaries from the patched detailed CSV."""
    method_map = {
        "Donor-HCP": "GHCP",
        "HCP": "HCP",
        "Std-CP": "Std-CP",
        "S-HCP": "S-HCP",
        "Pooling": "Pooling",
        "Subsampling": "Subsampling",
        "Repeated": "Repeated",
        "Pooled-Income-Quantile": "Pooled-Income-Quantile",
    }
    df = detailed.copy()
    df["method"] = df["method"].replace(method_map)
    df["alpha"] = ALPHA
    df["nominal_coverage"] = 1.0 - ALPHA
    df["plot_width"] = df["width"]

    summary_dir = PAPER_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    trials_path = summary_dir / "acs_trials_long.csv"
    keep = [
        "replicate", "method", "o", "coverage", "width", "width_income",
        "alpha", "nominal_coverage", "plot_width",
    ]
    df[keep].to_csv(trials_path, index=False)

    rows = []
    for (method, o), g in df.groupby(["method", "o"], dropna=False):
        cov = g["coverage"].dropna()
        w = g["width"].replace([np.inf, -np.inf], np.nan)
        w_fin = w.dropna()
        n_inf = int(np.isinf(g["width"].to_numpy(dtype=float)).sum()) if len(g) else 0
        rows.append({
            "alpha": ALPHA,
            "method": method,
            "o": int(o) if pd.notna(o) else o,
            "nominal_coverage": 1.0 - ALPHA,
            "coverage_mean": float(cov.mean()) if len(cov) else np.nan,
            "coverage_std": float(cov.std(ddof=1)) if len(cov) > 1 else (0.0 if len(cov) else np.nan),
            "coverage_se": float(cov.std(ddof=1) / np.sqrt(len(cov))) if len(cov) > 1 else (0.0 if len(cov) else np.nan),
            "coverage_n": float(len(cov)),
            "width_mean": float(w_fin.mean()) if len(w_fin) else np.nan,
            "width_std": float(w_fin.std(ddof=1)) if len(w_fin) > 1 else np.nan,
            "width_se": float(w_fin.std(ddof=1) / np.sqrt(len(w_fin))) if len(w_fin) > 1 else np.nan,
            "width_median": float(w_fin.median()) if len(w_fin) else np.nan,
            "width_q25": float(w_fin.quantile(0.25)) if len(w_fin) else np.nan,
            "width_q75": float(w_fin.quantile(0.75)) if len(w_fin) else np.nan,
            "width_min": float(w_fin.min()) if len(w_fin) else np.nan,
            "width_max": float(w_fin.max()) if len(w_fin) else np.nan,
            "width_n_total": float(len(g)),
            "width_n_finite": float(len(w_fin)),
            "width_n_infinite": float(n_inf),
            "width_finite_rate": float(len(w_fin) / len(g)) if len(g) else np.nan,
            "width_infinite_rate": float(n_inf / len(g)) if len(g) else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values(["method", "o"]).reset_index(drop=True)
    summary.to_csv(summary_dir / "acs_summary_by_alpha_o_method.csv", index=False)

    cov_tbl = summary.pivot_table(
        index="o", columns="method", values="coverage_mean", aggfunc="first"
    )
    wid_tbl = summary.pivot_table(
        index="o", columns="method", values="width_median", aggfunc="first"
    )
    cov_tbl.to_csv(summary_dir / "acs_coverage_table.csv")
    wid_tbl.to_csv(summary_dir / "acs_width_table.csv")
    print(f"Wrote summaries to {summary_dir}")


def main():
    print("Loading ACS cohort...")
    df, X, eligible, group_counts = _load_cohort()
    print(f"  n={len(df)}, eligible PUMAs={len(eligible)}")

    detailed = pd.read_csv(DETAILED)
    # backup once
    backup = DETAILED.with_name(DETAILED.stem + "_ols_stdcp_backup.csv")
    if not backup.exists():
        detailed.to_csv(backup, index=False)
        print(f"Backed up OLS Std-CP detailed CSV -> {backup.name}")

    print(f"Recomputing Std-CP (within-group RF) for {B} replicates...")
    new_rows = []
    for r in range(B):
        part = _stdcp_for_replicate(df, X, eligible, group_counts, r)
        if part is None:
            raise RuntimeError(f"replicate {r} failed to sample")
        new_rows.extend(part)
        if (r + 1) % 100 == 0:
            print(f"  {r + 1}/{B}")

    std_new = pd.DataFrame(new_rows)
    # Preserve income_target from existing Std-CP rows
    old_std = detailed[detailed["method"] == "Std-CP"][["replicate", "o", "income_target"]]
    std_new = std_new.merge(old_std, on=["replicate", "o"], how="left")

    other = detailed[detailed["method"] != "Std-CP"].copy()
    patched = pd.concat([other, std_new[detailed.columns]], ignore_index=True)
    patched = patched.sort_values(["replicate", "method", "o"]).reset_index(drop=True)
    patched.to_csv(DETAILED, index=False)
    print(f"Patched {DETAILED}")

    _rebuild_summaries(patched)

    # Print table numbers
    s = pd.read_csv(PAPER_ROOT / "summaries" / "acs_summary_by_alpha_o_method.csv")
    print("\n=== Updated table ingredients (α=0.2) ===")
    for method in ["GHCP", "Std-CP", "HCP"]:
        print(f"\n{method}:")
        sub = s[s["method"] == method].sort_values("o")
        for _, row in sub.iterrows():
            o = int(row["o"])
            cov = row["coverage_mean"]
            cse = row["coverage_se"]
            med = row["width_median"]
            wse = row["width_se"]
            n_inf = row["width_n_infinite"]
            if method == "Std-CP" and o == 0:
                print(f"  o={o}: cov={cov:.3f} ({cse:.3f}), width=∞  (n_inf={int(n_inf)})")
            elif pd.isna(med) or (method == "HCP" and o > 0):
                continue
            else:
                print(
                    f"  o={o}: cov={cov:.3f} ({cse:.3f}), "
                    f"med_width={med:,.0f} ({wse:,.0f})"
                )


if __name__ == "__main__":
    main()
