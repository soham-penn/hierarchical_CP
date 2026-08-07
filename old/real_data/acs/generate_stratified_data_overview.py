#!/usr/bin/env python3
"""Generate a stratified ACS data-overview file for PUMA strata.

This script mirrors the filtering and BA+ share stratification used by
`real_data/repeated_experiments_stratified_acs.py` and writes an overview CSV
with one row per stratum.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parents[2]
sys.path.append(str(BASE_DIR / "real_data"))
sys.path.append(str(BASE_DIR))

from acs.data_processing import load_and_clean_acs_pums
from repeated_experiments_stratified_acs import (
    compute_group_share_baplus,
)


EDUCATION_CODEBOOK = (
    "ACS SCHL education codes recoded into 4 categories: "
    "<HS (educ code <=15)  |  HS (educ 16-17)  |  SomeCollege (educ 18-20)  |  BAplus (educ >=21)"
)


def _stratification_process_description() -> str:
    """Return a detailed description of the stratification process."""
    return (
        "STRATIFICATION PROCESS (step-by-step):\n"
        "1. Education variable: educ_level (4 categories: <HS, HS, SomeCollege, BAplus)\n"
        "2. Stratification target: BA+ share (proportion) within each PUMA\n"
        "3. Compute share_baplus for each PUMA: "
        "(# individuals with educ_level==BAplus) / (# total individuals in PUMA)\n"
        "4. Sort eligible PUMAs (n>=20 individuals) by share_baplus value\n"
        "5. Use pandas pd.qcut(share_baplus, q=5, duplicates='drop') to bin PUMAs "
        "into 5 equal-sized quantile strata\n"
        "6. Each stratum is one interval of share_baplus; "
        "equal-sized bins means 45 PUMAs / 5 strata = 9 per stratum\n"
        "7. Lower quantile = lower proportion of BA+ (less educated PUMAs)\n"
        "8. Higher quantile = higher proportion of BA+ (more educated PUMAs)"
    )


def _q_label(rank: int, n_quantiles: int) -> str:
    if rank == 1:
        return f"Q{rank} (lowest BA+ share: least educated PUMAs)"
    if rank == n_quantiles:
        return f"Q{rank} (highest BA+ share: most educated PUMAs)"
    return f"Q{rank}"


def build_overview(
    data_path: Path,
    state: str,
    min_group_size: int,
    n_strata: int,
    age_min,
    age_max,
    yoep_min_year,
    yoep_window_years,
    min_hours,
    min_income,
    bottom_income_quantile,
    output_csv: Path,
) -> pd.DataFrame:
    df = load_and_clean_acs_pums(
        str(data_path),
        states_keep=[state],
        age_min=age_min,
        age_max=age_max,
        yoep_window_years=yoep_window_years,
        yoep_min_year=yoep_min_year,
        min_hours=min_hours,
        min_income=min_income,
        top_income_quantile=None,
        bottom_income_quantile=bottom_income_quantile,
    )

    if "puma" not in df.columns:
        raise ValueError("ACS source must contain PUMA to build a stratified overview")

    df = df.dropna(subset=["puma"]).copy()
    try:
        df["puma"] = df["puma"].astype(int)
    except Exception:
        df["puma"] = df["puma"].astype(str)

    counts_all = df.groupby("puma").size().sort_values(ascending=True)
    eligible = counts_all[counts_all >= min_group_size]
    eligible_groups = eligible.index.to_numpy().tolist()

    df_eligible = df[df["puma"].isin(eligible_groups)].copy()

    group_share_baplus = compute_group_share_baplus(df_eligible, "puma")

    # Match the experiment logic exactly: quantile-based bins of PUMA BAplus share.
    shares = pd.Series({g: group_share_baplus[g] for g in eligible_groups}, dtype=float)
    bins = pd.qcut(shares, q=int(n_strata), duplicates="drop")
    categories = list(bins.cat.categories)

    group_income_median = df_eligible.groupby("puma")["income"].median().to_dict()

    n_eligible_total = int(len(eligible_groups))
    n_strata_realized = int(len(categories))
    per_stratum_if_even = (
        float(n_eligible_total) / float(n_strata_realized) if n_strata_realized > 0 else np.nan
    )
    created_quantile_note = (
        "Created quantile strata via pd.qcut on PUMA-level BAplus share; "
        "not natural ACS education categories"
    )

    rows = []
    for idx, interval in enumerate(categories, start=1):
        stratum_name = str(interval)
        members = shares.index[bins == interval].tolist()
        stratum_df = df_eligible[df_eligible["puma"].isin(members)]
        puma_level_medians = [group_income_median[g] for g in members if g in group_income_median]

        rows.append(
            {
                "STRATUM_RANK": int(idx),
                "STRATUM_INTERPRETATION": _q_label(idx, n_strata_realized),
                "stratum": stratum_name,
                "STRATUM_SHARE_BAPLUS_INTERVAL": stratum_name,
                "STRATUM_SHARE_BAPLUS_MIN": float(min([group_share_baplus[g] for g in members])) if len(members) else np.nan,
                "STRATUM_SHARE_BAPLUS_MAX": float(max([group_share_baplus[g] for g in members])) if len(members) else np.nan,
                "STRATIFICATION_VARIABLE": "share_baplus (computed as proportion of individuals with BA+ degree within each PUMA)",
                "EDUCATION_CATEGORY_USED": "BAplus (ACS educ code >= 21; Bachelor's degree or higher)",
                "EDUCATION_CATEGORIES_AVAILABLE": "<HS | HS | SomeCollege | BAplus",
                "EDUCATION_CODEBOOK": EDUCATION_CODEBOOK,
                "STRATIFICATION_METHOD": "Quantile binning via pandas pd.qcut(share_baplus, q=n_strata, duplicates='drop')",
                "STRATIFICATION_DESCRIPTION": _stratification_process_description(),
                "n_pumas": int(len(members)),
                "EQUAL_SIZE_EXPLANATION": (
                    f"pd.qcut targets equal counts: {n_eligible_total} eligible PUMAs / {n_strata_realized} strata = {per_stratum_if_even:.1f} per stratum. "
                    f"Each stratum has {int(len(members))} PUMAs."
                ),
                "median_income_individual_level": float(stratum_df["income"].median()) if len(stratum_df) else np.nan,
                "median_income_puma_median": float(np.median(puma_level_medians)) if len(puma_level_medians) else np.nan,
                "BA_PLUS_SHARE_MEDIAN_IN_STRATUM": float(np.median([group_share_baplus[g] for g in members])) if len(members) else np.nan,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("stratum").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = BASE_DIR

    parser.add_argument(
        "--data_path",
        type=str,
        default=str(base_dir / "real_data" / "acs" / "data" / "acs_data_all50states.csv"),
    )
    parser.add_argument("--state", type=str, default="CA")
    parser.add_argument("--min_group_size", type=int, default=20)
    parser.add_argument("--n_strata", type=int, default=5)
    parser.add_argument("--age_min", type=int, default=25)
    parser.add_argument("--age_max", type=int, default=54)
    parser.add_argument("--no_age_filter", action="store_true")
    parser.add_argument("--yoep_min_year", type=int, default=2012)
    parser.add_argument("--yoep_window_years", type=int, default=2)
    parser.add_argument("--min_hours", type=int, default=40)
    parser.add_argument("--min_income", type=float, default=10000.0)
    parser.add_argument("--bottom_income_quantile", type=float, default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(base_dir / "real_data" / "acs" / "results" / "stratified" / "acs_stratified_data_overview.csv"),
    )
    args = parser.parse_args()

    age_min = None if args.no_age_filter else args.age_min
    age_max = None if args.no_age_filter else args.age_max

    out_df = build_overview(
        data_path=Path(args.data_path),
        state=args.state.upper(),
        min_group_size=args.min_group_size,
        n_strata=args.n_strata,
        age_min=age_min,
        age_max=age_max,
        yoep_min_year=args.yoep_min_year,
        yoep_window_years=args.yoep_window_years,
        min_hours=args.min_hours,
        min_income=args.min_income,
        bottom_income_quantile=args.bottom_income_quantile,
        output_csv=Path(args.output_csv),
    )

    print("Saved:", args.output_csv)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
