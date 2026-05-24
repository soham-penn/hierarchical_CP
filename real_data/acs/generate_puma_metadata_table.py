import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Allow running from anywhere while importing project modules.
THIS_DIR = Path(__file__).resolve().parent
REAL_DATA_DIR = THIS_DIR.parent
sys.path.append(str(REAL_DATA_DIR))

from acs.data_processing import load_and_clean_acs_pums  # noqa: E402


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    educ_dummies = pd.get_dummies(df["educ_level"], prefix="educ", drop_first=True)
    english_dummies = pd.get_dummies(df["english"], prefix="eng", drop_first=True)
    cow_dummies = pd.get_dummies(df["cow"], prefix="cow", drop_first=True)

    features = pd.concat(
        [
            df[["age", "age_sq", "hours", "entry_recency", "married", "female"]],
            educ_dummies,
            english_dummies,
            cow_dummies,
        ],
        axis=1,
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ACS PUMA eligibility metadata table.")
    parser.add_argument("--state", type=str, default="CA")
    parser.add_argument("--min_group_size", type=int, default=5)
    parser.add_argument("--n_groups", type=int, default=30)
    parser.add_argument("--group_seed", type=int, default=42)
    parser.add_argument("--yoep_window_years", type=int, default=2)
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(REAL_DATA_DIR / "acs/data/acs_data_all50states.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(REAL_DATA_DIR / "acs/results/acs_puma_eligibility_metadata.csv"),
    )
    args = parser.parse_args()

    state = args.state.upper()
    state_fips = 6 if state == "CA" else np.nan

    # Raw state-level PUMA count (before cleaning filters).
    raw = pd.read_csv(args.input_csv, usecols=["ST", "PUMA"])
    raw_state = raw[raw["ST"] == state_fips] if not np.isnan(state_fips) else raw.iloc[0:0]
    n_rows_raw_state = int(len(raw_state))
    n_pumas_raw_state = int(raw_state["PUMA"].dropna().nunique()) if n_rows_raw_state > 0 else 0

    # Cleaned data with the same filtering used by bootstrap_new ACS-PUMA runs.
    df = load_and_clean_acs_pums(
        args.input_csv,
        states_keep=[state],
        yoep_window_years=args.yoep_window_years,
        top_income_quantile=None,
    )

    try:
        df["puma"] = df["puma"].astype(int)
    except Exception:
        pass

    n_rows_after_filters = int(len(df))
    n_pumas_after_filters = int(df["puma"].nunique())

    counts_all = df.groupby("puma").size().sort_values(ascending=True)
    counts_eligible = counts_all[counts_all >= args.min_group_size]
    eligible_pumas = counts_eligible.index.tolist()

    rng = np.random.default_rng(args.group_seed)
    sampled_non_test = rng.choice(
        np.array(eligible_pumas), size=args.n_groups, replace=False
    ).tolist()
    sampled_non_test_set = set(sampled_non_test)

    features = build_feature_frame(df)
    covariate_cols = features.columns.tolist()
    full = pd.concat([df[["puma", "y", "income"]], features], axis=1)
    full = full[full["puma"].isin(eligible_pumas)].copy()

    agg_spec = {
        "sample_size": ("puma", "size"),
        "outcome_y_mean": ("y", "mean"),
        "income_mean": ("income", "mean"),
    }
    for c in covariate_cols:
        agg_spec[f"{c}_mean"] = (c, "mean")

    table = full.groupby("puma").agg(**agg_spec).reset_index()
    table["is_eligible_size_ge_threshold"] = True
    table["is_sampled_non_test"] = table["puma"].isin(sampled_non_test_set)
    table["set_role"] = np.where(table["is_sampled_non_test"], "sampled_non_test", "test")

    # Attach run metadata columns for reproducibility.
    table["state"] = state
    table["state_fips"] = state_fips
    table["n_rows_raw_state"] = n_rows_raw_state
    table["n_pumas_raw_state"] = n_pumas_raw_state
    table["n_rows_after_filters"] = n_rows_after_filters
    table["n_pumas_after_filters"] = n_pumas_after_filters
    table["min_group_size_threshold"] = args.min_group_size
    table["n_eligible_pumas"] = int(len(counts_eligible))
    table["n_sampled_non_test_pumas"] = args.n_groups
    table["group_selection_seed"] = args.group_seed
    table["yoep_window_years"] = args.yoep_window_years
    table["filter_foreign_born_only"] = True
    table["filter_age_range"] = "25-54"
    table["filter_hours_min"] = 20
    table["filter_drop_nonpositive_income"] = True
    table["filter_top_income_quantile"] = "none"
    table["filter_drop_missing_key_columns"] = True
    table["filter_description"] = (
        "state==CA; foreign-born only; age 25-54; YOEP within latest 2 years; "
        "hours>=20; income>0; drop missing key columns; eligible PUMA size>=5"
    )

    ordered_front = [
        "state",
        "state_fips",
        "n_rows_raw_state",
        "n_pumas_raw_state",
        "n_rows_after_filters",
        "n_pumas_after_filters",
        "min_group_size_threshold",
        "n_eligible_pumas",
        "n_sampled_non_test_pumas",
        "group_selection_seed",
        "yoep_window_years",
        "filter_foreign_born_only",
        "filter_age_range",
        "filter_hours_min",
        "filter_drop_nonpositive_income",
        "filter_top_income_quantile",
        "filter_drop_missing_key_columns",
        "filter_description",
        "puma",
        "set_role",
        "is_sampled_non_test",
        "sample_size",
        "outcome_y_mean",
        "income_mean",
    ]
    remaining = [c for c in table.columns if c not in ordered_front]
    table = table[ordered_front + remaining].sort_values("puma").reset_index(drop=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"state={state}, raw_pumas={n_pumas_raw_state}, after_filters={n_pumas_after_filters}, eligible={len(counts_eligible)}, sampled_non_test={args.n_groups}")


if __name__ == "__main__":
    main()
