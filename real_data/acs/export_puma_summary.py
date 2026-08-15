#!/usr/bin/env python3
"""Export PUMA-level summary table for the ACS true-marginal experiment cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
REAL_DATA_DIR = REPO_ROOT / "real_data"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REAL_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DATA_DIR))

from acs.data_processing import load_and_clean_acs_pums  # noqa: E402

DEFAULT_DHCP_WIDTH_O20 = {
    "ols_trim2_corr": 120_067.0,
    "rf_trim2_corr": 111_566.0,
}
MIN_PUMA_SIZE = 21


def _build_strata(shares: pd.Series, n_strata: int = 5) -> pd.Series:
    bins = pd.qcut(shares, q=n_strata, duplicates="drop")
    return bins.astype(str)


def export_puma_summary(
    *,
    acs_csv: Path,
    state: str = "CA",
    drop_top_income_pct: float = 0.02,
    min_puma_size: int = MIN_PUMA_SIZE,
    out_path: Path,
) -> pd.DataFrame:
    df = load_and_clean_acs_pums(
        str(acs_csv),
        states_keep=[state],
        age_min=25,
        age_max=54,
        yoep_min_year=2012,
        min_hours=40,
        min_income=10_000.0,
        drop_top_income_fraction=drop_top_income_pct if drop_top_income_pct > 0 else None,
        y_transform=lambda x: np.asarray(x, dtype=float),
    )
    df = df.dropna(subset=["puma"]).copy()
    df["puma"] = df["puma"].astype(int)

    shares = df.groupby("puma")["educ_level"].apply(lambda s: float((s == "BAplus").mean()))
    strata = _build_strata(shares)
    strata.name = "baplus_stratum"

    rows = []
    for puma, grp in df.groupby("puma"):
        income = grp["income"].astype(float)
        n = len(grp)
        row = {
            "puma": int(puma),
            "n_obs": n,
            "eligible_min_size": n >= min_puma_size,
            "target_eligible": n >= min_puma_size,
            "baplus_share": float(shares[puma]),
            "baplus_stratum": strata[puma],
            "income_min": float(income.min()),
            "income_median": float(income.median()),
            "income_mean": float(income.mean()),
            "income_max": float(income.max()),
            "income_range": float(income.max() - income.min()),
            "income_iqr": float(income.quantile(0.75) - income.quantile(0.25)),
        }
        for label, width in DEFAULT_DHCP_WIDTH_O20.items():
            row[f"dhcp_width_o20_{label}"] = width
            row[f"width_over_puma_range_{label}"] = (
                width / row["income_range"] if row["income_range"] > 0 else np.nan
            )
            row[f"width_over_puma_iqr_{label}"] = (
                width / row["income_iqr"] if row["income_iqr"] > 0 else np.nan
            )
            row[f"width_over_puma_median_{label}"] = (
                width / row["income_median"] if row["income_median"] > 0 else np.nan
            )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("puma").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acs_csv",
        type=Path,
        default=REPO_ROOT / "real_data" / "acs" / "data" / "acs_data_all50states.csv",
    )
    parser.add_argument("--state", default="CA")
    parser.add_argument("--drop_top_income_pct", type=float, default=0.02)
    parser.add_argument("--min_puma_size", type=int, default=MIN_PUMA_SIZE)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "real_data" / "acs" / "data" / "acs_ca_puma_summary.csv",
    )
    args = parser.parse_args()
    out = export_puma_summary(
        acs_csv=args.acs_csv,
        state=args.state,
        drop_top_income_pct=args.drop_top_income_pct,
        min_puma_size=args.min_puma_size,
        out_path=args.out,
    )
    eligible = out[out["eligible_min_size"]]
    print(f"Wrote {len(out)} PUMAs ({len(eligible)} eligible) to {args.out}")
    print(
        f"  Income range across PUMAs: "
        f"min=${eligible['income_min'].min():,.0f}, "
        f"max=${eligible['income_max'].max():,.0f}"
    )
    print(
        f"  Median PUMA income range: ${eligible['income_range'].median():,.0f} "
        f"(IQR ${eligible['income_iqr'].median():,.0f})"
    )
    for label, width in DEFAULT_DHCP_WIDTH_O20.items():
        ratio = width / eligible["income_range"].median()
        print(f"  GHCP width o=20 ({label}) ${width:,.0f} / median PUMA range = {ratio:.2f}x")


if __name__ == "__main__":
    main()
