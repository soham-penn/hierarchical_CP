#!/usr/bin/env python3
"""
Download ACS PUMS (2018) for California and save in the format expected by
load_and_clean_acs_pums().

Output:
    real_data/acs/data/acs_data_all50states.csv

Requires: pip install folktables pandas
"""

from pathlib import Path

import pandas as pd

OUT_PATH = Path(__file__).resolve().parent / "data" / "acs_data_all50states.csv"

# PUMS columns required by load_and_clean_acs_pums()
REQUIRED_COLS = [
    "ST", "AGEP", "NATIVITY", "YOEP", "WKHP", "PINCP",
    "SEX", "SCHL", "MAR", "ENG", "COW", "PUMA",
]


def main():
    try:
        from folktables import ACSDataSource
    except ImportError as exc:
        raise SystemExit("Install folktables: pip install folktables") from exc

    print("Downloading ACS 2018 PUMS for California via folktables...")
    cache_dir = OUT_PATH.parent / "folktables_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = ACSDataSource(
        survey_year="2018",
        horizon="1-Year",
        survey="person",
        root_dir=str(cache_dir),
    )
    acs_df = data.get_data(states=["CA"], download=True)

    missing = [c for c in REQUIRED_COLS if c not in acs_df.columns]
    if missing:
        raise ValueError(f"Downloaded ACS frame missing columns: {missing}")

    out = acs_df[REQUIRED_COLS].copy()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
