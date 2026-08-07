"""
One-time preprocessing script: extract the top-25% income ACS subset and save as CSV.

Run this once from the real_data/ directory:
    python3 acs/prepare_acs_data.py

Produces:
    acs/data/acs_top25_filtered.csv  -- top 25% income per state (~1180 rows)

This small CSV is used directly by bootstrap_proper.py, avoiding repeated
loading of the full 3.2M-row source file.
"""

import sys
from pathlib import Path

# Run from real_data/ directory
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from acs.data_processing import load_and_clean_acs_pums

DATA_DIR = Path(__file__).parent / 'data'
SRC_CSV = DATA_DIR / 'acs_data_all50states.csv'

print("=" * 60)
print("Preparing ACS top 25% income subset")
print("=" * 60)
df_top = load_and_clean_acs_pums(
    str(SRC_CSV),
    states_keep=None,
    top_income_quantile=0.75,   # keep top 25% => quantile threshold at 75th pct
)
out_top = DATA_DIR / 'acs_top25_filtered.csv'
df_top.to_csv(out_top, index=False)
print(f"\nSaved {len(df_top)} rows to {out_top}")

print("\nDone. Re-run this script only if the source data or filters change.")
