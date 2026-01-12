"""
Blood Pressure Data Loader

This script loads and combines the blood pressure Excel files from the
Ghana hypertension control trial (https://datadryad.org/dataset/doi:10.5061/dryad.16c9m51).

The dataset consists of 4 Excel files:
- Clinical Data.xlsx
- Demographic Data.xlsx  
- Observed Outcome Data.xlsx
- Site Characteristic Data.xlsx

Usage:
    python load_bp_data.py --data_dir path/to/excel/files --output bp_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_bp_excel_files(
    data_dir,
    output_path='data/bp_data.csv'
):
    """
    Load and combine blood pressure Excel files.

    Parameters:
    -----------
    data_dir : str
        Directory containing the Excel files
    output_path : str
        Where to save the combined CSV file

    Returns:
    --------
    pd.DataFrame : Combined blood pressure data
    """
    data_dir = Path(data_dir)

    print("=" * 80)
    print("LOADING BLOOD PRESSURE DATA FROM EXCEL FILES")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")

    # Expected file names (with + instead of spaces in actual files)
    files = {
        'clinical': 'Clinical+Data.xlsx',
        'demographic': 'Demographic+Data.xlsx',
        'outcome': 'Observed+Outcome+Data.xlsx',
        'site': 'Site+Characteristic+Data.xlsx'
    }

    # Load each file
    data = {}
    for key, filename in files.items():
        filepath = data_dir / filename
        print(f"\nLoading {filename}...", end=' ')

        if not filepath.exists():
            print(f"✗ File not found")
            print(f"  Expected: {filepath}")
            continue

        try:
            df = pd.read_excel(filepath)
            data[key] = df
            print(f"✓ ({len(df)} rows, {len(df.columns)} columns)")
            print(f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        except Exception as e:
            print(f"✗ Error: {e}")

    if not data:
        raise ValueError("No data files loaded!")

    # Try to merge the dataframes
    # This will depend on the actual structure of the files
    # We'll use a patient ID or similar key if available

    print("\n" + "=" * 80)
    print("COMBINING DATAFRAMES")
    print("=" * 80)

    # Start with clinical data if available
    if 'clinical' in data:
        combined = data['clinical'].copy()
        print(f"\nStarting with clinical data: {len(combined)} rows")

        # Try to merge with demographic data
        if 'demographic' in data:
            # Look for common ID column
            common_cols = set(combined.columns) & set(data['demographic'].columns)
            id_cols = [c for c in common_cols if 'id' in c.lower() or 'patient' in c.lower()]

            if id_cols:
                merge_key = id_cols[0]
                print(f"Merging with demographic data on '{merge_key}'...")
                combined = combined.merge(
                    data['demographic'],
                    on=merge_key,
                    how='left',
                    suffixes=('', '_demo')
                )
                print(f"  Result: {len(combined)} rows, {len(combined.columns)} columns")
            else:
                print("Warning: No common ID column found for demographic data")

        # Try to merge with outcome data
        if 'outcome' in data:
            common_cols = set(combined.columns) & set(data['outcome'].columns)
            id_cols = [c for c in common_cols if 'id' in c.lower() or 'patient' in c.lower()]

            if id_cols:
                merge_key = id_cols[0]
                print(f"Merging with outcome data on '{merge_key}'...")
                combined = combined.merge(
                    data['outcome'],
                    on=merge_key,
                    how='left',
                    suffixes=('', '_outcome')
                )
                print(f"  Result: {len(combined)} rows, {len(combined.columns)} columns")

        # Try to merge with site data
        if 'site' in data:
            common_cols = set(combined.columns) & set(data['site'].columns)
            site_cols = [c for c in common_cols if 'site' in c.lower() or 'clinic' in c.lower()]

            if site_cols:
                merge_key = site_cols[0]
                print(f"Merging with site data on '{merge_key}'...")
                combined = combined.merge(
                    data['site'],
                    on=merge_key,
                    how='left',
                    suffixes=('', '_site')
                )
                print(f"  Result: {len(combined)} rows, {len(combined.columns)} columns")

    else:
        # If clinical data not available, just use whatever we have
        combined = pd.concat(data.values(), axis=1)
        print(f"Combined all dataframes: {len(combined)} rows, {len(combined.columns)} columns")

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    combined.to_csv(output_path, index=False)
    print(f"✓ Saved ({output_path.stat().st_size / 1024:.1f} KB)")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    print(f"\nTotal observations: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")

    print("\nColumn names:")
    for i, col in enumerate(combined.columns, 1):
        print(f"  {i:2d}. {col}")

    # Look for key columns
    key_patterns = ['clinic', 'site', 'sbp', 'systolic', 'baseline', 'followup',
                   'treatment', 'arm', 'age', 'sex', 'bmi']

    print("\nKey columns found:")
    for pattern in key_patterns:
        matching = [c for c in combined.columns if pattern.lower() in c.lower()]
        if matching:
            print(f"  {pattern}: {matching}")

    # Summary statistics for numeric columns
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumeric columns summary:")
        print(combined[numeric_cols].describe().round(2))

    print("\n" + "=" * 80)
    print(f"Data saved to: {output_path}")
    print("=" * 80)
    print("\nNOTE: Please review the combined data and adjust column names")
    print("in data_processing.py if needed to match the actual structure.")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Load and combine blood pressure Excel files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing Excel files')
    parser.add_argument('--output', type=str, default='data/bp_data.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    load_bp_excel_files(
        data_dir=args.data_dir,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
