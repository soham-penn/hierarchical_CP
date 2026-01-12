"""
ACS Data Loader using Folktables

This script uses the folktables package to download and prepare ACS PUMS data
for the hierarchical conformal prediction experiments.

Installation:
    pip install folktables

Usage:
    python load_acs_data.py --year 2018 --output acs_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

try:
    from folktables import ACSDataSource
except ImportError:
    print("Error: folktables package not installed.")
    print("Install it with: pip install folktables")
    exit(1)

# Import state lists from data_processing
from data_processing import EMERGING_STATES, DEFAULT_TEST_STATES
TEST_STATES = DEFAULT_TEST_STATES
# Calculate training states
TRAINING_STATES = sorted([s for s in EMERGING_STATES if s not in TEST_STATES])


def download_acs_data(
    survey_year='2018',
    horizon='1-Year',
    output_path='data/acs_data.csv',
    states=None
):
    """
    Download ACS PUMS data using folktables.

    Parameters:
    -----------
    survey_year : str
        Year of ACS survey (default: '2018')
    horizon : str
        '1-Year' or '5-Year' (default: '1-Year')
    output_path : str
        Where to save the CSV file (default: 'data/acs_data.csv')
    states : list, optional
        List of state abbreviations (default: EMERGING_STATES)

    Returns:
    --------
    pd.DataFrame : Combined ACS data
    """
    if states is None:
        states = EMERGING_STATES

    print("=" * 80)
    print("DOWNLOADING ACS PUMS DATA USING FOLKTABLES")
    print("=" * 80)
    print(f"\nSurvey year: {survey_year}")
    print(f"Horizon: {horizon}")
    print(f"States to download: {len(states)}")
    print(f"  Emerging states: {states}")

    # Create data source
    data_source = ACSDataSource(
        survey_year=survey_year,
        horizon=horizon,
        survey='person'
    )

    # Download data for all states
    all_data = []

    for i, state in enumerate(states, 1):
        print(f"\n[{i}/{len(states)}] Downloading {state}...", end=' ')
        try:
            state_data = data_source.get_data(states=[state], download=True)
            all_data.append(state_data)
            print(f"✓ ({len(state_data):,} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    if not all_data:
        raise ValueError("No data downloaded!")

    # Combine all states
    print(f"\nCombining data from {len(all_data)} states...")
    combined_data = pd.concat(all_data, ignore_index=True)

    print(f"Total rows: {len(combined_data):,}")
    print(f"Total columns: {len(combined_data.columns)}")

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    combined_data.to_csv(output_path, index=False)
    print(f"✓ Saved ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    # Map state FIPS to abbreviations
    state_fips_map = {
        1: 'AL', 4: 'AZ', 5: 'AR', 8: 'CO', 10: 'DE', 13: 'GA', 15: 'HI', 16: 'ID',
        18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 23: 'ME', 27: 'MN', 28: 'MS',
        29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH', 35: 'NM', 37: 'NC',
        38: 'ND', 40: 'OK', 41: 'OR', 45: 'SC', 46: 'SD', 47: 'TN', 49: 'UT',
        53: 'WA', 54: 'WV', 56: 'WY'
    }

    if 'ST' in combined_data.columns:
        combined_data['state_abb'] = combined_data['ST'].map(state_fips_map)
        state_counts = combined_data['state_abb'].value_counts().sort_index()

        print("\nObservations by state:")
        for state in state_counts.index:
            if state in TEST_STATES:
                marker = " (TEST)"
            elif state in TRAINING_STATES:
                marker = " (TRAIN)"
            else:
                marker = ""
            print(f"  {state}: {state_counts[state]:6,}{marker}")

    # Key columns
    key_columns = ['AGEP', 'SEX', 'PINCP', 'WKHP', 'SCHL', 'MAR', 'NATIVITY', 'YOEP']
    available_columns = [c for c in key_columns if c in combined_data.columns]

    print(f"\nKey columns available: {available_columns}")

    if 'PINCP' in combined_data.columns:
        income = combined_data['PINCP'].dropna()
        print(f"\nIncome (PINCP) summary:")
        print(f"  Mean: ${income.mean():,.0f}")
        print(f"  Median: ${income.median():,.0f}")
        print(f"  90th percentile: ${income.quantile(0.9):,.0f}")

    if 'AGEP' in combined_data.columns:
        print(f"\nAge (AGEP) summary:")
        print(f"  Mean: {combined_data['AGEP'].mean():.1f}")
        print(f"  Range: [{combined_data['AGEP'].min()}, {combined_data['AGEP'].max()}]")

    print("\n" + "=" * 80)
    print(f"Data saved to: {output_path}")
    print("=" * 80)

    return combined_data


def main():
    parser = argparse.ArgumentParser(
        description='Download ACS PUMS data using folktables',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--year', type=str, default='2018',
                       help='Survey year')
    parser.add_argument('--horizon', type=str, default='1-Year',
                       choices=['1-Year', '5-Year'],
                       help='Survey horizon')
    parser.add_argument('--output', type=str, default='data/acs_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--all_states', action='store_true',
                       help='Download all 50 states (instead of just emerging states)')

    args = parser.parse_args()

    # Determine which states to download
    if args.all_states:
        # All state abbreviations
        states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]
    else:
        states = EMERGING_STATES

    download_acs_data(
        survey_year=args.year,
        horizon=args.horizon,
        output_path=args.output,
        states=states
    )


if __name__ == "__main__":
    main()
