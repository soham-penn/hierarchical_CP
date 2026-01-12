"""
ACS PUMS Data Processing Module

This module handles:
1. Loading ACS PUMS data
2. Filtering for recent immigrants in emerging destination states
3. Creating train/test splits by state
4. Building design matrices for hierarchical CP methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# State lists based on Migration Policy Institute (MPI)
NEW_DESTINATION_STATES = [
    "SC", "AL", "TN", "DE", "AR", "SD", "NV", "GA", "KY", "NC",
    "WY", "ID", "IN", "MS"
]

FASTEST_GROWING_FOREIGN_BORN_STATES = [
    "ND", "WV", "SD", "DE", "NE", "MN", "WY", "PA", "AK", "IN",
    "FL", "NV", "WA", "IA", "MD"
]

# Union of both lists (approximately 24 states)
EMERGING_STATES = sorted(list(set(NEW_DESTINATION_STATES + FASTEST_GROWING_FOREIGN_BORN_STATES)))

# Default test states (5 fastest-growing new destinations)
DEFAULT_TEST_STATES = ["SC", "AL", "TN", "DE", "AR"]


# FIPS code to state abbreviation mapping
FIPS_TO_STATE = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE',
    12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA',
    20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 25: 'MA', 26: 'MI',
    27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH',
    34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK',
    41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX',
    49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY'
}


def load_and_clean_acs_pums(
    pums_csv_path: str,
    states_keep: List[str] = None,
    age_min: int = 25,
    age_max: int = 54,
    yoep_window_years: int = 2,
    min_hours: int = 20,
    require_work_last_year: bool = True,
    drop_nonpositive_income: bool = True,
    top_income_quantile: float = None,
    y_transform = np.log1p
) -> pd.DataFrame:
    """
    Load and clean ACS PUMS data for recent immigrants.

    Parameters:
    -----------
    pums_csv_path : str
        Path to ACS PUMS CSV file
    states_keep : List[str]
        States to keep (default: None = all states)
    age_min : int
        Minimum age (default: 25)
    age_max : int
        Maximum age (default: 54)
    yoep_window_years : int
        Window for recent year of entry (default: 2 years)
    min_hours : int
        Minimum usual hours worked (default: 20)
    require_work_last_year : bool
        Require work last year (default: True)
    drop_nonpositive_income : bool
        Drop non-positive incomes (default: True)
    top_income_quantile : float
        If specified, keep only top X quantile by income (e.g., 0.90 for top 10%)
    y_transform : callable
        Transform for income (default: log1p)

    Returns:
    --------
    pd.DataFrame : Cleaned data with columns: state_abb, y, age, sex, educ, etc.
    """
    # states_keep = None means keep ALL states (no filtering)
    # To filter to emerging states only, pass states_keep=EMERGING_STATES explicitly

    # Column mapping (adjust based on your ACS PUMS file)
    col_map = {
        'ST': 'state_fips',
        'AGEP': 'age',
        'NATIVITY': 'nativity',
        'YOEP': 'yoep',
        'WKHP': 'hours',
        'PINCP': 'income',
        'SEX': 'sex',
        'SCHL': 'educ',
        'MAR': 'marital',
        'ENG': 'english',
        'COW': 'cow'
    }

    # Try to load only needed columns
    try:
        df = pd.read_csv(pums_csv_path, usecols=list(col_map.keys()))
        df = df.rename(columns=col_map)
    except:
        # If specific columns fail, load all and rename
        df = pd.read_csv(pums_csv_path)
        df = df.rename(columns=col_map)
        df = df[list(col_map.values())]

    print(f"Loaded {len(df)} rows from {pums_csv_path}")

    # Map FIPS to state abbreviation
    df['state_abb'] = df['state_fips'].map(FIPS_TO_STATE)
    df = df.dropna(subset=['state_abb'])

    if states_keep is not None:
        df = df[df['state_abb'].isin(states_keep)]
        print(f"After filtering to {len(states_keep)} specified states: {len(df)} rows")
    else:
        print(f"Keeping all states: {len(df)} rows")

    # Foreign-born (NATIVITY == 2 typically means foreign-born)
    df = df[df['nativity'] == 2]
    print(f"After filtering to foreign-born: {len(df)} rows")

    # Working-age
    df = df[(df['age'] >= age_min) & (df['age'] <= age_max)]
    print(f"After age filter [{age_min}, {age_max}]: {len(df)} rows")

    # Recent year of entry
    yoep_max = df['yoep'].max()
    yoep_cutoff = yoep_max - (yoep_window_years - 1)
    df = df[df['yoep'] >= yoep_cutoff]
    print(f"After recent entry filter (YOEP >= {yoep_cutoff}): {len(df)} rows")

    # Labor force attachment
    df = df[df['hours'] >= min_hours]
    df = df.dropna(subset=['income'])
    print(f"After labor force filter (hours >= {min_hours}): {len(df)} rows")

    if drop_nonpositive_income:
        df = df[df['income'] > 0]
        print(f"After dropping non-positive income: {len(df)} rows")

    # Top income quantile filter (applied per state)
    if top_income_quantile is not None:
        filtered_dfs = []
        for state in df['state_abb'].unique():
            state_df = df[df['state_abb'] == state]
            state_threshold = state_df['income'].quantile(top_income_quantile)
            state_filtered = state_df[state_df['income'] >= state_threshold]
            filtered_dfs.append(state_filtered)
        df = pd.concat(filtered_dfs, ignore_index=True)
        print(f"After top {int((1-top_income_quantile)*100)}% income filter (per state): {len(df)} rows across {df['state_abb'].nunique()} states")

    # Derived features
    df['entry_recency'] = yoep_max - df['yoep']
    df['age_sq'] = df['age'] ** 2

    # Recode education (SCHL codes)
    # Typically: 1-15 = <HS, 16-17 = HS, 18-20 = Some College, 21+ = BA+
    df['educ_level'] = pd.cut(
        df['educ'],
        bins=[-np.inf, 15, 17, 20, np.inf],
        labels=['<HS', 'HS', 'SomeCollege', 'BAplus']
    )

    # Binary indicators
    df['married'] = (df['marital'] == 1).astype(int)  # MAR==1 typically means married
    df['female'] = (df['sex'] == 2).astype(int)  # SEX==2 typically means female

    # Transform income to create outcome
    df['y'] = y_transform(df['income'])

    # Drop rows with any missing values in key columns
    key_cols = ['y', 'age', 'age_sq', 'hours', 'entry_recency', 'educ_level',
                'married', 'female', 'english', 'cow']
    df = df.dropna(subset=key_cols)
    print(f"After dropping missings in key columns: {len(df)} rows")

    # Check state counts
    print("\nFinal state counts:")
    print(df['state_abb'].value_counts().sort_index())

    return df


def build_design_matrix_acs(df: pd.DataFrame) -> np.ndarray:
    """
    Build design matrix X from cleaned ACS data.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ACS data

    Returns:
    --------
    np.ndarray : Design matrix of shape (n, p)
    """
    # Create dummy variables for categorical features
    educ_dummies = pd.get_dummies(df['educ_level'], prefix='educ', drop_first=True)
    english_dummies = pd.get_dummies(df['english'], prefix='eng', drop_first=True)
    cow_dummies = pd.get_dummies(df['cow'], prefix='cow', drop_first=True)

    # Combine all features
    X = pd.concat([
        df[['age', 'age_sq', 'hours', 'entry_recency', 'married', 'female']],
        educ_dummies,
        english_dummies,
        cow_dummies
    ], axis=1)

    return X.values


def create_acs_hierarchical_data(
    df: pd.DataFrame,
    X: np.ndarray,
    test_states: List[str],
    n_per_cal_group: int = 25,
    n_test_group: int = 30,
    o_observed: int = 15,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Create hierarchical data structures for one experiment replicate.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ACS data
    X : np.ndarray
        Design matrix
    test_states : List[str]
        States to use as test groups
    n_per_cal_group : int
        Number of observations per calibration group (default: 25)
    n_test_group : int
        Number of observations in test group (default: 30)
    o_observed : int
        Number of observed points in test group (default: 15)
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Dict : Dictionary with results for each test state
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Training states (calibration)
    train_states = [s for s in EMERGING_STATES if s not in test_states]

    results = {}

    for test_state in test_states:
        # Check if test state has enough data
        test_indices = np.where(df['state_abb'] == test_state)[0]
        if len(test_indices) < (o_observed + 1):
            print(f"Warning: Test state {test_state} has only {len(test_indices)} rows, need {o_observed + 1}")
            continue

        # Create calibration groups (one per training state)
        Z_calibration = []
        cal_states_used = []

        for state in train_states:
            state_indices = np.where(df['state_abb'] == state)[0]
            if len(state_indices) < 2:
                continue

            # Sample n_per_cal_group observations (or all if fewer)
            n_sample = min(n_per_cal_group, len(state_indices))
            sampled_indices = np.random.choice(state_indices, size=n_sample, replace=False)

            # Create Z list for this group
            Z_group = []
            for idx in sampled_indices:
                Z_group.append({
                    'X': X[idx, :],
                    'Y': df.iloc[idx]['y']
                })

            Z_calibration.append(Z_group)
            cal_states_used.append(state)

        # Create test group
        n_sample_test = min(n_test_group, len(test_indices))
        if n_sample_test < (o_observed + 1):
            print(f"Warning: Cannot create test group for {test_state}, insufficient data")
            continue

        sampled_test_indices = np.random.choice(test_indices, size=n_sample_test, replace=False)
        # Shuffle to create random "stream" order
        np.random.shuffle(sampled_test_indices)

        Z_test = []
        for idx in sampled_test_indices:
            Z_test.append({
                'X': X[idx, :],
                'Y': df.iloc[idx]['y']
            })

        # U vectors (constant 0 to avoid leakage, X contains demographics)
        U_calibration = np.zeros((len(Z_calibration), 1))
        U_test = np.zeros((1, 1))

        results[test_state] = {
            'U_calibration': U_calibration,
            'Z_calibration': Z_calibration,
            'U_test': U_test,
            'Z_test': Z_test,
            'cal_states_used': cal_states_used,
            'n_cal_groups': len(Z_calibration),
            'test_state': test_state
        }

    return results


def summarize_acs_data(df: pd.DataFrame, test_states: List[str]):
    """
    Print summary statistics of the ACS data.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ACS data
    test_states : List[str]
        Test states to highlight
    """
    print("\n" + "=" * 80)
    print("ACS DATA SUMMARY")
    print("=" * 80)

    print(f"\nTotal observations: {len(df)}")
    print(f"Total states: {df['state_abb'].nunique()}")

    print("\nState distribution:")
    state_counts = df['state_abb'].value_counts().sort_index()
    for state in state_counts.index:
        marker = " (TEST)" if state in test_states else ""
        print(f"  {state}: {state_counts[state]:4d}{marker}")

    print(f"\nOutcome (log1p(income)) statistics:")
    print(f"  Mean: {df['y'].mean():.3f}")
    print(f"  Std:  {df['y'].std():.3f}")
    print(f"  Min:  {df['y'].min():.3f}")
    print(f"  Max:  {df['y'].max():.3f}")

    print(f"\nAge statistics:")
    print(f"  Mean: {df['age'].mean():.1f}")
    print(f"  Range: [{df['age'].min()}, {df['age'].max()}]")

    print(f"\nEducation distribution:")
    print(df['educ_level'].value_counts())

    print(f"\nGender:")
    print(f"  Female: {df['female'].mean():.1%}")

    print(f"\nMarital status:")
    print(f"  Married: {df['married'].mean():.1%}")

    print("=" * 80 + "\n")
