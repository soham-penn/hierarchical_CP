"""
Blood Pressure Trial Data Processing Module

This module handles:
1. Loading blood pressure trial data
2. Filtering to treatment arm only
3. Creating train/test splits by clinic
4. Building design matrices for hierarchical CP methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def load_and_clean_bp_data(
    bp_csv_path: str,
    treatment_arm_only: bool = True,
    outcome_type: str = 'followup',  # 'followup' or 'reduction'
    min_clinic_size: int = 5
) -> pd.DataFrame:
    """
    Load and clean blood pressure trial data.

    Parameters:
    -----------
    bp_csv_path : str
        Path to blood pressure CSV file
    treatment_arm_only : bool
        Keep only treatment arm (default: True)
    outcome_type : str
        'followup' for follow-up SBP, 'reduction' for SBP reduction (default: 'followup')
    min_clinic_size : int
        Minimum clinic size to keep (default: 5)

    Returns:
    --------
    pd.DataFrame : Cleaned data with columns: clinic_id, y, baseline_sbp, age, etc.
    """
    # Load data
    df = pd.read_csv(bp_csv_path)
    print(f"Loaded {len(df)} rows from {bp_csv_path}")

    # Expected columns (adjust based on actual data structure):
    # - clinic_id or clinic or center: clinic identifier
    # - treatment or arm: treatment indicator (0/1 or placebo/treatment)
    # - baseline_sbp or sbp_baseline: baseline systolic blood pressure
    # - followup_sbp or sbp_followup: follow-up systolic blood pressure
    # - age: age
    # - sex or gender: sex/gender
    # - bmi: body mass index
    # - diabetes: diabetes indicator
    # ... other covariates ...

    # Standardize column names (adjust based on your data)
    col_rename = {}
    followup_assigned = False  # Track if we already assigned followup_sbp

    for col in df.columns:
        lower_col = col.lower()
        # Clinic/Site ID
        if 'site' in lower_col and 'number' in lower_col:
            col_rename[col] = 'clinic_id'
        elif 'clinic' in lower_col or 'center' in lower_col:
            col_rename[col] = 'clinic_id'
        # Treatment
        elif 'treatment' in lower_col or 'arm' in lower_col or ('tx' in lower_col and 'ctrl' in lower_col):
            col_rename[col] = 'treatment'
        # Baseline SBP
        elif 'sbp' in lower_col and 'baseline' in lower_col:
            col_rename[col] = 'baseline_sbp'
        # Follow-up SBP - prioritize 12 months over 6 months, only assign once
        elif not followup_assigned:
            if 'sbp' in lower_col and '12' in lower_col and 'month' in lower_col:
                col_rename[col] = 'followup_sbp'
                followup_assigned = True
            elif 'sbp' in lower_col and '6' in lower_col and 'month' in lower_col:
                col_rename[col] = 'followup_sbp'
                followup_assigned = True
        # Age
        elif col.lower() == 'age':
            col_rename[col] = 'age'
        # Gender
        elif col.lower() in ['sex', 'gender']:
            col_rename[col] = 'sex'
        # BMI
        elif col.lower() == 'bmi':
            col_rename[col] = 'bmi'

    if col_rename:
        df = df.rename(columns=col_rename)

    print(f"Columns after renaming: {list(df.columns)}")

    # Filter to treatment arm if requested
    if treatment_arm_only and 'treatment' in df.columns:
        # Assuming treatment==1 or treatment=='treatment' indicates treatment arm
        if df['treatment'].dtype == 'object':
            df = df[df['treatment'].str.lower().str.contains('treatment')]
        else:
            df = df[df['treatment'] == 1]
        print(f"After filtering to treatment arm: {len(df)} rows")

    # Check required columns exist
    required = ['clinic_id', 'baseline_sbp', 'followup_sbp']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop missing values in key columns
    df = df.dropna(subset=['clinic_id', 'baseline_sbp', 'followup_sbp'])
    print(f"After dropping missing values: {len(df)} rows")

    # Create outcome
    if outcome_type == 'followup':
        df['y'] = df['followup_sbp']
    elif outcome_type == 'reduction':
        df['y'] = df['baseline_sbp'] - df['followup_sbp']
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")

    # Filter clinics by minimum size
    clinic_counts = df['clinic_id'].value_counts()
    valid_clinics = clinic_counts[clinic_counts >= min_clinic_size].index
    df = df[df['clinic_id'].isin(valid_clinics)]
    print(f"After filtering clinics (min size {min_clinic_size}): {len(df)} rows across {len(valid_clinics)} clinics")

    # Create derived features if needed
    if 'age' in df.columns:
        df['age_sq'] = df['age'] ** 2
    if 'bmi' in df.columns:
        df['bmi_sq'] = df['bmi'] ** 2

    # Binary indicators
    if 'sex' in df.columns:
        # Assume 0=male, 1=female or M/F
        if df['sex'].dtype == 'object':
            df['female'] = (df['sex'].str.upper() == 'F').astype(int)
        else:
            df['female'] = df['sex'].astype(int)

    print("\nFinal clinic distribution:")
    print(df['clinic_id'].value_counts().sort_index())

    return df


def build_design_matrix_bp(df: pd.DataFrame) -> np.ndarray:
    """
    Build design matrix X from cleaned blood pressure data.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned blood pressure data

    Returns:
    --------
    np.ndarray : Design matrix of shape (n, p)
    """
    # Baseline features to include
    feature_cols = ['baseline_sbp']

    # Add other available features
    optional_cols = ['age', 'age_sq', 'bmi', 'bmi_sq', 'female']
    for col in optional_cols:
        if col in df.columns:
            feature_cols.append(col)

    # Handle categorical variables if present (e.g., diabetes, race)
    cat_cols = []
    for col in ['diabetes', 'race', 'ethnicity']:
        if col in df.columns:
            cat_cols.append(col)

    if cat_cols:
        # Create dummy variables
        dummies = pd.get_dummies(df[cat_cols], drop_first=True, prefix=cat_cols)
        X = pd.concat([df[feature_cols], dummies], axis=1)
    else:
        X = df[feature_cols]

    # Drop any remaining NaNs
    X = X.fillna(X.mean())

    return X.values


def create_bp_hierarchical_data(
    df: pd.DataFrame,
    X: np.ndarray,
    test_clinics: List,
    n_per_cal_group: int = 20,
    n_test_group: int = 25,
    o_observed: int = 12,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Create hierarchical data structures for blood pressure experiment.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned blood pressure data
    X : np.ndarray
        Design matrix
    test_clinics : List
        Clinic IDs to use as test groups
    n_per_cal_group : int
        Number of observations per calibration group (default: 20)
    n_test_group : int
        Number of observations in test group (default: 25)
    o_observed : int
        Number of observed points in test group (default: 12)
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    Dict : Dictionary with results for each test clinic
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Training clinics (calibration)
    all_clinics = df['clinic_id'].unique()
    train_clinics = [c for c in all_clinics if c not in test_clinics]

    results = {}

    for test_clinic in test_clinics:
        # Check if test clinic has enough data
        test_indices = np.where(df['clinic_id'] == test_clinic)[0]
        if len(test_indices) < (o_observed + 1):
            print(f"Warning: Test clinic {test_clinic} has only {len(test_indices)} obs, need {o_observed + 1}")
            continue

        # Create calibration groups (one per training clinic)
        Z_calibration = []
        cal_clinics_used = []

        for clinic in train_clinics:
            clinic_indices = np.where(df['clinic_id'] == clinic)[0]
            if len(clinic_indices) < 2:
                continue

            # Sample n_per_cal_group observations (or all if fewer)
            n_sample = min(n_per_cal_group, len(clinic_indices))
            sampled_indices = np.random.choice(clinic_indices, size=n_sample, replace=False)

            # Create Z list for this group
            Z_group = []
            for idx in sampled_indices:
                Z_group.append({
                    'X': X[idx, :],
                    'Y': df.iloc[idx]['y']
                })

            Z_calibration.append(Z_group)
            cal_clinics_used.append(clinic)

        # Create test group
        n_sample_test = min(n_test_group, len(test_indices))
        if n_sample_test < (o_observed + 1):
            print(f"Warning: Cannot create test group for clinic {test_clinic}")
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

        # U vectors (constant 0 to avoid leakage)
        U_calibration = np.zeros((len(Z_calibration), 1))
        U_test = np.zeros((1, 1))

        results[test_clinic] = {
            'U_calibration': U_calibration,
            'Z_calibration': Z_calibration,
            'U_test': U_test,
            'Z_test': Z_test,
            'cal_clinics_used': cal_clinics_used,
            'n_cal_groups': len(Z_calibration),
            'test_clinic': test_clinic
        }

    return results


def summarize_bp_data(df: pd.DataFrame, test_clinics: List):
    """
    Print summary statistics of the blood pressure data.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned blood pressure data
    test_clinics : List
        Test clinics to highlight
    """
    print("\n" + "=" * 80)
    print("BLOOD PRESSURE DATA SUMMARY")
    print("=" * 80)

    print(f"\nTotal observations: {len(df)}")
    print(f"Total clinics: {df['clinic_id'].nunique()}")

    print("\nClinic distribution:")
    clinic_counts = df['clinic_id'].value_counts().sort_index()
    for clinic in clinic_counts.index:
        marker = " (TEST)" if clinic in test_clinics else ""
        print(f"  Clinic {clinic}: {clinic_counts[clinic]:3d}{marker}")

    print(f"\nOutcome (y) statistics:")
    print(f"  Mean: {df['y'].mean():.2f}")
    print(f"  Std:  {df['y'].std():.2f}")
    print(f"  Min:  {df['y'].min():.2f}")
    print(f"  Max:  {df['y'].max():.2f}")

    print(f"\nBaseline SBP statistics:")
    print(f"  Mean: {df['baseline_sbp'].mean():.2f}")
    print(f"  Std:  {df['baseline_sbp'].std():.2f}")

    print(f"\nFollow-up SBP statistics:")
    print(f"  Mean: {df['followup_sbp'].mean():.2f}")
    print(f"  Std:  {df['followup_sbp'].std():.2f}")

    if 'age' in df.columns:
        print(f"\nAge statistics:")
        print(f"  Mean: {df['age'].mean():.1f}")
        print(f"  Range: [{df['age'].min()}, {df['age'].max()}]")

    if 'female' in df.columns:
        print(f"\nGender:")
        print(f"  Female: {df['female'].mean():.1%}")

    if 'bmi' in df.columns:
        print(f"\nBMI statistics:")
        print(f"  Mean: {df['bmi'].mean():.1f}")
        print(f"  Std:  {df['bmi'].std():.1f}")

    print("=" * 80 + "\n")


def select_test_clinics(df: pd.DataFrame, n_test_clinics: int = 5,
                       selection_method: str = 'random',
                       random_seed: Optional[int] = None) -> List:
    """
    Select test clinics from the data.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned blood pressure data
    n_test_clinics : int
        Number of test clinics to select (default: 5)
    selection_method : str
        Selection method: 'random', 'largest', 'smallest' (default: 'random')
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    List : List of test clinic IDs
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    clinic_counts = df['clinic_id'].value_counts()

    if selection_method == 'random':
        test_clinics = np.random.choice(
            clinic_counts.index,
            size=min(n_test_clinics, len(clinic_counts)),
            replace=False
        ).tolist()
    elif selection_method == 'largest':
        test_clinics = clinic_counts.nlargest(n_test_clinics).index.tolist()
    elif selection_method == 'smallest':
        test_clinics = clinic_counts.nsmallest(n_test_clinics).index.tolist()
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")

    return test_clinics
