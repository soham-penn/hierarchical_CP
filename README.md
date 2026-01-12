# Hierarchical Conformal Prediction - Python Implementation

Python implementation of hierarchical conformal prediction experiments comparing HCP++, HCP.sample, and baseline methods across both simulated (DGP) and real-world datasets.

## Overview

This repository implements and evaluates hierarchical conformal prediction methods for grouped/clustered data:

**Proposed Methods:**
- **HCP++** (HCP.plus): Donor group selection with adaptive calibration
- **HCP.sample**: Sample-splitting approach for hierarchical data

**Baseline Methods:**
- **HCP**: Standard hierarchical conformal prediction
- **Pooling**: Pool all calibration data (ignores group structure)
- **Subsampling**: Sample once from each group
- **Repeated**: Repeated subsampling across groups

## Repository Structure

```
hierarchical_new/
├── DGP/                          # Simulated data experiments
│   ├── dgp_specification.py      # DGP definitions (default & nonlinear)
│   ├── data_generation.py        # Calibration & test data generation
│   ├── summary_and_plots.py      # Plotting utilities
│   └── resultsDGP/               # Results storage
│       ├── files/                # Raw CSV results
│       ├── plots/                # Generated plots
│       └── summary_*.csv         # Summary statistics
│
├── methods/                      # Conformal prediction methods
│   ├── mu_methods.py             # μ-estimation (OLS and Random Forest)
│   ├── baseline_hcp.py           # HCP, pooling, subsampling, repeated
│   ├── hcp_plus.py               # HCP++ implementation
│   ├── hcp_sample.py             # HCP.sample implementation
│   └── experiments.py            # Experiment runner utilities
│
├── real_data/                    # Real data experiments
│   ├── acs/                      # ACS income prediction
│   │   ├── data/                 # Data files
│   │   ├── results/              # Results (sequential/ and marginal/)
│   │   ├── load_acs_data.py      # Download ACS PUMS data
│   │   ├── data_processing.py    # Data cleaning/filtering
│   │   ├── run_acs_sequential.py # Sequential experiments
│   │   └── run_acs_marginal.py   # Marginal experiments
│   │
│   ├── blood_pressure/           # BP clinical trial
│   │   ├── data/                 # Data files
│   │   ├── results/              # Results (marginal/)
│   │   ├── load_bp_data.py       # Load BP Excel files
│   │   ├── data_processing.py    # Data cleaning/filtering
│   │   └── run_bp_marginal.py    # Marginal experiments
│   │
│   └── README.md                 # Real data documentation
│
├── scores.py                     # Score functions & weighted quantile
├── run_experiments.py            # Main DGP experiment script
└── README.md                     # This file
```

## Installation

### Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib folktables
```

- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Machine learning models (Random Forest, OLS)
- `matplotlib`: Plotting
- `folktables`: ACS PUMS data download (for real data experiments)

### Setup

```bash
git clone <repository-url>
cd hierarchical_new
```

## Usage

### 1. Simulated Data (DGP) Experiments

Run experiments on simulated data to evaluate method performance under controlled conditions:

```bash
python run_experiments.py
```

**What this does:**
- Tests effect of number of observed points: `o ∈ {1, 15, 20, 50}`
- Tests effect of DGP: default (linear-ish) vs nonlinear
- Runs 25 replications per configuration
- Saves results to `DGP/resultsDGP/`

**Key parameters:**
- Calibration groups (K): 20
- Group sizes (Poisson λ): 20
- Miscoverage level (α): 0.1 (90% target coverage)
- Selection parameter (α_selection): 0.5
- Test groups: 100
- Random Forest trees: 50

### 2. Real Data Experiments

See [real_data/README.md](real_data/README.md) for detailed instructions on running experiments with:
- **ACS (American Community Survey)**: Income prediction for recent immigrants across US states
- **Blood Pressure Trial**: Systolic blood pressure prediction across clinic sites

Quick start:

```bash
# ACS experiments
cd real_data/acs
python3 load_acs_data.py --year 2018 --output data/acs_data_all50states.csv --all_states
python3 run_acs_marginal.py data/acs_data_all50states.csv --top_income_pct 25 --alpha 0.1

# Blood Pressure experiments
cd real_data/blood_pressure
python3 load_bp_data.py data/ --output data/bp_data.csv
python3 run_bp_marginal.py data/bp_data.csv --n_test_clinics 15 --alpha 0.2
```

## Experiment Types

### Sequential (Online) Experiments

Test observations arrive sequentially over time. Methods adapt as new observations are seen.

- **Setup**: Predict each new observation using all previously observed data
- **Coverage**: Tested at each time point
- **Use case**: Online learning, temporal data

### Marginal Coverage Experiments

Test coverage at specific quantiles of a key covariate.

- **Setup**:
  1. Order test group observations by a key variable (e.g., income, baseline SBP)
  2. For percentiles p ∈ {0, 25, 50, 75}:
     - Use observations before percentile p as history
     - Predict the observation at percentile p
     - Record coverage indicator
  3. Aggregate coverage across all test groups
- **Coverage**: Reported per percentile and overall
- **Use case**: Distribution-conditional coverage guarantees

## Methods Implementation

All methods are implemented in the `methods/` directory:

### Baseline Methods (`baseline_hcp.py`)

**HCP (Hierarchical CP):**
- Computes max residual quantile across calibration groups
- Conservative but valid under group exchangeability

**Pooling:**
- Pools all calibration data, ignoring group structure
- Assumes all groups are homogeneous

**Subsampling:**
- Samples one observation per group for calibration
- Reduces to standard split conformal

**Repeated Subsampling:**
- Averages quantiles over multiple subsamples
- More stable than single subsampling

### Proposed Methods

**HCP++ (`hcp_plus.py`):**
- Selects "donor groups" with similar group sizes to test group
- Uses test group history for better adaptation
- Key innovation: adaptive donor selection

**HCP.sample (`hcp_sample.py`):**
- Sample-splitting within groups for calibration
- Leverages test group data while maintaining validity
- Key innovation: within-group sample splitting

### μ-Estimation Methods (`mu_methods.py`)

Two approaches implemented:

**OLS (Ordinary Least Squares):**
- Fast, simple linear models
- Used for real data experiments (computational efficiency)
- Global model + group-level offset

**Random Forest:**
- Non-parametric, captures nonlinearities
- Used for DGP experiments
- Global model + group-level offset

## Results

### DGP Results

Saved to `DGP/resultsDGP/`:
- `files/`: Raw CSV results for each configuration
- `plots/`: Coverage and width comparison plots
- `summary_*.csv`: Aggregated statistics

### Real Data Results

**ACS Results** (`real_data/acs/results/`):
- `sequential/`: Sequential experiment results
  - `acs_sequential_detailed.csv`: Per-observation predictions
  - `acs_sequential_summary.csv`: Method summaries
- `marginal/`: Marginal experiment results
  - `acs_marginal_detailed.csv`: Per-percentile predictions
  - `acs_marginal_summary.csv`: Method summaries

**BP Results** (`real_data/blood_pressure/results/`):
- `marginal/`: Marginal experiment results
  - `bp_marginal_detailed.csv`: Per-observation predictions
  - `bp_marginal_summary.csv`: Method summaries

## Key Findings

### ACS Experiments
- **Setup**: 25 training states, 24 test states (emerging destinations)
  - 1,180 observations across 49 states
  - Per-state top 25% income filtering
- **Target**: 90% coverage (α=0.1)
- **Results**:
  - HCP++: 100% coverage, ~2.53 width, 6.7% infinite intervals
  - HCP.sample: 87% coverage, ~1.90 width, 0% infinite
  - HCP: 87% coverage, ~2.25 width, 0% infinite
  - Pooling: 85% coverage, ~1.99 width, 0% infinite

### Blood Pressure Experiments
- **Setup**: 17 training clinics, 15 test clinics
  - 605 observations across 32 clinics
- **Target**: 80% coverage (α=0.2)
- **Results**:
  - HCP++: 87% coverage, ~50.5 mmHg width, 0% infinite
  - HCP.sample: 90% coverage, ~48.3 mmHg width, 0% infinite
  - HCP: 100% coverage, ~56.9 mmHg width, 0% infinite
  - Pooling: 97% coverage, ~45.5 mmHg width, 0% infinite

## Technical Notes

### Conformal Prediction Guarantees

All methods provide **distribution-free finite-sample coverage guarantees** under:
- Exchangeability of calibration groups
- Test group exchangeable with calibration groups

### Group Structure

Methods handle hierarchical/grouped data where:
- **Groups**: States (ACS), Clinics (BP), or simulated clusters (DGP)
- **Within-group**: Individuals/observations
- **Challenge**: Limited data per group, heterogeneity across groups

### Computational Considerations

- **OLS**: Fast, scales well (used for real data)
- **Random Forest**: Slower, better for complex patterns (used for DGP)
- **Parallelization**: Not yet implemented (future work)

## Citation

[Add citation information when available]

## License

[Add license information]

## Contact

[Add contact information]
