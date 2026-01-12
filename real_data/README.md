# Real Data Analysis for Hierarchical Conformal Prediction

This directory contains code for applying hierarchical conformal prediction methods to two real-world datasets:

1. **ACS (American Community Survey)**: Earnings prediction for recent immigrants across US states
2. **Blood Pressure Trial**: Systolic blood pressure prediction across clinic groups

## Directory Structure

```
real_data/
├── acs/
│   ├── data/                        # Data files (created after download)
│   ├── results/                     # Results organized by experiment type
│   │   ├── sequential/              # Sequential (online) experiments
│   │   └── marginal/                # Marginal coverage experiments
│   ├── load_acs_data.py             # Download ACS PUMS data using folktables
│   ├── data_processing.py           # Data loading, cleaning, and processing
│   ├── run_acs_sequential.py        # Sequential experiment runner
│   └── run_acs_marginal.py          # Marginal experiment runner
│
├── blood_pressure/
│   ├── data/                        # Data files (BP Excel files)
│   ├── results/marginal/            # Marginal coverage experiments
│   ├── load_bp_data.py              # Load and combine BP Excel files
│   ├── data_processing.py           # Data loading, cleaning, and processing
│   └── run_bp_marginal.py           # Marginal experiment runner
│
└── README.md                        # This file
```

## Methods Evaluated

All experiments compare 6 hierarchical conformal prediction methods:

### Proposed Methods
1. **HCP++** (HCP.plus): Donor group selection with adaptive calibration
2. **HCP.sample**: Sample-splitting method with test group data

### Baseline Methods
3. **HCP**: Standard hierarchical CP (max quantile across groups)
4. **Pooling**: Pool all calibration data (ignores group structure)
5. **Subsampling**: Sample once from each group
6. **Repeated**: Repeated subsampling with averaging

---

## Dataset 1: ACS (American Community Survey)

### Overview

**Goal**: Predict log-income for recent foreign-born immigrants across US states

**Data Source**:
- ACS PUMS 2018 (Public Use Microdata Sample)
- Downloaded via `folktables` library

### Population & Filters

**Target Population:**
- **Age**: 25-54 (working age)
- **Nativity**: Foreign-born (NATIVITY == 2)
- **Year of entry**: Recent immigrants (YOEP >= 2017)
- **Labor force**: Hours worked >= 20
- **Income filter**: Top 25% by income **per state** (ensures all states represented)

**After filtering:**
- 1,180 observations across 49 states (MT excluded due to insufficient data)
- 24 emerging destination test states
- 25 traditional destination training states

### Outcome and Covariates

**Outcome (Y)**: `log(income + 1)`
- Range: [9.95, 13.86] (≈3.9 units)
- Corresponds to income range: [$21K, $1.04M]
- Per-state filtering ensures income distribution varies by state

**Covariates (X)**: 19 features including age, education, hours worked, etc.

### Experimental Setup

#### Training/Calibration
- **Training states**: 25 states (CA, TX, NY, NJ, IL, MA, VA, MI, CT, OH, etc.)
  - Total: ~788 calibration observations across 25 states
- **Split**: First half (13 states) fits μ-model, second half (12 states) computes conformity scores
- **μ estimation**:
  - **Baseline methods**: Use global μ estimate only (no within-group adjustment)
  - **HCP++/HCP.sample**: Use global μ + within-group offset
- **Key**: Test state data NEVER used in baseline calibration

#### Marginal Coverage Calculation
For each test state:
1. **Order observations by income** (ascending)
2. For percentiles p ∈ {0, 25, 50, 75}:
   - **0th**: No history, predict 1st observation (poorest)
   - **25th**: Use bottom 25% as history, predict next
   - **50th**: Use bottom 50% as history, predict median
   - **75th**: Use bottom 75% as history, predict next
3. Aggregate coverage across all test states

**Proposed Methods**: Can use test state history for adaptation

### Usage

```bash
cd real_data/acs

# Step 1: Download data (all 50 states)
python3 load_acs_data.py --year 2018 --output data/acs_data_all50states.csv --all_states

# Step 2: Run marginal experiments
python3 run_acs_marginal.py data/acs_data_all50states.csv --top_income_pct 25 --alpha 0.1

# Step 3: Run sequential experiments
python3 run_acs_sequential.py data/acs_data_all50states.csv \
    --top_income_pct 50 --n_training_states 19 \
    --test_states SC AL TN DE AR --alpha 0.1
```

### Results

**Location**: `results/marginal/` or `results/sequential/`

**Summary file** columns:
- Percentile coverage (0, 25, 50, 75)
- Overall coverage (average)
- Mean/median interval width
- Proportion of infinite intervals
- Target coverage and difference

**Actual Results** (marginal, α=0.1, 90% target coverage):
- HCP++: 100% coverage, ~2.53 width, 6.7% infinite intervals
- HCP.sample: 87% coverage, ~1.90 width, 0% infinite
- HCP: 87% coverage, ~2.25 width, 0% infinite
- Pooling: 85% coverage, ~1.99 width, 0% infinite

**Key Findings**:
- HCP++ achieves target coverage with slight overcoverage
- HCP.sample has narrower intervals but slightly under target
- All methods perform reasonably well with appropriate width-coverage tradeoff

---

## Dataset 2: Blood Pressure

### Overview

**Goal**: Predict SBP at 12 months across clinic sites

**Data Source**: Hypertension intervention trial (treatment arm)

### Population
- Treatment arm only (605 observations)
- 32 clinics (15 test, 17 training)
- Minimum 5 participants per clinic

### Outcome and Covariates

**Outcome (Y)**: Follow-up SBP at 12 months (mmHg)
- Range: [90.7, 198.0] mmHg
- Mean: 139.4 mmHg, Std: 18.8 mmHg

**Covariates (X)**: Baseline SBP

### Experimental Setup

#### Training/Calibration
- **Training clinics**: 17 smaller/medium clinics (~277 observations)
- **Split**: First half fits μ-model, second half computes scores
- **Key**: Test clinic data NEVER used in baseline calibration

#### Marginal Coverage Calculation
For each test clinic:
1. **Order observations by baseline SBP** (ascending)
2. For percentiles p ∈ {0, 25, 50, 75}: predict observation at percentile p
3. Aggregate coverage across all test clinics

### Usage

```bash
cd real_data/blood_pressure

# Step 1: Load data
python3 load_bp_data.py data/ --output data/bp_data.csv

# Step 2: Run marginal experiments
python3 run_bp_marginal.py data/bp_data.csv --n_test_clinics 15 --alpha 0.2
```

### Results

**Actual Results** (marginal, α=0.2, 80% target coverage):
- HCP++: 87% coverage, ~50.5 mmHg width, 0% infinite
- HCP.sample: 90% coverage, ~48.3 mmHg width, 0% infinite
- HCP: 100% coverage, ~56.9 mmHg width, 0% infinite
- Pooling: 97% coverage, ~45.5 mmHg width, 0% infinite

**Key Findings**:
- HCP++ and HCP.sample achieve close to target coverage
- Baseline methods (HCP, Pooling, Subsampling, Repeated) are conservative (over-cover)
- HCP.sample has the narrowest intervals among non-conservative methods
- Wide intervals reflect limited training data (17 clinics) and high residual variance

---

## Experiment Types

### Sequential (Online)
- Observations arrive over time
- Predict each new observation using all prior data
- Coverage tested at each time point

### Marginal
- Test coverage at specific covariate quantiles
- Order by key variable (income, baseline SBP)
- Predict at 0th, 25th, 50th, 75th percentiles
- Ensures coverage across covariate distribution

---

## Key Implementation Details

### Calibration Structure

**Baseline Methods (HCP, Pooling, Subsampling, Repeated):**
- Use ONLY training groups (never see test group data)
- Split training groups: half for μ-model, half for scores

**Proposed Methods (HCP++, HCP.sample):**
- Use same baseline calibration
- Additionally receive test group history as input
- Can adapt based on test group characteristics

### Why Separate load_*.py and data_processing.py?

- **load_*.py**: Data acquisition (download, read files)
- **data_processing.py**: Data transformation (filter, clean, features)
- Separation: I/O vs business logic

### Results Files

**Detailed** (`*_detailed.csv`):
- One row per prediction
- All methods, all predictions
- For: plotting, analysis, debugging

**Summary** (`*_summary.csv`):
- One row per method
- Aggregated metrics
- For: tables, comparisons

---

## Troubleshooting

**"No training states"**: Ensure `states_keep=None` to load all 50 states

**Many infinite intervals**: Insufficient calibration data

**100% coverage**: Very wide intervals (conservative), check widths vs outcome range

---

## Citation

[To be added]

## Contact

[To be added]
