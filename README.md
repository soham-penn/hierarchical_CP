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
hierarchical_CP/
├── DGP/                          # Simulated data experiments
│   ├── code/                     # DGP Python modules
│   │   ├── dgp_specification.py  # Generic DGP definitions
│   │   ├── data_generation.py    # Generic calibration/test generators
│   │   ├── experiments.py        # Core experiment runner
│   │   ├── summary_and_plots.py  # Summary helpers
│   │   ├── make_plots.py         # Standard effect-of-o plots
│   │   ├── joint_xy_marginal_common.py
│   │   ├── run_fixed_n51_k20_joint_xy_marginal.py
│   │   └── run_poisson_nmean21_k20_joint_xy_marginal.py
│   ├── results/                  # CSV outputs and summaries
│   │   └── files/                # Raw CSV results by run tag
│   └── plots/                    # Generated figures
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
│   │   ├── data/                 # Data files (excluded from git)
│   │   ├── results/              # Experimental results
│   │   ├── plots/                # Generated plots
│   │   ├── load_acs_data.py      # Download ACS PUMS data
│   │   ├── data_processing.py    # Data cleaning/filtering
│   │   └── run_acs.py            # Run ACS experiments
│   │
│   ├── blood_pressure/           # BP clinical trial
│   │   ├── data/                 # Data files (excluded from git)
│   │   ├── results/              # Experimental results
│   │   ├── plots/                # Generated plots
│   │   ├── load_bp_data.py       # Load BP Excel files
│   │   ├── data_processing.py    # Data cleaning/filtering
│   │   └── run_bp.py             # Run BP experiments
│   │
│   ├── plot_results.py           # Plotting utilities
│   ├── format_results.py         # Result formatting
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
git clone https://github.com/soham-penn/hierarchical_CP.git
cd hierarchical_CP
```

## Usage

### 1. Simulated Data (DGP) Experiments

The maintained DGP runners are:
- `DGP/code/run_fixed_n51_k20_joint_xy_marginal.py`
- `DGP/code/run_poisson_nmean21_k20_joint_xy_marginal.py`

Both use the same joint-Gaussian group model and differ only in group-size generation.

#### DGP setup (shared)

- Dimension: `d = 5`
- Number of calibration groups: `K = 20`
- Group covariate: `U_i ~ Unif([0,1]^5)`
- Joint data model per group:
  - `Z = (X, Y) ~ N(mu(U), Sigma(U))`
  - `mu(U) = (U_1^2, ..., U_5^2)`
  - `Sigma(U) = (1-rho) diag(U) + rho 11^T`, with `rho = 0.5`
- History sizes: `o ∈ {0, 5, 10, 15, 20, 25, 30, 35, 40}`
- Target index: `40` (0-based)
- Miscoverage level: `alpha = 0.2` (80% target coverage)
- Donor/sample selection parameter: `alpha_selection = 0.5`
- Repeated-subsampling baseline reps: `50`
- Test groups per experiment: `100`
- Number of outer experiments: `50`
- Mu estimators:
  - baselines: `create_mu_method_ols_global_only()`
  - hierarchical methods: `create_mu_method_ols_offset()`

#### Fixed-size run

```bash
.venv/bin/python -u DGP/code/run_fixed_n51_k20_joint_xy_marginal.py
```

- Group sizes: `N_i = 51` for all calibration and test groups.
- Output tag: `fixedN51_K20_d5_u01_rho05_joint_xy_marginal`

#### Poisson-size run

```bash
.venv/bin/python -u DGP/code/run_poisson_nmean21_k20_joint_xy_marginal.py
```

- Group sizes: `N_i = 1 + Poisson(20)`.
- Output tag: `poissonNmean21_K20_d5_u01_rho05_joint_xy_marginal`

#### Plot families produced for each run

- Set 1 (HCP as sole baseline, `o <= 20`):
  - `set1_side_by_side_dhcp_hcp_o_upto20.pdf`
- Set 2 (for each `o <= 40`: D-HCP, Std-CP, HCP):
  - `set2_coverage_dhcp_stdcp_hcp_by_o_upto40.pdf`
  - `set2_width_dhcp_stdcp_hcp_by_o_upto40.pdf`
- Set 3 (`o <= 40`, all baselines excluding Std-CP):
  - `set3_coverage_dhcp_with_baselines_no_stdcp_o_upto40.pdf`
  - `set3_width_dhcp_with_baselines_no_stdcp_o_upto40.pdf`
- Standard effect-of-o families (donor focus and sample focus):
  - `effect_of_o_coverage_donor_focus_lambda*.pdf`
  - `effect_of_o_width_donor_focus_lambda*.pdf`
  - `effect_of_o_coverage_sample_focus_lambda*.pdf`
  - `effect_of_o_width_sample_focus_lambda*.pdf`

#### Methods in result CSVs

`results_effect_of_o_joint_xy_marginal.csv` now includes o-dependent Std-CP columns:
- `coverage_stdcp`, `width_stdcp`, `infinite_stdcp`

HCP/Pooling/Subsampling/Repeated remain fixed across `o`; D-HCP and Std-CP vary with `o`.

### 2. Real Data Experiments

See [real_data/README.md](real_data/README.md) for detailed instructions on running experiments with:
- **ACS (American Community Survey)**: Income prediction for recent immigrants across US states
- **Blood Pressure Trial**: Systolic blood pressure prediction across clinic sites

Quick start:

```bash
# ACS experiments
cd real_data/acs
python3 load_acs_data.py --year 2018 --output data/acs_data_all50states.csv --all_states
python3 run_acs.py data/acs_data_all50states.csv --top_income_pct 25 --alpha 0.1

# Blood Pressure experiments
cd real_data/blood_pressure
python3 load_bp_data.py data/ --output data/bp_data.csv
python3 run_bp.py data/bp_data.csv --n_test_clinics 15 --alpha 0.2

# Generate plots (optional - automatically done by run scripts)
cd real_data
python3 plot_results.py --all
```

### ACS PUMA Repeated Experiment (Latest Locked Configuration)

The latest ACS rerun uses the single-state PUMA repeated-experiment pipeline in `real_data/repeated_experiments.py` with the top 2% income tail removed inside CA.

Command:

```bash
cd /path/to/hierarchical_cp
.venv/bin/python -u real_data/repeated_experiments.py \
  --dataset acs_puma \
  --B_acs 100 \
  --acs_state CA \
  --acs_n_groups 30 \
  --acs_min_group_size 20 \
  --acs_min_yoep 2012 \
  --acs_min_hours 40 \
  --acs_min_income 10000 \
  --acs_bottom_income_quantile 0.98 \
  --acs_no_age_filter \
  --acs_expected_eligible_pumas 60 \
  --acs_expected_test_pumas 30 \
  --acs_o_values 0,5,10,20
```

Run summary for this configuration:
- State: CA
- Foreign-born only
- Age filter: disabled (`--acs_no_age_filter`)
- Entry-year filter: `YOEP >= 2012`
- Labor filter: `hours >= 40`
- Income floor: `income >= 10,000`
- High-income trimming: keep bottom 98% in-state (`--acs_bottom_income_quantile 0.98`)
- Eligible PUMAs (size >= 20): 60
- Split each replicate: 30 non-test PUMAs + 30 test PUMAs
- History sizes evaluated: `o ∈ {0, 5, 10, 20}`

### NEW_RESULTS: Stratified ACS filtering details (May 2026)

Current stratified ACS experiments are run through `real_data/repeated_experiments_stratified_acs.py`.

Filtering used in the latest stratified run:
- State: CA only
- Foreign-born only (`nativity == 2`)
- Age filter: `25 <= age <= 54`
- Entry-year filter: `YOEP >= 2012`
- Hours filter: `hours >= 40`
- Income filters: `income > 0` and `income >= 10,000`
- No top-tail trimming in the latest stratified run (`--acs_bottom_income_quantile` not set)
- Keep PUMAs with size >= 20 after filtering

Observed outcome of this filter profile:
- Rows after full cleaning: 3,268
- Eligible PUMAs: 45
- Split: 20 non-test PUMAs + 25 test PUMAs

Why this differs from earlier 61-eligible write-ups:
- Earlier locked run in this repository used `--acs_no_age_filter --acs_bottom_income_quantile 0.98` and yielded 60 eligible PUMAs.
- Current stratified run uses a different setup (age filter enabled, different non-test count, stratified non-test sampling by BA+ share), so eligibility counts are not directly comparable.

Bootstrap and target definition:
- In each replicate, each selected PUMA is bootstrapped within-group with replacement at its original size.
- For each `o`, the target is the `(o+1)`-th bootstrapped individual (zero-based index `o`); the first `o` observations are the history for history-aware methods.
- Coverage is averaged over test PUMAs per replicate.
- HCP and other classic baselines are history-independent in construction; D-HCP and S-HCP vary with `o`.

Latest output artifacts:
- `real_data/acs/results/acs_new_detailed_bottom98.csv`
- `real_data/acs/results/acs_puma_filtered_summary_bottom98.csv`
- `real_data/acs/plots/acs_effect_of_o_coverage_main_compare_bottom98.(pdf|png)`
- `real_data/acs/plots/acs_effect_of_o_width_main_compare_bottom98.(pdf|png)`

## Experiment Design

### Varying History Experiments

Test how prediction intervals change as the amount of test-group history increases.

- **Key idea**: The target observation is always the **last** observation in each test group
  (same X, same Y across all conditions). Only the amount of history varies.
- **Setup**:
  1. For each test group, fix the target as the last observation
  2. For history fractions h ∈ {0%, 25%, 50%, 75%}:
     - Give the first `floor(h * (n-1))` observations as history
     - Predict the last observation
     - Record coverage indicator and interval width
  3. Aggregate coverage across all test groups
- **Expected behavior**:
  - Baseline methods (HCP, Pooling, Subsampling, Repeated) do **not** use test-group
    history, so their intervals are identical across all history fractions
  - HCP++ and HCP.sample **do** use history, so they should produce tighter/better
    intervals as history increases
- **Coverage**: Reported per history fraction and overall

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

Saved to `DGP/results/` and `DGP/plots/`:
- `results/files/`: Raw CSV results for each configuration
- `plots/`: Coverage and width comparison plots
- `results/lambda_*/summary_*.csv`: Aggregated statistics

### Real Data Results

**ACS Results** (`real_data/acs/results/`):
- `acs_detailed.csv`: Per-history-fraction predictions (576 predictions)
- `acs_summary.csv`: Method summaries with per-history-fraction coverage and width
- `acs_summary.md`: Formatted results table

**BP Results** (`real_data/blood_pressure/results/`):
- `bp_detailed.csv`: Per-history-fraction predictions (360 predictions)
- `bp_summary.csv`: Method summaries with per-history-fraction coverage and width
- `bp_summary.md`: Formatted results table

**Plots** (`real_data/*/plots/`):
- Coverage by method and history fraction
- Interval width by method and history fraction

## Key Findings

### ACS Experiments
- **Setup**: 25 training states, 24 test states (emerging destinations)
  - 1,180 observations across 49 states
  - Per-state top 25% income filtering
  - History fractions: 0%, 25%, 50%, 75%
  - All test states included (even with 1 observation)
- **Target**: 90% coverage (α=0.1)
- **Results** (576 predictions):
  - HCP: 96% overall coverage, ~2.25 width (identical across history fractions)
  - Pooling: 92% overall coverage, ~1.99 width (identical across history fractions)
  - HCP++: 95% overall coverage, ~2.48 width, 5.2% infinite intervals (varies with history)
  - HCP.sample: 90% overall coverage, ~1.98 width, 0% infinite (varies with history)

### Blood Pressure Experiments
- **Setup**: 17 training clinics, 15 test clinics
  - 605 observations across 32 clinics
  - History fractions: 0%, 25%, 50%, 75%
  - All test clinics included
- **Target**: 80% coverage (α=0.2)
- **Results** (360 predictions):
  - HCP: 93% overall coverage, ~56.9 mmHg width (identical across history fractions)
  - Pooling: 67% overall coverage, ~45.5 mmHg width (identical across history fractions)
  - HCP++: 87% overall coverage, ~50.1 mmHg width, 0% infinite (varies with history)
  - HCP.sample: 82% overall coverage, ~48.3 mmHg width, 0% infinite (varies with history)

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
