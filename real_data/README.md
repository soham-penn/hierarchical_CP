# Real Data Experiments for Hierarchical Conformal Prediction

This directory contains bootstrap experiments for hierarchical conformal prediction methods on three datasets:

1. Blood Pressure Trial: Systolic blood pressure prediction across clinic groups
2. ACS State-Level: Income prediction across US states
3. ACS PUMS: Income prediction for recent immigrants across PUMAs within a single state

## Directory Structure

```
real_data/
├── blood_pressure/
│   ├── data/                        # BP data files
│   ├── plots/                       # Bootstrap experiment plots
│   ├── results/                     # Bootstrap experiment results (CSV)
│   └── data_processing.py           # Data loading and cleaning
│
├── acs/
│   ├── data/                        # ACS state-level data
│   ├── plots/                       # Bootstrap experiment plots
│   ├── results/                     # Bootstrap experiment results (CSV)
│   └── data_processing.py           # Data loading and cleaning
│
├── acs_pums/
│   ├── data/                        # ACS PUMS data (auto-downloaded)
│   ├── plots/                       # Bootstrap experiment plots
│   ├── results/                     # Bootstrap experiment results (CSV)
│   ├── load_pums_data.py            # PUMS data loading and filtering
│   ├── bootstrap_pums.py            # Bootstrap experiments (proportion + fixed-o)
│   └── README.md                    # PUMS experiment documentation
│
├── repeated_experiments.py          # Repeated experiments for BP and ACS state-level
└── README.md                        # This file
```

## Methods Evaluated

All experiments compare 8 hierarchical conformal prediction methods:

Proposed Methods:
1. donor-HCP-randomized (D-HCP)
2. donor-HCP-derandomized (DD-HCP)
3. sample-HCP-randomized (S-HCP)
4. sample-HCP-derandomized (D-Sample-HCP)

Baseline Methods:
5. HCP: Standard hierarchical CP (weighted empirical quantile + infinity atom)
6. Pooling: Pool all calibration data (ignores group structure, no infinity)
7. Subsampling: Sample once from each group + infinity
8. Repeated: Repeated subsampling with averaging (50 repetitions)

## Experiment 1: Blood Pressure

### Overview

Goal: Predict systolic blood pressure at 12 months across clinic sites
Data Source: Hypertension intervention trial (treatment arm)
Groups: 32 clinics (17 non-test, 15 test)
Outcome: Follow-up SBP at 12 months (mmHg)
Covariates: Baseline SBP

### Bootstrap Design

For each of B=100 replicates:
1. Sample with replacement from each of the 32 clinics (size = original clinic size)
2. Randomly split 17 non-test clinics into training (8-9) and calibration (8-9)
3. Fit model on training clinics, compute scores on calibration clinics
4. Evaluate on all 15 test clinics at different history sizes o

History sizes (o values): {0, 4, 8, 12, 17} where 17 = min test clinic size - 1

### Usage

```bash
cd /path/to/hierarchical_cp/real_data

# Run BP repeated experiment (B=100 replicates)
python3 repeated_experiments.py --dataset bp --B_bp 100
```

### Output

Plots:
- blood_pressure/plots/bp.png - Main plot (excluding Repeated method)
- blood_pressure/plots/bp_all.png - All methods

Results:
- blood_pressure/results/bp_detailed.csv - Detailed results (all replicates)

## Experiment 2: ACS State-Level

### Overview

Goal: Predict income for recent immigrants across US states
Data Source: ACS PUMS 2018
Groups: 35 states (18 non-test, 17 test) after removing top-15 largest states
Outcome: log(income + 1)
Population: Foreign-born immigrants (YOEP >= 2017), top 25% income per state

### Bootstrap Design

For each of B=100 replicates:
1. Sample with replacement from each of the 35 states (size = original state size)
2. Randomly split 18 non-test states into training (9) and calibration (9)
3. Fit model on training states, compute scores on calibration states
4. Evaluate on all 17 test states at different history sizes o

History sizes (o values): {0, 2, 5, 7, 10} where 10 = min test state size - 1

### Usage

```bash
# Run ACS repeated experiment (B=100 replicates)
python3 repeated_experiments.py --dataset acs_top25 --B_acs 100

# Run both BP and ACS
python3 repeated_experiments.py --dataset all --B_bp 100 --B_acs 100
```

### Output

Plots:
- acs/plots/acs.png - Main plot (excluding Repeated method)
- acs/plots/acs_all.png - All methods

Results:
- acs/results/acs_detailed.csv - Detailed results (all replicates)

## Experiment 3: ACS PUMS (Within-State Hierarchical)

### Overview

Goal: Predict income for recent immigrants across PUMAs within a single state
Data Source: ACS PUMS 2018 via folktables
Groups: PUMAs within one state (configurable; default state is CA)
Outcome: Income (PINCP)
Population (repeated_experiments.py path):
- Foreign-born only
- Age in [25, 54]
- Recent entry window: last 4 years in YOEP
- Hours worked >= 0
- Income > 0
- Drop missing key fields used by covariates/outcome

This experiment addresses the concern that states may not be exchangeable. Within a single state, PUMAs are more likely to be exchangeable units.

### Stratified ACS filtering details (May 2026)

Current stratified run uses `real_data/repeated_experiments_stratified_acs.py` with:

```bash
.venv/bin/python -u real_data/repeated_experiments_stratified_acs.py \
   --B_acs 100 \
   --n_workers 6
```

Exact filtering used in this run:
- State filter: CA only
- Nativity filter: foreign-born only (`nativity == 2`)
- Age filter: `25 <= age <= 54` (enabled by default)
- Entry-year filter: `YOEP >= 2012` (`--acs_min_yoep 2012`)
- Labor filter: `hours >= 40` (`--acs_min_hours 40`)
- Income validity: `income > 0`
- Income floor: `income >= 10,000` (`--acs_min_income 10000`)
- Income upper-tail trimming: none in this run (`--acs_bottom_income_quantile` not set)
- Missing-data filter: drop rows missing key modeling fields (`y`, `age`, `age_sq`, `hours`, `entry_recency`, `educ_level`, `married`, `female`, `english`, `cow`)
- Group eligibility: keep PUMAs with at least 20 observations after all filters

Observed counts for this stratified run:
- Rows after full cleaning: 3,268
- Eligible PUMAs (size >= 20): 45
- Split: 20 non-test PUMAs (stratified sample) + 25 test PUMAs
- Stratification variable: PUMA-level BA+ share (`educ_level == BAplus`)
- Stratification scheme: 5 quantile bins (`pd.qcut`) over eligible PUMAs, balanced sampling across bins

Changes relative to your original setup text (61-eligible description):
- Age filter changed: original text uses age 25-54 and stratified run also uses age 25-54, so no change here.
- Bottom-98% trimming changed: original text keeps bottom 98%; current stratified run does not trim top incomes by default.
- Historical-group count changed: original text uses `K=30`; current stratified run uses `K=20` non-test PUMAs.
- Group assignment changed: original uses simple random non-test PUMA sampling; current run uses stratified sampling by BA+ share.
- Eligible-PUMA count changed from your earlier write-up (61) to 45 in the current run because this run combines age 25-54, stricter recent-entry and labor/income filters, and no forced split-count check.

Reproducibility checks run in this codebase:
- With stratified defaults (age 25-54, no bottom-98% trim): 45 eligible PUMAs.
- If bottom-98% trim is added (`--acs_bottom_income_quantile 0.98`) while keeping age 25-54: 44 eligible PUMAs.
- If using older locked settings (`--acs_no_age_filter --acs_bottom_income_quantile 0.98`): 60 eligible PUMAs.

So the main practical reason for 45 vs older 60/61-style counts is that the current stratified run is not the same filter profile as the older locked run (especially age filtering enabled and different split design), and your prior notes likely came from an earlier pipeline snapshot.

### New eligibility metadata CSV (requested table)

We now save a table with one row per eligible CA PUMA that includes:
- State/run metadata (state, raw/filtered/eligible counts, sampling seed, filters used)
- PUMA role (sampled non-test vs test)
- Sample size per PUMA
- Outcome summaries (`outcome_y_mean`, `income_mean`)
- Covariate summaries (means of the covariates used in the design matrix, including dummy-coded features)

File:
- acs/results/acs_puma_eligibility_metadata.csv

Generate/update it with:

```bash
cd /path/to/hierarchical_cp
.venv/bin/python \
   real_data/acs/generate_puma_metadata_table.py \
   --state CA \
   --min_group_size 10 \
   --n_groups 30 \
   --group_seed 42 \
   --yoep_window_years 4
```

### Two Experiment Modes

1. Proportion-based: o = {0%, 25%, 50%, 75%, 100%} of each test PUMA's size
2. Fixed-o: o = {0, floor(m/4), floor(m/2), floor(3m/4), m-1} where m = min test PUMA size

### Usage

```bash
cd /path/to/hierarchical_cp/real_data

# Stratified ACS PUMA experiment
../.venv/bin/python -u repeated_experiments_stratified_acs.py \
   --B_acs 100 \
   --n_workers 6
```

### Output

Results:
- acs/results/stratified/acs_stratified_summary.csv
- acs/results/stratified/acs_stratified_summary_long.csv
- acs/results/stratified/acs_stratified_data_overview.csv

## Parameters

All experiments use:
- alpha = 0.2 (target coverage = 80%)
- alpha_selection = 0.5 (CDF threshold for donor selection in HCP++/HCP.sample)
- n_repeated = 50 (repetitions for Repeated Subsampling)

## Implementation Details

### Bootstrap Procedure

For each replicate:
1. Bootstrap all groups (non-test + test) with replacement
2. Random split of non-test groups into training and calibration
3. Baselines (HCP, Pooling, Subsampling, Repeated):
   - Fit model on training groups
   - Compute scores on calibration groups
   - Same interval for all history sizes o (no history dependence)
4. HCP methods (HCP++, HCP.sample):
   - Fit model on complement of S_tilde (donor selection based on CDF)
   - Compute scores on S_tilde + test group history
   - Different interval for each history size o

### Key Differences from Previous Experiments

Old experiments used fixed test/calibration splits and sequential/marginal evaluation.

New experiments use bootstrap resampling to:
- Better quantify uncertainty (100 replicates)
- Evaluate coverage/width variability across different data splits
- Compare methods at multiple history sizes simultaneously

## Output Files

Each experiment produces:
- Plots: Boxplots showing coverage and interval width distributions across bootstrap replicates
- CSV: Detailed results with one row per (replicate, method, history_size) combination

CSV columns:
- replicate: Bootstrap replicate number (1 to B)
- method: Method name
- o or proportion: History size (absolute or proportion)
- coverage: Coverage rate (0 to 1)
- width: Mean interval width

## Notes

- All methods use global OLS regression (tau=0, no shrinkage in mu estimation)
- HCP++ and HCP.sample use shrinkage for scoring but global prediction for baselines
- Test groups are never used for training in baseline methods
- Bootstrap resampling is done with replacement within each group
- Group assignments (test vs non-test) are fixed across all bootstrap replicates

### Replicate and coverage details for acs_puma (repeated_experiments.py)

- Each bootstrap replicate resamples each selected PUMA independently with replacement, preserving that PUMA's original size N_k.
- One target is evaluated per test PUMA: the last bootstrapped observation of that PUMA.
- Coverage per replicate is the mean of covered/not-covered indicators across test PUMAs (equal weight per test PUMA).
- Baselines (HCP, Pooling, Subsampling, Repeated) do not use test history, so their replicate-level values are copied across all o values.
- donor/sample methods recompute intervals for each o using first o points as history and the last point as the target.

### HCP "infinite interval" sanity check

For the rerun above (CA, 30 sampled non-test PUMAs, min size 5):
- Detailed results file: acs/results/acs_new_detailed.csv
- NaN width rate (proxy for infinite intervals) is 0.0 for all methods, including HCP.
- HCP mean coverage is about 0.828, not near 1.

If you set --acs_min_group_size too low (for example 1), many test PUMAs have size 1 and m becomes 1, so only o=0 is evaluated; this can make behavior look unstable or overly conservative because test groups are tiny.
