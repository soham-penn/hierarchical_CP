# Pooled-Income Quantile Baseline for ACS True Marginal Experiments

## Overview

This directory contains scripts to add a naive unconditional pooled-income quantile baseline to the ACS true marginal experiments.

The **Pooled-Income-Quantile** method:
- Ignores all covariates, group structure, and test-group history length `o`
- Pools raw income values from calibration PUMAs only (excludes test PUMA)
- Computes empirical quantile-based prediction intervals
- For nominal coverage 1-α, uses quantiles at α/2 and 1-α/2
- Example: α=0.1 → use 5th and 95th percentiles

## Files

### Main Scripts

1. **`acs_true_marginal_add_pooled_quantile.py`**
   - Standalone script to generate pooled-quantile baseline results
   - Replicates the same trial structure as main ACS true marginal experiments
   - Outputs results compatible with existing format

2. **`acs/plot_true_marginal_with_pooled_quantile.py`**
   - Plotting script that combines main results with pooled-quantile baseline
   - Generates coverage and width comparison plots

## Usage

### Step 1: Generate Pooled-Quantile Results

Run for α=0.1 (90% nominal coverage):
```bash
cd /Users/soham/UPenn/Claude/hier_current/real_data
python3 acs_true_marginal_add_pooled_quantile.py \
    --B=1000 \
    --alpha=0.1 \
    --n_workers=6 \
    --n_puma_groups=20
```

Run for α=0.2 (80% nominal coverage):
```bash
python3 acs_true_marginal_add_pooled_quantile.py \
    --B=1000 \
    --alpha=0.2 \
    --n_workers=6 \
    --n_puma_groups=20
```

### Step 2: Generate Plots

For α=0.1:
```bash
cd /Users/soham/UPenn/Claude/hier_current/real_data
python3 acs/plot_true_marginal_with_pooled_quantile.py --alpha=0.1
```

For α=0.2:
```bash
python3 acs/plot_true_marginal_with_pooled_quantile.py --alpha=0.2
```

## Output Files

### Results (saved in `acs/results/true_marginal_alphaXX/`)

For α=0.1 (XX=10):
- `acs_true_marg_pooled_quantile_alpha10_detailed.csv` - Raw results (duplicated across o for plotting)
- `acs_true_marg_pooled_quantile_alpha10_o_independent_detailed.csv` - Raw results (o=-1, true o-independent)
- `acs_true_marg_pooled_quantile_alpha10_summary.csv` - Summary statistics

For α=0.2 (XX=20):
- Same pattern with `alpha20` suffix

### Plots (saved in `acs/NEW_PLOTS/true_marginal_with_pooled_quantile_alphaXX/`)

1. **Coverage vs Nominal Coverage**
   - `acs_true_marg_coverage_vs_nominal_alphaXX_with_pooled.pdf`
   - Shows empirical coverage vs nominal coverage
   - Includes error bars (binomial SE)
   - Compares D-HCP (varying o), HCP, and Pooled-Income-Quantile

2. **Width vs o**
   - `acs_true_marg_width_vs_o_alphaXX_with_pooled.pdf`
   - Shows median interval width vs test-group history length o
   - D-HCP: varies with o
   - HCP and Pooled-Income-Quantile: horizontal lines (o-independent)

## Column Descriptions

### Raw Results CSV Columns

- `trial_id`: Trial index (0 to B-1)
- `method`: Always "Pooled-Income-Quantile"
- `alpha`: Miscoverage level (e.g., 0.1 or 0.2)
- `nominal_coverage`: 1 - alpha
- `o`: Test-group history length (-1 for o-independent version, duplicated for plotting version)
- `coverage`: Binary coverage indicator (1 if target covered, 0 otherwise)
- `width`: Interval width in log1p(income) space
- `width_income`: Interval width in actual income dollars
- `lower`: Lower bound (log1p space)
- `upper`: Upper bound (log1p space)
- `y_target`: Target income (log1p space)
- `income_target`: Target income (actual dollars)
- `n_pooled`: Number of pooled calibration observations

### Summary CSV Columns

- `method`: "Pooled-Income-Quantile"
- `alpha`: Miscoverage level
- `nominal_coverage`: 1 - alpha
- `n_trials`: Number of successful trials
- `coverage_mean`: Empirical marginal coverage
- `coverage_se`: Binomial standard error = sqrt(p*(1-p)/n)
- `width_income_mean`: Mean interval width (dollars)
- `width_income_std`: Std dev of interval width (dollars)
- `width_income_median`: Median interval width (dollars)
- `width_log_mean`: Mean interval width (log space)
- `width_log_std`: Std dev of interval width (log space)
- `width_log_median`: Median interval width (log space)
- `n_pooled_mean`: Mean pooled sample size
- `n_pooled_std`: Std dev of pooled sample size

## Notes

1. **Same Trial Structure**: The pooled-quantile script uses the same seed (456) and sampling procedure as the main ACS true marginal experiments, ensuring trials align properly.

2. **O-Independence**: The Pooled-Income-Quantile method does NOT use test-group observations, so intervals are identical for all o values. The duplicated version repeats the same interval across o=[0,5,10,15,20] for easy comparison in plots.

3. **Coverage SE Formula**: Uses true marginal Bernoulli formula `sqrt(p*(1-p)/n)`, NOT the calibration-conditional formula.

4. **No Data Leakage**: The pooled-quantile method strictly uses only calibration PUMA incomes. The target PUMA and target observation are excluded from the pooled sample.

5. **Compatibility**: Output format matches the main experiment structure for easy integration into existing analysis pipelines.

## Example Output

```
Summary statistics:
  Trials: 1000
  Nominal coverage: 90.0%
  Empirical coverage: 0.8520 ± 0.0112
  Width (income): median=425000, mean=438000
  Width (log): median=4.123, mean=4.156
  Pooled sample size: mean=520.3
```

## Integration with Existing Results

To combine pooled-quantile results with existing main results:

```python
import pandas as pd

# Load main results
df_main = pd.read_csv('acs/results/true_marginal_alpha10/acs_true_marg_alpha10_detailed.csv')

# Load pooled-quantile results (duplicated version)
df_pooled = pd.read_csv('acs/results/true_marginal_alpha10/acs_true_marg_pooled_quantile_alpha10_detailed.csv')

# Combine
df_combined = pd.concat([df_main, df_pooled], ignore_index=True)

# Now you can plot or analyze all methods together
```
