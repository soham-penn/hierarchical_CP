# Latent Intercept Experiment Documentation

## Script Location
`run_latent_intercept_experiment.py`

## Overview
Complete experiment runner for latent intercept DGP with `tau_B ∈ {0, 6}` and `alpha ∈ {0.20, 0.05}`.

The experiment runs **twice** - once for each value of `tau_B`.

## Experimental Design

### Data Generating Process
- **Historical groups**: K = 20 groups, each with N = 21 observations
- **Target groups**: Size N = 36 observations each
- **Fixed target**: Always the 36th observation (index 35 in Python)
- **Dimension**: d = 5
- **Correlation parameter**: ρ = 0.5
- **Group covariates**: U_j ~ Uniform([0,5]^5)
- **Latent intercept**: B_j ~ N(0, tau_B^2)

#### DGP Formula
For each group j:
```
U_j ~ Uniform([0,5]^5)
B_j ~ N(0, tau_B^2)

Z_{j,i} ~ N(mu(U_j) + B_j·e_d, Sigma(U_j))

where:
  mu(U) = (U_1^2, ..., U_5^2)
  Sigma(U) = (1-ρ)·diag(U) + ρ·1·1^T
  e_d = (0,0,0,0,1) (only response Y gets shifted by B_j)
```

### Simulation Parameters
- **Outer experiments**: 50
- **Target groups per experiment**: 100
- **o values**: {0, 5, 10, 15, 20, 25, 30, 35}
- **alpha values**: {0.20, 0.05}
- **tau_B values**: {0, 6}
- **Parallelization**: 4 CPU cores

### Methods Evaluated

All methods are run on the **SAME dataset** before moving to the next dataset (fully paired comparison).

#### 1. **Donor-HCP with within-group training**
- **Source**: `methods/donor_hcp.py::compute_donor_hcp_randomized_interval()`
- **Mu method**: `create_mu_method_ols_offset()` (uses `m_j = floor(o/2)`)
- **Runs for**: All o ∈ {0,5,10,15,20,25,30,35}
- **Special case**: When o > 21, no eligible donors → falls back to S = {K+1}

#### 2. **Donor-HCP without within-group training**
- **Source**: Same as above
- **Mu method**: `create_mu_method_ols_global_only()` (pure global, m_j = 0)
- **Runs for**: All o values

#### 3. **S-HCP with within-group training**
- **Source**: `methods/sample_hcp.py::compute_sample_hcp_randomized_interval()`
- **Mu method**: `create_mu_method_ols_offset()`
- **Runs for**: All o values

#### 4. **Standard HCP**
- **Source**: `methods/baseline_hcp.py::compute_hcp_interval_radius()`
- **Runs**: Once per experiment (o-independent)
- **Split**: Uses donor-style split with o=0 to match D-HCP at o=0
- **Stored at**: o=0 in results

#### 5. **Pooling**
- **Source**: `methods/baseline_hcp.py::compute_pooling_interval_radius()`
- **Runs**: Once per experiment (o-independent)
- **Stored at**: o=0 in results

#### 6. **Subsampling**
- **Source**: `methods/baseline_hcp.py::compute_subsampling_once_interval_radius()`
- **Runs**: Once per experiment (o-independent)
- **Stored at**: o=0 in results

#### 7. **Repeated Subsampling**
- **Source**: `methods/baseline_hcp.py::compute_repeated_subsampling_interval_radius()`
- **Repetitions**: 50
- **Runs**: Once per experiment (o-independent)
- **Stored at**: o=0 in results

#### 8. **Standard CP**
- **Implementation**: Custom (lines 396-438)
- **Method**: Split first o observations 50/50 into train/cal, fit OLS, compute conformal quantile
- **Runs for**: All o ∈ {0,5,10,15,20,25,30,35}
- **Special case**: Returns infinite interval for o < 2

## Parallelization Strategy

**Parallelization is over OUTER EXPERIMENT REPLICATES**, not over methods.

Each worker:
1. Takes one (exp_id, tau_B, alpha) tuple
2. Generates the historical groups
3. Generates 100 target groups
4. For each target group, runs **all methods** on the same data
5. Returns complete results for that experiment

This ensures:
- ✓ All methods see the same realized data
- ✓ Fully paired comparison
- ✓ No data mismatch across methods
- ✓ Reproducible with fixed seeds

## Outputs

All outputs are saved to:
- `NEW_RESULTS/` - CSV files
- `NEW_PLOTS/` - PDF plots

### Result Files

#### 1. `raw_results_complete.csv`
Complete raw results with columns:
- `tau_B`: {0, 6}
- `alpha`: {0.20, 0.05}
- `experiment`: 0 to 49
- `target_group`: 0 to 99
- `o`: {0, 5, 10, 15, 20, 25, 30, 35}
- `method`: method name
- `coverage`: {0, 1}
- `width`: interval width (NaN if infinite)
- `lower`: lower endpoint
- `upper`: upper endpoint
- `infinite`: {0, 1}

#### 2. `summary_by_tau_alpha_o_method.csv`
Aggregated by (tau_B, alpha, o, method):
- `coverage_mean`, `coverage_std`
- `width_mean`, `width_median`, `width_std`
- `n_infinite`, `prop_infinite`

#### 3. `experiment_level_summary.csv`
Aggregated by (tau_B, alpha, experiment, o, method):
- `coverage`: mean per experiment
- `width`: median per experiment
- `n_infinite`: count per experiment

Used for creating boxplots.

#### 4. `randomization_stability.csv`
For Donor-HCP with within-training, measures effect of randomization:
- `tau_B`, `alpha`, `o`, `dataset_id`
- `mean_width`: mean width over 100 reruns
- `sd_upper`: standard deviation of upper endpoint
- `sd_width`: standard deviation of width
- `relative_instability`: sd(upper) / mean(width)
- `n_finite`: number of finite intervals

#### 5. `randomization_stability_summary.csv`
Aggregated stability metrics by (tau_B, alpha, o).

### Plots (for alpha = 0.20)

All plots are saved as PDF files in `NEW_PLOTS/`.

For each `tau_B ∈ {0, 6}`:

1. **`tauB{}_alpha20_dhcp_vs_hcp_o_upto20.pdf`**
   - D-HCP (with within) vs HCP
   - Coverage and Width subplots
   - o up to 20

2. **`tauB{}_alpha20_dhcp_vs_hcp_all_o.pdf`**
   - D-HCP (with within) vs HCP
   - Coverage and Width subplots
   - All o up to 35

3. **`tauB{}_alpha20_dhcp_vs_all_baselines.pdf`**
   - D-HCP (with within) vs HCP, Pooling, Subsampling, Repeated
   - Coverage and Width subplots
   - All o values

4. **`tauB{}_alpha20_dhcp_within_vs_no_within.pdf`**
   - D-HCP with vs without within-group training
   - Side-by-side comparison with HCP baseline
   - Uses hatching to distinguish no-within variant

## Running the Script

```bash
# Make executable
chmod +x run_latent_intercept_experiment.py

# Run
python3 run_latent_intercept_experiment.py
```

Expected runtime: ~30-60 minutes (depending on hardware)

## Reproducibility

- **Seeds**: Base seed = 12345, independent seed per experiment = BASE_SEED + exp_id
- **Donor-HCP split**: Uses fixed seed 123 (from method implementation)
- **Data generation**: Fully reproducible with same base seed

## Verification Checklist

✓ Historical groups have size 21
✓ Target groups have size 36
✓ Fixed target is observation 36 (index 35)
✓ All methods run on same data before data changes
✓ Parallelization is over outer experiments only
✓ Experiment runs twice (tau_B = 0 and tau_B = 6)
✓ Results include tau_B as a column/tag
✓ Baseline methods use donor-style split with o=0
✓ At o=0, HCP should match D-HCP (same split)
✓ When o > 21, D-HCP has no donors (expected behavior)
✓ Standard CP handles o < 2 by returning infinite interval

## Method Mapping from Repository

| Method in Spec | Repository Location | Notes |
|----------------|---------------------|-------|
| Donor-HCP (with within) | `methods/donor_hcp.py::compute_donor_hcp_randomized_interval` | Uses `create_mu_method_ols_offset()` |
| Donor-HCP (no within) | Same function | Uses `create_mu_method_ols_global_only()` |
| S-HCP (with within) | `methods/sample_hcp.py::compute_sample_hcp_randomized_interval` | Uses `create_mu_method_ols_offset()` |
| HCP | `methods/baseline_hcp.py::compute_hcp_interval_radius` | Weighted quantile with infinity |
| Pooling | `methods/baseline_hcp.py::compute_pooling_interval_radius` | Weighted quantile without infinity |
| Subsampling | `methods/baseline_hcp.py::compute_subsampling_once_interval_radius` | One score per group |
| Repeated | `methods/baseline_hcp.py::compute_repeated_subsampling_interval_radius` | Mean of R subsamples |
| Standard CP | Custom implementation in script | Split-conformal on test group only |
| OLS global learner | Custom in script (lines 294-321) | Y ~ 1 + U + X |
| Train/cal split | `methods/donor_hcp.py::get_donor_style_train_cal_split` | Donor-style with o=0 |

## Additional Notes

### Target Index Handling
- Target is **always** the 36th observation (index 35)
- For each o, only the first o observations are "revealed" to methods
- The same target point is used across all methods
- This creates a fair comparison where all methods predict the same Y

### Large o Values (o > 21)
When o > 21, there are no historical groups with N > o.
- Donor-HCP automatically falls back to S = {K+1} (test group only)
- This is correct behavior and is not a bug
- The method adapts gracefully to this case

### Baseline Method Storage
- HCP, Pooling, Subsampling, Repeated are o-independent
- Stored at o=0 to avoid duplication
- Summary/plotting code replicates them across all o values for comparison

### Randomization Stability
- Measures how much donor randomization affects results
- Fixes one dataset, reruns D-HCP 100 times
- Relative instability = sd(upper endpoint) / mean(width)
- Low values indicate stable randomization
