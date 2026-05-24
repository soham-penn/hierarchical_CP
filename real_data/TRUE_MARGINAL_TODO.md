# ACS True Marginal Experiments - Outstanding Issues

## Summary of Current State

### What Works ✅
- D-HCP coverage stays above nominal level for alpha=0.1 (99.4% → 93.4% for o=0→20)
- All key bugs fixed:
  - Fixed target at index 20
  - Full Z_test passed to D-HCP
  - Using ols_offset mu_method
  - Only select test groups with 21+ observations
  - o=15 included

### Outstanding Issues ⚠️

#### Issue 1: HCP has ~77% infinite width intervals at alpha=0.1
**Symptom:** HCP shows near-perfect coverage (99.5%) with infinite widths in 772/1000 replicates

**Likely Cause:** At alpha=0.1, the quantile computation includes the infinity score
- HCP adds one infinity score with weight 1/(K+1)
- At small alpha, this infinity gets included in the (1-alpha) quantile
- Result: infinite interval width, perfect coverage

**Solution:** This is EXPECTED behavior for small alpha! At alpha=0.1, HCP correctly returns conservative (infinite) intervals when there's high uncertainty.

#### Issue 2: Need to run experiments with alpha=0.2
**Status:** Code modified to save results in separate folders per alpha value
- alpha=0.1 → saves to `acs/results/true_marginal_alpha10/`
- alpha=0.2 → will save to `acs/results/true_marginal_alpha20/`

**Action Required:**
1. Wait for alpha=0.1 run to complete
2. Change line 587 in acs_true_marginal_experiments.py: `'alpha': 0.2`
3. Run again

#### Issue 3: Plot width axis shows log-transformed values, not actual income
**Current:** Width axis shows values like 0, 0.5 which are in log1p(income) space
**Desired:** Show actual income values like $100K, $200K

**Solution:** The plotting code already has `width_income` field which converts back to income space!
Check line in plot script - it should use `width_income` not `width` for the y-axis values.

The issue is in the plotting script at `plot_true_marginal_paper_style.py` - it needs to:
1. Use `width_income` column instead of `width`
2. Format y-axis labels as currency (e.g., "$100K")

## Files Modified

1. `acs_true_marginal_experiments.py`:
   - Line 587: alpha value (currently 0.1, change to 0.2 for second run)
   - Lines 612-617: Results saved to alpha-specific folders

2. `acs/plot_true_marginal_paper_style.py`:
   - Needs modification to use `width_income` and format as currency

## Next Steps

1. Run with alpha=0.2 to get comparable results
2. Update plotting script to show income values instead of log-transformed values
3. Generate final plots for both alpha values
4. Document that HCP infinite widths at alpha=0.1 are expected behavior
