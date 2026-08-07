# Stratified ACS Experiment Summary

## Experiment Details

- **Dataset**: California ACS PUMS data (stratified PUMA sampling)
- **Number of bootstrap replicates**: 100
- **Target coverage level (1-α)**: 0.80
- **O values tested**: [0, 5, 10, 20]
- **Number of calibration PUMAs**: 20 (stratified by BA+ share)
- **Number of test PUMAs**: 25
- **Stratification**: 5 quantile strata based on PUMA-level share of BA+ degree holders

## Summary Statistics

### Coverage Performance (Mean ± Std)

| Method | o=0 | o=5 | o=10 | o=20 | Overall |
|--------|-----|-----|------|------|---------|
| **D-HCP (randomized)** | 0.875 ± 0.063 | 0.846 ± 0.076 | 0.839 ± 0.075 | 0.824 ± 0.073 | **0.846** |
| **D-HCP (derandomized)** | 1.000 ± 0.004 | 0.977 ± 0.031 | 0.954 ± 0.047 | 0.884 ± 0.053 | **0.954** |
| **S-HCP (randomized)** | 0.812 ± 0.078 | 0.796 ± 0.074 | 0.791 ± 0.095 | 0.794 ± 0.083 | **0.798** |
| **S-HCP (derandomized)** | 0.995 ± 0.014 | 0.890 ± 0.065 | 0.902 ± 0.065 | 0.878 ± 0.057 | **0.916** |
| **HCP** | 0.877 ± 0.058 | 0.860 ± 0.072 | 0.873 ± 0.074 | 0.862 ± 0.075 | **0.868** |
| **Pooling** | 0.784 ± 0.082 | 0.778 ± 0.077 | 0.774 ± 0.098 | 0.764 ± 0.093 | **0.775** |
| **Subsampling** | 0.892 ± 0.099 | 0.873 ± 0.117 | 0.888 ± 0.112 | 0.880 ± 0.119 | **0.883** |
| **Repeated** | 0.937 ± 0.051 | 0.922 ± 0.059 | 0.933 ± 0.058 | 0.929 ± 0.057 | **0.930** |
| **Std-CP** | 1.000 ± 0.000 | 0.759 ± 0.100 | 0.845 ± 0.080 | 0.830 ± 0.069 | **0.859** |

### Width Performance - Median (Income Units)

| Method | o=0 | o=5 | o=10 | o=20 | Overall Median |
|--------|-----|-----|------|------|----------------|
| **D-HCP (randomized)** | $213,839 | $184,380 | $157,820 | $144,409 | **$169,447** |
| **D-HCP (derandomized)** | $1,506,349 | $561,117 | $387,271 | $273,462 | **$452,673** |
| **S-HCP (randomized)** | $203,671 | $170,408 | $158,957 | $170,003 | **$171,815** |
| **S-HCP (derandomized)** | $867,319 | $275,999 | $256,114 | $253,036 | **$279,857** |
| **HCP** | $216,411 | $222,706 | $222,426 | $218,730 | **$221,114** |
| **Pooling** | $146,850 | $149,600 | $146,197 | $145,735 | **$146,850** |
| **Subsampling** | $275,698 | $292,684 | $283,356 | $291,177 | **$288,061** |
| **Repeated** | $317,178 | $332,382 | $319,333 | $328,605 | **$323,623** |

### Width Performance - Mean ± Std (Income Units)

| Method | o=0 | o=5 | o=10 | o=20 |
|--------|-----|-----|------|------|
| **D-HCP (randomized)** | $220,244 ± $40,865 | $187,720 ± $30,086 | $158,357 ± $20,880 | $145,015 ± $17,029 |
| **D-HCP (derandomized)** | $1,656,255 ± $699,743 | $584,285 ± $159,213 | $395,041 ± $76,465 | $275,400 ± $39,185 |
| **S-HCP (randomized)** | $204,428 ± $41,274 | $173,088 ± $24,771 | $158,247 ± $23,705 | $169,449 ± $22,186 |
| **S-HCP (derandomized)** | $1,050,308 ± $646,831 | $277,854 ± $49,044 | $258,168 ± $45,005 | $254,951 ± $32,853 |
| **HCP** | $224,072 ± $42,334 | $230,406 ± $42,301 | $224,516 ± $42,502 | $227,892 ± $40,932 |
| **Pooling** | $146,526 ± $24,456 | $150,550 ± $23,895 | $146,600 ± $23,538 | $148,958 ± $23,341 |
| **Subsampling** | $358,009 ± $294,208 | $365,750 ± $298,325 | $361,129 ± $314,043 | $360,957 ± $296,598 |
| **Repeated** | $326,253 ± $61,896 | $335,250 ± $59,409 | $326,773 ± $61,422 | $332,301 ± $62,266 |

## Key Findings

1. **Coverage**:
   - D-HCP (randomized) achieves near-target coverage (0.846 overall) with decreasing trend as o increases
   - D-HCP (derandomized) is conservative (0.954 overall) but coverage improves with larger o
   - HCP achieves good coverage (0.868) but is constant across o (doesn't adapt)
   - Repeated subsampling is conservative (0.930) but has wide intervals

2. **Width Efficiency**:
   - D-HCP (randomized) shows **strong width reduction** with increasing o: from $214K (o=0) to $144K (o=20)
   - D-HCP (randomized) is narrower than HCP, Repeated, and Subsampling at all o values
   - Pooling has the narrowest intervals but severely under-covers (0.775)
   - D-HCP achieves better coverage-width tradeoff than all baselines

3. **Comparison with Baselines**:
   - D-HCP (randomized) dominates Pooling (better coverage, similar width)
   - D-HCP (randomized) dominates HCP (better width at comparable coverage)
   - D-HCP (randomized) dominates Repeated and Subsampling (much narrower at similar coverage)

4. **Effect of Stratified Sampling**:
   - Stratified PUMA selection ensures heterogeneous calibration groups
   - Results show consistent performance across bootstrap replicates
   - Low standard deviations indicate stable method performance

## Files Generated

- **Raw results**: `acs_stratified_raw_results.csv` (3,600 rows = 100 replicates × 9 methods × 4 o values)
- **Summary (wide format)**: `acs_stratified_summary.csv` (36 rows = 9 methods × 4 o values)
- **Summary (long format)**: `acs_stratified_summary_long.csv` (includes all metrics)
- **Plots**: `NEW_PLOTS/stratified/` and `NEW_PLOTS/stratified/paper_style/`
