# ACS group-size ignorability diagnostics

## Scope (read this first)

The real-data ACS study is an **empirical illustration under an approximate
working model**, not a test of the theoretical size-ignorability assumption
`N_j ⟂ (U_j, μ_j)`.

That assumption requires retained PUMA size to carry **no information** about
the PUMA-specific covariate and outcome distribution. Our filters (foreign-born,
recent entry, age, full-time hours) do **not** depend on income, but PUMAs with
many retained individuals may still differ systematically in occupation, education,
language, housing markets, and income dispersion.

These diagnostics relate observable group summaries to `N_j` using the same
cohort and pooled global RF predictor as the paper ACS runs.

## What is checked

1. **Y summaries:** mean(Y_j), sd(Y_j), and raw income moments
2. **X summaries:** group-level covariate means and category shares
3. **Residual summaries:** group-level moments of Y − μ_global(X)

Equal-size subsampling (`m0` per PUMA) within each replicate removes mechanical
precision differences. Stratified tertile and size-band analyses assess sensitivity.

## Sample construction

- Grouping: `puma`; eligible PUMAs: 212 (N_j ≥ 21)
- Individuals in eligible PUMAs: 11568
- N_j range: 21–252
- YOEP ≥ 2000; outcome scale: `income`
- Subsample m0=21; replicates B_sub=200

## Omnibus permutation tests (median across subsamples)

- Y summaries: median stat=0.0052, median perm p=0.6270, frac p<0.05=0.000
- X summaries: median stat=0.0975, median perm p=0.0750, frac p<0.05=0.390
- Residual summaries: median stat=0.1432, median perm p=0.0040, frac p<0.05=0.995

Omnibus statistic: sum of squared Spearman ρ; p-value from permuting N_j across PUMAs.

## Strongest univariate associations (median Spearman across subsamples)

### Y summaries

| summary | median ρ | median p | frac p<0.05 |
|---|---:|---:|---:|
| income_mean | 0.041 | 0.5497 | 0.000 |
| y_mean | 0.041 | 0.5497 | 0.000 |
| income_sd | -0.011 | 0.7446 | 0.000 |
| y_sd | -0.011 | 0.7446 | 0.000 |

### X summaries

| summary | median ρ | median p | frac p<0.05 |
|---|---:|---:|---:|
| hours_mean | -0.143 | 0.0369 | 0.630 |
| prop_married | -0.116 | 0.0915 | 0.345 |
| hours_sd | -0.116 | 0.0923 | 0.345 |
| age_mean | -0.110 | 0.1109 | 0.210 |
| age_sq_mean | -0.103 | 0.1336 | 0.165 |

### Residual summaries

| summary | median ρ | median p | frac p<0.05 |
|---|---:|---:|---:|
| resid_median | 0.242 | 0.0004 | 1.000 |
| resid_q25 | 0.182 | 0.0079 | 0.870 |
| resid_mean | 0.175 | 0.0109 | 0.850 |
| resid_q75 | 0.132 | 0.0546 | 0.475 |
| resid_sd | 0.004 | 0.7016 | 0.000 |

## Stratified sensitivity (high vs low N_j tertile means)

| summary | low tertile mean | high tertile mean | high − low |
|---|---:|---:|---:|
| y_mean | 6.114e+04 | 7.11e+04 | 9964 |
| income_mean | 6.114e+04 | 7.11e+04 | 9964 |
| resid_q75 | 7397 | 1.692e+04 | 9527 |
| resid_mean | -7063 | 1866 | 8929 |
| resid_median | -9503 | -3385 | 6118 |

## Size-band Spearman (within similar N_j)

| band | summary | ρ | p | n PUMAs |
|---|---|---:|---:|---:|
| 21-35 | y_mean | -0.137 | 0.2845 | 63 |
| 21-35 | y_sd | -0.121 | 0.3454 | 63 |
| 21-35 | age_mean | -0.012 | 0.9270 | 63 |
| 21-35 | hours_mean | 0.007 | 0.9562 | 63 |
| 21-35 | prop_female | 0.115 | 0.3689 | 63 |
| 21-35 | resid_sd | -0.027 | 0.8366 | 63 |
| 36-60 | y_mean | 0.050 | 0.6534 | 83 |
| 36-60 | y_sd | -0.011 | 0.9178 | 83 |
| 36-60 | age_mean | -0.059 | 0.5951 | 83 |
| 36-60 | hours_mean | -0.163 | 0.1411 | 83 |
| 36-60 | prop_female | 0.160 | 0.1478 | 83 |
| 36-60 | resid_sd | 0.060 | 0.5906 | 83 |
| 61+ | y_mean | 0.269 | 0.0292 | 66 |
| 61+ | y_sd | 0.192 | 0.1228 | 66 |
| 61+ | age_mean | -0.192 | 0.1233 | 66 |
| 61+ | hours_mean | -0.227 | 0.0669 | 66 |
| 61+ | prop_female | -0.093 | 0.4560 | 66 |
| 61+ | resid_sd | 0.196 | 0.1147 | 66 |

## Interpretation for the paper

- **Rejecting ignorability outright is not possible** from observables alone.
- **Non-trivial association between N_j and Y/X/residual summaries** indicates
  PUMA size may proxy unmodeled heterogeneity; report alongside coverage results.
- **Weak association is supportive but not definitive**; latent group structure
  not captured by these summaries would remain undetected.
