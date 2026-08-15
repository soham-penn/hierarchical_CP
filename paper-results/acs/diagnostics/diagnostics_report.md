# ACS group-size ignorability diagnostics

These are the paper ACS diagnostics (written 14 Aug 2026). They use the
same cohort and pooled global RF as the ACS experiment: 2018 ACS 1-year CA PUMS,
foreign-born, YOEP ≥ 2000, age 25–54, hours ≥ 40, unweighted, income in dollars,
RF 50 trees / min leaf 5. Equal-size subsamples use m0=21 and B_sub=200;
omnibus p-values use 499 permutations of N_j across PUMAs.

Source files in this folder: `omnibus_test_results.csv`,
`y_summary_spearman_results.csv`, `x_summary_spearman_results.csv`,
`residual_summary_spearman_results.csv`, `stratified_tertile_comparison.csv`,
`size_band_spearman.csv`, `group_sizes_and_group_summaries.csv`.

## Scope

The ACS study is an empirical illustration under an approximate working model,
not a test of N_j ⟂ (U_j, μ_j). Filters do not depend on income, but larger
retained PUMAs can still differ in occupation, education, language, and income.
These checks relate observable PUMA summaries to N_j.

## Sample construction

- Eligible PUMAs (N_j ≥ 21): 212
- Individuals in eligible PUMAs: 11568
- N_j: min=21, median=46.0, mean=54.6, max=252
- N_j quartiles: [32.75, 46.0, 68.0]
- Outcome: raw income; predictor: pooled RF
- Subsample m0=21; B_sub=200; n_perm=499

## Omnibus permutation tests

Statistic: sum of squared Spearman ρ between N_j and the summary vector.
Median/mean are across the 200 equal-size subsamples.

| test | median stat | mean stat | median perm p | frac p<0.05 |
|---|---:|---:|---:|---:|
| y_summaries | 0.0052 | 0.0060 | 0.6270 | 0.000 |
| x_summaries | 0.0975 | 0.1033 | 0.0750 | 0.390 |
| residual_summaries | 0.1432 | 0.1464 | 0.0040 | 0.995 |

## Spearman ρ(N_j, summary), median across subsamples

Education-share rows with empty ρ are constant after the paper X-filter
(entry recency and class of worker are excluded from the ACS predictor, but
group summaries still include the available covariates).

### Y summaries

| summary | median ρ | mean ρ | median p | frac p<0.05 |
|---|---:|---:|---:|---:|
| income_mean | 0.041 | 0.041 | 0.5497 | 0.000 |
| y_mean | 0.041 | 0.041 | 0.5497 | 0.000 |
| income_sd | -0.011 | -0.012 | 0.7446 | 0.000 |
| y_sd | -0.011 | -0.012 | 0.7446 | 0.000 |

### X summaries

| summary | median ρ | mean ρ | median p | frac p<0.05 |
|---|---:|---:|---:|---:|
| hours_mean | -0.143 | -0.145 | 0.0369 | 0.630 |
| prop_married | -0.116 | -0.118 | 0.0915 | 0.345 |
| hours_sd | -0.116 | -0.117 | 0.0923 | 0.345 |
| age_mean | -0.110 | -0.107 | 0.1109 | 0.210 |
| age_sq_mean | -0.103 | -0.103 | 0.1336 | 0.165 |
| entry_recency_mean | -0.100 | -0.101 | 0.1453 | 0.155 |
| english_mean | 0.048 | 0.047 | 0.4855 | 0.000 |
| entry_recency_sd | -0.042 | -0.041 | 0.5136 | 0.015 |
| age_sd | 0.030 | 0.031 | 0.5940 | 0.010 |
| age_sq_sd | 0.030 | 0.027 | 0.6183 | 0.010 |
| prop_female | 0.019 | 0.015 | 0.6240 | 0.000 |
| prop_educ_<HS | — | — | — | 0.000 |
| prop_educ_BAplus | — | — | — | 0.000 |
| prop_educ_HS | — | — | — | 0.000 |
| prop_educ_SomeCollege | — | — | — | 0.000 |

### Residual summaries (Y − μ̂_global)

| summary | median ρ | mean ρ | median p | frac p<0.05 |
|---|---:|---:|---:|---:|
| resid_median | 0.242 | 0.240 | 0.0004 | 1.000 |
| resid_q25 | 0.182 | 0.183 | 0.0079 | 0.870 |
| resid_mean | 0.175 | 0.176 | 0.0109 | 0.850 |
| resid_q75 | 0.132 | 0.132 | 0.0546 | 0.475 |
| resid_sd | 0.004 | 0.003 | 0.7016 | 0.000 |

## High vs low N_j tertile (full eligible cohort)

Low tertile: 72 PUMAs; high tertile: 69 PUMAs.

| summary | low tertile mean | high tertile mean | high − low |
|---|---:|---:|---:|
| y_mean | 6.114e+04 | 7.11e+04 | 9964 |
| income_mean | 6.114e+04 | 7.11e+04 | 9964 |
| resid_q75 | 7397 | 1.692e+04 | 9527 |
| resid_mean | -7063 | 1866 | 8929 |
| resid_median | -9503 | -3385 | 6118 |
| income_sd | 5.585e+04 | 6.184e+04 | 5996 |
| y_sd | 5.585e+04 | 6.184e+04 | 5996 |
| resid_q25 | -2.808e+04 | -2.251e+04 | 5574 |
| resid_sd | 4.939e+04 | 5.446e+04 | 5077 |
| age_sq_mean | 1512 | 1470 | -42.1079 |
| age_sq_sd | 581.3 | 591.5 | 10.1950 |
| age_mean | 38.1160 | 37.5469 | -0.5691 |
| hours_mean | 43.6045 | 43.0398 | -0.5647 |
| entry_recency_mean | 10.7983 | 10.2809 | -0.5174 |
| hours_sd | 7.1067 | 6.6998 | -0.4069 |
| age_sd | 7.4449 | 7.5644 | 0.1195 |
| english_mean | 1.8465 | 1.9244 | 0.0779 |
| prop_educ_SomeCollege | 0.1815 | 0.1271 | -0.0544 |
| entry_recency_sd | 5.4412 | 5.3927 | -0.0485 |
| prop_married | 0.6990 | 0.6610 | -0.0380 |
| prop_educ_BAplus | 0.4534 | 0.4898 | 0.0364 |
| prop_educ_HS | 0.1855 | 0.1594 | -0.0261 |
| prop_educ_<HS | 0.2357 | 0.2289 | -0.0068 |
| prop_female | 0.3803 | 0.3826 | 0.0024 |

## Size-band Spearman (full eligible cohort, within similar N_j)

| band | n PUMAs | summary | ρ | p |
|---|---:|---|---:|---:|
| 21-35 | 63 | y_mean | -0.137 | 0.2845 |
| 21-35 | 63 | y_sd | -0.121 | 0.3454 |
| 21-35 | 63 | age_mean | -0.012 | 0.9270 |
| 21-35 | 63 | hours_mean | 0.007 | 0.9562 |
| 21-35 | 63 | prop_female | 0.115 | 0.3689 |
| 21-35 | 63 | resid_sd | -0.027 | 0.8366 |
| 36-60 | 83 | y_mean | 0.050 | 0.6534 |
| 36-60 | 83 | y_sd | -0.011 | 0.9178 |
| 36-60 | 83 | age_mean | -0.059 | 0.5951 |
| 36-60 | 83 | hours_mean | -0.163 | 0.1411 |
| 36-60 | 83 | prop_female | 0.160 | 0.1478 |
| 36-60 | 83 | resid_sd | 0.060 | 0.5906 |
| 61+ | 66 | y_mean | 0.269 | 0.0292 |
| 61+ | 66 | y_sd | 0.192 | 0.1228 |
| 61+ | 66 | age_mean | -0.192 | 0.1233 |
| 61+ | 66 | hours_mean | -0.227 | 0.0669 |
| 61+ | 66 | prop_female | -0.093 | 0.4560 |
| 61+ | 66 | resid_sd | 0.196 | 0.1147 |

## Interpretation

- Y summaries are weakly related to N_j (omnibus median p ≈ 0.63).
- Residual summaries are strongly related to N_j (omnibus median p ≈ 0.004;
  resid_median median ρ ≈ 0.24 in every subsample).
- High-N_j PUMAs have mean income about $10k above low-N_j PUMAs.
- Size-ignorability is not established on this cohort; the ACS figures remain
  an empirical illustration. Validity evidence is the theory and the DGP,
  where N_j is generated independently of (U_j, B_j).
