# ACS group-size ignorability diagnostics

## What is being tested

These are **observable diagnostics**, not direct tests of the latent assumption
`N_j ⟂ (latent group law of group j)`.

We check two observable implications on PUMA groups:

1. **X-summary diagnostic:** is PUMA size `N_j` associated with group-level
   summaries of covariates?
2. **Residual-summary diagnostic:** after fitting the same pooled global predictor
   as the ACS experiment (ols, outcome scale `log1p`), is `N_j`
   associated with group-level summaries of residuals?

Equal-size subsampling (`m0` per group) is used within each subsample replicate
so precision differences across groups do not mechanically drive associations.

## Sample construction

- Grouping variable: `puma`
- Eligible groups: 43 (size ≥ 21)
- Total individuals in eligible groups: 1551
- PUMA size range: 21–126
- Subsample size m0: 21; subsample replicates B_sub: 200

## Census PUMA vs analytic `N_j` (read this first)

Census PUMAs are **geographic areas** designed to contain ~100,000+ residents.
They are **not** formed using education, income, or other covariates in this analysis.

In this script, `N_j` is **not** total PUMA population. It is the number of ACS rows
in the **heavily filtered analytic subpopulation** per PUMA (foreign-born, recent entry
since 2012, age 25–54, full-time hours, income ≥ \$10K, etc.). After filtering, most
CA PUMAs contain very few such individuals (median ~9); only PUMAs with `N_j ≥ 21` enter
the diagnostic (41 of 258 PUMAs with any data).

So a strong `N_j` vs X association means: **PUMAs where more recent immigrants in our
niche subpopulation happen to live also tend to have different covariate mixes**
(e.g. higher BA+ share, better English scores). That reflects **spatial sorting /
concentration of the subpopulation**, not how Census drew PUMA boundaries.

## Interpretation

- **Association between `N_j` and X summaries** suggests group size may depend on
  group composition (observable covariate mix).
- **Association between `N_j` and residual summaries** suggests group size may depend
  on unexplained group-specific outcome behavior after controlling for X via the
  pooled global model.
- **Lack of strong association is supportive but not definitive** for size ignorability;
  latent group heterogeneity not captured by these summaries would not be detected.

## Omnibus permutation tests (median across subsamples)

- X summaries: median omnibus stat = 1.0554, median perm p = 0.0040, fraction p<0.05 = 1.000
- Residual summaries: median omnibus stat = 0.4644, median perm p = 0.0220, fraction p<0.05 = 0.860

Omnibus statistic: sum of squared Spearman ρ across components; p-value from
permuting `N_j` across groups.

## Strongest univariate associations (median Spearman across subsamples)

### X summaries

| summary | median ρ | median p | frac p<0.05 |
|---|---:|---:|---:|
| prop_educ_BAplus | 0.515 | 0.0004 | 0.955 |
| english_mean | -0.459 | 0.0020 | 1.000 |
| age_sq_sd | -0.357 | 0.0189 | 0.860 |
| age_sq_mean | -0.344 | 0.0241 | 0.845 |
| age_sd | -0.338 | 0.0267 | 0.755 |

### Residual summaries

| summary | median ρ | median p | frac p<0.05 |
|---|---:|---:|---:|
| resid_q25 | 0.399 | 0.0081 | 0.965 |
| resid_median | 0.330 | 0.0309 | 0.720 |
| resid_mean | 0.288 | 0.0610 | 0.405 |
| resid_q75 | 0.276 | 0.0730 | 0.345 |
| resid_sd | 0.169 | 0.2782 | 0.075 |
