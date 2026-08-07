# ACS group-size ignorability diagnostics

## What is being tested

These are **observable diagnostics**, not direct tests of the latent assumption
`N_j ⟂ (latent group law of group j)`.

We check two observable implications on PUMA groups:

1. **X-summary diagnostic:** is PUMA size `N_j` associated with group-level
   summaries of covariates?
2. **Residual-summary diagnostic:** after fitting the same pooled global predictor
   as the ACS experiment (ols, outcome scale `income`), is `N_j`
   associated with group-level summaries of residuals?

Equal-size subsampling (`m0` per group) is used within each subsample replicate
so precision differences across groups do not mechanically drive associations.

## Sample construction

- Grouping variable: `puma`
- Eligible groups: 46 (size ≥ 21)
- Total individuals in eligible groups: 1676
- PUMA size range: 21–130
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

- X summaries: median omnibus stat = 0.9715, median perm p = 0.0040, fraction p<0.05 = 0.985
- Residual summaries: median omnibus stat = 0.1461, median perm p = 0.2480, fraction p<0.05 = 0.015

Omnibus statistic: sum of squared Spearman ρ across components; p-value from
permuting `N_j` across groups.

## Strongest univariate associations (median Spearman across subsamples)

### X summaries

| summary | median ρ | median p | frac p<0.05 |
|---|---:|---:|---:|
| prop_educ_BAplus | 0.509 | 0.0003 | 0.840 |
| english_mean | -0.457 | 0.0014 | 1.000 |
| age_sq_sd | -0.343 | 0.0197 | 0.840 |
| age_sq_mean | -0.336 | 0.0222 | 0.860 |
| age_mean | -0.323 | 0.0287 | 0.785 |

### Residual summaries

| summary | median ρ | median p | frac p<0.05 |
|---|---:|---:|---:|
| resid_sd | 0.222 | 0.1386 | 0.190 |
| resid_q75 | 0.196 | 0.1919 | 0.055 |
| resid_mean | 0.162 | 0.2813 | 0.005 |
| resid_median | 0.118 | 0.4338 | 0.000 |
| resid_q25 | 0.082 | 0.5769 | 0.005 |
