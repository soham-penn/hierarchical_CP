# ACS Stratified Coverage and Width Comparison

Target nominal coverage: 0.80. Widths and endpoints are in income dollars.

Endpoint caveat: a single lower/upper endpoint is meaningful only for methods that produce one global interval. For D-HCP, HCP, and pooled covariate CP, endpoints vary by replicate, target unit, and covariates, so the table reports `varies` rather than a fake universal interval.

## GitHub Table

| Method | Variant | o | Coverage mean | Coverage std | Lower endpoint | Upper endpoint | Width mean | Width std | Width median | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D-HCP | ACS stratified D-HCP randomized | 0 | 0.875 | 0.063 | varies | varies | $220,244 | $40,865 | $213,839 | Covariate/target-dependent endpoints |
| D-HCP | ACS stratified D-HCP randomized | 5 | 0.846 | 0.076 | varies | varies | $187,720 | $30,086 | $184,380 | Covariate/target-dependent endpoints |
| D-HCP | ACS stratified D-HCP randomized | 10 | 0.839 | 0.075 | varies | varies | $158,357 | $20,880 | $157,820 | Covariate/target-dependent endpoints |
| D-HCP | ACS stratified D-HCP randomized | 20 | 0.824 | 0.073 | varies | varies | $145,015 | $17,029 | $144,409 | Covariate/target-dependent endpoints |
| HCP | ACS stratified HCP baseline | 0 | 0.877 | 0.058 | varies | varies | $224,072 | $42,334 | $216,411 | Reported at o=0; endpoints vary |
| Pooled income quantile | No learning, no CP: stored empirical alpha/2 and 1-alpha/2 income quantiles | global | -- | -- | $18,000 | $150,000 | $132,000 | -- | $132,000 | Single global interval |
| Pooled mean CP | No learning CP: global mean of log-income with pooled absolute-residual split conformal radius | global | -- | -- | $17,820 | $150,000 | $132,180 | -- | $132,180 | Single global interval |
| Pooled covariate CP | Learns with covariates but ignores group structure: stored true-marginal Pooling baseline | global | -- | -- | varies | varies | $144,193 | $66,488 | $142,781 | Covariate-dependent endpoints |

## Endpoint Overlap

Only the two global intervals can be compared by fixed endpoints. The pooled income quantile interval is `[$18,000, $150,000]`; the pooled mean CP interval is approximately `[$17,820, $150,000]`. They almost exactly overlap, and the pooled income quantile interval lies completely inside the pooled mean CP interval.

For D-HCP, HCP, and pooled covariate CP, containment is not a single statement because each target can receive a different interval. To compare containment for those methods, we would need the raw lower/upper endpoints for every replicate and target, not just the summary widths.

## LaTeX Table

```latex
\begin{table}[t]
\centering
\caption{ACS stratified coverage and prediction set width comparison at nominal coverage 0.80. Widths and global endpoints are in income dollars.}
\label{tab:acs-stratified-width-comparison}
\begin{tabular}{llrrrrrrrr}
\toprule
Method & Variant & $o$ & Coverage & Cov. SD & Lower & Upper & Width Mean & Width SD & Width Median \\
\midrule
D-HCP & Stratified D-HCP & 0 & 0.875 & 0.063 & varies & varies & \$220{,}244 & \$40{,}865 & \$213{,}839 \\
D-HCP & Stratified D-HCP & 5 & 0.846 & 0.076 & varies & varies & \$187{,}720 & \$30{,}086 & \$184{,}380 \\
D-HCP & Stratified D-HCP & 10 & 0.839 & 0.075 & varies & varies & \$158{,}357 & \$20{,}880 & \$157{,}820 \\
D-HCP & Stratified D-HCP & 20 & 0.824 & 0.073 & varies & varies & \$145{,}015 & \$17{,}029 & \$144{,}409 \\
HCP & Stratified HCP & 0 & 0.877 & 0.058 & varies & varies & \$224{,}072 & \$42{,}334 & \$216{,}411 \\
Pooled income quantile & No learning, no CP & global & -- & -- & \$18{,}000 & \$150{,}000 & \$132{,}000 & -- & \$132{,}000 \\
Pooled mean CP & No learning CP & global & -- & -- & \$17{,}820 & \$150{,}000 & \$132{,}180 & -- & \$132{,}180 \\
Pooled covariate CP & Covariate CP, no groups & global & -- & -- & varies & varies & \$144{,}193 & \$66{,}488 & \$142{,}781 \\
\bottomrule
\end{tabular}
\end{table}
```

Source CSV: `paper_plots/acs_stratified_coverage_width_comparison.csv`.
