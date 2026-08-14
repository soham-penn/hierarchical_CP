# Archived / non-final artifacts

**Active paper deliverable:** [`paper-results/`](../paper-results/README.md)

- Simulations: `paper-results/dgp_true_marginal_rf/`
- ACS: `paper-results/acs/min21/`
- Raw DGP CSVs: `paper-results/results_marginal/dgp/`

## Keep (not archived — live under repo root / `paper-results/`)

| Path | Role |
|------|------|
| `paper-results/` | Canonical figures, tables, summaries, ACS suite, nested raw CSVs |
| `code/marginal/run_true_marginal_latent_intercept_rf_experiments.py` | Paper DGP runner |
| `code/marginal/run_true_marginal_latent_intercept_experiments.py` | Shared latent-intercept helpers |
| `code/marginal/plot_paper.py` | Active suites: `dgp_rf`, `acs`, … |
| `code/marginal/export_paper_tables_mean_width.py` | Mean-width LaTeX tables |
| `code/shared/dgp/`, `code/shared/plot_engine.py`, `methods/`, `scores.py` | Core used by the paper suite |
| `code/marginal/run_acs_experiments.py` | Core ACS runner |
| `code/marginal/run_acs_yoep_fb_min21_permute.py` | Paper ACS launcher |
| `code/paths.py` | Defaults → `paper-results/` |

## Contents of `old/`

| Path | What |
|------|------|
| `plots_marginal/` | Former top-level plot tree (pre–`paper-results` rename) |
| `new_poisson_marginal_scratch/` | Temporary Poi(25) staging dir; results merged into `paper-results/` |
| `plots_marginal_archives/` | Even older `plots_marginal/old/` dumps |
| `code/` | Archived runners (capture, conditional, alt ACS drivers, …) |
| `results_conditional/`, `plots_conditional/` | Conditional-calibration experiments |
| `results_marginal_dgp_old/` | Former `results_marginal/dgp/old/` |
| `dgp_true_marginal_rf_median_era/` | Pre–mean-width RF figures |
| `dgp_true_marginal_rf_capture_data/` | Intermediate capture dumps |
| `diagnostics/` | Ad-hoc diagnostics |

Regenerate final DGP plots/tables from shipped CSVs:

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
```
