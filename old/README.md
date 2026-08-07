# Archived / non-final artifacts

Active Simulations deliverable: `plots_marginal/dgp_true_marginal_rf/`  
(see that folder’s README for the paper section map).

Active ACS deliverable: `plots_marginal/acs/min21/` (see that folder’s README).

## Keep (not here)

| Path | Role |
|------|------|
| `plots_marginal/dgp_true_marginal_rf/` | Paper RF γ=5 figures, summaries, tables |
| `results_marginal/dgp/true_marg_latent_rf_gamma5p0_*` | Raw trials for that suite |
| `code/marginal/run_true_marginal_latent_intercept_rf_experiments.py` | Paper runner |
| `code/marginal/run_true_marginal_latent_intercept_experiments.py` | Shared latent-intercept helpers |
| `code/marginal/plot_paper.py` | Active suites: `dgp_rf`, `acs`, `acs_xgboost` |
| `code/marginal/export_paper_tables_mean_width.py` | Mean-width LaTeX tables |
| `code/shared/dgp/`, `code/shared/plot_engine.py`, `methods/`, `scores.py` | Core used by the paper suite |
| `code/marginal/run_acs_experiments.py` | Core ACS runner |
| `code/marginal/run_acs_yoep_fb_min21_permute.py` | Paper ACS launcher |
| `plots_marginal/acs/min21/` | Active ACS results and figures |

## Contents

| Path | What |
|------|------|
| `plots_marginal/acs/old/` | Superseded ACS experiment outputs |
| `code/marginal/acs_runners/` | Archived ACS driver scripts |
| `real_data/acs/` (archived files) | Diagnostics, alternate summaries |
| `code/conditional/` | Calibration-conditional experiments |
| `code/dgp_capture/`, `code/acs_capture/` | μ-capture pipelines |
| `code/run_true_marginal_experiments.py` | OLS joint-Gaussian true-marginal |
| `code/run_true_marginal_randomization_stability.py` | Randomization-stability runner |
| `code/plot_paper_full_suites.py` | Former multi-suite `plot_paper.py` |
| `code/joint_xy_marginal_common.py` | Shared draw helper for OLS path |
| `diagnostics/` | Ad-hoc diagnostics |
| `plots_marginal_archives/` | Former `plots_marginal/old/` |
| `plots_conditional/`, `results_conditional/` | Conditional paper outputs |
| `results_marginal_dgp_old/` | Former `results_marginal/dgp/old/` |
| `dgp_true_marginal_rf_median_era/` | Pre–mean-width RF figures |
| `dgp_true_marginal_rf_capture_data/` | Intermediate capture dumps |
| `.mplconfig/` | Matplotlib font cache |

Regenerate final DGP plots/tables:

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
```
