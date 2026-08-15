# Archived / non-final artifacts

**Keep live:** [`paper-results/`](../paper-results/README.md) (Sec. 3.1–3.2 + App. D.3–D.4), [`real_data/acs/`](../real_data/acs/README.md), and `code/`. Nothing under `old/` is required to reproduce the paper figures.

## Layout

```
old/
├── README.md                 # this file
├── code/                     # superseded runners / capture / conditional drivers
│   ├── acs_capture/
│   ├── dgp_capture/
│   ├── conditional/
│   └── marginal/             # alt ACS launchers (min31, yoep extended, …)
├── real_data/                # superseded real-data experiments
│   ├── acs/                  # old ACS plots/results/diagnostics variants
│   ├── blood_pressure/       # BP clinic study (not in paper)
│   └── scripts/              # bootstrap / stratified / early true-marginal runners
├── dgp/                      # archived simulation plot/result trees
│   ├── plots_marginal/       # pre–paper-results plot tree
│   ├── plots_marginal_archives/
│   ├── results_marginal_dgp_old/
│   ├── new_poisson_marginal_scratch/
│   ├── capture_data/         # frozen-μ capture dumps
│   └── median_era/           # pre–mean-width RF figures
├── conditional/              # conditional-calibration experiments
│   ├── plots/
│   └── results/
└── diagnostics/              # ad-hoc DGP/ACS diagnostic scratch
```

## Keep live (not archived)

| Path | Role |
|------|------|
| `paper-results/` | Canonical figures, tables, summaries, ACS suite, nested raw CSVs |
| `code/marginal/run_section_3_1.py` | Paper Sec. 3.1 launcher |
| `code/marginal/run_true_marginal_latent_intercept_rf_experiments.py` | Same as above (long name) |
| `code/marginal/plot_section_3_1.py` | Sec. 3.1 figures/tables |
| `code/marginal/run_section_3_2.py` | Paper Sec. 3.2 launcher |
| `code/marginal/run_acs_yoep_fb_min21_permute.py` | Same as Sec. 3.2 (long name) |
| `real_data/acs/` | ACS download, cleaning, paper plotter, size-ignorability diagnostics |
| `code/marginal/run_size_shift_sensitivity.py` | App. D.3 size–intercept \(\xi\) |
| `code/marginal/run_effect_of_weights.py` | App. D.4 merger-weight grid |

## What moved out of live `real_data/`

| Former path | Now |
|-------------|-----|
| `real_data/blood_pressure/` | `old/real_data/blood_pressure/` |
| `real_data/repeated_experiments*.py`, `acs_true_marginal_experiments.py`, `plot_results.py` | `old/real_data/scripts/` |
| `real_data/acs/results/`, `acs/plots/` | `old/real_data/acs/legacy_local/` |
| `real_data/acs/data/acs_top25_filtered.csv`, `acs_ca_puma_summary_trim2.csv` | `old/real_data/acs/data/` |

Regenerate final DGP plots/tables from shipped CSVs:

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/plot_paper.py --suite acs
```
