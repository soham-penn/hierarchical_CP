# Hierarchical Conformal Prediction

Python implementation of donor-HCP (GHCP), sample-HCP, and baseline conformal methods on simulated and real grouped data.

## Repository layout

```
hierarchical_cp/
├── code/
│   ├── paths.py
│   ├── shared/
│   │   ├── dgp/              # Simulation core
│   │   └── plot_engine.py    # Paper figure engine
│   └── marginal/             # True-marginal runners + paper plots
├── methods/                  # GHCP / HCP / μ-estimators
├── scores.py
├── results_marginal/dgp/     # Raw DGP trials (paper RF suite)
├── plots_marginal/
│   ├── dgp_true_marginal_rf/ # Simulations (figures, tables)
│   └── acs/
│       ├── min21/            # ACS paper suite (active)
│       └── old/              # Earlier ACS variants
├── real_data/acs/            # ACS preprocessing + plotting
└── old/                      # Archived code and experiments
```

Canonical **Simulations** map: [`plots_marginal/dgp_true_marginal_rf/README.md`](plots_marginal/dgp_true_marginal_rf/README.md)  
Canonical **ACS** map: [`plots_marginal/acs/README.md`](plots_marginal/acs/README.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Simulations (paper)

True-marginal latent intercept γ = 5, random-forest global μ, absolute residual score, B = 1000.

```bash
.venv/bin/python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.10 \
  --gamma 5 \
  --configs fixedN21,poissonNmean21 \
  --total_replicates 1000 \
  --n_workers 8 \
  --quantile-mode deterministic

.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
```

## ACS (paper)

Foreign-born California cohort, age 25–54, hours ≥ 40, YOEP ≥ 2000, PUMA size ≥ 21, target index 20.

```bash
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py --B 1000 --n_workers 8 --plot
```

Results and figures: `plots_marginal/acs/min21/`.  
B = 200 verification artifacts: `plots_marginal/acs/min21/demo_b200/`.

```bash
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

## Methods

| Label | Module |
|-------|--------|
| GHCP (donor-HCP) | `methods/donor_hcp.py` |
| Sample-HCP | `methods/sample_hcp.py` |
| HCP / pooling / sub / rep | `methods/baseline_hcp.py` |
| μ OLS / RF | `methods/mu_methods.py` |
| Scores / quantiles | `scores.py` |

## Archive (`old/`)

Superseded simulation suites, ACS experiment variants, diagnostics, and capture pipelines. See [`old/README.md`](old/README.md).
