# Hierarchical Conformal Prediction

Python implementation of donor-HCP (D-HCP), sample-HCP (S-HCP), and baseline conformal methods on simulated and real grouped data.

## Repository layout

```
hierarchical_cp/
├── code/
│   ├── paths.py                 # Central path constants
│   ├── shared/
│   │   ├── dgp/                 # Simulation core (data generation, experiment runner)
│   │   └── plot_engine.py       # Shared plotting engine for paper figures
│   ├── marginal/                # True-marginal experiments (regenerate all data each replicate)
│   │   ├── run_true_marginal_experiments.py
│   │   ├── run_true_marginal_rf_experiments.py
│   │   ├── run_true_marginal_latent_intercept_experiments.py
│   │   ├── run_true_marginal_latent_intercept_rf_experiments.py
│   │   ├── run_true_marginal_randomization_stability.py
│   │   ├── run_acs_experiments.py
│   │   ├── compare_latent_gamma5_ols_vs_rf.py
│   │   └── plot_paper.py        # Unified marginal plotting (--suite)
│   └── conditional/             # Calibration-conditional experiments (fixed calib, many test groups)
│       ├── latent_intercept_core.py
│       ├── run_latent_intercept_gamma5.py
│       └── plot_paper.py
├── methods/                     # Conformal methods and μ-estimators
├── scores.py                    # Conformal scores and weighted quantiles
├── results_marginal/            # CSV outputs for true-marginal runs
│   ├── dgp/
│   └── acs/
├── results_conditional/         # CSV outputs for calibration-conditional runs
│   └── dgp/
├── plots_marginal/              # Paper figures (true marginal)
└── plots_conditional/           # Paper figures (calibration conditional)
```

Real-data assets (ACS PUMS download, cleaning) remain under `real_data/acs/`.

## Experiment designs

### True marginal (`code/marginal/`)

Each replicate draws **new calibration groups and a new test group**. One test observation per replicate (target index 35 in simulation; index 20 in ACS). This evaluates **marginal** coverage over the data-generating process.

**DGP (joint Gaussian):** K=20 groups, d=5, U ~ Unif[1,5]^d, Z=(X,Y) multivariate normal with mean μ(U)=U² and covariance Σ(U)=(1−ρ)diag(U)+ρ11ᵀ, ρ=0.5.

| Config | Group sizes |
|--------|-------------|
| `fixedN21` | N_k = 21, target N = 36 |
| `poissonNmean21` | N_k = 1 + Poisson(20) |

**Latent intercept variant:** each group draws B_j ~ N(0, γ²) added to the Y mean (γ=5 for paper runs).

**Methods:** D-HCP (with/without within-group training), S-HCP variants, baselines (HCP, pooling, subsampling, repeated subsampling, Std-CP). Global μ via OLS or Random Forest.

**ACS:** California ACS income; stratified PUMA sampling; row permutation within PUMAs for exchangeability. See `run_acs_experiments.py --design marginal`.

### Calibration conditional (`code/conditional/`)

Fix calibration data within each outer experiment; generate **100 test groups** per outer run; 50 outer experiments. Evaluates performance conditional on a fixed calibration sample (legacy design, used for latent-intercept τ_B=5 paper figures).

## Running experiments

From the repo root (with `.venv` activated):

```bash
# DGP true marginal, OLS
python code/marginal/run_true_marginal_experiments.py \
  --alphas 0.05,0.10 --configs fixedN21,poissonNmean21 --total_replicates 1000 --n_workers 18

# DGP true marginal, latent intercept gamma=5, OLS
python code/marginal/run_true_marginal_latent_intercept_experiments.py \
  --alphas 0.05,0.10 --gamma 5

# DGP true marginal, latent intercept gamma=5, Random Forest
python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.10 --configs fixedN21,poissonNmean21 --total_replicates 1000 --n_workers 18 --gamma 5

# Calibration-conditional latent intercept gamma=5
python code/conditional/run_latent_intercept_gamma5.py

# ACS true marginal
python code/marginal/run_acs_experiments.py --alpha 0.10 --B 1000 --permute-rows
```

Results land in `results_marginal/dgp/` or `results_marginal/acs/` (marginal) and `results_conditional/dgp/` (conditional). Each run writes `*_raw_results_complete.csv` under a tagged subdirectory.

## Plotting

Two plotting entry points share `code/shared/plot_engine.py`:

```bash
# Marginal suites: dgp_ols | dgp_rf | dgp_latent_gamma5 | acs
python code/marginal/plot_paper.py --suite dgp_ols
python code/marginal/plot_paper.py --all

# Calibration-conditional
python code/conditional/plot_paper.py --suite latent_gamma5
```

Outputs go to `plots_marginal/<suite>/` or `plots_conditional/dgp_latent_intercept_gamma5/`.

## Methods

| Label | Module | Description |
|-------|--------|-------------|
| D-HCP | `methods/donor_hcp.py` | Donor group selection + hierarchical calibration |
| S-HCP | `methods/sample_hcp.py` | Sample-split hierarchical CP |
| HCP, Pooling, Sub, Rep | `methods/baseline_hcp.py` | Standard baselines on global μ |
| μ OLS / RF | `methods/mu_methods.py` | Global predictor with optional within-group shrinkage |

Conformal thresholds use `scores.py` (deterministic or randomized weighted quantile).

## Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib folktables
```

## ACS data

```bash
python real_data/acs/download_acs_ca_pums.py   # if needed
```

ACS experiments expect `real_data/acs/data/acs_data_all50states.csv`.
