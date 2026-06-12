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
│   │   ├── run_true_marginal_latent_intercept_experiments.py
│   │   ├── run_true_marginal_latent_intercept_rf_experiments.py
│   │   ├── run_true_marginal_randomization_stability.py
│   │   ├── run_acs_experiments.py
│   │   └── plot_paper.py        # Unified marginal plotting (--suite)
│   └── conditional/             # Calibration-conditional experiments (fixed calib, many test groups)
│       ├── latent_intercept_core.py
│       ├── run_latent_intercept_gamma5.py
│       └── plot_paper.py
├── methods/                     # Conformal methods and μ-estimators
├── scores.py                    # Conformal scores and weighted quantiles
├── results_marginal/            # CSV outputs for true-marginal runs (raw CSVs gitignored)
├── results_conditional/         # CSV outputs for calibration-conditional runs
├── plots_marginal/              # Paper figures + summary tables (tracked in git)
└── plots_conditional/           # Paper figures + summary tables (tracked in git)
```

Real-data assets (ACS PUMS download, cleaning) live under `real_data/acs/`.

## What is version-controlled

| Tracked in git | Not tracked (regenerate locally) |
|--------------|-----------------------------------|
| All code under `code/`, `methods/`, `scores.py` | `*raw_results*.csv`, `chunks/`, `*.log` |
| Paper PDFs and summary CSVs under `plots_marginal/` and `plots_conditional/` | ACS raw PUMS (`acs_data_all50states.csv`, `folktables_cache/`) |
| Randomization-stability summary CSVs in `results_marginal/dgp/` | ACS `*_detailed.csv` (large per-run outputs) |

Cloning the repo gives you the **exact figures** in the paper folders. To reproduce the underlying numbers from scratch, run the experiments below (several hours on a laptop) and then regenerate plots.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below assume the virtual environment is active and you are in the repo root.

### Shared experiment settings (paper runs)

These settings match the figures currently in the repository:

- **Miscoverage grid** (`PAPER_ALPHA_GRID` in `code/shared/plot_engine.py`):
  `0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25` (nine values)
- **Replicates:** 1000 per configuration
- **Conformal quantile:** `--quantile-mode deterministic` (default)
- **Workers:** `--n_workers 8` is a reasonable default on an 8-core machine; increase if you have more cores
- **Group-size configs:** `fixedN21` (N_k = 21) and `poissonNmean21` (N_k = 1 + Poisson(20))

## Reproducing paper experiments

### 1. DGP true marginal — OLS (`dgp_ols`)

Joint Gaussian DGP, global μ via OLS. Results → `results_marginal/dgp/true_marg_{config}_alpha{tag}/`.

```bash
ALPHAS="0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25"

python code/marginal/run_true_marginal_experiments.py \
  --alphas "$ALPHAS" \
  --configs fixedN21,poissonNmean21 \
  --total_replicates 1000 \
  --n_workers 8 \
  --quantile-mode deterministic
```

**Randomization stability** (Poisson panel in `dgp_ols` figures; ~1–2 hours):

```bash
python code/marginal/run_true_marginal_randomization_stability.py \
  --alphas "$ALPHAS" \
  --configs fixedN21,poissonNmean21 \
  --n_datasets 50 \
  --n_reruns 100 \
  --n_workers 8
```

Writes `true_marg_{config}_randomization_stability_{dataset_metrics,summary}.csv` under `results_marginal/dgp/`.

**Plots** → `plots_marginal/dgp_true_marginal/`:

```bash
python code/marginal/plot_paper.py --suite dgp_ols
```

### 2. DGP true marginal — latent intercept γ = 5, OLS (`dgp_latent_gamma5`)

Each group draws B_j ~ N(0, 25) added to the Y mean. Default `--alphas` is already the full paper grid.

```bash
python code/marginal/run_true_marginal_latent_intercept_experiments.py \
  --gamma 5 \
  --configs fixedN21,poissonNmean21 \
  --total_replicates 1000 \
  --n_workers 8 \
  --quantile-mode deterministic
```

Results → `results_marginal/dgp/true_marg_latent_gamma5p0_{config}_alpha{tag}/`.

**Plots** → `plots_marginal/dgp_true_marginal_latent_gamma5/`:

```bash
python code/marginal/plot_paper.py --suite dgp_latent_gamma5
```

### 3. DGP true marginal — latent intercept γ = 5, Random Forest (`dgp_rf`)

Global μ via random forest (50 trees, min leaf 5). The committed figures use **α ∈ {0.05, 0.10}** only (four configurations total); a full nine-alpha grid can be run by omitting `--alphas`.

```bash
python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.10 \
  --gamma 5 \
  --configs fixedN21,poissonNmean21 \
  --total_replicates 1000 \
  --n_workers 8 \
  --quantile-mode deterministic
```

Results → `results_marginal/dgp/true_marg_latent_rf_gamma5p0_{config}_alpha{tag}/`.  
Expect several hours per α on a laptop (RF is much slower than OLS).

**Plots** → `plots_marginal/dgp_true_marginal_rf/`:

```bash
python code/marginal/plot_paper.py --suite dgp_rf
```

### 4. ACS true marginal (`acs`)

**Data** (one-time download, not in git):

```bash
python real_data/acs/download_acs_ca_pums.py
```

This writes `real_data/acs/data/acs_data_all50states.csv` (California 2018 PUMS via `folktables`).

**Experiments** — marginal design, row permutation within PUMAs (default), target index 20, B = 1000:

```bash
for alpha in 0.05 0.075 0.10 0.125 0.15 0.175 0.20; do
  python code/marginal/run_acs_experiments.py \
    --design marginal \
    --alpha "$alpha" \
    --B 1000 \
    --n_workers 8 \
    --quantile-mode deterministic
done
```

Results → `results_marginal/acs/true_marginal_permuted_alpha{tag}/acs_true_marg_alpha{tag}_detailed.csv`.

**Plots** → `plots_marginal/acs/`:

```bash
python code/marginal/plot_paper.py --suite acs
```

### 5. Calibration-conditional latent intercept γ = 5 (`conditional`)

Fix calibration data within each outer experiment; 50 outer runs × 100 test groups. Includes a randomization-stability block.

```bash
python code/conditional/run_latent_intercept_gamma5.py
```

Results → `results_conditional/dgp/latent_gamma5_{fixedN21,poissonNmean21}/`.

**Plots** → `plots_conditional/dgp_latent_intercept_gamma5/`:

```bash
python code/conditional/plot_paper.py --suite latent_gamma5
```

### Regenerate all marginal figures at once

```bash
python code/marginal/plot_paper.py --all
```

## Experiment designs

### True marginal (`code/marginal/`)

Each replicate draws **new calibration groups and a new test group**. One test observation per replicate (target index 35 in simulation; index 20 in ACS). This evaluates **marginal** coverage over the data-generating process.

**DGP (joint Gaussian):** K = 20 groups, d = 5, U ~ Unif[1, 5]^d, Z = (X, Y) multivariate normal with mean μ(U) = U² and covariance Σ(U) = (1 − ρ) diag(U) + ρ 11ᵀ, ρ = 0.5.

| Config | Group sizes |
|--------|-------------|
| `fixedN21` | N_k = 21, target N = 36 |
| `poissonNmean21` | N_k = 1 + Poisson(20) |

**Latent intercept variant:** each group draws B_j ~ N(0, γ²) added to the Y mean (γ = 5 for paper runs).

**Methods:** D-HCP (with/without within-group training), S-HCP variants, baselines (HCP, pooling, subsampling, repeated subsampling, Std-CP). Global μ via OLS or random forest.

**ACS:** California ACS income; stratified PUMA sampling; row permutation within PUMAs for exchangeability.

### Calibration conditional (`code/conditional/`)

Fix calibration data within each outer experiment; generate **100 test groups** per outer run; **50** outer experiments. Evaluates performance conditional on a fixed calibration sample.

## Methods

| Label | Module | Description |
|-------|--------|-------------|
| D-HCP | `methods/donor_hcp.py` | Donor group selection + hierarchical calibration |
| S-HCP | `methods/sample_hcp.py` | Sample-split hierarchical CP |
| HCP, Pooling, Sub, Rep | `methods/baseline_hcp.py` | Standard baselines on global μ |
| μ OLS / RF | `methods/mu_methods.py` | Global predictor with optional within-group shrinkage |

Conformal thresholds use `scores.py` (`--quantile-mode deterministic` by default).

## Quick reference: output locations

| Suite | Experiment runner | Results directory | Plot directory |
|-------|-------------------|-------------------|----------------|
| `dgp_ols` | `run_true_marginal_experiments.py` | `results_marginal/dgp/true_marg_*` | `plots_marginal/dgp_true_marginal/` |
| `dgp_latent_gamma5` | `run_true_marginal_latent_intercept_experiments.py` | `results_marginal/dgp/true_marg_latent_gamma5p0_*` | `plots_marginal/dgp_true_marginal_latent_gamma5/` |
| `dgp_rf` | `run_true_marginal_latent_intercept_rf_experiments.py` | `results_marginal/dgp/true_marg_latent_rf_gamma5p0_*` | `plots_marginal/dgp_true_marginal_rf/` |
| `acs` | `run_acs_experiments.py` | `results_marginal/acs/true_marginal_permuted_*` | `plots_marginal/acs/` |
| conditional | `run_latent_intercept_gamma5.py` | `results_conditional/dgp/latent_gamma5_*` | `plots_conditional/dgp_latent_intercept_gamma5/` |
