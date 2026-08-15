# Hierarchical Conformal Prediction

Python implementation of **GHCP** (Algorithm 1 in the paper: restricted donor hierarchical conformal prediction), HCP, and conformal baselines on simulated and ACS grouped data.

**Paper outputs live in [`paper-results/`](paper-results/README.md)** (figures, tables, summaries, raw trial CSVs).

Notation below follows the paper: initial test-group sample size \(o\), local merger weight \(\lambda_{\mathrm{local}}\) in (3), restriction \(\eta\), donated size \(N_{J_0}\).

---

## Quick start

```bash
git clone <repo-url> hierarchical_cp
cd hierarchical_cp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Defaults in `code/paths.py` write to `paper-results/` (no env vars required).

### Option A — use the shipped results (no re-run)

| Deliverable | Path |
|-------------|------|
| Simulation figures (Sec. 3.1, App. D.2) | `paper-results/dgp/figures/` |
| Simulation LaTeX tables | `paper-results/dgp/tables/` |
| Simulation raw CSVs | `paper-results/results/dgp/` |
| ACS figures / results (Sec. 3.2, App. D.5) | `paper-results/acs/` |
| ACS size-ignorability diagnostics | `paper-results/acs/diagnostics/` |
| App. D.3 size–intercept coupling | `paper-results/size_shift/` |
| App. D.4 merger-weight grid | `paper-results/effect_of_weights/` |

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

### Option B — re-run from scratch

```bash
# Main simulations (Sec. 3.1): several hours
.venv/bin/python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.1,0.15,0.2 \
  --gamma 5 \
  --configs fixedN21,poissonNmean25 \
  --total_replicates 1000 \
  --n_workers 8 \
  --quantile-mode deterministic

.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/export_paper_tables_mean_width.py
.venv/bin/python code/marginal/export_paper_tables_baselines.py

# ACS (Sec. 3.2): download PUMS once, then experiments
.venv/bin/python real_data/acs/download_acs_ca_pums.py
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot
.venv/bin/python code/marginal/recompute_acs_stdcp_randomized_min21.py --B 1000 --n_workers 7 --alphas 0.05,0.1,0.15,0.2
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

Umbrella (main DGP + ACS):

```bash
bash paper-results/logs/run_alg1_suite.sh
```

Appendix D.3–D.4:

```bash
bash paper-results/logs/run_appendix_sensitivity.sh
```

---

## Repository layout

```
hierarchical_cp/
├── README.md
├── requirements.txt
├── scores.py
├── methods/                  # Algorithm 1 (GHCP), HCP, μ-estimators
├── code/
│   ├── paths.py
│   ├── shared/dgp/           # hierarchical Gaussian DGP
│   ├── shared/plot_engine.py
│   └── marginal/             # runners + paper plotting
├── paper-results/            # ★ paper deliverable
├── real_data/acs/            # ACS download + cohort filters
└── old/                      # archived (not needed for paper figures)
```

---

## Paper experiments (what is implemented)

Shared GHCP implementation (`methods/donor_hcp.py`), matching Algorithm 1:

- Restricted donor pool \(S_\eta\) with **\(\eta=0.5\)** (code argument `alpha_selection=0.5`).
- Local training size \(\tau=\lfloor o/2\rfloor\); local predictor = mean of first \(\tau\) outcomes.
- Merger (paper (3)):
  \(\widetilde\mu_j=(1-\lambda_{\mathrm{local}})\widehat\mu^{\mathrm{global}}+\lambda_{\mathrm{local}}\overline Y_j\),
  \(\lambda_{\mathrm{local}}=\tau/(|S_{\mathrm{train}}|+\tau)\) when \(c=1\).
  (Code stores the global weight \(w_g=1-\lambda_{\mathrm{local}}\); figures use \(\lambda_{\mathrm{local}}\).)
- Score \(s=|Y-\widetilde\mu|\). GHCP/HCP quantiles **deterministic**; Std-CP **randomized**.
- RF global learner: 50 trees, min leaf 5, `random_state=123`.
- Replicates \(B=1000\), \(K=20\) reference groups.

### Simulations (Sec. 3.1)

\(d=5\), \(U_j\sim\mathrm{Unif}([1,5]^d)\), \(B_j\sim N(0,\gamma^2)\) with \(\gamma=5\),
\(\mu(U)=(U_1^2,\ldots,U_d^2)^\top\), \(\Sigma(U)=(1-\rho)\mathrm{diag}(U)+\rho\mathbf{1}\mathbf{1}^\top\), \(\rho=0.5\).
Prediction for a fixed later observation in the test stream (index 35 on Poisson / index 20 on fixed \(N=21\)); history uses the first \(o\) rows, \(o\in\{0,5,10,15,20\}\).

| Config | Group sizes | Role |
|--------|-------------|------|
| `fixedN21` | \(N_j\equiv 21\) | Sec. 3.1.1 equal-size |
| `poissonNmean25` | \(N_j\sim\mathrm{Poi}(25)\) (redraw \(N=0\)) | **main Poisson design** |

### ACS (Sec. 3.2)

2018 ACS 1-year California PUMS. Foreign-born, YOEP \(\ge 2000\), age 25–54, hours \(\ge 40\), PUMA size \(\ge 21\). Target is the individual at **position 21** (index 20); first \(o\) permuted records are the initial sample. Details: [`real_data/acs/README.md`](real_data/acs/README.md).

### Appendix sensitivities

| Appendix | Folder | Command |
|----------|--------|---------|
| D.3 size ignorability | `paper-results/size_shift/` | `run_size_shift_sensitivity.py` |
| D.4 merger weight | `paper-results/effect_of_weights/` | `run_effect_of_weights.py`, `run_effect_of_weights_ud_narrow.py` |

---

## Reproducibility

| Quantity | Simulations | ACS |
|----------|-------------|-----|
| Suite / DGP seed | `BASE_SEED=457` | `456` |
| Quantile base seed | `457` | `456` |
| RF `random_state` | `123` | `123` |
| Chunk \(i\) RNG | `np.random.seed(457 + 1000*i)` | per-replicate formulas in the ACS README |

| Config | Chunk size | Chunks for \(B=1000\) |
|--------|------------|------------------------|
| `fixedN21` | 125 | 8 |
| `poissonNmean25` | 25 | 40 |

`--n_workers` only changes parallelism, not chunk seeds.

---

## Code map (paper → file)

| Paper | Code |
|-------|------|
| Algorithm 1, \(S_\eta\), donated \(N_{J_0}\) | `methods/donor_hcp.py` |
| HCP / pooling / subsampling | `methods/baseline_hcp.py` |
| Merger (3), RF/OLS, Bayes \(E[Y\mid X,U]\) | `methods/mu_methods.py` |
| Scores / quantiles | `scores.py` |
| Sec. 3.1 runner | `code/marginal/run_true_marginal_latent_intercept_rf_experiments.py` |
| Sec. 3.2 runner | `code/marginal/run_acs_yoep_fb_min21_permute.py` |
| Figures / tables | `code/marginal/plot_paper.py`, `export_paper_tables_*.py` |

Archive: [`old/README.md`](old/README.md).
