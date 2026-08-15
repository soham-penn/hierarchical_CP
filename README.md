# Hierarchical conformal prediction (GHCP)

This repository implements **GHCP**, Algorithm 1 of the paper (restricted donor
hierarchical conformal prediction), together with HCP and the conformal
baselines used in Sections 3.1–3.2. Paper figures, tables, and trial CSVs are
under [`paper-results/`](paper-results/README.md).

The paper name is **GHCP**, not “HCP++”. There is **no** blood-pressure /
clinical-trial study in the paper (an early prototype lives only under
`old/`). ACS evaluation **does not** order individuals by income or score
percentiles; each replicate draws a uniform permutation of records within each
PUMA, which preserves within-group exchangeability. See
[`real_data/acs/README.md`](real_data/acs/README.md).

Notation: initial test-group sample size \(o\), merger weight
\(\lambda_{\mathrm{local}}\) in paper (3), restriction \(\eta\), donated size
\(N_{J_0}\).

---

## Which code produced the paper

Every shipped table and figure uses the same GHCP implementation:

| Setting | Paper |
|---------|--------|
| Method | Algorithm 1, restricted donor pool \(S_\eta\), \(\eta=0.5\) (`alpha_selection=0.5`) |
| Local sample | \(\tau=\lfloor o/2\rfloor\); local predictor = mean of the first \(\tau\) outcomes |
| Merger | paper **(3)** with **\(c=1\)**: \(\lambda_{\mathrm{local}}=\tau/(\lvert S_{\mathrm{train}}\rvert+\tau)\) |
| Score | \(\lvert Y-\widetilde\mu\rvert\) |
| Global \(\mu\) | random forest, 50 trees, min leaf 5, `random_state=123` |
| GHCP / HCP quantile | **deterministic** |
| ACS Std-CP | studentized, **randomized** |
| Replicates | \(B=1000\), \(K=20\) reference groups |

Code stores the global weight \(w_g=1-\lambda_{\mathrm{local}}\); figures and
captions use \(\lambda_{\mathrm{local}}\) only.

**Not used for any paper table or figure** (present in `methods/` for
exploratory extras only):

- merger exponent \(c\neq 1\)
- empirical-Bayes / random-effects weight \(w_g=(\hat\sigma^2/\tau)/(\hat\tau_B^2+\hat\sigma^2/\tau)\) (depends on other groups’ residuals; not covered by Appendix B.1)
- ridge residual correction with clip \(0.5\)
- local-RF offset in place of the group mean
- blood-pressure data, 50-state ACS, per-state top-25% income filters

Empirical-Bayes is gated on `mu_method["merger"] in {"bayes_re","bayes","re"}`.
Section 3.1/3.2 runners never set that flag.

---

## Reproduce a paper section

Install once, then use the section launchers. Defaults write to
`paper-results/` (`code/paths.py`); no environment variables are required.

```bash
git clone <repo-url> hierarchical_cp
cd hierarchical_cp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Paper | Launcher | What it does |
|-------|----------|----------------|
| **Sec. 3.1** simulations | `code/marginal/run_section_3_1.py` | \(\gamma=5\) DGP, \(N_j\equiv 21\) and \(N_j\sim\mathrm{Poi}(25)\), all \(\alpha\) |
| Sec. 3.1 figures/tables | `code/marginal/plot_section_3_1.py` | figures + LaTeX from shipped or newly written CSVs |
| **Sec. 3.2** ACS | `code/marginal/run_section_3_2.py` | CA PUMS income, uniform within-PUMA permutation |
| Sec. 3.2 figures | `code/marginal/plot_paper.py --suite acs` | ACS coverage/width panels |
| App. D.3 | `code/marginal/run_size_shift_sensitivity.py` | size–intercept coupling \(\xi\) |
| App. D.4 | `code/marginal/run_effect_of_weights.py` | \(\lambda_{\mathrm{local}}\) grid, original DGP \(\gamma=5\) |
| App. D.4 (\(U_d\)-narrow) | `code/marginal/run_effect_of_weights_ud_narrow.py` | \(\gamma=0\), RF + Bayes \(E[Y\mid X,U]\) |

Full table/figure → file map: [`paper-results/README.md`](paper-results/README.md).

### Section 3.1 (simulations)

```bash
.venv/bin/python code/marginal/run_section_3_1.py --n_workers 8
.venv/bin/python code/marginal/plot_section_3_1.py
```

`run_section_3_1.py` is the public name for
`run_true_marginal_latent_intercept_rf_experiments.py` (same flags: `--alphas`,
`--gamma 5`, `--configs fixedN21,poissonNmean25`, `--total_replicates 1000`,
`--quantile-mode deterministic`). Merger \(c=1\).

Shipped outputs (no re-run): `paper-results/dgp/figures/`,
`paper-results/dgp/tables/`. Raw trial CSVs (`*raw_results*.csv`) are
gitignored; replotting from trials needs a local copy or a re-run.

### Section 3.2 (ACS)

```bash
.venv/bin/python real_data/acs/download_acs_ca_pums.py   # gitignored extract
.venv/bin/python code/marginal/run_section_3_2.py --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot
.venv/bin/python code/marginal/recompute_acs_stdcp_randomized_min21.py --B 1000 --n_workers 7 --alphas 0.05,0.1,0.15,0.2
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

`run_section_3_2.py` is the public name for
`run_acs_yoep_fb_min21_permute.py`. Within-group mode is **mean** (\(c=1\)).
Rows are **randomly permuted** within each PUMA each replicate
(`permute_rows=True`); they are not sorted by income.

### Appendix D.3–D.4

```bash
bash paper-results/logs/run_appendix_sensitivity.sh
```

Umbrella for Sec. 3.1 + 3.2: `bash paper-results/logs/run_alg1_suite.sh`.

---

## Designs (paper)

### Simulations (Sec. 3.1)

\(d=5\), \(U_j\sim\mathrm{Unif}([1,5]^d)\), \(B_j\sim N(0,\gamma^2)\) with
\(\gamma=5\), \(\rho=0.5\). History is the **first \(o\)** observations of the
test stream; the target is a later index held fixed across \(o\in\{0,5,10,15,20\}\).

| Config | Group sizes | Target index | Role |
|--------|-------------|--------------|------|
| `fixedN21` | refs \(N_j\equiv 21\); test stream 36 | 35 | Sec. 3.1.1 equal-size refs |
| `poissonNmean25` | \(N_j\sim\mathrm{Poi}(25)\) (redraw \(N=0\)) | 35 | main Poisson design |

Both designs hold the scored \(Y\) at test-stream **index 35**. Paper panels use
\(o\in\{0,5,10,15,20\}\); CSVs also store \(o\in\{25,30,35\}\).

### ACS (Sec. 3.2)

2018 ACS 1-year California PUMS. Foreign-born, YOEP \(\ge 2000\), age 25–54,
hours \(\ge 40\), PUMA size \(\ge 21\). Target: individual at **position 21**
(index 20). Details: [`real_data/acs/README.md`](real_data/acs/README.md).

---

## Repository layout

```
hierarchical_cp/
├── README.md
├── requirements.txt
├── scores.py
├── methods/                  # Algorithm 1 (GHCP), HCP, S-HCP, μ-estimators
├── code/
│   ├── paths.py
│   ├── shared/dgp/           # hierarchical Gaussian DGP
│   ├── shared/plot_engine.py
│   └── marginal/             # paper runners + plotting
├── paper-results/            # paper deliverable
├── real_data/acs/            # ACS download + cohort filters
└── old/                      # archive (not needed for paper figures)
```

File names that **do not exist** (older prototypes): `run_experiments.py`,
`methods/hcp_plus.py`, `methods/hcp_sample.py`, `DGP/dgp_specification.py`.
Current locations:

| Role | Path |
|------|------|
| Algorithm 1 | `methods/donor_hcp.py` (`compute_donor_hcp_randomized_interval`) |
| HCP / pooling / subsampling | `methods/baseline_hcp.py` |
| S-HCP (sample conformal) | `methods/sample_hcp.py` |
| Merger (3), RF/OLS | `methods/mu_methods.py` |
| DGP spec | `code/shared/dgp/dgp_specification.py` |
| Scores / quantiles | `scores.py` |

Archive: [`old/README.md`](old/README.md).

---

## Reproducibility seeds

| Quantity | Simulations | ACS |
|----------|-------------|-----|
| Suite seed | `BASE_SEED=457` | `456` |
| Quantile base seed | `457` | `456` |
| RF `random_state` | `123` | `123` |
| Chunk \(i\) RNG | `np.random.seed(457 + 1000*i)` | formulas in the ACS README |

| Config | Chunk size | Chunks for \(B=1000\) |
|--------|------------|------------------------|
| `fixedN21` | 125 | 8 |
| `poissonNmean25` | 25 | 40 |

`--n_workers` is concurrency only: chunk count (and therefore chunk seeds)
comes from \(B\) / chunk size, not from the worker count. Use `--n_workers 8`
for Section 3.1. Raw trial CSVs (`*raw_results*.csv`) are gitignored; GitHub
ships figures, summaries, and manifests. Replotting from trials requires a
local CSV or a re-run.

Appendix D.3 default \(\xi\in\{-0.75,0,0.5,0.75\}\).
