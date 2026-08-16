# Generalized Hierarchical Conformal Prediction (GHCP)

GHCP builds finite-sample, distribution-free prediction sets for grouped data
when a few observations from the **test group** are already available. Standard
hierarchical conformal prediction (HCP) covers a new group but does not use
those initial records. GHCP restores the needed symmetry by assigning the test
group a randomly donated reference-group size, then uses the initial sample
both to fit a local mean and to calibrate.

This repository is the implementation behind the paper figures and tables
(Sections 3.1–3.2 and Appendix D). Outputs live in
[`paper-results/`](paper-results/README.md).

**Implemented method.** Restricted-donor GHCP (Algorithm 1) with
$\eta=0.5$, local training size $\tau=\lfloor o/2\rfloor$, and merger
$\lambda_{\mathrm{local}}=\tau/(\lvert S_{\mathrm{train}}\rvert+\tau)$
(paper (3) with $c=1$). The score is $\lvert Y-\widetilde\mu\rvert$. The
global predictor is a random forest (50 trees, min leaf 5,
`random_state=123`). GHCP and HCP quantiles are deterministic; ACS Std-CP is
studentized and randomized. Paper runs use $B=1000$ replicates and $K=20$
reference groups. Notation: $o$ is the initial test-group sample size;
$\eta$ is the donor restriction.

| Paper | Launcher |
|-------|----------|
| Sec. 3.1 simulations | `code/marginal/run_section_3_1.py` |
| Sec. 3.1 figures / tables | `code/marginal/plot_section_3_1.py` |
| Sec. 3.2 ACS | `code/marginal/run_section_3_2.py` |
| Sec. 3.2 figures | `code/marginal/plot_paper.py --suite acs` |
| App. D.3 size–intercept coupling | `code/marginal/run_size_shift_sensitivity.py` |
| App. D.4 merger-weight grid | `code/marginal/run_effect_of_weights.py` |
| App. D.4, narrow $U_d$ | `code/marginal/run_effect_of_weights_ud_narrow.py` |

File map: [`paper-results/README.md`](paper-results/README.md).
ACS filters and seeds: [`real_data/acs/README.md`](real_data/acs/README.md).

---

## Install

```bash
git clone https://github.com/soham-penn/hierarchical_CP.git hierarchical_cp
cd hierarchical_cp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Defaults write to `paper-results/` (`code/paths.py`).

### Section 3.1 (simulations)

```bash
.venv/bin/python code/marginal/run_section_3_1.py --n_workers 8
.venv/bin/python code/marginal/plot_section_3_1.py
```

Defaults: $\gamma=5$, `fixedN21` and `poissonNmean25`, $B=1000$,
deterministic quantiles. Shipped figures and tables are already in
`paper-results/dgp/`. Raw trial CSVs (`*raw_results*.csv`) are gitignored.

### Section 3.2 (ACS)

```bash
.venv/bin/python real_data/acs/download_acs_ca_pums.py
.venv/bin/python code/marginal/run_section_3_2.py --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot
.venv/bin/python code/marginal/recompute_acs_stdcp_randomized_min21.py --B 1000 --n_workers 7 --alphas 0.05,0.1,0.15,0.2
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

Each replicate draws a uniform permutation of records within each selected
PUMA (exchangeable within group). The target is the individual at position 21.

### Appendix D.3–D.4

```bash
bash paper-results/logs/run_appendix_sensitivity.sh
```

Sec. 3.1 + 3.2 together: `bash paper-results/logs/run_alg1_suite.sh`.

---

## Designs

### Simulations (Sec. 3.1)

$d=5$, $U_j\sim\mathrm{Unif}([1,5]^d)$, $B_j\sim N(0,\gamma^2)$ with
$\gamma=5$, $\rho=0.5$. History is the first $o$ observations of the test
stream. The scored outcome is held at test-stream index 35 for every
$o\in\{0,5,10,15,20\}$.

| Config | Group sizes | Target index |
|--------|-------------|--------------|
| `fixedN21` | $N_j\equiv 21$; test stream length 36 | 35 |
| `poissonNmean25` | $N_j\sim\mathrm{Poi}(25)$ (redraw $N=0$) | 35 |

CSVs also store $o\in\{25,30,35\}$.

### ACS (Sec. 3.2)

2018 ACS 1-year California PUMS. Foreign-born, year of entry $\ge 2000$,
age 25–54, usual hours $\ge 40$, PUMA size $\ge 21$. Outcome: annual
income (dollars). Details:
[`real_data/acs/README.md`](real_data/acs/README.md).

---

## Layout

```
hierarchical_cp/
├── methods/          # GHCP (Algorithm 1), HCP, S-HCP, μ-estimators
├── scores.py
├── code/marginal/    # paper runners and plotting
├── code/shared/dgp/  # hierarchical Gaussian DGP
├── paper-results/    # figures, tables, summaries
├── real_data/acs/    # ACS download and cohort filters
└── old/              # archived prototypes
```

| Role | Path |
|------|------|
| GHCP | `methods/donor_hcp.py` |
| HCP / pooling / subsampling | `methods/baseline_hcp.py` |
| S-HCP | `methods/sample_hcp.py` |
| Merger and RF / OLS | `methods/mu_methods.py` |
| DGP | `code/shared/dgp/dgp_specification.py` |

Archived material: [`old/README.md`](old/README.md).

---

## Seeds

| Quantity | Simulations | ACS |
|----------|-------------|-----|
| Suite seed | `BASE_SEED=457` | `456` |
| Quantile base seed | `457` | `456` |
| RF `random_state` | `123` | `123` |
| Chunk $i$ RNG | `np.random.seed(457 + 1000*i)` | see the ACS README |

| Config | Chunk size | Chunks for $B=1000$ |
|--------|------------|----------------------|
| `fixedN21` | 125 | 8 |
| `poissonNmean25` | 25 | 40 |

`--n_workers` only sets concurrency. Chunk count (and chunk seeds) is
$B$ divided by chunk size; use `--n_workers 8` for Section 3.1.

Appendix D.3 default $\xi\in\{-0.75,0,0.5,0.75\}$.
