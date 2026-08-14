# Hierarchical Conformal Prediction

Python implementation of **GHCP** (donor hierarchical conformal prediction), sample-HCP, and baseline conformal methods on simulated and real grouped data.

**Paper outputs live in [`paper-results/`](paper-results/README.md).** That folder is the single place to look for figures, tables, summaries, and the raw trial CSVs used to build them.

---

## Quick start (download → reproduce)

```bash
git clone <repo-url> hierarchical_cp
cd hierarchical_cp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Defaults in `code/paths.py` already point at `paper-results/` (no env vars required). Optional overrides:

```bash
export HCP_PLOTS_MARGINAL="$PWD/paper-results"
export HCP_RESULTS_MARGINAL="$PWD/paper-results/results_marginal"
```

### Option A — use the shipped results (no re-run)

Paper figures / tables / summaries are already under:

| Deliverable | Path |
|-------------|------|
| Simulation figures | `paper-results/dgp_true_marginal_rf/figures/` |
| Simulation summaries | `paper-results/dgp_true_marginal_rf/summaries/` |
| Simulation LaTeX tables | `paper-results/dgp_true_marginal_rf/tables/` |
| Simulation raw CSVs | `paper-results/results_marginal/dgp/` |
| ACS figures / results | `paper-results/acs/min21/` |
| ACS size-ignorability diagnostics | `paper-results/acs/min21/diagnostics/` |

Regenerate figures/tables from the shipped CSVs only:

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

### Option B — re-run the full paper simulation suite (exact CSVs)

Seeds and chunk layout are part of the experiment identity (see [Reproducibility](#reproducibility) below).

```bash
# Several hours on a laptop. Writes under paper-results/.
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
```

Or one script for DGP + ACS:

```bash
bash paper-results/logs/run_alg1_suite.sh
```

---

## Repository layout

```
hierarchical_cp/
├── README.md                 # this file
├── requirements.txt
├── scores.py
├── methods/                  # GHCP / HCP / μ-estimators
├── code/
│   ├── paths.py              # defaults → paper-results/
│   ├── shared/dgp/           # simulation core
│   ├── shared/plot_engine.py
│   └── marginal/             # runners + paper plotting
├── paper-results/            # ★ PAPER DELIVERABLE (look here)
│   ├── README.md
│   ├── dgp_true_marginal_rf/ # figures, summaries, tables, seed manifests
│   ├── results_marginal/dgp/ # raw trial CSVs + run_manifest.json
│   ├── acs/min21/            # ACS paper suite
│   └── logs/                 # suite scripts
├── real_data/acs/            # ACS preprocessing helpers
├── results_marginal/         # legacy top-level raw dir (unused by defaults)
└── old/                      # archived code + superseded plot trees
    ├── plots_marginal/       # pre–paper-results plot tree
    └── new_poisson_marginal_scratch/
```

Detail maps:
- Simulations: [`paper-results/dgp_true_marginal_rf/README.md`](paper-results/dgp_true_marginal_rf/README.md)
- Paper-results overview: [`paper-results/README.md`](paper-results/README.md)
- ACS data prep: [`real_data/acs/README.md`](real_data/acs/README.md)

---

## Simulation designs (paper)

True-marginal latent intercept \(\gamma=5\), RF global \(\mu\) (50 trees, min leaf 5), absolute residual score, \(B=1000\), \(K=20\).

| Config | Group sizes | Role |
|--------|-------------|------|
| `fixedN21` | \(N_j \equiv 21\) | equal sizes |
| `poissonNmean25` | \(N_j \sim \mathrm{Poi}(25)\) (reject \(N=0\)) | **main Poisson design** |

Legacy \(N_j \sim 1+\mathrm{Poi}(20)\) (`poissonNmean21`) is archived under
`paper-results/results_marginal/dgp/_archive_poissonNmean21/`.

Merger weights use \(c=1\) (Eq. (4)). GHCP/HCP conformal quantiles are **deterministic**; Std-CP uses **randomized** split conformal.

---

## Reproducibility

### Seeds

| Quantity | Value |
|----------|-------|
| `BASE_SEED` (DGP RNG) | `457` |
| Quantile base seed | `457` |
| RF `random_state` | `123` |

Each RNG **chunk** \(i\) is seeded with:

```text
np.random.seed(BASE_SEED + 1000 * i)
```

Chunk size is part of the identity of the shipped CSVs:

| Config | Chunk size | Chunks for \(B=1000\) |
|--------|------------|------------------------|
| `fixedN21` | 125 | 8 |
| `poissonNmean25` | 25 | 40 |

The RF runner applies these defaults automatically when you pass
`--configs fixedN21,poissonNmean25` (override with `--chunk-size` only if you
intentionally want a different experiment). Parallelism (`--n_workers`) does
**not** change chunk seeds — only which chunks run concurrently.

### Manifests

After each run the runner writes `run_manifest.json` next to the complete CSV.
Shipped manifests:

- `paper-results/dgp_true_marginal_rf/fixedN21_seeds_manifest.json`
- `paper-results/dgp_true_marginal_rf/poissonNmean25_seeds_manifest.json`
- `paper-results/results_marginal/dgp/true_marg_latent_rf_gamma5p0_*_multialpha/run_manifest.json`

### What changed vs older trees

| Old path | New path |
|----------|----------|
| `plots_marginal/` | `old/plots_marginal/` (archived) |
| `new_plots_marginal/` | **`paper-results/`** (renamed; canonical) |
| `new_poisson_marginal/` | `old/new_poisson_marginal_scratch/` (staging; results merged into `paper-results`) |
| Poisson design `1+Poi(20)` | **`Poi(25)`** (`poissonNmean25`) |

Code defaults (`code/paths.py`) now resolve to `paper-results/` without setting env vars.

---

## ACS (paper)

**2018 ACS 1-year California PUMS** (2010 PUMA vintage), foreign-born, age 25–54, hours ≥ 40, YOEP ≥ 2000, PUMA size ≥ 21, target index 20. Full preprocessing, variable recodings, seeds, and RF protocol: [`real_data/acs/README.md`](real_data/acs/README.md).

```bash
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py --B 1000 --n_workers 8 --plot
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

Artifacts: `paper-results/acs/min21/`.

---

## Methods (code map)

| Label | Module |
|-------|--------|
| GHCP (donor-HCP) | `methods/donor_hcp.py` |
| Sample-HCP | `methods/sample_hcp.py` |
| HCP / pooling / sub / rep | `methods/baseline_hcp.py` |
| μ OLS / RF | `methods/mu_methods.py` |
| Scores / quantiles | `scores.py` |
| DGP runners | `code/marginal/run_true_marginal_latent_intercept_rf_experiments.py` |
| Plot / table export | `code/marginal/plot_paper.py`, `export_paper_tables_*.py` |

---

## Archive (`old/`)

Superseded runners, capture tooling, conditional-calibration experiments, and former plot trees (`old/plots_marginal/`, etc.). See [`old/README.md`](old/README.md).
