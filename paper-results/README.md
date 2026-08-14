# Paper results (canonical deliverable)

All paper figures, tables, summaries, and the raw trial CSVs used to build them
live **here**. Defaults in `code/paths.py` point at this directory.

```
paper-results/
├── README.md                      # this file
├── dgp_true_marginal_rf/          # Simulations deliverable
│   ├── figures/                   # PDFs cited in the paper
│   ├── summaries/                 # trial / method summaries
│   ├── tables/                    # LaTeX tables + section body
│   ├── fixedN21_seeds_manifest.json
│   ├── poissonNmean25_seeds_manifest.json
│   └── README.md                  # section map + exact re-run commands
├── results_marginal/
│   └── dgp/                       # raw CSVs (B=1000 × α × o)
│       ├── true_marg_latent_rf_gamma5p0_fixedN21_*
│       ├── true_marg_latent_rf_gamma5p0_poissonNmean25_*
│       └── _archive_poissonNmean21/   # legacy 1+Poi(20)
├── acs/min21/                     # ACS paper suite
└── logs/
    ├── run_alg1_suite.sh          # full DGP + ACS re-run
    └── run_alg1_resume.sh
```

## What this suite is

Algorithm-1–aligned GHCP with:

1. **Strain:** \(\mathrm{Strain}=[K]\setminus S_\eta\) — donor rows never enter global training.
2. **Merger:** Eq. (4) with \(c=1\): \(w_g=|\mathrm{Strain}|/(|\mathrm{Strain}|+\tau)\), \(\tau=\lfloor o/2\rfloor\).
3. **\(S_\eta\) cardinality:** \(|S_\eta|=q_\eta+1\) so after donor removal
   \(|\mathrm{Scal}|=q_\eta=\lceil(1-\eta)K\rceil\) (same calib size as HCP).

## Simulation designs

| Config | Group sizes | Chunk size (RNG) | Notes |
|--------|-------------|------------------|-------|
| `fixedN21` | \(N_j\equiv 21\) | 125 | equal sizes |
| `poissonNmean25` | \(N_j\sim\mathrm{Poi}(25)\) (reject 0) | 25 | **main Poisson** |

Shared: \(\gamma=5\), RF (50 trees, leaf 5), \(B=1000\), \(\alpha\in\{0.05,0.1,0.15,0.2\}\),
`BASE_SEED=457`, chunk \(i\) → `np.random.seed(457 + 1000*i)`, RF `random_state=123`,
deterministic GHCP/HCP quantiles (Std-CP randomized).

## Reproduce from a clean clone

Env vars are optional (defaults already target `paper-results/`):

```bash
cd /path/to/hierarchical_cp
source .venv/bin/activate   # after pip install -r requirements.txt

# 1) Re-run simulations (exact shipped CSVs; several hours)
.venv/bin/python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.1,0.15,0.2 \
  --gamma 5 \
  --configs fixedN21,poissonNmean25 \
  --total_replicates 1000 \
  --n_workers 8 \
  --quantile-mode deterministic

# 2) Figures + summaries + mean-width / baseline tables
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/export_paper_tables_mean_width.py
.venv/bin/python code/marginal/export_paper_tables_baselines.py

# 3) ACS (optional; separate wall time)
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py --B 1000 --n_workers 8 --plot
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

Umbrella script (DGP + ACS):

```bash
bash paper-results/logs/run_alg1_suite.sh
```

**Plots only** (use shipped raw CSVs; no experiment re-run):

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

## Where to look first

| Question | Open |
|----------|------|
| Simulation figure cited in the paper | `dgp_true_marginal_rf/figures/` |
| Mean-width / coverage tables | `dgp_true_marginal_rf/tables/` |
| Per-(α, o, method) summaries | `dgp_true_marginal_rf/summaries/` |
| Exact seeds / chunk layout | `*_seeds_manifest.json` and `results_marginal/dgp/*/run_manifest.json` |
| ACS coverage/width plots | `acs/min21/figures/` |
| How to re-run DGP exactly | `dgp_true_marginal_rf/README.md` |

## History of this folder

| Former name | Status |
|-------------|--------|
| `new_plots_marginal/` | Renamed to **`paper-results/`** |
| `plots_marginal/` | Moved to `old/plots_marginal/` |
| `new_poisson_marginal/` | Scratch Poi(25) staging; moved to `old/new_poisson_marginal_scratch/` (contents merged here) |
