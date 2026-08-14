# Simulations (DGP): true-marginal RF, \(\gamma = 5\)

Canonical **Simulations** deliverable for the paper. Parent folder:
[`paper-results/`](../README.md).

| Role | Path |
|------|------|
| Figures | `figures/` |
| Trial / summary CSVs | `summaries/` |
| LaTeX tables + section body | `tables/` |
| Seed / run manifests | `*_seeds_manifest.json`, `../results_marginal/dgp/*/run_manifest.json` |
| Raw trial CSVs | `../results_marginal/dgp/true_marg_latent_rf_gamma5p0_*` |

## Design

- **DGP:** latent intercept \(\gamma=5\) on the joint-Gaussian hierarchical model (\(K=20\), \(d=5\), \(\rho=0.5\)).
- **Global \(\mu\):** random forest (50 trees, min leaf 5), merger \(c=1\).
- **Score:** absolute residual.
- **Methods:** restricted GHCP (\(\eta=0.5\)) with WGT, GHCP without WGT, baseline HCP (+ baselines / Std-CP in extra figures).
- **Replicates:** \(B=1000\); deterministic conformal quantile (Std-CP randomized).
- **Configs:**
  - `fixedN21` — \(N_j \equiv 21\)
  - `poissonNmean25` — \(N_j \sim \mathrm{Poi}(25)\) (reject \(N=0\))

Legacy `poissonNmean21` (\(N_j \sim 1+\mathrm{Poi}(20)\)) is under
`../results_marginal/dgp/_archive_poissonNmean21/`.

## Paper section map

### 1. Benefit of GHCP vs HCP

**(i) Non-triviality when baselines are infinite** — fixed \(N_j=21\), \(\alpha=0.05\)  
Table `tab:dhcp-hcp-alpha005` in `tables/`.

**(ii) Width gains when intervals are finite** — fixed \(N_j=21\), \(\alpha=0.1\)

| Artifact | File |
|----------|------|
| Figure | `figures/fixedN21_alpha0p1_coverage_width_by_o.pdf` |
| Table | `tables/` (`tab:dhcp-hcp-alpha01`) |

**(iii) Randomized donor sizes** — \(N_j \sim \mathrm{Poi}(25)\), \(\alpha=0.1\)

| Artifact | File |
|----------|------|
| Figure | `figures/poisson_alpha0p1_coverage_width_by_o.pdf` |
| Table | `tables/` (`tab:dhcp-hcp-poisson-alpha01`) |

### 2. Within-group training

Fixed \(N_j=21\), \(\alpha=0.1\); GHCP with vs without WGT.

| Artifact | File |
|----------|------|
| Figure | `figures/fixedN21_3_alpha0p1_within_vs_no_within.pdf` |
| Table | `tables/` (`tab:sim-with-vs-without-training`) |

Width entries are **means** (`width_mean`); SEs are `width_se`.

## Exact reproduction

### Seeds and chunks

| Config | `BASE_SEED` | Chunk size | Chunks (\(B=1000\)) | Chunk seed |
|--------|-------------|------------|---------------------|------------|
| `fixedN21` | 457 | 125 | 8 | `457 + 1000*i` |
| `poissonNmean25` | 457 | 25 | 40 | `457 + 1000*i` |

RF fits use `random_state=123`. Quantile seeds use `quantile_base_seed=457`.

`--n_workers` only changes parallelism; chunk IDs (and therefore seeds) are
fixed by chunk size. The runner’s per-config defaults match the table above.

### Commands

From the repo root (venv active). Defaults already write to `paper-results/`.

```bash
# Experiments (several hours)
.venv/bin/python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.1,0.15,0.2 \
  --gamma 5 \
  --configs fixedN21,poissonNmean25 \
  --total_replicates 1000 \
  --n_workers 8 \
  --quantile-mode deterministic

# Figures + summaries + tables
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/export_paper_tables_mean_width.py
.venv/bin/python code/marginal/export_paper_tables_baselines.py
```

Plots-only (shipped raw CSVs already present):

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
```

Active code: `code/marginal/run_true_marginal_latent_intercept_rf_experiments.py`,
`run_true_marginal_latent_intercept_experiments.py`, `code/shared/dgp/`,
`code/shared/plot_engine.py`, `methods/`, `scores.py`.

## Extra figures

Regenerating `dgp_rf` also writes baseline panels, Std-CP overlays, \(\alpha=0.05\)
panels, multi-\(\alpha\) boxplots, etc. The paper Simulations subsection uses the
three figures above plus the tables in `tables/`.
