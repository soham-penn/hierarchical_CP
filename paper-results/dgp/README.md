# Simulations (Sec. 3.1): true-marginal RF, \(\gamma=5\)

Parent: [`paper-results/`](../README.md).

| Role | Path |
|------|------|
| Figures | `figures/` |
| Summaries | `summaries/` |
| LaTeX tables | `tables/` |
| Seeds | `*_seeds_manifest.json`, `../results/dgp/*/run_manifest.json` |
| Raw CSVs | `../results/dgp/true_marg_latent_rf_gamma5p0_*` |

## Design (paper Sec. 3.1)

- \(K=20\), \(d=5\), \(U_j\stackrel{\mathrm{i.i.d.}}{\sim}\mathrm{Unif}([1,5]^d)\),
  \(B_j\stackrel{\mathrm{i.i.d.}}{\sim}N(0,\gamma^2)\) with \(\gamma=5\).
- \(Z_{j,i}\mid U_j,B_j\sim\mathcal N_d(\mu(U_j)+B_je_d,\Sigma(U_j))\),
  \(\mu(U)=(U_1^2,\ldots,U_d^2)^\top\),
  \(\Sigma(U)=(1-\rho)\mathrm{diag}(U)+\rho\mathbf{1}\mathbf{1}^\top\), \(\rho=0.5\).
- Restricted GHCP (Algorithm 1), **\(\eta=0.5\)**.
- Local mean on first \(\tau=\lfloor o/2\rfloor\) outcomes; merger (3) with \(c=1\).
- RF: 50 trees, min leaf 5; score \(|Y-\widetilde\mu|\).
- \(B=1000\); GHCP/HCP quantiles deterministic; Std-CP randomized.
- \(o\in\{0,5,10,15,20\}\).

| Config | \(N_j\) | Target index | Test-stream length |
|--------|---------|--------------|--------------------|
| `fixedN21` | \(21\) | 20 | 21 (history \(0{:}o-1\), predict index 20) |
| `poissonNmean25` | \(\mathrm{Poi}(25)\), redraw 0 | 35 | \(\ge 36\) (history \(0{:}o-1\), predict index 35) |

The Poisson target is held at a fixed later index so the same \(Y\) is scored for every \(o\).

## Paper section map

### GHCP vs HCP (Sec. 3.1.1)

| Setting | Figure | Table |
|---------|--------|-------|
| \(N_j\equiv 21\), \(\alpha=0.05\) (infinite-width cases) | — | `tables/` `tab:dhcp-hcp-alpha005` |
| \(N_j\equiv 21\), \(\alpha=0.1\) | `figures/fixedN21_alpha0p1_coverage_width_by_o.pdf` | `tab:dhcp-hcp-alpha01` |
| \(N_j\sim\mathrm{Poi}(25)\), \(\alpha=0.1\) | `figures/poisson_alpha0p1_coverage_width_by_o.pdf` | `tab:dhcp-hcp-poisson-alpha01` |

### Within-group training (Sec. 3.1.2)

Fixed \(N_j=21\), \(\alpha=0.1\): GHCP with vs without the local mean in (3).

| Artifact | File |
|----------|------|
| Figure | `figures/fixedN21_3_alpha0p1_within_vs_no_within.pdf` |

### Extra (App. D.2)

Baseline panels (pooling, subsampling, …) and Std-CP overlays are also written by
`plot_paper.py --suite dgp_rf`. App. D.1 Bayes-vs-RF main table is a separate
suite (not this RF-only folder). App. D.3–D.4 are `../size_shift/` and
`../effect_of_weights/`.

## Exact reproduction

| Config | `BASE_SEED` | Chunk size | Chunks | Chunk seed |
|--------|-------------|------------|--------|------------|
| `fixedN21` | 457 | 125 | 8 | `457 + 1000*i` |
| `poissonNmean25` | 457 | 25 | 40 | `457 + 1000*i` |

```bash
.venv/bin/python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.1,0.15,0.2 --gamma 5 \
  --configs fixedN21,poissonNmean25 \
  --total_replicates 1000 --n_workers 8 --quantile-mode deterministic

.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/export_paper_tables_mean_width.py
.venv/bin/python code/marginal/export_paper_tables_baselines.py
```

Plots only (shipped CSVs): `.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf`

Runners: `run_true_marginal_latent_intercept_rf_experiments.py`,
`run_true_marginal_latent_intercept_experiments.py`.
