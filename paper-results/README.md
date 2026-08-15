# Paper results (canonical deliverable)

Figures, tables, and summaries live **here**. Raw trial CSVs
(`*raw_results*.csv`, ACS `*_detailed.csv`) are gitignored under
`results/` and `acs/results/`; GitHub ships figures, summaries, and
manifests. Defaults in `code/paths.py` point at this directory. Notation
follows the paper (\(o\), \(\lambda_{\mathrm{local}}\), \(\eta\), \(\gamma\)).

Launchers named after paper sections (wrappers around the long runner names):

```bash
.venv/bin/python code/marginal/run_section_3_1.py --n_workers 8   # Sec. 3.1
.venv/bin/python code/marginal/plot_section_3_1.py               # Sec. 3.1 figures/tables
.venv/bin/python code/marginal/run_section_3_2.py --skip_stdcp --plot  # Sec. 3.2
```

Rebuild **figures/tables from stored CSVs** (no experiments):

```bash
.venv/bin/python code/marginal/plot_section_3_1.py
.venv/bin/python code/marginal/plot_paper.py --suite acs
.venv/bin/python code/marginal/plot_size_shift_sensitivity.py
.venv/bin/python code/marginal/plot_effect_of_weights.py --gamma 5
.venv/bin/python code/marginal/plot_effect_of_weights_ud_narrow.py
```

```
paper-results/
├── dgp/                       # Sec. 3.1 simulations (+ App. D.1–D.2 extras)
│   ├── figures/
│   ├── summaries/
│   ├── tables/
│   └── *_seeds_manifest.json
├── results/dgp/               # raw CSVs (B=1000; `*raw_results*.csv` gitignored)
│   ├── true_marg_latent_rf_gamma5p0_fixedN21_*
│   ├── true_marg_latent_rf_gamma5p0_poissonNmean25_*
│   ├── size_shift_*
│   └── effect_of_weights_*
├── acs/                       # Sec. 3.2 ACS (+ App. D.5)
├── size_shift/                # App. D.3
├── effect_of_weights/         # App. D.4
└── logs/                      # suite scripts
```

## GHCP as implemented (all suites)

Restricted GHCP, Algorithm 1, \(\eta=0.5\):

- \(\tau=\lfloor o/2\rfloor\), \(\lambda_{\mathrm{local}}=\tau/(|S_{\mathrm{train}}|+\tau)\) (paper (3), \(c=1\)).
- RF 50 trees / min leaf 5, absolute residual, \(K=20\), \(B=1000\).
- GHCP/HCP quantiles deterministic; Std-CP randomized + studentized (ACS).

## Main simulations (Sec. 3.1)

| Config | Group sizes | Chunk size | Target index |
|--------|-------------|------------|--------------|
| `fixedN21` | refs \(N_j\equiv 21\); test stream 36 | 125 | 35 |
| `poissonNmean25` | \(N_j\sim\mathrm{Poi}(25)\), redraw 0 | 25 | 35 |

Shared: \(\gamma=5\), \(U_j\sim\mathrm{Unif}([1,5]^5)\), \(\rho=0.5\), paper
panels \(o\in\{0,5,10,15,20\}\) (CSVs also store \(25,30,35\)),
\(\alpha\in\{0.05,0.1,0.15,0.2\}\), `BASE_SEED=457`. Both designs hold the
scored outcome at test-stream **index 35**. History uses the **first \(o\)**
rows of that stream.

```bash
.venv/bin/python code/marginal/run_section_3_1.py --n_workers 8
.venv/bin/python code/marginal/plot_section_3_1.py
```

(`run_section_3_1.py` ≡ `run_true_marginal_latent_intercept_rf_experiments.py`
with paper defaults; plots + `export_paper_tables_*.py`.)

Detail: [`dgp/README.md`](dgp/README.md).

### Section 3.1 artifact map

| Paper | Figure / table | Reproduce |
|-------|----------------|-----------|
| Sec. 3.1.1, \(N_j\equiv 21\), \(\alpha=0.1\) | `dgp/figures/fixedN21_alpha0p1_coverage_width_by_o.pdf`, `dgp/tables/` `tab:dhcp-hcp-alpha01` | `run_section_3_1.py` then `plot_section_3_1.py` |
| Sec. 3.1.1, \(N_j\equiv 21\), \(\alpha=0.05\) | `tab:dhcp-hcp-alpha005` | same |
| Sec. 3.1.1, Poi(25), \(\alpha=0.1\) | `dgp/figures/poisson_alpha0p1_coverage_width_by_o.pdf`, `tab:dhcp-hcp-poisson-alpha01` | same |
| Sec. 3.1.2 within-group training | `dgp/figures/fixedN21_3_alpha0p1_within_vs_no_within.pdf` | same |
| App. D.2 extra baselines | `dgp/figures/*_all_baselines_*.pdf`, `tab:dhcp-hcp-*-extra` | same |
| App. D.2 Std-CP overlay | `dgp/figures/*_with_stdcp.pdf` | same |

Folders `dgp/figures/merger_compare/`, `re_vs_std/`, and `oracleB_vs_re/` are
exploratory extras, **not** paper tables. They are the only place an
empirical-Bayes merger was ever run.

## ACS (Sec. 3.2)

2018 ACS 1-year CA PUMS; foreign-born; YOEP \(\ge 2000\); age 25–54; hours \(\ge 40\);
eligible PUMAs size \(\ge 21\). Target = **position 21** (index 20); permute rows
within PUMA each replicate; \(K=20\) reference PUMAs + 1 test PUMA;
\(\alpha\in\{0.05,0.1,0.15,0.2\}\).

Prep + filter checklist: [`../real_data/acs/README.md`](../real_data/acs/README.md).
Suite notes: [`acs/README.md`](acs/README.md).

```bash
.venv/bin/python real_data/acs/download_acs_ca_pums.py
.venv/bin/python code/marginal/run_section_3_2.py \
  --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot
.venv/bin/python code/marginal/recompute_acs_stdcp_randomized_min21.py \
  --B 1000 --n_workers 7 --alphas 0.05,0.1,0.15,0.2
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

## Appendix D.3–D.4

```bash
# D.3 size-shift (Poi(25), γ=5, ξ∈{−0.75, 0, 0.5, 0.75})
.venv/bin/python code/marginal/run_size_shift_sensitivity.py --B 1000 --n_workers 6
.venv/bin/python code/marginal/plot_size_shift_sensitivity.py

# D.4 weights, original DGP (γ=5)
.venv/bin/python code/marginal/run_effect_of_weights.py --B 1000 --n_workers 6 --gamma 5
.venv/bin/python code/marginal/plot_effect_of_weights.py --gamma 5

# D.4 weights, narrow U_d, γ=0, RF + Bayes E[Y|X,U]
.venv/bin/python code/marginal/run_effect_of_weights_ud_narrow.py --B 1000 --n_workers 6
.venv/bin/python code/marginal/plot_effect_of_weights_ud_narrow.py
```

Extra γ-grid (not in the paper appendix text): `--gamma 0.2` and `--gamma 0`.

Umbrella scripts: `logs/run_alg1_suite.sh` (main DGP+ACS),
`logs/run_appendix_sensitivity.sh` (D.3–D.4).
