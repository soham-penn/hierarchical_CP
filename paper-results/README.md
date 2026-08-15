# Paper results (canonical deliverable)

Figures, tables, summaries, and the raw trial CSVs used to build them live **here**.
Defaults in `code/paths.py` point at this directory. Notation follows the paper
(\(o\), \(\lambda_{\mathrm{local}}\), \(\eta\), \(\gamma\)).

```
paper-results/
├── dgp/                       # Sec. 3.1 simulations (+ App. D.1–D.2 extras)
│   ├── figures/
│   ├── summaries/
│   ├── tables/
│   └── *_seeds_manifest.json
├── results/dgp/               # raw CSVs (B=1000)
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
| `fixedN21` | \(N_j\equiv 21\) | 125 | 20 |
| `poissonNmean25` | \(N_j\sim\mathrm{Poi}(25)\), redraw 0 | 25 | 35 |

Shared: \(\gamma=5\), \(U_j\sim\mathrm{Unif}([1,5]^5)\), \(\rho=0.5\),
\(o\in\{0,5,10,15,20\}\), \(\alpha\in\{0.05,0.1,0.15,0.2\}\), `BASE_SEED=457`.

History always uses the **first \(o\)** observations of the test stream; the
prediction target is a later index (20 or 35) held fixed across \(o\).

```bash
.venv/bin/python code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas 0.05,0.1,0.15,0.2 --gamma 5 \
  --configs fixedN21,poissonNmean25 \
  --total_replicates 1000 --n_workers 8 --quantile-mode deterministic

.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
.venv/bin/python code/marginal/export_paper_tables_mean_width.py
.venv/bin/python code/marginal/export_paper_tables_baselines.py
```

Detail: [`dgp/README.md`](dgp/README.md).

## ACS (Sec. 3.2)

2018 ACS 1-year CA PUMS; foreign-born; YOEP \(\ge 2000\); age 25–54; hours \(\ge 40\);
eligible PUMAs size \(\ge 21\). Target = **position 21** (index 20); permute rows
within PUMA each replicate; \(K=20\) reference PUMAs + 1 test PUMA;
\(\alpha\in\{0.05,0.1,0.15,0.2\}\).

Prep + filter checklist: [`../real_data/acs/README.md`](../real_data/acs/README.md).
Suite notes: [`acs/README.md`](acs/README.md).

```bash
.venv/bin/python real_data/acs/download_acs_ca_pums.py
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py \
  --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot
.venv/bin/python code/marginal/recompute_acs_stdcp_randomized_min21.py \
  --B 1000 --n_workers 7 --alphas 0.05,0.1,0.15,0.2
.venv/bin/python code/marginal/plot_paper.py --suite acs
```

## Appendix D.3–D.4

```bash
# D.3 size-shift (Poi(25), γ=5, ξ∈{0,0.25,0.50,0.75})
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
