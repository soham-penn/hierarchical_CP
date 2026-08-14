# ACS real-data experiments (paper)

California ACS PUMS income prediction with donor-HCP (GHCP), sample-HCP, and conformal baselines.

**Interpretation:** This is an **empirical illustration** under the paper's working
model, not evidence that the theoretical size-ignorability assumption holds in
ACS. See `min21/diagnostics/` for PUMA-size vs group-summary checks.

Cohort filters, variable recodings, and the full reproducibility checklist:
**[`real_data/acs/README.md`](../../real_data/acs/README.md)** (ACS year, PUMS type, PUMA vintage,
weights, income handling, RF settings, seeds, permutation protocol).

## Active suite: `min21/`

| Setting | Value |
|---------|--------|
| Cohort | CA, foreign-born, YOEP ≥ 2000, age 25–54, usual hours ≥ 40 |
| Individuals | 12,285 |
| PUMAs | 265 total; 212 eligible (size ≥ 21) |
| Target index | 20 (person at position 21 within PUMA) |
| o values | 0, 5, 10, 15, 20 |
| Row order | Permuted within PUMA each replicate |
| Global μ | Random forest (50 trees, min leaf 5), mean within-PUMA shrinkage |
| GHCP score | Absolute \|Y − μ\| |
| Std-CP | Studentized local half/half RF (recomputed with same seeds); finite mainly at larger $o$ |

### Directory layout

```
min21/
├── results/          # Per-alpha summaries (detailed CSVs local / gitignored)
├── figures/          # Paper-style PDFs (B = 1000)
├── summaries/        # Coverage/width tables for plotting
├── diagnostics/      # Size-ignorability falsification checks (N_j vs summaries)
├── stdcp_studentized/# Std-CP-only recompute manifests
├── logs/             # Run logs
├── seeds_manifest.json
└── demo_b200/        # B = 200 verification runs (figures + summaries)
```

### Reproduce

```bash
# Production GHCP/HCP/baselines (B = 1000), skip Std-CP for speed
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py \
  --B 1000 --n_workers 7 --skip_stdcp --plot

# Recompute studentized local Std-CP with the same seeds; patch results + replot
.venv/bin/python code/marginal/recompute_acs_stdcp_studentized_min21.py \
  --B 1000 --n_workers 7 --alphas 0.2,0.1
```

Verification only (B = 200):

```bash
.venv/bin/python code/marginal/run_acs_yoep_fb_min21_permute.py \
  --B 200 --n_workers 7 --skip_stdcp --plot
```

Plot from saved results:

```bash
.venv/bin/python real_data/acs/plot_dgp_style_paper_plots.py --suite rf_yoep_fb_min21
```

Size-ignorability diagnostics (regenerate `diagnostics/`):

```bash
.venv/bin/python real_data/acs/run_size_ignorability_diagnostics.py
```

Core runner (all flags): `code/marginal/run_acs_experiments.py`  
Cohort filters and design matrix: `real_data/acs/data_processing.py`  
Std-CP recompute: `code/marginal/recompute_acs_stdcp_studentized_min21.py`

## Archive

Earlier ACS variants (stratified OLS/RF, studentized-only runs, YOEP-extended cohorts without age/hours filters, min31 permute, diagnostics) are under [`old/`](old/).
