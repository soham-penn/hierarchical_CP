# ACS real-data experiments (paper Sec. 3.2)

California ACS PUMS income prediction with GHCP (Algorithm 1, $\eta=0.5$), HCP, and conformal baselines.

Empirical illustration under the paper working model; size-ignorability checks are in `diagnostics/`.

Full download / filters / recodes / seeds: [`real_data/acs/README.md`](../../real_data/acs/README.md).

```bash
pip install -r requirements.txt
.venv/bin/python real_data/acs/download_acs_ca_pums.py   # gitignored extract
```

The paper cohort (12,285 rows) is filtered at runtime — no separate cleaned CSV.

## Settings (match Sec. 3.2)

| Setting | Value |
|---------|--------|
| Extract | 2018 ACS 1-year CA PUMS (2010 PUMA vintage) |
| Cohort | Foreign-born, YOEP $\ge 2000$, age 25–54, usual hours $\ge 40$ |
| Individuals / PUMAs | 12,285 people; 265 CA PUMAs; **212 eligible** ($N_j\ge 21$) |
| Target | Individual at **position 21** (index 20) |
| Initial sample $o$ | $0,5,10,15,20$ (first $o$ permuted records) |
| Local training | $\tau=\lfloor o/2\rfloor$ as in paper (3) |
| Row order | Uniform permutation within each selected PUMA |
| Design | `uniform_one_target`: 20 reference PUMAs + 1 test PUMA |
| Global $\mu$ | RF, 50 trees, min leaf 5 |
| Score | Absolute $\lvert Y-\widetilde\mu\rvert$ |
| $U$ | Scalar 0 (no PUMA-level features) |
| Std-CP | Studentized, randomized, local RF min leaf 5 |
| $\alpha$ | $\{0.05,0.10,0.15,0.20\}$ (main text uses $0.1$) |
| $B$ | 1000 |

### Layout

```
acs/
├── results/       # per-α summaries (detailed CSVs gitignored)
├── figures/
├── summaries/
├── diagnostics/
├── logs/
└── seeds_manifest.json
```

### Reproduce

```bash
.venv/bin/python code/marginal/run_section_3_2.py \
  --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot

.venv/bin/python code/marginal/recompute_acs_stdcp_randomized_min21.py \
  --B 1000 --n_workers 7 --alphas 0.05,0.1,0.15,0.2

.venv/bin/python code/marginal/plot_paper.py --suite acs
.venv/bin/python real_data/acs/plot_dgp_style_paper_plots.py --suite rf_yoep_fb_min21

.venv/bin/python real_data/acs/run_size_ignorability_diagnostics.py
```

Core runner: `code/marginal/run_acs_experiments.py`
Public launcher: `code/marginal/run_section_3_2.py`  
Filters: `real_data/acs/data_processing.py`
