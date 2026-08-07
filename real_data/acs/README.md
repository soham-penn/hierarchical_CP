# ACS data and preprocessing

Entry point for the paper ACS suite: `code/marginal/run_acs_yoep_fb_min21_permute.py`  
Low-level runner: `code/marginal/run_acs_experiments.py`

## Data

Download California PUMS (via `folktables`):

```bash
python real_data/acs/download_acs_ca_pums.py
```

Output: `real_data/acs/data/acs_data_all50states.csv`

PUMA-level summary for the filtered cohort:

```bash
python real_data/acs/export_puma_summary.py
```

## Cohort filters (`load_and_clean_acs_pums`)

Applied in order in `data_processing.py`:

| Step | Filter |
|------|--------|
| State | California (`--acs_state CA`) |
| Nativity | Foreign-born (`NATIVITY == 2`) |
| Age | 25–54 (default; disable with `--no_age_filter`) |
| Year of entry | YOEP ≥ 2000 (`--yoep_min_year 2000`) |
| Labor force | Usual hours ≥ 40 (default; disable with `--no_hours_filter`) |
| Income | Drop missing; drop non-positive |
| Covariates | Drop rows missing any model column |
| Income floor / trim | None in paper runs (`--min_income 0`, `--drop_top_income_pct 0`) |

Paper extract: **12,285** individuals, **212** PUMAs with size ≥ 21.

## Predictors (`build_design_matrix_acs`)

Continuous: age, age², hours, married, female  
Factors: education, English proficiency  
Excluded by default: entry_recency (YOEP-derived), class-of-worker dummies (`--exclude_cow`, default True)

## Replicate design (`--design uniform_one_target`)

Each replicate:

1. Draw 20 calibration PUMAs uniformly from eligible PUMAs (size ≥ 21).
2. Draw one target PUMA from the remainder.
3. Permute rows within each PUMA (`permute_rows=True`).
4. Fix prediction target at index 20; history uses indices 0, …, o − 1.

No row bootstrap; group sizes are fixed at observed ACS counts.

## Conformal settings (paper)

| Method | Score / notes |
|--------|----------------|
| GHCP, S-HCP, baselines | Absolute \|Y − μ\| |
| Std-CP | Studentized (local RF σ on absolute residuals); `--stdcp_score_type studentized` |
| Partial Std-CP | `--stdcp_o_values 20` (α = 0.1 production run) |

Quantile mode: deterministic (`--quantile-base-seed 456`).

## Plotting

```bash
python real_data/acs/plot_dgp_style_paper_plots.py --suite rf_yoep_fb_min21
```

Writes to `plots_marginal/acs/min21/figures/`.

## Archive

Superseded scripts, diagnostics, and alternate cohort summaries are under `old/real_data/acs/` in the repository root.
