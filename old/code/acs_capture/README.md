# ACS capture / apply pipeline

Run **once** (expensive RF), store **full CSV artifacts**, then apply different
within-group corrections or global OLS **without refitting RF**.

## Why the previous run took ~5 hours

Per replicate the code evaluates roughly:

    (# test PUMAs) × (# o values) × 2 methods (Donor-HCP, S-HCP)

Each call runs `fit_global` → **50-tree Random Forest** on the complement group set
(`S_comp`). With ~40 test PUMAs, 5 o values, 2 methods:

    ~400 RF trainings × 1000 replicates ≈ 400,000 RF fits

Filtering the ACS cohort is cheap; **RF training inside D-HCP/S-HCP dominates**.

The old `replicate_cache/*.pkl` only stored PUMA draws and `(X,Y)` — not
`mu_global` or fitted models — so “recompute from cache” still refit RF.

## New workflow

### 1. Capture (run once)

```bash
.venv/bin/python code/marginal/acs_capture/run_capture.py \
  --suite rf_2012_mean \
  --predictor rf \
  --within_group_mode mean \
  --B 1000 --n_workers 6
```

Writes `plots_marginal/acs/rf_2012_mean/capture_data/`:

| File | Contents |
|------|----------|
| `experiment_config.csv` | Seeds, alpha, o_values, cohort flags |
| `feature_names.csv` | `f_0`, `f_1`, … |
| `observations.csv` | Every row: replicate, puma, row_idx, y, income, features |
| `replicate_design.csv` | Calib / test PUMA assignment per replicate |
| `baseline_*.csv` | Baseline HCP train/calib split + `mu_global` on calib rows |
| `global_fits.csv` | Each `fit_global` call: fit_id, comp PUMAs, call metadata |
| `mu_global.csv` | **μ_RF(x)** at every (fit_id, puma, row_idx) for **all** groups in the HCP call (calib, test, complement) — required for fast apply |
| `mu_predictions.csv` | μ_shrunk / correction output at each `predict_group_mu` call (audit; includes puma, row_idx) |
| `hcp_calls.csv` | Per-call intervals, μ_hat, coverage (audit trail) |

### 2. Apply correction / OLS global (fast)

```bash
.venv/bin/python code/marginal/acs_capture/apply_acs_capture.py \
  --capture_dir plots_marginal/acs/rf_2012_mean/capture_data \
  --within_group_mode correction \
  --global_predictor rf \
  --output_suite rf_2012_correction
```

- **`--global_predictor rf`**: uses `mu_global.csv` (no RF retrain).
- **`--global_predictor ols`**: fits OLS from `observations.csv` per `global_fits` row (seconds).

Only `fit_group_adjustment` / conformal scoring changes between mean and correction.

## Existing `rf_2012_mean` capture run

The completed `run_capture.py` run stored `mu_global` only for **complement** PUMAs
(a bug in an earlier `capture_mu.py` version). That is **not enough** for fast
correction apply. Re-run capture with the fixed script, or use apply with RF
refit (slow, ~1h with 6 workers).

## Scripts (do not use `run_acs_experiments.py --cache_dir` for this)

| Script | Role |
|--------|------|
| `code/marginal/acs_capture/run_capture.py` | One-shot capture + summary results |
| `code/marginal/acs_capture/apply_acs_capture.py` | Apply within-group / OLS from CSV |
| `code/marginal/run_acs_experiments.py` | Legacy path (summary-only; unchanged behavior) |
