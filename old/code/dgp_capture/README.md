# DGP RF capture / apply

Run **once** (expensive RF), store **full CSV artifacts**, then apply different
within-group modes (**mean** / **correction** / **none**) **without refitting RF**.

## Why capture is required

Each D-HCP / S-HCP call fits a global RF on the complement group set `S_comp`.
With 8 `o` values, 2 HCP methods, and 1000 replicates, that is thousands of RF
fits. Recomputing within-group adjustment alone must **not** refit RF.

## Artifacts (`capture_data/`)

| File | Contents |
|------|----------|
| `experiment_config.csv` | Seeds, alpha, o_values, gamma, RF hyperparams, DGP flags |
| `feature_names.csv` | `x_*` and `u_*` column names |
| `observations.csv` | Every row: experiment, group_id, role, row_idx, y, x_*, u_* |
| `experiment_design.csv` | Calib / test group assignment per experiment |
| `baseline_hcp_split.csv` | Train / calib slots for baseline HCP split |
| `baseline_mu_global.csv` | **μ_RF** on all calib + test rows under the baseline fit |
| `global_fits.csv` | Each `fit_global` call: fit_id, S_comp slots, call metadata |
| `mu_global.csv` | **μ_RF(x,u)** at every `(fit_id, group_id, row_idx)` for **all** groups in the HCP call (calib + test + complement) |
| `mu_predictions.csv` | Shrunk / corrected μ at each `predict_group_mu` (audit) |
| `hcp_calls.csv` | Per-call intervals, μ_hat, coverage (audit trail) |

**Critical:** `mu_global.csv` must cover **every** group slot in each HCP call, not only
`S_comp`. Apply uses these values only — it never refits RF.

## Workflow

### 1. Capture (mean within-group, RF global)

```bash
.venv/bin/python code/marginal/dgp_capture/run_capture.py \
  --gamma 5 \
  --configs fixedN21,poissonNmean21 \
  --alphas 0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25 \
  --total_replicates 1000 \
  --n_workers 8
```

Writes:

- Capture: `plots_marginal/dgp_true_marginal_rf/capture_data/{config}_{alpha_tag}/`
- Plot-ready results: `results_marginal/dgp/true_marg_latent_rf_gamma5p0_{config}_{alpha_tag}/`

### 2. Apply a different within-group mode (no RF)

```bash
.venv/bin/python code/marginal/dgp_capture/apply_dgp_capture.py \
  --capture_dir plots_marginal/dgp_true_marginal_rf/capture_data/fixedN21_alpha20 \
  --within_group_mode correction \
  --output_tag corr
```

`--within_group_mode` choices: `mean`, `correction`, `none` (no within-group / pure global).

### 3. Plots

```bash
.venv/bin/python code/marginal/plot_paper.py --suite dgp_rf
```

## Within-group modes

| Mode | `fit_group_adjustment` | Prediction |
|------|------------------------|------------|
| `mean` | Sample mean of Y on first `tau=⌊o/2⌋` rows | Shrink μ_RF toward group mean |
| `correction` | Ridge residual correction on Y − μ_RF | μ_RF + clip(g_j(x)) |
| `none` | Ignored (`tau_override=0`) | Pure μ_RF |

All three use the **same** stored `mu_global.csv`.
