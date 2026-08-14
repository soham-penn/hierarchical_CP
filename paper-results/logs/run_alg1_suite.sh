#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HCP_PLOTS_MARGINAL="$ROOT/paper-results"
export HCP_RESULTS_MARGINAL="$ROOT/paper-results/results_marginal"
LOG="$ROOT/paper-results/logs/suite.log"
PY="$ROOT/.venv/bin/python"
DGP_WORKERS=8
ACS_WORKERS=7
ALPHAS_ALL="0.05,0.1,0.15,0.2"

exec >>"$LOG" 2>&1
echo "======== START suite $(date) ========"
echo "PLOTS=$HCP_PLOTS_MARGINAL"
echo "RESULTS=$HCP_RESULTS_MARGINAL"
echo "Configs: fixedN21 (chunk=125) + poissonNmean25 (chunk=25); BASE_SEED=457"

echo "===== DGP shared-alpha sims $(date) ====="
"$PY" -u code/marginal/run_true_marginal_latent_intercept_rf_experiments.py \
  --alphas "$ALPHAS_ALL" \
  --gamma 5 \
  --configs fixedN21,poissonNmean25 \
  --total_replicates 1000 \
  --n_workers "$DGP_WORKERS" \
  --quantile-mode deterministic

echo "===== DGP plots/tables $(date) ====="
"$PY" -u code/marginal/plot_paper.py --suite dgp_rf
"$PY" -u code/marginal/export_paper_tables_mean_width.py || true
"$PY" -u code/marginal/export_paper_tables_baselines.py || true

echo "===== ACS GHCP/HCP (skip Std-CP) $(date) ====="
"$PY" -u code/marginal/run_acs_yoep_fb_min21_permute.py \
  --alphas "$ALPHAS_ALL" \
  --B 1000 --n_workers "$ACS_WORKERS" --skip_stdcp --plot

echo "===== ACS Std-CP studentized recompute $(date) ====="
"$PY" -u code/marginal/recompute_acs_stdcp_studentized_min21.py \
  --B 1000 --n_workers "$ACS_WORKERS" --alphas "$ALPHAS_ALL"

echo "======== DONE $(date) ========"
