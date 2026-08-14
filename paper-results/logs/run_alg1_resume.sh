#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HCP_PLOTS_MARGINAL="$ROOT/paper-results"
export HCP_RESULTS_MARGINAL="$ROOT/paper-results/results_marginal"
LOG="$ROOT/paper-results/logs/suite.log"
PY="$ROOT/.venv/bin/python"
ACS_WORKERS=7
ALPHAS_ALL="0.05,0.10,0.15,0.20"

exec >>"$LOG" 2>&1
echo "======== RESUME plots+ACS $(date) ========"

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
