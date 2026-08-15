#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HCP_PLOTS_MARGINAL="$ROOT/paper-results"
export HCP_RESULTS_MARGINAL="$ROOT/paper-results/results"
LOG="$ROOT/paper-results/logs/suite.log"
PY="$ROOT/.venv/bin/python"
DGP_WORKERS=8
ACS_WORKERS=7
ALPHAS_ALL="0.05,0.1,0.15,0.2"

exec >>"$LOG" 2>&1
echo "======== START suite $(date) ========"
echo "PLOTS=$HCP_PLOTS_MARGINAL"
echo "RESULTS=$HCP_RESULTS_MARGINAL"
echo "Configs: fixedN21 (chunk=125) + poissonNmean25 (chunk=25); BASE_SEED=457; γ=5; η=0.5"
echo "Appendix D.3–D.4: paper-results/logs/run_appendix_sensitivity.sh"

echo "===== DGP shared-alpha sims $(date) ====="
"$PY" -u code/marginal/run_section_3_1.py --n_workers "$DGP_WORKERS"

echo "===== DGP plots/tables $(date) ====="
"$PY" -u code/marginal/plot_section_3_1.py

echo "===== ACS GHCP/HCP (skip Std-CP) $(date) ====="
"$PY" -u code/marginal/run_section_3_2.py \
  --alphas "$ALPHAS_ALL" \
  --B 1000 --n_workers "$ACS_WORKERS" --skip_stdcp --plot

echo "===== ACS Std-CP studentized randomized recompute $(date) ====="
"$PY" -u code/marginal/recompute_acs_stdcp_randomized_min21.py \
  --B 1000 --n_workers "$ACS_WORKERS" --alphas "$ALPHAS_ALL"

echo "======== DONE $(date) ========"
