#!/usr/bin/env bash
# Appendix D.3–D.4 sensitivity experiments (does not overwrite Sec. 3.1 CSVs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HCP_PLOTS_MARGINAL="$ROOT/paper-results"
export HCP_RESULTS_MARGINAL="$ROOT/paper-results/results"
LOG="$ROOT/paper-results/logs/appendix_sensitivity.log"
PY="$ROOT/.venv/bin/python"
WORKERS="${WORKERS:-6}"

mkdir -p "$ROOT/paper-results/logs"
exec >>"$LOG" 2>&1
echo "======== START appendix D.3–D.4 $(date) ========"

echo "===== D.3 size-shift $(date) ====="
"$PY" -u code/marginal/run_size_shift_sensitivity.py --B 1000 --n_workers "$WORKERS"
"$PY" -u code/marginal/plot_size_shift_sensitivity.py

echo "===== D.4 merger weights, gamma=5 $(date) ====="
"$PY" -u code/marginal/run_effect_of_weights.py --B 1000 --n_workers "$WORKERS" --gamma 5
"$PY" -u code/marginal/plot_effect_of_weights.py --gamma 5

echo "===== D.4 ud_narrow gamma=0 RF+Bayes $(date) ====="
"$PY" -u code/marginal/run_effect_of_weights_ud_narrow.py --B 1000 --n_workers "$WORKERS"
"$PY" -u code/marginal/plot_effect_of_weights_ud_narrow.py

echo "======== DONE $(date) ========"
