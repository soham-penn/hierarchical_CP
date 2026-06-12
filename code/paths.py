"""Central path configuration for the reorganized repository."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
SHARED_DGP = CODE_ROOT / "shared" / "dgp"

RESULTS_MARGINAL = REPO_ROOT / "results_marginal"
RESULTS_CONDITIONAL = REPO_ROOT / "results_conditional"
PLOTS_MARGINAL = REPO_ROOT / "plots_marginal"
PLOTS_CONDITIONAL = REPO_ROOT / "plots_conditional"

RESULTS_DGP_MARGINAL = RESULTS_MARGINAL / "dgp"
RESULTS_ACS_MARGINAL = RESULTS_MARGINAL / "acs"
RESULTS_DGP_CONDITIONAL = RESULTS_CONDITIONAL / "dgp"
