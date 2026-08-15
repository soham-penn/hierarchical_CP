"""Central path configuration for paper outputs.

Defaults point at ``paper-results/`` (figures + nested raw CSVs).

Override if needed::

    export HCP_PLOTS_MARGINAL=/path/to/paper-results
    export HCP_RESULTS_MARGINAL=/path/to/paper-results/results
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
SHARED_DGP = CODE_ROOT / "shared" / "dgp"

# Paper deliverable root (figures, summaries, tables, ACS suite).
PLOTS_MARGINAL = Path(
    os.environ.get("HCP_PLOTS_MARGINAL", str(REPO_ROOT / "paper-results"))
).expanduser().resolve()

# Raw trial CSVs live under paper-results/results/ by default.
# HCP_RESULTS_MARGINAL still accepted; also honor HCP_RESULTS.
_default_results = PLOTS_MARGINAL / "results"
RESULTS_MARGINAL = Path(
    os.environ.get(
        "HCP_RESULTS",
        os.environ.get("HCP_RESULTS_MARGINAL", str(_default_results)),
    )
).expanduser().resolve()

RESULTS_DGP_MARGINAL = RESULTS_MARGINAL / "dgp"
RESULTS_ACS_MARGINAL = RESULTS_MARGINAL / "acs"

# Simulation figure/table suite (formerly dgp_true_marginal_rf/).
DGP_PAPER = PLOTS_MARGINAL / "dgp"

# Archived under old/ (conditional calibration, alt DGP suites, legacy plots).
OLD_ROOT = REPO_ROOT / "old"
RESULTS_CONDITIONAL = OLD_ROOT / "conditional" / "results"
PLOTS_CONDITIONAL = OLD_ROOT / "conditional" / "plots"
RESULTS_DGP_CONDITIONAL = RESULTS_CONDITIONAL / "dgp"
