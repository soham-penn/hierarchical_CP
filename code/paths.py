"""Central path configuration for paper outputs.

Defaults point at ``paper-results/`` (figures + nested raw CSVs).

Override if needed::

    export HCP_PLOTS_MARGINAL=/path/to/paper-results
    export HCP_RESULTS_MARGINAL=/path/to/paper-results/results_marginal
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

# Raw trial CSVs live under paper-results/results_marginal/ by default.
RESULTS_MARGINAL = Path(
    os.environ.get(
        "HCP_RESULTS_MARGINAL",
        str(PLOTS_MARGINAL / "results_marginal"),
    )
).expanduser().resolve()

RESULTS_DGP_MARGINAL = RESULTS_MARGINAL / "dgp"
RESULTS_ACS_MARGINAL = RESULTS_MARGINAL / "acs"

# Archived under old/ (conditional calibration, alt DGP suites, legacy plots).
OLD_ROOT = REPO_ROOT / "old"
RESULTS_CONDITIONAL = OLD_ROOT / "results_conditional"
PLOTS_CONDITIONAL = OLD_ROOT / "plots_conditional"
RESULTS_DGP_CONDITIONAL = RESULTS_CONDITIONAL / "dgp"
