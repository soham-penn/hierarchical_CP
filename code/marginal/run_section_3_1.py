#!/usr/bin/env python3
"""Section 3.1 entry point (simulations).

Paper DGP: latent intercept γ=5, RF global μ, merger (3) with c=1,
η=0.5, deterministic GHCP/HCP quantiles.

This is a public name for
``run_true_marginal_latent_intercept_rf_experiments.py``. Defaults match the
shipped Sec. 3.1 CSVs: ``fixedN21`` and ``poissonNmean25``,
α∈{0.05,0.1,0.15,0.2}, B=1000.

Example::

    .venv/bin/python code/marginal/run_section_3_1.py --n_workers 8
    .venv/bin/python code/marginal/plot_section_3_1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.marginal.run_true_marginal_latent_intercept_rf_experiments import main

if __name__ == "__main__":
    main()
