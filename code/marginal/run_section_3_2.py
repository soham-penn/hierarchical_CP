#!/usr/bin/env python3
"""Section 3.2 entry point (ACS).

California PUMS income, Algorithm 1 with η=0.5 and merger (3) c=1
(``--within_group_mode mean``). Each replicate applies a **uniform permutation**
of records within each PUMA (not an income ranking).

This is a public name for ``run_acs_yoep_fb_min21_permute.py``.

Example::

    .venv/bin/python real_data/acs/download_acs_ca_pums.py
    .venv/bin/python code/marginal/run_section_3_2.py \\
        --alphas 0.05,0.1,0.15,0.2 --B 1000 --n_workers 7 --skip_stdcp --plot
    .venv/bin/python code/marginal/recompute_acs_stdcp_randomized_min21.py \\
        --B 1000 --n_workers 7 --alphas 0.05,0.1,0.15,0.2
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.marginal.run_acs_yoep_fb_min21_permute import main

if __name__ == "__main__":
    main()
