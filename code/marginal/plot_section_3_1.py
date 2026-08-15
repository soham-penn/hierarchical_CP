#!/usr/bin/env python3
"""Rebuild Section 3.1 figures and LaTeX tables from shipped (or newly run) CSVs.

Writes ``paper-results/dgp/figures/`` and ``paper-results/dgp/tables/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def main() -> None:
    cmds = [
        [PY, "-u", "code/marginal/plot_paper.py", "--suite", "dgp_rf"],
        [PY, "-u", "code/marginal/export_paper_tables_mean_width.py"],
        [PY, "-u", "code/marginal/export_paper_tables_baselines.py"],
    ]
    for cmd in cmds:
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
