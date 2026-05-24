#!/usr/bin/env python3
"""Run fixed-N (N=21) DGP experiments."""
import multiprocessing as _mp
_mp.set_start_method("fork", force=True)

import run_latent_intercept_experiment as rle

if __name__ == "__main__":
    rle.POISSON_MODE = False
    rle.N_WORKERS = 6
    rle.main()
