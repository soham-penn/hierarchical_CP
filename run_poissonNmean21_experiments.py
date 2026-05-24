#!/usr/bin/env python3
"""Run Poisson-size (N_i = 1 + Poi(20)) DGP experiments."""
import multiprocessing as _mp
_mp.set_start_method("fork", force=True)

import run_latent_intercept_experiment as rle

if __name__ == "__main__":
    rle.POISSON_MODE = True
    rle.N_WORKERS = 6
    rle.main()
