#!/usr/bin/env python3
"""Driver to run requested DGP experiments sequentially.

Runs:
1) Fixed-size groups (N=21)
2) Poisson-size groups (N_i = 1 + Poi(20))

Results are written to top-level NEW_RESULTS/NEW_PLOTS with filename prefixes
so outputs from fixed and Poisson runs do not overwrite each other.
"""

import run_latent_intercept_experiment as rle

def main() -> None:
	# Fixed N run (N=21)
	rle.POISSON_MODE = False
	rle.N_WORKERS = 6
	print("Running fixed-N experiments (N=21) — POISSON_MODE=False")
	rle.main()

	# Poisson run (N_i = 1 + Poi(20))
	rle.POISSON_MODE = True
	rle.N_WORKERS = 6
	print("Running Poisson-size experiments (N_i = 1 + Poi(20)) — POISSON_MODE=True")
	rle.main()


if __name__ == "__main__":
	main()
