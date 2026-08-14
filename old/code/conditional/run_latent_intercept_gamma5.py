#!/usr/bin/env python3
"""Run latent-intercept DGP experiments with gamma=5 only.

Each group draws B_j ~ Normal(0, gamma^2) i.i.d.; with gamma=5 this is N(0, 25).
The shift is applied to the response coordinate only (Y gets +B_j).

Writes results to results_conditional/dgp/latent_gamma5_{fixedN21,poissonNmean21}/.
Paper plots: python code/conditional/plot_paper.py --suite latent_gamma5
"""

from __future__ import annotations

import multiprocessing as _mp

try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import code.conditional.latent_intercept_core as rle
GAMMA = 5.0


def _run_mode(*, poisson: bool, out_name: str, skip_stability: bool = False) -> None:
    rle.POISSON_MODE = poisson
    rle.GAMMA_VALUES = [GAMMA]
    rle.OUT_DIR = REPO_ROOT / "results_conditional" / "dgp" / out_name
    rle.OUT_DIR.mkdir(parents=True, exist_ok=True)
    rle.N_WORKERS = 6
    rle.plot_requested_sets = lambda *_args, **_kwargs: None  # paper plots handled separately

    mode = "poissonNmean21" if poisson else "fixedN21"
    print("=" * 80)
    print(f"Latent intercept paper run: {mode}, gamma={GAMMA}")
    print(f"Output directory: {rle.OUT_DIR}")
    print("=" * 80)

    if skip_stability:
        _main_without_stability()
    else:
        rle.main()


def _main_without_stability() -> None:
    """Run main experiment loop only (no randomization-stability block)."""
    mode_tag = rle._mode_tag()
    print(f"Running latent-intercept experiment with {rle.N_WORKERS} cores (no stability).")
    print(f"Run mode: {mode_tag}")

    all_results = []
    for gamma_val in rle.GAMMA_VALUES:
        for alpha in rle.ALPHA_VALUES:
            args = [(exp_id, gamma_val, alpha) for exp_id in range(rle.N_EXPERIMENTS)]
            print(f"[Main] gamma={gamma_val}, alpha={alpha} ({len(args)} outer experiments)")
            from multiprocessing import Pool

            with Pool(rle.N_WORKERS) as pool:
                chunks = list(pool.imap_unordered(rle.run_experiment_wrapper, args))
            all_results.append(rle.pd.concat(chunks, ignore_index=True))

    df_all = rle.pd.concat(all_results, ignore_index=True)
    raw_path = rle.OUT_DIR / f"{mode_tag}_raw_results_complete.csv"
    df_all.to_csv(raw_path, index=False)

    summary = (
        df_all.groupby(["gamma", "alpha", "o", "method"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            width_mean=("width", "mean"),
            width_median=("width", "median"),
            width_std=("width", "std"),
            proportion_infinite=("infinite", "mean"),
            n_infinite=("infinite", "sum"),
        )
    )
    summary["base_seed"] = rle.BASE_SEED
    summary_path = rle.OUT_DIR / f"{mode_tag}_summary_by_gamma_alpha_o_method.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    _run_mode(poisson=False, out_name="latent_gamma5_fixedN21", skip_stability=False)
    _run_mode(poisson=True, out_name="latent_gamma5_poissonNmean21", skip_stability=False)


if __name__ == "__main__":
    main()
