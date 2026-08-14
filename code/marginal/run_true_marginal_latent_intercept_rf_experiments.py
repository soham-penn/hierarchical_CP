#!/usr/bin/env python3
"""True-marginal latent-intercept DGP (gamma) with Random Forest global predictors."""

from __future__ import annotations

import argparse
import functools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.shared.dgp.experiments import run_experiments_outer
from code.shared.plot_engine import PAPER_ALPHA_GRID_STR
from code.paths import RESULTS_DGP_MARGINAL
from code.marginal.run_true_marginal_latent_intercept_experiments import (
    BASE_SEED,
    DEFAULT_GAMMA,
    alpha_to_tag,
    build_configs,
    gamma_tag,
    offset_experiment_column,
    patch_fixed_size_generators,
    patch_poisson_generators,
    split_counts,
    without_within_group_training,
)
from methods import (
    create_mu_method_random_forest_global_only,
    create_mu_method_random_forest_offset,
)

RESULTS_PREFIX = "true_marg_latent_rf"
RF_NTREE = 50
RF_NODESIZE = 5
RF_RANDOM_STATE = 123
N_WORKERS_DEFAULT = 18

print = functools.partial(print, flush=True)


def run_chunk(worker_id, number_experiments_chunk, experiment_offset, config):
    np.random.seed(BASE_SEED + 1000 * worker_id)

    gamma = float(config["gamma"])
    rho = float(config["rho"])
    min_target_n = int(config["target_index"]) + 1

    if config.get("generation_mode") == "fixed":
        patch_fixed_size_generators(
            fixed_n=config["fixed_n"],
            target_n=config["target_n"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=rho,
            gamma=gamma,
        )
    elif config.get("generation_mode") == "poisson":
        patch_poisson_generators(
            lambda_poisson=config["lambda_poisson"],
            dimension=config["dimension"],
            u_min=config["u_min"],
            u_max=config["u_max"],
            rho=rho,
            gamma=gamma,
            min_target_n=min_target_n,
            size_offset=int(config.get("size_offset", 1)),
        )
    else:
        raise ValueError(f"Unknown generation_mode: {config.get('generation_mode')}")

    mu_baseline = create_mu_method_random_forest_global_only(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE
    )
    # c=1 → Eq. (4): w_g = |Strain| / (|Strain| + τ)
    mu_hcp = create_mu_method_random_forest_offset(
        ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
    )
    mu_hcp_no_within = without_within_group_training(
        create_mu_method_random_forest_offset(
            ntree=RF_NTREE, nodesize=RF_NODESIZE, random_state=RF_RANDOM_STATE, c=1.0
        )
    )

    results = run_experiments_outer(
        number_experiments=number_experiments_chunk,
        number_groups_k=config["number_groups_k"],
        lambda_Poisson=config.get("lambda_poisson", 21),
        dgp_specification={
            "dimension": config["dimension"],
            "u_min": config["u_min"],
            "u_max": config["u_max"],
        },
        o_values=config["o_values"],
        target_index=config["target_index"],
        alpha=config["alpha"],
        alphas=config.get("alphas"),
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=1,
        mu_method_baseline=mu_baseline,
        mu_method_hcp=mu_hcp,
        mu_method_hcp_no_within=mu_hcp_no_within,
        show_progress=False,
        quantile_mode=config.get("quantile_mode", "deterministic"),
        quantile_base_seed=config.get("quantile_base_seed", BASE_SEED),
    )

    results = offset_experiment_column(results, experiment_offset)
    results["worker_id"] = worker_id
    results["gamma"] = gamma
    return results


def run_experiment(config_name: str, config: dict, n_workers: int, chunk_size: int = 25) -> pd.DataFrame:
    """Run one design (fixed/poisson). If config['alphas'] has many levels, results
    include an ``alpha`` column and are also split into per-α output folders.

    Chunking controls RNG streams: each chunk ``i`` is seeded with
    ``BASE_SEED + 1000 * i``. Shipped paper CSVs use:
      - ``fixedN21``: ``chunk_size=125`` (8 chunks for B=1000)
      - ``poissonNmean25``: ``chunk_size=25`` (40 chunks for B=1000)
    """
    alphas = config.get("alphas")
    if alphas is None:
        alphas = [float(config["alpha"])]
    else:
        alphas = [float(a) for a in alphas]

    # Shared working dir when multi-alpha; also write per-alpha complete CSVs.
    multi = len(alphas) > 1
    tag_base = f"{RESULTS_PREFIX}_{gamma_tag(config['gamma'])}_{config_name}"
    out_dir = RESULTS_DGP_MARGINAL / (f"{tag_base}_multialpha" if multi else f"{tag_base}_{alpha_to_tag(alphas[0])}")
    # config_name already may include alpha tag when single-alpha; keep prior behavior
    if not multi:
        out_dir = RESULTS_DGP_MARGINAL / f"{RESULTS_PREFIX}_{gamma_tag(config['gamma'])}_{config_name}"
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"TRUE MARGINAL LATENT RF: {config_name}")
    print("=" * 96)
    print(f"gamma: {config['gamma']}  |  RF ntree={RF_NTREE}, nodesize={RF_NODESIZE}")
    print(f"Generation mode: {config.get('generation_mode')}")
    print(f"Workers: {n_workers}, replicates: {config['total_replicates']}, alphas: {alphas}")
    print("=" * 96)

    # Chunk layout fixes RNG streams (seed = BASE_SEED + 1000 * chunk_id).
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        n_chunks = int(n_workers)
        max_chunk = None
    else:
        n_chunks = max(int(n_workers), int(np.ceil(config["total_replicates"] / chunk_size)))
        max_chunk = chunk_size
    chunk_sizes = split_counts(config["total_replicates"], n_chunks)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()
    results_parts = []
    print(
        f"Submitting {n_chunks} chunks "
        f"(sizes={chunk_sizes[:3]}{'...' if len(chunk_sizes) > 3 else ''}; "
        f"seed_i = {BASE_SEED}+1000*i)"
    )

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        for worker_id, (csize, offset) in enumerate(zip(chunk_sizes, offsets)):
            if csize == 0:
                continue
            futures.append(ex.submit(run_chunk, worker_id, csize, offset, config))

        for i, fut in enumerate(as_completed(futures), start=1):
            part = fut.result()
            results_parts.append(part)
            worker_id = int(part["worker_id"].iloc[0])
            chunk_path = chunk_dir / f"raw_results_chunk_worker_{worker_id}.csv"
            part.to_csv(chunk_path, index=False)
            print(f"[{i}/{len(futures)}] chunk {worker_id} done ({len(part)} rows, offset {offsets[worker_id]})")

    results = pd.concat(results_parts, ignore_index=True)
    sort_cols = [c for c in ["alpha", "experiment", "experiment_id", "o_observed", "worker_id"] if c in results.columns]
    if sort_cols:
        results = results.sort_values(sort_cols).reset_index(drop=True)

    raw_csv = out_dir / f"{out_dir.name}_raw_results_complete.csv"
    results.to_csv(raw_csv, index=False)
    print(f"Saved: {raw_csv} ({len(results)} rows)")

    # Split into per-alpha dirs expected by plot_paper / existing layout
    if multi and "alpha" in results.columns:
        import re as _re
        design = _re.sub(r"_alpha[0-9p.]+$", "", config_name)
        for a in alphas:
            a_tag = alpha_to_tag(a)
            tag = f"{RESULTS_PREFIX}_{gamma_tag(config['gamma'])}_{design}_{a_tag}"
            a_dir = RESULTS_DGP_MARGINAL / tag
            a_dir.mkdir(parents=True, exist_ok=True)
            sub = results[np.isclose(results["alpha"].astype(float), float(a))].copy()
            out_csv = a_dir / f"{tag}_raw_results_complete.csv"
            sub.to_csv(out_csv, index=False)
            print(f"Saved per-alpha: {out_csv} ({len(sub)} rows)")

    # Reproducibility manifest next to the complete CSV.
    import json
    from datetime import datetime, timezone

    manifest = {
        "config_name": config_name,
        "generation_mode": config.get("generation_mode"),
        "lambda_poisson": config.get("lambda_poisson"),
        "size_offset": config.get("size_offset"),
        "size_law": (
            f"N = {config.get('size_offset', 0)} + Poi({config.get('lambda_poisson')})"
            if config.get("generation_mode") == "poisson"
            else f"N = {config.get('fixed_n')} (fixed); target_n = {config.get('target_n')}"
        ),
        "gamma": float(config["gamma"]),
        "total_replicates": int(config["total_replicates"]),
        "alphas": list(alphas),
        "shared_alphas": multi,
        "o_values": list(config["o_values"]),
        "target_index": int(config["target_index"]),
        "number_groups_k": int(config["number_groups_k"]),
        "quantile_mode": config.get("quantile_mode", "deterministic"),
        "quantile_base_seed": int(config.get("quantile_base_seed", BASE_SEED)),
        "base_seed": BASE_SEED,
        "chunk_seed_formula": "np.random.seed(BASE_SEED + 1000 * chunk_id)",
        "n_workers": int(n_workers),
        "chunk_size_arg": chunk_size if chunk_size > 0 else None,
        "n_chunks": int(n_chunks),
        "chunk_sizes": list(map(int, chunk_sizes)),
        "rf": {"ntree": RF_NTREE, "nodesize": RF_NODESIZE, "random_state": RF_RANDOM_STATE, "c": 1.0},
        "rows": int(len(results)),
        "saved": str(raw_csv),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote manifest: {manifest_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="True-marginal latent-intercept DGP with RF global predictors."
    )
    parser.add_argument("--alphas", type=str, default=PAPER_ALPHA_GRID_STR)
    parser.add_argument("--configs", type=str, default="fixedN21,poissonNmean25")
    parser.add_argument("--total_replicates", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--n_workers", type=int, default=N_WORKERS_DEFAULT)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Replicates per RNG chunk (chunk i seeded with BASE_SEED+1000*i). "
            "Default: 125 for fixedN21 / poissonNmean21, 25 for poissonNmean25 "
            "(matches shipped CSVs). Pass 0 for one chunk per worker."
        ),
    )
    parser.add_argument(
        "--quantile-mode",
        choices=["deterministic", "randomized"],
        default="deterministic",
    )
    parser.add_argument("--quantile-base-seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--shared-alphas",
        action="store_true",
        default=True,
        help="Evaluate all --alphas on the same replicates (default: on).",
    )
    parser.add_argument(
        "--no-shared-alphas",
        action="store_true",
        help="Legacy: separate full run per alpha.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    shared = not args.no_shared_alphas

    # Per-config chunk sizes so a single suite command reproduces shipped CSVs.
    paper_chunk = {
        "fixedN21": 125,
        "poissonNmean25": 25,
        "poissonNmean21": 125,
    }

    def _chunk_for(name: str) -> int:
        if args.chunk_size is not None:
            return int(args.chunk_size)
        return int(paper_chunk.get(name, 25))

    if shared and len(alphas) > 1:
        # One pass per design; all alphas share DGP draws / GHCP fits.
        configs = build_configs(
            total_replicates=args.total_replicates,
            alpha=alphas[0],
            gamma=args.gamma,
            quantile_mode=args.quantile_mode,
            quantile_base_seed=args.quantile_base_seed,
        )
        for name in config_names:
            cfg = dict(configs[name])
            cfg["alphas"] = alphas
            cfg["alpha"] = alphas[0]
            run_experiment(name, cfg, args.n_workers, chunk_size=_chunk_for(name))
    else:
        for alpha in alphas:
            configs = build_configs(
                total_replicates=args.total_replicates,
                alpha=alpha,
                gamma=args.gamma,
                quantile_mode=args.quantile_mode,
                quantile_base_seed=args.quantile_base_seed,
            )
            alpha_tag = alpha_to_tag(alpha)
            for name in config_names:
                cfg = dict(configs[name])
                cfg["alphas"] = None
                run_experiment(
                    f"{name}_{alpha_tag}", cfg, args.n_workers, chunk_size=_chunk_for(name)
                )


if __name__ == "__main__":
    main()
