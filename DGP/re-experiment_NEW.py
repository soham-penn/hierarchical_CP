#!/usr/bin/env python3
"""Run marginal DGP experiment with fixed group size N_i = 21 in parallel on 5 cores.

This version:
- parallelizes over experiment chunks
- saves combined raw results to NEW_RESULTS_2/
- also saves worker-level raw chunk CSVs
- saves the summary CSV
- does not make any plots
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()

# If this file is at .../hier_current/DGP/re-experiment_NEW.py,
# then project root is .../hier_current
PROJECT_ROOT = SCRIPT_PATH.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import DGP.code.experiments as exp_mod
from DGP.code.experiments import run_experiments_outer
from DGP.code.summary_and_plots import summarize_methods
from DGP.code.joint_xy_marginal_common import draw_group_joint_xy
from methods import create_mu_method_ols_global_only, create_mu_method_ols_offset


FIXED_N = 21
DIMENSION = 5
U_MIN = 1.0
U_MAX = 5.0
RHO = 0.5

N_WORKERS = 5
BASE_SEED = 123


def generate_calibration_data_fixed(number_groups, lambda_Poisson, dgp_specification):
    u_cal = np.random.uniform(
        low=dgp_specification["u_min"],
        high=dgp_specification["u_max"],
        size=(number_groups, dgp_specification["dimension"]),
    )

    n_vec = np.full(number_groups, FIXED_N, dtype=int)
    z_cal = []
    for j in range(number_groups):
        z_cal.append(draw_group_joint_xy(u_cal[j, :], int(n_vec[j]), rho=RHO))

    return {
        "U_calibration": u_cal,
        "Z_calibration": z_cal,
        "sample_size_vector": n_vec,
    }


def generate_test_group_fixed(lambda_Poisson, dgp_specification, o_observed, fixed_U=None):
    d = dgp_specification["dimension"]

    if FIXED_N <= o_observed:
        raise ValueError(
            f"Fixed group size N={FIXED_N} is not greater than o_observed={o_observed}."
        )

    if fixed_U is not None:
        u_test = np.asarray(fixed_U, dtype=float).reshape(1, -1)
    else:
        u_test = np.random.uniform(
            low=dgp_specification["u_min"],
            high=dgp_specification["u_max"],
            size=(1, d),
        )

    n_test = FIXED_N
    z_test = draw_group_joint_xy(u_test[0, :], n_test, rho=RHO)

    return {
        "U_test": u_test,
        "Z_test": z_test,
        "N_test": n_test,
    }


def patch_fixed_size_generators():
    exp_mod.generate_calibration_data = generate_calibration_data_fixed
    exp_mod.generate_test_group = generate_test_group_fixed


def split_counts(total, n_parts):
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def offset_experiment_column(df, offset):
    out = df.copy()
    for col in ["experiment", "experiment_id", "exp_id"]:
        if col in out.columns:
            out[col] = out[col] + offset
            break
    return out


def run_chunk(
    worker_id,
    number_experiments_chunk,
    experiment_offset,
    number_groups_k,
    lambda_dummy,
    dgp_dummy,
    o_vec,
    target_idx,
    alpha,
    number_test_groups,
):
    np.random.seed(BASE_SEED + 1000 * worker_id)
    patch_fixed_size_generators()

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    results = run_experiments_outer(
        number_experiments=number_experiments_chunk,
        number_groups_k=number_groups_k,
        lambda_Poisson=lambda_dummy,
        dgp_specification=dgp_dummy,
        o_values=o_vec,
        target_index=target_idx,
        alpha=alpha,
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=number_test_groups,
        mu_method_baseline=mu_baseline,
        mu_method_hcp=mu_hcp,
        show_progress=False,
    )

    results = offset_experiment_column(results, experiment_offset)
    results["worker_id"] = worker_id
    return results


def main():
    patch_fixed_size_generators()

    tag = "fixedN21_alpha005_K20_d5_u15_rho05_joint_xy_marginal_parallel5"
    out_dir = PROJECT_ROOT / "NEW_RESULTS_2" / tag
    chunk_dir = out_dir / "chunks"

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    alpha = 0.05
    total_number_experiments = 50
    number_test_groups = 100
    number_groups_k = 20
    lambda_dummy = 21
    o_vec = [0, 5, 10, 15, 20]
    target_idx = 20

    print("=" * 96)
    print(
        f"MARGINAL DGP EXPERIMENT (FIXED N={FIXED_N}, K={number_groups_k}, d={DIMENSION}, "
        f"U~Unif[{U_MIN},{U_MAX}]^d, rho={RHO}, workers={N_WORKERS})"
    )
    print("Project root:", PROJECT_ROOT)
    print("Joint generation: Z=(X,Y) ~ N(mu(U), Sigma(U))")
    print("mu(U)=(U_1^2,...,U_d^2), Sigma(U)=(1-rho)diag(U)+rho 11^T")
    print("=" * 96)
    print(f"o_values={o_vec}, target_index={target_idx}")

    dgp_dummy = {
        "dimension": DIMENSION,
        "u_min": U_MIN,
        "u_max": U_MAX,
    }

    chunk_sizes = split_counts(total_number_experiments, N_WORKERS)
    offsets = np.cumsum([0] + chunk_sizes[:-1]).tolist()

    futures = []
    results_parts = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for worker_id, (chunk_size, offset) in enumerate(zip(chunk_sizes, offsets)):
            if chunk_size == 0:
                continue
            futures.append(
                ex.submit(
                    run_chunk,
                    worker_id,
                    chunk_size,
                    offset,
                    number_groups_k,
                    lambda_dummy,
                    dgp_dummy,
                    o_vec,
                    target_idx,
                    alpha,
                    number_test_groups,
                )
            )

        for i, fut in enumerate(as_completed(futures), start=1):
            part = fut.result()
            results_parts.append(part)

            worker_id = int(part["worker_id"].iloc[0]) if "worker_id" in part.columns else i - 1
            chunk_path = chunk_dir / f"raw_results_chunk_worker_{worker_id}.csv"
            part.to_csv(chunk_path, index=False)

            print(
                f"[{i}/{len(futures)}] worker chunk completed "
                f"with {len(part)} rows -> {chunk_path.name}"
            )

    results_o = pd.concat(results_parts, ignore_index=True)
    results_o["test_sample_size_o"] = results_o["o_observed"]

    sort_cols = [
        c for c in ["experiment", "experiment_id", "exp_id", "test_sample_size_o", "worker_id"]
        if c in results_o.columns
    ]
    if sort_cols:
        results_o = results_o.sort_values(sort_cols).reset_index(drop=True)

    raw_csv = out_dir / "raw_results_effect_of_o_joint_xy_marginal.csv"
    results_o.to_csv(raw_csv, index=False)

    summary_list_o = []
    for o in sorted(results_o["test_sample_size_o"].unique()):
        df_o = results_o[results_o["test_sample_size_o"] == o]
        tab = summarize_methods(df_o, alpha=alpha, number_test_groups=number_test_groups)
        tab["test_sample_size_o"] = o
        summary_list_o.append(tab)

    summary_o = pd.concat(summary_list_o, ignore_index=True)
    summary_csv = out_dir / "summary_effect_of_o_joint_xy_marginal.csv"
    summary_o.to_csv(summary_csv, index=False)

    print("\n" + "=" * 96)
    print("MARGINAL DGP RUN COMPLETED (FIXED GROUP SIZE, PARALLEL, NO PLOTS)")
    print("=" * 96)
    print(f"Combined raw results: {raw_csv}")
    print(f"Chunk raw results:    {chunk_dir}")
    print(f"Summary:              {summary_csv}")


if __name__ == "__main__":
    main()