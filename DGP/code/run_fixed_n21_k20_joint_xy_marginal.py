#!/usr/bin/env python3
"""Run marginal DGP experiment with fixed group size N_i = 21."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import DGP.code.experiments as exp_mod
from DGP.code.experiments import run_experiments_outer
from DGP.code.summary_and_plots import summarize_methods
from DGP.code.make_plots import make_effect_of_o_plots
from DGP.code.joint_xy_marginal_common import (
    draw_group_joint_xy,
    generate_requested_plots_from_results,
)
from methods import create_mu_method_ols_global_only, create_mu_method_ols_offset


FIXED_N = 21
DIMENSION = 5
U_MIN = 1
U_MAX = 5
RHO = 0.5


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


def main():
    np.random.seed(123)
    patch_fixed_size_generators()

    tag = "fixedN21_K20_d5_u01_rho05_joint_xy_marginal"
    dir_files = ROOT / "DGP" / "results" / "files" / tag
    dir_plots = ROOT / "DGP" / "plots" / tag
    dir_summ = ROOT / "DGP" / "results" / tag

    dir_files.mkdir(parents=True, exist_ok=True)
    dir_plots.mkdir(parents=True, exist_ok=True)
    dir_summ.mkdir(parents=True, exist_ok=True)

    alpha = 0.1
    number_test_groups = 100
    number_groups_k = 20
    lambda_dummy = 21
    o_vec = [0, 5, 10, 15, 20]
    target_idx = 20

    print("=" * 96)
    print(
        f"MARGINAL DGP EXPERIMENT (FIXED N={FIXED_N}, K={number_groups_k}, d={DIMENSION}, "
        f"U~Unif[{U_MIN},{U_MAX}]^d, rho={RHO})"
    )
    print("Joint generation: Z=(X,Y) ~ N(mu(U), Sigma(U))")
    print("mu(U)=(U_1^2,...,U_d^2), Sigma(U)=(1-rho)diag(U)+rho 11^T")
    print("=" * 96)
    print(f"o_values={o_vec}, target_index={target_idx}")

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    dgp_dummy = {
        "dimension": DIMENSION,
        "u_min": U_MIN,
        "u_max": U_MAX,
    }

    results_o = run_experiments_outer(
        number_experiments=50,
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
        show_progress=True,
    )
    results_o["test_sample_size_o"] = results_o["o_observed"]
    results_o.to_csv(dir_files / "results_effect_of_o_joint_xy_marginal.csv", index=False)

    summary_list_o = []
    for o in sorted(results_o["test_sample_size_o"].unique()):
        df_o = results_o[results_o["test_sample_size_o"] == o]
        tab = summarize_methods(df_o, alpha=alpha, number_test_groups=number_test_groups)
        tab["test_sample_size_o"] = o
        summary_list_o.append(tab)
    summary_o = pd.concat(summary_list_o, ignore_index=True)
    summary_o.to_csv(dir_summ / "summary_effect_of_o_joint_xy_marginal.csv", index=False)

    generate_requested_plots_from_results(results_o, alpha=alpha, dir_plots=dir_plots)

    make_effect_of_o_plots(
        csv_path=dir_files / "results_effect_of_o_joint_xy_marginal.csv",
        output_dir=dir_plots,
        lam=lambda_dummy,
        alpha=alpha,
    )

    print("\n" + "=" * 96)
    print("MARGINAL DGP RUN COMPLETED (FIXED GROUP SIZE)")
    print("=" * 96)
    print(f"Raw results: {dir_files}")
    print(f"Summary:     {dir_summ}")
    print(f"Plots:       {dir_plots}")


if __name__ == "__main__":
    main()
