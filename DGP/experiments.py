"""
Experiment Runner

This module contains functions for running experiments comparing different
hierarchical conformal prediction methods.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from DGP.data_generation import generate_calibration_data, generate_test_group
from .baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius
)
from .hcp_plus import compute_hcp_plus_interval
from .hcp_sample import compute_hcp_sample_interval
from scores import absolute_residual_score


def run_one_experiment(number_groups_k, lambda_Poisson, dgp_specification,
                       o_observed, alpha, number_subsampling_repetitions,
                       alpha_selection, number_test_groups,
                       mu_method_baseline, mu_method_hcp):
    """
    Run one experiment comparing different methods.

    Parameters:
    -----------
    number_groups_k : int
        Number of calibration groups
    lambda_Poisson : float
        Poisson parameter for group sizes
    dgp_specification : dict
        DGP specification
    o_observed : int
        Number of observed points in test group
    alpha : float
        Miscoverage level
    number_subsampling_repetitions : int
        Number of repetitions for repeated subsampling
    alpha_selection : float
        Selection parameter
    number_test_groups : int
        Number of test groups to evaluate
    mu_method_baseline : dict
        μ-method for baseline methods
    mu_method_hcp : dict
        μ-method for HCP++ and HCP.sample

    Returns:
    --------
    DataFrame : Results for this experiment
    """
    # Generate calibration data
    cal = generate_calibration_data(
        number_groups=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_specification
    )
    U_cal = cal['U_calibration']
    Z_cal = cal['Z_calibration']

    # Split calibration groups for baseline methods
    K0 = number_groups_k // 2
    train_idx = list(range(K0))
    calib_idx = list(range(K0, number_groups_k))

    # Fit baseline model
    model_baseline = mu_method_baseline['fit_global'](
        U_matrix=U_cal,
        Z_list=Z_cal,
        group_index_vector=train_idx
    )

    # Compute scores for baseline methods
    scores_list = []
    for j in calib_idx:
        Zj = Z_cal[j]
        Uj = U_cal[j, :]
        yj = np.array([z['Y'] for z in Zj])
        Xj = np.array([z['X'] for z in Zj])

        muj = np.array([
            mu_method_baseline['predict_global'](
                model_global=model_baseline,
                x_vector=Xj[i],
                u_vector=Uj
            ) for i in range(len(Xj))
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    # Compute interval radii for baseline methods
    T_hcp = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep = compute_repeated_subsampling_interval_radius(
        scores_list, alpha, number_subsampling_repetitions
    )

    # Initialize result arrays
    cov_hcppp = np.zeros(number_test_groups, dtype=bool)
    cov_hcpsamp = np.zeros(number_test_groups, dtype=bool)
    cov_hcp = np.zeros(number_test_groups, dtype=bool)
    cov_pool = np.zeros(number_test_groups, dtype=bool)
    cov_sub = np.zeros(number_test_groups, dtype=bool)
    cov_rep = np.zeros(number_test_groups, dtype=bool)

    wid_hcppp = np.full(number_test_groups, np.nan)
    wid_hcpsamp = np.full(number_test_groups, np.nan)
    wid_hcp = np.full(number_test_groups, np.nan)
    wid_pool = np.full(number_test_groups, np.nan)
    wid_sub = np.full(number_test_groups, np.nan)
    wid_rep = np.full(number_test_groups, np.nan)

    inf_hcppp = 0
    inf_hcpsamp = 0
    inf_hcp = 0
    inf_pool = 0
    inf_sub = 0
    inf_rep = 0

    # Evaluate on test groups
    for t in range(number_test_groups):
        test = generate_test_group(
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_observed=o_observed
        )
        U_test = test['U_test']
        Z_test = test['Z_test']
        N_test = test['N_test']

        Y_test = np.array([z['Y'] for z in Z_test])
        test_index = o_observed  # 0-indexed
        if N_test < test_index + 1:
            continue

        true_target = Y_test[test_index]
        X_target = Z_test[test_index]['X']
        mu_test_hat = mu_method_baseline['predict_global'](
            model_global=model_baseline,
            x_vector=X_target,
            u_vector=U_test[0, :]
        )

        # HCP++
        res_pp = compute_hcp_plus_interval(
            U_calibration=U_cal,
            Z_calibration=Z_cal,
            U_test=U_test,
            Z_test=Z_test,
            o_observed=o_observed,
            alpha=alpha,
            alpha_selection=alpha_selection,
            mu_method=mu_method_hcp
        )
        int_pp = res_pp['interval']
        cov_hcppp[t] = (int_pp[0] <= true_target <= int_pp[1])
        if np.isfinite(int_pp[0]) and np.isfinite(int_pp[1]):
            wid_hcppp[t] = int_pp[1] - int_pp[0]
        else:
            inf_hcppp += 1

        # HCP.sample
        res_hs = compute_hcp_sample_interval(
            U_calibration=U_cal,
            Z_calibration=Z_cal,
            U_test=U_test,
            Z_test=Z_test,
            o_observed=o_observed,
            alpha=alpha,
            test_index_target=test_index,
            alpha_selection=alpha_selection,
            mu_method=mu_method_hcp
        )
        int_hs = res_hs['interval']
        cov_hcpsamp[t] = (int_hs[0] <= true_target <= int_hs[1])
        if np.isfinite(int_hs[0]) and np.isfinite(int_hs[1]):
            wid_hcpsamp[t] = int_hs[1] - int_hs[0]
        else:
            inf_hcpsamp += 1

        # Baseline HCP
        if np.isfinite(T_hcp):
            int_hcp = (mu_test_hat - T_hcp, mu_test_hat + T_hcp)
        else:
            int_hcp = (-np.inf, np.inf)
        cov_hcp[t] = (int_hcp[0] <= true_target <= int_hcp[1])
        if np.isfinite(int_hcp[0]) and np.isfinite(int_hcp[1]):
            wid_hcp[t] = int_hcp[1] - int_hcp[0]
        else:
            inf_hcp += 1

        # Pooling
        if np.isfinite(T_pool):
            int_pool = (mu_test_hat - T_pool, mu_test_hat + T_pool)
        else:
            int_pool = (-np.inf, np.inf)
        cov_pool[t] = (int_pool[0] <= true_target <= int_pool[1])
        if np.isfinite(int_pool[0]) and np.isfinite(int_pool[1]):
            wid_pool[t] = int_pool[1] - int_pool[0]
        else:
            inf_pool += 1

        # Subsampling
        if np.isfinite(T_sub):
            int_sub = (mu_test_hat - T_sub, mu_test_hat + T_sub)
        else:
            int_sub = (-np.inf, np.inf)
        cov_sub[t] = (int_sub[0] <= true_target <= int_sub[1])
        if np.isfinite(int_sub[0]) and np.isfinite(int_sub[1]):
            wid_sub[t] = int_sub[1] - int_sub[0]
        else:
            inf_sub += 1

        # Repeated subsampling
        if np.isfinite(T_rep):
            int_rep = (mu_test_hat - T_rep, mu_test_hat + T_rep)
        else:
            int_rep = (-np.inf, np.inf)
        cov_rep[t] = (int_rep[0] <= true_target <= int_rep[1])
        if np.isfinite(int_rep[0]) and np.isfinite(int_rep[1]):
            wid_rep[t] = int_rep[1] - int_rep[0]
        else:
            inf_rep += 1

    # Return results as DataFrame
    return pd.DataFrame({
        'coverage_hcp_plus': [np.mean(cov_hcppp)],
        'coverage_hcp_sample': [np.mean(cov_hcpsamp)],
        'coverage_hcp': [np.mean(cov_hcp)],
        'coverage_pool': [np.mean(cov_pool)],
        'coverage_sub': [np.mean(cov_sub)],
        'coverage_rep': [np.mean(cov_rep)],
        'width_hcp_plus': [np.nanmedian(wid_hcppp)],
        'width_hcp_sample': [np.nanmedian(wid_hcpsamp)],
        'width_hcp': [np.nanmedian(wid_hcp)],
        'width_pool': [np.nanmedian(wid_pool)],
        'width_sub': [np.nanmedian(wid_sub)],
        'width_rep': [np.nanmedian(wid_rep)],
        'infinite_hcp_plus': [inf_hcppp],
        'infinite_hcp_sample': [inf_hcpsamp],
        'infinite_hcp': [inf_hcp],
        'infinite_pool': [inf_pool],
        'infinite_sub': [inf_sub],
        'infinite_rep': [inf_rep]
    })


def run_experiments_outer(number_experiments, number_groups_k, lambda_Poisson,
                          dgp_specification, o_observed, alpha=0.1,
                          number_subsampling_repetitions=50,
                          alpha_selection=0.1, number_test_groups=100,
                          mu_method_baseline=None, mu_method_hcp=None,
                          show_progress=True):
    """
    Run multiple experiments (outer loop).

    Parameters:
    -----------
    number_experiments : int
        Number of experiments to run
    number_groups_k : int
        Number of calibration groups per experiment
    lambda_Poisson : float
        Poisson parameter for group sizes
    dgp_specification : dict
        DGP specification
    o_observed : int
        Number of observed points in test group
    alpha : float
        Miscoverage level (default: 0.1)
    number_subsampling_repetitions : int
        Number of repetitions for repeated subsampling (default: 50)
    alpha_selection : float
        Selection parameter (default: 0.1)
    number_test_groups : int
        Number of test groups per experiment (default: 100)
    mu_method_baseline : dict
        μ-method for baseline methods
    mu_method_hcp : dict
        μ-method for HCP++ and HCP.sample
    show_progress : bool
        Whether to print progress (default: True)

    Returns:
    --------
    DataFrame : Combined results from all experiments
    """
    if show_progress:
        print(f"Running {number_experiments} experiments sequentially...")

    results_list = []
    for e in range(number_experiments):
        if show_progress and (e + 1) % 5 == 0:
            print(f"  Completed {e + 1}/{number_experiments} experiments")

        res = run_one_experiment(
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_specification,
            o_observed=o_observed,
            alpha=alpha,
            number_subsampling_repetitions=number_subsampling_repetitions,
            alpha_selection=alpha_selection,
            number_test_groups=number_test_groups,
            mu_method_baseline=mu_method_baseline,
            mu_method_hcp=mu_method_hcp
        )
        res['experiment'] = e + 1
        results_list.append(res)

    combined = pd.concat(results_list, ignore_index=True)
    # Reorder columns to put experiment first
    cols = ['experiment'] + [c for c in combined.columns if c != 'experiment']
    return combined[cols]
