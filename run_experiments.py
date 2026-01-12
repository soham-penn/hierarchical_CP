"""
Main Experiment Script for Hierarchical Conformal Prediction

This script runs experiments comparing different hierarchical conformal prediction methods:
- HCP++ (HCP.plus)
- HCP.sample
- HCP (baseline)
- Pooling
- Subsampling
- Repeated subsampling

Results are saved to DGP/resultsDGP/files/ and summaries to DGP/resultsDGP/
Plots are saved to DGP/resultsDGP/plots/
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import DGP and data generation
from DGP import (
    create_dgp_specification_default,
    create_dgp_specification_nonlinear
)

# Import methods
from methods import (
    create_mu_method_random_forest_offset,
    create_mu_method_random_forest_global_only,
    run_experiments_outer
)

# Import summary and plotting
from summary_and_plots import (
    summarize_methods,
    plot_effect_of_o_coverage_2x2,
    plot_effect_of_o_width_2x2,
    plot_effect_of_meanvar_coverage_1x2,
    plot_effect_of_meanvar_width_1x2
)


def run_experiments_effect_of_o(o_vector=[1, 15, 20, 50],
                                number_experiments=25,
                                number_groups_k=20,
                                lambda_Poisson=20,
                                alpha=0.1,
                                number_subsampling_repetitions=50,
                                alpha_selection=0.5,
                                number_test_groups=100,
                                ntree_rf=50,
                                nodesize_rf=5):
    """
    Run experiments varying the number of observed points o.

    Parameters:
    -----------
    o_vector : list
        Different values of o to test
    number_experiments : int
        Number of experiments per configuration
    number_groups_k : int
        Number of calibration groups
    lambda_Poisson : float
        Poisson parameter for group sizes
    alpha : float
        Miscoverage level
    number_subsampling_repetitions : int
        Number of repetitions for repeated subsampling
    alpha_selection : float
        Selection parameter
    number_test_groups : int
        Number of test groups per experiment
    ntree_rf : int
        Number of trees in random forest
    nodesize_rf : int
        Minimum node size in random forest

    Returns:
    --------
    DataFrame : Combined results
    """
    d = 5
    dgp_o = create_dgp_specification_default(
        dimension=d,
        u_min=0,
        u_max=1
    )

    mu_baseline = create_mu_method_random_forest_global_only(
        ntree=ntree_rf, mtry=None, nodesize=nodesize_rf
    )
    mu_hcp = create_mu_method_random_forest_offset(
        ntree=ntree_rf, mtry=None, nodesize=nodesize_rf
    )

    results_list = []

    for o_cur in o_vector:
        print(f"\nRunning experiments for o = {o_cur}")
        res = run_experiments_outer(
            number_experiments=number_experiments,
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp_o,
            o_observed=o_cur,
            alpha=alpha,
            number_subsampling_repetitions=number_subsampling_repetitions,
            alpha_selection=alpha_selection,
            number_test_groups=number_test_groups,
            mu_method_baseline=mu_baseline,
            mu_method_hcp=mu_hcp,
            show_progress=True
        )
        res['test_sample_size_o'] = o_cur
        results_list.append(res)

    return pd.concat(results_list, ignore_index=True)


def run_experiments_effect_of_mean_variance(number_experiments=25,
                                           number_groups_k=20,
                                           lambda_Poisson=20,
                                           o_observed=15,
                                           alpha=0.1,
                                           number_subsampling_repetitions=50,
                                           alpha_selection=0.5,
                                           number_test_groups=100,
                                           ntree_rf=50,
                                           nodesize_rf=5):
    """
    Run experiments comparing different DGPs (default vs nonlinear).

    Parameters:
    -----------
    number_experiments : int
        Number of experiments per DGP
    number_groups_k : int
        Number of calibration groups
    lambda_Poisson : float
        Poisson parameter for group sizes
    o_observed : int
        Number of observed points in test group
    alpha : float
        Miscoverage level
    number_subsampling_repetitions : int
        Number of repetitions for repeated subsampling
    alpha_selection : float
        Selection parameter
    number_test_groups : int
        Number of test groups per experiment
    ntree_rf : int
        Number of trees in random forest
    nodesize_rf : int
        Minimum node size in random forest

    Returns:
    --------
    DataFrame : Combined results
    """
    d = 5
    dgp_linearish = create_dgp_specification_default(
        dimension=d,
        u_min=0,
        u_max=1
    )
    dgp_nonlinear = create_dgp_specification_nonlinear(
        dimension=d,
        u_min=0,
        u_max=1
    )

    mu_baseline = create_mu_method_random_forest_global_only(
        ntree=ntree_rf, mtry=None, nodesize=nodesize_rf
    )
    mu_hcp = create_mu_method_random_forest_offset(
        ntree=ntree_rf, mtry=None, nodesize=nodesize_rf
    )

    dgp_list = {
        'linearish': dgp_linearish,
        'nonlinear': dgp_nonlinear
    }

    all_results = []

    for name, dgp in dgp_list.items():
        print(f"\nRunning experiments for DGP = {name}")
        res = run_experiments_outer(
            number_experiments=number_experiments,
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp,
            o_observed=o_observed,
            alpha=alpha,
            number_subsampling_repetitions=number_subsampling_repetitions,
            alpha_selection=alpha_selection,
            number_test_groups=number_test_groups,
            mu_method_baseline=mu_baseline,
            mu_method_hcp=mu_hcp,
            show_progress=True
        )
        res['dgp_name'] = name
        all_results.append(res)

    return pd.concat(all_results, ignore_index=True)


def main():
    """
    Main function to run all experiments and generate results.
    """
    # Set random seed for reproducibility
    np.random.seed(123)

    # Create output directories if they don't exist
    Path("DGP/resultsDGP/files").mkdir(parents=True, exist_ok=True)
    Path("DGP/resultsDGP/plots").mkdir(parents=True, exist_ok=True)

    alpha = 0.1
    number_test_groups = 100

    print("=" * 80)
    print("HIERARCHICAL CONFORMAL PREDICTION EXPERIMENTS")
    print("=" * 80)

    # ----- Effect of o -----
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Effect of o (number of observed points)")
    print("=" * 80)

    results_o = run_experiments_effect_of_o(
        o_vector=[1, 15, 20, 50],
        number_experiments=25,
        number_groups_k=20,
        lambda_Poisson=20,
        alpha=alpha,
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=number_test_groups,
        ntree_rf=50,
        nodesize_rf=5
    )

    # Save raw results
    results_o.to_csv("DGP/resultsDGP/files/results_effect_of_o.csv", index=False)
    print("\nSaved raw results to: DGP/resultsDGP/files/results_effect_of_o.csv")

    # Generate and save summary
    summary_list_o = []
    for o in sorted(results_o['test_sample_size_o'].unique()):
        df_o = results_o[results_o['test_sample_size_o'] == o]
        tab = summarize_methods(df_o, alpha=alpha, number_test_groups=number_test_groups)
        tab['test_sample_size_o'] = o
        summary_list_o.append(tab)

    summary_o = pd.concat(summary_list_o, ignore_index=True)
    summary_o.to_csv("DGP/resultsDGP/summary_effect_of_o.csv", index=False)
    print("Saved summary to: DGP/resultsDGP/summary_effect_of_o.csv")

    # Generate and save plots
    print("\nGenerating plots for effect of o...")
    plot_effect_of_o_coverage_2x2(
        results_o, alpha=alpha,
        save_path="DGP/resultsDGP/plots/effect_of_o_coverage.png"
    )
    print("Saved plot to: DGP/resultsDGP/plots/effect_of_o_coverage.png")

    plot_effect_of_o_width_2x2(
        results_o,
        save_path="DGP/resultsDGP/plots/effect_of_o_width.png"
    )
    print("Saved plot to: DGP/resultsDGP/plots/effect_of_o_width.png")

    # ----- Effect of mean and variance -----
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Effect of mean and variance (DGP comparison)")
    print("=" * 80)

    results_mv = run_experiments_effect_of_mean_variance(
        number_experiments=25,
        number_groups_k=20,
        lambda_Poisson=20,
        o_observed=15,
        alpha=alpha,
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=number_test_groups,
        ntree_rf=50,
        nodesize_rf=5
    )

    # Save raw results
    results_mv.to_csv("DGP/resultsDGP/files/results_effect_of_mean_variance.csv", index=False)
    print("\nSaved raw results to: DGP/resultsDGP/files/results_effect_of_mean_variance.csv")

    # Generate and save summary
    summary_list_mv = []
    for name in results_mv['dgp_name'].unique():
        df_dgp = results_mv[results_mv['dgp_name'] == name]
        tab = summarize_methods(df_dgp, alpha=alpha, number_test_groups=number_test_groups)
        tab['dgp_name'] = name
        summary_list_mv.append(tab)

    summary_mv = pd.concat(summary_list_mv, ignore_index=True)
    summary_mv.to_csv("DGP/resultsDGP/summary_effect_of_mean_variance.csv", index=False)
    print("Saved summary to: DGP/resultsDGP/summary_effect_of_mean_variance.csv")

    # Generate and save plots
    print("\nGenerating plots for effect of mean and variance...")
    plot_effect_of_meanvar_coverage_1x2(
        results_mv, alpha=alpha,
        save_path="DGP/resultsDGP/plots/effect_of_mean_variance_coverage.png"
    )
    print("Saved plot to: DGP/resultsDGP/plots/effect_of_mean_variance_coverage.png")

    plot_effect_of_meanvar_width_1x2(
        results_mv,
        save_path="DGP/resultsDGP/plots/effect_of_mean_variance_width.png"
    )
    print("Saved plot to: DGP/resultsDGP/plots/effect_of_mean_variance_width.png")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print("\nResults summary:")
    print(f"  - Raw results: DGP/resultsDGP/files/")
    print(f"  - Summaries: DGP/resultsDGP/")
    print(f"  - Plots: DGP/resultsDGP/plots/")
    print()


if __name__ == "__main__":
    main()
