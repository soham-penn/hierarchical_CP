"""
Main Experiment Script for Hierarchical Conformal Prediction

This script runs experiments comparing different hierarchical conformal prediction methods:
- donor-HCP randomized / derandomized
- sample-HCP randomized / derandomized
- HCP (baseline)
- Pooling
- Subsampling
- Repeated subsampling

Results are saved to DGP/results/files/ and summaries to DGP/results/
Plots are saved to DGP/plots/
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import DGP and data generation
from DGP import (
    create_dgp_specification_default,
    create_dgp_specification_nonlinear,
    create_dgp_specification_heteroscedastic
)

# Import methods
from methods import (
    create_mu_method_ols_global_only,
    create_mu_method_ols_offset,
)

# Import experiment runner and summary/plotting
from DGP.code.experiments import run_experiments_outer, run_experiments_conditional
from DGP.code.summary_and_plots import (
    summarize_methods,
    plot_effect_of_o_coverage_2x2,
    plot_effect_of_o_width_2x2,
    plot_effect_of_meanvar_coverage_1x2,
    plot_effect_of_meanvar_width_1x2,
    plot_conditional_coverage_line,
)


def run_experiments_effect_of_o(o_vector=[0, 15, 20, 50],
                                target_index=50,
                                number_experiments=25,
                                number_groups_k=12,
                                lambda_Poisson=5,
                                alpha=0.1,
                                number_subsampling_repetitions=50,
                                alpha_selection=0.5,
                                number_test_groups=100):
    """
    Run experiments varying the number of observed points o.

    All o values share the same calibration data and the same test groups.
    The prediction target is always Z_test[target_index], so baseline methods
    produce identical results across o values — making the comparison fair.

    Parameters
    ----------
    o_vector : list of int
        History sizes to evaluate.
    target_index : int
        0-based index of the fixed prediction target in each test group.
    """
    d = 5
    dgp_o = create_dgp_specification_default(dimension=d, u_min=0, u_max=1)
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    print(f"\nRunning experiments for o_values={o_vector}, target_index={target_index}")
    results = run_experiments_outer(
        number_experiments=number_experiments,
        number_groups_k=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp_o,
        o_values=o_vector,
        target_index=target_index,
        alpha=alpha,
        number_subsampling_repetitions=number_subsampling_repetitions,
        alpha_selection=alpha_selection,
        number_test_groups=number_test_groups,
        mu_method_baseline=mu_baseline,
        mu_method_hcp=mu_hcp,
        show_progress=True,
    )
    # Keep test_sample_size_o for backward compat with summary/plot functions
    results['test_sample_size_o'] = results['o_observed']
    return results


def run_experiments_effect_of_mean_variance(number_experiments=25,
                                           number_groups_k=12,
                                           lambda_Poisson=5,
                                           o_observed=15,
                                           alpha=0.1,
                                           number_subsampling_repetitions=50,
                                           alpha_selection=0.5,
                                           number_test_groups=100):
    """
    Run experiments comparing different DGPs (default vs nonlinear).
    """
    d = 5
    dgp_linearish = create_dgp_specification_default(dimension=d, u_min=0, u_max=1)
    dgp_nonlinear = create_dgp_specification_nonlinear(dimension=d, u_min=0, u_max=1)
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp = create_mu_method_ols_offset()

    dgp_list = {'linearish': dgp_linearish, 'nonlinear': dgp_nonlinear}
    all_results = []

    for name, dgp in dgp_list.items():
        print(f"\nRunning experiments for DGP = {name}")
        res = run_experiments_outer(
            number_experiments=number_experiments,
            number_groups_k=number_groups_k,
            lambda_Poisson=lambda_Poisson,
            dgp_specification=dgp,
            o_values=[o_observed],
            target_index=o_observed,
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


def run_experiments_effect_of_o_conditional(
        o_values=None,
        target_index=50,
        u_grid=None,
        number_experiments=25,
        number_groups_k=12,
        lambda_Poisson=5,
        alpha=0.1,
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=50):
    """
    Run U-conditional-coverage experiments varying o.

    For each experiment, test groups are generated with U[0] fixed to each
    point in ``u_grid`` (and U[1:] = 0.5).  X and Y are drawn from
    P(X, Y | U) as usual.  This isolates the effect of the group covariate
    value on coverage — the guarantee that HCP++ / HCP.sample provide.

    Parameters
    ----------
    o_values : list of int
        History sizes to evaluate.
    target_index : int
        0-based index of the fixed prediction target (test group must have
        at least target_index + 1 observations).
    u_grid : array-like or None
        Grid of U[0] values to condition on.  Defaults to 9 evenly-spaced
        points in (0, 1).
    number_test_groups : int
        Test groups per grid point per experiment.
    """
    if o_values is None:
        o_values = [0, 15, 20, 50]
    if u_grid is None:
        u_grid = np.linspace(0.1, 0.9, 9)

    d = 5
    dgp = create_dgp_specification_default(dimension=d, u_min=0, u_max=1)
    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp      = create_mu_method_ols_offset()

    print(f"\nRunning U-conditional experiments: o_values={o_values}, "
          f"target_index={target_index}, u_grid size={len(u_grid)}")
    return run_experiments_conditional(
        number_experiments=number_experiments,
        number_groups_k=number_groups_k,
        lambda_Poisson=lambda_Poisson,
        dgp_specification=dgp,
        o_values=o_values,
        target_index=target_index,
        u_grid=u_grid,
        alpha=alpha,
        number_subsampling_repetitions=number_subsampling_repetitions,
        alpha_selection=alpha_selection,
        number_test_groups=number_test_groups,
        mu_method_baseline=mu_baseline,
        mu_method_hcp=mu_hcp,
        show_progress=True,
    )


def main(lambda_val=20):
    """
    Main function to run all experiments and generate results.

    Parameters
    ----------
    lambda_val : int
        Poisson rate for group sizes. Results are saved under:
        - DGP/results/files/lambda_<lambda_val>/
        - DGP/results/lambda_<lambda_val>/
        - DGP/plots/lambda_<lambda_val>/
    """
    # Set random seed for reproducibility
    np.random.seed(123)

    # Lambda-specific output directories
    tag = f"lambda_{lambda_val}"
    dir_files = Path(f"DGP/results/files/{tag}")
    dir_plots = Path(f"DGP/plots/{tag}")
    dir_summ  = Path(f"DGP/results/{tag}")
    dir_files.mkdir(parents=True, exist_ok=True)
    dir_plots.mkdir(parents=True, exist_ok=True)
    dir_summ.mkdir(parents=True, exist_ok=True)

    alpha = 0.2
    number_test_groups = 100

    # Fixed o values and target_index (same for all lambda settings).
    o_vec = [0, 5, 10, 20]
    target_idx = 20   # predict Z_test[20]; test groups forced to have N ≥ 21

    mean_N = 1 + lambda_val
    print(f"  lambda = {lambda_val}, mean_N ≈ {mean_N}")
    print(f"  target_index = {target_idx}, o_values = {o_vec}")

    print("=" * 80)
    print("HIERARCHICAL CONFORMAL PREDICTION EXPERIMENTS")
    print("=" * 80)

    # ----- Effect of o -----
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Effect of o (number of observed points)")
    print("=" * 80)

    results_o = run_experiments_effect_of_o(
        o_vector=o_vec,
        target_index=target_idx,
        number_experiments=25,
        number_groups_k=12,
        lambda_Poisson=lambda_val,
        alpha=alpha,
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=number_test_groups,
    )

    # Save raw results
    results_o.to_csv(dir_files / "results_effect_of_o.csv", index=False)
    print(f"\nSaved raw results to: {dir_files / 'results_effect_of_o.csv'}")

    # Generate and save summary
    summary_list_o = []
    for o in sorted(results_o['test_sample_size_o'].unique()):
        df_o = results_o[results_o['test_sample_size_o'] == o]
        tab = summarize_methods(df_o, alpha=alpha, number_test_groups=number_test_groups)
        tab['test_sample_size_o'] = o
        summary_list_o.append(tab)

    summary_o = pd.concat(summary_list_o, ignore_index=True)
    summary_o.to_csv(dir_summ / "summary_effect_of_o.csv", index=False)
    print(f"Saved summary to: {dir_summ / 'summary_effect_of_o.csv'}")

    # Generate and save plots
    print("\nGenerating plots for effect of o...")
    plot_effect_of_o_coverage_2x2(
        results_o, alpha=alpha,
        save_path=str(dir_plots / "effect_of_o_coverage.png")
    )
    print(f"Saved plot to: {dir_plots / 'effect_of_o_coverage.png'}")

    plot_effect_of_o_width_2x2(
        results_o,
        save_path=str(dir_plots / "effect_of_o_width.png")
    )
    print(f"Saved plot to: {dir_plots / 'effect_of_o_width.png'}")

    # ----- Effect of mean and variance -----
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Effect of mean and variance (DGP comparison)")
    print("=" * 80)

    results_mv = run_experiments_effect_of_mean_variance(
        number_experiments=25,
        number_groups_k=12,
        lambda_Poisson=lambda_val,
        o_observed=o_vec[-1],
        alpha=alpha,
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=number_test_groups,
    )

    # Save raw results
    results_mv.to_csv(dir_files / "results_effect_of_mean_variance.csv", index=False)
    print(f"\nSaved raw results to: {dir_files / 'results_effect_of_mean_variance.csv'}")

    # Generate and save summary
    summary_list_mv = []
    for name in results_mv['dgp_name'].unique():
        df_dgp = results_mv[results_mv['dgp_name'] == name]
        tab = summarize_methods(df_dgp, alpha=alpha, number_test_groups=number_test_groups)
        tab['dgp_name'] = name
        summary_list_mv.append(tab)

    summary_mv = pd.concat(summary_list_mv, ignore_index=True)
    summary_mv.to_csv(dir_summ / "summary_effect_of_mean_variance.csv", index=False)
    print(f"Saved summary to: {dir_summ / 'summary_effect_of_mean_variance.csv'}")

    # Generate and save plots
    print("\nGenerating plots for effect of mean and variance...")
    plot_effect_of_meanvar_coverage_1x2(
        results_mv, alpha=alpha,
        save_path=str(dir_plots / "effect_of_mean_variance_coverage.png")
    )
    print(f"Saved plot to: {dir_plots / 'effect_of_mean_variance_coverage.png'}")

    plot_effect_of_meanvar_width_1x2(
        results_mv,
        save_path=str(dir_plots / "effect_of_mean_variance_width.png")
    )
    print(f"Saved plot to: {dir_plots / 'effect_of_mean_variance_width.png'}")

    # ----- Conditional coverage -----
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: U-conditional coverage (coverage vs U[0] on a grid)")
    print("=" * 80)
    print()
    print("Design: for each experiment, test groups are generated with U[0]"
          " fixed to each grid point (U[1:] = 0.5); X and Y are drawn from"
          " P(X,Y|U).  Repeating many times yields coverage curves vs U[0]"
          " → line plots with uncertainty bands (± 1 std across experiments).")

    u_grid = np.linspace(0.1, 0.9, 9)

    results_cond = run_experiments_effect_of_o_conditional(
        o_values=o_vec,
        target_index=target_idx,
        u_grid=u_grid,
        number_experiments=25,
        number_groups_k=12,
        lambda_Poisson=lambda_val,
        alpha=alpha,
        number_subsampling_repetitions=50,
        alpha_selection=0.5,
        number_test_groups=50,
    )

    # Save raw results
    results_cond.to_csv(
        dir_files / "results_conditional_coverage.csv", index=False
    )
    print(f"\nSaved raw results to:"
          f" {dir_files / 'results_conditional_coverage.csv'}")

    # Plots
    print("\nGenerating conditional coverage plots …")
    plot_conditional_coverage_line(
        results_cond, alpha=alpha,
        save_path=str(dir_plots / "conditional_coverage.png")
    )
    print(f"Saved plot to: {dir_plots / 'conditional_coverage.png'}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print("\nResults summary:")
    print(f"  - Raw results: {dir_files}/")
    print(f"  - Summaries: {dir_summ}/")
    print(f"  - Plots: {dir_plots}/")
    print()


if __name__ == "__main__":
    import sys
    lam = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(lambda_val=lam)
