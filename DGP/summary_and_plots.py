"""
Summary and Plotting Functions

This module contains functions for summarizing experimental results and
creating plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summarize_methods(results_df, alpha, number_test_groups):
    """
    Summarize results across methods.

    Parameters:
    -----------
    results_df : DataFrame
        Results from experiments
    alpha : float
        Miscoverage level
    number_test_groups : int
        Number of test groups per experiment

    Returns:
    --------
    DataFrame : Summary statistics for each method
    """
    methods = ['HCP++', 'HCP.sample', 'HCP', 'Pooling', 'Subsampling', 'Repeated']
    cov_cols = ['coverage_hcp_plus', 'coverage_hcp_sample',
                'coverage_hcp', 'coverage_pool',
                'coverage_sub', 'coverage_rep']
    width_cols = ['width_hcp_plus', 'width_hcp_sample',
                  'width_hcp', 'width_pool',
                  'width_sub', 'width_rep']
    inf_cols = ['infinite_hcp_plus', 'infinite_hcp_sample',
                'infinite_hcp', 'infinite_pool',
                'infinite_sub', 'infinite_rep']

    summary_list = []
    n_exp = len(results_df)
    total_intervals = n_exp * number_test_groups

    for i, method in enumerate(methods):
        cov_vec = results_df[cov_cols[i]].values
        width_vec = results_df[width_cols[i]].values
        inf_vec = results_df[inf_cols[i]].values

        cov_mean = np.mean(cov_vec)
        cov_std = np.std(cov_vec, ddof=1)

        width_median = np.nanmedian(width_vec)
        width_std = np.nanstd(width_vec, ddof=1)

        inf_total = np.sum(inf_vec)
        inf_perc = 100 * inf_total / total_intervals

        summary_list.append({
            'Method': method,
            'Coverage_Mean': cov_mean,
            'Coverage_Std': cov_std,
            'Width_Median_Finite': width_median,
            'Width_Std_Finite': width_std,
            'Infinite_Intervals': int(inf_total),
            'Infinite_Percentage': inf_perc,
            'Target_Coverage': 1 - alpha,
            'Coverage_Difference': cov_mean - (1 - alpha)
        })

    return pd.DataFrame(summary_list)


def plot_effect_of_o_coverage_2x2(results_o, alpha=0.1, save_path=None):
    """
    Plot coverage results for different values of o in a 2x2 grid.

    Parameters:
    -----------
    results_o : DataFrame
        Results with 'test_sample_size_o' column
    alpha : float
        Miscoverage level (default: 0.1)
    save_path : str or None
        Path to save the plot (if None, displays instead)
    """
    o_vals = sorted(results_o['test_sample_size_o'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, o in enumerate(o_vals):
        if idx >= 4:
            break

        df_o = results_o[results_o['test_sample_size_o'] == o]

        # Prepare data as list of arrays for boxplot
        cov_data = [
            df_o['coverage_hcp_plus'].values,
            df_o['coverage_hcp_sample'].values,
            df_o['coverage_hcp'].values,
            df_o['coverage_pool'].values,
            df_o['coverage_sub'].values,
            df_o['coverage_rep'].values
        ]
        labels = ['HCP++', 'HCP.sample', 'HCP', 'Pooling', 'Subsample', 'Repeated']

        ax = axes[idx]
        bp = ax.boxplot(cov_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.axhline(y=1 - alpha, linestyle='--', linewidth=2, color='red',
                  label=f'Target ({1-alpha:.1f})')
        ax.set_title(f'Coverage for o = {o}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Estimated Coverage', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        ax.set_ylim([0, 1.05])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_effect_of_o_width_2x2(results_o, save_path=None):
    """
    Plot width results for different values of o in a 2x2 grid.

    Parameters:
    -----------
    results_o : DataFrame
        Results with 'test_sample_size_o' column
    save_path : str or None
        Path to save the plot (if None, displays instead)
    """
    o_vals = sorted(results_o['test_sample_size_o'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, o in enumerate(o_vals):
        if idx >= 4:
            break

        df_o = results_o[results_o['test_sample_size_o'] == o]

        # Prepare data as list of arrays for boxplot
        wid_data = [
            df_o['width_hcp_plus'].values,
            df_o['width_hcp_sample'].values,
            df_o['width_hcp'].values,
            df_o['width_pool'].values,
            df_o['width_sub'].values,
            df_o['width_rep'].values
        ]
        labels = ['HCP++', 'HCP.sample', 'HCP', 'Pooling', 'Subsample', 'Repeated']

        ax = axes[idx]
        bp = ax.boxplot(wid_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'Interval Width for o = {o}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Median Interval Width', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_effect_of_meanvar_coverage_1x2(results_mv, alpha=0.1, save_path=None):
    """
    Plot coverage results for different DGPs in a 1x2 grid.

    Parameters:
    -----------
    results_mv : DataFrame
        Results with 'dgp_name' column
    alpha : float
        Miscoverage level (default: 0.1)
    save_path : str or None
        Path to save the plot (if None, displays instead)
    """
    dgp_vals = sorted(results_mv['dgp_name'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, name in enumerate(dgp_vals):
        if idx >= 2:
            break

        df_dgp = results_mv[results_mv['dgp_name'] == name]

        # Prepare data as list of arrays for boxplot
        cov_data = [
            df_dgp['coverage_hcp_plus'].values,
            df_dgp['coverage_hcp_sample'].values,
            df_dgp['coverage_hcp'].values,
            df_dgp['coverage_pool'].values,
            df_dgp['coverage_sub'].values,
            df_dgp['coverage_rep'].values
        ]
        labels = ['HCP++', 'HCP.sample', 'HCP', 'Pooling', 'Subsample', 'Repeated']

        ax = axes[idx]
        bp = ax.boxplot(cov_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.axhline(y=1 - alpha, linestyle='--', linewidth=2, color='red',
                  label=f'Target ({1-alpha:.1f})')
        ax.set_title(f'Coverage, DGP = {name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Estimated Coverage', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        ax.set_ylim([0, 1.05])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_effect_of_meanvar_width_1x2(results_mv, save_path=None):
    """
    Plot width results for different DGPs in a 1x2 grid.

    Parameters:
    -----------
    results_mv : DataFrame
        Results with 'dgp_name' column
    save_path : str or None
        Path to save the plot (if None, displays instead)
    """
    dgp_vals = sorted(results_mv['dgp_name'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, name in enumerate(dgp_vals):
        if idx >= 2:
            break

        df_dgp = results_mv[results_mv['dgp_name'] == name]

        # Prepare data as list of arrays for boxplot
        wid_data = [
            df_dgp['width_hcp_plus'].values,
            df_dgp['width_hcp_sample'].values,
            df_dgp['width_hcp'].values,
            df_dgp['width_pool'].values,
            df_dgp['width_sub'].values,
            df_dgp['width_rep'].values
        ]
        labels = ['HCP++', 'HCP.sample', 'HCP', 'Pooling', 'Subsample', 'Repeated']

        ax = axes[idx]
        bp = ax.boxplot(wid_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(f'Interval Width, DGP = {name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Median Interval Width', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
