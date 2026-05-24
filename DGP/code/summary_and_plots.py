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
    triplets = [
        ('donor-HCP-randomized', 'coverage_donor_hcp_randomized', 'width_donor_hcp_randomized', 'infinite_donor_hcp_randomized'),
        ('donor-HCP-derandomized', 'coverage_donor_hcp_derandomized', 'width_donor_hcp_derandomized', 'infinite_donor_hcp_derandomized'),
        ('sample-HCP-randomized', 'coverage_sample_hcp_randomized', 'width_sample_hcp_randomized', 'infinite_sample_hcp_randomized'),
        ('sample-HCP-derandomized', 'coverage_sample_hcp_derandomized', 'width_sample_hcp_derandomized', 'infinite_sample_hcp_derandomized'),
        ('Std-CP', 'coverage_stdcp', 'width_stdcp', 'infinite_stdcp'),
        ('HCP', 'coverage_hcp', 'width_hcp', 'infinite_hcp'),
        ('Pooling', 'coverage_pool', 'width_pool', 'infinite_pool'),
        ('Subsampling', 'coverage_sub', 'width_sub', 'infinite_sub'),
        ('Repeated', 'coverage_rep', 'width_rep', 'infinite_rep'),
    ]

    available = [t for t in triplets if t[1] in results_df.columns]

    summary_list = []
    n_exp = len(results_df)
    total_intervals = n_exp * number_test_groups

    for method, cov_col, width_col, inf_col in available:
        cov_vec = results_df[cov_col].values
        width_vec = results_df[width_col].values
        inf_vec = results_df[inf_col].values

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
            df_o['coverage_donor_hcp_randomized'].values,
            df_o['coverage_sample_hcp_randomized'].values,
            df_o['coverage_hcp'].values,
            df_o['coverage_pool'].values,
            df_o['coverage_sub'].values,
            df_o['coverage_rep'].values
        ]
        labels = [
            'donor-HCP-randomized', 'sample-HCP-randomized',
            'HCP', 'Pooling', 'Subsample', 'Repeated'
        ]

        ax = axes[idx]
        bp = ax.boxplot(cov_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'moccasin', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
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

        # Prepare data as list of arrays for boxplot (drop inf/nan)
        raw = [
            df_o['width_donor_hcp_randomized'].values,
            df_o['width_sample_hcp_randomized'].values,
            df_o['width_hcp'].values,
            df_o['width_pool'].values,
            df_o['width_sub'].values,
            df_o['width_rep'].values
        ]
        wid_data = [v[np.isfinite(v)] for v in raw]
        labels = [
            'donor-HCP-randomized', 'sample-HCP-randomized',
            'HCP', 'Pooling', 'Subsample', 'Repeated'
        ]

        ax = axes[idx]
        bp = ax.boxplot(wid_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'moccasin', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_yscale('log')
        ax.set_title(f'Interval Width for o = {o}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Median Interval Width (log scale)', fontsize=14)
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
            df_dgp['coverage_donor_hcp_randomized'].values,
            df_dgp['coverage_sample_hcp_randomized'].values,
            df_dgp['coverage_hcp'].values,
            df_dgp['coverage_pool'].values,
            df_dgp['coverage_sub'].values,
            df_dgp['coverage_rep'].values
        ]
        labels = [
            'donor-HCP-randomized', 'sample-HCP-randomized',
            'HCP', 'Pooling', 'Subsample', 'Repeated'
        ]

        ax = axes[idx]
        bp = ax.boxplot(cov_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'moccasin', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
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

        # Prepare data as list of arrays for boxplot (drop inf/nan)
        raw = [
            df_dgp['width_donor_hcp_randomized'].values,
            df_dgp['width_sample_hcp_randomized'].values,
            df_dgp['width_hcp'].values,
            df_dgp['width_pool'].values,
            df_dgp['width_sub'].values,
            df_dgp['width_rep'].values
        ]
        wid_data = [v[np.isfinite(v)] for v in raw]
        labels = [
            'donor-HCP-randomized', 'sample-HCP-randomized',
            'HCP', 'Pooling', 'Subsample', 'Repeated'
        ]

        ax = axes[idx]
        bp = ax.boxplot(wid_data, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'moccasin', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_yscale('log')
        ax.set_title(f'Interval Width, DGP = {name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Median Interval Width (log scale)', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ======================================================================
# CONDITIONAL COVERAGE PLOTS
# ======================================================================

def plot_conditional_coverage_line(results_cond, alpha=0.1, save_path=None):
    """
    Line graph of U-conditional coverage vs U[0], one subplot per o value.

    For each method, draws:
      * a solid / dashed line for the mean coverage across experiments
      * a shaded ±1-std band to convey experiment-to-experiment variation

    This visualises how well each method covers the true outcome as the
    group covariate U[0] moves across its support [0, 1].

    Parameters
    ----------
    results_cond : DataFrame
        Output of ``run_experiments_conditional`` — must contain columns
        ``o_observed``, ``u_grid_val``, and ``coverage_*``.
    alpha : float
        Miscoverage level (draws a horizontal reference line at 1−alpha).
    save_path : str or None
        If given, saves to that path; otherwise calls ``plt.show()``.
    """
    o_vals   = sorted(results_cond['o_observed'].unique())
    n_o      = len(o_vals)
    n_cols   = min(2, n_o)
    n_rows   = (n_o + n_cols - 1) // n_cols

    methods  = [
        'donor-HCP-randomized',
        'sample-HCP-randomized',
        'HCP', 'Pooling', 'Subsampling', 'Repeated'
    ]
    cov_cols = [
        'coverage_donor_hcp_randomized',
        'coverage_sample_hcp_randomized',
        'coverage_hcp', 'coverage_pool', 'coverage_sub', 'coverage_rep'
    ]
    colors   = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#7f7f7f']
    lstyles  = ['-', '-', '--', '--', ':', ':']
    markers  = ['o', 's', '^', 'D', 'v', 'P']
    msize    = [6, 5, 5, 5, 5, 5]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8 * n_cols, 6 * n_rows),
                             squeeze=False)

    for idx, o in enumerate(o_vals):
        row, col = divmod(idx, n_cols)
        ax       = axes[row, col]
        df_o     = results_cond[results_cond['o_observed'] == o]
        u_vals   = sorted(df_o['u_grid_val'].unique())

        for mi, (method, col_name, color, ls) in enumerate(
                zip(methods, cov_cols, colors, lstyles)):
            means, stds = [], []
            for u_g in u_vals:
                v = df_o[df_o['u_grid_val'] == u_g][col_name].values
                means.append(np.mean(v))
                stds.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
            means = np.array(means)
            stds  = np.array(stds)
            ax.plot(u_vals, means, color=color, linestyle=ls,
                    linewidth=2, label=method,
                    marker=markers[mi], markersize=msize[mi],
                    zorder=10 - mi)
            ax.fill_between(u_vals,
                            np.clip(means - stds, 0, 1),
                            np.clip(means + stds, 0, 1),
                            color=color, alpha=0.12)

        ax.axhline(y=1 - alpha, color='black', linestyle='--',
                   linewidth=1.5, label=f'Target ({1 - alpha:.2f})')
        ax.set_title(f'U-Conditional Coverage  (o = {o})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(r'$U_1$ (first group covariate)', fontsize=12)
        ax.set_ylabel('Coverage', fontsize=12)
        ax.set_ylim(-0.02, 1.07)
        ax.legend(fontsize=9, loc='lower center', ncol=4)
        ax.grid(True, alpha=0.3)

    for idx in range(n_o, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



