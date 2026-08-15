"""
Generate plots for ACS and BP experimental results.

Creates side-by-side subplots showing:
- Coverage by method and percentile
- Width by method and percentile
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_experiment_results(summary_csv, output_path, title, target_coverage, ylabel_width="Width"):
    """
    Create a 2-subplot figure showing coverage and width by method and percentile.

    Parameters:
    -----------
    summary_csv : str
        Path to summary CSV file
    output_path : str
        Path to save the plot
    title : str
        Title for the overall figure
    target_coverage : float
        Target coverage level (e.g., 0.9)
    ylabel_width : str
        Label for width y-axis (default: "Width")
    """
    # Read summary data
    df = pd.read_csv(summary_csv)
    df = df.set_index('method')

    # Define percentiles and their visual properties
    percentiles = [0, 25, 50, 75]
    markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Methods (x-axis) - reorder to desired sequence
    desired_order = ['HCP.sample', 'HCP++', 'HCP', 'Pooling', 'Subsampling', 'Repeated']
    # Keep only methods that exist in the data
    methods = [m for m in desired_order if m in df.index]
    # Add any methods not in desired order (shouldn't happen, but defensive)
    for m in df.index:
        if m not in methods:
            methods.append(m)

    # Reorder dataframe
    df = df.reindex(methods)
    x_pos = np.arange(len(methods))

    # --- Subplot 1: Coverage ---
    for i, pct in enumerate(percentiles):
        col_name = str(pct)
        if col_name in df.columns:
            ax1.plot(x_pos, df[col_name].values,
                    marker=markers[i], markersize=8, linewidth=2,
                    color=colors[i], label=f'{pct}th percentile')

    # Add target coverage line
    ax1.axhline(y=target_coverage, color='black', linestyle='--',
                linewidth=1.5, label=f'Target ({target_coverage:.0%})', alpha=0.7)

    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax1.set_title('Coverage by Method and Percentile', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.set_ylim([0, 1.05])

    # --- Subplot 2: Width ---
    for i, pct in enumerate(percentiles):
        col_name = f'width_{pct}'
        if col_name in df.columns:
            ax2.plot(x_pos, df[col_name].values,
                    marker=markers[i], markersize=8, linewidth=2,
                    color=colors[i], label=f'{pct}th percentile')

    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel(ylabel_width, fontsize=12, fontweight='bold')
    ax2.set_title('Interval Width by Method and Percentile', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(loc='best', framealpha=0.9)

    # Set y-axis to start from 0 for better comparison
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim([0, ymax * 1.1])

    # Overall title
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate plots for ACS and BP results')
    parser.add_argument('--acs', action='store_true', help='Generate ACS plots')
    parser.add_argument('--bp', action='store_true', help='Generate BP plots')
    parser.add_argument('--all', action='store_true', help='Generate all plots')

    args = parser.parse_args()

    # If no specific option, do all
    if not (args.acs or args.bp or args.all):
        args.all = True

    base_dir = Path(__file__).parent

    if args.acs or args.all:
        print("\n" + "="*80)
        print("Generating ACS plots...")
        print("="*80)

        acs_summary = base_dir / 'acs' / 'results' / 'acs_summary.csv'
        acs_plots_dir = base_dir / 'acs' / 'plots'
        acs_plots_dir.mkdir(parents=True, exist_ok=True)
        acs_output = acs_plots_dir / 'acs_results.png'

        plot_experiment_results(
            summary_csv=str(acs_summary),
            output_path=str(acs_output),
            title='ACS Income Prediction Results',
            target_coverage=0.9,
            ylabel_width='Width (log income units)'
        )

    if args.bp or args.all:
        print("\n" + "="*80)
        print("Generating BP plots...")
        print("="*80)

        bp_summary = base_dir / 'blood_pressure' / 'results' / 'bp_summary.csv'
        bp_plots_dir = base_dir / 'blood_pressure' / 'plots'
        bp_plots_dir.mkdir(parents=True, exist_ok=True)
        bp_output = bp_plots_dir / 'bp_results.png'

        plot_experiment_results(
            summary_csv=str(bp_summary),
            output_path=str(bp_output),
            title='Blood Pressure Prediction Results',
            target_coverage=0.8,
            ylabel_width='Width (mmHg)'
        )

    print("\n" + "="*80)
    print("All plots generated successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
