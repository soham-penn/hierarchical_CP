#!/usr/bin/env python3
"""
Plot ACS true marginal results INCLUDING pooled-income quantile baseline.

This script creates plots comparing:
- D-HCP (varying o)
- HCP baseline
- Pooled-Income-Quantile baseline (o-independent)

Generates:
1. Coverage vs nominal coverage
2. Width vs nominal coverage
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


# -----------------------------------------------------------------------------
# Plotting configuration
# -----------------------------------------------------------------------------
O_COLORS = {
    0: "#0072B2",
    5: "#00A1D5",
    10: "#00B894",
    15: "#F39C12",
    20: "#E74C3C",
}

FONT_TICK = 24
FONT_LABEL = 24
FONT_TITLE = 26
FONT_LEGEND = 22


def load_results(alpha, base_dir):
    """Load ACS true marginal results for a given alpha."""
    alpha_str = f"alpha{int(alpha*100):02d}"
    results_dir = base_dir / f"acs/results/true_marginal_{alpha_str}"

    # Load main experiment results
    main_file = results_dir / f"acs_true_marg_{alpha_str}_detailed.csv"
    if not main_file.exists():
        print(f"Warning: Main results not found: {main_file}")
        return None, None

    df_main = pd.read_csv(main_file)

    # Load pooled-quantile results (duplicated version for plotting)
    pooled_file = results_dir / f"acs_true_marg_pooled_quantile_{alpha_str}_detailed.csv"
    if not pooled_file.exists():
        print(f"Warning: Pooled-quantile results not found: {pooled_file}")
        df_pooled = None
    else:
        df_pooled = pd.read_csv(pooled_file)

    return df_main, df_pooled


def compute_coverage_stats(df, method, o=None):
    """Compute coverage mean and binomial SE."""
    if o is None:
        d = df[df['method'] == method]
    else:
        d = df[(df['method'] == method) & (df['o'] == o)]

    cov_vals = d['coverage'].dropna().values
    n = len(cov_vals)

    if n == 0:
        return {'mean': np.nan, 'se': np.nan, 'n': 0}

    p = np.mean(cov_vals)
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0

    return {'mean': p, 'se': se, 'n': n}


def compute_width_stats(df, method, o=None, metric='width_income'):
    """Compute width median and IQR."""
    if o is None:
        d = df[df['method'] == method]
    else:
        d = df[(df['method'] == method) & (df['o'] == o)]

    vals = d[metric].dropna().values
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return {'median': np.nan, 'q25': np.nan, 'q75': np.nan, 'n': 0}

    return {
        'median': np.median(vals),
        'q25': np.percentile(vals, 25),
        'q75': np.percentile(vals, 75),
        'n': len(vals),
    }


def plot_coverage_vs_nominal(df_main, df_pooled, alpha, output_file):
    """Plot coverage vs nominal coverage with pooled-quantile baseline."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_alpha(0.0)

    nominal_coverage = 1 - alpha
    o_vals = sorted([o for o in df_main['o'].unique() if o >= 0])

    # Plot D-HCP for different o values
    dhcp_coverages = []
    dhcp_ses = []
    for o in o_vals:
        stats = compute_coverage_stats(df_main, 'Donor-HCP', o)
        if not np.isnan(stats['mean']):
            dhcp_coverages.append(stats['mean'])
            dhcp_ses.append(stats['se'])
            ax.errorbar(
                nominal_coverage, stats['mean'],
                yerr=stats['se'],
                fmt='o', color=O_COLORS[o], markersize=10,
                capsize=5, capthick=2, elinewidth=2,
                label=f'D-HCP ($o={o}$)',
                zorder=3,
            )

    # Plot HCP baseline (o=0 only, since it doesn't use test group)
    hcp_stats = compute_coverage_stats(df_main, 'HCP', 0)
    if not np.isnan(hcp_stats['mean']):
        ax.errorbar(
            nominal_coverage, hcp_stats['mean'],
            yerr=hcp_stats['se'],
            fmt='s', color='#333333', markersize=10,
            capsize=5, capthick=2, elinewidth=2,
            label='HCP',
            zorder=3,
        )

    # Plot Pooled-Income-Quantile baseline (o-independent)
    if df_pooled is not None:
        pooled_stats = compute_coverage_stats(df_pooled, 'Pooled-Income-Quantile', 0)  # Use o=0 from duplicated
        if not np.isnan(pooled_stats['mean']):
            ax.errorbar(
                nominal_coverage, pooled_stats['mean'],
                yerr=pooled_stats['se'],
                fmt='^', color='#E74C3C', markersize=12,
                capsize=5, capthick=2, elinewidth=2,
                label='Pooled-Income-Quantile',
                zorder=3,
            )

    # Reference line y = x
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Nominal', zorder=1)

    # Formatting
    ax.set_xlabel(f'Nominal Coverage (1 - α)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Empirical Coverage', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_title(f'ACS True Marginal: Coverage vs Nominal (α={alpha})',
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_facecolor('white')

    # Set axis limits
    ax.set_xlim(nominal_coverage - 0.05, nominal_coverage + 0.05)
    ax.set_ylim(max(0.5, nominal_coverage - 0.15), 1.02)

    ax.legend(loc='lower right', fontsize=FONT_LEGEND, frameon=True,
              fancybox=False, edgecolor='#222222', framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_width_vs_o(df_main, df_pooled, alpha, output_file):
    """Plot width vs o with pooled-quantile baseline."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_alpha(0.0)

    o_vals = sorted([o for o in df_main['o'].unique() if o >= 0])

    # Plot D-HCP widths vs o
    dhcp_medians = []
    dhcp_o = []
    for o in o_vals:
        stats = compute_width_stats(df_main, 'Donor-HCP', o)
        if not np.isnan(stats['median']):
            dhcp_o.append(o)
            dhcp_medians.append(stats['median'])
            ax.plot(o, stats['median'], 'o', color=O_COLORS[o], markersize=10, zorder=3)

    if len(dhcp_o) > 0:
        ax.plot(dhcp_o, dhcp_medians, '-', color='#0072B2', linewidth=2,
                label='D-HCP', alpha=0.7, zorder=2)

    # Plot HCP baseline (horizontal line, o-independent)
    hcp_stats = compute_width_stats(df_main, 'HCP', 0)
    if not np.isnan(hcp_stats['median']):
        ax.axhline(hcp_stats['median'], color='#333333', linestyle='--',
                   linewidth=2, label='HCP', zorder=2)

    # Plot Pooled-Income-Quantile baseline (horizontal line, o-independent)
    if df_pooled is not None:
        pooled_stats = compute_width_stats(df_pooled, 'Pooled-Income-Quantile', 0)
        if not np.isnan(pooled_stats['median']):
            ax.axhline(pooled_stats['median'], color='#E74C3C', linestyle=':',
                       linewidth=2.5, label='Pooled-Income-Quantile', zorder=2)

    # Formatting
    ax.set_xlabel('$o$ (Test Group History Length)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Median Interval Width (Income $)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_title(f'ACS True Marginal: Width vs $o$ (α={alpha})',
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_facecolor('white')

    ax.set_xlim(-1, max(o_vals) + 1)
    ax.legend(loc='upper right', fontsize=FONT_LEGEND, frameon=True,
              fancybox=False, edgecolor='#222222', framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot ACS true marginal with pooled-quantile baseline')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha value to plot (default: 0.1)')
    args = parser.parse_args()

    print("=" * 70)
    print("ACS True Marginal: Plots with Pooled-Income-Quantile Baseline")
    print("=" * 70)

    # Paths
    base_dir = Path(__file__).parent.parent
    alpha_str = f"alpha{int(args.alpha*100):02d}"
    output_dir = base_dir / f"acs/NEW_PLOTS/true_marginal_with_pooled_quantile_{alpha_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results for alpha={args.alpha}...")
    df_main, df_pooled = load_results(args.alpha, base_dir)

    if df_main is None:
        print("Error: Main results not found. Please run experiments first.")
        return

    print(f"  Main results: {len(df_main)} rows")
    if df_pooled is not None:
        print(f"  Pooled-quantile results: {len(df_pooled)} rows")
    else:
        print("  Warning: Pooled-quantile results not found. Run acs_true_marginal_add_pooled_quantile.py first.")

    print(f"\nGenerating plots...")

    # Plot 1: Coverage vs nominal
    plot_coverage_vs_nominal(
        df_main, df_pooled, args.alpha,
        output_dir / f"acs_true_marg_coverage_vs_nominal_{alpha_str}_with_pooled.pdf"
    )

    # Plot 2: Width vs o
    plot_width_vs_o(
        df_main, df_pooled, args.alpha,
        output_dir / f"acs_true_marg_width_vs_o_{alpha_str}_with_pooled.pdf"
    )

    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
