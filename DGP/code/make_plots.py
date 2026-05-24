"""
Generate effect-of-o boxplots for DGP experiments.

For each lambda, this script creates three plot families (coverage + width):
1) donor_focus: D-HCP and all baselines
2) sample_focus: S-HCP and all baselines

Baselines are shown with black shaded styling across all families.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

# ── Aesthetic constants ──────────────────────────────────────────────────────
O_COLORS = {
    0:  '#2166ac',   # dark blue
    5:  '#4393c3',   # medium blue
    15: '#00B894',   # green-teal
    10: '#f4a582',   # salmon
    20: '#b2182b',   # dark red
    25: '#8E44AD',   # purple
    30: '#C0392B',   # dark orange-red
    35: '#2C3E50',   # slate
    40: '#7F8C8D',   # gray
}

BASELINE_COLOR = '#888888'          # grey fill for baseline boxes
BASELINE_EDGE  = '#444444'          # darker grey edge

METHOD_RENAME = {
    'donor_hcp_randomized':   'D-HCP',
    'sample_hcp_randomized':  'S-HCP',
    'hcp':        'HCP',
    'pool':       'Pooling',
    'sub':        'Subsampling',
    'rep':        'Repeated',
}

BASELINES = ['hcp', 'pool', 'sub', 'rep']

PLOT_FAMILIES = {
    'donor_focus': {
        'title': 'Donor Focus',
        'methods': ['donor_hcp_randomized'],
    },
    'sample_focus': {
        'title': 'Sample Focus',
        'methods': ['sample_hcp_randomized'],
    },
}


def _boxplot_style(ax, data, pos, color, width=0.55, edge_color=None,
                   hatch=None, alpha=0.75, linestyle='-'):
    """Draw a single styled boxplot and return the artist."""
    if edge_color is None:
        edge_color = color
    bp = ax.boxplot(
        data, positions=[pos], widths=width,
        patch_artist=True, manage_ticks=False,
        medianprops  = dict(color='black', linewidth=1.4),
        whiskerprops = dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        capprops     = dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        flierprops   = dict(marker='.', color=edge_color, markersize=3, alpha=0.4),
        boxprops     = dict(facecolor=color, alpha=alpha, edgecolor=edge_color,
                            linewidth=1.0, linestyle=linestyle),
    )
    if hatch:
        for patch in bp['boxes']:
            patch.set_hatch(hatch)
    return bp


def make_effect_of_o_plots(csv_path, output_dir, lam, alpha=0.2):
    """
    Read *results_effect_of_o.csv* and produce two figures
    (coverage + width) saved into *output_dir*.
    """
    df = pd.read_csv(csv_path)
    o_values = sorted(df['o_observed'].unique())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    title_fs = 28
    label_fs = 24
    tick_fs = 20
    legend_fs = 20

    for family_name, family_cfg in PLOT_FAMILIES.items():
        proposed_methods = family_cfg['methods']
        family_title = family_cfg['title']

        for metric, ylabel, target_line in [
            ('coverage', 'Coverage', 1 - alpha),
            ('width',    'Interval Width (log scale)', None),
        ]:
            fig, ax = plt.subplots(figsize=(16.0, 6.2))

            pos = 0          # running x position
            tick_positions = []
            tick_labels    = []

            # Proposed methods (colour = o value)
            for m_key in proposed_methods:
                col = f'{metric}_{m_key}'
                group_start = pos
                for o_val in o_values:
                    vals = df.loc[df['o_observed'] == o_val, col].dropna().values
                    if len(vals) == 0:
                        pos += 1
                        continue
                    color = O_COLORS.get(o_val, '#555555')
                    _boxplot_style(ax, vals, pos, color, width=0.58)
                    pos += 1
                group_end = pos - 1
                mid = (group_start + group_end) / 2
                tick_positions.append(mid)
                tick_labels.append(METHOD_RENAME[m_key])
                pos += 1.2

            # Baselines (black shaded, one box each)
            baseline_step = 3.4
            for m_key in BASELINES:
                col = f'{metric}_{m_key}'
                vals = df[col].dropna().values
                _boxplot_style(
                    ax,
                    vals,
                    pos,
                    color='#111111',
                    width=0.58,
                    edge_color='#111111',
                    alpha=0.28,
                    linestyle='--',
                    hatch='///',
                )
                tick_positions.append(pos)
                tick_labels.append(METHOD_RENAME[m_key])
                pos += baseline_step

            # Axes and grid
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=tick_fs, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=label_fs)
            ax.set_xlim(-0.8, pos - 0.4)
            ax.grid(axis='y', alpha=0.25, linewidth=0.6)
            ax.tick_params(axis='y', labelsize=tick_fs)

            if target_line is not None:
                ax.axhline(target_line, color='black', linewidth=1.2,
                           linestyle='--', zorder=0, alpha=0.55)
                ax.text(pos - 0.6, target_line + 0.008,
                        f'{target_line:.0%}', fontsize=8.5, ha='right',
                        va='bottom', color='black', alpha=0.7)

            if metric == 'coverage':
                ax.set_ylim(0.20, 1.05)
            else:
                ax.set_yscale('log')
                ax.set_ylim(bottom=0.20)

            # Legend
            legend_handles = []
            for o_val in o_values:
                color = O_COLORS.get(o_val, '#555555')
                legend_handles.append(
                    mpatches.Patch(facecolor=color, alpha=0.75,
                                   edgecolor=color,
                                   label=f'o = {o_val}'))
            legend_handles.append(
                mpatches.Patch(facecolor='#111111', alpha=0.28,
                               edgecolor='#111111', hatch='///',
                               linestyle='--', label='Baselines'))
            if target_line is not None:
                legend_handles.append(
                    mlines.Line2D([], [], color='black', linewidth=1.2,
                                  linestyle='--', alpha=0.55,
                                  label=f'Target ({target_line:.0%})'))

            ax.legend(handles=legend_handles, loc='lower left',
                      fontsize=legend_fs, framealpha=0.85, edgecolor='#cccccc',
                      ncol=2)

            ax.set_title(
                f'{ylabel}  —  {family_title}  —  λ = {lam}',
                fontsize=title_fs, fontweight='bold', pad=12)

            plt.tight_layout()
            fname = output_dir / f'effect_of_o_{metric}_{family_name}_lambda{lam}.pdf'
            fig.savefig(str(fname), bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {fname}')


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    for lam in [5, 15]:
        csv = base_dir / 'results' / 'files' / f'lambda_{lam}' / 'results_effect_of_o.csv'
        out = base_dir / 'plots' / f'lambda_{lam}'
        print(f'\n=== Lambda = {lam} ===')
        make_effect_of_o_plots(csv, out, lam)
    print('\nDone.')
