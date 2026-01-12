"""
Utility functions for formatting experimental results as markdown tables.
"""

import pandas as pd
import numpy as np


def csv_to_markdown_table(csv_path, md_path, title, target_coverage):
    """
    Convert summary CSV to a formatted markdown table.

    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    md_path : str
        Path to output markdown file
    title : str
        Title for the markdown document
    target_coverage : float
        Target coverage level (e.g., 0.9 for 90%)
    """
    df = pd.read_csv(csv_path)

    # Round numeric columns
    numeric_cols = ['0', '25', '50', '75', 'overall_coverage', 'mean_width',
                    'median_width', 'coverage_diff']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    # Rename columns for better display
    df = df.rename(columns={
        '0': '0th %ile',
        '25': '25th %ile',
        '50': '50th %ile',
        '75': '75th %ile',
        'overall_coverage': 'Overall Coverage',
        'mean_width': 'Mean Width',
        'median_width': 'Median Width',
        'total_infinite': 'Prop Infinite',
        'prop_infinite': 'Prop Infinite',
        'coverage_diff': 'Coverage - Target'
    })

    # Drop target_coverage column
    if 'target_coverage' in df.columns:
        df = df.drop(columns=['target_coverage'])

    # Reorder columns
    col_order = ['method', '0th %ile', '25th %ile', '50th %ile', '75th %ile',
                 'Overall Coverage', 'Mean Width', 'Median Width',
                 'Prop Infinite', 'Coverage - Target']
    df = df[[c for c in col_order if c in df.columns]]

    # Rename method column
    df = df.rename(columns={'method': 'Method'})

    # Build markdown table manually (no tabulate dependency)
    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"**Target Coverage:** {target_coverage:.1%}\n")
    lines.append("## Coverage by Percentile and Method\n")

    # Header row
    headers = df.columns.tolist()
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for _, row in df.iterrows():
        values = [str(v) for v in row.values]
        lines.append("| " + " | ".join(values) + " |")

    lines.append("\n**Notes:**")
    lines.append("- Coverage values show the proportion of true outcomes falling within prediction intervals")
    lines.append("- Width metrics are in the units of the outcome variable")
    lines.append("- Coverage - Target shows deviation from the nominal coverage level\n")

    # Write to file
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Markdown table written to {md_path}")


def add_data_summary_to_markdown(md_path, data_csv, group_col, outcome_col,
                                   original_outcome_col=None, is_test_group_func=None):
    """
    Append data summary table to existing markdown file.

    Parameters:
    -----------
    md_path : str
        Path to markdown file to append to
    data_csv : str
        Path to filtered data CSV
    group_col : str
        Column name for grouping (e.g., 'state_abb' or 'clinic_id')
    outcome_col : str
        Column name for transformed outcome (e.g., 'y')
    original_outcome_col : str, optional
        Column name for original outcome (e.g., 'income')
    is_test_group_func : function, optional
        Function that takes group ID and returns True if it's a test group
    """
    df = pd.read_csv(data_csv)

    # Compute statistics per group
    summary_rows = []

    for group in sorted(df[group_col].unique()):
        group_df = df[df[group_col] == group]
        n = len(group_df)

        # Outcome statistics (transformed)
        y_mean = group_df[outcome_col].mean()
        y_median = group_df[outcome_col].median()
        y_std = group_df[outcome_col].std()
        y_min = group_df[outcome_col].min()
        y_max = group_df[outcome_col].max()
        y_range = y_max - y_min

        row = {
            'Group': group,
            'N': n,
            'Y Min': round(y_min, 2),
            'Y Max': round(y_max, 2),
            'Y Range': round(y_range, 2),
            'Y Mean': round(y_mean, 2),
            'Y Median': round(y_median, 2),
            'Y Std': round(y_std, 2)
        }

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort by test/train if function provided, then by group name
    if is_test_group_func:
        # Create temporary column for sorting
        summary_df['_is_test'] = summary_df['Group'].apply(is_test_group_func)
        summary_df = summary_df.sort_values(['_is_test', 'Group'], ascending=[False, True])
        summary_df = summary_df.drop(columns=['_is_test'])
        summary_df = summary_df.reset_index(drop=True)

    # Build markdown table
    lines = []
    lines.append("\n\n## Data Summary by Group\n")

    # Header
    headers = summary_df.columns.tolist()
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for _, row in summary_df.iterrows():
        values = [str(v) for v in row.values]
        lines.append("| " + " | ".join(values) + " |")

    # Append to existing file
    with open(md_path, 'a') as f:
        f.write('\n'.join(lines))

    print(f"Data summary appended to {md_path}")
