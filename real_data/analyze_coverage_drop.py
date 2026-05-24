#!/usr/bin/env python3
"""
Analyze the coverage drop pattern from the existing results.

We can't see the actual intervals, but we can analyze:
1. Which replicates fail at o=20 but pass at o=0
2. Pattern of failures across o values
3. Whether failures are systematic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('acs/results/true_marginal/acs_true_marg_detailed.csv')

# Focus on D-HCP
dhcp = df[df['method'] == 'Donor-HCP'].copy()

print("="*80)
print("ANALYZING COVERAGE DROP PATTERN")
print("="*80)

# Create a pivot table: replicates x o values
pivot = dhcp.pivot_table(
    index='replicate',
    columns='o',
    values='coverage',
    aggfunc='first'
)

print(f"\nTotal replicates: {len(pivot)}")
print(f"o values: {sorted(dhcp['o'].unique())}")

# Analyze transition patterns
print(f"\n{'='*80}")
print("TRANSITION ANALYSIS:")
print(f"{'='*80}")

for o_from, o_to in [(0, 5), (5, 10), (10, 15), (15, 20)]:
    if o_from in pivot.columns and o_to in pivot.columns:
        # Replicates that pass at o_from
        pass_from = pivot[pivot[o_from] == 1.0]

        # How many of those fail at o_to?
        fail_to = pass_from[pass_from[o_to] == 0.0]

        transition_fail_rate = len(fail_to) / len(pass_from) * 100 if len(pass_from) > 0 else 0

        print(f"\no={o_from} → o={o_to}:")
        print(f"  Replicates passing at o={o_from}: {len(pass_from)}")
        print(f"  Of those, failing at o={o_to}: {len(fail_to)}")
        print(f"  Transition failure rate: {transition_fail_rate:.1f}%")

# Check if certain replicates consistently fail
print(f"\n{'='*80}")
print("REPLICATE-SPECIFIC PATTERNS:")
print(f"{'='*80}")

# Count failures per replicate across all o values
pivot_fail_count = (pivot == 0.0).sum(axis=1)

print(f"\nDistribution of failures per replicate:")
print(pivot_fail_count.value_counts().sort_index())

# Analyze specific problematic replicates
worst_replicates = pivot_fail_count[pivot_fail_count >= 3].index.tolist()
print(f"\nReplicates with 3+ failures: {len(worst_replicates)}")

if len(worst_replicates) > 0:
    print("\nExample worst replicates (first 5):")
    for rep_idx in worst_replicates[:5]:
        cov_vals = pivot.loc[rep_idx]
        print(f"  Rep {rep_idx}: {dict(cov_vals)}")

# Check width patterns
print(f"\n{'='*80}")
print("WIDTH ANALYSIS:")
print(f"{'='*80}")

# For replicates that fail at o=20, are their widths unusually narrow?
for o in sorted(dhcp['o'].unique()):
    subset = dhcp[dhcp['o'] == o]
    passing = subset[subset['coverage'] == 1.0]
    failing = subset[subset['coverage'] == 0.0]

    if len(passing) > 0 and len(failing) > 0:
        width_pass = passing['width_income'].mean()
        width_fail = failing['width_income'].mean()
        width_diff_pct = (width_pass - width_fail) / width_pass * 100

        print(f"\no={o}:")
        print(f"  Mean width (passing): ${width_pass:,.0f}")
        print(f"  Mean width (failing): ${width_fail:,.0f}")
        print(f"  Difference: {width_diff_pct:.1f}% {'wider' if width_diff_pct > 0 else 'narrower'} for passing")

print(f"\n{'='*80}")
print("HYPOTHESIS:")
print(f"{'='*80}")

print("\nIf passing replicates have WIDER intervals than failing ones,")
print("this suggests intervals are getting TOO NARROW at higher o.")
print("\nThis could be due to:")
print("1. D-HCP using more test observations → tighter intervals")
print("2. But not enough calibration data to properly quantify uncertainty")
print("3. Result: overconfident (too narrow) intervals that undercover")

# Calculate overall statistics
print(f"\n{'='*80}")
print("OVERALL STATISTICS:")
print(f"{'='*80}")

stats = dhcp.groupby('o').agg({
    'coverage': ['mean', 'std'],
    'width_income': ['mean', 'std']
})

print(f"\n{'o':>3s}  {'Coverage':>10s}  {'Cov Std':>10s}  {'Width':>15s}  {'Width Std':>15s}")
print("-" * 70)
for o in sorted(dhcp['o'].unique()):
    cov_mean = stats.loc[o, ('coverage', 'mean')]
    cov_std = stats.loc[o, ('coverage', 'std')]
    width_mean = stats.loc[o, ('width_income', 'mean')]
    width_std = stats.loc[o, ('width_income', 'std')]

    print(f"{o:3d}  {cov_mean:10.4f}  {cov_std:10.4f}  {width_mean:15,.0f}  {width_std:15,.0f}")

print("\n" + "="*80)
