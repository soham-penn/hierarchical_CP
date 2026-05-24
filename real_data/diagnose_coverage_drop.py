#!/usr/bin/env python3
"""
Diagnose why D-HCP coverage drops from 88% to 78% as o increases from 0 to 20.

Examine a few specific replicates to understand:
1. How intervals change with o
2. Why they stop covering the true target at higher o
3. Whether the problem is in mu_center or in the quantile q
"""

import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('acs/results/true_marginal/acs_true_marg_detailed.csv')

# Focus on D-HCP
dhcp = df[df['method'] == 'Donor-HCP'].copy()

print("="*80)
print("DIAGNOSING COVERAGE DROP FOR DONOR-HCP")
print("="*80)

# Find some replicates that fail at o=20 but pass at o=0
dhcp_pivot = dhcp.pivot_table(
    index='replicate',
    columns='o',
    values='coverage',
    aggfunc='first'
)

# Replicates that pass at o=0 but fail at o=20
passing_at_0 = dhcp_pivot[dhcp_pivot[0] == 1.0]
failing_at_20 = passing_at_0[passing_at_0[20] == 0.0]

print(f"\nReplicates passing at o=0: {len(passing_at_0)}")
print(f"Replicates passing at o=0 but failing at o=20: {len(failing_at_20)}")
print(f"Failure rate: {len(failing_at_20) / len(passing_at_0) * 100:.1f}%")

# Pick a few examples
example_reps = failing_at_20.head(5).index.tolist()

print(f"\n{'='*80}")
print(f"EXAMINING {len(example_reps)} EXAMPLE REPLICATES THAT FAIL AT o=20:")
print(f"{'='*80}")

for rep_idx in example_reps:
    print(f"\n{'='*80}")
    print(f"Replicate {rep_idx}:")
    print(f"{'='*80}")

    rep_data = dhcp[dhcp['replicate'] == rep_idx].sort_values('o')

    for _, row in rep_data.iterrows():
        o = int(row['o'])
        cov = row['coverage']
        width_income = row['width_income']

        # NOTE: We don't have mu_hat or interval bounds in the CSV
        # This is a limitation - we'd need to modify the experiment to save these

        print(f"  o={o:2d}: coverage={int(cov)}, width={width_income:,.0f}")

    # Check trend
    widths = rep_data.set_index('o')['width_income']
    print(f"\n  Width change: {widths[0]:,.0f} (o=0) → {widths[20]:,.0f} (o=20)")
    print(f"  Width decreased by: {(widths[0] - widths[20])/widths[0]*100:.1f}%")

print(f"\n{'='*80}")
print("AGGREGATE STATISTICS BY o:")
print(f"{'='*80}")

# Compute statistics by o
stats = dhcp.groupby('o').agg({
    'coverage': ['mean', 'std'],
    'width_income': ['mean', 'median', 'std']
})

print(f"\n{'o':>3s}  {'Cov Mean':>8s}  {'Cov Std':>8s}  {'Width Mean':>12s}  {'Width Median':>14s}  {'Width Std':>12s}")
print("-" * 80)
for o in sorted(dhcp['o'].unique()):
    cov_mean = stats.loc[o, ('coverage', 'mean')]
    cov_std = stats.loc[o, ('coverage', 'std')]
    width_mean = stats.loc[o, ('width_income', 'mean')]
    width_median = stats.loc[o, ('width_income', 'median')]
    width_std = stats.loc[o, ('width_income', 'std')]

    print(f"{o:3d}  {cov_mean:8.4f}  {cov_std:8.4f}  {width_mean:12,.0f}  {width_median:14,.0f}  {width_std:12,.0f}")

# Check if width decrease correlates with coverage drop
print(f"\n{'='*80}")
print("HYPOTHESIS: Coverage drops because intervals get too narrow")
print(f"{'='*80}")

width_change = (stats.loc[0, ('width_income', 'mean')] - stats.loc[20, ('width_income', 'mean')]) / stats.loc[0, ('width_income', 'mean')] * 100
cov_change = (stats.loc[0, ('coverage', 'mean')] - stats.loc[20, ('coverage', 'mean')]) * 100

print(f"\nWidth decreased by: {width_change:.1f}%")
print(f"Coverage decreased by: {cov_change:.1f} percentage points")
print(f"\nIf width decreases by 26% but coverage drops by 10pp, this suggests")
print(f"the intervals are shrinking TOO MUCH relative to the uncertainty.")

print(f"\n{'='*80}")
print("NEXT STEPS:")
print(f"{'='*80}")
print("1. Check if tau override helps stabilize coverage")
print("2. Examine if the donor weighting is causing issues at high o")
print("3. Compare with DGP results to see if this is expected behavior")
print("4. Verify the group-level offset is being computed correctly")

