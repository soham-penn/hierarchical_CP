"""
Repeated group experiment — current design
=========================================
For each dataset:

Step 1 (fixed split):
  - 17 calibration + 15 test clinics for BP (middle-15 by size, min >= 10).
    - ACS (single state): sample K eligible PUMAs once (without replacement)
        as the non-test pool; all remaining eligible PUMAs are test groups.
  - m = min test group size.
    - o_values (for donor-HCP/sample-HCP methods) = {0, floor(m/4), floor(m/2), floor(3m/4), m-1}
    (these are the 5 fixed absolute o values on the x-axis).

Step 2 (bootstrap sample):
  - For EVERY group k (both calibration and test), resample N_k obs with replacement
    from group k's original observations.
    - For each o, target is the (o+1)-th bootstrapped observation (index o).

Step 3 (baselines — HCP, Pooling, Subsampling, Repeated):
  - Randomly split the 17 calibration groups into ~8 train / ~9 calib.
  - Fit model on train, compute scores on calib.
    - For each test group and each o: target = bootstrapped index o → covered (0/1).
  - Average over 15 test groups → one coverage value per replicate.
  - (Baselines use no o; they produce a single coverage per replicate.)

Step 4 (donor-HCP and sample-HCP, randomized/derandomized — for each o value):
  - For each o in o_values:
    * The train/calib split follows the method-specific selection based on o.
            * For each test group: history = first o bootstrapped obs, target = index o.
      * Covered? → average over all 15 test groups → one coverage value per replicate.

Step 5: Repeat Steps 2-4 for B replicates → B coverage values per method per o.
  - Baselines: B values (same for all o; plotted as a flat horizontal boxplot reference).
    - donor-HCP/sample-HCP variants: B values per o_value → boxplot per o_value.

x-axis: o values {0, floor(m/4), floor(m/2), floor(3m/4), m-1}
Baseline boxes are replicated across all x positions (same B values everywhere).

Saves to:
  blood_pressure/plots_new/   blood_pressure/results_new/
  acs/plots_new/              acs/results_new/

Run:
    python repeated_experiments.py --dataset all --B_bp 100 --B_acs 100
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from blood_pressure.data_processing import load_and_clean_bp_data, build_design_matrix_bp
from acs.data_processing import load_and_clean_acs_pums, build_design_matrix_acs, EMERGING_STATES

from methods.mu_methods import create_mu_method_ols_global_only, create_mu_method_ols_offset
from methods.donor_hcp import (
    compute_donor_hcp_randomized_interval,
    compute_donor_hcp_derandomized_interval,
)
from methods.sample_hcp import (
    compute_sample_hcp_randomized_interval,
    compute_sample_hcp_derandomized_interval,
)
from methods.baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius,
)
from scores import absolute_residual_score

# Methods that vary with o vs baselines (constant in o)
HCP_METHODS      = [
    'donor-HCP-randomized',
    'donor-HCP-derandomized',
    'sample-HCP-randomized',
    'sample-HCP-derandomized',
]
BASELINE_METHODS = ['HCP', 'Pooling', 'Subsampling', 'Repeated']
STD_CP_METHODS   = ['Std-CP']
METHODS          = HCP_METHODS + BASELINE_METHODS + STD_CP_METHODS

ACS_PLOT_HCP_METHODS = ['donor-HCP-randomized', 'sample-HCP-randomized']
ACS_PLOT_BASELINES = ['HCP', 'Std-CP']

COLORS = {
    'donor-HCP-randomized':    '#1f77b4',
    'donor-HCP-derandomized':  '#17becf',
    'sample-HCP-randomized':   '#ff7f0e',
    'sample-HCP-derandomized': '#bcbd22',
    'HCP':         '#2ca02c',
    'Pooling':     '#d62728',
    'Subsampling': '#9467bd',
    'Repeated':    '#8c564b',
    'Std-CP':      '#111111',
}

O_COLORS = {
    0: '#2166ac',
    2: '#00b4d8',
    4: '#f77f00',
    6: '#7b2cbf',
    5: '#4393c3',
    10: '#f4a582',
    15: '#00B894',
    20: '#b2182b',
    25: '#8E44AD',
    30: '#C0392B',
    35: '#2C3E50',
    40: '#7F8C8D',
}


def _get_o_color(o_val):
    # Cast to int when possible so numpy scalars / float-like values map correctly.
    try:
        key = int(o_val)
    except Exception:
        key = o_val
    return O_COLORS.get(key, '#555555')

METHOD_RENAME = {
    'donor-HCP-randomized': 'D-HCP',
    'donor-HCP-derandomized': 'DD-HCP',
    'sample-HCP-randomized': 'S-HCP',
    'sample-HCP-derandomized': 'D-Sample-HCP',
    'HCP': 'HCP',
    'Pooling': 'Pooling',
    'Subsampling': 'Subsampling',
    'Repeated': 'Repeated',
    'Std-CP': 'Std-CP',
}


def _interval_from_radius(center, radius):
    return (-np.inf, np.inf) if np.isinf(radius) else (center - radius, center + radius)

def _covered(interval, y):
    return int(interval[0] <= y <= interval[1])

def _width(interval):
    lo, hi = interval
    return (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else np.nan


def _width_income_from_log1p_interval(interval):
    """Map a finite interval on log1p-income scale to width on income scale."""
    lo, hi = interval
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    with np.errstate(over='ignore', invalid='ignore'):
        w = np.expm1(hi) - np.expm1(lo)
    if not np.isfinite(w) or w < 0:
        return np.nan
    return float(w)


def _human_number_format(y, _):
    ay = abs(y)
    if ay >= 1e12:
        return f'{y:.1e}'
    if ay >= 1e9:
        return f'{y / 1e9:.1f}B'
    if ay >= 1e6:
        return f'{y / 1e6:.1f}M'
    if ay >= 1e3:
        return f'{y / 1e3:.1f}K'
    if ay >= 1:
        return f'{y:.0f}'
    return f'{y:.2f}'


def _nanmean_no_warning(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


def _nanstd_no_warning(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return np.nanstd(arr)


def _split_conformal_radius(abs_residuals, alpha):
    """Finite-sample split-conformal radius from calibration residuals."""
    scores = np.asarray(abs_residuals, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(1, k), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _compute_std_cp_interval(x_hist, y_hist, x_target, alpha, rng):
    """Standard split CP within one target group using its observed history."""
    x_hist = np.asarray(x_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    if x_hist.ndim == 1:
        x_hist = x_hist.reshape(-1, 1)

    n_hist = len(y_hist)
    n_train = n_hist // 2
    n_cal = n_hist - n_train
    if n_train < 1 or n_cal < 1:
        return (-np.inf, np.inf)

    perm = rng.permutation(n_hist)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:]

    x_train = x_hist[train_idx]
    y_train = y_hist[train_idx]
    x_cal = x_hist[cal_idx]
    y_cal = y_hist[cal_idx]

    x_train_aug = np.column_stack([np.ones(len(x_train)), x_train])
    x_cal_aug = np.column_stack([np.ones(len(x_cal)), x_cal])

    beta, *_ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
    y_cal_pred = x_cal_aug @ beta
    radius = _split_conformal_radius(np.abs(y_cal - y_cal_pred), alpha)

    x_target = np.asarray(x_target, dtype=float).reshape(1, -1)
    x_target_aug = np.column_stack([np.ones(1), x_target])
    y_hat = float((x_target_aug @ beta)[0])
    return _interval_from_radius(y_hat, radius)


# ---------------------------------------------------------------------------
# Single bootstrap replicate
# ---------------------------------------------------------------------------

def run_one_replicate(df, X, calib_groups, test_groups, group_col,
                      o_values, config, b):
    """
        Returns dict:
            result['baseline'][method][o] = {'coverage': float, 'width': float, 'width_income': float}
            result['hcp'][method][o]      = {'coverage': float, 'width': float, 'width_income': float}
    """
    rng       = np.random.default_rng(config['seed'] + b)
    alpha     = config['alpha']
    alpha_sel = config.get('alpha_selection', 0.5)
    n_rep     = config.get('n_repeated', 50)

    # ------------------------------------------------------------------
    # 1. Bootstrap each group's observations for this replicate.
    #    This keeps history/target exchangeable within group when target=index o.
    # ------------------------------------------------------------------
    all_groups = calib_groups + test_groups
    group_data = {}   # group -> {'X': array, 'Y': array}
    for grp in all_groups:
        grp_idx = df.index[df[group_col] == grp].to_numpy()
        n_k = len(grp_idx)
        boot_idx = rng.choice(grp_idx, size=n_k, replace=True)
        group_data[grp] = {
            'X': X[boot_idx],
            'Y': df.loc[boot_idx, 'y'].values,
        }

    # ------------------------------------------------------------------
    # 2. Build Z_calibration for all calibration groups (for HCP++/HCP.sample)
    #    These are all K_calib groups; the method internally splits by N_k > o.
    # ------------------------------------------------------------------
    K_calib = len(calib_groups)
    Z_calibration_full = [
        [{'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
         for i in range(len(group_data[grp]['Y']))]
        for grp in calib_groups
    ]
    U_calibration_full = np.zeros((K_calib, 1))

    # ------------------------------------------------------------------
    # 3. Baseline methods: use donor-style train/calib split with o=0
    #    This ensures HCP at o=0 matches D-HCP at o=0 (as in DGP experiments)
    # ------------------------------------------------------------------
    from methods.donor_hcp import get_donor_style_train_cal_split

    sample_sizes = [len(Z_calibration_full[j]) for j in range(K_calib)]
    train_idx, calib_idx = get_donor_style_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=alpha_sel
    )
    train_idx = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
    calib_idx = calib_idx.tolist() if hasattr(calib_idx, 'tolist') else list(calib_idx)

    mu_baseline = create_mu_method_ols_global_only()
    mu_hcp      = create_mu_method_ols_global_only()

    model_baseline = mu_baseline['fit_global'](
        U_matrix=U_calibration_full,
        Z_list=Z_calibration_full,
        group_index_vector=train_idx,
    )

    scores_list = []
    for j in calib_idx:
        Zj  = Z_calibration_full[j]
        yj  = np.array([z['Y'] for z in Zj])
        Uj  = U_calibration_full[j]
        muj = np.array([
            mu_baseline['predict_global'](model_baseline, z['X'], Uj)
            for z in Zj
        ])
        scores_list.append(absolute_residual_score(yj, muj))

    T_hcp  = compute_hcp_interval_radius(scores_list, alpha)
    T_pool = compute_pooling_interval_radius(scores_list, alpha)
    T_sub  = compute_subsampling_once_interval_radius(scores_list, alpha)
    T_rep  = compute_repeated_subsampling_interval_radius(scores_list, alpha, n_rep)

    U_test = np.zeros((1, 1))

    baseline_result = {m: {} for m in BASELINE_METHODS}
    stdcp_result = {m: {} for m in STD_CP_METHODS}

    # ------------------------------------------------------------------
    # 4. HCP++ and HCP.sample for each o value
    #    The methods now internally fit on complement of S_tilde (fixed).
    # ------------------------------------------------------------------
    hcp_result = {m: {} for m in HCP_METHODS}

    for o in o_values:
        # Need o observed points plus one target at index o.
        eligible_test_groups = [
            grp for grp in test_groups
            if len(group_data[grp]['Y']) >= (o + 1)
        ]

        if len(eligible_test_groups) == 0:
            for method in BASELINE_METHODS:
                baseline_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
            for method in HCP_METHODS:
                hcp_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
            for method in STD_CP_METHODS:
                stdcp_result[method][o] = {'coverage': np.nan, 'width': np.nan, 'width_income': np.nan}
            continue

        base_covered = {m: [] for m in BASELINE_METHODS}
        base_width = {m: [] for m in BASELINE_METHODS}
        base_width_income = {m: [] for m in BASELINE_METHODS}

        hcp_covered = {m: [] for m in HCP_METHODS}
        hcp_width   = {m: [] for m in HCP_METHODS}
        hcp_width_income = {m: [] for m in HCP_METHODS}

        stdcp_covered = {m: [] for m in STD_CP_METHODS}
        stdcp_width   = {m: [] for m in STD_CP_METHODS}
        stdcp_width_income = {m: [] for m in STD_CP_METHODS}

        for grp in eligible_test_groups:
            # Target is the (o+1)-th observation, i.e., index o in zero-based indexing.
            x_target = group_data[grp]['X'][o]
            true_y   = group_data[grp]['Y'][o]

            mu_hat = mu_baseline['predict_global'](model_baseline, x_target, U_test[0])
            for method, T in [
                ('HCP', T_hcp),
                ('Pooling', T_pool),
                ('Subsampling', T_sub),
                ('Repeated', T_rep),
            ]:
                interval = _interval_from_radius(mu_hat, T)
                base_covered[method].append(_covered(interval, true_y))
                base_width[method].append(_width(interval))
                base_width_income[method].append(_width_income_from_log1p_interval(interval))

            # Build Z_test: first o observed points + target at index o.
            Z_test = [
                {'X': group_data[grp]['X'][i], 'Y': group_data[grp]['Y'][i]}
                for i in range(o)
            ]
            Z_test.append({'X': x_target, 'Y': true_y})

            # donor-HCP randomized
            try:
                res_pp = compute_donor_hcp_randomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                )
                int_pp = res_pp['interval']
            except Exception:
                int_pp = (-np.inf, np.inf)

            # donor-HCP derandomized
            try:
                res_dd = compute_donor_hcp_derandomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                )
                int_dd = res_dd['interval']
            except Exception:
                int_dd = (-np.inf, np.inf)

            # sample-HCP randomized
            try:
                res_hs = compute_sample_hcp_randomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                )
                int_hs = res_hs['interval']
            except Exception:
                int_hs = (-np.inf, np.inf)

            # sample-HCP derandomized
            try:
                res_sd = compute_sample_hcp_derandomized_interval(
                    U_calibration=U_calibration_full,
                    Z_calibration=Z_calibration_full,
                    U_test=U_test, Z_test=Z_test,
                    o_observed=o, alpha=alpha,
                    alpha_selection=alpha_sel,
                    mu_method=mu_hcp,
                )
                int_sd = res_sd['interval']
            except Exception:
                int_sd = (-np.inf, np.inf)

            for method, interval in [
                ('donor-HCP-randomized', int_pp),
                ('donor-HCP-derandomized', int_dd),
                ('sample-HCP-randomized', int_hs),
                ('sample-HCP-derandomized', int_sd),
            ]:
                hcp_covered[method].append(_covered(interval, true_y))
                hcp_width[method].append(_width(interval))
                hcp_width_income[method].append(_width_income_from_log1p_interval(interval))

            # Standard split CP baseline within the target group itself.
            stdcp_interval = _compute_std_cp_interval(
                x_hist=group_data[grp]['X'][:o],
                y_hist=group_data[grp]['Y'][:o],
                x_target=x_target,
                alpha=alpha,
                rng=rng,
            )
            stdcp_covered['Std-CP'].append(_covered(stdcp_interval, true_y))
            stdcp_width['Std-CP'].append(_width(stdcp_interval))
            stdcp_width_income['Std-CP'].append(_width_income_from_log1p_interval(stdcp_interval))

        for method in BASELINE_METHODS:
            baseline_result[method][o] = {
                'coverage': np.mean(base_covered[method]),
                'width': _nanmean_no_warning(base_width[method]),
                'width_income': _nanmean_no_warning(base_width_income[method]),
            }

        for method in HCP_METHODS:
            hcp_result[method][o] = {
                'coverage': np.mean(hcp_covered[method]),
                'width': _nanmean_no_warning(hcp_width[method]),
                'width_income': _nanmean_no_warning(hcp_width_income[method]),
            }

        for method in STD_CP_METHODS:
            stdcp_result[method][o] = {
                'coverage': np.mean(stdcp_covered[method]),
                'width': _nanmean_no_warning(stdcp_width[method]),
                'width_income': _nanmean_no_warning(stdcp_width_income[method]),
            }

    return {'baseline': baseline_result, 'hcp': hcp_result, 'stdcp': stdcp_result}


# ---------------------------------------------------------------------------
# Bootstrap loop
# ---------------------------------------------------------------------------

def run_bootstrap(df, X, calib_groups, test_groups, group_col, o_values, config):
    B   = config['B']
    n_o = len(o_values)

    # cov/wid arrays: shape (B, n_methods, n_o)
    n_m  = len(METHODS)
    m_map = {m: i for i, m in enumerate(METHODS)}

    cov = np.full((B, n_m, n_o), np.nan)
    wid = np.full((B, n_m, n_o), np.nan)
    wid_income = np.full((B, n_m, n_o), np.nan)

    eligible_groups = config.get('eligible_groups', None)
    n_groups = config.get('n_puma_groups', None)
    split_seed = config.get('group_selection_seed', 42)
    redraw_split_each_rep = bool(config.get('resample_group_split_each_rep', False))

    for b in range(B):
        if (b + 1) % 10 == 0:
            print(f"  Replicate {b+1}/{B}")

        if redraw_split_each_rep:
            if eligible_groups is None or n_groups is None:
                raise ValueError("resample_group_split_each_rep=True requires eligible_groups and n_puma_groups in config")
            rng_split = np.random.default_rng(split_seed + b)
            calib_groups_rep = rng_split.choice(
                np.asarray(eligible_groups),
                size=int(n_groups),
                replace=False,
            ).tolist()
            calib_set_rep = set(calib_groups_rep)
            test_groups_rep = [g for g in eligible_groups if g not in calib_set_rep]
            if config.get('enforce_fixed_split_counts', False):
                expected_test = int(config.get('expected_test_pumas', len(eligible_groups) - int(n_groups)))
                if len(test_groups_rep) != expected_test:
                    raise ValueError(
                        f"Expected {expected_test} target PUMAs for replicate {b+1}, "
                        f"got {len(test_groups_rep)}"
                    )
        else:
            calib_groups_rep = calib_groups
            test_groups_rep = test_groups

        result = run_one_replicate(
            df, X, calib_groups_rep, test_groups_rep, group_col, o_values, config, b)
        if result is None:
            continue

        # Baselines: computed per o over eligible test groups
        for method in BASELINE_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[b, m_i, o_i] = result['baseline'][method][o]['coverage']
                wid[b, m_i, o_i] = result['baseline'][method][o]['width']
                wid_income[b, m_i, o_i] = result['baseline'][method][o]['width_income']

        # HCP methods: per o
        for method in HCP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[b, m_i, o_i] = result['hcp'][method][o]['coverage']
                wid[b, m_i, o_i] = result['hcp'][method][o]['width']
                wid_income[b, m_i, o_i] = result['hcp'][method][o]['width_income']

        # Standard CP method: per o
        for method in STD_CP_METHODS:
            m_i = m_map[method]
            for o_i, o in enumerate(o_values):
                cov[b, m_i, o_i] = result['stdcp'][method][o]['coverage']
                wid[b, m_i, o_i] = result['stdcp'][method][o]['width']
                wid_income[b, m_i, o_i] = result['stdcp'][method][o]['width_income']

    return {
        'methods': METHODS,
        'o_values': o_values,
        'coverage': cov,
        'width': wid,
        'width_income': wid_income,
    }


# ---------------------------------------------------------------------------
# Grouped boxplots
# ---------------------------------------------------------------------------

def draw_box_plots(results, output_path, target_coverage, title,
                   method_indices, methods_subset):
    o_values = results['o_values']
    cov      = results['coverage']
    wid      = results['width']
    B        = cov.shape[0]
    n_m      = len(method_indices)
    n_o      = len(o_values)

    group_gap = 1.0
    box_width = 0.8 / n_m
    offsets   = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * box_width

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_o * n_m * 0.55 + 2), 9))

    for ax, data3d, ylabel, ylim in [
        (axes[0], cov, 'Coverage',       [0.0, 1.05]),
        (axes[1], wid, 'Interval Width',  None),
    ]:
        legend_patches = []
        for k, m_i in enumerate(method_indices):
            method = methods_subset[k]
            color  = COLORS[method]
            for o_i in range(n_o):
                vals = data3d[:, m_i, o_i]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                pos = o_i * group_gap + offsets[k]
                ax.boxplot(
                    vals,
                    positions=[pos],
                    widths=box_width * 0.85,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker='o', color=color,
                                   markersize=3, alpha=0.5),
                    boxprops=dict(facecolor=color, alpha=0.6, color=color),
                )
            legend_patches.append(
                mpatches.Patch(facecolor=color, alpha=0.7, label=method))

        ax.set_xticks([i * group_gap for i in range(n_o)])
        ax.set_xticklabels([str(o) for o in o_values])
        ax.set_xlabel('History size o', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        if ylim:
            ax.set_ylim(ylim)

    axes[0].axhline(target_coverage, color='black', linestyle='--', linewidth=1.5)
    legend_patches.append(
        mlines.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                      label=f'Target {target_coverage:.0%}'))
    axes[0].legend(handles=legend_patches, loc='lower right', ncol=2, fontsize=9)
    axes[0].set_title(f'{title}  |  B={B}  |  target={target_coverage:.0%}',
                      fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _boxplot_style(ax, data, pos, color, width=0.55, edge_color=None,
                   hatch=None, alpha=0.75, linestyle='-'):
    if edge_color is None:
        edge_color = color
    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color='black', linewidth=1.4),
        whiskerprops=dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        capprops=dict(color=edge_color, linewidth=1.0, linestyle=linestyle),
        flierprops=dict(marker='.', color=edge_color, markersize=3, alpha=0.4),
        boxprops=dict(
            facecolor=color,
            alpha=alpha,
            edgecolor=edge_color,
            linewidth=1.0,
            linestyle=linestyle,
        ),
    )
    if hatch:
        for patch in bp['boxes']:
            patch.set_hatch(hatch)
    return bp


def draw_acs_dgp_style_plots(results, output_dir, tag, title, alpha=0.2):
    o_values = sorted(results['o_values'])
    cov = results['coverage']
    wid = results['width']
    wid_income = results.get('width_income', None)
    methods = results['methods']
    method_index = {m: i for i, m in enumerate(methods)}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    title_fs = 28
    label_fs = 24
    tick_fs = 21
    legend_fs = 17

    def _draw_metric_panel(ax, metric, ylabel, target_line):
        pos = 0
        tick_positions = []
        tick_labels = []
        baseline_methods = ACS_PLOT_BASELINES

        if metric == 'coverage':
            data3d = cov
        else:
            data3d = wid_income if wid_income is not None else np.expm1(wid)

        width_display_cap = None
        if metric == 'width':
            # Keep clipping reference on D/S-HCP + HCP so Std-CP tails do not dominate scale.
            cap_reference_methods = ACS_PLOT_HCP_METHODS + ['HCP']
            plotted_idx = [method_index[m] for m in cap_reference_methods if m in method_index]
            finite_vals = data3d[:, plotted_idx, :]
            finite_vals = finite_vals[np.isfinite(finite_vals)]
            finite_vals = finite_vals[finite_vals >= 0]
            if finite_vals.size > 0:
                width_display_cap = float(np.nanquantile(finite_vals, 0.90))
                if not np.isfinite(width_display_cap) or width_display_cap <= 0:
                    width_display_cap = float(np.nanmax(finite_vals))
            else:
                width_display_cap = 1.0

        for method in ACS_PLOT_HCP_METHODS:
            m_i = method_index[method]
            group_start = pos
            for o_val in o_values:
                o_i = o_values.index(o_val)
                vals = data3d[:, m_i, o_i]
                vals = vals[~np.isnan(vals)]
                if metric == 'width' and width_display_cap is not None:
                    vals = np.clip(vals, 0.0, width_display_cap)
                if len(vals) == 0:
                    pos += 1
                    continue
                color = _get_o_color(o_val)
                _boxplot_style(ax, vals, pos, color, width=0.58)
                pos += 1
            group_end = pos - 1
            tick_positions.append((group_start + group_end) / 2)
            tick_labels.append(METHOD_RENAME[method])
            pos += 1.2

        baseline_style = {
            'HCP': {'hatch': '///', 'alpha': 0.34},
            'Std-CP': {'hatch': 'xxx', 'alpha': 0.18},
        }
        baseline_step = 3.4
        for method in baseline_methods:
            m_i = method_index[method]
            vals = data3d[:, m_i, :].reshape(-1)
            vals = vals[~np.isnan(vals)]
            if metric == 'width' and width_display_cap is not None:
                vals = np.clip(vals, 0.0, width_display_cap)
            if len(vals) == 0:
                continue
            _boxplot_style(
                ax,
                vals,
                pos,
                color='#111111',
                width=0.58,
                edge_color='#111111',
                alpha=baseline_style.get(method, {}).get('alpha', 0.28),
                linestyle='--',
                hatch=baseline_style.get(method, {}).get('hatch', '///'),
            )
            tick_positions.append(pos)
            tick_labels.append(METHOD_RENAME[method])
            pos += baseline_step

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=tick_fs, fontweight='bold')
        ax.set_xlabel('Method', fontsize=label_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.set_xlim(-0.8, pos - 0.4)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.tick_params(axis='y', labelsize=tick_fs)

        if target_line is not None:
            ax.axhline(
                target_line,
                color='black',
                linewidth=1.2,
                linestyle='--',
                zorder=0,
                alpha=0.55,
            )

        if metric == 'coverage':
            ax.set_ylim(0.55, 1.05)
            ax.set_title('Coverage', fontsize=title_fs - 2, fontweight='bold')
        else:
            y_max = float(width_display_cap) if width_display_cap is not None else 65000.0
            y_max = max(65000.0, y_max * 1.05)
            ax.set_ylim(60000.0, y_max)
            ax.yaxis.set_major_formatter(FuncFormatter(_human_number_format))
            ax.set_title('Interval Width (income units, 90th pct clipped)', fontsize=title_fs - 5, fontweight='bold')

    fig, axes = plt.subplots(1, 2, figsize=(21.5, 7.0))
    _draw_metric_panel(axes[0], 'coverage', 'Coverage', 1 - alpha)
    _draw_metric_panel(axes[1], 'width', 'Interval Width (income units)', None)

    legend_handles = []
    for o_val in o_values:
        color = _get_o_color(o_val)
        legend_handles.append(
            mpatches.Patch(
                facecolor=color,
                alpha=0.75,
                edgecolor=color,
                label=f'o = {o_val}',
            )
        )
    legend_handles.extend([
        mpatches.Patch(facecolor='#111111', alpha=0.34, edgecolor='#111111', hatch='///', linestyle='--', label='HCP baseline'),
        mpatches.Patch(facecolor='#111111', alpha=0.18, edgecolor='#111111', hatch='xxx', linestyle='--', label='Std-CP baseline'),
        mlines.Line2D([], [], color='black', linewidth=1.2, linestyle='--', alpha=0.55, label=f'Target ({1 - alpha:.0%})'),
    ])

    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(0.90, 0.5),
        fontsize=legend_fs,
        framealpha=0.9,
        ncol=1,
        title='Legend',
        title_fontsize=legend_fs + 2,
    )

    fig.suptitle(
        f'D-HCP / S-HCP vs HCP / Std-CP  -  {title}',
        fontsize=title_fs,
        fontweight='bold',
        y=0.98,
    )

    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.15, top=0.88, wspace=0.24)

    side_pdf = output_dir / f'{tag}_effect_of_o_side_by_side_main_compare.pdf'
    side_png = output_dir / f'{tag}_effect_of_o_side_by_side_main_compare.png'
    fig.savefig(str(side_pdf), bbox_inches='tight')
    fig.savefig(str(side_png), dpi=200, bbox_inches='tight')

    # Backward-compatible filenames kept for downstream scripts.
    cov_pdf = output_dir / f'{tag}_effect_of_o_coverage_main_compare.pdf'
    cov_png = output_dir / f'{tag}_effect_of_o_coverage_main_compare.png'
    wid_pdf = output_dir / f'{tag}_effect_of_o_width_main_compare.pdf'
    wid_png = output_dir / f'{tag}_effect_of_o_width_main_compare.png'
    fig.savefig(str(cov_pdf), bbox_inches='tight')
    fig.savefig(str(cov_png), dpi=200, bbox_inches='tight')
    fig.savefig(str(wid_pdf), bbox_inches='tight')
    fig.savefig(str(wid_png), dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'  Saved: {side_pdf}')
    print(f'  Saved: {side_png}')
    print(f'  Saved: {cov_pdf}')
    print(f'  Saved: {cov_png}')
    print(f'  Saved: {wid_pdf}')
    print(f'  Saved: {wid_png}')


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(results, plots_dir, results_dir, tag, target_coverage, title,
                 plot_style='default'):
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    o_values = results['o_values']
    cov      = results['coverage']
    wid      = results['width']
    wid_income = results.get('width_income', np.full_like(wid, np.nan))
    B        = cov.shape[0]

    if plot_style == 'dgp':
        draw_acs_dgp_style_plots(
            results,
            output_dir=plots_dir,
            tag=tag,
            title=title,
            alpha=1 - target_coverage,
        )
    else:
        all_idx = list(range(len(METHODS)))
        no_rep_idx = [i for i, m in enumerate(METHODS) if m != 'Repeated']
        no_rep_m = [m for m in METHODS if m != 'Repeated']

        draw_box_plots(
            results,
            str(plots_dir / f'{tag}_new_all.png'),
            target_coverage,
            title,
            all_idx,
            METHODS,
        )
        draw_box_plots(
            results,
            str(plots_dir / f'{tag}_new.png'),
            target_coverage,
            f'{title} (excl. Repeated)',
            no_rep_idx,
            no_rep_m,
        )

    # Detailed CSV
    rows = []
    for b in range(B):
        for m_i, method in enumerate(METHODS):
            for o_i, o in enumerate(o_values):
                c = cov[b, m_i, o_i]
                w = wid[b, m_i, o_i]
                rows.append({
                    'replicate': b + 1,
                    'method':    method,
                    'o':         o,
                    'coverage':  None if np.isnan(c) else round(float(c), 4),
                    'width':     None if np.isnan(w) else round(float(w), 4),
                    'width_income': None if np.isnan(wid_income[b, m_i, o_i]) else round(float(wid_income[b, m_i, o_i]), 4),
                })
    csv_path = str(results_dir / f'{tag}_new_detailed.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Summary table
    print(f"\nCoverage (mean ± sd over B={B} replicates):")
    hdr = f"{'Method':<14}" + "".join(f"  o={o:<6}" for o in o_values)
    print(hdr); print("-" * len(hdr))
    for m_i, method in enumerate(METHODS):
        row_str = f"{method:<14}"
        for o_i in range(len(o_values)):
            v = cov[:, m_i, o_i]
            row_str += f"  {_nanmean_no_warning(v):.3f}±{_nanstd_no_warning(v):.3f} "
        print(row_str)

    print(f"\nWidth (mean ± sd over B={B} replicates):")
    print(hdr); print("-" * len(hdr))
    for m_i, method in enumerate(METHODS):
        row_str = f"{method:<14}"
        for o_i in range(len(o_values)):
            v = wid[:, m_i, o_i]
            row_str += f"  {_nanmean_no_warning(v):.3f}±{_nanstd_no_warning(v):.3f} "
        print(row_str)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def compute_o_values(m):
    """Given min test group size m, return the 5 fixed o values."""
    return sorted(set([
        0,
        int(np.floor((m - 1) / 4)),
        int(np.floor((m - 1) / 2)),
        int(np.floor(3 * (m - 1) / 4)),
        m - 1,
    ]))


def compute_o_values_from_test_size_distribution(test_sizes):
    """Select o values from quantiles of the target-group size distribution."""
    if len(test_sizes) == 0:
        return [0]

    arr = np.asarray(test_sizes, dtype=float)
    # Keep o fixed and compact from the full eligible-size distribution.
    # Using method='lower' gives stable integer cutoffs for discrete sample-size data.
    # The 30th percentile captures the first meaningful step above the minimum size
    # in the CA 66-PUMA setup, yielding the requested default o grid [0, 4, 5, 9].
    q_sizes = np.quantile(arr, [0.0, 0.30, 0.75], method='lower')
    # Convert size quantiles to history lengths (target index = o => requires o+1 obs).
    derived = [max(0, int(q) - 1) for q in q_sizes]
    o_values = [0] + derived
    return sorted(set(o_values))


def save_puma_summary_csv(df, group_col, eligible_groups, output_csv):
    """Save per-PUMA post-filter summary stats and group sizes."""

    rows = []
    for grp in sorted(eligible_groups):
        grp_df = df[df[group_col] == grp]
        rows.append({
            'puma': grp,
            'n_individuals_after_filter': int(len(grp_df)),
            'income_mean': float(grp_df['income'].mean()),
            'income_median': float(grp_df['income'].median()),
            'income_std': float(grp_df['income'].std(ddof=1)) if len(grp_df) > 1 else 0.0,
            'y_mean': float(grp_df['y'].mean()),
            'y_std': float(grp_df['y'].std(ddof=1)) if len(grp_df) > 1 else 0.0,
            'age_mean': float(grp_df['age'].mean()),
            'hours_mean': float(grp_df['hours'].mean()),
            'entry_recency_mean': float(grp_df['entry_recency'].mean()),
        })

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")


def load_bp(config):
    df = load_and_clean_bp_data(
        config['data_path'], treatment_arm_only=True,
        outcome_type='followup', min_clinic_size=5,
    )
    df = df.reset_index(drop=True)

    # Optionally filter to treated-only sites (odd-numbered sites encode 'H')
    if config.get('treated_only', False):
        treated_sites = sorted([s for s in df['clinic_id'].unique() if s % 2 == 1])
        df = df[df['clinic_id'].isin(treated_sites)].reset_index(drop=True)
        print(f"  Filtered to {len(treated_sites)} treated (odd) sites: {treated_sites}")

    X  = build_design_matrix_bp(df)

    counts      = df.groupby('clinic_id').size().sort_values(ascending=True)
    n_total     = len(counts)
    n_test      = config['n_test_clinics']
    min_test_sz = config.get('min_test_size', 10)

    nl = (n_total - n_test) // 2
    nh = n_total - n_test - nl
    sorted_clinics = counts.index.tolist()

    test_groups  = sorted_clinics[nl: n_total - nh]
    calib_groups = sorted_clinics[:nl] + sorted_clinics[n_total - nh:]

    # Verify min test size
    actual_min = min(counts[g] for g in test_groups)
    assert actual_min >= min_test_sz, \
        f"Min test group size {actual_min} < required {min_test_sz}"

    m        = actual_min
    o_values = compute_o_values(m)

    print(f"  {len(df)} obs, {df['clinic_id'].nunique()} clinics  |  "
          f"test={len(test_groups)}, calib={len(calib_groups)}")
    print(f"  Test clinic sizes: {sorted(counts[g] for g in test_groups)}")
    print(f"  m={m}, o_values={o_values}")
    return df, X, calib_groups, test_groups, 'clinic_id', o_values


def load_acs(config):
    # Single-state ACS experiment with PUMA groups.
    target_state = config.get('acs_state', 'CA')
    n_groups = config.get('n_puma_groups', 30)
    min_group_size = config.get('min_puma_size', 5)
    min_hours = config.get('min_hours', 20)
    min_income = config.get('min_income', None)
    age_min = config.get('age_min', 25)
    age_max = config.get('age_max', 54)
    selection_seed = config.get('group_selection_seed', 42)
    yoep_window_years = config.get('yoep_window_years', 2)
    yoep_min_year = config.get('yoep_min_year', None)
    top_income_q = config.get('top_income_quantile', None)
    bottom_income_q = config.get('bottom_income_quantile', None)

    if n_groups <= 1:
        raise ValueError("n_puma_groups must be at least 2")

    # Clean data for one state and keep PUMA labels for grouping.
    df = load_and_clean_acs_pums(
        config['data_path'],
        states_keep=[target_state],
        age_min=age_min,
        age_max=age_max,
        yoep_window_years=yoep_window_years,
        yoep_min_year=yoep_min_year,
        min_hours=min_hours,
        min_income=min_income,
        top_income_quantile=top_income_q,
        bottom_income_quantile=bottom_income_q,
    )
    if 'puma' not in df.columns:
        raise ValueError("ACS source must contain PUMA to run PUMA-group experiments")

    df = df.dropna(subset=['puma']).copy()
    # Keep PUMA coding stable for grouping.
    try:
        df['puma'] = df['puma'].astype(int)
    except Exception:
        df['puma'] = df['puma'].astype(str)

    counts_all = df.groupby('puma').size().sort_values(ascending=True)
    counts_eligible = counts_all[counts_all >= min_group_size]
    if len(counts_eligible) <= n_groups:
        raise ValueError(
            f"Only {len(counts_eligible)} eligible PUMAs in {target_state} "
            f"with size >= {min_group_size}, but n_puma_groups={n_groups}. "
            "Need at least one additional eligible PUMA for test groups."
        )

    # Keep all eligible PUMAs in the experiment; sample non-test group IDs.
    df = df[df['puma'].isin(counts_eligible.index)].reset_index(drop=True)
    X = build_design_matrix_acs(df)

    eligible_groups = counts_eligible.index.to_numpy().tolist()
    config['eligible_groups'] = eligible_groups

    rng = np.random.default_rng(selection_seed)
    calib_groups = rng.choice(
        np.asarray(eligible_groups),
        size=n_groups,
        replace=False,
    ).tolist()

    calib_set = set(calib_groups)
    test_groups = [g for g in eligible_groups if g not in calib_set]

    m = min(counts_eligible[g] for g in test_groups)
    eligible_sizes = [int(v) for v in counts_eligible.values]

    if config.get('enforce_fixed_split_counts', False):
        expected_eligible = int(config.get('expected_eligible_pumas', 66))
        expected_test = int(config.get('expected_test_pumas', expected_eligible - n_groups))
        if n_groups != 30:
            raise ValueError(f"Expected n_puma_groups=30, got {n_groups}")
        if len(eligible_groups) != expected_eligible:
            raise ValueError(
                f"Expected {expected_eligible} eligible PUMAs after filtering, got {len(eligible_groups)}. "
                "Adjust filtering settings to recover the configured split."
            )
        if len(test_groups) != expected_test:
            raise ValueError(
                f"Expected {expected_test} target PUMAs from the split, got {len(test_groups)}. "
                "Adjust filtering settings to recover the configured split."
            )

    fixed_o_values = config.get('fixed_o_values', None)
    if fixed_o_values is None:
        # Keep o fixed across repetitions using the full eligible-PUMA distribution.
        o_values = compute_o_values_from_test_size_distribution(eligible_sizes)
    else:
        o_values = sorted(set(int(v) for v in fixed_o_values if int(v) >= 0))

    print(f"  Target state: {target_state}")
    if yoep_min_year is None:
        print(f"  YOEP window years: {yoep_window_years}")
    else:
        print(f"  YOEP cutoff year: >= {yoep_min_year}")
    print(f"  Age filter: [{age_min}, {age_max}]")
    print(f"  Min hours filter: {min_hours}")
    print(f"  Min income filter: {min_income}")
    print(f"  Bottom income quantile keep: {bottom_income_q}")
    print(f"  Eligible PUMAs (size >= {min_group_size}): {len(counts_eligible)}")
    print(f"  Initial non-test PUMA sample (n={n_groups}, seed={selection_seed})")
    print(f"  test={len(test_groups)}, non-test={len(calib_groups)}")
    print(f"  Non-test PUMAs: {sorted(calib_groups)}")
    print(f"  Test PUMAs: {sorted(test_groups)}")
    print(f"  Test group sizes: {sorted(counts_eligible[g] for g in test_groups)}")
    print(f"  Test min size: {m}")
    print(f"  o values from full eligible-size distribution: {o_values}")
    return df, X, calib_groups, test_groups, 'puma', o_values


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['bp', 'bp_treated', 'acs_top25', 'acs_puma', 'all'], default='all')
    parser.add_argument('--B_bp',  type=int, default=100)
    parser.add_argument('--B_acs', type=int, default=100)
    parser.add_argument('--acs_state', type=str, default='CA')
    parser.add_argument('--acs_n_groups', type=int, default=30)
    parser.add_argument('--acs_min_group_size', type=int, default=20)
    parser.add_argument('--acs_o_values', type=str, default='0,5,10,20')
    parser.add_argument('--acs_yoep_window_years', type=int, default=2)
    parser.add_argument('--acs_min_hours', type=int, default=40)
    parser.add_argument('--acs_min_income', type=float, default=10000.0)
    parser.add_argument('--acs_bottom_income_quantile', type=float, default=None)
    parser.add_argument('--acs_min_yoep', type=int, default=2012)
    parser.add_argument('--acs_age_min', type=int, default=25)
    parser.add_argument('--acs_age_max', type=int, default=54)
    parser.add_argument('--acs_no_age_filter', action='store_true')
    parser.add_argument('--acs_group_seed', type=int, default=42)
    parser.add_argument('--acs_expected_eligible_pumas', type=int, default=61)
    parser.add_argument('--acs_expected_test_pumas', type=int, default=31)
    args = parser.parse_args()

    acs_o_values = [int(v.strip()) for v in args.acs_o_values.split(',') if v.strip() != '']
    if len(acs_o_values) == 0:
        acs_o_values = [10, 15, 20, 25]

    if args.dataset in ('bp', 'all'):
        print("=" * 70)
        print("BLOOD PRESSURE — New bootstrap experiment")
        print("=" * 70)
        bp_config = {
            'data_path':       str(base_dir / 'blood_pressure/data/bp_data.csv'),
            'B':               args.B_bp,
            'seed':            123,
            'alpha':           0.2,
            'alpha_selection': 0.5,
            'n_test_clinics':  15,
            'min_test_size':   10,
            'n_repeated':      50,
        }
        df, X, calib_grps, test_grps, gcol, o_vals = load_bp(bp_config)
        res = run_bootstrap(df, X, calib_grps, test_grps, gcol, o_vals, bp_config)
        save_outputs(
            res,
            plots_dir       = base_dir / 'blood_pressure/plots',
            results_dir     = base_dir / 'blood_pressure/results',
            tag             = 'bp',
            target_coverage = 0.8,
            title           = 'Blood Pressure — Bootstrap',
        )

    if args.dataset == 'bp_treated':
        print("=" * 70)
        print("BLOOD PRESSURE (TREATED ONLY) — 16 treated sites, 10 calib / 6 test, alpha=0.25")
        print("=" * 70)
        bp_config = {
            'data_path':       str(base_dir / 'blood_pressure/data/bp_data.csv'),
            'B':               args.B_bp,
            'seed':            123,
            'alpha':           0.25,
            'alpha_selection': 0.5,
            'treated_only':    True,
            'n_test_clinics':  6,
            'min_test_size':   10,
            'n_repeated':      50,
        }
        df, X, calib_grps, test_grps, gcol, o_vals = load_bp(bp_config)
        res = run_bootstrap(df, X, calib_grps, test_grps, gcol, o_vals, bp_config)
        save_outputs(
            res,
            plots_dir       = base_dir / 'blood_pressure/plots',
            results_dir     = base_dir / 'blood_pressure/results',
            tag             = 'bp_treated',
            target_coverage = 0.75,
            title           = 'BP Treated Only — 10 calib / 6 test, α=0.25',
        )

    if args.dataset in ('acs_top25', 'acs_puma', 'all'):
        print("=" * 70)
        print("ACS (single-state fixed PUMA groups) — Repeated experiment")
        print("=" * 70)
        age_min = None if args.acs_no_age_filter else args.acs_age_min
        age_max = None if args.acs_no_age_filter else args.acs_age_max
        acs_config = {
            'data_path':         str(base_dir / 'acs/data/acs_data_all50states.csv'),
            'B':                 args.B_acs,
            'seed':              456,
            'alpha':             0.2,
            'alpha_selection':   0.5,
            'acs_state':         args.acs_state.upper(),
            'n_puma_groups':     args.acs_n_groups,
            'min_puma_size':     args.acs_min_group_size,
            'age_min':           age_min,
            'age_max':           age_max,
            'min_hours':         args.acs_min_hours,
            'min_income':        args.acs_min_income,
            'yoep_min_year':     args.acs_min_yoep,
            'yoep_window_years': args.acs_yoep_window_years,
            'group_selection_seed': args.acs_group_seed,
            'top_income_quantile': None,
            'bottom_income_quantile': args.acs_bottom_income_quantile,
            'fixed_o_values':    acs_o_values,
            'n_repeated':        50,
            'enforce_fixed_split_counts': True,
            'expected_eligible_pumas': args.acs_expected_eligible_pumas,
            'expected_test_pumas': args.acs_expected_test_pumas,
            'resample_group_split_each_rep': True,
        }
        df, X, calib_grps, test_grps, gcol, o_vals = load_acs(acs_config)
        save_puma_summary_csv(
            df=df,
            group_col=gcol,
            eligible_groups=acs_config['eligible_groups'],
            output_csv=base_dir / 'acs/results/acs_puma_filtered_summary.csv',
        )
        res = run_bootstrap(df, X, calib_grps, test_grps, gcol, o_vals, acs_config)
        save_outputs(
            res,
            plots_dir       = base_dir / 'acs/plots',
            results_dir     = base_dir / 'acs/results',
            tag             = 'acs',
            target_coverage = 0.8,
            title           = f"ACS ({acs_config['acs_state']} fixed PUMA groups, alpha=0.2)",
            plot_style      = 'dgp',
        )
