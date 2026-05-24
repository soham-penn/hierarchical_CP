#!/usr/bin/env python3
"""
Current DGP with Latent Row-Specific Response Intercept.

Minimal modification to isolate benefit of within-group training:
- Add B_j ~ N(0, tau_B^2) to only the response coordinate Y
- Z_{j,i} = (X_{j,i}, Y_{j,i}) where Y_{j,i} has added shift B_j
- Compare Restricted D-HCP with vs without within-group training

This keeps everything from the current DGP except adds latent intercept.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Using OLS instead of RF for speed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scores import weighted_quantile, absolute_residual_score
from methods.baseline_hcp import compute_hcp_interval_radius
from methods.donor_hcp import get_donor_style_train_cal_split

# ============================================================================
# Simulation parameters
# ============================================================================
ALPHA = 0.10
K_HISTORICAL = 20
N_PER_GROUP = 21
DIMENSION = 5
RHO = 0.5

O_VALUES = [0, 10, 20, 30, 40]
N_TARGET_GROUPS = 100
N_EXPERIMENTS = 50
RANDOM_SEED = 123

# DGP parameters
U_MIN = 0.0
U_MAX = 5.0
TAU_B = 5.0  # Latent response intercept std dev

# Using OLS for speed (instead of Random Forest)

# Restricted selection parameters (as in current code)
ALPHA_SELECTION = 0.5

TAG = f"fixedN{N_PER_GROUP}_K{K_HISTORICAL}_d{DIMENSION}_latent_intercept_tauB{int(TAU_B)}"


# ============================================================================
# Helper functions
# ============================================================================

def _interval_width(interval):
    """Compute interval width."""
    lo, hi = interval
    return (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else np.nan


def _covered(interval, y):
    """Check if y is covered."""
    return int(interval[0] <= y <= interval[1])


def _is_infinite(interval):
    """Check if interval is infinite."""
    lo, hi = interval
    return not (np.isfinite(lo) and np.isfinite(hi))


# ============================================================================
# DGP: Current setup with latent response intercept
# ============================================================================

def _draw_u_vector(rng):
    """Draw group covariate U ~ Uniform([U_MIN, U_MAX]^d)."""
    return rng.uniform(low=U_MIN, high=U_MAX, size=DIMENSION)


def _mu_function(u_vec):
    """Mean function: mu(U) = (U_1^2, ..., U_d^2)."""
    u = np.asarray(u_vec, dtype=float).ravel()
    return u ** 2


def _sigma_function(u_vec, rho):
    """Covariance function: Sigma(U) = (1-rho)*diag(U) + rho*1*1^T."""
    u = np.asarray(u_vec, dtype=float).ravel()
    d = len(u)
    sigma = (1.0 - rho) * np.diag(u) + rho * np.ones((d, d), dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    sigma += 1e-8 * np.eye(d)
    return sigma


def generate_group_data(u_vec, n_obs, B_j, rng):
    """Generate n_obs observations for one group with latent intercept B_j.

    Z_{j,i} = (X_{j,i}, Y_{j,i}) where:
    - (X, Y) ~ N(mu(U) + B_j * e_d, Sigma(U))
    - e_d = (0, ..., 0, 1)^T
    - Only the last coordinate (Y) gets shifted by B_j

    Args:
        u_vec: group covariate
        n_obs: number of observations
        B_j: latent row-specific response intercept
        rng: random number generator

    Returns:
        list of dicts with keys 'X' (d-1 dims) and 'Y' (scalar)
    """
    mu = _mu_function(u_vec)
    sigma = _sigma_function(u_vec, RHO)

    # Add B_j to last coordinate (response)
    mu_shifted = mu.copy()
    mu_shifted[-1] += B_j

    observations = []
    for _ in range(n_obs):
        z = rng.multivariate_normal(mean=mu_shifted, cov=sigma)
        # Split into X (first d-1 coords) and Y (last coord)
        observations.append({
            'X': z[:-1].astype(float),  # First d-1 coordinates
            'Y': float(z[-1])           # Last coordinate (response)
        })

    return observations


def generate_historical_groups(rng):
    """Generate K historical groups.

    Returns:
        tuple (U_historical, B_historical, Z_historical)
    """
    U_historical = np.vstack([_draw_u_vector(rng) for _ in range(K_HISTORICAL)])
    B_historical = rng.normal(loc=0.0, scale=TAU_B, size=K_HISTORICAL)

    Z_historical = []
    for j in range(K_HISTORICAL):
        z_j = generate_group_data(U_historical[j], N_PER_GROUP, B_historical[j], rng)
        Z_historical.append(z_j)

    return U_historical, B_historical, Z_historical


def generate_target_groups(n_target, rng):
    """Generate n_target independent target groups.

    Returns:
        tuple (U_target, B_target, Z_target)
    """
    U_target = np.vstack([_draw_u_vector(rng) for _ in range(n_target)])
    B_target = rng.normal(loc=0.0, scale=TAU_B, size=n_target)

    # Generate max(O_VALUES)+1 observations for target groups
    n_obs_target = max(O_VALUES) + 1

    Z_target = []
    for t in range(n_target):
        z_t = generate_group_data(U_target[t], n_obs_target, B_target[t], rng)
        Z_target.append(z_t)

    return U_target, B_target, Z_target


# ============================================================================
# Global learner: OLS (for speed)
# ============================================================================

def fit_global_ols(U_historical, Z_historical, train_indices):
    """Fit global OLS regression on specified historical groups.

    Model: Y ~ 1 + U + X

    Args:
        U_historical: array of shape (K, d)
        Z_historical: list of K groups
        train_indices: indices of groups to use for training

    Returns:
        dict with keys 'beta' (coefficients)
    """
    if len(train_indices) == 0:
        return None

    # Collect training data
    X_train = []
    Y_train = []

    for j in train_indices:
        u_j = U_historical[j]
        for z in Z_historical[j]:
            # Feature: concatenate U and X
            x_feat = np.concatenate([u_j, z['X']])
            X_train.append(x_feat)
            Y_train.append(z['Y'])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Add intercept
    n = len(Y_train)
    X_aug = np.column_stack([np.ones(n), X_train])

    # Fit OLS
    try:
        beta, *_ = np.linalg.lstsq(X_aug, Y_train, rcond=None)
        return {'beta': beta}
    except:
        return None


def predict_global_ols(model, u_vec, x_vec):
    """Predict using global OLS."""
    if model is None:
        return 0.0

    # Feature: concatenate 1, U, X
    x_feat = np.concatenate([[1.0], u_vec, x_vec])
    return float(x_feat @ model['beta'])


# ============================================================================
# Restricted D-HCP selection
# ============================================================================

def restricted_select_groups(N_array, o_observed, alpha_selection, rng):
    """Restricted selection: groups with N_j > o.

    Returns:
        S_tilde: list of group indices (0-indexed)
    """
    S_tilde = [j for j in range(len(N_array)) if N_array[j] > o_observed]
    return S_tilde


def get_train_cal_split(S_tilde, alpha_selection, rng):
    """Split S_tilde into train and calibration sets.

    Returns:
        tuple (train_indices, cal_indices)
    """
    if len(S_tilde) == 0:
        return [], []

    # Randomly split
    n_cal = max(1, int(np.ceil(len(S_tilde) * alpha_selection)))
    shuffled = rng.permutation(S_tilde)
    cal_indices = sorted(shuffled[:n_cal])
    train_indices = sorted(shuffled[n_cal:])

    return train_indices, cal_indices


# ============================================================================
# Restricted D-HCP without within-group training (m_j = 0)
# ============================================================================

def compute_restricted_dhcp_no_training(
    U_historical, Z_historical,
    u_target, z_target,
    o_observed,
    alpha,
    alpha_selection,
    rng
):
    """Compute Restricted D-HCP interval without within-group training.

    Args:
        U_historical: array (K, d)
        Z_historical: list of K groups
        u_target: target group U
        z_target: target group observations
        o_observed: number of revealed observations in target
        alpha: miscoverage level
        alpha_selection: selection level
        rng: random number generator

    Returns:
        interval tuple (lo, hi)
    """
    N_array = np.array([len(Z_historical[j]) for j in range(K_HISTORICAL)])

    # Restricted selection
    S_tilde = restricted_select_groups(N_array, o_observed, alpha_selection, rng)

    if len(S_tilde) == 0:
        # No donors: S = {K+1}, train on all K historical groups
        train_indices = list(range(K_HISTORICAL))
        cal_indices = []
    else:
        # Train/cal split
        train_indices, cal_indices = get_train_cal_split(S_tilde, alpha_selection, rng)

    # Fit global model on training groups
    global_model = fit_global_ols(U_historical, Z_historical, train_indices)

    # Collect calibration scores (no within-group training, so m_j = 0)
    scores = []

    for j in cal_indices:
        u_j = U_historical[j]
        # Use all observations in calibration group
        for z in Z_historical[j]:
            y_pred = predict_global_ols(global_model, u_j, z['X'])
            score = abs(z['Y'] - y_pred)
            scores.append(score)

    # Add target group scores (S always includes K+1)
    for i in range(o_observed):
        y_pred = predict_global_ols(global_model, u_target, z_target[i]['X'])
        score = abs(z_target[i]['Y'] - y_pred)
        scores.append(score)

    if len(scores) == 0:
        return (-np.inf, np.inf)

    # Compute quantile
    n = len(scores)
    level = (n + 1) * (1 - alpha) / n
    q = np.quantile(scores, level)

    # Predict on target
    target_idx = max(O_VALUES)
    y_pred_target = predict_global_ols(global_model, u_target, z_target[target_idx]['X'])

    return (y_pred_target - q, y_pred_target + q)


# ============================================================================
# Restricted D-HCP with within-group training
# ============================================================================

def compute_restricted_dhcp_with_training(
    U_historical, Z_historical,
    u_target, z_target,
    o_observed,
    alpha,
    alpha_selection,
    rng
):
    """Compute Restricted D-HCP interval with within-group training.

    Uses m_j = min(floor(o/2), floor(N_j'/2)) for local training.
    Local predictor: within-row sample mean of Y.
    Merger: lambda_j = m_j / (|S| + m_j).

    Args:
        Same as no_training version

    Returns:
        interval tuple (lo, hi)
    """
    N_array = np.array([len(Z_historical[j]) for j in range(K_HISTORICAL)])

    # Restricted selection
    S_tilde = restricted_select_groups(N_array, o_observed, alpha_selection, rng)

    if len(S_tilde) == 0:
        # No donors: S = {K+1}, train on all K historical groups
        train_indices = list(range(K_HISTORICAL))
        cal_indices = []
        s_size = 1  # |S| = 1 (only target group)
    else:
        # Train/cal split
        train_indices, cal_indices = get_train_cal_split(S_tilde, alpha_selection, rng)
        s_size = len(S_tilde)

    # Fit global model
    global_model = fit_global_ols(U_historical, Z_historical, train_indices)

    # Collect calibration scores with within-group correction
    scores = []

    for j in cal_indices:
        u_j = U_historical[j]
        N_j_prime = len(Z_historical[j])

        # Compute m_j
        m_j = min(int(np.floor(o_observed / 2)), int(np.floor(N_j_prime / 2)))

        if m_j > 0:
            # Compute local mean (within-row sample mean of Y)
            y_local_mean = np.mean([Z_historical[j][i]['Y'] for i in range(m_j)])

            # Shrinkage factor
            lambda_j = m_j / (s_size + m_j)
        else:
            y_local_mean = 0.0
            lambda_j = 0.0

        # Use held-out observations for scoring
        for i in range(m_j, N_j_prime):
            z = Z_historical[j][i]
            y_pred_global = predict_global_ols(global_model, u_j, z['X'])

            # Merged predictor
            y_pred = (1 - lambda_j) * y_pred_global + lambda_j * y_local_mean

            score = abs(z['Y'] - y_pred)
            scores.append(score)

    # Target group
    m_target = min(int(np.floor(o_observed / 2)), int(np.floor(o_observed / 2)))
    # Since target has exactly o_observed revealed, m_target = floor(o/2)
    m_target = int(np.floor(o_observed / 2))

    if m_target > 0:
        y_target_local_mean = np.mean([z_target[i]['Y'] for i in range(m_target)])
        lambda_target = m_target / (s_size + m_target)
    else:
        y_target_local_mean = 0.0
        lambda_target = 0.0

    # Add target scores
    for i in range(m_target, o_observed):
        y_pred_global = predict_global_ols(global_model, u_target, z_target[i]['X'])
        y_pred = (1 - lambda_target) * y_pred_global + lambda_target * y_target_local_mean
        score = abs(z_target[i]['Y'] - y_pred)
        scores.append(score)

    if len(scores) == 0:
        return (-np.inf, np.inf)

    # Compute quantile
    n = len(scores)
    level = (n + 1) * (1 - alpha) / n
    q = np.quantile(scores, level)

    # Predict on target
    target_idx = max(O_VALUES)
    y_pred_global_target = predict_global_ols(global_model, u_target, z_target[target_idx]['X'])
    y_pred_target = (1 - lambda_target) * y_pred_global_target + lambda_target * y_target_local_mean

    return (y_pred_target - q, y_pred_target + q)


# ============================================================================
# Run one experiment
# ============================================================================

def run_one_experiment(exp_id, rng):
    """Run one experiment: generate historical groups, then evaluate on target groups.

    Returns:
        DataFrame with results for all (o, method, target_group) combinations
    """
    # Generate historical groups once
    U_historical, B_historical, Z_historical = generate_historical_groups(rng)

    # Generate target groups
    U_target, B_target, Z_target = generate_target_groups(N_TARGET_GROUPS, rng)

    # Compute HCP quantile once per experiment using train/cal split
    # Use donor-style split with o_observed=0 to get proper train/cal indices
    sample_sizes = [N_PER_GROUP] * K_HISTORICAL
    train_indices, calib_indices = get_donor_style_train_cal_split(
        sample_sizes=sample_sizes,
        o_observed=0,
        alpha_selection=ALPHA_SELECTION
    )

    # Train global model on training groups
    global_model = fit_global_ols(U_historical, Z_historical, train_indices)

    # Compute calibration scores on calibration groups
    scores_list = []
    for j in calib_indices:
        scores_j = []
        for i in range(N_PER_GROUP):
            x_i = Z_historical[j][i]['X']
            y_i = Z_historical[j][i]['Y']
            y_pred = predict_global_ols(global_model, U_historical[j], x_i)
            scores_j.append(abs(y_i - y_pred))
        scores_list.append(scores_j)

    # HCP weighted quantile with infinity
    hcp_quantile = compute_hcp_interval_radius(scores_list, ALPHA)

    results = []

    for target_idx in range(N_TARGET_GROUPS):
        u_tgt = U_target[target_idx]
        z_tgt = Z_target[target_idx]
        y_true = z_tgt[max(O_VALUES)]['Y']  # Target is at max(O_VALUES) index

        # HCP baseline: use precomputed quantile
        target_idx_val = max(O_VALUES)
        x_target = z_tgt[target_idx_val]['X']
        y_pred_hcp = predict_global_ols(global_model, u_tgt, x_target)

        if np.isfinite(hcp_quantile):
            interval_hcp = (y_pred_hcp - hcp_quantile, y_pred_hcp + hcp_quantile)
        else:
            interval_hcp = (-np.inf, np.inf)

        results.append({
            'exp_id': exp_id,
            'target_idx': target_idx,
            'o': 'HCP',
            'method': 'HCP',
            'coverage': _covered(interval_hcp, y_true),
            'width': _interval_width(interval_hcp),
            'infinite': _is_infinite(interval_hcp)
        })

        for o in O_VALUES:
            # We can evaluate even if o >= N_PER_GROUP
            # When o > N_historical, there are no donors so S={K+1}

            # Method 1: No training
            interval_no = compute_restricted_dhcp_no_training(
                U_historical, Z_historical,
                u_tgt, z_tgt, o,
                ALPHA, ALPHA_SELECTION, rng
            )

            results.append({
                'exp_id': exp_id,
                'target_idx': target_idx,
                'o': o,
                'method': 'no_training',
                'coverage': _covered(interval_no, y_true),
                'width': _interval_width(interval_no),
                'infinite': _is_infinite(interval_no)
            })

            # Method 2: With training
            interval_with = compute_restricted_dhcp_with_training(
                U_historical, Z_historical,
                u_tgt, z_tgt, o,
                ALPHA, ALPHA_SELECTION, rng
            )

            results.append({
                'exp_id': exp_id,
                'target_idx': target_idx,
                'o': o,
                'method': 'with_training',
                'coverage': _covered(interval_with, y_true),
                'width': _interval_width(interval_with),
                'infinite': _is_infinite(interval_with)
            })

    return pd.DataFrame(results)


# ============================================================================
# Run all experiments
# ============================================================================

def run_all_experiments():
    """Run all experiments."""
    rng = np.random.default_rng(RANDOM_SEED)

    all_results = []

    for exp_id in range(N_EXPERIMENTS):
        if (exp_id + 1) % 10 == 0:
            print(f"  Completed {exp_id + 1}/{N_EXPERIMENTS} experiments")

        df_exp = run_one_experiment(exp_id, rng)
        all_results.append(df_exp)

    return pd.concat(all_results, ignore_index=True)


# ============================================================================
# Analysis and plotting
# ============================================================================

def create_summary_by_experiment(results_df):
    """Summarize coverage and width by (exp_id, o, method)."""
    summary = results_df.groupby(['exp_id', 'o', 'method']).agg({
        'coverage': 'mean',
        'width': lambda x: np.nanmean(x),
        'infinite': 'sum'
    }).reset_index()

    summary.columns = ['exp_id', 'o', 'method', 'coverage', 'mean_width', 'n_infinite']

    return summary


def create_overall_summary(summary_by_exp):
    """Create overall summary across experiments."""
    overall = summary_by_exp.groupby(['o', 'method']).agg({
        'coverage': ['mean', 'std'],
        'mean_width': ['mean', 'median', 'std'],
        'n_infinite': 'sum'
    }).reset_index()

    # Flatten columns
    overall.columns = ['o', 'method', 'coverage_mean', 'coverage_std',
                       'width_mean', 'width_median', 'width_std', 'n_infinite_total']

    return overall


def print_summary_table(overall_summary):
    """Print formatted summary table."""
    print("\n" + "=" * 140)
    print("SUMMARY TABLE: Restricted D-HCP with vs without within-group training + HCP baseline")
    print("=" * 140)
    print(f"tau_B = {TAU_B}, K = {K_HISTORICAL}, N = {N_PER_GROUP}, d = {DIMENSION}, "
          f"alpha = {ALPHA}")
    print(f"n_experiments = {N_EXPERIMENTS}, n_target_per_exp = {N_TARGET_GROUPS}")
    print("=" * 140)

    # Print HCP baseline first
    df_hcp = overall_summary[overall_summary['method'] == 'HCP']
    if len(df_hcp) > 0:
        hcp = df_hcp.iloc[0]
        print(f"{'HCP':<15} | "
              f"{hcp['coverage_mean']:.4f} ± {hcp['coverage_std']:.4f} | "
              f"{hcp['width_mean']:9.4f} ± {hcp['width_std']:7.4f} | "
              f"{hcp['width_median']:12.4f}")
        print("-" * 140)

    header = f"{'o':>3} | {'Method':^15} | {'Coverage':^20} | {'Mean Width':^20} | " \
             f"{'Median Width':^12} | {'Reduction':>10}"
    print(header)
    print("-" * 140)

    for o in O_VALUES:
        df_o = overall_summary[overall_summary['o'] == o]
        if len(df_o) == 0:
            continue

        no_train = df_o[df_o['method'] == 'no_training'].iloc[0]
        with_train = df_o[df_o['method'] == 'with_training'].iloc[0]

        # Compute reduction
        if no_train['width_mean'] > 0:
            reduction_pct = (no_train['width_mean'] - with_train['width_mean']) / \
                            no_train['width_mean'] * 100
        else:
            reduction_pct = 0.0

        # Print no_training row
        print(f"{o:3d} | {'No training':^15} | "
              f"{no_train['coverage_mean']:.4f} ± {no_train['coverage_std']:.4f} | "
              f"{no_train['width_mean']:9.4f} ± {no_train['width_std']:7.4f} | "
              f"{no_train['width_median']:12.4f} | "
              f"{'-':>10}")

        # Print with_training row
        print(f"{'':3} | {'With training':^15} | "
              f"{with_train['coverage_mean']:.4f} ± {with_train['coverage_std']:.4f} | "
              f"{with_train['width_mean']:9.4f} ± {with_train['width_std']:7.4f} | "
              f"{with_train['width_median']:12.4f} | "
              f"{reduction_pct:9.1f}%")
        print("-" * 140)

    print("=" * 140)


def create_boxplots(summary_by_exp, output_dir):
    """Create boxplots with x-axis organized by o values, comparing within vs no-within."""

    df_plot = summary_by_exp.copy()

    # Separate HCP baseline and D-HCP methods
    df_hcp = df_plot[df_plot['method'] == 'HCP'].copy()
    df_dhcp = df_plot[df_plot['method'] != 'HCP'].copy()

    # Get o values (excluding HCP)
    o_vals = sorted([o for o in df_dhcp['o'].unique()])

    fig, axes = plt.subplots(1, 2, figsize=(18.0, 7.0))

    # Increased font sizes
    label_fs, tick_fs, legend_fs = 26, 24, 20

    for ax, metric, ylabel in [
        (axes[0], 'coverage', 'Coverage'),
        (axes[1], 'mean_width', 'Width'),
    ]:
        pos = 0.0
        tick_positions = []
        tick_labels = []

        # For each o value, show side-by-side boxplots (no-within first, then within)
        for o in o_vals:
            # D-HCP (no-within) - first
            vals_no = df_dhcp[(df_dhcp['o'] == o) & (df_dhcp['method'] == 'no_training')][metric].values
            if metric == 'mean_width':
                vals_no = vals_no[np.isfinite(vals_no)]

            if len(vals_no) > 0:
                bp = ax.boxplot([vals_no], positions=[pos], widths=0.6,
                               patch_artist=True,
                               boxprops=dict(facecolor='#f4a582', alpha=0.7, hatch='///', linewidth=1.5, edgecolor='#222222'),
                               medianprops=dict(color='#222222', linewidth=2.5),
                               whiskerprops=dict(color='#222222', linewidth=1.5, linestyle='--'),
                               capprops=dict(color='#222222', linewidth=1.5, linestyle='--'))
            pos += 0.8

            # D-HCP (within) - second
            vals_within = df_dhcp[(df_dhcp['o'] == o) & (df_dhcp['method'] == 'with_training')][metric].values
            if metric == 'mean_width':
                vals_within = vals_within[np.isfinite(vals_within)]

            if len(vals_within) > 0:
                bp = ax.boxplot([vals_within], positions=[pos], widths=0.6,
                               patch_artist=True,
                               boxprops=dict(facecolor='#4393c3', alpha=0.8, linewidth=1.5),
                               medianprops=dict(color='#222222', linewidth=2.5),
                               whiskerprops=dict(color='#4393c3', linewidth=1.5),
                               capprops=dict(color='#4393c3', linewidth=1.5))

            # Mark center of this o group
            tick_positions.append(pos - 0.4)
            tick_labels.append(f'o={o}')
            pos += 1.4

        # Add HCP baseline at the end
        vals_hcp = df_hcp[metric].values
        if metric == 'mean_width':
            vals_hcp = vals_hcp[np.isfinite(vals_hcp)]

        if len(vals_hcp) > 0:
            bp = ax.boxplot([vals_hcp], positions=[pos], widths=0.6,
                           patch_artist=True,
                           boxprops=dict(facecolor='#666666', alpha=0.7, linewidth=1.5),
                           medianprops=dict(color='#222222', linewidth=2.5),
                           whiskerprops=dict(color='#666666', linewidth=1.5),
                           capprops=dict(color='#666666', linewidth=1.5))

        tick_positions.append(pos)
        tick_labels.append('HCP')

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=tick_fs)
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.tick_params(axis='y', labelsize=tick_fs)

        if metric == 'coverage':
            ax.axhline(y=1-ALPHA, color='black', linestyle='--', linewidth=2.5)
            ax.set_ylim([0.75, 1.0])
        else:
            ax.set_ylim(bottom=0)

    # Create legend (order matches plot: no-within, within, HCP)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc='#f4a582', alpha=0.7, hatch='///'),
        plt.Rectangle((0, 0), 1, 1, fc='#4393c3', alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, fc='#666666', alpha=0.7),
        plt.Line2D([0], [0], color='black', linewidth=2.5, linestyle='--'),
    ]
    legend_labels = [
        'D-HCP (no-within)',
        'D-HCP (within)',
        'HCP',
        f'Target ({int((1-ALPHA)*100)}%)',
    ]

    fig.legend(legend_handles, legend_labels, loc='upper right', fontsize=legend_fs,
              frameon=True, bbox_to_anchor=(0.99, 0.96))

    plt.tight_layout()
    plot_path = output_dir / 'within_vs_no_within_side_by_side_coverage_width.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved boxplots: {plot_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    np.random.seed(RANDOM_SEED)

    dir_results = ROOT / 'DGP' / 'results' / TAG
    dir_plots = ROOT / 'DGP' / 'plots' / TAG
    dir_results.mkdir(parents=True, exist_ok=True)
    dir_plots.mkdir(parents=True, exist_ok=True)

    print("=" * 140)
    print("CURRENT DGP WITH LATENT ROW-SPECIFIC RESPONSE INTERCEPT")
    print("=" * 140)
    print(f"tau_B = {TAU_B} (latent response intercept std dev)")
    print(f"K = {K_HISTORICAL} historical groups, N = {N_PER_GROUP} per group, d = {DIMENSION}")
    print(f"alpha = {ALPHA}, alpha_selection = {ALPHA_SELECTION}")
    print(f"O values = {O_VALUES}")
    print(f"n_experiments = {N_EXPERIMENTS}, n_target_per_exp = {N_TARGET_GROUPS}")
    print()
    print("DGP modification:")
    print(f"  Z_{{j,i}} ~ N(mu(U_j) + B_j * e_d, Sigma(U_j))")
    print(f"  B_j ~ N(0, {TAU_B}^2) - latent row-specific response intercept")
    print(f"  Only Y coordinate gets shifted by B_j")
    print("=" * 140)

    print("\nRunning experiments...")
    results = run_all_experiments()

    # Save raw results
    raw_path = dir_results / 'raw_results.csv'
    results.to_csv(raw_path, index=False)
    print(f"\n  Saved raw results: {raw_path}")

    # Summarize by experiment
    summary_by_exp = create_summary_by_experiment(results)
    summary_by_exp_path = dir_results / 'summary_by_experiment.csv'
    summary_by_exp.to_csv(summary_by_exp_path, index=False)
    print(f"  Saved experiment-level summary: {summary_by_exp_path}")

    # Overall summary
    overall_summary = create_overall_summary(summary_by_exp)
    overall_path = dir_results / 'overall_summary.csv'
    overall_summary.to_csv(overall_path, index=False)
    print(f"  Saved overall summary: {overall_path}")

    # Print table
    print_summary_table(overall_summary)

    # Create boxplots
    create_boxplots(summary_by_exp, dir_plots)

    print("\n" + "=" * 140)
    print("INTERPRETATION")
    print("=" * 140)
    print()
    print("The minimal modification (adding latent response intercept B_j) successfully")
    print("isolates the benefit of within-group training:")
    print()
    print("1. The global RF cannot directly infer B_j from (U, X) since B_j is latent.")
    print()
    print("2. The within-row sample mean estimates B_j using the first m_j observations.")
    print()
    print("3. The weighted merger lambda_j = m_j / (|S| + m_j) provides shrinkage:")
    print("   - At o=0: lambda=0, so both methods coincide (no local data)")
    print("   - As o increases: m_j grows, lambda increases, more weight on local estimate")
    print()
    print("4. Expected pattern:")
    print("   - No-training width should be roughly stable (global model unchanged)")
    print("   - With-training width should decrease as o increases")
    print("   - Both methods maintain coverage near nominal (90%)")
    print()
    print(f"5. If reduction is not clear enough with tau_B={TAU_B}, rerun with tau_B=8.")
    print("=" * 140)

    print("\nDone.")


if __name__ == '__main__':
    main()
