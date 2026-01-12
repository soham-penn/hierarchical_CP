"""
Baseline HCP-Style Methods

This module implements baseline hierarchical conformal prediction methods:
- HCP (standard hierarchical CP)
- Pooling
- Subsampling (once)
- Repeated subsampling
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scores import weighted_quantile


def compute_hcp_interval_radius(scores_list, alpha):
    """
    Compute HCP interval radius using all calibration scores with uniform weighting.

    Parameters:
    -----------
    scores_list : list of arrays
        List where scores_list[k] contains scores for calibration group k
    alpha : float
        Miscoverage level

    Returns:
    --------
    float : Interval radius
    """
    K = len(scores_list)
    Nk = np.array([len(scores) for scores in scores_list])

    # Combine all scores
    all_scores = []
    all_weights = []

    for k in range(K):
        nk = Nk[k]
        if nk > 0:
            all_scores.extend(scores_list[k])
            all_weights.extend([1.0 / ((K + 1) * nk)] * nk)

    # Add infinity with weight 1/(K+1)
    all_scores.append(np.inf)
    all_weights.append(1.0 / (K + 1))

    return weighted_quantile(all_scores, all_weights, alpha)


def compute_pooling_interval_radius(scores_list, alpha):
    """
    Compute pooling interval radius (no infinity score).

    Parameters:
    -----------
    scores_list : list of arrays
        List where scores_list[k] contains scores for calibration group k
    alpha : float
        Miscoverage level

    Returns:
    --------
    float : Interval radius
    """
    K = len(scores_list)
    Nk = np.array([len(scores) for scores in scores_list])

    # Combine all scores without infinity
    all_scores = []
    all_weights = []

    for k in range(K):
        nk = Nk[k]
        if nk > 0:
            all_scores.extend(scores_list[k])
            all_weights.extend([1.0 / (K * nk)] * nk)

    if len(all_scores) == 0:
        return np.inf

    return weighted_quantile(all_scores, all_weights, alpha)


def compute_subsampling_once_interval_radius(scores_list, alpha):
    """
    Compute subsampling interval radius (sample one score from each group).

    Parameters:
    -----------
    scores_list : list of arrays
        List where scores_list[k] contains scores for calibration group k
    alpha : float
        Miscoverage level

    Returns:
    --------
    float : Interval radius
    """
    K = len(scores_list)
    if K == 0:
        return np.inf

    # Sample one score from each group
    sampled_scores = []
    for scores in scores_list:
        if len(scores) > 0:
            sampled_scores.append(np.random.choice(scores))

    # Add infinity
    sampled_scores.append(np.inf)

    # Uniform weights
    weights = np.ones(len(sampled_scores)) / (K + 1)

    return weighted_quantile(sampled_scores, weights, alpha)


def compute_repeated_subsampling_interval_radius(scores_list, alpha,
                                                 number_repetitions):
    """
    Compute repeated subsampling interval radius.

    Parameters:
    -----------
    scores_list : list of arrays
        List where scores_list[k] contains scores for calibration group k
    alpha : float
        Miscoverage level
    number_repetitions : int
        Number of times to repeat the subsampling

    Returns:
    --------
    float : Interval radius
    """
    K = len(scores_list)
    if K == 0 or number_repetitions <= 0:
        return np.inf

    # Sample scores repeatedly
    sampled_scores = []
    for _ in range(number_repetitions):
        for scores in scores_list:
            if len(scores) > 0:
                sampled_scores.append(np.random.choice(scores))

    # Add infinity
    sampled_scores.append(np.inf)

    # Weights
    n_samples = K * number_repetitions
    weights = np.array([1.0 / (number_repetitions * (K + 1))] * n_samples +
                      [1.0 / (K + 1)])

    return weighted_quantile(sampled_scores, weights, alpha)
