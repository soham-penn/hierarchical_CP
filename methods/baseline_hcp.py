"""
Baseline HCP-Style Methods

Implements four baseline hierarchical conformal prediction methods:
  - HCP       : weighted quantile with all calibration scores + one infinity
  - Pooling   : weighted quantile with all calibration scores, no infinity
  - Subsampling     : one score sampled per group + infinity, uniform weights
  - Repeated Subsampling : mean of R independent single-subsampling quantiles
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scores import weighted_quantile


def compute_hcp_interval_radius(scores_list, alpha):
    """
    Compute the HCP interval radius.

    Each calibration group j contributes N_j scores with weight 1/((K+1)*N_j),
    and one infinity score has weight 1/(K+1).  Total weight = 1.

    Parameters
    ----------
    scores_list : list of array-like
        Conformity scores per calibration group.
    alpha : float
        Miscoverage level; returns the (1-alpha)-quantile.

    Returns
    -------
    float
        Interval radius (may be inf if scores_list is empty).
    """
    K = len(scores_list)
    if K == 0:
        return np.inf

    all_scores, all_weights = [], []
    for scores in scores_list:
        n = len(scores)
        if n > 0:
            all_scores.extend(scores)
            all_weights.extend([1.0 / ((K + 1) * n)] * n)

    all_scores.append(np.inf)
    all_weights.append(1.0 / (K + 1))

    return weighted_quantile(all_scores, all_weights, alpha)


def compute_pooling_interval_radius(scores_list, alpha):
    """
    Compute the Pooling interval radius.

    Each calibration group j contributes N_j scores with weight 1/(K*N_j).
    No infinity score is added.

    Parameters
    ----------
    scores_list : list of array-like
        Conformity scores per calibration group.
    alpha : float
        Miscoverage level; returns the (1-alpha)-quantile.

    Returns
    -------
    float
        Interval radius (inf if no scores).
    """
    K = len(scores_list)
    if K == 0:
        return np.inf

    all_scores, all_weights = [], []
    for scores in scores_list:
        n = len(scores)
        if n > 0:
            all_scores.extend(scores)
            all_weights.extend([1.0 / (K * n)] * n)

    if not all_scores:
        return np.inf

    return weighted_quantile(all_scores, all_weights, alpha)


def compute_subsampling_once_interval_radius(scores_list, alpha):
    """
    Compute the Subsampling (once) interval radius.

    Sample one score uniformly at random from each non-empty calibration group,
    append one infinity, and apply uniform weights 1/(K+1).

    Parameters
    ----------
    scores_list : list of array-like
        Conformity scores per calibration group.
    alpha : float
        Miscoverage level; returns the (1-alpha)-quantile.

    Returns
    -------
    float
        Interval radius.
    """
    K = len(scores_list)
    if K == 0:
        return np.inf

    sampled = [np.random.choice(s) for s in scores_list if len(s) > 0]
    sampled.append(np.inf)
    weights = np.ones(len(sampled)) / (K + 1)
    return weighted_quantile(sampled, weights, alpha)


def compute_repeated_subsampling_interval_radius(scores_list, alpha,
                                                 number_repetitions):
    """
    Compute the Repeated Subsampling interval radius.

    Runs `number_repetitions` independent single-subsampling draws and returns
    the mean of the resulting quantiles.  This matches the theoretical definition
    (average over repetitions) and guarantees the repeated estimate is <=
    the single-subsampling estimate in expectation (variance reduction, same mean).

    Parameters
    ----------
    scores_list : list of array-like
        Conformity scores per calibration group.
    alpha : float
        Miscoverage level; returns the (1-alpha)-quantile.
    number_repetitions : int
        Number of independent subsampling draws to average over.

    Returns
    -------
    float
        Interval radius.
    """
    K = len(scores_list)
    if K == 0 or number_repetitions <= 0:
        return np.inf

    quantiles = []
    for _ in range(number_repetitions):
        sampled = [np.random.choice(s) for s in scores_list if len(s) > 0]
        sampled.append(np.inf)
        w = np.ones(len(sampled)) / (K + 1)
        quantiles.append(weighted_quantile(sampled, w, alpha))

    return float(np.mean(quantiles))
