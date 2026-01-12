"""
Score Functions and Weighted Quantile Computation

This module defines conformity score functions and weighted quantile computation.
"""

import numpy as np


def absolute_residual_score(y, mu):
    """
    Compute absolute residual score: |y - mu|

    Parameters:
    -----------
    y : array-like or float
        Observed values
    mu : array-like or float
        Predicted values

    Returns:
    --------
    array-like or float : Absolute residuals
    """
    return np.abs(y - mu)


def weighted_quantile(values, weights, alpha):
    """
    Compute weighted quantile using the weighted empirical CDF approach.

    This function computes the (1-alpha)-quantile of the weighted distribution.

    Parameters:
    -----------
    values : array-like
        Values to compute quantile from
    weights : array-like
        Weights for each value (must be non-negative)
    alpha : float
        Miscoverage level (returns 1-alpha quantile)

    Returns:
    --------
    float : The weighted quantile
    """
    if len(values) == 0:
        return np.inf

    values = np.asarray(values)
    weights = np.asarray(weights)

    if len(values) != len(weights):
        raise ValueError("weighted_quantile: length mismatch between values and weights")

    if np.any(weights < 0):
        raise ValueError("weighted_quantile: negative weights not allowed")

    # Sort values and corresponding weights
    sort_indices = np.argsort(values)
    sorted_values = values[sort_indices]
    sorted_weights = weights[sort_indices]

    # Compute cumulative sum of weights
    cumsum_weights = np.cumsum(sorted_weights)

    # Find the smallest value where cumsum >= 1 - alpha
    threshold = 1 - alpha
    idx = np.searchsorted(cumsum_weights, threshold, side='left')

    if idx >= len(sorted_values):
        return np.inf

    return sorted_values[idx]
