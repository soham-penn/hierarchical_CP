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


def _resolve_rng(rng=None, random_seed=None):
    if rng is not None:
        return rng
    if random_seed is not None:
        return np.random.default_rng(random_seed)
    return None


def _aggregate_weighted_support(values, weights, atol=1e-12):
    """
    Sort scores and sum weights at tied values.

    Returns unique support points, weight mass per point, and cumulative mass.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]

    unique_vals = []
    unique_weights = []
    current_val = values[0]
    current_w = float(weights[0])

    for v, w in zip(values[1:], weights[1:]):
        if v == current_val:
            current_w += float(w)
        else:
            unique_vals.append(current_val)
            unique_weights.append(current_w)
            current_val = v
            current_w = float(w)
    unique_vals.append(current_val)
    unique_weights.append(current_w)

    unique_vals = np.asarray(unique_vals, dtype=float)
    unique_weights = np.asarray(unique_weights, dtype=float)
    total_w = float(np.sum(unique_weights))
    if total_w <= atol:
        return unique_vals, unique_weights, np.zeros_like(unique_weights)

    unique_weights = unique_weights / total_w
    cdf = np.cumsum(unique_weights)
    return unique_vals, unique_weights, cdf


def randomized_weighted_quantile(
    scores,
    weights=None,
    alpha=None,
    beta=None,
    random_seed=None,
    rng=None,
    return_info=False,
    atol=1e-12,
):
    """
    Randomized weighted conformal quantile at target coverage beta = 1 - alpha.

    If beta lies in a jump of the weighted empirical CDF, the deterministic
    conformal quantile is conservative. Randomizing between the previous and
    current threshold with probability
        gamma = (beta - F(q^-)) / (F(q) - F(q^-))
    gives exact target coverage over the auxiliary randomization.

    Parameters
    ----------
    scores : array-like
        Conformity scores (may include ``np.inf``).
    weights : array-like or None
        Non-negative weights; normalized internally. Equal weights if None.
    alpha : float or None
        Miscoverage level; used when ``beta`` is None (beta = 1 - alpha).
    beta : float or None
        Target cumulative mass (coverage level).
    random_seed, rng : optional
        Randomness for threshold selection only (``np.random.default_rng``).
    return_info : bool
        If True, return a dict with diagnostic fields; otherwise return q only.
    atol : float
        Numerical tolerance for jump / boundary checks.

    Returns
    -------
    float or dict
        Randomized threshold ``q`` (may be ``np.inf``), or info dict if
        ``return_info=True``.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        info = {
            "q_randomized": np.inf,
            "q_upper": np.inf,
            "q_lower": np.nan,
            "gamma_upper": np.nan,
            "cdf_lower": 0.0,
            "cdf_upper": 1.0,
            "beta": np.nan if beta is None else float(beta),
            "randomized": False,
            "quantile_mode": "randomized",
        }
        return info if return_info else np.inf

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if scores.shape != weights.shape:
        raise ValueError("randomized_weighted_quantile: length mismatch")

    if np.any(weights < -atol):
        raise ValueError("randomized_weighted_quantile: negative weights not allowed")

    if beta is None:
        if alpha is None:
            raise ValueError("randomized_weighted_quantile: need alpha or beta")
        beta = 1.0 - float(alpha)
    else:
        beta = float(beta)

    total_w = float(np.sum(weights))
    if total_w <= atol:
        info = {
            "q_randomized": np.inf,
            "q_upper": np.inf,
            "q_lower": np.nan,
            "gamma_upper": np.nan,
            "cdf_lower": 0.0,
            "cdf_upper": 1.0,
            "beta": beta,
            "randomized": False,
            "quantile_mode": "randomized",
        }
        return info if return_info else np.inf

    weights = weights / total_w

    support, masses, cdf = _aggregate_weighted_support(scores, weights, atol=atol)
    if support.size == 0:
        info = {
            "q_randomized": np.inf,
            "q_upper": np.inf,
            "q_lower": np.nan,
            "gamma_upper": np.nan,
            "cdf_lower": 0.0,
            "cdf_upper": 1.0,
            "beta": beta,
            "randomized": False,
            "quantile_mode": "randomized",
        }
        return info if return_info else np.inf

    j = int(np.searchsorted(cdf, beta, side="left"))
    if j >= support.size:
        j = support.size - 1

    b = float(cdf[j])
    a = float(cdf[j - 1]) if j > 0 else 0.0
    q_upper = float(support[j])
    q_lower = float(support[j - 1]) if j > 0 else 0.0

    randomized = False
    gamma_upper = 1.0

    if beta <= a + atol:
        q = q_lower
        gamma_upper = 0.0
    elif b - a <= atol:
        q = q_upper
        gamma_upper = 1.0
    else:
        gamma_upper = (beta - a) / (b - a)
        gamma_upper = float(np.clip(gamma_upper, 0.0, 1.0))
        if gamma_upper <= atol:
            q = q_lower
        elif gamma_upper >= 1.0 - atol:
            q = q_upper
        else:
            randomized = True
            local_rng = _resolve_rng(rng=rng, random_seed=random_seed)
            if local_rng is None:
                raise ValueError(
                    "randomized_weighted_quantile: rng or random_seed required "
                    "when beta lies inside a CDF jump"
                )
            q = q_upper if local_rng.random() < gamma_upper else q_lower

    info = {
        "q_randomized": q,
        "q_upper": q_upper,
        "q_lower": q_lower,
        "gamma_upper": gamma_upper,
        "cdf_lower": a,
        "cdf_upper": b,
        "beta": beta,
        "randomized": randomized,
        "quantile_mode": "randomized",
        "randomized_threshold": randomized,
    }
    return info if return_info else q


def conformal_threshold(
    scores,
    weights=None,
    alpha=None,
    beta=None,
    quantile_mode="deterministic",
    random_seed=None,
    rng=None,
    return_info=False,
    atol=1e-12,
):
    """
    Select a conformal threshold using deterministic or randomized weighted quantiles.

    Parameters
    ----------
    quantile_mode : {"deterministic", "randomized"}
        ``deterministic`` uses :func:`weighted_quantile`; ``randomized`` uses
        :func:`randomized_weighted_quantile`.
    """
    if quantile_mode not in ("deterministic", "randomized"):
        raise ValueError(
            f"conformal_threshold: unknown quantile_mode={quantile_mode!r}"
        )

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        info = {
            "q_randomized": np.inf,
            "q_upper": np.inf,
            "q_lower": np.nan,
            "gamma_upper": np.nan,
            "cdf_lower": 0.0,
            "cdf_upper": 1.0,
            "beta": np.nan if beta is None and alpha is None else (
                float(beta) if beta is not None else 1.0 - float(alpha)
            ),
            "randomized": False,
            "quantile_mode": quantile_mode,
            "randomized_threshold": False,
        }
        return info if return_info else np.inf

    if alpha is None and beta is None:
        raise ValueError("conformal_threshold: need alpha or beta")

    if quantile_mode == "randomized":
        return randomized_weighted_quantile(
            scores=scores,
            weights=weights,
            alpha=alpha,
            beta=beta,
            random_seed=random_seed,
            rng=rng,
            return_info=return_info,
            atol=atol,
        )

    if weights is None:
        weights = np.ones(scores.size, dtype=float)

    q = weighted_quantile(scores, weights, alpha)
    if not return_info:
        return q

    beta_val = float(beta) if beta is not None else 1.0 - float(alpha)
    info = {
        "q_randomized": q,
        "q_upper": q,
        "q_lower": q,
        "gamma_upper": 1.0,
        "cdf_lower": beta_val,
        "cdf_upper": beta_val,
        "beta": beta_val,
        "randomized": False,
        "quantile_mode": "deterministic",
        "randomized_threshold": False,
    }
    return info


def make_quantile_seed(base_seed, experiment_id=0, target_id=0, o=0, method=""):
    """
    Deterministic seed for conformal threshold randomization in experiments.
    """
    method_hash = sum(ord(c) for c in str(method)) % 10007
    return int(base_seed) + 1000003 * int(experiment_id) + 10007 * int(target_id) + 101 * int(o) + method_hash


def merge_quantile_info(result_dict, quantile_info):
    """Attach quantile diagnostics to a method result dictionary."""
    if quantile_info is None:
        return result_dict
    if isinstance(quantile_info, dict):
        for key in (
            "q_randomized",
            "q_lower",
            "q_upper",
            "gamma_upper",
            "cdf_lower",
            "cdf_upper",
            "beta",
            "randomized",
            "randomized_threshold",
            "quantile_mode",
        ):
            if key in quantile_info:
                result_dict[key] = quantile_info[key]
    return result_dict
