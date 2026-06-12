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
from scores import conformal_threshold


def _threshold_from_scores(scores, weights, alpha, quantile_mode="deterministic",
                           random_seed=None, rng=None, return_info=False):
    return conformal_threshold(
        scores=scores,
        weights=weights,
        alpha=alpha,
        quantile_mode=quantile_mode,
        random_seed=random_seed,
        rng=rng,
        return_info=return_info,
    )


def compute_hcp_interval_radius(
    scores_list,
    alpha,
    quantile_mode="deterministic",
    random_seed=None,
    rng=None,
    return_info=False,
):
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
    quantile_mode : {"deterministic", "randomized"}
        How to choose the conformal threshold from weighted scores.
    random_seed, rng : optional
        Randomness for randomized threshold selection only.
    return_info : bool
        If True, return a dict with threshold diagnostics.

    Returns
    -------
    float or dict
        Interval radius (may be inf if scores_list is empty), or info dict.
    """
    K = len(scores_list)
    if K == 0:
        out = _threshold_from_scores([], [], alpha, quantile_mode=quantile_mode,
                                    random_seed=random_seed, rng=rng, return_info=True)
        return out if return_info else out["q_randomized"]

    all_scores, all_weights = [], []
    for scores in scores_list:
        n = len(scores)
        if n > 0:
            all_scores.extend(scores)
            all_weights.extend([1.0 / ((K + 1) * n)] * n)

    all_scores.append(np.inf)
    all_weights.append(1.0 / (K + 1))

    return _threshold_from_scores(
        all_scores, all_weights, alpha,
        quantile_mode=quantile_mode,
        random_seed=random_seed,
        rng=rng,
        return_info=return_info,
    )


def compute_pooling_interval_radius(
    scores_list,
    alpha,
    quantile_mode="deterministic",
    random_seed=None,
    rng=None,
    return_info=False,
):
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
    quantile_mode : {"deterministic", "randomized"}
    random_seed, rng : optional
    return_info : bool

    Returns
    -------
    float or dict
        Interval radius (inf if no scores), or info dict.
    """
    K = len(scores_list)
    if K == 0:
        out = _threshold_from_scores([], [], alpha, quantile_mode=quantile_mode,
                                    random_seed=random_seed, rng=rng, return_info=True)
        return out if return_info else out["q_randomized"]

    all_scores, all_weights = [], []
    for scores in scores_list:
        n = len(scores)
        if n > 0:
            all_scores.extend(scores)
            all_weights.extend([1.0 / (K * n)] * n)

    if not all_scores:
        out = _threshold_from_scores([], [], alpha, quantile_mode=quantile_mode,
                                    random_seed=random_seed, rng=rng, return_info=True)
        return out if return_info else out["q_randomized"]

    return _threshold_from_scores(
        all_scores, all_weights, alpha,
        quantile_mode=quantile_mode,
        random_seed=random_seed,
        rng=rng,
        return_info=return_info,
    )


def compute_subsampling_once_interval_radius(
    scores_list,
    alpha,
    quantile_mode="deterministic",
    random_seed=None,
    rng=None,
    return_info=False,
):
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
    quantile_mode : {"deterministic", "randomized"}
    random_seed, rng : optional
        ``rng`` is also used for per-group score subsampling when provided.
    return_info : bool

    Returns
    -------
    float or dict
        Interval radius.
    """
    K = len(scores_list)
    if K == 0:
        out = _threshold_from_scores([], [], alpha, quantile_mode=quantile_mode,
                                    random_seed=random_seed, rng=rng, return_info=True)
        return out if return_info else out["q_randomized"]

    sample_rng = _resolve_sample_rng(rng, random_seed)
    sampled = [sample_rng.choice(s) for s in scores_list if len(s) > 0]
    sampled.append(np.inf)
    weights = np.ones(len(sampled)) / (K + 1)
    return _threshold_from_scores(
        sampled, weights, alpha,
        quantile_mode=quantile_mode,
        random_seed=random_seed,
        rng=rng,
        return_info=return_info,
    )


def compute_repeated_subsampling_interval_radius(
    scores_list,
    alpha,
    number_repetitions,
    quantile_mode="deterministic",
    random_seed=None,
    rng=None,
    return_info=False,
):
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
    quantile_mode : {"deterministic", "randomized"}
    random_seed, rng : optional
    return_info : bool

    Returns
    -------
    float or dict
        Interval radius (mean quantile; info dict summarizes last draw only).
    """
    K = len(scores_list)
    if K == 0 or number_repetitions <= 0:
        out = _threshold_from_scores([], [], alpha, quantile_mode=quantile_mode,
                                    random_seed=random_seed, rng=rng, return_info=True)
        return out if return_info else out["q_randomized"]

    sample_rng = _resolve_sample_rng(rng, random_seed)
    quantiles = []
    last_info = None
    for rep in range(number_repetitions):
        sampled = [sample_rng.choice(s) for s in scores_list if len(s) > 0]
        sampled.append(np.inf)
        w = np.ones(len(sampled)) / (K + 1)
        q_seed = None if random_seed is None else int(random_seed) + 1009 * rep
        q_out = _threshold_from_scores(
            sampled, w, alpha,
            quantile_mode=quantile_mode,
            random_seed=q_seed,
            rng=rng,
            return_info=return_info,
        )
        if return_info:
            quantiles.append(q_out["q_randomized"])
            last_info = q_out
        else:
            quantiles.append(q_out)

    mean_q = float(np.mean(quantiles))
    if not return_info:
        return mean_q

    if last_info is None:
        last_info = {}
    last_info = dict(last_info)
    last_info["q_randomized"] = mean_q
    last_info["repeated_subsampling_mean"] = True
    return last_info


def _resolve_sample_rng(rng, random_seed):
    if rng is not None:
        return rng
    if random_seed is not None:
        return np.random.default_rng(random_seed)
    return np.random.default_rng()
