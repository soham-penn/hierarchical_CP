#!/usr/bin/env python3
"""Diagnostics for randomized weighted conformal quantile selection."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scores import randomized_weighted_quantile, weighted_quantile, conformal_threshold


def test_beta_on_cdf_knot_matches_deterministic():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4) / 4.0
    alpha = 0.5
    beta = 0.5
    det = weighted_quantile(scores, weights, alpha)
    info = randomized_weighted_quantile(
        scores, weights, beta=beta, random_seed=0, return_info=True
    )
    assert info["randomized"] is False
    assert info["q_randomized"] == det


def test_inf_jump_gamma_half():
    scores = np.array([1.0, 2.0, np.inf])
    weights = np.array([0.4, 0.4, 0.2])
    beta = 0.9
    info = randomized_weighted_quantile(
        scores, weights, beta=beta, random_seed=0, return_info=True
    )
    assert np.isclose(info["cdf_lower"], 0.8)
    assert np.isclose(info["cdf_upper"], 1.0)
    assert np.isclose(info["gamma_upper"], 0.5)
    assert info["q_upper"] == np.inf
    assert info["q_lower"] == 2.0

    rng = np.random.default_rng(2024)
    n = 20000
    upper_count = 0
    for _ in range(n):
        q = randomized_weighted_quantile(
            scores, weights, beta=beta, rng=rng, return_info=False
        )
        if q == np.inf:
            upper_count += 1
    freq = upper_count / n
    assert abs(freq - 0.5) < 0.03, f"empirical upper freq {freq}"


def test_tied_scores_aggregate_weights():
    scores = np.array([1.0, 1.0, 2.0])
    weights = np.array([0.2, 0.3, 0.5])
    beta = 0.55
    info = randomized_weighted_quantile(
        scores, weights, beta=beta, random_seed=1, return_info=True
    )
    assert info["q_upper"] == 2.0
    assert info["cdf_lower"] == 0.5
    assert info["cdf_upper"] == 1.0
    assert np.isclose(info["gamma_upper"], 0.1)


def test_equal_weights_all_finite():
    scores = np.linspace(0.0, 1.0, 8)
    alpha = 0.1
    det = weighted_quantile(scores, np.ones(len(scores)), alpha)
    q = conformal_threshold(
        scores, weights=None, alpha=alpha, quantile_mode="deterministic"
    )
    assert q == det


def main():
    test_beta_on_cdf_knot_matches_deterministic()
    test_inf_jump_gamma_half()
    test_tied_scores_aggregate_weights()
    test_equal_weights_all_finite()
    print("All randomized weighted quantile diagnostics passed.")


if __name__ == "__main__":
    main()
