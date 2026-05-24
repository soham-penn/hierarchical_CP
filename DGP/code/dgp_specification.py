"""
Data Generating Process (DGP) Specification Module

This module defines DGP specification objects for hierarchical conformal prediction simulations.
All experiments use U ~ Unif(0,1)^d.
"""

import numpy as np


def _coord(vec, idx, default=0.0):
    """Safe coordinate access for variable-dimension DGPs."""
    return vec[idx] if idx < len(vec) else default


def create_dgp_specification_default(dimension, u_min=0, u_max=1, rho_X=0.5):
    """
    Create a default (OLS-friendly) DGP specification.

    Regression is linear in X with a U-interaction term so that OLS can
    fit it well while the hierarchical structure still matters.

    The covariance Σ(U) of X | U is *U-dependent*:

        Σ(U) = s(u₁) · [(1 − ρ)I + ρ 11ᵀ]
        s(u₁) = 0.2 + 3 sin²(2π u₁)

    This creates peaks in X-spread at u₁ = 0.25, 0.75 and valleys at
    u₁ = 0, 0.5, 1.  Combined with an oscillating noise SD, the residual
    |Y − μ̂(X)| varies non-monotonically with U, which causes HCP's
    pooled quantile to over-cover at some U and under-cover at others.

    Parameters
    ----------
    dimension : int
        Dimension of U and X vectors
    u_min, u_max : float
        Range for U (default: 0, 1)
    rho_X : float
        Compound-symmetry base correlation (default: 0.5)
    """

    def mean_X_given_U(u_vector):
        """Mean of X given U: u²."""
        return u_vector ** 2

    def covariance_X(dimension_inner, u_vector):
        """U-dependent covariance: scale(u₁) × compound-symmetry.

        scale(u₁) = 0.2 + 3 sin²(2π u₁)
          - Peaks (≈ 3.2) at u₁ = 0.25, 0.75
          - Valleys (≈ 0.2) at u₁ = 0, 0.5, 1
        """
        scale = 0.2 + 3.0 * np.sin(2 * np.pi * u_vector[0]) ** 2
        base = (1 - rho_X) * np.eye(dimension_inner) + \
               rho_X * np.ones((dimension_inner, dimension_inner))
        return scale * base

    def regression_Y(x_vector, u_vector):
        """
        OLS-friendly regression:
          2*x₁ + x₂ + 0.5*x₃ + 3*u₁*x₁ + u₂
        Linear in X, with U-X interaction so group structure matters.
        """
        x1 = _coord(x_vector, 0)
        x2 = _coord(x_vector, 1)
        x3 = _coord(x_vector, 2)
        u1 = _coord(u_vector, 0)
        u2 = _coord(u_vector, 1)
        return (2 * x1 + x2 + 0.5 * x3 + 3 * u1 * x1 + u2)

    def noise_sd_Y(x_vector, u_vector):
        """Noise SD: 0.5 + 2 sin²(2π u₁) — oscillates in phase with Σ(U).

        Peaks (≈ 2.5) at u₁ = 0.25, 0.75; valleys (0.5) at u₁ = 0, 0.5, 1.
        Combined with the U-dependent covariance, this creates a ~5:1 ratio
        in residual SD across the U-grid, driving oscillating conditional
        coverage for methods that pool across all U values (HCP, Pooling).
        """
        return 0.5 + 2.0 * np.sin(2 * np.pi * u_vector[0]) ** 2

    return {
        'dimension': dimension,
        'u_min': u_min,
        'u_max': u_max,
        'mean_X_given_U': mean_X_given_U,
        'covariance_X': covariance_X,
        'regression_Y': regression_Y,
        'noise_sd_Y': noise_sd_Y
    }


def create_dgp_specification_nonlinear(dimension, u_min=0, u_max=1, rho_X=0.5):
    """
    Nonlinear DGP that is hard for pooling / OLS.

    The regression has strong nonlinearity and group-level heterogeneity:
      - sin(π x₁ x₂): OLS cannot capture
      - exp(u₁) * x₁²: group-level scaling of a nonlinear feature
      - |x₃ − u₃|: V-shaped, group-shifted

    Uses the same U-dependent covariance as the default DGP.

    Parameters
    ----------
    dimension : int
    u_min, u_max : float
    rho_X : float
    """

    def mean_X_given_U(u_vector):
        """Mean of X given U: sin(2πu) + u²"""
        return np.sin(2 * np.pi * u_vector) + u_vector ** 2

    def covariance_X(dimension_inner, u_vector):
        """U-dependent covariance: scale(u₁) × compound-symmetry."""
        scale = 0.2 + 3.0 * np.sin(2 * np.pi * u_vector[0]) ** 2
        base = (1 - rho_X) * np.eye(dimension_inner) + \
               rho_X * np.ones((dimension_inner, dimension_inner))
        return scale * base

    def regression_Y(x_vector, u_vector):
        """
        Strongly nonlinear regression:
          4 * sin(π * x₁ * x₂)       — nonlinear interaction, OLS can't fit
        + exp(u₁) * x₁²              — group-scaled quadratic
        + 2 * |x₃ − u₃|             — V-shaped, group-shifted
        """
        x1 = _coord(x_vector, 0)
        x2 = _coord(x_vector, 1)
        x3 = _coord(x_vector, 2)
        u1 = _coord(u_vector, 0)
        u3 = _coord(u_vector, 2)
        term1 = 4 * np.sin(np.pi * x1 * x2)
        term2 = np.exp(u1) * x1 ** 2
        term3 = 2 * np.abs(x3 - u3)
        return term1 + term2 + term3

    def noise_sd_Y(x_vector, u_vector):
        """Oscillating noise: 0.5 + 1.2 sin²(2π u₁)."""
        return 0.5 + 1.2 * np.sin(2 * np.pi * u_vector[0]) ** 2

    return {
        'dimension': dimension,
        'u_min': u_min,
        'u_max': u_max,
        'mean_X_given_U': mean_X_given_U,
        'covariance_X': covariance_X,
        'regression_Y': regression_Y,
        'noise_sd_Y': noise_sd_Y
    }


def create_dgp_specification_heteroscedastic(dimension, u_min=0, u_max=1,
                                              rho_X=0.5, noise_base=0.3,
                                              noise_x_scale=1.5):
    """
    DGP with X-dependent heteroscedasticity — designed to break conditional
    validity of marginal conformal methods.

    Uses the same U-dependent covariance as the default DGP.  The main
    driver of conditional invalidity is the noise SD that grows
    quadratically in x₁:

        noise_sd(x, u) = noise_base + noise_x_scale * x₁²

    Parameters
    ----------
    dimension : int
    u_min, u_max : float
    rho_X : float
        Compound-symmetry base correlation (default 0.5).
    noise_base : float
        Baseline noise SD (default 0.3).
    noise_x_scale : float
        Coefficient on x₁² in the noise SD (default 1.5).
    """

    def mean_X_given_U(u_vector):
        """Mean of X given U: u²."""
        return u_vector ** 2

    def covariance_X(dimension_inner, u_vector):
        """U-dependent covariance: scale(u₁) × compound-symmetry."""
        scale = 0.2 + 3.0 * np.sin(2 * np.pi * u_vector[0]) ** 2
        base = (1 - rho_X) * np.eye(dimension_inner) + \
               rho_X * np.ones((dimension_inner, dimension_inner))
        return scale * base

    def regression_Y(x_vector, u_vector):
        """
        Mildly nonlinear regression:
          3*x₁ + 2*sin(π*x₂) + x₁*u₁
        """
        return 3 * x_vector[0] + 2 * np.sin(np.pi * x_vector[1]) + \
               x_vector[0] * u_vector[0]

    def noise_sd_Y(x_vector, u_vector):
        """
        X-dependent heteroscedastic noise:
          noise_base + noise_x_scale * x₁²
        """
        return noise_base + noise_x_scale * x_vector[0] ** 2

    return {
        'dimension': dimension,
        'u_min': u_min,
        'u_max': u_max,
        'mean_X_given_U': mean_X_given_U,
        'covariance_X': covariance_X,
        'regression_Y': regression_Y,
        'noise_sd_Y': noise_sd_Y
    }
