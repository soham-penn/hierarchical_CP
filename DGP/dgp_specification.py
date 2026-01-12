"""
Data Generating Process (DGP) Specification Module

This module defines DGP specification objects for hierarchical conformal prediction simulations.
All experiments use U ~ Unif(0,1)^d.
"""

import numpy as np


def create_dgp_specification_default(dimension, u_min=0, u_max=1, rho_X=0.7):
    """
    Create a default DGP specification.

    Parameters:
    -----------
    dimension : int
        Dimension of U and X vectors
    u_min : float
        Minimum value for U (default: 0)
    u_max : float
        Maximum value for U (default: 1)
    rho_X : float
        Correlation parameter for X covariance (default: 0.7)

    Returns:
    --------
    dict : DGP specification containing functions and parameters
    """

    def mean_X_given_U(u_vector):
        """Mean of X given U: u^2"""
        return u_vector ** 2

    def covariance_X(dimension_inner):
        """Covariance matrix for X with compound symmetry structure"""
        cov = (1 - rho_X) * np.eye(dimension_inner) + \
              rho_X * np.ones((dimension_inner, dimension_inner))
        return cov

    def regression_Y(x_vector, u_vector):
        """
        Regression function for Y given X and U:
        10 * sin(pi * x1 * x2) + 2 * u1^2 * x1^2
        """
        term1 = 10 * np.sin(np.pi * x_vector[0] * x_vector[1])
        term2 = 2 * u_vector[0]**2 * x_vector[0]**2
        return term1 + term2

    def noise_sd_Y(u_vector):
        """Standard deviation of noise for Y: 0.5 * (1 + 0.3 * u1)"""
        return 0.5 * (1 + 0.3 * u_vector[0])

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
    Create a nonlinear/heteroskedastic DGP specification to stress-test methods.

    Parameters:
    -----------
    dimension : int
        Dimension of U and X vectors
    u_min : float
        Minimum value for U (default: 0)
    u_max : float
        Maximum value for U (default: 1)
    rho_X : float
        Correlation parameter for X covariance (default: 0.5)

    Returns:
    --------
    dict : DGP specification containing functions and parameters
    """

    def mean_X_given_U(u_vector):
        """Mean of X given U: sin(2*pi*u) + u^2"""
        return np.sin(2 * np.pi * u_vector) + u_vector ** 2

    def covariance_X(dimension_inner):
        """
        Covariance matrix for X with compound symmetry and
        heterogeneous variances
        """
        base_matrix = (1 - rho_X) * np.eye(dimension_inner) + \
                     rho_X * np.ones((dimension_inner, dimension_inner))
        # Scale diagonal by (1 + i/5) for i=1,...,dimension_inner
        for i in range(dimension_inner):
            base_matrix[i, i] *= (1 + (i + 1) / 5)
        return base_matrix

    def regression_Y(x_vector, u_vector):
        """
        Complex nonlinear regression function:
        - 5 * sin(pi * x1 * u2)
        - 3 * x2^2 * sign(u3 - 0.5)
        - 2 * exp(-(x3 - u3)^2)
        - x4^3 * (u1 - 0.3)
        - sin(3*x5 + 2*u5)
        """
        term1 = 5 * np.sin(np.pi * x_vector[0] * u_vector[1])
        term2 = 3 * (x_vector[1]**2) * np.sign(u_vector[2] - 0.5)
        term3 = 2 * np.exp(-(x_vector[2] - u_vector[2])**2)
        term4 = (x_vector[3]**3) * (u_vector[0] - 0.3)
        term5 = np.sin(3 * x_vector[4] + 2 * u_vector[4])
        return term1 + term2 + term3 + term4 + term5

    def noise_sd_Y(u_vector):
        """
        Heteroskedastic noise standard deviation:
        0.2 + 0.8 * |u1 - 0.5| + 0.4 * I(u2 > 0.5)
        """
        sd = 0.2 + 0.8 * np.abs(u_vector[0] - (u_max + u_min) / 2)
        sd += 0.4 * (u_vector[1] > (u_min + u_max) / 2)
        return sd

    return {
        'dimension': dimension,
        'u_min': u_min,
        'u_max': u_max,
        'mean_X_given_U': mean_X_given_U,
        'covariance_X': covariance_X,
        'regression_Y': regression_Y,
        'noise_sd_Y': noise_sd_Y
    }
