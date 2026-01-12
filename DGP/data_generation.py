"""
Calibration and Test Data Generation

This module contains functions for generating calibration groups and test groups
according to the specified DGP.
"""

import numpy as np


def generate_calibration_data(number_groups, lambda_Poisson, dgp_specification):
    """
    Generate calibration data with hierarchical structure.

    Parameters:
    -----------
    number_groups : int
        Number of calibration groups (K)
    lambda_Poisson : float
        Parameter for Poisson distribution of group sizes
    dgp_specification : dict
        DGP specification object

    Returns:
    --------
    dict with keys:
        - 'U_calibration': ndarray of shape (K, d) - group-level covariates
        - 'Z_calibration': list of lists - observations for each group
        - 'sample_size_vector': array of group sizes
    """
    d = dgp_specification['dimension']

    # Generate group-level covariates U ~ Unif(u_min, u_max)^d
    U_cal = np.random.uniform(
        low=dgp_specification['u_min'],
        high=dgp_specification['u_max'],
        size=(number_groups, d)
    )

    # Generate group sizes: 1 + Poisson(lambda)
    N = 1 + np.random.poisson(lam=lambda_Poisson, size=number_groups)

    # Get covariance matrix for X
    Sigma_X = dgp_specification['covariance_X'](d)

    # Generate observations for each group
    Z_cal = []
    for j in range(number_groups):
        Uj = U_cal[j, :]
        Nj = N[j]

        # Mean of X given U
        mu_X = dgp_specification['mean_X_given_U'](Uj)

        # Generate X ~ N(mu_X, Sigma_X)
        X_mat = np.random.multivariate_normal(
            mean=mu_X,
            cov=Sigma_X,
            size=Nj
        )

        # Generate Y for each observation
        group_obs = []
        for i in range(Nj):
            x = X_mat[i, :]
            mu_Y = dgp_specification['regression_Y'](x, Uj)
            sd_Y = dgp_specification['noise_sd_Y'](Uj)
            y = np.random.normal(loc=mu_Y, scale=sd_Y)

            group_obs.append({'X': x, 'Y': y})

        Z_cal.append(group_obs)

    return {
        'U_calibration': U_cal,
        'Z_calibration': Z_cal,
        'sample_size_vector': N
    }


def generate_test_group(lambda_Poisson, dgp_specification, o_observed):
    """
    Generate a test group.

    Parameters:
    -----------
    lambda_Poisson : float
        Parameter for Poisson distribution of group size
    dgp_specification : dict
        DGP specification object
    o_observed : int
        Number of observed points (ensures N_test > o_observed)

    Returns:
    --------
    dict with keys:
        - 'U_test': ndarray of shape (1, d) - group-level covariate
        - 'Z_test': list - observations for test group
        - 'N_test': int - size of test group
    """
    d = dgp_specification['dimension']

    # Generate group-level covariate U ~ Unif(u_min, u_max)^d
    U_test = np.random.uniform(
        low=dgp_specification['u_min'],
        high=dgp_specification['u_max'],
        size=(1, d)
    )

    # Generate group size: 1 + Poisson(lambda), ensuring N_test > o_observed
    N_test = 1 + np.random.poisson(lam=lambda_Poisson)
    if N_test <= o_observed:
        N_test = o_observed + 1

    # Get covariance matrix for X
    Sigma_X = dgp_specification['covariance_X'](d)

    # Mean of X given U
    mu_X = dgp_specification['mean_X_given_U'](U_test[0, :])

    # Generate X ~ N(mu_X, Sigma_X)
    X_mat = np.random.multivariate_normal(
        mean=mu_X,
        cov=Sigma_X,
        size=N_test
    )

    # Generate Y for each observation
    Z_test = []
    for i in range(N_test):
        x = X_mat[i, :]
        mu_Y = dgp_specification['regression_Y'](x, U_test[0, :])
        sd_Y = dgp_specification['noise_sd_Y'](U_test[0, :])
        y = np.random.normal(loc=mu_Y, scale=sd_Y)

        Z_test.append({'X': x, 'Y': y})

    return {
        'U_test': U_test,
        'Z_test': Z_test,
        'N_test': N_test
    }
