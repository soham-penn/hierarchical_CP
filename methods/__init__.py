"""
Hierarchical Conformal Prediction Methods Package

This package implements various hierarchical conformal prediction methods
including HCP++, HCP.sample, and baseline methods.
"""

from .mu_methods import (
    create_mu_method_random_forest_offset,
    create_mu_method_random_forest_global_only,
    create_mu_method_ols_offset,
    create_mu_method_ols_global_only
)
from .baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius
)
from .hcp_plus import compute_hcp_plus_interval
from .hcp_sample import compute_hcp_sample_interval

__all__ = [
    'create_mu_method_random_forest_offset',
    'create_mu_method_random_forest_global_only',
    'create_mu_method_ols_offset',
    'create_mu_method_ols_global_only',
    'compute_hcp_interval_radius',
    'compute_pooling_interval_radius',
    'compute_subsampling_once_interval_radius',
    'compute_repeated_subsampling_interval_radius',
    'compute_hcp_plus_interval',
    'compute_hcp_sample_interval'
]
