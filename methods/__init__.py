"""Hierarchical conformal prediction methods.

Modules:
  - mu_methods: RF / OLS global predictors (optional within-group offset)
  - baseline_hcp: HCP, pooling, subsampling, repeated subsampling
  - donor_hcp: donor-HCP (GHCP) randomized and derandomized
  - sample_hcp: sample-HCP randomized and derandomized
"""

from .mu_methods import (
    create_mu_method_random_forest_offset,
    create_mu_method_random_forest_global_only,
    create_mu_method_ols_offset,
    create_mu_method_ols_global_only,
)
from .baseline_hcp import (
    compute_hcp_interval_radius,
    compute_pooling_interval_radius,
    compute_subsampling_once_interval_radius,
    compute_repeated_subsampling_interval_radius,
)
from .donor_hcp import (
    compute_donor_hcp_randomized_interval,
    compute_donor_hcp_derandomized_interval,
)
from .sample_hcp import (
    compute_sample_hcp_randomized_interval,
    compute_sample_hcp_derandomized_interval,
)

__all__ = [
    # mu-methods
    'create_mu_method_random_forest_offset',
    'create_mu_method_random_forest_global_only',
    'create_mu_method_ols_offset',
    'create_mu_method_ols_global_only',
    # baseline methods
    'compute_hcp_interval_radius',
    'compute_pooling_interval_radius',
    'compute_subsampling_once_interval_radius',
    'compute_repeated_subsampling_interval_radius',
    # Donor-HCP and Sample-HCP (renamed API)
    'compute_donor_hcp_randomized_interval',
    'compute_donor_hcp_derandomized_interval',
    'compute_sample_hcp_randomized_interval',
    'compute_sample_hcp_derandomized_interval',
]
