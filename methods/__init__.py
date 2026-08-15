"""Hierarchical conformal prediction methods.

  - donor_hcp: GHCP (paper Algorithm 1; restricted donor pool, η=0.5)
  - baseline_hcp: HCP, pooling, subsampling, repeated subsampling
  - mu_methods: RF / OLS; paper (3) with c=1 for all paper tables
  - sample_hcp: S-HCP baseline (legacy name HCP.sample)

The paper never refers to GHCP as HCP++. That was an old working name in
``donor_hcp.py``.
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
