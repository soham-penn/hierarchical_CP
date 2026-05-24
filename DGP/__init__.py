"""
Data Generating Process (DGP) Package

This package contains DGP specifications and data generation functions
for hierarchical conformal prediction experiments.
"""

from .code.dgp_specification import (
    create_dgp_specification_default,
    create_dgp_specification_nonlinear,
    create_dgp_specification_heteroscedastic
)
from .code.data_generation import (
    generate_calibration_data,
    generate_test_group
)

__all__ = [
    'create_dgp_specification_default',
    'create_dgp_specification_nonlinear',
    'create_dgp_specification_heteroscedastic',
    'generate_calibration_data',
    'generate_test_group'
]
