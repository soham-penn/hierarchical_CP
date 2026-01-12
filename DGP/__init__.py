"""
Data Generating Process (DGP) Package

This package contains DGP specifications and data generation functions
for hierarchical conformal prediction experiments.
"""

from .dgp_specification import (
    create_dgp_specification_default,
    create_dgp_specification_nonlinear
)
from .data_generation import (
    generate_calibration_data,
    generate_test_group
)

__all__ = [
    'create_dgp_specification_default',
    'create_dgp_specification_nonlinear',
    'generate_calibration_data',
    'generate_test_group'
]
