"""
Extensions for Py2JAGS

This package contains specialized interfaces and utilities for specific models
and use cases with Py2JAGS.

Available extensions:
- ez_diffusion: EZ-diffusion model interface
"""

from .ez_diffusion import bayesian_parameter_estimation, Parameters

__all__ = ["bayesian_parameter_estimation", "Parameters"] 