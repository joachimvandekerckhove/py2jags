"""
Py2JAGS - A Python interface to JAGS (Just Another Gibbs Sampler)

This package provides a Python interface to JAGS that mimics the Trinity MATLAB toolbox.
It works by creating JAGS script files and calling JAGS via system commands.

Main Components:
- core: Main interface to run JAGS
- runner: JAGS execution engine
- reader: CODA file parsing
- utils: Utility functions and option parsing
- mcmc_samples: MCMC analysis and visualization

Examples:
    >>> from py2jags import run_jags
    >>> samples = run_jags(
    ...     model_string=model_string,
    ...     data_dict=data_dict,
    ...     monitorparams=['theta', 'sigma'],
    ...     nchains=4,
    ...     nburnin=1000,
    ...     nsamples=5000
    ... )
    >>> print(samples.summary())
    >>> fig = samples.traceplot()
    >>> print(samples.diagnostics())
    >>> print(samples.info())
"""

from .core import run_jags
from .runner import JagsRunner
from .reader import CodaReader
from .utils import parse_options, summary_stats, write_s_data
from .mcmc_samples import MCMCSamples

__version__ = "0.1.0"
__author__ = "Joachim Vandekerckhove"
__email__ = ""
__license__ = "GPL-3.0"

__all__ = [
    "run_jags",
    "JagsRunner", 
    "CodaReader",
    "parse_options",
    "summary_stats",
    "write_s_data",
    "MCMCSamples"
] 