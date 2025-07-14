"""
EZ-Diffusion interface for Py2JAGS

This module provides a wrapper for using Py2JAGS with EZ-diffusion models,
using the proper EZ-DDM formulas and summary statistics approach.
"""

import os
import tempfile
import shutil
import numpy as np
from typing import List

from ..core import run_jags
from ..mcmc_samples import MCMCSamples


class Parameters:
    """Simple parameter storage class for compatibility"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def bayesian_parameter_estimation(rt_data: List[float], 
                                accuracy_data: List[int],
                                n_samples: int = 5000,
                                n_burnin: int = 2000,
                                n_chains: int = 4,
                                verbosity: int = 2,
                                debug: bool = False) -> MCMCSamples:
    """
    Estimate EZ-diffusion parameters using JAGS via Py2JAGS
    
    Uses summary statistics approach with proper EZ-DDM formulas
    following the visual-ez.R implementation.
    
    Args:
        rt_data: Response time data
        accuracy_data: Accuracy data (0/1)
        n_samples: Number of MCMC samples
        n_burnin: Number of burn-in samples
        n_chains: Number of MCMC chains
        verbosity: Verbosity level
        debug: Debug mode - prints working directory and skips cleanup
        return_samples_object: Whether to return an MCMCSamples object (default: True)
    
    Returns:
        If return_samples_object=True: MCMCSamples object with full MCMC analysis capabilities
        If return_samples_object=False: Parameters object with point estimates (backward compatibility)
    """
    
    # Create temporary directory for JAGS files
    temp_dir = tempfile.mkdtemp(prefix='py2jags_ez_')
    
    try:
        # Create JAGS model
        model_string = _create_ez_diffusion_model()
        
        # Prepare summary statistics data (single cell)
        rt_array = np.array(rt_data)
        acc_array = np.array(accuracy_data)
        
        n_trials = len(rt_data)
        n_correct = np.sum(acc_array)
        mean_rt = np.mean(rt_array)
        var_rt = np.var(rt_array, ddof=1)  # Sample variance
        
        data_dict = {
            'correct': [n_correct],  # Total correct responses
            'meanRT': [mean_rt],     # Mean response time
            'varRT': [var_rt],       # Variance of response time
            'nTrials': [n_trials],   # Number of trials
            'nCell': 1               # Single cell
        }
        
        # Create initialization files
        init_files = []
        for chain in range(n_chains):
            init_file = os.path.join(temp_dir, f'init_{chain+1}.R')
            with open(init_file, 'w') as f:
                # Initialize with reasonable starting values
                f.write(f'boundary <- {0.8 + 0.2 * chain}\n')
                f.write(f'drift <- {0.3 + 0.1 * chain}\n')
                f.write(f'ndt <- {0.25 + 0.05 * chain}\n')
            init_files.append(init_file)
        
        # Run JAGS using Py2JAGS
        samples = run_jags(
            model_string=model_string,
            data_dict=data_dict,
            init=init_files,
            monitorparams=['boundary', 'drift', 'ndt'],
            nchains=n_chains,
            nadapt=1000,
            nburnin=n_burnin,
            nsamples=n_samples,
            thin=1,
            workingdir=temp_dir,
            parallel=True,
            verbosity=verbosity,
            debug=debug,
        )
        
        return samples
        
    finally:
        # Clean up temporary directory unless debug mode is enabled
        if not debug:
            shutil.rmtree(temp_dir)


def _create_ez_diffusion_model() -> str:
    """Create EZ-diffusion JAGS model using proper EZ-DDM formulas"""
    
    return """
    model {
        # Priors for DDM parameters
        boundary ~ dnorm(1.50, pow(0.20, -2))T(0.10, 3.00)
        drift ~ dnorm(0.50, pow(0.50, -2))
        ndt ~ dnorm(0.30, pow(0.03, -2))T(0.05, )
        
        # EZ-DDM forward equations (from visual-ez.R)
        ey <- exp(-boundary * drift)
        Pc <- 1 / (1 + ey)
        PRT <- 2 * pow(drift, 3) / boundary * 
               pow(ey + 1, 2) / (2 * (-boundary) * 
                   drift * ey - ey * ey + 1)
        MDT <- (boundary / (2 * drift)) * 
               (1 - ey) / (1 + ey)
        MRT <- MDT + ndt
        
        # Sampling distributions for summary statistics
        for (c in 1:nCell) {
            correct[c] ~ dbin(Pc, nTrials[c])
            varRT[c] ~ dnorm(1/PRT, 
                           0.5 * (nTrials[c] - 1) * PRT * PRT)
            meanRT[c] ~ dnorm(MRT, PRT * nTrials[c])
        }
    }
    """ 