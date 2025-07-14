"""
Utility functions for Py2JAGS

This module provides utility functions for option parsing, file handling,
and summary statistics generation.
"""

import os
import random
import tempfile
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional


def parse_options(engine: str, **kwargs) -> Dict[str, Any]:
    """
    Parse input options for JAGS execution
    
    Parameters
    ----------
    engine : str
        Engine type ('jags')
    **kwargs : dict
        Input options
        
    Returns
    -------
    dict
        Parsed and validated options
    """
    
    # Default options for all engines
    defaults_all = {
        'model_file': None,
        'model_string': None,
        'data_file': None,
        'data_dict': None,
        'outputname': 'samples',
        'init': None,
        'modelfilename': None,
        'datafilename': None,
        'initfilename': None,
        'scriptfilename': None,
        'logfilename': None,
        'engine': 'jags',
        'nchains': 4,
        'nburnin': 1000,
        'nsamples': 5000,
        'monitorparams': None,
        'thin': 1,
        'refresh': 1000,
        'workingdir': None,
        'verbosity': 0,
        'saveoutput': True,
        'readonly': False,
        'parallel': os.name == 'posix',  # True on Unix-like systems
        'allowunderscores': False,
        'debug': False
    }
    
    # JAGS-specific defaults
    defaults_jags = {
        'engine': 'jags',
        'showwarnings': True,
        'modules': [],
        'seed': random.randint(1, 10000),
        'maxcores': 0,  # 0 means auto-detect
        'jags_executable': 'jags'  # Name/path of JAGS executable
    }
    
    # Merge defaults
    if engine.lower() == 'jags':
        defaults = {**defaults_all, **defaults_jags}
    else:
        raise ValueError(f"Unknown engine: {engine}")
    
    # Update with user options
    options = {**defaults, **kwargs}
    options['engine'] = engine.lower()
    
    # Validate and process options
    options = _validate_options(options)
    
    return options


def _validate_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and process options"""
    
    # Validate model specification - must have either model or model_string, but not both
    has_model_file = options.get('model_file') is not None
    has_model_string = options.get('model_string') is not None
    
    if has_model_file and has_model_string:
        raise ValueError("Cannot specify both 'model_file' and 'model_string'. Use one or the other.")
    
    if not has_model_file and not has_model_string:
        raise ValueError("Must specify either 'model_file' (file path) or 'model_string' (model code).")
    
    # Validate data specification - must have either data or data_dict, but not both
    has_data_file = options.get('data_file') is not None
    has_data_dict = options.get('data_dict') is not None
    
    if has_data_file and has_data_dict:
        raise ValueError("Cannot specify both 'data_file' and 'data_dict'. Use one or the other.")
    
    if not has_data_file and not has_data_dict:
        raise ValueError("Must specify either 'data_file' (file path) or 'data_dict' (data dictionary).")
    
    # Handle working directory - create timestamped temp dir if none provided
    if options['workingdir'] is None:
        # Create timestamped temporary directory in /tmp/
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_dir = tempfile.mkdtemp(prefix=f'py2jags_{timestamp}_')
        options['workingdir'] = temp_dir
        
        if options.get('verbosity', 0) > 0:
            print(f"Created working directory: {temp_dir}")
    else:
        # Ensure user-specified working directory exists
        if not os.path.exists(options['workingdir']):
            os.makedirs(options['workingdir'])
        # Convert to absolute path to avoid issues with os.chdir()
        options['workingdir'] = os.path.abspath(options['workingdir'])
    
    # Handle model_string - create temporary file if provided
    if has_model_string:
        model_file = os.path.join(options['workingdir'], 'model.jags')
        
        # Write model string to temporary file
        with open(model_file, 'w') as f:
            f.write(options['model_string'])
        
        # Update options to use the created file
        options['model_file'] = model_file
        
        if options.get('verbosity', 0) > 0:
            print(f"Created model file from string: {model_file}")
    
    # Handle data_dict - create temporary file if provided
    if has_data_dict:
        data_file = os.path.join(options['workingdir'], 'data.S')
        
        # Write data dictionary to temporary file using write_s_data
        write_s_data(data_file, **options['data_dict'])
        
        # Update options to use the created file
        options['data_file'] = data_file
        
        if options.get('verbosity', 0) > 0:
            print(f"Created data file from dictionary: {data_file}")
    
    # Convert relative paths to absolute paths
    for key in ['model_file', 'data_file']:
        if options[key] and not os.path.isabs(options[key]):
            options[key] = os.path.abspath(options[key])
    
    # Ensure monitorparams is a list
    if options['monitorparams'] is not None:
        if isinstance(options['monitorparams'], str):
            options['monitorparams'] = [options['monitorparams']]
    
    # Ensure modules is a list
    if isinstance(options['modules'], str):
        options['modules'] = [options['modules']]
    
    # Auto-detect maxcores if 0
    if options['maxcores'] == 0:
        options['maxcores'] = os.cpu_count() or 1
    
    return options


def generate_temp_filename(
    options: Dict[str, Any], 
    prefix: str, 
    chain: Optional[int] = None
) -> str:
    """Generate a temporary filename"""
    
    if chain is not None:
        filename = f"{options['outputname']}_{prefix}_{chain}"
    else:
        filename = f"{options['outputname']}_{prefix}"
    
    return os.path.join(options['workingdir'], filename)


def write_s_data(filename: str, **data) -> None:
    """Write data to a JAGS data file"""
    
    with open(filename, 'w') as f:
        for key, value in data.items():
            if isinstance(value, (int, float)):
                f.write(f'{key} <- {value}\n')
            elif isinstance(value, str):
                # Handle quoted strings properly
                if value.startswith('"') and value.endswith('"'):
                    f.write(f'{key} <- {value}\n')  # Already quoted
                else:
                    f.write(f'{key} <- "{value}"\n')  # Add quotes
            elif isinstance(value, (list, np.ndarray)):
                if np.array(value).ndim == 1:
                    # Vector
                    values_str = ', '.join(map(str, value))
                    f.write(f'{key} <- c({values_str})\n')
                else:
                    # Matrix - flatten in column-major order for JAGS/R compatibility
                    arr = np.array(value)
                    f.write(f'{key} <- structure(\n')
                    f.write(f'  c({", ".join(map(str, arr.flatten(order="F")))}),\n')
                    f.write(f'  .Dim = c{arr.shape}\n')
                    f.write(f')\n')


def summary_stats(coda: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from CODA samples
    
    Parameters
    ----------
    coda : dict
        CODA samples dictionary
        
    Returns
    -------
    dict
        Dictionary with stats, chains, diagnostics, and info
    """
    
    samples = coda['samples']
    
    # Initialize output dictionaries
    stats = {}
    chains = {}
    diagnostics = {}
    
    for param_name, param_samples in samples.items():
        # param_samples should be a 2D array: [iterations, chains]
        if param_samples.ndim == 1:
            param_samples = param_samples.reshape(-1, 1)
        
        # Convert to list of chains format: [chain1, chain2, ...]
        # where each chain is a 1D array of samples
        chains[param_name] = [param_samples[:, i] for i in range(param_samples.shape[1])]
        
        # Calculate summary statistics
        param_stats = {
            'mean': np.mean(param_samples),
            'std': np.std(param_samples, ddof=1),
            'median': np.median(param_samples),
            'q025': np.percentile(param_samples, 2.5),
            'q975': np.percentile(param_samples, 97.5),
            'min': np.min(param_samples),
            'max': np.max(param_samples)
        }
        
        # Calculate convergence diagnostics (simple Rhat)
        if param_samples.shape[1] > 1:  # Multiple chains
            rhat = _calculate_rhat(param_samples)
            n_eff = _calculate_eff_sample_size(param_samples)
            mcmc_se = _calculate_mcmc_error(param_samples)
        else:
            rhat = np.nan
            n_eff = param_samples.shape[0]  # Single chain, n_eff = n_samples
            mcmc_se = np.std(param_samples, ddof=1) / np.sqrt(param_samples.shape[0])
        
        param_stats['rhat'] = rhat
        param_stats['n_eff'] = n_eff
        param_stats['mcmc_se'] = mcmc_se
        
        stats[param_name] = param_stats
        diagnostics[param_name] = {
            'rhat': rhat,
            'n_eff': n_eff,
            'mcmc_se': mcmc_se
        }
    
    info = {
        'n_parameters': len(samples),
        'n_samples': param_samples.shape[0] if samples else 0,
        'n_chains': param_samples.shape[1] if samples else 0
    }
    
    return {
        'stats': stats,
        'chains': chains,
        'diagnostics': diagnostics,
        'info': info
    }


def _calculate_rhat(samples: np.ndarray) -> float:
    """
    Calculate R-hat convergence diagnostic
    
    Parameters
    ----------
    samples : ndarray
        Array of shape [iterations, chains]
        
    Returns
    -------
    float
        R-hat statistic
    """
    
    n_iterations, n_chains = samples.shape
    
    if n_chains < 2:
        return 1.0
    
    # Chain means
    chain_means = np.mean(samples, axis=0)
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n_iterations * np.var(chain_means, ddof=1)
    
    # Within-chain variance
    W = np.mean([np.var(chain, ddof=1) for chain in samples.T])
    
    # Pooled variance estimate
    var_plus = ((n_iterations - 1) / n_iterations) * W + (1 / n_iterations) * B
    
    # R-hat
    rhat = np.sqrt(var_plus / W) if W > 0 else 1.0
    
    return float(rhat)


def _calculate_eff_sample_size(samples: np.ndarray) -> float:
    """
    Calculate effective sample size
    
    Parameters
    ----------
    samples : ndarray
        Array of shape [iterations, chains]
        
    Returns
    -------
    float
        Effective sample size
    """
    
    n_iterations, n_chains = samples.shape
    
    if n_chains < 2:
        return float(n_iterations)
    
    # Simple approximation: total samples adjusted for autocorrelation
    # For multiple chains, estimate autocorrelation using lag-1 correlation
    all_samples = samples.flatten()
    
    # Calculate lag-1 autocorrelation 
    if len(all_samples) > 1:
        autocorr = np.corrcoef(all_samples[:-1], all_samples[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0
    
    # Effective sample size accounting for autocorrelation
    n_eff = len(all_samples) * (1 - autocorr) / (1 + autocorr) if autocorr < 1.0 else len(all_samples)
    
    nominal_samples = n_chains * n_iterations
    n_eff = np.min([n_eff, nominal_samples])
    
    return float(max(1.0, n_eff))


def _calculate_mcmc_error(samples: np.ndarray) -> float:
    """
    Calculate MCMC error (Monte Carlo standard error)
    
    Parameters
    ----------
    samples : ndarray
        Array of shape [iterations, chains]
        
    Returns
    -------
    float
        MCMC error
    """
    
    # Standard error of the mean
    all_samples = samples.flatten()
    n_eff = _calculate_eff_sample_size(samples)
    
    mcmc_error = np.std(all_samples, ddof=1) / np.sqrt(n_eff)
    
    return float(mcmc_error)


def set_executable_permissions(filepath: str) -> None:
    """Set executable permissions on a file"""
    import stat
    current_permissions = os.stat(filepath).st_mode
    os.chmod(filepath, current_permissions | stat.S_IEXEC) 