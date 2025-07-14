"""
Main JAGS interface for Py2JAGS

This module provides the main run_jags function that replicates the Trinity
MATLAB toolbox interface for running JAGS.
"""

import os
import platform
from .runner import JagsRunner
from .reader import CodaReader
from .utils import parse_options, summary_stats
from .mcmc_samples import MCMCSamples


def run_jags(**kwargs) -> MCMCSamples:
    """
    Execute a call to JAGS
    
    This function will execute a call to JAGS. Supply a set of options
    through keyword arguments. Based on Trinity's calljags function.
    
    Parameters
    ----------
    model_file : str, optional
        Path to JAGS model file (mutually exclusive with model_string)
    model_string : str, optional
        JAGS model code as string (mutually exclusive with model_file)
    data_file : str, optional
        Path to data file (mutually exclusive with data_dict)
    data_dict : dict, optional
        Data as dictionary (mutually exclusive with data_file)
    outputname : str, optional
        Name for output files (default: 'samples')
    init : list, optional
        List of initial value files for each chain
    nchains : int, optional
        Number of chains (default: 4)
    nburnin : int, optional
        Number of burn-in iterations (default: 1000)
    nsamples : int, optional
        Number of samples (default: 5000)
    monitorparams : list, optional
        List of parameters to monitor
    thin : int, optional
        Thinning interval (default: 1)
    modules : list, optional
        JAGS modules to load
    seed : int, optional
        Random seed
    workingdir : str, optional
        Working directory (default: timestamped temp directory in /tmp/)
    verbosity : int, optional
        Verbosity level (default: 0)
    saveoutput : bool, optional
        Save JAGS output (default: True)
    parallel : bool, optional
        Run chains in parallel (default: True on Unix)
    maxcores : int, optional
        Maximum cores for parallel execution (default: 0 = auto)
    showwarnings : bool, optional
        Show JAGS warnings (default: True)
    debug : bool, optional
        Debug mode - prints working directory and skips cleanup (default: False)
    jags_executable : str, optional
        Name or path of JAGS executable (default: 'jags')
        
    Returns
    -------
        MCMCSamples
            MCMCSamples object with built-in analysis methods
    
    Examples
    --------
    >>> samples = run_jags(model_string=model_string, data_dict=data_dict)
    >>> print(samples.summary())
    """
    
    # Parse and validate options
    options = parse_options('jags', **kwargs)
    
    # Check system compatibility
    if platform.system() != 'Linux':
        raise RuntimeError("Py2JAGS currently only supports Linux")
    
    # Run JAGS
    runner = JagsRunner(options)
    options = runner.run()
    
    # Process CODA output
    reader = CodaReader(options)
    coda = reader.read()
    
    # Generate summary statistics
    output = summary_stats(coda)
    
    # Prepare return values
    stats = output['stats']
    chains = output['chains'] 
    diagnostics = output['diagnostics']
    info = output['info']
    info['options'] = options
    
    # Add debug info to return values
    if options.get('debug', False):
        info['debug'] = {
            'workingdir': options['workingdir'],
            'cleanup_skipped': True
        }
    
    # Print debug info if requested
    if options.get('debug', False):
        print(f"\n=== DEBUG MODE ===")
        wdir = options['workingdir']
        print(f"Working directory: {wdir}")
        print(f"Temporary files were NOT cleaned up for debugging purposes")
        print("Directory contents:")
        os.system(f"ls -1d {wdir}/*")
        print(f"==================\n")
    
    return MCMCSamples(stats, chains, diagnostics, info)

