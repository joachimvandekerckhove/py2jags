"""
MCMC Samples class for storing and analyzing JAGS output

This module provides a comprehensive class for handling MCMC samples
with built-in methods for summaries, plotting, and diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import re
from collections import defaultdict
import warnings


class MCMCSamples:
    """
    A comprehensive class for storing and analyzing MCMC samples
    
    This class provides methods for summary statistics, plotting,
    diagnostics, and data manipulation of MCMC output.
    """
    
    def __init__(self, stats: Dict, chains: Dict, diagnostics: Dict, info: Dict):
        """
        Initialize MCMCSamples from run_jags output
        
        Parameters
        ----------
        stats : dict
            Summary statistics from run_jags
        chains : dict
            Raw MCMC chains from run_jags
        diagnostics : dict
            Convergence diagnostics from run_jags
        info : dict
            Additional information from run_jags
        """
        self.stats = stats
        self.chains = chains
        self.diagnostics = diagnostics
        self.info = info
        
        # Extract basic information
        self.parameter_names = list(chains.keys()) if chains else []
        self.n_chains = len(chains[self.parameter_names[0]]) if self.parameter_names else 0
        self.n_samples = len(chains[self.parameter_names[0]][0]) if self.parameter_names and self.n_chains > 0 else 0
        self.n_parameters = len(self.parameter_names)
        
        # Store options for reference
        self.options = info.get('options', {})
        
    @classmethod
    def from_run_jags(cls, stats: Dict, chains: Dict, diagnostics: Dict, info: Dict):
        """
        Create MCMCSamples from run_jags output
        
        Parameters
        ----------
        stats, chains, diagnostics, info : dict
            Output from run_jags function
            
        Returns
        -------
        MCMCSamples
            Initialized MCMCSamples object
        """
        return cls(stats, chains, diagnostics, info)
    
    def summary(self, 
                parameter_regex: Optional[str] = None, 
                summary_fcn: List[Callable] = [np.mean, np.std, np.median]) -> pd.DataFrame:
        """
        Generate summary statistics table
        
        Parameters
        ----------
        parameters : list of str, optional
            Parameter names to include. If None, includes all parameters.
        summary_fcn : list of Callable
            Functions to compute summary statistics. Default: [np.mean, np.std, np.median]
            
        Returns
        -------
        pd.DataFrame
            Summary statistics table
        """
        if parameter_regex is None:
            parameters = self.parameter_names
        else:
            parameters = self.filter_parameters(parameter_regex)
        
        summary_data = []
        
        for param in parameters:
            if param not in self.chains:
                continue
                
            # Combine all chains for this parameter
            all_samples = np.concatenate(self.chains[param])
            
            # Calculate statistics
            row = {
                'parameter': param,
                **{fcn.__name__: fcn(all_samples) for fcn in summary_fcn}
            }
            
            summary_data.append(row)
        
        # Handle empty summary_data
        if not summary_data:
            # Create empty DataFrame with proper columns
            columns = ['parameter'] + [fcn.__name__ for fcn in summary_fcn]
            df = pd.DataFrame(columns=columns)
            df.set_index('parameter', inplace=True)
            print(df)
            return
        
        df = pd.DataFrame(summary_data)
        df.set_index('parameter', inplace=True)
        print(df)
        return
    
    def filter_parameters(self, pattern: str) -> List[str]:
        """
        Filter parameter names by regex pattern
        
        Parameters
        ----------
        pattern : str
            Regular expression pattern to match parameter names
            
        Returns
        -------
        list of str
            Matching parameter names
        """
        regex = re.compile(pattern)
        return [param for param in self.parameter_names if regex.search(param)]
    
    def get_samples(self, parameter: str, chain: Optional[int] = None) -> np.ndarray:
        """
        Get samples for a specific parameter
        
        Parameters
        ----------
        parameter : str
            Parameter name
        chain : int, optional
            Chain number (0-based). If None, returns all chains concatenated.
            
        Returns
        -------
        np.ndarray
            MCMC samples
        """
        if parameter not in self.chains:
            raise ValueError(f"Parameter '{parameter}' not found")
        
        if chain is None:
            return np.concatenate(self.chains[parameter])
        else:
            if chain >= self.n_chains:
                raise ValueError(f"Chain {chain} not found (only {self.n_chains} chains)")
            return np.array(self.chains[parameter][chain])
    
    def trace_plot(self, parameters: Optional[List[str]] = None, 
                  chains: Optional[List[int]] = None,
                  figsize: Optional[Tuple[float, float]] = None,
                  ncols: int = 2) -> plt.Figure:
        """
        Create trace plots for parameters
        
        Parameters
        ----------
        parameters : list of str, optional
            Parameters to plot. If None, plots all parameters.
        chains : list of int, optional
            Chain numbers to plot. If None, plots all chains.
        figsize : tuple, optional
            Figure size (width, height)
        ncols : int
            Number of columns in subplot grid
            
        Returns
        -------
        matplotlib.Figure
            Figure object
        """
        if parameters is None:
            parameters = self.parameter_names[:12]  # Limit to first 12 for readability
        
        if chains is None:
            chains = list(range(self.n_chains))
        
        n_params = len(parameters)
        nrows = (n_params + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (ncols * 4, nrows * 3)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(parameters):
            row, col = i // ncols, i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            
            for chain_idx in chains:
                samples = self.get_samples(param, chain_idx)
                ax.plot(samples, alpha=0.7, label=f'Chain {chain_idx + 1}')
            
            ax.set_title(f'{param}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            if len(chains) > 1:
                ax.legend()
        
        # Hide empty subplots
        for i in range(n_params, nrows * ncols):
            row, col = i // ncols, i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def density_plot(self, parameters: Optional[List[str]] = None,
                     chains: Optional[List[int]] = None,
                     figsize: Optional[Tuple[float, float]] = None,
                     ncols: int = 2) -> plt.Figure:
        """
        Create density plots for parameters
        
        Parameters
        ----------
        parameters : list of str, optional
            Parameters to plot. If None, plots all parameters.
        chains : list of int, optional
            Chain numbers to plot. If None, plots all chains combined.
        figsize : tuple, optional
            Figure size (width, height)
        ncols : int
            Number of columns in subplot grid
            
        Returns
        -------
        matplotlib.Figure
            Figure object
        """
        if parameters is None:
            parameters = self.parameter_names[:12]  # Limit to first 12
        
        n_params = len(parameters)
        nrows = (n_params + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (ncols * 4, nrows * 3)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(parameters):
            row, col = i // ncols, i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            
            if chains is None:
                # Plot combined density
                samples = self.get_samples(param)
                ax.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue')
                sns.kdeplot(samples, ax=ax, color='darkblue')
            else:
                # Plot separate densities for each chain
                for chain_idx in chains:
                    samples = self.get_samples(param, chain_idx)
                    sns.kdeplot(samples, ax=ax, alpha=0.7, label=f'Chain {chain_idx + 1}')
                if len(chains) > 1:
                    ax.legend()
            
            ax.set_title(f'{param}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        # Hide empty subplots
        for i in range(n_params, nrows * ncols):
            row, col = i // ncols, i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def autocorrelation_plot(self, parameters: Optional[List[str]] = None,
                      max_lag: int = 50,
                      figsize: Optional[Tuple[float, float]] = None,
                      ncols: int = 2) -> plt.Figure:
        """
        Create autocorrelation plots for parameters
        
        Parameters
        ----------
        parameters : list of str, optional
            Parameters to plot. If None, plots all parameters.
        max_lag : int
            Maximum lag to compute
        figsize : tuple, optional
            Figure size (width, height)
        ncols : int
            Number of columns in subplot grid
            
        Returns
        -------
        matplotlib.Figure
            Figure object
        """
        if parameters is None:
            parameters = self.parameter_names[:8]  # Limit for readability
        
        n_params = len(parameters)
        nrows = (n_params + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (ncols * 4, nrows * 3)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(parameters):
            row, col = i // ncols, i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            
            # Compute autocorrelation for each chain
            for chain_idx in range(self.n_chains):
                samples = self.get_samples(param, chain_idx)
                autocorr = self._compute_autocorr(samples, max_lag)
                ax.plot(range(max_lag + 1), autocorr, alpha=0.7, 
                       label=f'Chain {chain_idx + 1}')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='0.1 threshold')
            ax.set_title(f'{param}')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.set_ylim(-0.2, 1.0)
            if self.n_chains > 1:
                ax.legend()
        
        # Hide empty subplots
        for i in range(n_params, nrows * ncols):
            row, col = i // ncols, i % ncols
            ax = axes[row, col] if nrows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _compute_autocorr(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation function"""
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag + 1]
    
    def pair_plot(self, parameters: Optional[List[str]] = None,
                   figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Create pairwise scatter plots
        
        Parameters
        ----------
        parameters : list of str, optional
            Parameters to plot. If None, uses first 5 parameters.
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns
        -------
        matplotlib.Figure
            Figure object
        """
        if parameters is None:
            parameters = self.parameter_names[:5]  # Limit for readability
        
        n_params = len(parameters)
        if n_params < 2:
            raise ValueError("Need at least 2 parameters for pairwise plots")
        
        if figsize is None:
            figsize = (n_params * 2, n_params * 2)
        
        fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
        
        for i, param1 in enumerate(parameters):
            for j, param2 in enumerate(parameters):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    samples = self.get_samples(param1)
                    ax.hist(samples, bins=30, alpha=0.7, color='skyblue')
                    ax.set_title(param1)
                elif i > j:
                    # Lower triangle: scatter plot
                    samples1 = self.get_samples(param1)
                    samples2 = self.get_samples(param2)
                    ax.scatter(samples2, samples1, alpha=0.5, s=1)
                    ax.set_xlabel(param2)
                    ax.set_ylabel(param1)
                else:
                    # Upper triangle: correlation coefficient
                    samples1 = self.get_samples(param1)
                    samples2 = self.get_samples(param2)
                    corr = np.corrcoef(samples1, samples2)[0, 1]
                    ax.text(0.5, 0.5, f'r = {corr:.3f}', 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=ax.transAxes, fontsize=14)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def show_diagnostics(self, parameter_regex: Optional[str] = None) -> None:
        """
        Generate convergence diagnostics table
        
        Parameters
        ----------
        parameter_regex : str, optional
            Regular expression pattern to match parameter names
            
        Returns
        -------
        None
        """
        if parameter_regex is None:
            parameters = self.parameter_names
        else:
            parameters = self.filter_parameters(parameter_regex)
        
        diag_data = []
        
        for param in parameters:
            if param not in self.diagnostics:
                continue
            
            diag = self.diagnostics[param]
            
            # Determine convergence status
            rhat = diag.get('Rhat', diag.get('rhat', np.nan))
            converged = '✅ ' if not np.isnan(rhat) and rhat < 1.1 \
                else '❌ ' if not np.isnan(rhat) else '❓ '
            
            # Get effective sample size and round it
            n_eff = diag.get('n_eff', np.nan)
            if not np.isnan(n_eff):
                n_eff = round(n_eff)
            
            # Calculate total sample size
            n_total = self.n_chains * self.n_samples
            
            row = {
                'parameter': param,
                'rhat': rhat,
                'mcmc_se': diag.get('mcmc_se', np.nan),
                'n_eff': n_eff,
                'n_total': n_total,
                'converged': converged
            }
            
            diag_data.append(row)
        
        # Handle empty diag_data
        if not diag_data:
            # Create empty DataFrame with proper columns
            columns = ['parameter', 'rhat', 'mcmc_se', 'n_eff', 'n_total', 'converged']
            df = pd.DataFrame(columns=columns)
            df.set_index('parameter', inplace=True)
            print(df)
            return
        
        df = pd.DataFrame(diag_data)
        df.set_index('parameter', inplace=True)        

        print(df)
    
    def effective_sample_size(self, parameter: str) -> float:
        """
        Get effective sample size for a parameter
        
        Parameters
        ----------
        parameter : str
            Parameter name
            
        Returns
        -------
        float
            Effective sample size
        """
        return self.diagnostics.get(parameter, {}).get('n_eff', np.nan)
    
    def rhat(self, parameter: str) -> float:
        """
        Get R-hat statistic for a parameter
        
        Parameters
        ----------
        parameter : str
            Parameter name
            
        Returns
        -------
        float
            R-hat statistic
        """
        return self.diagnostics.get(parameter, {}).get('Rhat', np.nan)
    
    def to_dataframe(self, parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert samples to pandas DataFrame
        
        Parameters
        ----------
        parameters : list of str, optional
            Parameters to include. If None, includes all parameters.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with samples (columns are parameters, rows are iterations)
        """
        if parameters is None:
            parameters = self.parameter_names
        
        data = {}
        for param in parameters:
            if param in self.chains:
                data[param] = self.get_samples(param)
        
        return pd.DataFrame(data)
    
    def export_samples(self, filename: str, parameters: Optional[List[str]] = None,
                      format: str = 'csv') -> None:
        """
        Export samples to file
        
        Parameters
        ----------
        filename : str
            Output filename
        parameters : list of str, optional
            Parameters to export. If None, exports all parameters.
        format : str
            Export format ('csv', 'parquet', 'pickle')
        """
        df = self.to_dataframe(parameters)
        
        if format == 'csv':
            df.to_csv(filename, index=False)
        elif format == 'parquet':
            df.to_parquet(filename, index=False)
        elif format == 'pickle':
            df.to_pickle(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"MCMCSamples(n_parameters={self.n_parameters}, "
                f"n_chains={self.n_chains}, n_samples={self.n_samples})")
    
    def __str__(self) -> str:
        """String representation"""
        info = [
            f"MCMC Samples",
            f"  Parameters: {self.n_parameters}",
            f"  Chains: {self.n_chains}",
            f"  Samples per chain: {self.n_samples}",
            f"  Total samples: {self.n_chains * self.n_samples}"
        ]
        
        if self.parameter_names:
            info.append(f"  Parameter names: {', '.join(self.parameter_names[:5])}")
            if len(self.parameter_names) > 5:
                info.append(f"    ... and {len(self.parameter_names) - 5} more")
        
        return '\n'.join(info)
