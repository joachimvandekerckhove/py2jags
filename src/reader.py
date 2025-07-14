"""
CODA Reader for Py2JAGS

This module handles reading and processing JAGS CODA output files,
replicating the functionality of Trinity's jags2coda.m and readcoda.m.
"""

import os
import numpy as np
from typing import Dict, Any


class CodaReader:
    """
    CODA file reader that processes JAGS output files
    """
    
    def __init__(self, options: Dict[str, Any]):
        """
        Initialize CODA reader
        
        Parameters
        ----------
        options : dict
            Options dictionary with coda_files and workingdir
        """
        self.options = options
        self.coda_files = options['coda_files']
        self.workingdir = options['workingdir']
        self.verbosity = options.get('verbosity', 0)
    
    def read(self) -> Dict[str, Any]:
        """
        Read CODA files and return samples dictionary
        
        Returns
        -------
        dict
            Dictionary with 'samples' key containing parameter samples
        """
        
        # Read index and chain files
        samples = self._read_coda_files()
        
        return {'samples': samples}
    
    def _read_coda_files(self) -> Dict[str, np.ndarray]:
        """Read CODA index and chain files"""
        
        samples = {}
        
        # Process each chain
        for i, coda_stem in enumerate(self.coda_files):
            chain_samples = self._read_single_chain(coda_stem, i + 1)
            
            # Merge with existing samples
            for param_name, param_values in chain_samples.items():
                if param_name not in samples:
                    # Initialize with first chain
                    samples[param_name] = param_values.reshape(-1, 1)
                else:
                    # Concatenate with existing chains
                    samples[param_name] = np.column_stack([
                        samples[param_name], param_values.reshape(-1, 1)
                    ])
        
        return samples
    
    def _read_single_chain(self, coda_stem: str, chain_num: int) -> Dict[str, np.ndarray]:
        """
        Read a single chain's CODA files
        
        Parameters
        ----------
        coda_stem : str
            CODA file stem (e.g., "samples_1_")
        chain_num : int
            Chain number
            
        Returns
        -------
        dict
            Dictionary mapping parameter names to sample arrays
        """
        
        # Construct file paths
        index_file = os.path.join(self.workingdir, f"{coda_stem}index.txt")
        chain_file = os.path.join(self.workingdir, f"{coda_stem}chain1.txt")
        
        # Check if files exist
        if not os.path.exists(index_file):
            # Try alternative naming
            index_file = os.path.join(self.workingdir, f"{coda_stem[:-1]}index.txt")
            chain_file = os.path.join(self.workingdir, f"{coda_stem[:-1]}.txt")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"CODA index file not found: {index_file}")
        
        if not os.path.exists(chain_file):
            raise FileNotFoundError(f"CODA chain file not found: {chain_file}")
        
        # Read index file
        param_index = self._read_index_file(index_file)
        
        # Read chain file
        chain_data = self._read_chain_file(chain_file)
        
        # Extract parameter samples using index
        samples = self._extract_samples(param_index, chain_data)
        
        return samples
    
    def _read_index_file(self, index_file: str) -> Dict[str, tuple]:
        """
        Read CODA index file
        
        Parameters
        ----------
        index_file : str
            Path to index file
            
        Returns
        -------
        dict
            Dictionary mapping parameter names to (start, end) indices
        """
        
        param_index = {}
        
        with open(index_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line: parameter_name start_index end_index
                parts = line.split()
                if len(parts) >= 3:
                    param_name = parts[0]
                    start_idx = int(parts[1])
                    end_idx = int(parts[2])
                    
                    # Clean parameter name (remove brackets, etc.)
                    param_name = self._clean_parameter_name(param_name)
                    
                    param_index[param_name] = (start_idx, end_idx)
        
        return param_index
    
    def _read_chain_file(self, chain_file: str) -> np.ndarray:
        """
        Read CODA chain file
        
        Parameters
        ----------
        chain_file : str
            Path to chain file
            
        Returns
        -------
        ndarray
            Array of all sampled values
        """
        
        values = []
        
        with open(chain_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse line: iteration_number value
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        values.append(value)
                    except ValueError:
                        continue
        
        return np.array(values)
    
    def _extract_samples(self, param_index: Dict[str, tuple], 
                        chain_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract parameter samples using index information
        
        Parameters
        ----------
        param_index : dict
            Parameter index mapping names to (start, end) positions
        chain_data : ndarray
            Raw chain data
            
        Returns
        -------
        dict
            Dictionary mapping parameter names to sample arrays
        """
        
        samples = {}
        
        for param_name, (start_idx, end_idx) in param_index.items():
            # Convert to 0-based indexing
            start_pos = start_idx - 1
            end_pos = end_idx
            
            # Extract samples for this parameter
            if end_pos <= len(chain_data):
                param_samples = chain_data[start_pos:end_pos]
                samples[param_name] = param_samples
            else:
                print(f"Warning: Parameter {param_name} index out of bounds")
        
        return samples
    
    def _clean_parameter_name(self, param_name: str) -> str:
        """
        Clean parameter name by removing/replacing special characters
        
        Parameters
        ----------
        param_name : str
            Original parameter name
            
        Returns
        -------
        str
            Cleaned parameter name
        """
        
        # Replace brackets and commas with underscores
        cleaned = param_name.replace('[', '_')
        cleaned = cleaned.replace(',', '_')
        cleaned = cleaned.replace(']', '')
        cleaned = cleaned.replace('.', '')
        
        # Remove any double underscores
        while '__' in cleaned:
            cleaned = cleaned.replace('__', '_')
        
        # Remove trailing underscore
        if cleaned.endswith('_'):
            cleaned = cleaned[:-1]
        
        return cleaned 