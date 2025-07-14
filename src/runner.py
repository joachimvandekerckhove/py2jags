"""
JAGS Runner for Py2JAGS

This module handles JAGS script generation and execution, replicating the
functionality of Trinity's calljags_lnx.m file.
"""

import os
import subprocess
import tempfile
import shutil
import re
import time
from typing import Dict, List, Any, Optional
from .utils import generate_temp_filename, set_executable_permissions


class JagsRunner:
    """
    JAGS execution engine that creates scripts and runs JAGS via system commands
    """
    
    def __init__(self, options: Dict[str, Any]):
        """
        Initialize JAGS runner
        
        Parameters
        ----------
        options : dict
            Options dictionary from parse_options
        """
        self.options = options
        self.original_dir = os.getcwd()
        
        # Set JAGS executable path
        self.jags_executable = options.get('jags_executable', 'jags')
        self.jags_path = None  # Will be set in _launch_jags
        
    def run(self) -> Dict[str, Any]:
        """
        Execute JAGS with the given options
        
        Returns
        -------
        dict
            Updated options with execution results
        """
        
        start_time = time.time()
        
        try:
            # Change to working directory
            os.chdir(self.options['workingdir'])
            
            # Generate JAGS scripts for each chain
            script_start = time.time()
            self._make_jags_scripts()
            script_time = time.time() - script_start
            
            # Launch JAGS
            jags_start = time.time()
            self._launch_jags()
            jags_time = time.time() - jags_start
            
            # Save JAGS output
            output_start = time.time()
            self._save_jags_output()
            output_time = time.time() - output_start
            
            # Check for errors
            self._error_checking()
            
        finally:
            # Always return to original directory
            os.chdir(self.original_dir)
            
        total_time = time.time() - start_time
        
        # Store timing information
        self.options['timing'] = {
            'total_time': total_time,
            'script_generation_time': script_time,
            'jags_execution_time': jags_time,
            'output_processing_time': output_time
        }
        
        # Print timing information if verbose
        if self.options['verbosity'] > 0:
            self._print_timing_info()
            
        return self.options
    
    def _print_timing_info(self):
        """Print timing information in a formatted way"""
        timing = self.options['timing']
        
        print(f"\n=== JAGS Execution Timing ===")
        print(f"Script generation: {timing['script_generation_time']:.3f}s")
        print(f"JAGS execution:    {timing['jags_execution_time']:.3f}s")
        print(f"Output processing: {timing['output_processing_time']:.3f}s")
        print(f"Total time:        {timing['total_time']:.3f}s")
        print(f"=============================")
    
    def _make_jags_scripts(self):
        """Create JAGS script files for each chain"""
        
        model = self.options['model_file']
        datafile = self.options['data_file']
        outputname = self.options['outputname']
        modules = self.options['modules']
        nchains = self.options['nchains']
        nburnin = self.options['nburnin']
        nsamples = self.options['nsamples']
        monitorparams = self.options['monitorparams']
        thin = self.options['thin']
        init = self.options['init']
        seed = self.options['seed']
        
        script_files = []
        coda_files = []
        
        for ch in range(nchains):
            # Generate filenames
            script_file = generate_temp_filename(self.options, 'script', ch + 1)
            coda_file = f"{outputname}_{ch + 1}_"
            seed_file = f"{os.path.splitext(datafile)[0]}_{ch + 1}.seed"
            
            script_files.append(script_file)
            coda_files.append(coda_file)
            
            # Create seed file with RNG settings and optional initial values
            with open(seed_file, 'w') as f:
                f.write(f'.RNG.name <- "base::Mersenne-Twister"\n')
                f.write(f'.RNG.seed <- {seed + ch + 1}\n')
                
                # Add initial values if provided
                if init and len(init) > ch and init[ch]:
                    f.write('\n# Initial values\n')
                    for param, value in init[ch].items():
                        f.write(f'{param} <- {value}\n')
            
            # Create JAGS script
            with open(script_file, 'w') as f:
                # Load modules
                if modules:
                    for module in modules:
                        f.write(f'load {module}\n')
                
                # Basic JAGS commands
                f.write(f'model in "{model}"\n')
                f.write(f'data in "{datafile}"\n')
                f.write('compile, nchains(1)\n')
                
                # Load parameters from seed file (now includes initial values)
                f.write(f'parameters in "{seed_file}"\n')
                
                f.write('initialize\n')
                f.write(f'update {nburnin}\n')
                
                # Monitor parameters
                if monitorparams:
                    for param in monitorparams:
                        f.write(f'monitor set {param}, thin({thin})\n')
                
                # Monitor deviance if dic module loaded
                if 'dic' in modules:
                    f.write('monitor deviance\n')
                
                # Sample
                f.write(f'update {nsamples * thin}\n')
                f.write(f"coda *, stem('{coda_file}')\n")
            
            # Make script executable
            set_executable_permissions(script_file)
        
        self.options['scriptfile'] = script_files
        self.options['coda_files'] = coda_files
    
    def _launch_jags(self):
        """Launch JAGS execution"""
        
        nchains = self.options['nchains']
        verbosity = self.options['verbosity']
        doparallel = self.options['parallel']
        script_files = self.options['scriptfile']
        maxcores = self.options['maxcores']
        
        # Find JAGS executable
        self.jags_path = shutil.which(self.jags_executable)
        if not self.jags_path:
            raise RuntimeError(f"JAGS executable '{self.jags_executable}' not found in PATH")
        
        if verbosity >= 2:
            print(f"Using JAGS executable: {self.jags_path}")
        
        if doparallel and nchains > 1:
            self._run_parallel(script_files, maxcores, verbosity)
        else:
            self._run_serial(script_files, verbosity)
    
    def _run_parallel(self, script_files: List[str], maxcores: int, verbosity: int):
        """Run JAGS chains in parallel using GNU parallel"""
        
        # Check if GNU parallel is available
        parallel_path = shutil.which('parallel')
        if not parallel_path:
            if verbosity > 0:
                print("GNU parallel not found, falling back to serial execution")
            self._run_serial(script_files, verbosity)
            return
        
        # Create temporary batch file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            batch_file = f.name
            for script in script_files:
                f.write(f'{self.jags_path} {script}\n')
        
        try:
            # Prepare parallel command
            cmd = f'cat {batch_file} | parallel --max-procs {maxcores}'
            
            if verbosity > 0:
                print(f'Running {len(script_files)} chains (parallel execution)')
                print(f'$ {cmd}')
            
            # Execute with timing
            chain_start = time.time()
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            chain_time = time.time() - chain_start
            
            if verbosity > 0:
                print(f'Parallel execution completed in {chain_time:.3f}s')
            
            if result.returncode != 0:
                error_msg = self._parse_jags_errors(result.stderr + result.stdout)
                raise RuntimeError(f"JAGS parallel execution failed:\n{error_msg}")
            
            self.options['status'] = [result.returncode] * len(script_files)
            self.options['result'] = [result.stdout] * len(script_files)
            
        finally:
            # Clean up batch file
            if os.path.exists(batch_file) and not self.options.get('debug', False):
                os.unlink(batch_file)
    
    def _run_serial(self, script_files: List[str], verbosity: int):
        """Run JAGS chains serially"""
        
        status = []
        result = []
        chain_times = []
        
        for i, script in enumerate(script_files):
            cmd = f'{self.jags_path} {script}'
            
            if verbosity > 0:
                print(f'Running chain {i + 1} of {len(script_files)} (serial execution)')
            
            chain_start = time.time()
            proc_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            chain_time = time.time() - chain_start
            chain_times.append(chain_time)
            
            if verbosity > 0:
                print(f'Chain {i + 1} completed in {chain_time:.3f}s')
            
            status.append(proc_result.returncode)
            result.append(proc_result.stdout)
            
            if proc_result.returncode != 0:
                print(f"Warning: Chain {i + 1} returned non-zero exit code")
                if verbosity >= 1:
                    print(f"STDERR: {proc_result.stderr}")
        
        # Store individual chain times
        self.options['chain_times'] = chain_times
        
        # Check if any chains failed
        if any(s != 0 for s in status):
            failed_results = [result[i] for i, s in enumerate(status) if s != 0]
            error_msg = '\n'.join(failed_results)
            raise RuntimeError(f"JAGS execution failed:\n{error_msg}")
        
        self.options['status'] = status
        self.options['result'] = result
    
    def _save_jags_output(self):
        """Save JAGS output to files"""
        
        if not self.options['saveoutput']:
            return
        
        doparallel = self.options['parallel']
        result = self.options['result']
        outputname = self.options['outputname']
        
        if doparallel:
            filename = f'jags_output_{outputname}.txt'
            with open(filename, 'w') as f:
                f.write(result[0] if result else '')
        else:
            for i, output in enumerate(result):
                filename = f'jags_output_{outputname}_{i + 1}.txt'
                with open(filename, 'w') as f:
                    f.write(output)
    
    def _error_checking(self):
        """Check JAGS output for errors and warnings"""
        
        verbosity = self.options['verbosity']
        doparallel = self.options['parallel']
        showwarnings = self.options['showwarnings']
        result = self.options['result']
        
        def check_result(output: str, chain_id: Optional[int] = None):
            """Check a single result for errors and warnings"""
            
            # Check for errors
            error_patterns = [r"can't", r"RUNTIME ERROR", r"syntax error", r"failure"]
            for pattern in error_patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    if chain_id is not None:
                        msg = f"Error encountered in JAGS (chain {chain_id}):\n{output}"
                    else:
                        msg = f"Error encountered in JAGS (all chains):\n{output}"
                    raise RuntimeError(f"JAGS error:\n{msg}")
            
            # Check for warnings
            if showwarnings and re.search(r"WARNING", output, re.IGNORECASE):
                if chain_id is not None:
                    print(f"JAGS warning (chain {chain_id}):")
                else:
                    print("JAGS warning (all chains):")
                print(output)
            
            # Print output if verbose
            if verbosity >= 2:
                if chain_id is not None:
                    print(f"JAGS output (chain {chain_id}):\n{output}")
                else:
                    print(f"JAGS output (all chains):\n{output}")
        
        if doparallel:
            check_result(result[0] if result else "")
        else:
            for i, output in enumerate(result):
                check_result(output, i + 1)
    
    def _parse_jags_errors(self, error_text: str) -> str:
        """Parse and format JAGS error messages"""
        
        # Simple error parsing - could be enhanced
        lines = error_text.split('\n')
        error_lines = [line for line in lines if 
                      any(keyword in line.lower() for keyword in 
                          ['error', 'warning', 'fail', 'exception'])]
        
        if error_lines:
            return '\n'.join(error_lines)
        else:
            return error_text 