# Py2JAGS

**A Python interface to JAGS (Just Another Gibbs Sampler)**

Py2JAGS provides a Python interface to JAGS that replicates the functionality of the Trinity MATLAB toolbox. It works by creating JAGS script files and calling JAGS via system commands, making it easy to run Bayesian analysis in Python.

## Features

- **Trinity-compatible API**: Familiar interface for users of the Trinity MATLAB toolbox
- **Direct JAGS integration**: No Python wrappers needed, uses JAGS directly
- **CODA file parsing**: Automatic parsing of JAGS output files
- **Convergence diagnostics**: Built-in R-hat calculation and other diagnostics
- **Parallel execution**: Support for GNU parallel when available
- **Professional structure**: Clean, extensible codebase with proper documentation

## Installation

### Prerequisites

1. **JAGS**: Must be installed and accessible from the command line
   ```bash
   # Ubuntu/Debian
   sudo apt-get install jags
   
   # macOS with Homebrew
   brew install jags
   
   # Or download from: https://mcmc-jags.sourceforge.io/
   ```

2. **Python dependencies**: NumPy is required
   ```bash
   pip install numpy
   ```

### Installing Py2JAGS

Clone or download this repository and add the src directory to your Python path:

```python
import sys
sys.path.append('/path/to/py2jags/src')

# Import functions directly
from core import run_jags
from utils import write_s_data

# Or import extensions
from extensions.ez_diffusion import bayesian_parameter_estimation
```

## Quick Start

### Basic Usage

```python
from core import run_jags

# Run a JAGS model
stats, chains, diagnostics, info = run_jags(
    model='model.jags',           # Path to JAGS model file
    data='data.R',                # Path to data file
    monitorparams=['theta', 'sigma'],  # Parameters to monitor
    nchains=4,                    # Number of chains
    nburnin=1000,                 # Burn-in iterations
    nsamples=5000,                # Sample iterations
    thin=1,                       # Thinning interval
    workingdir='./jags_output'    # Working directory
)

# Access results
print(f"theta estimate: {stats['theta']['mean']:.3f}")
print(f"R-hat: {diagnostics['theta']['rhat']:.3f}")
```

### Using Extensions

```python
from extensions.ez_diffusion import bayesian_parameter_estimation

# Estimate EZ-diffusion parameters
results = bayesian_parameter_estimation(
    rt_data=response_times,
    accuracy_data=accuracy,
    n_samples=5000,
    n_chains=4
)

print(f"Boundary: {results.boundary:.3f}")
print(f"Drift: {results.drift:.3f}")
print(f"Non-decision time: {results.ndt:.3f}")
```

## Package Structure

```
py2jags/                      # Package root
├── README.md                 # This file
├── src/                      # Source code
│   ├── __init__.py           # Main package interface
│   ├── core.py               # Main run_jags function
│   ├── runner.py             # JAGS execution engine
│   ├── reader.py             # CODA file parsing
│   ├── utils.py              # Utility functions
│   └── extensions/           # Specialized modules
│       ├── __init__.py
│       └── ez_diffusion.py
├── basic_demo.py             # Linear regression demo
├── matrix_3d_demo.py         # 3D matrix estimation demo
└── ez_diffusion_demo.py      # EZ-diffusion demo
```

## Demos

### Basic Demo
```bash
python basic_demo.py
```
Demonstrates basic linear regression with JAGS.

### 3D Matrix Demo
```bash
python matrix_3d_demo.py
```
Shows estimation of a 3D matrix of parameters (groups × conditions × time).

### EZ-Diffusion Demo
```bash
python ez_diffusion_demo.py
```
Demonstrates parameter estimation for the EZ-diffusion model.

## API Reference

### Main Functions

#### `run_jags(**kwargs)`
Main interface to run JAGS models.

**Parameters:**
- `model`: Path to JAGS model file
- `data`: Path to data file  
- `monitorparams`: List of parameters to monitor
- `nchains`: Number of chains (default: 4)
- `nburnin`: Burn-in iterations (default: 1000)
- `nsamples`: Sample iterations (default: 5000)
- `thin`: Thinning interval (default: 1)
- `workingdir`: Working directory (default: 'wdir')
- `parallel`: Use parallel execution (default: True)
- `verbosity`: Verbosity level (default: 0)

**Returns:**
- `stats`: Parameter summary statistics
- `chains`: Raw MCMC samples
- `diagnostics`: Convergence diagnostics
- `info`: Additional information

### Extensions

#### `bayesian_parameter_estimation(rt_data, accuracy_data, **kwargs)`
Estimate EZ-diffusion parameters from response time and accuracy data.

**Parameters:**
- `rt_data`: List of response times
- `accuracy_data`: List of accuracy values (0/1)
- `n_samples`: Number of samples (default: 5000)
- `n_chains`: Number of chains (default: 4)

**Returns:**
- `Parameters` object with boundary, drift, and ndt attributes

## Development

### Adding New Extensions

1. Create a new module in `src/py2jags/extensions/`
2. Implement your interface functions
3. Add imports to `src/py2jags/extensions/__init__.py`
4. Create a demo in `demos/`

### Running Tests

```bash
# Run all demos to test functionality
python basic_demo.py
python matrix_3d_demo.py
python ez_diffusion_demo.py
```

## Requirements

- Python 3.6+
- NumPy
- JAGS 4.0+
- GNU parallel (optional, for parallel execution)

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Common Issues

1. **"JAGS not found"**: Ensure JAGS is installed and in your PATH
2. **"Permission denied"**: Check file permissions in working directory
3. **"Module not found"**: Ensure the src directory is in your Python path

### Getting Help

1. Check the demos for usage examples
2. Review the API documentation above
3. Ensure all prerequisites are installed
4. Check JAGS installation: `jags --help`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests/demos for new functionality
4. Submit a pull request

## Acknowledgments

- Based on the Trinity MATLAB toolbox
- Uses JAGS for Bayesian computation
- Inspired by the need for a pure Python JAGS interface 