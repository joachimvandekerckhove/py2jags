"""
EZ-Diffusion Demo for Py2JAGS

This demo shows how to use the EZ-diffusion extension to estimate
diffusion model parameters from response time and accuracy data.
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import the package modules directly
from src.extensions.ez_diffusion import bayesian_parameter_estimation


def generate_synthetic_data(n_trials: int = 500, 
                          boundary: float = 1.0,
                          drift: float = 0.5, 
                          ndt: float = 0.3) -> tuple:
    """
    Generate synthetic EZ-diffusion data for testing
    
    Args:
        n_trials: Number of trials to generate
        boundary: True boundary parameter
        drift: True drift parameter  
        ndt: True non-decision time parameter
        
    Returns:
        tuple: (rt_data, accuracy_data, true_params)
    """
    
    np.random.seed(42)  # For reproducibility
    
    # Calculate EZ-DDM parameters using proper formulas (from visual-ez.R)
    ey = np.exp(-boundary * drift)
    Pc = 1 / (1 + ey)
    PRT = 2 * (drift ** 3) / boundary * ((ey + 1) ** 2) / \
        (2 * (-boundary) * drift * ey - ey * ey + 1)
    MDT = (boundary / (2 * drift)) * (1 - ey) / (1 + ey)
    MRT = MDT + ndt
    
    # Generate accuracy data
    accuracy_data = np.random.binomial(1, Pc, n_trials).tolist()
    
    # Generate RT data using proper EZ-DDM variance
    rt_variance = 1 / PRT
    rt_data = np.random.normal(MRT, np.sqrt(rt_variance), n_trials)
    
    # Ensure positive RTs
    rt_data = np.maximum(rt_data, 0.1)
    rt_data = rt_data.tolist()
    
    true_params = {
        'boundary': boundary,
        'drift': drift,
        'ndt': ndt
    }
    
    return rt_data, accuracy_data, true_params


def run_ez_diffusion_demo():
    """Run the EZ-diffusion parameter estimation demo"""
    
    print("Py2JAGS EZ-Diffusion Demo")
    print("=" * 30)
    print()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    rt_data, accuracy_data, true_params = generate_synthetic_data()
    
    print(f"Generated {len(rt_data)} trials")
    print(f"Mean RT: {np.mean(rt_data):.3f} seconds")
    print(f"Accuracy: {np.mean(accuracy_data):.3f}")
    print()
    
    print("True parameter values:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.3f}")
    print()
    
    # Estimate parameters
    print("Estimating parameters using JAGS...")

    results = bayesian_parameter_estimation(
        rt_data=rt_data,
        accuracy_data=accuracy_data,
        n_samples=3000,      # Increased samples
        n_burnin=3000,       # Increased burn-in
        n_chains=4,          # More chains for better convergence
        verbosity=2,
        debug=True
    )
    
    # Display results using the summary method
    print("\nSummary statistics:")
    
    # Create lambda functions with custom names
    quantile_025 = lambda x: np.quantile(x, 0.025) 
    quantile_025.__name__ = "lower"
    
    quantile_975 = lambda x: np.quantile(x, 0.975)
    quantile_975.__name__ = "upper"
    
    results.summary(
        parameter_regex = "^(boundary|drift|ndt)$", 
        summary_fcn     = [
            np.mean, 
            np.std, 
            np.median,
            quantile_025,
            quantile_975
            ]
    )
    
    # Show the full convergence diagnostics table
    print("\nFull convergence diagnostics table:")
    results.show_diagnostics()
    
    # Make a pair plot of the parameters
    fig = results.pair_plot()
    fig.savefig("pair_plot.png")
    
    # Make a trace plot of the parameters
    fig = results.trace_plot()
    fig.savefig("trace_plot.png")
    
    # Make a histogram of the parameters
    fig = results.density_plot()
    fig.savefig("density.png")
    
    # Make an autocorrelation plot of the parameters
    fig = results.autocorrelation_plot()
    fig.savefig("autocorrelation.png")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_ez_diffusion_demo() 