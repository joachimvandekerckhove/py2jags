"""
Demo script for Py2JAGS

This script demonstrates how to use Py2JAGS to run a simple JAGS model.
It creates a basic linear regression example.
"""

import os
import sys
import numpy as np

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import the package modules directly
import src as p2


def create_demo_model_and_data():
    """Create a simple linear regression model and data for demonstration"""
    
    # Model file (simple linear regression)
    model_string = """
    model {
        # Priors
        alpha ~ dnorm(0, 0.001)
        beta ~ dnorm(0, 0.001)
        tau ~ dgamma(0.001, 0.001)
        
        # Likelihood
        for (i in 1:n) {
            y[i] ~ dnorm(mu[i], tau)
            mu[i] <- alpha + beta * x[i]
        }
        
        # Derived quantities
        sigma <- 1 / sqrt(tau)
    }
    """
    
    # Data file
    np.random.seed(42)
    
    alpha = 2
    beta = 1.5
    sigma = 0.5
    n = 20
    x = np.linspace(1, 10, n)
    y = alpha + beta * x + np.random.normal(0, sigma, n)
    
    data_dict = {
        'n': n,
        'x': x.tolist(),
        'y': y.tolist()
    }
    
    true_params = {
        'alpha': alpha,
        'beta': beta,
        'sigma': sigma
    }
    
    return model_string, data_dict, true_params


def run_demo():
    """Run the Py2JAGS demo"""
    
    print("Py2JAGS Demo")
    print("==============")
    
    # Create demo files
    model_string, data_dict, true_params = create_demo_model_and_data()
    
    # Run JAGS
    print("\nRunning JAGS...")
    samples = p2.run_jags(
        model_string=model_string,
        data_dict=data_dict,
        monitorparams=['alpha', 'beta', 'sigma'],
        nchains=2,
        nburnin=1000,
        nsamples=2000,
        thin=1,
        verbosity=1
    )
    
    print("\nResults:")
    lower = lambda x: np.quantile(x, 0.025)
    lower.__name__ = "lower"
    upper = lambda x: np.quantile(x, 0.975)
    upper.__name__ = "upper"
    samples.summary(
        parameter_regex = "^(alpha|beta|sigma)$", 
        summary_fcn     = [
            np.mean, 
            np.std, 
            np.median,
            lower,
            upper])

    print(f"Total parameters: {samples.n_parameters}")
    print(f"Samples per chain: {samples.n_samples}")
    print(f"Number of chains: {samples.n_chains}")
    
    print("\nTrue parameter values:")
    for param_name, param_value in true_params.items():
        print(f"{param_name}: {param_value:.3f}")
    
    

if __name__ == "__main__":
    run_demo() 