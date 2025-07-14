"""
3D Matrix Demo for Py2JAGS

This script demonstrates how to use Py2JAGS to estimate a 3D matrix variable.
It creates a hierarchical model where we estimate means for different groups,
conditions, and time points.
"""

import numpy as np
import os
import sys

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import the package modules directly
import src as p2


def generate_3d_data():
    """Generate synthetic 3D data for demonstration
    
    Returns:
        tuple: (data_dict, true_means) where data_dict is observations and 
        true_means is the true 3D matrix
    """
    np.random.seed(42)
    
    # Dimensions: 3 groups, 4 conditions, 5 time points
    n_groups = 3
    n_conditions = 4
    n_times = 5
    n_obs_per_cell = 10  # Number of observations per cell
    
    # Generate true means for each cell of the 3D matrix
    true_means = np.random.normal(0, 2, (n_groups, n_conditions, n_times))
    
    # Add some structure to make it more interesting
    for g in range(n_groups):
        for c in range(n_conditions):
            for t in range(n_times):
                # Add group effect
                true_means[g, c, t] += g * 0.5
                # Add condition effect
                true_means[g, c, t] += c * 0.3
                # Add time trend
                true_means[g, c, t] += t * 0.2
    
    # Generate observations
    data = []
    group_idx = []
    condition_idx = []
    time_idx = []
    
    for g in range(n_groups):
        for c in range(n_conditions):
            for t in range(n_times):
                # Generate observations for this cell
                cell_obs = np.random.normal(
                    true_means[g, c, t], 
                    scale=1.0, 
                    size=n_obs_per_cell)
                data.extend(cell_obs)
                group_idx.extend([g + 1] * n_obs_per_cell)
                condition_idx.extend([c + 1] * n_obs_per_cell)
                time_idx.extend([t + 1] * n_obs_per_cell)
    
    data_dict = {
        'y': data,
        'group': group_idx,
        'condition': condition_idx,
        'time': time_idx,
        'N': len(data),
        'G': n_groups,
        'C': n_conditions,
        'T': n_times
    }
    return data_dict, true_means


def create_jags_model():
    """Create JAGS model for 3D matrix estimation"""
    
    model_string = """
    model {
        # Likelihood
        for (i in 1:N) {
            y[i] ~ dnorm(mu[group[i], condition[i], time[i]], tau)
        }
        
        # Priors for the 3D matrix of means
        for (g in 1:G) {
            for (c in 1:C) {
                for (t in 1:T) {
                    mu[g, c, t] ~ dnorm(0, 0.001)
                }
            }
        }
        
        # Prior for precision
        tau ~ dgamma(0.001, 0.001)
        sigma <- 1/sqrt(tau)
        
        # Derived quantities for interpretation
        # Group effects (marginal means across conditions and time)
        for (g in 1:G) {
            group_mean[g] <- mean(mu[g, 1:C, 1:T])
        }
        
        # Condition effects (marginal means across groups and time)
        for (c in 1:C) {
            condition_mean[c] <- mean(mu[1:G, c, 1:T])
        }
        
        # Time effects (marginal means across groups and conditions)
        for (t in 1:T) {
            time_mean[t] <- mean(mu[1:G, 1:C, t])
        }
        
        # Overall mean
        overall_mean <- mean(mu[1:G, 1:C, 1:T])
    }
    """
    
    return model_string


def run_3d_demo():
    """Run the 3D matrix demo"""
    
    print("Py2JAGS 3D Matrix Demo")
    print("======================")
    
    # Generate synthetic data
    print("Generating synthetic 3D data...")
    data_dict, true_means = generate_3d_data()
    
    print(f"Data dimensions: {data_dict['G']} groups × {data_dict['C']} " + \
          f"conditions × {data_dict['T']} time points")
    print(f"Total observations: {data_dict['N']}")
    print(f"True means shape: {true_means.shape}")
    
    # Create JAGS model
    model_string = create_jags_model()
    
    # Run JAGS
    print("Running JAGS model...")
    samples = p2.run_jags(
        model_string=model_string,
        data_dict=data_dict,
        init=None,
        monitorparams=[
            'mu',
            'sigma',
            'group_mean',
            'condition_mean',
            'time_mean',
            'overall_mean'
        ],
        nchains=3,
        nadapt=1000,
        nburnin=2000,
        nsamples=5000,
        thin=2,
        parallel=True,
        verbosity=2
    )
    
    print("\nResults:")
    print("=" * 50)
    
    # Display overall statistics
    samples.summary("overall_mean|sigma", summary_fcn=[np.mean, np.std, np.median])
    
    # Display group effects
    print("\nGroup Effects:")
    for g in range(data_dict['G']):
        param_name = f'group_mean_{g+1}'
        true_group_mean = true_means[g, :, :].mean()
        if param_name in samples.stats:
            print(f"  Group {g+1}: {samples.stats[param_name]['mean']:.3f} " + \
                  f"(true: {true_group_mean:.3f})")
        else:
            print(f"  Group {g+1}: Parameter {param_name} not found in stats")
    
    # Display condition effects
    print("\nCondition Effects:")
    for c in range(data_dict['C']):
        param_name = f'condition_mean_{c+1}'
        true_condition_mean = true_means[:, c, :].mean()
        if param_name in samples.stats:
            print(f"  Condition {c+1}: {samples.stats[param_name]['mean']:.3f} " + \
                  f"(true: {true_condition_mean:.3f})")
        else:
            print(f"  Condition {c+1}: Parameter {param_name} not found in stats")
    
    # Display time effects
    print("\nTime Effects:")
    for t in range(data_dict['T']):
        param_name = f'time_mean_{t+1}'
        true_time_mean = true_means[:, :, t].mean()
        if param_name in samples.stats:
            print(f"  Time {t+1}: {samples.stats[param_name]['mean']:.3f} " + \
                  f"(true: {true_time_mean:.3f})")
        else:
            print(f"  Time {t+1}: Parameter {param_name} not found in stats")
    
    # Display sample of 3D matrix estimates
    print("\n3D Matrix Sample (Group 1, All Conditions × Time Points):")
    print("Condition\\Time", end="")
    for t in range(data_dict['T']):
        print(f"    T{t+1:1d}", end="")
    print()
    
    for c in range(data_dict['C']):
        print(f"  Cond {c+1:1d}     ", end="")
        for t in range(data_dict['T']):
            param_name = f'mu_1_{c+1}_{t+1}'
            if param_name in samples.stats:
                print(f"{samples.stats[param_name]['mean']:6.2f}", end="")
            else:
                print(f"  N/A ", end="")
        print()
    
    print("\nTrue values (Group 1):")
    print("Condition\\Time", end="")
    for t in range(data_dict['T']):
        print(f"    T{t+1:1d}", end="")
    print()
    
    for c in range(data_dict['C']):
        print(f"  Cond {c+1:1d}     ", end="")
        for t in range(data_dict['T']):
            print(f"{true_means[0, c, t]:6.2f}", end="")
        print()
    
    # Display convergence diagnostics
    print("\nConvergence Diagnostics (sample):")
    samples.show_diagnostics()
    
    print(f"\nModel successfully estimated {data_dict['G']} × " + \
          f"{data_dict['C']} × {data_dict['T']} = " + \
          f"{data_dict['G']*data_dict['C']*data_dict['T']} parameters!")
    print("Demo completed successfully!")


if __name__ == "__main__":
    run_3d_demo() 