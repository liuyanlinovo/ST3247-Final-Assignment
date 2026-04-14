import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from simulator import simulate

def get_data_path(filename):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)

def load_observed_data():
    try:
        df_inf = pd.read_csv(get_data_path('infected_timeseries.csv'))
        df_rew = pd.read_csv(get_data_path('rewiring_timeseries.csv'))
        df_deg = pd.read_csv(get_data_path('final_degree_histograms.csv'))

        # Average results across 40 replicates
        obs_inf = df_inf.groupby('time')['infected_fraction'].mean().values
        obs_rew = df_rew.groupby('time')['rewire_count'].mean().values
        obs_deg = df_deg.groupby('degree')['count'].mean().values
        
        return np.concatenate([obs_inf, obs_rew, obs_deg])
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        exit()

def calculate_distance(sim_data, obs_data):
    return np.sqrt(np.mean((sim_data - obs_data)**2))

def run_abc_inference(obs_data, num_simulations=1000):
    print(f"Starting ABC inference ({num_simulations} simulations)...")
    results = []
    
    for i in range(num_simulations):
        # Sample parameters from uniform priors
        beta = np.random.uniform(0.05, 0.50)
        gamma = np.random.uniform(0.02, 0.20)
        rho = np.random.uniform(0.0, 0.8)
        
        try:
            inf, rew, deg = simulate(beta, gamma, rho)
            sim_data = np.concatenate([inf, rew, deg])
            dist = calculate_distance(sim_data, obs_data)
            results.append([beta, gamma, rho, dist])
        except Exception as e:
            continue
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{num_simulations}")

    return pd.DataFrame(results, columns=['beta', 'gamma', 'rho', 'distance'])

if __name__ == "__main__":
    x_obs = load_observed_data()
    
    df_results = run_abc_inference(x_obs, num_simulations=2000)
    
    # Accept top 5% samples as posterior
    quantile = 0.05
    threshold = df_results['distance'].quantile(quantile)
    posterior_samples = df_results[df_results['distance'] <= threshold]
    
    # Output results
    print("Results are as follows.")
    means = posterior_samples[['beta', 'gamma', 'rho']].mean()
    print(f"Inferred Beta:  {means['beta']:.4f}")
    print(f"Inferred Gamma: {means['gamma']:.4f}")
    print(f"Inferred Rho:   {means['rho']:.4f}")

    # Visualization
    plt.figure(figsize=(15, 5))
    params = [('beta', 'Beta'), ('gamma', 'Gamma'), ('rho', 'Rho')]
    for i, (col, name) in enumerate(params):
        plt.subplot(1, 3, i+1)
        plt.hist(posterior_samples[col], bins=15, color='skyblue', edgecolor='black')
        plt.axvline(means[col], color='red', linestyle='--')
        plt.title(f'Posterior of {name}')
        plt.xlabel('Value')
    
    plt.tight_layout()
    plt.savefig('inference_result.png')
    print("\nPlot saved as 'inference_result.png'")
    plt.show()
