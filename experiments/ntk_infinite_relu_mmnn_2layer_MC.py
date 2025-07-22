# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
from scipy.linalg import eigvalsh
from ntk import compute_ntk_2layer_montecarlo, relu, relu_dot
from tqdm import tqdm

# %% [markdown]
"""
# Experiment: 2-Layer NTK Analysis with ReLU Activation

We analyze the Neural Tangent Kernel for a 2-layer network with ReLU activation:
1. Compute NTK matrices for different (sigma_A, sigma_c) configurations
2. Plot the resulting matrices and their eigenvalue spectra
"""

# %%
# we set up base parameters
key = random.PRNGKey(42)
input_dim = 5  # we fix input dimension
n_samples = 8  # we fix sample size
ranks = [10,1]  # we use ranks [10,1] for 2-layer network
beta = 1.0  # we fix random bias scaling
n_mc_samples = 1000  # we fix number of Monte Carlo samples

# we test different sigma_A and sigma_c combinations
sigma_As = [1.0, 1.1, 1.2, 1.3, 1.4, jnp.sqrt(2)]
sigma_cs = [1.0]

# we generate test data
X = random.normal(key, (n_samples, input_dim))
X = X / jnp.linalg.norm(X, axis=1, keepdims=True)  # we normalize to unit sphere

# we compute and plot NTK matrices for each configuration
for i, sigma_A in tqdm(enumerate(sigma_As), desc="Testing σA values"):
    for j, sigma_c in tqdm(enumerate(sigma_cs), desc=f"Testing σc values for σA={sigma_A}"):
        try:
            print(f"\nComputing NTK for σA={sigma_A}, σc={sigma_c}")
            
            ntk = compute_ntk_2layer_montecarlo(
                X,
                ranks=ranks,
                sigma_A=sigma_A,
                sigma_c=sigma_c,
                beta=beta,
                activation_fn=relu,
                activation_dot_fn=relu_dot,
                key=key,
                n_samples=n_mc_samples
            )
            
            # we plot the NTK matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(ntk, cmap='viridis', annot=True, fmt='.3f')
            plt.title(f'NTK Matrix (σA={sigma_A}, σc={sigma_c})')
            plt.tight_layout()
            plt.show()
            
            # we plot eigenvalue spectrum
            eigenvals = jnp.linalg.eigvalsh(ntk)
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(eigenvals) + 1), jnp.sort(eigenvals)[::-1], 'o-')
            plt.title(f'Eigenvalue Spectrum (σA={sigma_A}, σc={sigma_c})')
            plt.xlabel('Index')
            plt.ylabel('Eigenvalue')
            plt.grid(True)
            plt.show()
            
            print(f"Min eigenvalue: {jnp.min(eigenvals):.3f}")
            print(f"Max eigenvalue: {jnp.max(eigenvals):.3f}")
            print(f"Condition number: {jnp.max(eigenvals)/jnp.min(eigenvals):.3f}")
            
        except Exception as e:
            print(f"Error at σA={sigma_A}, σc={sigma_c}: {str(e)}")
            continue

# %%
