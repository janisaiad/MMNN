# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
from scipy.linalg import eigvalsh
from ntk import compute_ntk_nngp_recursive, sin, sin_dot
from tqdm import tqdm

# %% [markdown]
"""
# Experiment: NTK Spectrum Analysis with Phase Transition

We analyze the eigenvalue spectrum of Neural Tangent Kernels to find stable growth regimes:
1. First identify (sigma_A, sigma_c) pairs that give stable growth
2. Plot spectrum for a stable configuration and analyze growth rates
"""

# %%
# we set up base parameters
key = random.PRNGKey(42)
input_dim = 5  # we fix input dimension to 10
rank = 10  # we fix rank to 10
beta = 1.0  # we fix random bias scaling
n_samples = 8  # we fix sample size
depths = list(range(1, 5))  # we test depths 1 to 10

# we test different sigma_A and sigma_c combinations
sigma_As = [1.0,1.1,1.2,1.3,1.4,jnp.sqrt(2)]
sigma_cs = [1.0]

# we generate test data
X = random.normal(key, (n_samples, input_dim))
X = X / jnp.linalg.norm(X, axis=1, keepdims=True)  # we normalize to unit sphere
# we store growth rates for heatmap
growth_rates = jnp.zeros((len(sigma_As), len(sigma_cs)))
linreg_coeffs = jnp.zeros((len(sigma_As), len(sigma_cs)))
stable_configs = []
for i, sigma_A in tqdm(enumerate(sigma_As), desc="Testing σA values", total=len(sigma_As)):
    for j, sigma_c in tqdm(enumerate(sigma_cs), desc=f"Testing σc values for σA={sigma_A}", leave=False, total=len(sigma_cs)):
        max_eigenval_ratio = 0
        try:
            eigenvals_by_depth = []
            l2_norms_by_depth = []  # we track L2 norms of differences
            ntk_norms_by_depth = []  # we track NTK norms
            prev_ntk = None  # we store previous NTK for diff computation
            
            # we create basis vectors for I and J decomposition
            I = jnp.eye(n_samples)  # identity matrix

            J = (jnp.ones((n_samples, n_samples)) - I )   # normalized all-ones matrix
            J = J / jnp.linalg.norm(J)
            I = I / jnp.linalg.norm(I)
        
            # we store projections for each depth
            I_projs = []
            J_projs = []
            all_eigenvals_for_plot = [] # we store all eigenvalues for plotting their growth
            
            for L in tqdm(depths, desc=f"Testing depths for σA={sigma_A}, σc={sigma_c}", leave=False):
                print(f"Testing depth {L} for σA={sigma_A}, σc={sigma_c}")
                d_hidden = [rank] * (L-1)
                kernels = compute_ntk_nngp_recursive(X, L, d_hidden, sigma_A, sigma_c, beta, activation_fn=sin, activation_dot_fn=sin_dot)
                ntk_final = kernels['ntk'][L]
                # we ensure symmetry by averaging with transpose
                ntk_final = (ntk_final + ntk_final.T) / 2
                
                
                ntk_norm = jnp.linalg.norm(ntk_final, ord='fro')
                ntk_norms_by_depth.append(ntk_norm)
                
                I_proj_coeff = jnp.abs(jnp.sum(ntk_final * I))  # normalized projection coefficient for I
                J_proj_coeff = jnp.abs(jnp.sum(ntk_final * J))  # normalized projection coefficient for J    
                I_projs.append(I_proj_coeff)
                J_projs.append(J_proj_coeff)
            
                eigenvals = eigvalsh(ntk_final)
                all_eigenvals_for_plot.append(eigenvals/ntk_norm) # we collect eigenvalues for the plot
                l2_norms_by_depth.append(jnp.linalg.norm(ntk_final - I*I_proj_coeff - J*J_proj_coeff, ord='fro'))
                
                
                if jnp.any(jnp.isnan(eigenvals)): #check for NaN values early
                    raise ValueError("NaN eigenvalues detected")
                    
                eigenvals_by_depth.append(jnp.max(eigenvals))
                print(ntk_final)

            
            plt.figure(figsize=(10, 8)) #we plot the growth of each eigenvalue vs depth
            all_eigenvals_for_plot = jnp.array(all_eigenvals_for_plot)
            for k in range(n_samples):
                plt.plot(depths, all_eigenvals_for_plot[:, k], marker='o', linestyle='-', alpha=0.7, label=f'Eigenvalue k={k+1}')
            plt.yscale('log')
            plt.title(f'Eigenvalue Growth vs. Depth (σA={sigma_A}, σc={sigma_c})')
            plt.xlabel('Depth (L)')
            plt.ylabel('Eigenvalue (log scale)')
            plt.legend()
            plt.grid(True)
            plt.show()

            
            # we plot NTK norms in log space
            plt.figure(figsize=(8, 6))
            norms = jnp.array(ntk_norms_by_depth)  # we convert list to array before taking log
            plt.plot(depths, norms, marker='o')
            
            # we perform linear regression on log norms
            depths_array = jnp.array(depths, dtype=jnp.float32)  # we convert depths to float array
            coeffs = jnp.polyfit(jnp.log(depths_array), jnp.log(norms), deg=1)
            growth_rate = coeffs[0]  # slope is the growth rate
            
            plt.plot(depths_array, jnp.exp(coeffs[0] * jnp.log(depths_array) + coeffs[1]), 'r--', 
                    label=f'Linear fit (growth rate = {growth_rate:.3f})')
            plt.title(f'Log NTK Norm vs Depth (σA={sigma_A}, σc={sigma_c})')
            plt.xlabel('Depth L')
            plt.ylabel('Log NTK Norm')
            plt.legend()
            plt.yscale('log')
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Error at σA={sigma_A}, σc={sigma_c}: {str(e)}")  # we log the specific error
            growth_rates = growth_rates.at[i, j].set(jnp.nan)
            continue
        
        
        
# we handle the case where all configs failed
if jnp.all(jnp.isnan(growth_rates)):
    print("Warning: All configurations resulted in NaN values")
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, 'No valid data to display\nAll configurations unstable', 
             horizontalalignment='center', verticalalignment='center')
    plt.title('Growth Rate Analysis Failed')
    plt.tight_layout()
    plt.show()
else:
    # we plot exponential growth rate heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(growth_rates, 
                xticklabels=[f"{x:.2f}" for x in sigma_cs],
                yticklabels=[f"{x:.2f}" for x in sigma_As],
                cmap='viridis',
                annot=True, fmt='.3f')
    plt.title('Exponential Growth Rate by Configuration')
    plt.xlabel('σc')
    plt.ylabel('σA')
    plt.tight_layout()
    plt.show()

    # we plot linear regression coefficient heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(linreg_coeffs, 
                xticklabels=[f"{x:.2f}" for x in sigma_cs],
                yticklabels=[f"{x:.2f}" for x in sigma_As],
                cmap='viridis',
                annot=True, fmt='.3f')
    plt.title('Linear Regression Coefficient (Log NTK Norm) by Configuration')
    plt.xlabel('σc')
    plt.ylabel('σA')
    plt.tight_layout()
    plt.show()

# %%
if len(stable_configs) > 0:
    # we choose the most stable configuration (lowest max ratio)
    stable_configs.sort(key=lambda x: x[2])
    best_sigma_A, best_sigma_c, _, best_growth = stable_configs[0]
    print(f"\nBest configuration: sigma_A={best_sigma_A}, sigma_c={best_sigma_c}, exp_growth={best_growth:.3f}")

    # we plot spectrum for the stable configuration
    plt.figure(figsize=(10, 6))
    data_types = ['Sphere', 'Gaussian', 'Cube']
    data_generators = [
        lambda k, n, d: random.normal(k, (n, d)) / jnp.linalg.norm(random.normal(k, (n, d)), axis=1, keepdims=True),
        lambda k, n, d: random.normal(k, (n, d)),
        lambda k, n, d: 2 * random.uniform(k, (n, d)) - 1
    ]

    for idx, (data_type, generator) in enumerate(tqdm(list(zip(data_types, data_generators)), desc="Testing data distributions")):
        eigenvalues_by_depth = []
        X = generator(key, n_samples, input_dim)
        
        for L in tqdm(depths, desc=f"Computing depths for {data_type}", leave=False):
            d_hidden = [rank] * (L-1)
            kernels = compute_ntk_nngp_recursive(X, L, d_hidden, best_sigma_A, best_sigma_c, beta)
            ntk_final = kernels['ntk'][L]
            # we ensure symmetry by averaging with transpose
            ntk_final = (ntk_final + ntk_final.T) / 2
            eigenvals = eigvalsh(ntk_final)
            eigenvalues_by_depth.append(eigenvals)
        
        eigenvalues_by_depth = jnp.array(eigenvalues_by_depth)
        for i in range(n_samples):
            plt.plot(depths, eigenvalues_by_depth[:, i],
                    alpha=0.5, label=f'{data_type} n={i+1}' if i==0 else None)

    plt.title(f'Stable NTK Eigenspectrum (σA={best_sigma_A:.2f}, σc={best_sigma_c:.2f})')
    plt.xlabel('Network Depth (L)')
    plt.ylabel('Eigenvalues')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No stable configurations found")

# %% [markdown]
"""
## Observations:

1. Found stable configuration with σA={best_sigma_A:.2f}, σc={best_sigma_c:.2f}, exp_growth={best_growth:.3f}
2. This configuration shows controlled growth of eigenvalues with depth
3. Different data distributions still show distinct but stable spectral patterns
4. Heatmap reveals patterns in exponential growth rates across configurations
"""
