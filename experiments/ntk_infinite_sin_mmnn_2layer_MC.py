# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
from scipy.linalg import eigvalsh
from ntk import compute_ntk_2layer_montecarlo, sin, sin_dot
from tqdm import tqdm

# %% [markdown]
"""
# Experiment: 2-Layer NTK Analysis with ReLU Activation

We analyze the Neural Tangent Kernel for a 2-layer network with ReLU activation:
1. Compute NTK matrices for different (sigma_A, sigma_c) configurations
2. Plot the resulting matrices and their eigenvalue spectra
3. Analyze projection onto a*I + b*J subspace
"""

# %%
# we set up base parameters
key = random.PRNGKey(42)
input_dim = 5  # we fix input dimension
n_samples = 8  # we fix sample size
ranks_to_test = range(2,100,5) # we test different ranks[0] values
beta = 1.0  # we fix random bias scaling
n_mc_samples = 1000  # we fix number of Monte Carlo samples

# we test different sigma_A and sigma_c combinations
sigma_As = [jnp.sqrt(2)]
sigma_cs = [1.0]

# we generate test data
X = random.normal(key, (n_samples, input_dim))
X = X / jnp.linalg.norm(X, axis=1, keepdims=True)  # we normalize to unit sphere

# we store l2 losses and coordinates for each rank
l2_losses = {rank: [] for rank in ranks_to_test}
i_coords = {rank: [] for rank in ranks_to_test}  # we store I coordinates
j_coords = {rank: [] for rank in ranks_to_test}  # we store J coordinates

def compute_projection_matrices(n):
    """we compute normalized I and J matrices for projection"""
    I = jnp.eye(n)
    J = jnp.ones((n,n)) - jnp.eye(n)
    I_norm = jnp.linalg.norm(I, 'fro')
    J_norm = jnp.linalg.norm(J, 'fro')
    return I/I_norm, J/J_norm

def project_onto_IJ_space(K, I_norm, J_norm):
    """we project matrix K onto span{I,J}"""
    K_norm = jnp.linalg.norm(K, 'fro')
    K = K/K_norm  # we normalize K
    
    # we compute coefficients
    a = jnp.sum(K * I_norm)
    b = jnp.sum(K * J_norm)
    
    # we compute projection
    proj = a*I_norm + b*J_norm
    
    # we compute L2 loss
    l2_loss = jnp.linalg.norm(K - proj, 'fro')
    return l2_loss, a, b

# we compute and plot NTK matrices for each configuration
with tqdm(total=len(ranks_to_test)*len(sigma_As)*len(sigma_cs), desc="Overall Progress") as pbar:
    for rank in ranks_to_test:
        print(f"\nTesting rank[0]={rank}")
        
        I_norm, J_norm = compute_projection_matrices(n_samples)
        
        for sigma_A in sigma_As:
            for sigma_c in sigma_cs:
                try:
                    ntk = compute_ntk_2layer_montecarlo(
                        X,
                        ranks=[rank,1],
                        sigma_A=sigma_A,
                        sigma_c=sigma_c,
                        beta=beta,
                        activation_fn=sin,
                        activation_dot_fn=sin_dot,
                        key=key,
                        n_samples=n_mc_samples
                    )
                    
                    # we compute L2 loss and coordinates from projection
                    l2_loss, i_coord, j_coord = project_onto_IJ_space(ntk, I_norm, J_norm)
                    l2_losses[rank].append(l2_loss)
                    i_coords[rank].append(i_coord)
                    j_coords[rank].append(j_coord)
                    
                    # we plot the NTK matrix
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(ntk, cmap='viridis', annot=True, fmt='.3f')
                    plt.title(f'NTK Matrix (rank={rank}, σA={sigma_A}, σc={sigma_c})')
                    plt.tight_layout()
                    plt.show()
                    
                    # we plot eigenvalue spectrum
                    eigenvals = jnp.linalg.eigvalsh(ntk)
                    plt.figure(figsize=(8, 6))
                    plt.plot(range(1, len(eigenvals) + 1), jnp.sort(eigenvals)[::-1], 'o-')
                    plt.title(f'Eigenvalue Spectrum (rank={rank}, σA={sigma_A}, σc={sigma_c})')
                    plt.xlabel('Index')
                    plt.ylabel('Eigenvalue')
                    plt.grid(True)
                    plt.show()
                    
                    print(f"Min eigenvalue: {jnp.min(eigenvals):.3f}")
                    print(f"Max eigenvalue: {jnp.max(eigenvals):.3f}")
                    print(f"Condition number: {jnp.max(eigenvals)/jnp.min(eigenvals):.3f}")
                    print(f"L2 loss from I,J projection: {l2_loss:.3f}")
                    print(f"I coordinate: {i_coord:.3f}")
                    print(f"J coordinate: {j_coord:.3f}")
                    
                except Exception as e:
                    print(f"Error at rank={rank}, σA={sigma_A}, σc={sigma_c}: {str(e)}")
                    continue
                
                pbar.update(1)

# %%
# we compute mean losses and fit power law
mean_losses = [jnp.mean(jnp.array(l2_losses[rank])) for rank in ranks_to_test]
ranks_array = jnp.array(list(ranks_to_test))
log_ranks = jnp.log(ranks_array)
log_losses = jnp.log(jnp.array(mean_losses))

# we fit power law using linear regression in log space
A = jnp.vstack([log_ranks, jnp.ones_like(log_ranks)]).T
m, b = jnp.linalg.lstsq(A, log_losses, rcond=None)[0]
power_law = jnp.exp(b) * ranks_array**m

# we plot L2 loss vs rank in log-log space
plt.figure(figsize=(10, 6))
plt.loglog(ranks_array, mean_losses, 'o', label='Data')
plt.loglog(ranks_array, power_law, 'r-', label=f'Fit: y ∝ x^{m:.2f}')
plt.title('Mean L2 Loss vs Rank (Log-Log)')
plt.xlabel('Rank (log scale)')
plt.ylabel('Mean L2 Loss (log scale)')
plt.legend()
plt.grid(True)
plt.show()

# we plot I and J coordinates vs rank
plt.figure(figsize=(12, 6))
mean_i_coords = [jnp.mean(jnp.array(i_coords[rank])) for rank in ranks_to_test]
mean_j_coords = [jnp.mean(jnp.array(j_coords[rank])) for rank in ranks_to_test]
plt.plot(list(ranks_to_test), mean_i_coords, 'o-', label='I coordinate')
plt.plot(list(ranks_to_test), mean_j_coords, 'o-', label='J coordinate')
plt.title('Mean I and J Coordinates vs Rank')
plt.xlabel('Rank')
plt.ylabel('Coordinate Value')
plt.legend()
plt.grid(True)
plt.show()

# %%
