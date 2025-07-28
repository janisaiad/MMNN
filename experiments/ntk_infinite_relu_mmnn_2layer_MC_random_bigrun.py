# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
from scipy.linalg import eigvalsh
from ntk import compute_ntk_2layer_montecarlo_random_field, relu, relu_dot # type: ignore
from tqdm import tqdm
import os
import pathlib
import numpy as np  # we add numpy for saving arrays

# Create output directories
pathlib.Path("figures/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun").mkdir(parents=True, exist_ok=True)

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
key = random.PRNGKey(41)
input_dims = range(2, 9)  # we test input dimensions from 1 to 8
n_samples_list = [4, 8, 16]  # we test different sample sizes
ranks_to_test = [1,2,5,10,20,50,100]  # we test different ranks[0] values
betas = [0.0, 0.1, 0.2, 1.0]  # we test different beta values
n_mc_samples = 300  # we fix number of Monte Carlo samples

# we test different sigma_A and sigma_c combinations
sigma_As = [jnp.sqrt(2)]
sigma_cs = [1.0]

# we store results for all configurations
results = {}

for input_dim in input_dims:
    for n_samples in n_samples_list:
        for beta in betas:
            config_key = f"dim{input_dim}_n{n_samples}_beta{beta}"
            results[config_key] = {
                'l2_losses': {rank: [] for rank in ranks_to_test},
                'i_coords': {rank: [] for rank in ranks_to_test},
                'j_coords': {rank: [] for rank in ranks_to_test},
                'ntk_stds_00': {rank: [] for rank in ranks_to_test},
                'ntk_stds_01': {rank: [] for rank in ranks_to_test},
                'mean_ntks': {rank: None for rank in ranks_to_test},
                'eigenvals': {rank: None for rank in ranks_to_test},
                'ntk_values': {rank: None for rank in ranks_to_test}  # we store all NTK values
            }

            # we generate test data
            X = random.normal(key, (n_samples, input_dim))
            X = X / jnp.linalg.norm(X, axis=1, keepdims=True)  # we normalize to unit sphere

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
            n_samples_ntk = 300  # we fix number of samples for NTK distribution analysis

            with tqdm(total=len(ranks_to_test)*len(sigma_As)*len(sigma_cs), 
                     desc=f"Progress for dim={input_dim}, n={n_samples}, beta={beta}") as pbar:
                for rank in ranks_to_test:
                    print(f"\nTesting rank[0]={rank}")
                    
                    I_norm, J_norm = compute_projection_matrices(n_samples)
                    
                    for sigma_A in sigma_As:
                        for sigma_c in sigma_cs:
                            try:
                                # we store NTK values for distribution analysis
                                ntk_samples = []
                                l2_loss_samples = []
                                i_coord_samples = []
                                j_coord_samples = []
                                
                                # we compute multiple NTK samples
                                for i in range(n_samples_ntk):
                                    ntk = compute_ntk_2layer_montecarlo_random_field(
                                        X,
                                        ranks=[rank,1],
                                        sigma_A=sigma_A,
                                        sigma_c=sigma_c,
                                        beta=beta,
                                        activation_fn=relu,
                                        activation_dot_fn=relu_dot,
                                        key=random.split(key)[1]+i*10,
                                        n_samples=n_mc_samples
                                    )
                                    ntk_samples.append(ntk)
                                    if jnp.isnan(ntk).any():
                                        print(f"NTK matrix contains NaNs at rank={rank}, σA={sigma_A}, σc={sigma_c}, sample={i}")
                                        continue
                                        
                                    l2_loss, i_coord, j_coord = project_onto_IJ_space(ntk, I_norm, J_norm)
                                    l2_loss_samples.append(l2_loss)
                                    i_coord_samples.append(i_coord)
                                    j_coord_samples.append(j_coord)

                                # we convert to arrays and store results
                                ntk_samples = jnp.array(ntk_samples)
                                results[config_key]['l2_losses'][rank].append(jnp.mean(jnp.array(l2_loss_samples)))
                                results[config_key]['i_coords'][rank].append(jnp.mean(jnp.array(i_coord_samples)))
                                results[config_key]['j_coords'][rank].append(jnp.mean(jnp.array(j_coord_samples)))
                                results[config_key]['ntk_stds_00'][rank].append(jnp.std(ntk_samples[:, 0, 0]))
                                results[config_key]['ntk_stds_01'][rank].append(jnp.std(ntk_samples[:, 0, 1]))
                                results[config_key]['mean_ntks'][rank] = jnp.mean(ntk_samples, axis=0)
                                results[config_key]['eigenvals'][rank] = jnp.array([jnp.linalg.eigvalsh(ntk) for ntk in ntk_samples])
                                results[config_key]['ntk_values'][rank] = ntk_samples  # we store all NTK values

                                # we save plots
                                plt.figure(figsize=(15, 5))
                                for idx, (i, j) in enumerate([(0,0), (0,1), (1,1)]):
                                    plt.subplot(1, 3, idx+1)
                                    entry_samples = ntk_samples[:, i, j]
                                    plt.hist(entry_samples, bins='auto', density=True)
                                    plt.axvline(jnp.mean(entry_samples), color='r', linestyle='--',
                                              label=f'Mean: {jnp.mean(entry_samples):.3f}\nStd: {jnp.std(entry_samples):.3f}')
                                    plt.title(f'NTK Distribution at ({i},{j})')
                                    plt.legend()
                                plt.suptitle(f'NTK Entry Distributions (dim={input_dim}, n={n_samples}, beta={beta}, rank={rank})')
                                plt.tight_layout()
                                plt.savefig(f'figures/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun/dist_dim{input_dim}_n{n_samples}_beta{beta}_rank{rank}.png')
                                plt.close()

                                # we plot mean NTK heatmap
                                plt.figure(figsize=(8, 6))
                                mean_ntk = results[config_key]['mean_ntks'][rank]
                                sns.heatmap(mean_ntk, cmap='viridis', annot=True, fmt='.3f')
                                plt.title(f'Mean NTK Matrix (dim={input_dim}, n={n_samples}, beta={beta}, rank={rank})')
                                plt.savefig(f'figures/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun/mean_ntk_dim{input_dim}_n{n_samples}_beta{beta}_rank{rank}.png')
                                plt.close()

                                # we convert JAX arrays to numpy before saving
                                np.save(f'data/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun/ntk_samples_dim{input_dim}_n{n_samples}_beta{beta}_rank{rank}.npy', 
                                        np.array(ntk_samples))
                                np.save(f'data/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun/mean_ntk_dim{input_dim}_n{n_samples}_beta{beta}_rank{rank}.npy',
                                        np.array(mean_ntk))
                                np.save(f'data/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun/eigenvals_dim{input_dim}_n{n_samples}_beta{beta}_rank{rank}.npy',
                                        np.array(results[config_key]['eigenvals'][rank]))

                            except Exception as e:
                                print(f"Error at dim={input_dim}, n={n_samples}, beta={beta}, rank={rank}: {str(e)}")
                                continue
                            
                            pbar.update(1)

# we convert results dictionary to numpy arrays before saving
numpy_results = {}
for key in results:
    numpy_results[key] = {k: np.array(v) if isinstance(v, (list, jnp.ndarray)) else v 
                         for k, v in results[key].items()}
np.save('data/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun/all_results.npy', numpy_results)

# we plot summary statistics across configurations
for config_key in results:
    dim, n, beta = config_key.split('_')
    plt.figure(figsize=(10, 6))
    mean_losses = [jnp.mean(jnp.array(results[config_key]['l2_losses'][rank])) for rank in ranks_to_test]
    plt.loglog(list(ranks_to_test), mean_losses, 'o-')
    plt.title(f'L2 Loss vs Rank ({config_key})')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('L2 Loss (log scale)')
    plt.grid(True)
    plt.savefig(f'figures/ntk_infinite_relu_mmnn_2layer_MC_random_bigrun/loss_summary_{config_key}.png')
    plt.close()

# %%
