# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import jax.numpy as jnp

# we define base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # we get the project root directory

# %%
# we load and combine individual results
data_dir = os.path.join(BASE_PATH, 'data', 'ntk_infinite_relu_mmnn_2layer_MC_random_bigrun')
results = {}

# we get all unique parameters from filenames
files = os.listdir(data_dir)
param_sets = set()
for f in files:
    if f.startswith('eigenvals_'):
        # we extract parameters from filename (e.g., 'eigenvals_dim2_n4_beta0.0_rank1.npy')
        params = '_'.join(f.split('_')[1:4])  # we get 'dim2_n4_beta0.0'
        param_sets.add(params)

# we load data for each parameter set
for params in param_sets:
    key = params  # e.g., 'dim2_n4_beta0.0'
    results[key] = {
        'eigenvals': {},
        'mean_ntks': {},
        'ntk_samples': {},
        'l2_losses': {},
        'ntk_stds_00': {},
        'ntk_stds_01': {}
    }
    
    # we find all files for this parameter set
    for f in files:
        if params in f:
            rank = int(f.split('_rank')[-1].split('.')[0])
            
            if f.startswith('eigenvals_'):
                results[key]['eigenvals'][rank] = np.load(os.path.join(data_dir, f))
            elif f.startswith('mean_ntk_'):
                results[key]['mean_ntks'][rank] = np.load(os.path.join(data_dir, f))
            elif f.startswith('ntk_samples_'):
                samples = np.load(os.path.join(data_dir, f))
                results[key]['ntk_samples'][rank] = samples
                
                # we compute L2 losses and standard deviations
                if samples is not None:
                    results[key]['l2_losses'][rank] = np.mean((samples - np.mean(samples))**2)
                    results[key]['ntk_stds_00'][rank] = np.std(np.diagonal(samples, axis1=1, axis2=2))
                    off_diag = samples - np.diagonal(samples, axis1=1, axis2=2)[:, :, None]
                    results[key]['ntk_stds_01'][rank] = np.std(off_diag)

# we extract parameters from results
input_dims = sorted(list(set([int(k.split('_')[0].replace('dim','')) for k in results.keys()])))  # we get unique dimensions
n_samples_list = sorted(list(set([int(k.split('_')[1].replace('n','')) for k in results.keys()])))  # we get unique sample sizes
betas = sorted(list(set([float(k.split('_')[2].replace('beta','')) for k in results.keys()])))  # we get unique betas
ranks = sorted(list(set([int(r) for k in results.keys() for r in results[k]['l2_losses'].keys()])))  # we get ranks

# we create output directories
output_dir = os.path.join(BASE_PATH, "figures", "after_big_run_statistics")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
# we analyze scaling with rank
plt.figure(figsize=(15, 5))
for beta in betas:
    for n in n_samples_list:
        for d in input_dims:
            key = f"dim{d}_n{n}_beta{beta}"
            if key in results:
                # we get mean L2 losses across ranks
                valid_ranks = [r for r in ranks if r in results[key]['l2_losses']]  # we only use ranks that exist
                if valid_ranks:  # we plot only if we have valid ranks
                    mean_losses = [np.mean(results[key]['l2_losses'][r]) for r in valid_ranks]
                    plt.subplot(131)
                    plt.loglog(valid_ranks, mean_losses, 'o-', label=f'd={d}, n={n}, β={beta}')
                    
                    # we get std of diagonal entries
                    stds_00 = [np.mean(results[key]['ntk_stds_00'][r]) for r in valid_ranks]
                    plt.subplot(132)
                    plt.loglog(valid_ranks, stds_00, 'o-', label=f'd={d}, n={n}, β={beta}')
                    
                    # we get std of off-diagonal entries
                    stds_01 = [np.mean(results[key]['ntk_stds_01'][r]) for r in valid_ranks]
                    plt.subplot(133)
                    plt.loglog(valid_ranks, stds_01, 'o-', label=f'd={d}, n={n}, β={beta}')

plt.subplot(131)
plt.title('L2 Loss vs Rank')
plt.xlabel('Rank')
plt.ylabel('L2 Loss')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(132)
plt.title('Diagonal STD vs Rank')
plt.xlabel('Rank')
plt.ylabel('STD')
plt.grid(True)

plt.subplot(133)
plt.title('Off-diagonal STD vs Rank')
plt.xlabel('Rank')
plt.ylabel('STD')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rank_scaling.png'), bbox_inches='tight')
plt.close()

