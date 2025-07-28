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
from scipy import stats

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
# we analyze scaling with rank for beta=0.0
plt.figure(figsize=(10, 12))

# we store slopes and data for beta=0.0
slopes_data = []  # we store (dim, n_samples, metric_type, slope, r_value)

for n in n_samples_list:
    for d in input_dims:
        key = f"dim{d}_n{n}_beta0.0"
        if key in results:
            valid_ranks = [r for r in ranks if r in results[key]['l2_losses']]
            if valid_ranks:
                log_ranks = np.log(valid_ranks)
                
                # L2 loss
                plt.subplot(311)
                mean_losses = [np.mean(results[key]['l2_losses'][r]) for r in valid_ranks]
                log_losses = np.log(mean_losses)
                slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_losses)
                plt.loglog(valid_ranks, mean_losses, 'o-', label=f'd={d}, n={n} (slope={slope:.3f}, R²={r_value**2:.3f})')
                slopes_data.append((d, n, 'L2 Loss', slope, r_value))
                
                # Diagonal STD
                plt.subplot(312)
                stds_00 = [np.mean(results[key]['ntk_stds_00'][r]) for r in valid_ranks]
                log_diag = np.log(stds_00)
                slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_diag)
                plt.loglog(valid_ranks, stds_00, 'o-', label=f'd={d}, n={n} (slope={slope:.3f}, R²={r_value**2:.3f})')
                slopes_data.append((d, n, 'Diagonal STD', slope, r_value))
                
                # Off-diagonal STD
                plt.subplot(313)
                stds_01 = [np.mean(results[key]['ntk_stds_01'][r]) for r in valid_ranks]
                log_offdiag = np.log(stds_01)
                slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_offdiag)
                plt.loglog(valid_ranks, stds_01, 'o-', label=f'd={d}, n={n} (slope={slope:.3f}, R²={r_value**2:.3f})')
                slopes_data.append((d, n, 'Off-diagonal STD', slope, r_value))

# we customize plots
plt.subplot(311)
plt.title('L2 Loss vs Rank (β=0.0)')
plt.xlabel('Rank')
plt.ylabel('L2 Loss')
plt.grid(True)
plt.legend()

plt.subplot(312)
plt.title('Diagonal STD vs Rank (β=0.0)')
plt.xlabel('Rank')
plt.ylabel('STD')
plt.grid(True)
plt.legend()

plt.subplot(313)
plt.title('Off-diagonal STD vs Rank (β=0.0)')
plt.xlabel('Rank')
plt.ylabel('STD')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rank_scaling_beta0.png'), bbox_inches='tight', dpi=300)
plt.close()

# we print slopes in a formatted table
print("\nSlopes and R² values for β=0.0:")
print("-" * 80)
print(f"{'Dimension':^10} | {'n_samples':^10} | {'Metric':^15} | {'Slope':^10} | {'R²':^10}")
print("-" * 80)
for d, n, metric, slope, r_value in slopes_data:
    print(f"{d:^10} | {n:^10} | {metric:^15} | {slope:^10.3f} | {r_value**2:^10.3f}")
print("-" * 80)

# %%
# we analyze convergence values as a function of beta
plt.figure(figsize=(15, 5))

# we use the highest rank available as asymptotic value
max_rank = max(ranks)

# we store convergence data for each n and beta
convergence_data = {n: {'betas': [], 'l2_losses': [], 'diag_stds': [], 'offdiag_stds': []} for n in n_samples_list}

for beta in betas:
    for n in n_samples_list:
        # we average over dimensions for stability
        l2_losses = []
        diag_stds = []
        offdiag_stds = []
        
        for d in input_dims:
            key = f"dim{d}_n{n}_beta{beta}"
            if key in results and max_rank in results[key]['l2_losses']:
                l2_losses.append(np.mean(results[key]['l2_losses'][max_rank]))
                diag_stds.append(np.mean(results[key]['ntk_stds_00'][max_rank]))
                offdiag_stds.append(np.mean(results[key]['ntk_stds_01'][max_rank]))
        
        if l2_losses:  # we only add if we have data
            convergence_data[n]['betas'].append(beta)
            convergence_data[n]['l2_losses'].append(np.mean(l2_losses))
            convergence_data[n]['diag_stds'].append(np.mean(diag_stds))
            convergence_data[n]['offdiag_stds'].append(np.mean(offdiag_stds))

# we plot convergence values vs beta
plt.subplot(131)
for n in n_samples_list:
    if convergence_data[n]['betas']:  # we only plot if we have data
        plt.semilogy(convergence_data[n]['betas'], convergence_data[n]['l2_losses'], 'o-', 
                label=f'n={n}')
plt.title(f'Asymptotic L2 Loss vs β\n(rank={max_rank})')
plt.xlabel('β')
plt.ylabel('L2 Loss (log scale)')
plt.grid(True)
plt.legend()

plt.subplot(132)
for n in n_samples_list:
    if convergence_data[n]['betas']:
        plt.semilogy(convergence_data[n]['betas'], convergence_data[n]['diag_stds'], 'o-',
                label=f'n={n}')
plt.title(f'Asymptotic Diagonal STD vs β\n(rank={max_rank})')
plt.xlabel('β')
plt.ylabel('STD (log scale)')
plt.grid(True)
plt.legend()

plt.subplot(133)
for n in n_samples_list:
    if convergence_data[n]['betas']:
        plt.semilogy(convergence_data[n]['betas'], convergence_data[n]['offdiag_stds'], 'o-',
                label=f'n={n}')
plt.title(f'Asymptotic Off-diagonal STD vs β\n(rank={max_rank})')
plt.xlabel('β')
plt.ylabel('STD (log scale)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_vs_beta.png'), bbox_inches='tight', dpi=300)
plt.close()

# we print convergence values in a table
print("\nAsymptotic values vs β:")
print("-" * 100)
print(f"{'n_samples':^10} | {'β':^10} | {'L2 Loss':^20} | {'Diagonal STD':^20} | {'Off-diagonal STD':^20}")
print("-" * 100)
for n in n_samples_list:
    for i, beta in enumerate(convergence_data[n]['betas']):
        print(f"{n:^10} | {beta:^10.1f} | {convergence_data[n]['l2_losses'][i]:^20.3e} | "
              f"{convergence_data[n]['diag_stds'][i]:^20.3e} | {convergence_data[n]['offdiag_stds'][i]:^20.3e}")
print("-" * 100)

