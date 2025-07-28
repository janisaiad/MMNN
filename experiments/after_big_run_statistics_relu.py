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
def plot_convergence(rank_to_use, output_filename):
    plt.figure(figsize=(15, 5))
    
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
                if key in results and rank_to_use in results[key]['l2_losses']:
                    l2_losses.append(np.mean(results[key]['l2_losses'][rank_to_use]))
                    diag_stds.append(np.mean(results[key]['ntk_stds_00'][rank_to_use]))
                    offdiag_stds.append(np.mean(results[key]['ntk_stds_01'][rank_to_use]))
            
            if l2_losses:  # we only add if we have data
                convergence_data[n]['betas'].append(beta)
                convergence_data[n]['l2_losses'].append(np.mean(l2_losses))
                convergence_data[n]['diag_stds'].append(np.mean(diag_stds))
                convergence_data[n]['offdiag_stds'].append(np.mean(offdiag_stds))
    
    # we plot convergence values vs beta
    plt.subplot(131)
    for n in n_samples_list:
        if convergence_data[n]['betas']:  # we only plot if we have data
            betas_array = np.array(convergence_data[n]['betas'])
            l2_losses_array = np.array(convergence_data[n]['l2_losses'])
            
            # we compute slope for β>0
            nonzero_mask = betas_array > 0
            if np.sum(nonzero_mask) > 1:  # we need at least 2 points for regression
                log_betas = np.log(betas_array[nonzero_mask])
                log_l2 = np.log(l2_losses_array[nonzero_mask])
                slope, _, r_value, _, _ = stats.linregress(log_betas, log_l2)
                label = f'n={n} (slope={slope:.3f}, R²={r_value**2:.3f})'
            else:
                label = f'n={n}'
            
            plt.loglog(betas_array, l2_losses_array, 'o-', label=label)
    
    plt.title(f'Asymptotic L2 Loss vs β\n(rank={rank_to_use})')
    plt.xlabel('β (log scale)')
    plt.ylabel('L2 Loss (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.legend()

    plt.subplot(132)
    for n in n_samples_list:
        if convergence_data[n]['betas']:
            betas_array = np.array(convergence_data[n]['betas'])
            diag_stds_array = np.array(convergence_data[n]['diag_stds'])
            
            # we compute slope for β>0
            nonzero_mask = betas_array > 0
            if np.sum(nonzero_mask) > 1:
                log_betas = np.log(betas_array[nonzero_mask])
                log_diag = np.log(diag_stds_array[nonzero_mask])
                slope, _, r_value, _, _ = stats.linregress(log_betas, log_diag)
                label = f'n={n} (slope={slope:.3f}, R²={r_value**2:.3f})'
            else:
                label = f'n={n}'
            
            plt.loglog(betas_array, diag_stds_array, 'o-', label=label)
    
    plt.title(f'Asymptotic Diagonal STD vs β\n(rank={rank_to_use})')
    plt.xlabel('β (log scale)')
    plt.ylabel('STD (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.legend()

    plt.subplot(133)
    for n in n_samples_list:
        if convergence_data[n]['betas']:
            betas_array = np.array(convergence_data[n]['betas'])
            offdiag_stds_array = np.array(convergence_data[n]['offdiag_stds'])
            
            # we compute slope for β>0
            nonzero_mask = betas_array > 0
            if np.sum(nonzero_mask) > 1:
                log_betas = np.log(betas_array[nonzero_mask])
                log_offdiag = np.log(offdiag_stds_array[nonzero_mask])
                slope, _, r_value, _, _ = stats.linregress(log_betas, log_offdiag)
                label = f'n={n} (slope={slope:.3f}, R²={r_value**2:.3f})'
            else:
                label = f'n={n}'
            
            plt.loglog(betas_array, offdiag_stds_array, 'o-', label=label)
    
    plt.title(f'Asymptotic Off-diagonal STD vs β\n(rank={rank_to_use})')
    plt.xlabel('β (log scale)')
    plt.ylabel('STD (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', dpi=300)
    plt.close()

    # we print slopes for β>0
    print(f"\nSlopes for β>0 at rank {rank_to_use}:")
    print("-" * 100)
    print(f"{'n_samples':^10} | {'Metric':^15} | {'Slope':^10} | {'R²':^10}")
    print("-" * 100)
    
    for n in n_samples_list:
        if convergence_data[n]['betas']:
            betas_array = np.array(convergence_data[n]['betas'])
            nonzero_mask = betas_array > 0
            if np.sum(nonzero_mask) > 1:
                # L2 Loss slope
                log_betas = np.log(betas_array[nonzero_mask])
                log_l2 = np.log(np.array(convergence_data[n]['l2_losses'])[nonzero_mask])
                slope_l2, _, r_value_l2, _, _ = stats.linregress(log_betas, log_l2)
                print(f"{n:^10} | {'L2 Loss':^15} | {slope_l2:^10.3f} | {r_value_l2**2:^10.3f}")
                
                # Diagonal STD slope
                log_diag = np.log(np.array(convergence_data[n]['diag_stds'])[nonzero_mask])
                slope_diag, _, r_value_diag, _, _ = stats.linregress(log_betas, log_diag)
                print(f"{n:^10} | {'Diagonal STD':^15} | {slope_diag:^10.3f} | {r_value_diag**2:^10.3f}")
                
                # Off-diagonal STD slope
                log_offdiag = np.log(np.array(convergence_data[n]['offdiag_stds'])[nonzero_mask])
                slope_offdiag, _, r_value_offdiag, _, _ = stats.linregress(log_betas, log_offdiag)
                print(f"{n:^10} | {'Off-diag STD':^15} | {slope_offdiag:^10.3f} | {r_value_offdiag**2:^10.3f}")
    print("-" * 100)

# we create plots for all ranks
for rank in ranks:
    output_filename = f'convergence_vs_beta_rank{rank}.png'
    plot_convergence(rank, output_filename)

# we print a message about the main plot
print("\nMain plot (rank=100) has been saved as 'convergence_vs_beta_rank100.png'")

