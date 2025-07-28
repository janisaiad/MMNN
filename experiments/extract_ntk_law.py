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
from scipy import stats
import os
from sklearn.neighbors import KernelDensity

# we define base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# we define kernels to test
KERNELS = ['gaussian', 'epanechnikov', 'exponential', 'cosine']
BANDWIDTHS = [0.1, 0.2, 0.5]  # we test different bandwidths

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
        'ntk_samples': {}
    }
    
    # we find all files for this parameter set
    for f in files:
        if params in f and f.startswith('ntk_samples_'):
            rank = int(f.split('_rank')[-1].split('.')[0])
            samples = np.load(os.path.join(data_dir, f))
            results[key]['ntk_samples'][rank] = samples

# we extract parameters
input_dims = sorted(list(set([int(k.split('_')[0].replace('dim','')) for k in results.keys()])))
n_samples_list = sorted(list(set([int(k.split('_')[1].replace('n','')) for k in results.keys()])))
betas = sorted(list(set([float(k.split('_')[2].replace('beta','')) for k in results.keys()])))
ranks = sorted(list(set([int(r) for k in results.keys() for r in results[k]['ntk_samples'].keys()])))

# we create output directories
output_dir = os.path.join(BASE_PATH, "figures", "ntk_law_analysis")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
def analyze_ntk_distribution(rank, beta, kernel='gaussian', bandwidth=0.2):
    """we analyze NTK distribution for a specific rank and beta"""
    plt.figure(figsize=(15, 5))
    
    # we collect all NTK values for this rank and beta
    diagonal_values = []
    offdiagonal_values = []
    all_values = []
    
    for n in n_samples_list:
        for d in input_dims:
            key = f"dim{d}_n{n}_beta{beta}"
            if key in results and rank in results[key]['ntk_samples']:
                samples = results[key]['ntk_samples'][rank]
                if samples is not None:
                    # we collect diagonal values
                    diag = np.diagonal(samples, axis1=1, axis2=2).flatten()
                    diagonal_values.extend(diag)
                    
                    # we collect off-diagonal values
                    off_diag = samples.reshape(-1, samples.shape[-1]**2)
                    off_diag = off_diag[:, ~np.eye(samples.shape[-1], dtype=bool)].flatten()
                    offdiagonal_values.extend(off_diag)
                    
                    # we collect all values
                    all_values.extend(samples.flatten())
    
    if not all_values:  # we skip if no data
        return
    
    # we convert to arrays
    diagonal_values = np.array(diagonal_values)
    offdiagonal_values = np.array(offdiagonal_values)
    all_values = np.array(all_values)
    
    # we fit KDE for each type of values
    for i, (values, title) in enumerate([
        (diagonal_values, 'Diagonal Values'),
        (offdiagonal_values, 'Off-diagonal Values'),
        (all_values, 'All Values')
    ]):
        plt.subplot(1, 3, i+1)
        
        # we compute basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # we fit KDE
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        values_reshaped = values.reshape(-1, 1)
        kde.fit(values_reshaped)
        
        # we evaluate KDE on a grid
        x_grid = np.linspace(min(values), max(values), 1000).reshape(-1, 1)
        log_dens = kde.score_samples(x_grid)
        
        # we plot histogram and KDE
        plt.hist(values, bins=50, density=True, alpha=0.5, label='Histogram')
        plt.plot(x_grid, np.exp(log_dens), 'r-', label=f'KDE ({kernel})')
        
        # we plot normal distribution for comparison
        x = np.linspace(min(values), max(values), 1000)
        plt.plot(x, stats.norm.pdf(x, mean_val, std_val), 'g--', label='Normal')
        
        plt.title(f'{title}\nrank={rank}, β={beta}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ntk_distribution_rank{rank}_beta{beta}_{kernel}_bw{bandwidth}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

# %%
# we analyze distributions for each rank and beta
for rank in ranks:
    for beta in betas:
        for kernel in KERNELS:
            for bandwidth in BANDWIDTHS:
                analyze_ntk_distribution(rank, beta, kernel, bandwidth)
                print(f"Analyzed distribution for rank={rank}, β={beta}, kernel={kernel}, bandwidth={bandwidth}")

# we print completion message
print("\nAnalysis complete. Results saved in 'figures/ntk_law_analysis/'")
