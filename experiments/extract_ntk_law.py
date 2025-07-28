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
import jax.numpy as jnp
from jax import random
from scipy.stats import norm, multivariate_normal

# we import theoretical formulas
def get_rho_eff(rho, beta, norm_x=1.0, norm_x_prime=1.0):
    numerator = rho + beta**2
    denominator = jnp.sqrt((norm_x**2 + beta**2) * (norm_x_prime**2 + beta**2))
    return numerator / denominator

def K1(rho_eff, beta, norm_x=1.0, norm_x_prime=1.0):
    rho_eff = jnp.clip(rho_eff, -1.0, 1.0)
    variance_product = jnp.sqrt((norm_x**2 + beta**2) * (norm_x_prime**2 + beta**2))
    term_in_parentheses = jnp.sqrt(1 - rho_eff**2) + (jnp.pi - jnp.arccos(rho_eff)) * rho_eff
    return variance_product * (1 / (2 * jnp.pi)) * term_in_parentheses

def K_dot(rho_eff):
    rho_eff = jnp.clip(rho_eff, -1.0, 1.0)
    return (1 / (2 * jnp.pi)) * (jnp.pi - jnp.arccos(rho_eff))

def ntk_formula(rho, beta, norm_x=1.0, norm_x_prime=1.0):
    rho_eff = get_rho_eff(rho, beta, norm_x, norm_x_prime)
    k1_val = K1(rho_eff, beta, norm_x, norm_x_prime)
    k_dot_val = K_dot(rho_eff)
    return k1_val + (rho + beta**2) * k_dot_val

# we define base path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# we define kernels to test
KERNELS = ['gaussian', 'epanechnikov', 'exponential', 'cosine']
BANDWIDTHS = [0.1, 0.2, 0.5]

# we define rho bins (10 bins between -1 and 1)
N_RHO_BINS = 10
RHO_EDGES = np.linspace(-1, 1, N_RHO_BINS + 1)
RHO_CENTERS = (RHO_EDGES[:-1] + RHO_EDGES[1:]) / 2

# %%
# we load and combine individual results
data_dir = os.path.join(BASE_PATH, 'data', 'ntk_infinite_relu_mmnn_2layer_MC_random_bigrun')
results = {}

# we get all unique parameters from filenames
files = os.listdir(data_dir)
param_sets = set()
for f in files:
    if f.startswith('eigenvals_'):
        params = '_'.join(f.split('_')[1:4])  # we get 'dim2_n4_beta0.0'
        param_sets.add(params)

# we load data for each parameter set
for params in param_sets:
    key = params
    results[key] = {
        'ntk_samples': {}
    }
    
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
    """we analyze NTK distribution for specific rho bins and compare with theory"""
    
    # we collect NTK values and corresponding dot products
    ntk_values = []
    dot_products = []
    
    for n in n_samples_list:
        for d in input_dims:
            key = f"dim{d}_n{n}_beta{beta}"
            if key in results and rank in results[key]['ntk_samples']:
                samples = results[key]['ntk_samples'][rank]
                if samples is not None:
                    # we generate input vectors
                    key_rng = random.PRNGKey(41)
                    X = random.normal(key_rng, (n, d))
                    X = X / jnp.linalg.norm(X, axis=1, keepdims=True)
                    
                    # we compute dot products
                    dots = jnp.dot(X, X.T)
                    
                    # we collect values
                    for i in range(n):
                        for j in range(n):
                            ntk_values.extend(samples[:, i, j])
                            dot_products.extend([dots[i, j]] * len(samples))
    
    if not ntk_values:
        return
    
    ntk_values = np.array(ntk_values)
    dot_products = np.array(dot_products)
    
    # we create figure for distributions by rho and theory comparison
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(4, 3)
    
    # we plot distributions for each rho bin
    distributions_by_rho = {}
    for i, (rho_min, rho_max) in enumerate(zip(RHO_EDGES[:-1], RHO_EDGES[1:])):
        rho_center = RHO_CENTERS[i]
        
        # we get values in this rho bin
        mask = (dot_products >= rho_min) & (dot_products < rho_max)
        ntk_in_bin = ntk_values[mask]
        
        if len(ntk_in_bin) > 0:
            distributions_by_rho[rho_center] = {
                'values': ntk_in_bin,
                'mean': np.mean(ntk_in_bin),
                'std': np.std(ntk_in_bin)
            }
            
            # we plot distribution for this rho bin
            plt.subplot(gs[i//3, i%3])
            
            # we plot histogram
            plt.hist(ntk_in_bin, bins=50, density=True, alpha=0.5, label='Empirical')
            
            # we fit and plot KDE
            if len(ntk_in_bin) > 1:
                kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
                kde.fit(ntk_in_bin.reshape(-1, 1))
                x_grid = np.linspace(min(ntk_in_bin), max(ntk_in_bin), 1000).reshape(-1, 1)
                log_dens = kde.score_samples(x_grid)
                plt.plot(x_grid, np.exp(log_dens), 'r-', label=f'KDE ({kernel})')
            
            # we compute theoretical value
            theo_val = ntk_formula(rho_center, beta)
            plt.axvline(theo_val, color='g', linestyle='--', 
                       label=f'Theory: {theo_val:.3f}')
            
            plt.title(f'ρ ≈ {rho_center:.2f}\nμ={np.mean(ntk_in_bin):.3f}, σ={np.std(ntk_in_bin):.3f}')
            plt.xlabel('NTK Value')
            plt.ylabel('Density')
            plt.grid(True)
            if i == 0:
                plt.legend()
    
    # we plot theory vs empirical means
    ax_theory = plt.subplot(gs[3, :])
    
    # we sort by rho for plotting
    rhos = sorted(distributions_by_rho.keys())
    means = [distributions_by_rho[r]['mean'] for r in rhos]
    stds = [distributions_by_rho[r]['std'] for r in rhos]
    
    # we plot empirical means with error bars
    plt.errorbar(rhos, means, yerr=stds, fmt='o', label='Empirical (mean ± std)', 
                capsize=5, color='blue', alpha=0.6)
    
    # we plot theoretical curve
    rho_fine = np.linspace(-1, 1, 100)
    theo_values = [ntk_formula(r, beta) for r in rho_fine]
    plt.plot(rho_fine, theo_values, 'r-', label='Theoretical', alpha=0.8)
    
    plt.title(f'NTK Mean vs ρ (rank={rank}, β={beta})')
    plt.xlabel('ρ')
    plt.ylabel('NTK Value')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(f'NTK Distributions by ρ with Theory Comparison\n(rank={rank}, β={beta})', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ntk_distributions_theory_rank{rank}_beta{beta}_{kernel}_bw{bandwidth}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # we print statistics and comparison with theory
    print(f"\nStatistics and Theory Comparison for rank={rank}, β={beta}:")
    print("-" * 100)
    print(f"{'ρ':^10} | {'Count':^8} | {'Emp Mean':^12} | {'Emp Std':^12} | {'Theory':^12} | {'Diff':^12}")
    print("-" * 100)
    for rho in sorted(distributions_by_rho.keys()):
        values = distributions_by_rho[rho]['values']
        emp_mean = np.mean(values)
        emp_std = np.std(values)
        theo_val = ntk_formula(rho, beta)
        diff = emp_mean - theo_val
        print(f"{rho:^10.3f} | {len(values):^8d} | {emp_mean:^12.3f} | {emp_std:^12.3f} | "
              f"{theo_val:^12.3f} | {diff:^12.3f}")
    print("-" * 100)

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
