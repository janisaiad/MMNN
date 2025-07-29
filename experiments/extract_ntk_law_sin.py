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
data_dir = os.path.join(BASE_PATH, 'data', 'ntk_infinite_sin_mmnn_2layer_MC_random_bigrun')
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
output_dir = os.path.join(BASE_PATH, "figures", "ntk_law_analysis_sin")
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
                            ntk_values.extend(samples[:, i, j].tolist())  # we convert to list
                            dot_products.extend([float(dots[i, j])] * len(samples))  # we ensure float
    
    if not ntk_values:
        return
    
    ntk_values = np.array(ntk_values)
    dot_products = np.array(dot_products)
    
    # we first collect all distributions by rho to optimize global bandwidth
    distributions_by_rho = {}
    for i, (rho_min, rho_max) in enumerate(zip(RHO_EDGES[:-1], RHO_EDGES[1:])):
        rho_center = RHO_CENTERS[i]
        mask = (dot_products >= rho_min) & (dot_products < rho_max)
        ntk_in_bin = ntk_values[mask]
        if len(ntk_in_bin) > 0:
            distributions_by_rho[rho_center] = {
                'values': ntk_in_bin,
                'mean': float(np.mean(ntk_in_bin)),  # we ensure float
                'std': float(np.std(ntk_in_bin))     # we ensure float
            }
    
    # we compute optimal bandwidth using Silverman's rule for each bin
    def compute_silverman_bandwidth(data):
        n = len(data)
        sigma = np.std(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        # we use min to be robust to outliers
        scale = min(sigma, iqr/1.34)
        return 0.9 * scale * n**(-0.2)

    # we compute average Silverman bandwidth across all bins
    all_bandwidths = []
    for rho in distributions_by_rho:
        values = distributions_by_rho[rho]['values']
        if len(values) > 1:
            all_bandwidths.append(compute_silverman_bandwidth(values))
    
    optimal_bw = float(np.mean(all_bandwidths))
    
    # we create figure for distributions by rho
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(4, 3)
    
    # we plot distributions for each rho bin with optimal bandwidth
    for i, rho in enumerate(sorted(distributions_by_rho.keys())):
        values = distributions_by_rho[rho]['values']
        emp_mean = float(np.mean(values))
        emp_std = float(np.std(values))
        
        plt.subplot(gs[i//3, i%3])
        
        # we plot normalized histogram
        plt.hist(values, bins=50, density=True, alpha=0.5, label='Empirical')
        
        if len(values) > 1:
            kde = KernelDensity(kernel=kernel, bandwidth=optimal_bw)
            kde.fit(values.reshape(-1, 1))
            
            # we extend grid for better visualization
            x_grid = np.linspace(min(values) - 3*emp_std, max(values) + 3*emp_std, 2000)
            log_dens = kde.score_samples(x_grid.reshape(-1, 1))
            dens = np.exp(log_dens)
            
            # we ensure proper normalization
            dx = x_grid[1] - x_grid[0]
            dens = dens / (np.sum(dens) * dx)
            
            # we compute moments for verification
            fitted_mean = float(np.sum(x_grid * dens) * dx)
            fitted_var = float(np.sum((x_grid - fitted_mean)**2 * dens) * dx)
            fitted_std = float(np.sqrt(fitted_var))
            
            plt.plot(x_grid, dens, 'r-', 
                    label=f'KDE fit (σ={fitted_std:.3f})')
        
        plt.title(f'ρ ≈ {rho:.2f}\nμ={emp_mean:.3f}, σ={emp_std:.3f}\nn={len(values)}')
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
    
    plt.title(f'NTK Mean vs ρ (rank={rank}, β={beta})\nGlobal bandwidth = {optimal_bw:.3f}')
    plt.xlabel('ρ')
    plt.ylabel('NTK Value')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(f'NTK Distributions by ρ (rank={rank}, β={beta})', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ntk_distributions_sin_rank{rank}_beta{beta}_{kernel}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # we print statistics
    print(f"\nStatistics for rank={rank}, β={beta} (bandwidth = {optimal_bw:.3f}):")
    print("-" * 120)
    print(f"{'ρ':^10} | {'Count':^8} | {'Emp Mean':^12} | {'Emp Std':^12} | {'KDE Std':^12} | {'Theory':^12} | {'Diff':^12}")
    print("-" * 120)
    for rho in sorted(distributions_by_rho.keys()):
        values = distributions_by_rho[rho]['values']
        emp_mean = float(np.mean(values))
        emp_std = float(np.std(values))
        
        # we compute KDE std for this bin
        kde = KernelDensity(kernel=kernel, bandwidth=optimal_bw)
        kde.fit(values.reshape(-1, 1))
        x_grid = np.linspace(min(values) - 3*emp_std, max(values) + 3*emp_std, 2000)
        log_dens = kde.score_samples(x_grid.reshape(-1, 1))
        dens = np.exp(log_dens)
        dx = x_grid[1] - x_grid[0]
        dens = dens / (np.sum(dens) * dx)
        
        fitted_mean = float(np.sum(x_grid * dens) * dx)
        fitted_var = float(np.sum((x_grid - fitted_mean)**2 * dens) * dx)
        fitted_std = float(np.sqrt(fitted_var))
        
        theo_val = ntk_formula(rho, beta)
        diff = emp_mean - theo_val
        print(f"{rho:^10.3f} | {len(values):^8d} | {emp_mean:^12.3f} | {emp_std:^12.3f} | "
              f"{fitted_std:^12.3f} | {theo_val:^12.3f} | {diff:^12.3f}")
    print("-" * 120)

    # we store min values for later analysis
    return {
        'beta': beta,
        'rank': rank,
        'min_values': {rho: np.min(distributions_by_rho[rho]['values']) for rho in distributions_by_rho}
    }

# we define Hill estimator function
def hill_estimator(data):
    """Computes the Hill estimator for the tail index alpha."""
    # we only use positive data
    data = data[data > 0]
    if len(data) < 2:
        return np.nan
    
    # we use top 10% of data for tail estimation
    k = int(len(data) * 0.1)
    if k < 2:
        k = len(data) // 2
    if k < 2:
        return np.nan
        
    sorted_data = np.sort(data)
    
    # we compute log of k largest values
    log_top_k = np.log(sorted_data[-k:])
    # we compute log of threshold value
    log_threshold = np.log(sorted_data[-k-1])
    
    # we compute 1/alpha
    inv_alpha = np.mean(log_top_k) - log_threshold
    
    return 1.0 / inv_alpha if inv_alpha > 0 else np.nan

def perform_full_analysis(rank, beta, kernel):
    """we perform full analysis including distributions and tail analysis."""
    
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
                            ntk_values.extend(samples[:, i, j].tolist())  # we convert to list
                            dot_products.extend([float(dots[i, j])] * len(samples))  # we ensure float
    
    if not ntk_values:
        return
    
    ntk_values = np.array(ntk_values)
    dot_products = np.array(dot_products)
    
    # we first collect all distributions by rho to optimize global bandwidth
    distributions_by_rho = {}
    for i, (rho_min, rho_max) in enumerate(zip(RHO_EDGES[:-1], RHO_EDGES[1:])):
        rho_center = RHO_CENTERS[i]
        mask = (dot_products >= rho_min) & (dot_products < rho_max)
        ntk_in_bin = ntk_values[mask]
        if len(ntk_in_bin) > 0:
            distributions_by_rho[rho_center] = {
                'values': ntk_in_bin,
                'mean': float(np.mean(ntk_in_bin)),  # we ensure float
                'std': float(np.std(ntk_in_bin))     # we ensure float
            }
    
    # we compute optimal bandwidth using Silverman's rule for each bin
    def compute_silverman_bandwidth(data):
        n = len(data)
        sigma = np.std(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        # we use min to be robust to outliers
        scale = min(sigma, iqr/1.34)
        return 0.9 * scale * n**(-0.2)

    # we compute average Silverman bandwidth across all bins
    all_bandwidths = []
    for rho in distributions_by_rho:
        values = distributions_by_rho[rho]['values']
        if len(values) > 1:
            all_bandwidths.append(compute_silverman_bandwidth(values))
    
    optimal_bw = float(np.mean(all_bandwidths))
    
    # --- Plot 1: NTK Distributions by Rho ---
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(4, 3)
    
    for i, rho in enumerate(sorted(distributions_by_rho.keys())):
        values = distributions_by_rho[rho]['values']
        
        # we truncate data at the 99th quantile
        q99 = np.percentile(values, 99)
        truncated_values = values[values <= q99]
        
        emp_mean = float(np.mean(truncated_values))
        emp_std = float(np.std(truncated_values))
        
        plt.subplot(gs[i//3, i%3])
        
        plt.hist(truncated_values, bins=50, density=True, alpha=0.5, label='Empirical')
        
        if len(truncated_values) > 1:
            kde = KernelDensity(kernel=kernel, bandwidth=optimal_bw)
            kde.fit(truncated_values.reshape(-1, 1))
            
            x_grid = np.linspace(min(truncated_values), max(truncated_values), 2000)
            log_dens = kde.score_samples(x_grid.reshape(-1, 1))
            dens = np.exp(log_dens)
            
            dx = x_grid[1] - x_grid[0]
            dens = dens / (np.sum(dens) * dx)
            
            fitted_mean = float(np.sum(x_grid * dens) * dx)
            fitted_var = float(np.sum((x_grid - fitted_mean)**2 * dens) * dx)
            fitted_std = float(np.sqrt(fitted_var))
            
            plt.plot(x_grid, dens, 'r-', 
                    label=f'KDE fit (σ={fitted_std:.3f})')
        
        plt.title(f'ρ ≈ {rho:.2f}\nμ={emp_mean:.3f}, σ={emp_std:.3f}\nn={len(truncated_values)}')
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
    
    plt.title(f'NTK Mean vs ρ (rank={rank}, β={beta})\nGlobal bandwidth = {optimal_bw:.3f}')
    plt.xlabel('ρ')
    plt.ylabel('NTK Value')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(f'NTK Distributions by ρ (rank={rank}, β={beta})', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ntk_distributions_sin_rank{rank}_beta{beta}_{kernel}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # --- Plot 2: Log-Log KDE and Tail Analysis ---
    fig2 = plt.figure(figsize=(20, 15))
    gs2 = plt.GridSpec(4, 3)
    
    # we plot log-log KDE for each rho bin
    for i, rho in enumerate(sorted(distributions_by_rho.keys())):
        values = distributions_by_rho[rho]['values']
        
        # we truncate data at the 99th quantile
        q99 = np.percentile(values, 99)
        truncated_values = values[values <= q99]
        
        if len(truncated_values) < 10:
            continue
            
        # we find the mode of the distribution to analyze the right tail
        kde_for_mode = KernelDensity(kernel=kernel, bandwidth=optimal_bw)
        kde_for_mode.fit(truncated_values.reshape(-1, 1))
        x_grid_mode = np.linspace(min(truncated_values), max(truncated_values), 2000)
        log_dens_mode = kde_for_mode.score_samples(x_grid_mode.reshape(-1, 1))
        mode = x_grid_mode[np.argmax(log_dens_mode)]
        
        # we filter for the right tail, starting from the mode
        tail_data = truncated_values[truncated_values > mode]
        
        # we only use positive data from the tail for log-log plots
        positive_tail_data = tail_data[tail_data > 0]
        
        if len(positive_tail_data) < 10:
            continue
            
        plt.subplot(gs2[i//3, i%3])
        
        # we fit KDE on the tail data
        kde_tail = KernelDensity(kernel=kernel, bandwidth=optimal_bw)
        kde_tail.fit(positive_tail_data.reshape(-1, 1))
        
        # we create log-spaced grid for plotting
        x_grid = np.logspace(np.log10(min(positive_tail_data)), np.log10(max(positive_tail_data)), 1000)
        log_dens = kde_tail.score_samples(x_grid.reshape(-1, 1))
        dens = np.exp(log_dens)
        
        # we plot KDE in log-log scale
        plt.loglog(x_grid, dens, 'r-', label='KDE of Tail')
        
        # we estimate slope from KDE tail
        log_x = np.log(x_grid)
        log_y = np.log(dens)
        valid_indices = np.isfinite(log_y)
        if np.sum(valid_indices) > 1:
            slope, _, _, _, _ = stats.linregress(log_x[valid_indices], log_y[valid_indices])
            alpha_kde = -slope - 1
        else:
            alpha_kde = np.nan
        
        # we estimate slope with Hill estimator on the raw tail data
        alpha_hill = hill_estimator(positive_tail_data)
        
        plt.title(f'ρ ≈ {rho:.2f}\nα_kde={alpha_kde:.2f}, α_hill={alpha_hill:.2f}')
        plt.xlabel('NTK Value (log)')
        plt.ylabel('Density (log)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
    
    plt.suptitle(f'Log-Log KDE and Tail Analysis (rank={rank}, β={beta})', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ntk_loglog_kde_sin_rank{rank}_beta{beta}_{kernel}.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig2)

# %%
# we analyze distributions and tails for each rank, beta, and kernel
for rank in ranks:
    for beta in betas:
        for kernel in KERNELS:
            perform_full_analysis(rank, beta, kernel)
            print(f"Full analysis complete for rank={rank}, β={beta}, kernel={kernel}")

# %%
# we analyze distributions and collect min values
min_value_data = []
for rank in ranks:
    for beta in betas:
        for kernel in KERNELS:
            result = analyze_ntk_distribution(rank, beta, kernel)
            if result is not None:
                min_value_data.append(result)

# we plot min values vs rank for each beta
plt.figure(figsize=(15, 10))

for beta in betas:
    beta_data = [d for d in min_value_data if d['beta'] == beta]
    if beta_data:
        ranks_for_beta = sorted([d['rank'] for d in beta_data])
        
        # we collect min values across all rhos for each rank
        min_vals = []
        for rank in ranks_for_beta:
            rank_data = next(d for d in beta_data if d['rank'] == rank)
            min_vals.append(min(rank_data['min_values'].values()))
        
        plt.subplot(121)
        plt.plot(ranks_for_beta, min_vals, 'o-', label=f'β={beta}')

        plt.subplot(122)
        plt.loglog(ranks_for_beta, min_vals, 'o-', label=f'β={beta}')

plt.subplot(121)
plt.title('Minimum NTK Value vs Rank')
plt.xlabel('Rank')
plt.ylabel('Min NTK Value')
plt.grid(True)
plt.legend()

plt.subplot(122)
plt.title('Minimum NTK Value vs Rank (log-log)')
plt.xlabel('Rank')
plt.ylabel('Min NTK Value')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'min_ntk_vs_rank_sin.png'), bbox_inches='tight', dpi=300)
plt.close()

# we print min value analysis
print("\nMinimum NTK Value Analysis:")
print("-" * 80)
print(f"{'β':^10} | {'Rank':^10} | {'Min Value':^15} | {'ρ at min':^15}")
print("-" * 80)
for data in min_value_data:
    beta = data['beta']
    rank = data['rank']
    min_val = min(data['min_values'].values())
    rho_at_min = min(data['min_values'].items(), key=lambda x: x[1])[0]
    print(f"{beta:^10.2f} | {rank:^10d} | {min_val:^15.3f} | {rho_at_min:^15.3f}")
print("-" * 80)

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
