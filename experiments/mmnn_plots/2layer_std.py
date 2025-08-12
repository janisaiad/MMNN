import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from ntk.ntk_infinite import compute_ntk_2layer_montecarlo_random_field, relu, relu_dot
from tqdm import tqdm
import os
import pathlib
import numpy as np
from scipy.stats import linregress

# we create output directory
pathlib.Path("figures/2layer_std_analysis").mkdir(parents=True, exist_ok=True)

# we set up base parameters
key = random.PRNGKey(42)
n_samples = 2  # we fix sample size for the input data
input_dim = 2 # we fix input dimension
ranks_to_test = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # we test different ranks[0] values
beta = 0.0  # we fix beta value
n_mc_samples = 100  # we fix number of Monte Carlo samples for NTK computation
n_ntk_samples = 50 # we number of NTK samples to compute std deviation

# we test one sigma_A and sigma_c combination
sigma_A = jnp.sqrt(2)
sigma_c = 1.0

# we store results
stds = []

# we create a subdirectory for distribution plots
dist_plot_dir = f"figures/2layer_std_analysis/distributions_beta{beta}_n{n_samples}_d{input_dim}"
pathlib.Path(dist_plot_dir).mkdir(parents=True, exist_ok=True)


# we generate test data once
X = random.normal(key, (n_samples, input_dim))
X = X / jnp.linalg.norm(X, axis=1, keepdims=True)  # we normalize to unit sphere

with tqdm(total=len(ranks_to_test), desc="Computing NTK std dev vs rank") as pbar:
    for rank in ranks_to_test:
        ntk_samples = []
        for i in tqdm(range(n_ntk_samples), desc=f"Rank {rank} NTK samples", leave=False):
            # we compute one NTK sample
            ntk = compute_ntk_2layer_montecarlo_random_field(
                X,
                ranks=[rank,1],
                sigma_A=sigma_A,
                sigma_c=sigma_c,
                beta=beta,
                activation_fn=relu,
                activation_dot_fn=relu_dot,
                key=random.split(key)[1]+i,
                n_samples=n_mc_samples
            )
            if not jnp.isnan(ntk).any():
                ntk_samples.append(ntk)

        if ntk_samples:
            # we compute std of diagonal elements
            ntk_samples_array = jnp.array(ntk_samples)
            diag_samples = ntk_samples_array[:, 0, 0]
            std_diag = jnp.std(diag_samples)
            mean_diag = jnp.mean(diag_samples)
            stds.append(std_diag)

            # we plot the ntk distribution
            plt.figure(figsize=(8, 6))
            plt.hist(diag_samples, bins='auto', density=True)
            plt.axvline(mean_diag, color='r', linestyle='--', label=f'Mean: {mean_diag:.3f}')
            plt.title(f'NTK[0,0] Distribution (rank={rank})')
            plt.xlabel("NTK[0,0] Value")
            plt.ylabel("Density")
            plt.text(0.05, 0.95, f'Std: {std_diag:.2e}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.legend()
            
            dist_plot_filename = f"{dist_plot_dir}/ntk_dist_rank{rank}.png"
            plt.savefig(dist_plot_filename)
            plt.close()
        else:
            stds.append(np.nan) # we append nan if all computations failed

        pbar.update(1)

# we filter out nan values for the fit
ranks_for_fit = np.array([r for r, s in zip(ranks_to_test, stds) if s and not np.isnan(s) and s > 0])
stds_for_fit = np.array([s for s in stds if s and not np.isnan(s) and s > 0])

# we fit a slope to the log-log plot
if len(ranks_for_fit) > 1:
    log_ranks = np.log(ranks_for_fit)
    log_stds = np.log(stds_for_fit)
    slope, intercept, _, _, _ = linregress(log_ranks, log_stds)
    fit_label = f'Fit: slope={slope:.2f}'
else:
    slope = np.nan
    intercept = np.nan
    fit_label = "Not enough data to fit slope"

# we plot the results
plt.figure(figsize=(10, 6))
plt.loglog(ranks_to_test, stds, 'o-', label=f'beta={beta}, n_samples={n_samples}, d={input_dim}')

if not np.isnan(slope):
    # we plot the fitted line
    fit_line = np.exp(intercept) * (ranks_to_test ** slope)
    plt.loglog(ranks_to_test, fit_line, '--', label=fit_label)

plt.xlabel('Rank (log scale)')
plt.ylabel('Std Dev of NTK[0,0] (log scale)')
plt.title('NTK Diagonal Stdev vs. Rank for 2-Layer MMNN')
plt.grid(True)
plt.legend()
plt.tight_layout()

# we save the plot
output_filename = f"figures/2layer_std_analysis/std_vs_rank_beta{beta}_n{n_samples}_d{input_dim}.png"
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")

plt.show()
