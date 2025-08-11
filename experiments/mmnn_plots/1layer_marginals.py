import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy import stats

# i add project root to python path to allow imports from the 'model' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mmnn.mmnn_jax import MMNNJax

# --- configuration ---
num_nets = 1000  # i define the number of random networks to sample for the distributions
net_depth = 1  # i want a 1-layer mmnn
net_width = 4096  # i set the width of the hidden layer
ranks = [1, 1]  # i define the ranks for the mmnn layers
widths = [net_width] * net_depth

# i define the specific input points for which to plot the marginal distributions
x_points = jnp.array([0.1, 0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)

# --- model and data generation ---
print(f"generating outputs from {num_nets} networks for {len(x_points)} points...")

# i initialize the jax model
model_jax = MMNNJax(ranks=ranks, widths=widths, resnet=False, fix_wb=False)

# we jit the apply function for better performance
@jax.jit
def apply_model(params, x):
    return model_jax.apply({'params': params}, x)

key = jax.random.PRNGKey(0)
outputs = []

for i in range(num_nets):
    if (i + 1) % 1000 == 0:
        print(f"  ... network {i + 1}/{num_nets}")
    key, subkey = jax.random.split(key)
    # we initialize parameters for each network instance
    params = model_jax.init(subkey, x_points)['params']
    
    # we compute the model's output for the specified points
    y_output = apply_model(params, x_points)
    outputs.append(np.array(y_output).flatten())

# this results in a matrix of shape (num_nets, num_points)
outputs_matrix = np.array(outputs)

# --- visualization ---
print("plotting marginal distributions...")
fig, axes = plt.subplots(1, len(x_points), figsize=(20, 4), sharey=False)

for i in range(len(x_points)):
    ax = axes[i]
    point_outputs = outputs_matrix[:, i]

    # i use the freedman-diaconis rule to determine the optimal number of bins.
    # this method is robust to outliers.
    iqr = stats.iqr(point_outputs, rng=(25, 75))
    if iqr > 0:
        bin_width = 2 * iqr / (len(point_outputs)**(1/3))
        num_bins = int((point_outputs.max() - point_outputs.min()) / bin_width)
        num_bins = min(num_bins, 150) # we cap at 150 bins to avoid noisy histograms
    else:
        num_bins = 30 # we fallback for distributions with no variance

    ax.hist(point_outputs, bins=num_bins, density=True, alpha=0.75, label='Histogram')
    
    # we overlay a kernel density estimate (kde) for a smoother plot
    kde = stats.gaussian_kde(point_outputs)
    x_range = np.linspace(point_outputs.min(), point_outputs.max(), 500)
    ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')

    ax.set_title(f"x = {x_points[i].item():.1f}")
    ax.set_xlabel("model output")
    if i == 0:
        ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

fig.suptitle(f"marginal distributions for a 1-layer mmnn (width={net_width}, activation=relu)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])


# --- saving the figure ---
output_dir = 'figures/mmnn_plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '1layer_marginals.png')
plt.savefig(output_path)
print(f"plot saved to {output_path}")

plt.show()

