import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# i add project root to python path to allow imports from `model` module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ntk.ntk_infinite import compute_nngp_1layer, relu as jax_relu
from mmnn.mmnn_jax import MMNNJax
import jax
import jax.numpy as jnp

num_nets_empirical = 5000 # i define the number of networks for the empirical covariance calculation
net_depth = 1 # i want a 1-layer mmnn
net_width = 4096 # i set the width of the hidden layer, a large width allows for more complexity

ranks = [1,1] # i define the ranks for the mmnn layers
widths = [net_width] * net_depth # i define the widths for the mmnn layers

# --- empirical covariance calculation ---
print(f"calculating empirical covariance with {num_nets_empirical} networks...")
outputs = []
# i define the input range for our 1d function, smaller for covariance matrix visualization
x_domain_jax = jnp.linspace(-jnp.pi, jnp.pi, 100).reshape(-1, 1)

key = jax.random.PRNGKey(0)

# i initialize the jax model as defined in mmnn_jax.py
model_jax = MMNNJax(ranks=ranks, widths=widths, resnet=False, fix_wb=False)

# we jit the apply function for speed
@jax.jit
def apply_model(params, x):
    return model_jax.apply({'params': params}, x)

for i in range(num_nets_empirical):
    if (i+1) % 500 == 0:
        print(f"  ... network {i+1}/{num_nets_empirical}")
    key, subkey = jax.random.split(key)
    # we initialize parameters for each network instance
    params = model_jax.init(subkey, x_domain_jax)['params']
    
    # we compute the model's output
    y_output = apply_model(params, x_domain_jax)
    outputs.append(np.array(y_output).flatten())

outputs_matrix = np.array(outputs) # we create a matrix of shape (num_nets, num_points)
# we calculate the covariance. `rowvar=false` indicates that columns are variables (points in x_domain)
# and rows are observations (different networks).
empirical_cov = np.cov(outputs_matrix, rowvar=False)

# --- theoretical covariance calculation ---
print("calculating theoretical nngp kernel...")
# i use the imported function to compute the theoretical nngp kernel (covariance matrix).
key, subkey = jax.random.split(key)
nngp_kernel = compute_nngp_1layer(
    X=x_domain_jax,
    ranks=[net_width],
    activation_fn=jax_relu,
    key=subkey,
    sigma_A=1.0,
    sigma_c=1.0,
    beta=0.0,
    n_samples=10000
)
theoretical_cov = np.array(nngp_kernel)

# --- visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# i plot the theoretical covariance matrix
im1 = axes[0].imshow(theoretical_cov, cmap='viridis')
axes[0].set_title("theoretical covariance (nngp)")
fig.colorbar(im1, ax=axes[0])

# i plot the empirical covariance matrix
im2 = axes[1].imshow(empirical_cov, cmap='viridis')
axes[1].set_title(f"empirical covariance ({num_nets_empirical} nets)")
fig.colorbar(im2, ax=axes[1])

# i plot the difference
diff = theoretical_cov - empirical_cov
vmax = np.abs(diff).max()
im3 = axes[2].imshow(diff, cmap='coolwarm', vmin=-vmax, vmax=vmax)
axes[2].set_title("difference (theoretical - empirical)")
fig.colorbar(im3, ax=axes[2])

fig.suptitle(f"nngp kernel comparison for 1-layer mmnn (width={net_width}, activation=relu)")
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_dir = 'figures/mmnn_plots' # i define the output directory
os.makedirs(output_dir, exist_ok=True) # i create the output directory if it doesn't exist
output_path = os.path.join(output_dir, 'nngp_comparison_jax.png') # i define the output file path
plt.savefig(output_path) # i save the figure
print(f"plot saved to {output_path}")

plt.show() # i display the plot
