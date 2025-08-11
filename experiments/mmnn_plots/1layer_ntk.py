
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# i add project root to python path to allow imports from `model` module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ntk.ntk_infinite import compute_ntk_1layer, relu as jax_relu

# --- parameters ---
net_width = 4096  # i set the width of the hidden layer, a large width allows for more complexity
x_domain = jnp.linspace(-1, 1, 100)  # i define the input range for our 1d function
betas = [0.0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]  # i define the values of beta to test
ntk_results = {}  # i initialize a dictionary to store the results

key = jax.random.PRNGKey(0)

# --- ntk calculation ---
print("calculating ntk values for different betas...")
for beta in betas:
    print(f"  ... beta = {beta}")
    ntk_values_for_beta = []
    for x_val in x_domain:
        # i create the input matrix for a single x_val and the constant 1
        X_input = jnp.array([[x_val], [1.0]])
        
        # i compute the 2x2 ntk matrix
        key, subkey = jax.random.split(key)
        ntk_matrix = compute_ntk_1layer(
            X=X_input,
            ranks=[net_width],
            activation_fn=jax_relu,
            key=subkey,
            beta=beta,
            sigma_A=1.0,
            sigma_c=1.0,
            n_samples=10000
        )
        # i extract the off-diagonal element which corresponds to ntk(x, 1)
        ntk_value = ntk_matrix[0, 1]
        ntk_values_for_beta.append(ntk_value)
    
    ntk_results[beta] = np.array(ntk_values_for_beta)

# --- visualization ---
plt.figure(figsize=(10, 6))

for beta, ntk_values in ntk_results.items():
    plt.plot(x_domain, ntk_values, label=f'beta = {beta}')

plt.title(f'1-layer mmnn ntk value of ntk(x, 1) for various beta (width={net_width})')
plt.xlabel("x")
plt.ylabel("ntk(x, 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- save and show plot ---
output_dir = 'figures/mmnn_plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '1layer_ntk_beta_variation.png')
plt.savefig(output_path)
print(f"plot saved to {output_path}")

plt.show()