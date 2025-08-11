
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# i add project root to python path to allow imports from `model` module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ntk.ntk_infinite import compute_ntk_1layer, relu as jax_relu

# --- theoretical formula functions ---
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

# --- parameters ---
net_width = 4096  # i set the width of the hidden layer, a large width allows for more complexity
x_domain = jnp.linspace(-1, 1, 100)  # i define the input range for our 1d function
betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # i define the values of beta to test
ntk_results = {}  # i initialize a dictionary to store the results
ntk_formula_results = {} # i initialize a dictionary for the theoretical formula results

key = jax.random.PRNGKey(0)

# --- ntk calculation ---
print("calculating ntk values for different betas...")
for beta in betas:
    print(f"  ... beta = {beta}")
    ntk_values_for_beta = []
    ntk_formula_values_for_beta = []
    for x_val in x_domain:
        # i create the input matrix for a single x_val and the constant 1
        X_input = jnp.array([[x_val], [1.0]])
        
        # i compute the 2x2 ntk matrix from simulation
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
        
        # i compute the theoretical ntk value
        # rho is the dot product of x and 1, which is x_val. norm_x is |x_val|, norm_x_prime is 1.
        rho = x_val
        norm_x = jnp.abs(x_val)
        norm_x_prime = 1.0
        formula_value = ntk_formula(rho, beta, norm_x, norm_x_prime)
        ntk_formula_values_for_beta.append(formula_value)
    
    ntk_results[beta] = np.array(ntk_values_for_beta)
    ntk_formula_results[beta] = np.array(ntk_formula_values_for_beta)

# --- visualization ---
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))

for i, beta in enumerate(betas):
    # i plot the simulated mmnn ntk
    plt.plot(x_domain, ntk_results[beta], color=colors[i], linestyle='-', label=f'MMNN beta = {beta}')
    
    # i determine the linestyle for the mlp formula
    if beta in [0.0, 0.5, 1.0]:
        linestyle = '--'
    else:
        linestyle = '-.'
        
    # i plot the theoretical mlp ntk
    plt.plot(x_domain, ntk_formula_results[beta], color=colors[i], linestyle=linestyle, label=f'MLP Formula beta = {beta}')

plt.title(f'1-layer MMNN vs MLP NTK value of ntk(x, 1) for various beta (width={net_width})')
plt.xlabel("x")
plt.ylabel("ntk(x, 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- save and show plot ---
output_dir = 'figures/mmnn_plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '1layer_ntk_beta_comparison.png')
plt.savefig(output_path)
print(f"plot saved to {output_path}")

plt.show()