
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.stats import norm, multivariate_normal
from tqdm import tqdm
# i add project root to python path to allow imports from `model` module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ntk.ntk_infinite import compute_ntk_1layer, relu as jax_relu

# --- theoretical formula functions (mlp) ---
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

# --- experimental formula functions ---
def M00(s1, s2, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    return multivariate_normal.cdf([-s1, -s2], mean=mean, cov=cov, allow_singular=True)

def M10(s1, s2, rho):
    eps = 1e-8
    denominator = jnp.sqrt(1 - rho**2)
    arg_phi = (rho * s1 - s2) / jnp.maximum(denominator, eps)
    return norm.pdf(s1) * norm.cdf(arg_phi)

def M01(s1, s2, rho):
    return M10(s2, s1, rho)

def M11(s1, s2, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    term1 = rho * multivariate_normal.pdf([s1, s2], mean=mean, cov=cov, allow_singular=True)
    term2 = s1 * M10(s1, s2, rho)
    term3 = s2 * M01(s1, s2, rho)
    term4 = M00(s1, s2, rho)
    return term1 + term2 + term3 + term4

def inner_expectation(mu, sigma_1, sigma_2, rho):
    """
    we compute e[relu(x_1+mu)relu(x_2+mu)] for (x_1, x_2) bivariate normal with mean 0.
    this is equivalent to e[relu(y_1)relu(y_2)] for y~n((mu,mu), diag(sigma_1,sigma_2)*rho*diag(sigma_1,sigma_2))
    """
    eps = 1e-8
    sigma_1_safe = jnp.maximum(sigma_1, eps)
    sigma_2_safe = jnp.maximum(sigma_2, eps)

    s1 = -mu / sigma_1_safe
    s2 = -mu / sigma_2_safe
    
    rho_safe = jnp.clip(rho, -1.0 + eps, 1.0 - eps)

    m00 = M00(s1, s2, rho_safe)
    m10 = M10(s1, s2, rho_safe)
    m01 = M01(s1, s2, rho_safe)
    m11 = M11(s1, s2, rho_safe)

    # I(beta*b, rho) = sigma1*sigma2*M11 + mu*sigma2*M01 + mu*sigma1*M10 + mu^2*M00
    # where mu = beta*b
    
    res = (sigma_1 * sigma_2 * m11 +
           mu * sigma_2 * m01 +
           mu * sigma_1 * m10 +
           mu**2 * m00)
           
    return res

def calculate_total_expectation_riemann(beta, rho, n_points=50, sigma_1=1.0, sigma_2=1.0):
    b_points = jnp.linspace(-5, 5, n_points)
    db = b_points[1] - b_points[0]
    gaussian_weights = norm.pdf(b_points)
    
    mu_b = b_points * beta
    expectations = jnp.array([inner_expectation(mu, sigma_1, sigma_2, rho) for mu in mu_b])
    
    return jnp.sum(expectations * gaussian_weights * db)

# --- parameters ---
net_width = 4096  # i set the width of the hidden layer, a large width allows for more complexity
x_domain = jnp.linspace(-1, 1, 100)  # i define the input range for our 1d function
betas = [0.0, 0.2, 1.0]  # i define the values of beta to test
ntk_results = {}  # i initialize a dictionary to store the results
ntk_formula_results = {} # i initialize a dictionary for the theoretical formula results
experimental_formula_results = {} # i initialize a dictionary for the experimental formula results

key = jax.random.PRNGKey(0)

# --- ntk calculation ---
print("calculating ntk values for different betas...")
for beta in tqdm(betas):
    print(f"  ... beta = {beta}")
    ntk_values_for_beta = []
    ntk_formula_values_for_beta = []
    experimental_values_for_beta = []
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
        rho = x_val
        norm_x = jnp.abs(x_val)
        norm_x_prime = 1.0
        formula_value = ntk_formula(rho, beta, norm_x, norm_x_prime)
        ntk_formula_values_for_beta.append(formula_value)
        
        # i compute the experimental formula value
        experimental_value = calculate_total_expectation_riemann(beta, rho, sigma_1=norm_x, sigma_2=norm_x_prime)
        experimental_values_for_beta.append(experimental_value)
    
    ntk_results[beta] = np.array(ntk_values_for_beta)
    ntk_formula_results[beta] = np.array(ntk_formula_values_for_beta)
    experimental_formula_results[beta] = np.array(experimental_values_for_beta)

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
    
    # i plot the experimental formula ntk
    plt.plot(x_domain, experimental_formula_results[beta], color=colors[i], linestyle=':', label=f'Empirical Formula beta = {beta}')

plt.title(f'1-layer MMNN vs MLP NTK value of ntk(x, 1) for various beta (width={net_width})')
plt.xlabel("x")
plt.ylabel("ntk(x, 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- save and show plot ---
output_dir = 'figures/mmnn_plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '1layer_ntk_beta_comparison_with_empirical.png')
plt.savefig(output_path)
print(f"plot saved to {output_path}")

plt.show()