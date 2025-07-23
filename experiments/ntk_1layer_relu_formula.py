# %%
# we analyze the NTK which is rotation invariant, so we first compare the functional dependence in rho (correlation or dot product)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from jax import random
from tqdm import tqdm

# %%

# we define the NTK formula functions
def get_rho_eff(rho, beta, norm_x=1.0, norm_x_prime=1.0):
    numerator = rho + beta**2  # we compute numerator
    denominator = jnp.sqrt((norm_x**2 + beta**2) * (norm_x_prime**2 + beta**2))  # we compute denominator
    return numerator / denominator

def K1(rho_eff, beta, norm_x=1.0, norm_x_prime=1.0):
    rho_eff = jnp.clip(rho_eff, -1.0, 1.0)  # we avoid domain errors for arccos
    variance_product = jnp.sqrt((norm_x**2 + beta**2) * (norm_x_prime**2 + beta**2))
    term_in_parentheses = jnp.sqrt(1 - rho_eff**2) + (jnp.pi - jnp.arccos(rho_eff)) * rho_eff
    return variance_product * (1 / (2 * jnp.pi)) * term_in_parentheses

def K_dot(rho_eff):
    rho_eff = jnp.clip(rho_eff, -1.0, 1.0)  # we avoid domain errors for arccos
    return (1 / (2 * jnp.pi)) * (jnp.pi - jnp.arccos(rho_eff))

def ntk_formula(rho, beta, norm_x=1.0, norm_x_prime=1.0):
    rho_eff = get_rho_eff(rho, beta, norm_x, norm_x_prime)  # we compute effective correlation
    k1_val = K1(rho_eff, beta, norm_x, norm_x_prime)  # we compute K1 term
    k_dot_val = K_dot(rho_eff)  # we compute K_dot term
    return k1_val + (rho + beta**2) * k_dot_val



# %%
def I_0(mu, rho):
    """
    Computes I_0 = P(Z > -mu, Z' > -mu) for a bivariate normal distribution
    with correlation rho.
    By symmetry, equals P(Z < mu, Z' < mu).
    """
    rho = jnp.clip(rho, -0.9999, 0.9999)  # we handle edge cases
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    # we use scipy's multivariate normal since jax doesn't have cdf
    return multivariate_normal.cdf([mu, mu], mean=mean, cov=cov)

def I_1(mu, rho):
    """
    Computes I_1 = (1+rho) * phi(-mu) * Phi(-mu * sqrt((1-rho)/(1+rho)))
    """
    rho = jnp.clip(rho, -0.9999, 0.9999)  # we handle edge cases
    a = -mu
    factor = jnp.sqrt((1 - rho) / (1 + rho))
    # we use scipy's normal distribution
    return (1 + rho) * norm.pdf(a) * norm.cdf(a * factor)

def I_2(mu, rho):
    """
    Computes I_2 = rho * I_0 + (1-rho^2) * phi(-mu, -mu; rho)
    """
    rho = jnp.clip(rho, -0.9999, 0.9999)  # we handle edge cases
    a = -mu
    i0_val = I_0(mu, rho)
    # we use scipy's multivariate normal
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    pdf_val = multivariate_normal.pdf([a, a], mean=mean, cov=cov)
    return rho * i0_val + (1 - rho**2) * pdf_val

def inner_expectation(mu, rho):
    """
    Computes inner expectation I(mu, rho) = I_2 + 2*mu*I_1 + mu^2*I_0
    """
    sigma = 1.0  # we assume sigma=1 since Q1, Q2 are standard
    i0 = I_0(mu, rho)
    i1 = I_1(mu, rho)
    i2 = I_2(mu, rho)
    return sigma**2 * i2 + 2 * mu * sigma * i1 + mu**2 * i0

def two_layer_ntk(rho):
    """
    Computes the 2-layer NTK formula K(ρ) = (1/2π)(sqrt(1-ρ^2) + (π-arccos(ρ))ρ) + ρ*(1/2π)(π-arccos(ρ))
    """
    term1 = jnp.sqrt(1 - rho**2)
    term2 = (jnp.pi - jnp.arccos(rho)) * rho
    term3 = rho * (jnp.pi - jnp.arccos(rho))
    return (1/(2*jnp.pi)) * (term1 + term2) + (1/(2*jnp.pi)) * term3


# %%
def calculate_total_expectation_riemann(beta, rho, n_points=1000):
    """
    Computes total expectation E_b[I(b*beta, rho)] using Riemann sum with Gaussian weights.
    We integrate over [-5,5] which covers >99.99% of normal distribution.
    """
    b_points = jnp.linspace(-5, 5, n_points)  # we create grid points for b
    db = b_points[1] - b_points[0]  # we compute step size
    gaussian_weights = norm.pdf(b_points)  # we compute Gaussian weights
    expectations = jnp.array([inner_expectation(b * beta, rho) for b in tqdm(b_points, desc="Computing Riemann sum", leave=False)])  # we compute expectation for each b
    return jnp.sum(expectations * gaussian_weights * db)  # we return weighted sum

# %%
rhos = jnp.linspace(-0.99, 0.99, 50)  # we use 20 points for reasonable computation
betas = [0, 0.3, 0.5, 1.0,1.2,1.5, 2.0]  # we test different beta values
n_riemann_points = 50  # we set number of points for Riemann sum

for beta in tqdm(betas, desc="Computing expectations for different betas"):
    total_I_values = []
    for rho in tqdm(rhos, desc=f"Computing rho values for beta={beta}", leave=True):
        total_I_values.append(calculate_total_expectation_riemann(beta, rho, n_riemann_points))  # we compute Riemann sum expectation for each rho
    


# %%

plt.figure(figsize=(10, 7))

colors = plt.cm.rainbow(jnp.linspace(0, 1, len(betas)))  # we create color map for betas

# we plot both empirical and theoretical values with same colors
for beta, color in zip(betas, colors):
    # Empirical values
    plt.plot(rhos, total_I_values, '--', color=color, label=f'$\\beta = {beta}$ (empirical)')  # we use dashed lines for empirical
    
    # Theoretical values
    ntk_values = [ntk_formula(rho, beta) for rho in rhos]  # we compute theoretical NTK values
    plt.plot(rhos, ntk_values, '-', color=color, label=f'$\\beta = {beta}$ (theory)')  # we use solid lines for theoretical

plt.xlabel('$\\rho = x^T x\'$ (Original Inner Product)', fontsize=14)
plt.ylabel('NTK Value', fontsize=14)
plt.title('NTK for 1-layer network with bias $b_i = \\beta n_i$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)  # we add horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.5)  # we add vertical line at x=0
plt.show()


# %%


# %%
