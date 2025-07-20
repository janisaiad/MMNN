import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
from typing import Callable, Tuple, Union, List

@partial(jit, static_argnums=(0,))
def nd_gaussian_expectation(f: Callable, 
                          mus: Union[List[float], jnp.ndarray],
                          sigmas: Union[List[float], jnp.ndarray],
                          rho: Union[List[List[float]], jnp.ndarray],
                          num_points: int = 32) -> float:
    """
    I compute E[f(x)] where x follows a multivariate Gaussian distribution
    
    Args:
        f: Function to integrate that takes a vector input
        mus: Mean vector of the Gaussian
        sigmas: Standard deviation vector
        rho: Correlation matrix
        num_points: Number of points per dimension for grid integration
        
    Returns:
        Expectation value E[f(x)] where x ~ N(mu, Sigma)
    """
    # I convert inputs to arrays
    mus = jnp.array(mus)
    sigmas = jnp.array(sigmas)
    rho = jnp.array(rho)
    n_dims = len(mus)
    
    # cov matrix
    sigma_matrix = jnp.outer(sigmas, sigmas) * rho  # we multiply standard deviations with correlation
    
    # grid
    grid_points = []
    dxs = []
    for i in range(n_dims):
        x = jnp.linspace(mus[i] - 6*sigmas[i], mus[i] + 6*sigmas[i], num_points)
        grid_points.append(x)
        dxs.append(x[1] - x[0])
    
    # meshgrid
    grid_arrays = jnp.meshgrid(*grid_points, indexing='ij')
    grid_flat = jnp.stack([x.flatten() for x in grid_arrays], axis=-1)
    
    # weights to integrate
    diff = grid_flat - mus
    inv_sigma = jnp.linalg.inv(sigma_matrix)
    exponent = -0.5 * jnp.sum(diff @ inv_sigma * diff, axis=1)
    det = jnp.linalg.det(sigma_matrix)
    norm = jnp.power(2 * jnp.pi, n_dims/2) * jnp.sqrt(det)
    weights = jnp.exp(exponent) / norm
    
    # function values and integrate
    f_vals = vmap(f)(grid_flat)
    integral = jnp.sum(f_vals * weights) * jnp.prod(jnp.array(dxs))
    
    return integral












# to use it later because it's directly the law inside the activation
@partial(jit, static_argnums=(0,))
def variance_gamma_expectation(f: Callable, mu: float = 0.0, sigma: float = 1.0,
                             nu: float = 1.0, theta: float = 0.0,
                             num_points: int = 1000) -> float:
    """
    I compute the expectation of a function f under a Variance Gamma distribution
    
    Args:
        f: Function to integrate
        mu: Location parameter
        sigma: Scale parameter
        nu: Shape parameter (variance rate)
        theta: Skewness parameter
        num_points: Number of points for numerical integration
        
    Returns:
        Expectation value E[f(x)] where x follows VG distribution
    """
    # I create integration grid (wider range due to heavier tails)
    x = jnp.linspace(mu - 10*sigma, mu + 10*sigma, num_points)
    dx = x[1] - x[0]
    
    # I compute VG weights using the modified Bessel function
    z = jnp.sqrt(2*theta**2/sigma**2 + 2/nu)
    weights = (jnp.abs(x - mu)**(1/nu - 0.5) * 
              jnp.exp((theta*(x - mu))/sigma**2) *
              jnp.exp(-z*jnp.abs(x - mu)/sigma))
    weights = weights / (sigma * jnp.sqrt(2*jnp.pi*nu))
    
    # I compute function values and integrate
    f_vals = vmap(f)(x)
    return jnp.sum(f_vals * weights) * dx
