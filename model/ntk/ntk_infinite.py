# we derive a formula for the Neural Tangent Kernel (NTK) with a recursive relation similar to FCNN
# this comes from the chain rule and concatenation properties

# for layer l, the NTK kernel Θ^(l)(x,x') follows the recursive formula:
# Θ^(l)(x,x') = Θ^(l-1)(x,x') * (σ_A^2 * Σ_dot^(l)(x,x')) + Σ^(l)(x,x') + σ_c^2

# where:
# - Σ^(l) is the base NNGP kernel 
# - Σ_dot^(l) is the derivative kernel
# - σ_A and σ_c are scaling parameters

# we study the limiting behavior as the width of hidden layers goes to infinity
# while maintaining low rank structure

# the exact formula in our notation is:
# K_n(x,x') = 1 + σ_L(K_tilde_n-1) + σ_L_dot(K_tilde_n-1) * K_n-1

# This should be an analysis with analytic formulas for the kernels, but the computations are hard and currently in progress by hand
# we use scipy and jax to compute the kernels numerically

# for a fast implementation, we will use jax to compute some means with integrals numerically in cuda in utils.py

import numpy as np
import scipy.linalg as la
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from utils.means import nd_gaussian_expectation

# we define our activation function and its derivative
# here we use ReLU as a common example
@jit
def relu(x):
    return jnp.maximum(0, x)

@jit
def relu_dot(x):
    return (x > 0).astype(x.dtype)

def compute_ntk_nngp_recursive(X, L, d_hidden, sigma_A, sigma_c, beta, activation_fn=relu, activation_dot_fn=relu_dot):
    """
    we compute the NNGP and NTK kernels for a deep MMNN recursively for all layers.

    Args:
        X (jnp.ndarray): Input data of shape (num_samples, d_0).
        L (int): Total number of layers (depth).
        d_hidden (list[int]): List of hidden dimensions for layers 1 to L-1.
        sigma_A (float): Weight scaling parameter.
        sigma_c (float): Bias scaling parameter.
        beta (float): Random bias scaling parameter.
        activation_fn (callable): The activation function.
        activation_dot_fn (callable): The derivative of the activation function.

    Returns:
        dict: A dictionary containing the NNGP and NTK kernel matrices for each layer.
    """
    num_samples, d_0 = X.shape
    dims = [d_0] + d_hidden # we create a list of dimensions including input
    
    # we initialize storage for kernels
    kernels = {'nngp': {}, 'ntk': {}}
    
    # we initialize nngp_l_minus_1 and ntk_l_minus_1 for the first layer
    nngp_l_minus_1 = jnp.zeros((num_samples, num_samples))
    ntk_l_minus_1 = jnp.zeros((num_samples, num_samples))

    for l in range(1, L + 1):
        d_l_minus_1 = dims[l-1]
        
        # we compute the full physical nngp kernel of the previous layer
        k_nngp_prev = sigma_A**2 * nngp_l_minus_1 + sigma_c**2
        
        # we define the functions to integrate for the current layer's kernels
        f_nngp = lambda g: activation_fn(g[0]) * activation_fn(g[1])
        f_nngp_dot = lambda g: d_l_minus_1 * activation_dot_fn(g[0]) * activation_dot_fn(g[1])









        # we prepare a function to compute one element (i, j) of the kernel matrices
        @jit
        def compute_kernel_element(i, j):
            # we handle the base case and recursive step for pre-activation stats
            if l == 1:
                # for the first layer, h_0 is the deterministic input x
                k_prev_diag_i = jnp.dot(X[i], X[i])
                k_prev_diag_j = jnp.dot(X[j], X[j])
                k_prev_offdiag = jnp.dot(X[i], X[j])
                
                var_i = k_prev_diag_i + beta**2
                var_j = k_prev_diag_j + beta**2
                cov = k_prev_offdiag + beta**2
            else:
                # for subsequent layers, h_{l-1} is a gaussian process
                k_prev_diag_i = k_nngp_prev[i, i]
                k_prev_diag_j = k_nngp_prev[j, j]
                k_prev_offdiag = k_nngp_prev[i, j]
                
                var_i = d_l_minus_1 * k_prev_diag_i + beta**2
                var_j = d_l_minus_1 * k_prev_diag_j + beta**2
                cov = d_l_minus_1 * k_prev_offdiag + beta**2

            # we set up parameters for the 2d gaussian expectation
            mus = jnp.array([0.0, 0.0])
            sigmas = jnp.sqrt(jnp.array([var_i, var_j]))
            rho_val = cov / (sigmas[0] * sigmas[1])
            # we clip rho_val to avoid numerical instability, staying away from singular boundaries
            eps = 1e-6
            rho_val = jnp.clip(rho_val, -1.0 + eps, 1.0 - eps)
            rho = jnp.array([[1.0, rho_val], [rho_val, 1.0]])

            # we compute the base kernels for the current layer
            nngp_l_ij = nd_gaussian_expectation(f_nngp, mus, sigmas, rho)
            nngp_dot_l_ij = nd_gaussian_expectation(f_nngp_dot, mus, sigmas, rho)
            
            # we compute the ntk for the current layer using the recursive formula
            ntk_l_ij = ntk_l_minus_1[i, j] * (sigma_A**2 * nngp_dot_l_ij) + nngp_l_ij + sigma_c**2
            
            return nngp_l_ij, ntk_l_ij










        # we use vmap to compute the full kernel matrices efficiently
        # we create index pairs for all upper-triangular elements, including diagonal
        idx_i, idx_j = jnp.triu_indices(num_samples)
        
        # we compute the upper-triangular parts
        nngp_triu, ntk_triu = vmap(compute_kernel_element)(idx_i, idx_j)
        
        # we build the full symmetric matrices
        # first, we create an upper triangular matrix
        nngp_l_upper = jnp.zeros_like(nngp_l_minus_1).at[idx_i, idx_j].set(nngp_triu)
        ntk_l_upper = jnp.zeros_like(ntk_l_minus_1).at[idx_i, idx_j].set(ntk_triu)

        # we make it symmetric by adding the transpose and subtracting the diagonal which was counted twice
        nngp_l = nngp_l_upper + nngp_l_upper.T - jnp.diag(jnp.diag(nngp_l_upper))
        ntk_l = ntk_l_upper + ntk_l_upper.T - jnp.diag(jnp.diag(ntk_l_upper))
        
        # we store the computed kernels
        kernels['nngp'][l] = nngp_l
        kernels['ntk'][l] = ntk_l
        
        # we update the kernels for the next iteration
        nngp_l_minus_1 = nngp_l
        ntk_l_minus_1 = ntk_l
        
    return kernels
