import pytest
import jax.numpy as jnp
from jax import random
from ntk.ntk_infinite import compute_ntk_2layer_montecarlo, relu, relu_dot

def test_ntk_2layer_montecarlo_basic():
    """
    we test basic functionality of the 2-layer Monte Carlo NTK computation
    """
    key = random.PRNGKey(0)
    X = random.normal(key, (4, 3))  # i create small test input
    
    result = compute_ntk_2layer_montecarlo(
        X, 
        ranks=[5,5],
        sigma_A=jnp.sqrt(2),
        sigma_c=1.0,
        beta=1.0,
        activation_fn=relu,
        activation_dot_fn=relu_dot,
        key=key,
        n_samples=100
    )
    
    assert result.shape == (4, 4)  # i check output shape matches input samples
    assert not jnp.any(jnp.isnan(result))  # i verify no NaN values
    assert jnp.all(jnp.isfinite(result))  # i verify all values are finite

def test_ntk_2layer_montecarlo_symmetry():
    """
    we test that the output kernel is symmetric
    """
    key = random.PRNGKey(1)
    X = random.normal(key, (3, 2))
    
    result = compute_ntk_2layer_montecarlo(X,key=key,n_samples=100)
    
    assert jnp.allclose(result, result.T, rtol=1e-5)  # i check symmetry

def test_ntk_2layer_montecarlo_positive_definite():
    """
    we test that the output kernel is positive definite
    """
    key = random.PRNGKey(2)
    X = random.normal(key, (3, 2))
    
    result = compute_ntk_2layer_montecarlo(X,key=key,n_samples=100)
    
    eigenvals = jnp.linalg.eigvalsh(result)
    assert jnp.all(eigenvals > -1e-10)  # i allow for small numerical errors

def test_ntk_2layer_montecarlo_parameter_effects():
    """
    we test that changing parameters affects the output
    """
    key = random.PRNGKey(3)
    X = random.normal(key, (3, 2))
    
    result1 = compute_ntk_2layer_montecarlo(X, sigma_A=1.0,key=key,n_samples=100)
    result2 = compute_ntk_2layer_montecarlo(X, sigma_A=2.0,key=key,n_samples=100)
    
    assert not jnp.allclose(result1, result2)  # i verify different parameters give different results

def test_ntk_2layer_montecarlo_input_validation():
    """
    we test input validation
    """
    key = random.PRNGKey(4)
    
    with pytest.raises(Exception):  # i expect error for empty input
        compute_ntk_2layer_montecarlo(jnp.array([]),key=key,n_samples=100)
        
    with pytest.raises(Exception):  # i expect error for 1D input
        compute_ntk_2layer_montecarlo(jnp.array([1,2,3]),key=key,n_samples=100)

if __name__ == "__main__":
    test_ntk_2layer_montecarlo_basic()
    test_ntk_2layer_montecarlo_symmetry()
    test_ntk_2layer_montecarlo_positive_definite()
    test_ntk_2layer_montecarlo_parameter_effects()
    test_ntk_2layer_montecarlo_input_validation()