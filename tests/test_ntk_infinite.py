import pytest
import jax.numpy as jnp
import jax
from ntk.ntk_infinite import compute_ntk_nngp_recursive, relu, relu_dot # type: ignore

def test_compute_ntk_nngp_recursive():
    # we set up test parameters
    batch_size = 4
    input_dim = 3
    L = 3  # we test with 3 layers
    d_hidden = [5, 4]  # we test with hidden dimensions [5,4], last in 1 implicitely
    sigma_A = 1.0
    sigma_c = 0.1
    beta = 0.01

    # we create random test input data
    rng = jax.random.PRNGKey(0)
    X = jnp.array(jax.random.normal(rng, (batch_size, input_dim)))

    # we compute kernels
    kernels = compute_ntk_nngp_recursive(
        X, L, d_hidden, sigma_A, sigma_c, beta,
        activation_fn=relu, activation_dot_fn=relu_dot
    )

    # we verify the output structure
    assert 'nngp' in kernels and 'ntk' in kernels, "we expect both nngp and ntk kernels"
    assert len(kernels['nngp']) == L, "we expect nngp kernels for all layers"
    assert len(kernels['ntk']) == L, "we expect ntk kernels for all layers"

    # we verify kernel shapes
    for l in range(1, L+1):
        assert kernels['nngp'][l].shape == (batch_size, batch_size), "we expect correct nngp shape"
        assert kernels['ntk'][l].shape == (batch_size, batch_size), "we expect correct ntk shape"

    # we verify symmetry of kernels
    for l in range(1, L+1):
        print(kernels['nngp'][l])
        assert jnp.allclose(kernels['nngp'][l], kernels['nngp'][l].T, atol=1e-6), "we expect symmetric nngp"
        assert jnp.allclose(kernels['ntk'][l], kernels['ntk'][l].T, atol=1e-6), "we expect symmetric ntk"

    # we verify positive semi-definiteness
    for l in range(1, L+1):
        eigvals_nngp = jnp.linalg.eigvalsh(kernels['nngp'][l])
        eigvals_ntk = jnp.linalg.eigvalsh(kernels['ntk'][l])
        assert jnp.all(eigvals_nngp >= -1e-10), "we expect non-negative eigenvalues for nngp"
        assert jnp.all(eigvals_ntk >= -1e-10), "we expect non-negative eigenvalues for ntk"

    # we verify diagonal elements are positive
    for l in range(1, L+1):
        assert jnp.all(jnp.diag(kernels['nngp'][l]) > 0), "we expect positive diagonal elements for nngp"
        assert jnp.all(jnp.diag(kernels['ntk'][l]) > 0), "we expect positive diagonal elements for ntk"

if __name__ == "__main__":
    test_compute_ntk_nngp_recursive()