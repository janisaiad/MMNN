import numpy as np

from experiments.spectral_self_attention.maxcut_ou import (
    binary_hamiltonian,
    cut_values,
    exact_maxcut,
)


def test_binary_attention_energy_is_exactly_affine_in_maxcut() -> None:
    rng = np.random.default_rng(7)
    upper = np.triu(rng.uniform(0.1, 2.0, size=(6, 6)), 1)
    weights = upper + upper.T
    signs = rng.choice([-1.0, 1.0], size=(20, 6))
    gamma = 1.27

    cuts = cut_values(signs, weights)
    energy = binary_hamiltonian(signs, weights, gamma)
    edge_weight = np.sum(np.triu(weights, 1))

    expected = -edge_weight * np.exp(-gamma) - 2.0 * np.sinh(gamma) * cuts
    np.testing.assert_allclose(energy, expected, rtol=1e-12, atol=1e-12)


def test_exact_maxcut_triangle() -> None:
    weights = np.ones((3, 3)) - np.eye(3)
    optimum, signs = exact_maxcut(weights)

    assert optimum == 2.0
    assert cut_values(signs, weights) == 2.0
