import numpy as np

from experiments.spectral_self_attention.equilibrium_catalogue import (
    classify_beta_zero_equilibrium,
    cluster_equilibrium_residual,
    equilibrium_certificate,
    evaluate_spectral_gram_system,
    factor_spectral_grams,
    regular_polygon,
    regular_simplex,
)
from experiments.spectral_self_attention.simulator import (
    attention_weights,
    diagnostics,
    energy,
    integrate,
    is_linearly_stable_pure_mode,
    normalize,
    pure_mode_linear_rates,
    pure_mode_tangent_jacobian,
    random_sphere,
    vector_field,
)
from experiments.spectral_self_attention.mixed_equilibria import (
    embed_mixed_balanced_state,
    mixed_balanced_roots,
    mixed_balanced_state,
    mixed_three_roots,
    mixed_three_state,
    root_diagnostics,
)


def test_vector_field_is_tangent_and_attention_is_stochastic():
    rng = np.random.default_rng(4)
    x = random_sphere(rng, batch=3, n_tokens=5, dimension=4)
    eigenvalues = np.array([3.0, 1.0, -0.5, -2.0])
    weights = attention_weights(x, eigenvalues, beta=1.3)
    field = vector_field(x, eigenvalues, beta=1.3)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-13)
    np.testing.assert_allclose(np.sum(x * field, axis=-1), 0.0, atol=1e-13)


def test_pure_modes_are_equilibria_for_every_sign_pattern():
    eigenvalues = np.array([2.0, 0.0, -3.0])
    signs = np.array([1.0, -1.0, -1.0, 1.0])
    for mode in range(3):
        x = np.zeros((1, 4, 3))
        x[0, :, mode] = signs
        np.testing.assert_allclose(vector_field(x, eigenvalues, beta=2.0), 0.0, atol=1e-14)


def test_energy_is_non_decreasing_along_resolved_trajectory():
    rng = np.random.default_rng(12)
    x0 = random_sphere(rng, batch=4, n_tokens=4, dimension=4)
    eigenvalues = np.array([2.0, 0.6, -0.5, -2.8])
    trajectory = integrate(x0, eigenvalues, 0.8, t_final=2.0, dt=0.01, save_every=10)
    increments = np.diff(trajectory.energies, axis=0)
    assert np.min(increments) > -2e-10
    np.testing.assert_allclose(np.linalg.norm(trajectory.states, axis=-1), 1.0, atol=2e-13)


def test_homogeneous_stability_matches_theorem_5_1():
    eigenvalues = np.array([2.0, 1.0, -3.0])
    assert is_linearly_stable_pure_mode(eigenvalues, 1.0, mode=0, n_plus=5, n_minus=0)
    assert not is_linearly_stable_pure_mode(eigenvalues, 1.0, mode=1, n_plus=5, n_minus=0)
    assert not is_linearly_stable_pure_mode(eigenvalues, 1.0, mode=2, n_plus=5, n_minus=0)


def test_balanced_split_closed_form_thresholds():
    beta = 0.7
    eigenvalues = np.array([2.0, 0.5, -3.0])
    for mode in (0, 2):
        rates = pure_mode_linear_rates(eigenvalues, beta, mode, 3, 3)
        assert np.max(rates) < 0

    # Raising the positive transverse eigenvalue above the negative-mode bound
    # destabilizes the split supported on lambda=-3.
    threshold = 3.0 * np.tanh(3.0 * beta)
    unstable = np.array([threshold + 0.05, 0.5, -3.0])
    assert not is_linearly_stable_pure_mode(unstable, beta, mode=2, n_plus=3, n_minus=3)


def test_finite_difference_jacobian_matches_block_spectrum():
    eigenvalues = np.array([2.0, 0.5, -3.0])
    for mode, split in ((0, (3, 2)), (2, (2, 3)), (0, (5, 0))):
        exact = np.sort(pure_mode_linear_rates(eigenvalues, 0.7, mode, *split))
        numeric = np.sort(
            np.linalg.eigvals(
                pure_mode_tangent_jacobian(eigenvalues, 0.7, mode, *split)
            ).real
        )
        np.testing.assert_allclose(numeric, exact, atol=2e-9)


def test_repeated_eigenvalue_diagnostics_selects_eigenspace():
    eigenvalues = np.array([3.0, 3.0, 1.0, -1.0])
    u = np.array([0.6, 0.8, 0.0, 0.0])
    x = np.tile(u, (4, 1))
    result = diagnostics(x, eigenvalues, beta=1.0)
    assert result.geometry[0] == "consensus"
    np.testing.assert_allclose(result.selected_group_mass[0], 1.0)


def test_zero_matrix_is_stationary():
    rng = np.random.default_rng(7)
    x0 = random_sphere(rng, batch=2, n_tokens=3, dimension=4)
    trajectory = integrate(x0, np.zeros(4), 5.0, t_final=1.0, dt=0.05)
    np.testing.assert_allclose(trajectory.states[0], trajectory.states[-1])
    np.testing.assert_allclose(energy(x0, np.zeros(4), 5.0), 9 / 10)


def test_new_mixed_three_token_equilibrium():
    roots = mixed_three_roots(positive=2.0, magnitude=3.0, beta=1.5)
    stable = [root for root in roots if root_diagnostics(root, 2.0, 3.0, 1.5)["linearly_stable"]]
    assert stable
    np.testing.assert_allclose(stable[0], 0.02390527, atol=2e-7)
    state = mixed_three_state(stable[0])
    np.testing.assert_allclose(
        vector_field(state, np.array([2.0, -3.0]), 1.5), 0.0, atol=2e-12
    )


def test_balanced_mixed_family_explains_small_beta_negative_definite_state():
    roots = mixed_balanced_roots(
        center_eigenvalue=-0.4,
        polar_eigenvalue=-4.0,
        beta=0.03,
        n_center=1,
        n_each_polar=2,
    )
    assert len(roots) == 1
    np.testing.assert_allclose(roots[0], -0.1141124228, atol=2e-10)
    state = mixed_balanced_state(roots[0], n_center=1, n_each_polar=2)
    np.testing.assert_allclose(
        vector_field(state, np.array([-0.4, -4.0]), 0.03), 0.0, atol=2e-12
    )
    embedded = embed_mixed_balanced_state(
        roots[0], 4, center_mode=0, polar_mode=3, n_center=1, n_each_polar=2
    )
    np.testing.assert_allclose(
        vector_field(embedded, np.array([-0.4, -1.0, -2.0, -4.0]), 0.03),
        0.0,
        atol=2e-12,
    )


def test_spectral_gram_characterization_on_mixed_equilibrium():
    roots = mixed_balanced_roots(2.0, -3.0, 1.5)
    state = embed_mixed_balanced_state(
        roots[1], 4, center_mode=0, polar_mode=3
    )
    certificate = equilibrium_certificate(
        state, np.array([2.0, 0.0, -0.5, -3.0]), 1.5
    )
    assert certificate.normalized_field < 2e-12
    assert certificate.multiplier_equation < 2e-12
    assert certificate.spectral_gram_equation < 2e-12
    values = [2.0, 0.0, -0.5, -3.0]
    grams = [state[:, [index]] @ state[:, [index]].T for index in range(4)]
    gram_certificate = evaluate_spectral_gram_system(
        values, grams, 1.5, eigenspace_dimensions=[1, 1, 1, 1]
    )
    assert gram_certificate.equation_residual < 2e-12
    assert gram_certificate.diagonal_residual < 2e-12
    assert gram_certificate.rank_violations == (0, 0, 0, 0)
    reconstructed = factor_spectral_grams(grams, [1, 1, 1, 1])
    reconstructed_certificate = equilibrium_certificate(
        reconstructed, np.asarray(values), 1.5
    )
    assert reconstructed_certificate.normalized_field < 2e-12


def test_beta_zero_complete_classes():
    centered = regular_polygon(5)
    assert classify_beta_zero_equilibrium(centered, np.array([2.0, -1.0])) == "zero_mean_output"
    eigenline = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    assert classify_beta_zero_equilibrium(eigenline, np.array([2.0, -1.0])) == "unbalanced_eigenline"
    generic = normalize(np.array([[1.0, 0.2], [0.3, 1.0]]))
    assert classify_beta_zero_equilibrium(generic, np.array([2.0, -1.0])) == "not_equilibrium"


def test_regular_simplex_and_polygon_in_repeated_eigenspace():
    simplex = regular_simplex(4)
    np.testing.assert_allclose(
        cluster_equilibrium_residual(simplex, np.ones(4), np.full(3, -2.0), 1.7),
        0.0,
        atol=2e-12,
    )
    polygon = regular_polygon(7, phase=0.13)
    np.testing.assert_allclose(
        cluster_equilibrium_residual(polygon, np.ones(7), np.full(2, 3.0), 0.9),
        0.0,
        atol=2e-12,
    )
