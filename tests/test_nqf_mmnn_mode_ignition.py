from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "feynman"
    / "nqf_mmnn_mode_ignition.py"
)
SPEC = spec_from_file_location("nqf_mmnn_mode_ignition", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _symmetric_structure(rng: np.random.Generator, samples: int, p: int) -> np.ndarray:
    raw = rng.normal(size=(samples, p, p))
    return 0.5 * (raw + np.swapaxes(raw, 1, 2))


def test_nqf_ntk_is_exact_parameter_gradient_gram() -> None:
    rng = np.random.default_rng(3)
    structure = _symmetric_structure(rng, samples=7, p=5)
    weights = rng.normal(size=(5, 4))
    order_parameter = weights @ weights.T
    closed = MODULE.nqf_ntk(structure, order_parameter)
    explicit = MODULE.explicit_parameter_ntk(structure, weights)
    assert np.allclose(closed, explicit, rtol=2e-13, atol=2e-13)


def test_all_valence_gaussian_vertex_reduces_to_trace_words() -> None:
    rng = np.random.default_rng(31)
    matrices = [
        _symmetric_structure(rng, samples=1, p=4)[0] for _ in range(3)
    ]
    epsilon = 0.07
    components = 9
    covariance = MODULE.gaussian_trace_cumulant(
        matrices[:2], epsilon=epsilon, component_count=components
    )
    expected_covariance = (
        2.0 * epsilon**2 * np.trace(matrices[0] @ matrices[1]) / components
    )
    assert np.isclose(covariance, expected_covariance, rtol=2e-14, atol=2e-14)

    third = MODULE.gaussian_trace_cumulant(
        matrices, epsilon=epsilon, component_count=components
    )
    expected_third = (
        4.0
        * epsilon**3
        * (
            np.trace(matrices[0] @ matrices[1] @ matrices[2])
            + np.trace(matrices[0] @ matrices[2] @ matrices[1])
        )
        / components**2
    )
    assert np.isclose(third, expected_third, rtol=2e-14, atol=2e-14)


def test_orthogonal_feature_formula_gives_complete_ntk_spectrum() -> None:
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(9, 5)))
    column_scales = np.geomspace(0.7, 1.8, 5)
    feature_eigenvalues = q * column_scales
    modes = np.geomspace(0.02, 1.1, 5)
    kernel = MODULE.orthogonal_ntk(feature_eigenvalues, modes)
    actual = np.linalg.eigvalsh(kernel)[::-1]
    labeled = MODULE.orthogonal_ntk_eigenvalues(feature_eigenvalues, modes)
    expected = np.concatenate((np.sort(labeled)[::-1], np.zeros(4)))
    assert np.allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_logistic_solution_satisfies_closed_mode_ode() -> None:
    rates = np.array([0.2, 0.7, 1.4])
    competition = np.array([0.8, 1.1, 1.7])
    initial = np.array([2e-4, 3e-4, 5e-4])
    t = 2.3
    step = 1e-6
    z_minus, z, z_plus = MODULE.orthogonal_logistic(
        rates, competition, initial, np.array([t - step, t, t + step])
    )
    numerical = (z_plus - z_minus) / (2.0 * step)
    expected = (rates - competition * z) * z
    assert np.allclose(numerical, expected, rtol=2e-8, atol=2e-11)


def test_isotropic_matrix_solution_resums_the_riccati_flow() -> None:
    rng = np.random.default_rng(8)
    target = _symmetric_structure(rng, samples=1, p=4)[0] * 0.15
    raw = rng.normal(size=(4, 3))
    initial = 0.03 * raw @ raw.T
    isotropy = 0.2
    t = 0.7
    step = 2e-6
    minus = MODULE.isotropic_solution(target, initial, isotropy, t - step)
    center = MODULE.isotropic_solution(target, initial, isotropy, t)
    plus = MODULE.isotropic_solution(target, initial, isotropy, t + step)
    numerical = (plus - minus) / (2.0 * step)
    expected = MODULE.isotropic_rhs(target, center, isotropy)
    assert np.allclose(numerical, expected, rtol=2e-7, atol=2e-9)


def test_wishart_and_stiefel_log_clock_moments_match_sampling() -> None:
    rng = np.random.default_rng(12)
    epsilon = 1e-6
    samples = 300_000
    p, d = 40, 8
    for ensemble in ("gaussian", "stiefel"):
        seeds = MODULE.sample_initial_modes(
            ensemble,
            epsilon=epsilon,
            component_count=d,
            ambient_dimension=p,
            samples=samples,
            rng=rng,
        )
        log_seed = np.log(seeds / epsilon)
        if ensemble == "gaussian":
            mean, variance = MODULE.gaussian_log_seed_moments(d)
        else:
            mean, variance = MODULE.stiefel_log_seed_moments(p, d)
        assert np.isclose(np.mean(log_seed), mean, atol=5e-3)
        assert np.isclose(np.var(log_seed), variance, rtol=1.2e-2)


def test_full_rank_stiefel_seed_has_no_clock_disorder() -> None:
    rng = np.random.default_rng(14)
    epsilon = 3e-7
    seeds = MODULE.sample_initial_modes(
        "stiefel",
        epsilon=epsilon,
        component_count=32,
        ambient_dimension=32,
        samples=1000,
        rng=rng,
    )
    mean, variance = MODULE.stiefel_log_seed_moments(32, 32)
    assert np.array_equal(seeds, np.full(1000, epsilon))
    assert mean == 0.0
    assert variance == 0.0
