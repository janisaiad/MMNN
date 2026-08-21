from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "feynman"
    / "benigni_paquette_mmnn_spectrum.py"
)
SPEC = spec_from_file_location("benigni_paquette_mmnn_spectrum", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_bp_explicit_population_is_probability_with_unit_mean() -> None:
    population = MODULE.bp_explicit_population(0.5, points=300, bins=500)
    assert np.isclose(np.sum(population.weights), 1.0, atol=1e-12)
    assert np.isclose(population.mean, 1.0, rtol=4e-3)


def test_homogeneous_concatenation_has_expected_scaling() -> None:
    population = MODULE.bp_explicit_population(0.4, points=250, bins=400)
    raw = MODULE.homogeneous_concatenated_population(population, 5)
    normalized = MODULE.homogeneous_concatenated_population(
        population, 5, normalize_by_depth=True
    )
    assert np.isclose(raw.mean, 5.0 * population.mean)
    assert np.isclose(normalized.mean, population.mean)


def test_mp_linear_response_matches_finite_difference() -> None:
    population = MODULE.bp_explicit_population(0.5, points=250, bins=400)
    gamma = 0.2
    z = 1.3 + 0.4j
    scale = 1.25
    epsilon = 2e-5
    perturbation = MODULE.scale_mixture_perturbation(population, scale)
    analytic = MODULE.mp_population_linear_response(
        z, population, gamma, perturbation
    )
    perturbed = MODULE.mixed_population(population, scale, epsilon)
    finite_difference = (
        MODULE.mp_stieltjes(z, perturbed, gamma)
        - MODULE.mp_stieltjes(z, population, gamma)
    ) / epsilon
    assert np.allclose(analytic, finite_difference, rtol=2e-3, atol=2e-4)
