from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "feynman"
    / "deformed_wigner_dyson.py"
)
SPEC = spec_from_file_location("deformed_wigner_dyson", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_semicircle_moments_are_catalan():
    moments = MODULE.deformed_wigner_moments(np.zeros(20), sigma=0.7, max_order=8)
    expected = np.zeros(9)
    expected[0] = 1.0
    expected[2] = 0.7**2
    expected[4] = 2 * 0.7**4
    expected[6] = 5 * 0.7**6
    expected[8] = 14 * 0.7**8
    np.testing.assert_allclose(moments, expected, rtol=2e-13, atol=2e-13)


def test_first_two_deformed_moments():
    eta = np.asarray([0.1, 0.4, 1.2, 2.0])
    sigma = 0.3
    moments = MODULE.deformed_wigner_moments(eta, sigma, max_order=4)
    assert np.isclose(moments[1], eta.mean())
    assert np.isclose(moments[2], np.mean(eta**2) + sigma**2)


def test_local_overlap_sum_rule():
    eta = np.arange(1, 33, dtype=float) ** -1.5
    grid = np.linspace(-0.3, 1.2, 100)
    m, diagonal = MODULE.solve_dyson(eta, grid, sigma=0.08, imaginary_part=0.01)
    density, local, overlaps = MODULE.dyson_density_and_overlaps(m, diagonal)
    active = density > 1e-10
    np.testing.assert_allclose(overlaps[active].sum(axis=1), 1.0, atol=2e-12)
    np.testing.assert_allclose(local.mean(axis=1), density, atol=2e-12)
