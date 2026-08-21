from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import sympy as sp


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "feynman"
    / "exact_orthogonal_weingarten.py"
)
SPEC = spec_from_file_location("exact_orthogonal_weingarten", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_second_projector_cumulant_is_gamma_vertex():
    n, r = sp.symbols("n r", positive=True, integer=True)
    gamma = MODULE.gamma_defect(n, r)
    diagonal_covariance = MODULE.projector_entry_cumulant(
        ((0, 0), (0, 0)), n, r
    )
    off_diagonal_variance = MODULE.projector_entry_cumulant(
        ((0, 1), (0, 1)), n, r
    )
    assert sp.simplify(diagonal_covariance - 2 * gamma * (1 - 1 / n)) == 0
    assert sp.simplify(off_diagonal_variance - gamma) == 0


def test_diagonal_moments_match_beta_distribution():
    n, r = 9, 4
    for order in range(1, 4):
        actual = MODULE.projector_entry_moment(
            tuple((0, 0) for _ in range(order)), n, r
        )
        expected = (
            sp.Rational(n, r) ** order
            * sp.rf(sp.Rational(r, 2), order)
            / sp.rf(sp.Rational(n, 2), order)
        )
        assert sp.simplify(actual - expected) == 0


def test_full_rank_projector_cumulants_vanish_exactly():
    n = 8
    entries = ((0, 0), (0, 1), (1, 1))
    assert MODULE.projector_entry_moment(entries, n, n) == 0
    for order in (2, 3):
        assert MODULE.projector_entry_cumulant(entries[:order], n, n) == 0
