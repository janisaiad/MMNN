from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import sympy as sp


SCRIPT_DIR = Path(__file__).parents[1] / "experiments" / "feynman"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "exact_mmnn_gram_wick.py"
SPEC = spec_from_file_location("exact_mmnn_gram_wick", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_covariance_is_the_exact_wishart_rank_vertex():
    r = sp.symbols("r", positive=True, integer=True)
    assert MODULE.gram_entry_cumulant(((0, 1), (0, 1)), r) == 1 / r
    assert MODULE.gram_entry_cumulant(((0, 0), (0, 0)), r) == 2 / r
    assert MODULE.gram_entry_cumulant(((0, 0), (1, 1)), r) == 0


def test_diagonal_moments_and_cumulants_match_normalized_chi_square():
    r = 7
    for order in range(1, 6):
        entries = ((0, 0),) * order
        expected_moment = sp.prod(1 + sp.Rational(2 * j, r) for j in range(order))
        expected_cumulant = (
            sp.Integer(1)
            if order == 1
            else 2 ** (order - 1) * sp.factorial(order - 1) / r ** (order - 1)
        )
        assert sp.simplify(MODULE.gram_entry_moment(entries, r) - expected_moment) == 0
        assert sp.simplify(MODULE.gram_entry_cumulant(entries, r) - expected_cumulant) == 0


def test_only_connected_wick_pairings_enter_a_cumulant():
    for order in range(2, 6):
        tau = MODULE.canonical_pairing(order)
        connected = MODULE.gram_cumulant_coefficients(order, 5)
        assert connected
        assert all(
            MODULE.join_component_count(pairing, tau) == 1
            for pairing, _ in connected
        )
        assert all(coefficient == sp.Rational(1, 5 ** (order - 1)) for _, coefficient in connected)
