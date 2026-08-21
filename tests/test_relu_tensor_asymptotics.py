from importlib.util import module_from_spec, spec_from_file_location
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).parents[1] / "experiments" / "feynman"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "relu_tensor_asymptotics.py"
SPEC = spec_from_file_location("relu_tensor_asymptotics", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CERTIFICATE_SCRIPT = SCRIPT_DIR / "certify_relu_collision_sources.py"
CERTIFICATE_SPEC = spec_from_file_location(
    "certify_relu_collision_sources", CERTIFICATE_SCRIPT
)
CERTIFICATE_MODULE = module_from_spec(CERTIFICATE_SPEC)
assert CERTIFICATE_SPEC.loader is not None
sys.modules[CERTIFICATE_SPEC.name] = CERTIFICATE_MODULE
CERTIFICATE_SPEC.loader.exec_module(CERTIFICATE_MODULE)


def test_diagonal_coefficients_match_exact_single_input_polynomials():
    component = (0, 0, 0, 0)
    assert MODULE.normalized_leading_coefficient("V", component) == 5
    assert MODULE.normalized_leading_coefficient("D", component) == Fraction(3, 2)
    assert MODULE.normalized_leading_coefficient("F", component) == Fraction(1, 2)
    assert MODULE.normalized_leading_coefficient("A", component) == Fraction(7, 12)
    assert MODULE.normalized_leading_coefficient("B", component) == Fraction(1, 6)


def test_all_plotted_coefficients_are_nonzero():
    for tensor in MODULE.POWERS:
        for component in MODULE.PLOT_COMPONENTS:
            assert MODULE.normalized_leading_coefficient(tensor, component) > 0


def test_A_collision_sectors_and_crossed_B_formula():
    assert MODULE.normalized_leading_coefficient("A", (0, 0, 1, 1)) == Fraction(7, 12)
    assert MODULE.normalized_leading_coefficient("A", (0, 0, 1, 2)) == Fraction(47, 240)
    assert MODULE.normalized_leading_coefficient("A", (0, 1, 0, 1)) == Fraction(227, 2880)
    assert MODULE.normalized_leading_coefficient("A", (0, 1, 0, 2)) == Fraction(149, 1920)
    assert MODULE.normalized_leading_coefficient("A", (0, 1, 2, 3)) == Fraction(37, 480)
    assert MODULE.normalized_leading_coefficient("B", (0, 1, 2, 3)) == Fraction(1, 288)


def test_A_source_term_certificate_sums_to_all_five_exact_sources():
    expected = {
        "both_diagonal": Fraction(7, 4),
        "one_off_diagonal": Fraction(47, 40),
        "same_off_diagonal_pair": Fraction(227, 320),
        "one_shared_label": Fraction(447, 640),
        "four_distinct_labels": Fraction(111, 160),
    }
    for collision_class, total in expected.items():
        assert CERTIFICATE_MODULE.exact_source_total(collision_class) == total


def test_angle_map_has_the_critical_quadratic_drift():
    theta = 2e-4
    drift = MODULE.relu_angle_map(theta) - theta
    expected = -(theta**2) / (3.0 * np.pi)
    assert np.isclose(drift, expected, rtol=2e-3, atol=1e-13)
