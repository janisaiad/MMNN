from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).parents[1] / "experiments" / "feynman"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "run_dyson_powerlaw_early_stopping.py"
SPEC = spec_from_file_location("run_dyson_powerlaw_early_stopping", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_spectral_cost_has_continuous_zero_rate_limit():
    rate = np.asarray([0.0, 1e-16, 1e-8])
    cost = MODULE.spectral_cost(rate, sample_size=17, time=0.4, beta=2.0)
    assert np.isclose(cost[0], 0.5 + 0.4 / 2.0)
    assert np.isclose(cost[1], cost[0])
    assert np.isclose(cost[2], cost[0], rtol=2e-6)


def test_normalized_local_spectral_measures_have_unit_mass():
    grid = np.linspace(-1.0, 1.0, 1001)
    centers = np.asarray([-0.3, 0.4])
    epsilon = 0.03
    diagonal = 1.0 / (centers[None, :] - (grid[:, None] + 1j * epsilon))
    local = MODULE.normalized_local_density(grid, diagonal)
    np.testing.assert_allclose(np.trapezoid(local, grid, axis=0), 1.0)
