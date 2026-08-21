from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).parents[1] / "experiments" / "feynman"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "run_mmnn_ntk_variance_scaling.py"
SPEC = spec_from_file_location("run_mmnn_ntk_variance_scaling", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stiefel_frozen_factor_has_exact_normalized_projector_gram():
    rng = np.random.default_rng(1)
    W = MODULE.frozen_matrix(rng, width=9, input_rank=4, ensemble="stiefel")
    np.testing.assert_allclose(W.T @ W / 9.0, np.eye(4), atol=2e-15)


def test_small_variance_estimate_is_reproducible_and_positive():
    angles = np.asarray([0.1, 0.8, 1.7])
    inputs = np.stack((np.cos(angles), np.sin(angles)), axis=1) * np.sqrt(2.0)
    first = MODULE.estimate_ntk_statistics(
        inputs,
        width=10,
        rank=3,
        depth=3,
        samples=8,
        ensemble="gaussian",
        seed=5,
    )
    second = MODULE.estimate_ntk_statistics(
        inputs,
        width=10,
        rank=3,
        depth=3,
        samples=8,
        ensemble="gaussian",
        seed=5,
    )
    assert first == second
    assert first["offdiagonal_variance_mean"] > 0.0
