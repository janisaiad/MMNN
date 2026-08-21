from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "feynman"
    / "run_exact_relu_tensor_recursions.py"
)
SPEC = spec_from_file_location("exact_relu_recursions", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_orthant_moments_independent_gaussians():
    queries = MODULE.OrthantMomentTable.build_queries(4)
    table = MODULE.OrthantMomentTable(
        np.eye(4), queries, MODULE.CubatureConfig(rtol=1e-8, atol=1e-11)
    )
    assert np.isclose(table.get((0, 1, 2, 3), ()), 1.0 / 16.0, rtol=2e-6)
    expected = (1.0 / np.sqrt(2.0 * np.pi)) ** 4
    assert np.isclose(
        table.get((0, 1, 2, 3), (0, 1, 2, 3)), expected, rtol=2e-6
    )


def test_rank_one_single_input_exact_polynomials():
    variance = 1.7
    queries = MODULE.OrthantMomentTable.build_queries(1)
    table = MODULE.OrthantMomentTable(
        np.asarray([[variance]]), queries, MODULE.CubatureConfig()
    )
    assert np.isclose(table.get((0, 0, 0, 0), ()), 0.5)
    assert np.isclose(
        table.get((0, 0, 0, 0), (0, 0, 0, 0)), 1.5 * variance**2
    )


def test_single_input_tensor_closed_forms():
    inputs = np.asarray([[1.3, -0.7]])
    tensors, kernels = MODULE.run_recursions(
        inputs,
        depth=7,
        weight_variance=2.0,
        config=MODULE.CubatureConfig(),
    )
    layers = np.arange(1, 8, dtype=float)
    q0 = float((inputs @ inputs.T / inputs.shape[1]).item())
    # Book normalization: K=2 q0 and Theta=q0 L.
    assert np.allclose([row["ntk"][0, 0] for row in kernels], q0 * layers)
    assert np.allclose(tensors["V"][:, 0, 0, 0, 0], 20.0 * q0**2 * (layers - 1.0))
    assert np.allclose(
        tensors["F"][:, 0, 0, 0, 0],
        2.0 * q0**2 * layers * (layers - 1.0),
    )
    assert np.allclose(
        tensors["D"][:, 0, 0, 0, 0],
        6.0 * q0**2 * layers * (layers - 1.0),
    )
    assert np.allclose(
        tensors["A"][:, 0, 0, 0, 0],
        q0**2
        * layers
        * (layers - 1.0)
        * (7.0 * layers - 2.0)
        / 3.0,
    )
    assert np.allclose(
        tensors["B"][:, 0, 0, 0, 0],
        q0**2 * layers * (layers - 1.0) * (2.0 * layers - 1.0) / 3.0,
    )
