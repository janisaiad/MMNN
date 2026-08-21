from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "feynman"
    / "exact_mmnn_ntk_recursion.py"
)
SPEC = spec_from_file_location("exact_mmnn_ntk_recursion", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def random_layers(seed: int = 0):
    rng = np.random.default_rng(seed)
    ranks = (2, 3, 2, 1)
    widths = (5, 4, 6)
    layers = []
    for input_rank, output_rank, width in zip(ranks, ranks[1:], widths):
        layers.append(
            MODULE.MMNNLayer(
                W=rng.normal(size=(width, input_rank)),
                b=0.1 * rng.normal(size=width),
                A=rng.normal(size=(output_rank, width)),
                c=0.1 * rng.normal(size=output_rank),
            )
        )
    return layers


def test_pathwise_recursion_equals_full_parameter_jacobian():
    rng = np.random.default_rng(2)
    inputs = rng.normal(size=(4, 2))
    layers = random_layers(3)
    recursive = MODULE.exact_pathwise_ntk(inputs, layers)
    explicit = MODULE.explicit_parameter_jacobian_ntk(inputs, layers)
    np.testing.assert_allclose(recursive, explicit, rtol=2e-13, atol=2e-13)


def test_one_layer_direct_kernel_is_random_feature_kernel_times_identity():
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(3, 2))
    layer = random_layers(5)[0]
    _, caches = MODULE.forward_with_cache(inputs, [layer])
    features = caches[0][2]
    expected_scalar = features @ features.T / features.shape[1] + 1.0
    expected = np.einsum("xy,ab->xyab", expected_scalar, np.eye(layer.A.shape[0]))
    np.testing.assert_allclose(MODULE.exact_pathwise_ntk(inputs, [layer]), expected)


def test_concatenated_factor_metric_is_exact_jacobian_contraction():
    rng = np.random.default_rng(6)
    W = rng.normal(size=(7, 3))
    rank = W.shape[1]
    # M_{mu,nu}=sum_a W_{mu,a} A_{a,nu}/sqrt(r).
    # Fixing the right index nu, its Jacobian with respect to A[:,nu] is W/sqrt(r).
    jacobian = W / np.sqrt(rank)
    np.testing.assert_allclose(
        jacobian @ jacobian.T, MODULE.concatenated_factor_tangent_metric(W)
    )
