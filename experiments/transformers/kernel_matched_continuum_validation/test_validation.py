"""Fast regression tests for the validation machinery."""

from __future__ import annotations

import numpy as np
import torch

from .common import (
    geometric_midpoint_grid,
    orthonormalize,
    rbf_kernel,
    ritz_inverse_metric,
    symmetric_preconditioned_spectrum,
)
from .run_pde_validation import (
    TorchMetric,
    apply_hessian,
    elliptic_matrix,
    solve_hb,
)


def test_quadrature_weighted_softmax_is_exact() -> None:
    nodes, weights = geometric_midpoint_grid(48)
    kernel = rbf_kernel(nodes, nodes, 0.15)
    expected = kernel * weights[None, :]
    expected /= expected.sum(axis=1, keepdims=True)
    logits = -0.5 * ((nodes[:, None] - nodes[None, :]) / 0.15) ** 2
    logits += np.log(weights)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    observed = np.exp(logits)
    observed /= observed.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-14)


def test_ritz_metric_is_spd_and_improves_spectrum() -> None:
    rng = np.random.default_rng(12)
    dimension, rank = 40, 6
    eigenvectors = orthonormalize(rng.normal(size=(dimension, dimension)))
    update = np.zeros(dimension)
    update[:rank] = np.geomspace(30.0, 2.0, rank)
    hessian = np.eye(dimension) + (eigenvectors * update[None, :]) @ eigenvectors.T
    metric = ritz_inverse_metric(hessian, eigenvectors[:, :rank])
    assert np.linalg.eigvalsh(metric)[0] > 0.0
    spectrum = symmetric_preconditioned_spectrum(hessian, metric)
    np.testing.assert_allclose(spectrum, np.ones_like(spectrum), rtol=1e-11, atol=1e-11)


def test_variable_coefficient_elliptic_matrix_is_spd() -> None:
    coefficient = np.exp(
        0.3
        * np.sin(np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False))[:, None]
        * np.cos(np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False))[None, :]
    )
    matrix = elliptic_matrix(coefficient)
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=1e-13)
    assert np.linalg.eigvalsh(matrix.toarray())[0] > 0.0


def test_heavy_ball_solves_low_rank_posterior() -> None:
    generator = torch.Generator().manual_seed(23)
    dimension, context = 64, 7
    sensitivity = torch.randn(
        dimension, context, generator=generator, dtype=torch.float64
    ) / 4.0
    rhs = torch.randn(dimension, 3, generator=generator, dtype=torch.float64)
    gram = torch.eye(context, dtype=torch.float64) + sensitivity.T @ sensitivity
    exact = rhs - sensitivity @ torch.linalg.solve(gram, sensitivity.T @ rhs)
    maximum = 1.0 + float(torch.linalg.eigvalsh(sensitivity.T @ sensitivity)[-1])
    solution = solve_hb(
        sensitivity,
        rhs,
        TorchMetric(name="identity"),
        depth=120,
        mu=1.0,
        ell=maximum,
    )
    relative = torch.linalg.vector_norm(
        rhs - apply_hessian(sensitivity, solution)
    ) / torch.linalg.vector_norm(rhs)
    assert float(relative) < 1e-9
    torch.testing.assert_close(solution, exact, rtol=1e-8, atol=1e-9)
