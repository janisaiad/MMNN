from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from poc import (  # noqa: E402
    BoundedGeometryDesignPolicy,
    DesignPolicy,
    DiverseDesignPolicy,
    FixedGeometryGP,
    GeometryConfig,
    TiedKernelLoop,
    exact_krr,
    query_weights,
)


def test_kernel_geometry_is_frozen_and_gradients_reach_controller() -> None:
    cfg = GeometryConfig(grid_size=20, min_context=4, max_context=6, loop_depth=4)
    task = FixedGeometryGP(cfg, torch.device("cpu"))
    latent, observed = task.sample(3)
    indices = task.random_context(3, 5)
    x_context, y_context = task.gather_context(observed, indices)
    model = TiedKernelLoop(cfg.true_lengthscale, cfg.ridge, cfg.loop_depth, cfg.grid_size)
    prediction, _ = model(x_context, y_context, task.grid)
    (prediction - latent).square().mean().backward()

    assert model.kernel_is_frozen
    assert model.lengthscale.grad is None
    assert any(parameter.grad is not None for parameter in model.controller.parameters())


def test_more_iterations_approach_exact_krr_for_stable_fixed_coefficients() -> None:
    cfg = GeometryConfig(grid_size=20, min_context=5, max_context=5, loop_depth=4)
    task = FixedGeometryGP(cfg, torch.device("cpu"))
    _, observed = task.sample(4)
    indices = task.random_context(4, 5)
    x_context, y_context = task.gather_context(observed, indices)
    model = TiedKernelLoop(cfg.true_lengthscale, cfg.ridge, cfg.loop_depth, cfg.grid_size)
    with torch.no_grad():
        model.controller[-1].bias.copy_(torch.tensor([-0.9, -5.0]))
    exact = exact_krr(x_context, y_context, task.grid, cfg.true_lengthscale, cfg.ridge)
    shallow, _ = model(x_context, y_context, task.grid, depth=1)
    deep, _ = model(x_context, y_context, task.grid, depth=64)

    assert torch.mean((deep - exact).square()) < torch.mean((shallow - exact).square())


def test_design_is_unique_and_differentiable() -> None:
    cfg = GeometryConfig(grid_size=20, loop_depth=4)
    task = FixedGeometryGP(cfg, torch.device("cpu"))
    latent, observed = task.sample(3)
    policy = DesignPolicy(width=16)
    weights = query_weights(task.grid)
    x_context, y_context, selected, _ = policy.select(
        observed, task.grid, task.kernel, weights, budget=4, stochastic=True
    )
    model = TiedKernelLoop(cfg.true_lengthscale, cfg.ridge, cfg.loop_depth, cfg.grid_size)
    prediction, _ = model(x_context, y_context, task.grid)
    loss = ((prediction - latent).square() * weights).sum(-1).mean()
    loss.backward()

    assert all(torch.unique(row).numel() == 4 for row in selected)
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_diverse_policy_penalizes_kernel_covered_candidates() -> None:
    cfg = GeometryConfig(grid_size=20)
    task = FixedGeometryGP(cfg, torch.device("cpu"))
    policy = DiverseDesignPolicy(width=16)
    weights = query_weights(task.grid)
    coverage = task.kernel[:, 10].unsqueeze(0)
    base_logits = DesignPolicy.logits(policy, task.grid, weights, coverage, 1, 4, 1)
    diverse_logits = policy.logits(task.grid, weights, coverage, 1, 4, 1)

    penalty = base_logits - diverse_logits
    assert penalty[0, 10] > penalty[0, 0]
    assert policy.raw_repulsion.requires_grad


def test_bounded_policy_never_selects_near_duplicate_locations() -> None:
    cfg = GeometryConfig(grid_size=64)
    task = FixedGeometryGP(cfg, torch.device("cpu"))
    policy = BoundedGeometryDesignPolicy(width=16)
    _, observed = task.sample(16)
    _, _, selected, _ = policy.select(
        observed,
        task.grid,
        task.kernel,
        query_weights(task.grid),
        budget=8,
        stochastic=True,
    )
    for row in selected:
        selected_kernel = task.kernel[row][:, row].clone()
        selected_kernel.fill_diagonal_(0.0)
        assert selected_kernel.max() <= 0.85


def test_bounded_geometry_policy_cannot_overpower_fixed_repulsion() -> None:
    cfg = GeometryConfig(grid_size=64)
    task = FixedGeometryGP(cfg, torch.device("cpu"))
    policy = BoundedGeometryDesignPolicy(width=16)
    weights = query_weights(task.grid)
    coverage = task.kernel[:, 32].unsqueeze(0)
    logits = policy.logits(task.grid, weights, coverage, 1, 8, 1)

    assert logits.max() <= 2.0
    assert not any("repulsion" in name for name, _ in policy.named_parameters())
