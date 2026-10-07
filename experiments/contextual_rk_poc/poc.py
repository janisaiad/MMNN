#!/usr/bin/env python3
"""Low-cost proof of concept for contextual, order-constrained RK policies.

The experiment deliberately uses a scalar Riccati family with a closed-form
solution.  This keeps the teacher exact and makes the numerical effect of the
RK coefficients identifiable without also learning an ODE model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class Config:
    seed: int = 20260911
    train_size: int = 1024
    test_size: int = 4096
    train_steps: int = 1600
    n_steps: int = 8
    horizon: float = 0.8
    learning_rate: float = 0.05
    grid_size: int = 601


@dataclass
class Batch:
    a: Tensor
    y0: Tensor
    target: Tensor
    horizon: float

    def exact_path(self, n_steps: int) -> Tensor:
        times = torch.linspace(0.0, self.horizon, n_steps + 1)[:, None]
        root_a = torch.sqrt(self.a)[None, :]
        phase0 = torch.atan(self.y0[None, :] / root_a)
        return root_a * torch.tan(root_a * times + phase0)


def exact_riccati(a: Tensor, y0: Tensor, horizon: float) -> Tensor:
    """Exact solution of y' = a + y^2 for positive a, before blow-up."""
    root_a = torch.sqrt(a)
    return root_a * torch.tan(root_a * horizon + torch.atan(y0 / root_a))


def sample_batch(size: int, seed: int, horizon: float, split: str) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    if split in {"train", "iid"}:
        a_lo, a_hi = 0.1, 1.0
        y_lo, y_hi = -0.5, 0.5
    elif split == "ood":
        a_lo, a_hi = 0.02, 1.2
        y_lo, y_hi = -0.7, 0.55
    else:
        raise ValueError(f"Unknown split: {split}")

    a = a_lo + (a_hi - a_lo) * torch.rand(size, generator=generator)
    y0 = y_lo + (y_hi - y_lo) * torch.rand(size, generator=generator)
    target = exact_riccati(a, y0, horizon)
    if not torch.isfinite(target).all() or target.abs().max() >= 10.0:
        raise RuntimeError("Sampling interval crossed or approached Riccati blow-up")
    return Batch(a=a, y0=y0, target=target, horizon=horizon)


def normalized_error(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target) / (1.0 + target.abs())


def terminal_objective(prediction: Tensor, target: Tensor) -> Tensor:
    return normalized_error(prediction, target).square().mean()


def dynamic_features(y: Tensor, a: Tensor, step: int, n_steps: int) -> Tensor:
    """Four cheap, dimensionless-ish context features for the scalar toy."""
    nonlinear_fraction = y.square() / (a + y.square())
    signed_state = 2.0 * y / (1.0 + y.abs())
    time_feature = torch.full_like(y, 2.0 * step / max(1, n_steps - 1) - 1.0)
    return torch.stack(
        [torch.ones_like(y), 2.0 * nonlinear_fraction - 1.0, signed_state, time_feature]
    )


def static_features(y0: Tensor, a: Tensor) -> Tensor:
    nonlinear_fraction = y0.square() / (a + y0.square())
    signed_state = 2.0 * y0 / (1.0 + y0.abs())
    scaled_parameter = torch.tanh(2.0 * (a - 0.55))
    return torch.stack(
        [torch.ones_like(y0), 2.0 * nonlinear_fraction - 1.0, signed_state, scaled_parameter]
    )


class ContextPolicy(nn.Module):
    """Affine-sigmoid policy; only four trainable scalars."""

    def __init__(self, alpha_lo: float, alpha_hi: float, mode: str) -> None:
        super().__init__()
        if mode not in {"static", "dynamic"}:
            raise ValueError("mode must be 'static' or 'dynamic'")
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.mode = mode
        self.weights = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        y: Tensor,
        a: Tensor,
        step: int,
        n_steps: int,
        y0: Tensor,
        context_y0: Tensor | None = None,
        context_a: Tensor | None = None,
    ) -> Tensor:
        if self.mode == "dynamic":
            features = dynamic_features(y, a, step, n_steps)
        else:
            ctx_y0 = y0 if context_y0 is None else context_y0
            ctx_a = a if context_a is None else context_a
            features = static_features(ctx_y0, ctx_a)
        logits = (self.weights[:, None] * features).sum(dim=0)
        return self.alpha_lo + (self.alpha_hi - self.alpha_lo) * torch.sigmoid(logits)


class OpenLoopPolicy(nn.Module):
    """One learned alpha per time step, shared by all tasks (S4S-like ablation)."""

    def __init__(self, n_steps: int, alpha_lo: float, alpha_hi: float) -> None:
        super().__init__()
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.logits = nn.Parameter(torch.zeros(n_steps))

    def forward(
        self,
        y: Tensor,
        a: Tensor,
        step: int,
        n_steps: int,
        y0: Tensor,
        context_y0: Tensor | None = None,
        context_a: Tensor | None = None,
    ) -> Tensor:
        del a, n_steps, y0, context_y0, context_a
        alpha = self.alpha_lo + (self.alpha_hi - self.alpha_lo) * torch.sigmoid(
            self.logits[step]
        )
        return alpha.expand_as(y)


Chooser = Callable[[Tensor, Tensor, int, int, Tensor, Tensor | None, Tensor | None], Tensor]


def constant_policy(alpha: float) -> Chooser:
    def choose(
        y: Tensor,
        a: Tensor,
        step: int,
        n_steps: int,
        y0: Tensor,
        context_y0: Tensor | None = None,
        context_a: Tensor | None = None,
    ) -> Tensor:
        del a, step, n_steps, y0, context_y0, context_a
        return torch.full_like(y, alpha)

    return choose


def analytic_local_cancellation_policy(
    y: Tensor,
    a: Tensor,
    step: int,
    n_steps: int,
    y0: Tensor,
    context_y0: Tensor | None = None,
    context_a: Tensor | None = None,
) -> Tensor:
    """Cancel Riccati's third-order local-error term analytically.

    For f(y)=a+y^2, matching the h^3 Taylor coefficient gives
    alpha=2/3+(4/3)y^2/(a+y^2).  This highly problem-specific reference also
    exposes how favorable the learned policy features are.
    """
    del step, n_steps, y0, context_y0, context_a
    nonlinear_fraction = y.square() / (a + y.square())
    return 2.0 / 3.0 + 4.0 * nonlinear_fraction / 3.0


def vector_policy(alpha: Tensor) -> Chooser:
    def choose(
        y: Tensor,
        a: Tensor,
        step: int,
        n_steps: int,
        y0: Tensor,
        context_y0: Tensor | None = None,
        context_a: Tensor | None = None,
    ) -> Tensor:
        del y, a, step, n_steps, y0, context_y0, context_a
        return alpha

    return choose


def rollout_rk2(
    batch: Batch,
    n_steps: int,
    chooser: Chooser,
    *,
    context_y0: Tensor | None = None,
    context_a: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Order-two family parameterized by alpha at every step.

    a21=c2=alpha, b2=1/(2 alpha), b1=1-b2.  Thus b^T 1=1
    and b^T c=1/2 identically for every policy output.
    """
    y = batch.y0
    h = batch.horizon / n_steps
    trajectory = [y]
    controls = []
    for step in range(n_steps):
        alpha = chooser(y, batch.a, step, n_steps, batch.y0, context_y0, context_a)
        k1 = batch.a + y.square()
        stage = y + alpha * h * k1
        k2 = batch.a + stage.square()
        b2 = 1.0 / (2.0 * alpha)
        y = y + h * ((1.0 - b2) * k1 + b2 * k2)
        trajectory.append(y)
        controls.append(alpha)
    return y, torch.stack(trajectory), torch.stack(controls)


def rollout_rk4(batch: Batch, n_steps: int) -> tuple[Tensor, Tensor]:
    y = batch.y0
    h = batch.horizon / n_steps
    trajectory = [y]
    for _ in range(n_steps):
        k1 = batch.a + y.square()
        k2 = batch.a + (y + 0.5 * h * k1).square()
        k3 = batch.a + (y + 0.5 * h * k2).square()
        k4 = batch.a + (y + h * k3).square()
        y = y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        trajectory.append(y)
    return y, torch.stack(trajectory)


def train_policy(
    policy: nn.Module, batch: Batch, config: Config
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    checkpoints = {0, config.train_steps // 4, config.train_steps // 2, 3 * config.train_steps // 4, config.train_steps - 1}
    history: list[dict[str, float]] = []
    for iteration in range(config.train_steps):
        prediction, _, _ = rollout_rk2(batch, config.n_steps, policy)
        data_loss = terminal_objective(prediction, batch.target)
        regularizer = sum(parameter.square().sum() for parameter in policy.parameters())
        loss = data_loss + 1.0e-12 * regularizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration in checkpoints:
            history.append(
                {
                    "iteration": iteration,
                    "objective": float(data_loss.detach()),
                    "regularized_loss": float(loss.detach()),
                }
            )
    return policy.eval(), history


@torch.no_grad()
def coefficient_grid(
    batch: Batch, n_steps: int, alpha_lo: float, alpha_hi: float, grid_size: int
) -> tuple[float, Tensor, Tensor, Tensor]:
    """Return best population alpha and the per-task oracle on the same grid."""
    grid = torch.linspace(alpha_lo, alpha_hi, grid_size)
    y = batch.y0[None, :].expand(grid_size, -1).clone()
    a = batch.a[None, :]
    h = batch.horizon / n_steps
    alpha = grid[:, None]
    for _ in range(n_steps):
        k1 = a + y.square()
        stage = y + alpha * h * k1
        k2 = a + stage.square()
        b2 = 1.0 / (2.0 * alpha)
        y = y + h * ((1.0 - b2) * k1 + b2 * k2)
    squared_errors = normalized_error(y, batch.target[None, :]).square()
    population_index = int(squared_errors.mean(dim=1).argmin())
    oracle_errors, oracle_indices = squared_errors.min(dim=0)
    oracle_alphas = grid[oracle_indices]
    oracle_predictions = y.gather(0, oracle_indices[None, :]).squeeze(0)
    return float(grid[population_index]), oracle_alphas, oracle_predictions, oracle_errors


@torch.no_grad()
def method_metrics(
    name: str,
    split: str,
    batch: Batch,
    prediction: Tensor,
    trajectory: Tensor,
    nfe: int,
    controls: Tensor | None,
) -> dict[str, float | str | int]:
    absolute = (prediction - batch.target).abs()
    normalized = normalized_error(prediction, batch.target)
    exact_path = batch.exact_path(trajectory.shape[0] - 1)
    normalized_path = normalized_error(trajectory, exact_path)
    # Compare all 16-NFE methods on the same five physical times.  Native
    # trajectories have nine points for RK2 and five for RK4.
    native_steps = trajectory.shape[0] - 1
    if native_steps % 4 != 0:
        raise ValueError("Path metric expects a step count divisible by four")
    common_indices = torch.arange(0, native_steps + 1, native_steps // 4)
    common_trajectory = trajectory[common_indices]
    common_exact_path = batch.exact_path(4)
    common_normalized_path = normalized_error(common_trajectory, common_exact_path)
    result: dict[str, float | str | int] = {
        "method": name,
        "split": split,
        "nfe": nfe,
        "terminal_nmse": float(normalized.square().mean()),
        "terminal_nrmse": float(normalized.square().mean().sqrt()),
        "terminal_mae": float(absolute.mean()),
        "terminal_median_abs": float(absolute.median()),
        "terminal_p90_abs": float(torch.quantile(absolute, 0.9)),
        "terminal_max_abs": float(absolute.max()),
        "path_nrmse": float(normalized_path.square().mean().sqrt()),
        "path_common_nrmse": float(common_normalized_path.square().mean().sqrt()),
    }
    if controls is not None:
        result.update(
            {
                "alpha_mean": float(controls.mean()),
                "alpha_min": float(controls.min()),
                "alpha_max": float(controls.max()),
            }
        )
    else:
        result.update({"alpha_mean": math.nan, "alpha_min": math.nan, "alpha_max": math.nan})
    return result


def manual_adjoint_check() -> dict[str, float]:
    """Check a hand-derived discrete adjoint against autograd and differences."""
    a = 0.7
    y0 = 0.2
    horizon = 0.8
    n_steps = 8
    h = horizon / n_steps
    target = math.sqrt(a) * math.tan(
        math.sqrt(a) * horizon + math.atan(y0 / math.sqrt(a))
    )
    controls = [0.72, 0.81, 0.95, 1.10, 1.25, 1.38, 1.52, 1.66]

    states: list[float] = [y0]
    stages: list[float] = []
    for alpha in controls:
        y = states[-1]
        k1 = a + y * y
        stage = y + alpha * h * k1
        k2 = a + stage * stage
        y_next = y + h * (k1 + (k2 - k1) / (2.0 * alpha))
        stages.append(stage)
        states.append(y_next)

    scale = 1.0 + abs(target)
    costate = (states[-1] - target) / (scale * scale)
    manual = [0.0] * n_steps
    for step in reversed(range(n_steps)):
        y = states[step]
        stage = stages[step]
        alpha = controls[step]
        k1 = a + y * y
        k2 = a + stage * stage
        fprime_y = 2.0 * y
        fprime_stage = 2.0 * stage
        dflow_dalpha = 0.5 * h * (
            -(k2 - k1) / (alpha * alpha) + h * k1 * fprime_stage / alpha
        )
        dflow_dy = 1.0 + h * fprime_y + h / (2.0 * alpha) * (
            fprime_stage * (1.0 + alpha * h * fprime_y) - fprime_y
        )
        manual[step] = costate * dflow_dalpha
        costate *= dflow_dy

    alpha_tensor = torch.tensor(controls, requires_grad=True)
    y_tensor = torch.tensor(y0)
    for alpha in alpha_tensor:
        k1 = a + y_tensor.square()
        stage = y_tensor + alpha * h * k1
        k2 = a + stage.square()
        y_tensor = y_tensor + h * (k1 + (k2 - k1) / (2.0 * alpha))
    loss = 0.5 * ((y_tensor - target) / scale).square()
    loss.backward()
    automatic = alpha_tensor.grad.detach()
    manual_tensor = torch.tensor(manual)

    def scalar_loss(values: list[float]) -> float:
        y = y0
        for alpha in values:
            k1 = a + y * y
            stage = y + alpha * h * k1
            k2 = a + stage * stage
            y = y + h * (k1 + (k2 - k1) / (2.0 * alpha))
        return 0.5 * ((y - target) / scale) ** 2

    epsilon = 1.0e-6
    finite_difference = []
    for index in range(n_steps):
        plus = controls.copy()
        minus = controls.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference.append((scalar_loss(plus) - scalar_loss(minus)) / (2.0 * epsilon))
    finite_tensor = torch.tensor(finite_difference)
    denominator = max(float(automatic.abs().max()), 1.0e-15)
    return {
        "manual_vs_autograd_max_abs": float((manual_tensor - automatic).abs().max()),
        "manual_vs_autograd_max_rel": float((manual_tensor - automatic).abs().max()) / denominator,
        "finite_difference_vs_autograd_max_abs": float((finite_tensor - automatic).abs().max()),
        "autograd_gradient_max_abs": denominator,
    }


def closed_loop_adjoint_check() -> dict[str, float]:
    """Check the discrete adjoint of a four-weight dynamic feedback policy."""
    a = 0.7
    y0 = 0.2
    horizon = 0.8
    n_steps = 8
    h = horizon / n_steps
    alpha_lo = 0.5
    alpha_hi = 2.0
    weights = [-0.2, 0.8, -0.4, 0.6]
    target = math.sqrt(a) * math.tan(
        math.sqrt(a) * horizon + math.atan(y0 / math.sqrt(a))
    )

    states: list[float] = [y0]
    stages: list[float] = []
    alphas: list[float] = []
    policy_state_derivatives: list[float] = []
    policy_weight_derivatives: list[list[float]] = []
    for step in range(n_steps):
        y = states[-1]
        denominator = a + y * y
        features = [
            1.0,
            2.0 * y * y / denominator - 1.0,
            2.0 * y / (1.0 + abs(y)),
            2.0 * step / (n_steps - 1) - 1.0,
        ]
        logit = sum(weight * feature for weight, feature in zip(weights, features))
        sigmoid = 1.0 / (1.0 + math.exp(-logit))
        sigmoid_derivative = (alpha_hi - alpha_lo) * sigmoid * (1.0 - sigmoid)
        alpha = alpha_lo + (alpha_hi - alpha_lo) * sigmoid
        feature_state_derivatives = [
            0.0,
            4.0 * a * y / (denominator * denominator),
            2.0 / (1.0 + abs(y)) ** 2,
            0.0,
        ]
        policy_state_derivatives.append(
            sigmoid_derivative
            * sum(
                weight * derivative
                for weight, derivative in zip(weights, feature_state_derivatives)
            )
        )
        policy_weight_derivatives.append(
            [sigmoid_derivative * feature for feature in features]
        )

        k1 = a + y * y
        stage = y + alpha * h * k1
        k2 = a + stage * stage
        states.append(y + h * (k1 + (k2 - k1) / (2.0 * alpha)))
        stages.append(stage)
        alphas.append(alpha)

    scale = 1.0 + abs(target)
    costate = (states[-1] - target) / (scale * scale)
    manual = [0.0] * len(weights)
    feedback_terms: list[float] = []
    for step in reversed(range(n_steps)):
        y = states[step]
        stage = stages[step]
        alpha = alphas[step]
        k1 = a + y * y
        k2 = a + stage * stage
        fprime_y = 2.0 * y
        fprime_stage = 2.0 * stage
        dflow_dalpha = 0.5 * h * (
            -(k2 - k1) / (alpha * alpha) + h * k1 * fprime_stage / alpha
        )
        dflow_dy_fixed_alpha = 1.0 + h * fprime_y + h / (2.0 * alpha) * (
            fprime_stage * (1.0 + alpha * h * fprime_y) - fprime_y
        )
        for index, dalpha_dweight in enumerate(policy_weight_derivatives[step]):
            manual[index] += costate * dflow_dalpha * dalpha_dweight
        feedback_term = dflow_dalpha * policy_state_derivatives[step]
        feedback_terms.append(feedback_term)
        costate *= dflow_dy_fixed_alpha + feedback_term

    weight_tensor = torch.tensor(weights, requires_grad=True)
    y_tensor = torch.tensor(y0)
    a_tensor = torch.tensor(a)
    for step in range(n_steps):
        features = dynamic_features(y_tensor, a_tensor, step, n_steps)
        sigmoid = torch.sigmoid((weight_tensor * features).sum())
        alpha = alpha_lo + (alpha_hi - alpha_lo) * sigmoid
        k1 = a + y_tensor.square()
        stage = y_tensor + alpha * h * k1
        k2 = a + stage.square()
        y_tensor = y_tensor + h * (k1 + (k2 - k1) / (2.0 * alpha))
    loss = 0.5 * ((y_tensor - target) / scale).square()
    loss.backward()
    automatic = weight_tensor.grad.detach()
    manual_tensor = torch.tensor(manual)

    def scalar_loss(values: list[float]) -> float:
        y = y0
        for step in range(n_steps):
            denominator = a + y * y
            features = [
                1.0,
                2.0 * y * y / denominator - 1.0,
                2.0 * y / (1.0 + abs(y)),
                2.0 * step / (n_steps - 1) - 1.0,
            ]
            logit = sum(value * feature for value, feature in zip(values, features))
            sigmoid = 1.0 / (1.0 + math.exp(-logit))
            alpha = alpha_lo + (alpha_hi - alpha_lo) * sigmoid
            k1 = a + y * y
            stage = y + alpha * h * k1
            k2 = a + stage * stage
            y = y + h * (k1 + (k2 - k1) / (2.0 * alpha))
        return 0.5 * ((y - target) / scale) ** 2

    epsilon = 1.0e-6
    finite_difference = []
    for index in range(len(weights)):
        plus = weights.copy()
        minus = weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference.append((scalar_loss(plus) - scalar_loss(minus)) / (2.0 * epsilon))
    finite_tensor = torch.tensor(finite_difference)
    denominator = max(float(automatic.abs().max()), 1.0e-15)
    return {
        "manual_vs_autograd_max_abs": float((manual_tensor - automatic).abs().max()),
        "manual_vs_autograd_max_rel": float((manual_tensor - automatic).abs().max()) / denominator,
        "finite_difference_vs_autograd_max_abs": float((finite_tensor - automatic).abs().max()),
        "finite_difference_vs_autograd_max_rel": float(
            (finite_tensor - automatic).abs().max()
        )
        / denominator,
        "autograd_gradient_max_abs": denominator,
        "feedback_jacobian_term_max_abs": max(abs(value) for value in feedback_terms),
    }


@torch.no_grad()
def linear_negative_control(seed: int = 44) -> dict[str, float]:
    """All order-two two-stage members have the same linear stability polynomial."""
    generator = torch.Generator().manual_seed(seed)
    size = 512
    lam = -4.0 + 3.9 * torch.rand(size, generator=generator)
    y0 = -1.0 + 2.0 * torch.rand(size, generator=generator)
    h = 0.05
    n_steps = 10
    baseline = y0.clone()
    varied = y0.clone()
    for step in range(n_steps):
        alpha0 = torch.full_like(y0, 0.5)
        alpha1 = 0.5 + 1.5 * torch.rand(size, generator=generator)
        for alpha, state_name in [(alpha0, "baseline"), (alpha1, "varied")]:
            state = baseline if state_name == "baseline" else varied
            k1 = lam * state
            k2 = lam * (state + alpha * h * k1)
            b2 = 1.0 / (2.0 * alpha)
            updated = state + h * ((1.0 - b2) * k1 + b2 * k2)
            if state_name == "baseline":
                baseline = updated
            else:
                varied = updated
    return {
        "max_pairwise_difference": float((baseline - varied).abs().max()),
        "mean_pairwise_difference": float((baseline - varied).abs().mean()),
    }


def loglog_slope(step_counts: list[int], errors: list[float], tail: int = 3) -> float:
    xs = [math.log(1.0 / n) for n in step_counts[-tail:]]
    ys = [math.log(error) for error in errors[-tail:]]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    denominator = sum((x - xbar) ** 2 for x in xs)
    return numerator / denominator


def save_figure(
    metrics: list[dict[str, float | str | int]],
    convergence: dict[str, dict[str, list[float] | float]],
    output_path: Path,
) -> None:
    iid = {row["method"]: row for row in metrics if row["split"] == "iid"}
    bar_methods = [
        "midpoint_rk2",
        "ralston_rk2",
        "heun_rk2",
        "best_constant",
        "analytic_local_cancel",
        "open_loop",
        "static_context",
        "dynamic_context",
        "rk4_compute_matched",
    ]
    labels = [
        "midpoint",
        "Ralston",
        "Heun",
        "best const.",
        "analytic local",
        "open-loop",
        "static ctx",
        "dynamic ctx",
        "RK4, same NFE",
    ]
    values = [float(iid[name]["terminal_nmse"]) for name in bar_methods]

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].bar(
        range(len(values)),
        values,
        color=["#777777"] * 4
        + ["#e9c46a", "#d98c35", "#4c78a8", "#2a9d8f", "#8e5ea2"],
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(labels)), labels, rotation=38, ha="right")
    axes[0].set_ylabel("terminal normalized MSE")
    axes[0].set_title("IID test, 16 RHS evaluations")
    axes[0].grid(axis="y", which="both", alpha=0.25)

    for name, color, marker in [
        ("best_constant", "#777777", "o"),
        ("analytic_local_cancel", "#e0a100", "D"),
        ("static_context", "#4c78a8", "s"),
        ("dynamic_context", "#2a9d8f", "^"),
    ]:
        entry = convergence[name]
        axes[1].loglog(
            entry["n_steps"], entry["terminal_mae"], marker=marker, color=color, label=f"{name} (slope {entry['slope']:.2f})"
        )
    axes[1].set_xlabel("number of RK2 steps")
    axes[1].set_ylabel("terminal MAE")
    axes[1].set_title("Convergence / transfer (learned policies trained at N=8)")
    axes[1].grid(which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--train-size", type=int, default=Config.train_size)
    parser.add_argument("--test-size", type=int, default=Config.test_size)
    parser.add_argument("--train-steps", type=int, default=Config.train_steps)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
        train_steps=args.train_steps,
    )
    start_time = time.perf_counter()
    torch.manual_seed(config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = sample_batch(config.train_size, config.seed, config.horizon, "train")
    iid = sample_batch(config.test_size, config.seed + 1, config.horizon, "iid")
    ood = sample_batch(config.test_size, config.seed + 2, config.horizon, "ood")
    outside_mask = (
        (ood.a < 0.1)
        | (ood.a > 1.0)
        | (ood.y0 < -0.5)
        | (ood.y0 > 0.5)
    )
    outside = Batch(
        a=ood.a[outside_mask],
        y0=ood.y0[outside_mask],
        target=ood.target[outside_mask],
        horizon=ood.horizon,
    )

    best_alpha, _, _, _ = coefficient_grid(
        train, config.n_steps, 0.5, 2.0, config.grid_size
    )
    best_alpha_strict, _, _, _ = coefficient_grid(
        train, config.n_steps, 0.5, 1.0, config.grid_size
    )

    policies: dict[str, nn.Module] = {
        "open_loop": OpenLoopPolicy(config.n_steps, 0.5, 2.0),
        "static_context": ContextPolicy(0.5, 2.0, "static"),
        "dynamic_context": ContextPolicy(0.5, 2.0, "dynamic"),
        "dynamic_context_strict": ContextPolicy(0.5, 1.0, "dynamic"),
    }
    histories: dict[str, list[dict[str, float]]] = {}
    for name, policy in policies.items():
        policies[name], histories[name] = train_policy(policy, train, config)

    metrics: list[dict[str, float | str | int]] = []
    all_order_controls: list[Tensor] = []
    for split_name, batch in [
        ("iid", iid),
        ("expanded_box", ood),
        ("outside_train_support", outside),
    ]:
        methods: list[tuple[str, Chooser, int]] = [
            ("midpoint_rk2", constant_policy(0.5), 2 * config.n_steps),
            ("ralston_rk2", constant_policy(2.0 / 3.0), 2 * config.n_steps),
            ("heun_rk2", constant_policy(1.0), 2 * config.n_steps),
            ("best_constant", constant_policy(best_alpha), 2 * config.n_steps),
            ("best_constant_strict", constant_policy(best_alpha_strict), 2 * config.n_steps),
            (
                "analytic_local_cancel",
                analytic_local_cancellation_policy,
                2 * config.n_steps,
            ),
            ("open_loop", policies["open_loop"], 2 * config.n_steps),
            ("static_context", policies["static_context"], 2 * config.n_steps),
            ("dynamic_context", policies["dynamic_context"], 2 * config.n_steps),
            ("dynamic_context_strict", policies["dynamic_context_strict"], 2 * config.n_steps),
        ]
        for name, chooser, nfe in methods:
            prediction, trajectory, controls = rollout_rk2(batch, config.n_steps, chooser)
            metrics.append(
                method_metrics(name, split_name, batch, prediction, trajectory, nfe, controls)
            )
            all_order_controls.append(controls.detach())

        permutation = torch.randperm(
            batch.a.numel(), generator=torch.Generator().manual_seed(config.seed + 99)
        )
        prediction, trajectory, controls = rollout_rk2(
            batch,
            config.n_steps,
            policies["static_context"],
            context_y0=batch.y0[permutation],
            context_a=batch.a[permutation],
        )
        metrics.append(
            method_metrics(
                "static_context_shuffled",
                split_name,
                batch,
                prediction,
                trajectory,
                2 * config.n_steps,
                controls,
            )
        )

        _, oracle_alphas, oracle_prediction, _ = coefficient_grid(
            batch, config.n_steps, 0.5, 2.0, config.grid_size
        )
        oracle_prediction_2, oracle_trajectory, oracle_controls = rollout_rk2(
            batch, config.n_steps, vector_policy(oracle_alphas)
        )
        if not torch.allclose(oracle_prediction, oracle_prediction_2):
            raise RuntimeError("Oracle grid and rollout disagree")
        metrics.append(
            method_metrics(
                "oracle_per_task_constant",
                split_name,
                batch,
                oracle_prediction,
                oracle_trajectory,
                2 * config.n_steps,
                oracle_controls,
            )
        )

        rk4_prediction, rk4_trajectory = rollout_rk4(batch, config.n_steps // 2)
        metrics.append(
            method_metrics(
                "rk4_compute_matched",
                split_name,
                batch,
                rk4_prediction,
                rk4_trajectory,
                4 * (config.n_steps // 2),
                None,
            )
        )

    controls_flat = torch.cat([control.reshape(-1) for control in all_order_controls])
    b2 = 1.0 / (2.0 * controls_flat)
    b1 = 1.0 - b2
    order_checks = {
        "max_abs_b1_plus_b2_minus_1": float((b1 + b2 - 1.0).abs().max()),
        "max_abs_b2_c2_minus_half": float((b2 * controls_flat - 0.5).abs().max()),
        "max_abs_a21_minus_c2": 0.0,
        "all_weights_nonnegative": bool(((b1 >= 0.0) & (b2 >= 0.0)).all()),
    }

    _, iid_oracle_alphas, _, iid_oracle_errors = coefficient_grid(
        iid, config.n_steps, 0.5, 2.0, config.grid_size
    )
    oracle_summary = {
        "terminal_nmse": float(iid_oracle_errors.mean()),
        "alpha_quantiles": {
            str(q): float(torch.quantile(iid_oracle_alphas, q))
            for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        },
    }

    # The learned controllers are deliberately optimized at N=8, so pre-asymptotic
    # cancellation can make N=16 look worse.  Continue far enough to estimate the
    # actual asymptotic order instead of fitting only that transition region.
    convergence_counts = [8, 16, 32, 64, 128, 256, 512, 1024]
    convergence: dict[str, dict[str, list[float] | float]] = {}
    convergence_methods: dict[str, Chooser] = {
        "best_constant": constant_policy(best_alpha),
        "analytic_local_cancel": analytic_local_cancellation_policy,
        "static_context": policies["static_context"],
        "dynamic_context": policies["dynamic_context"],
    }
    with torch.no_grad():
        for name, chooser in convergence_methods.items():
            errors = []
            for count in convergence_counts:
                prediction, _, _ = rollout_rk2(iid, count, chooser)
                errors.append(float((prediction - iid.target).abs().mean()))
            convergence[name] = {
                "n_steps": convergence_counts,
                "terminal_mae": errors,
                "slope": loglog_slope(convergence_counts, errors),
            }

    iid_by_name = {row["method"]: row for row in metrics if row["split"] == "iid"}
    outside_by_name = {
        row["method"]: row
        for row in metrics
        if row["split"] == "outside_train_support"
    }
    with torch.no_grad():
        dynamic_prediction, _, dynamic_controls = rollout_rk2(
            iid, config.n_steps, policies["dynamic_context"]
        )
        constant_prediction, _, _ = rollout_rk2(
            iid, config.n_steps, constant_policy(best_alpha)
        )
        matched_rk4_prediction, _ = rollout_rk4(iid, config.n_steps // 2)
        dynamic_absolute = (dynamic_prediction - iid.target).abs()
        constant_absolute = (constant_prediction - iid.target).abs()
        matched_rk4_absolute = (matched_rk4_prediction - iid.target).abs()
    paired_comparisons = {
        "dynamic_beats_best_constant_abs_fraction": float(
            (dynamic_absolute < constant_absolute).double().mean()
        ),
        "dynamic_beats_matched_rk4_abs_fraction": float(
            (dynamic_absolute < matched_rk4_absolute).double().mean()
        ),
        "dynamic_alpha_above_one_fraction": float(
            (dynamic_controls > 1.0).double().mean()
        ),
        "rk4_to_dynamic_median_abs_ratio": float(
            matched_rk4_absolute.median() / dynamic_absolute.median()
        ),
    }
    headline = {
        "best_alpha_train": best_alpha,
        "best_alpha_strict_train": best_alpha_strict,
        "nmse_gain_dynamic_vs_best_constant": float(iid_by_name["best_constant"]["terminal_nmse"])
        / float(iid_by_name["dynamic_context"]["terminal_nmse"]),
        "nmse_gain_dynamic_vs_open_loop": float(iid_by_name["open_loop"]["terminal_nmse"])
        / float(iid_by_name["dynamic_context"]["terminal_nmse"]),
        "rk4_nmse_over_dynamic_nmse": float(iid_by_name["rk4_compute_matched"]["terminal_nmse"])
        / float(iid_by_name["dynamic_context"]["terminal_nmse"]),
        "nmse_gain_static_vs_best_constant": float(iid_by_name["best_constant"]["terminal_nmse"])
        / float(iid_by_name["static_context"]["terminal_nmse"]),
        "nmse_gain_strict_dynamic_vs_strict_constant": float(iid_by_name["best_constant_strict"]["terminal_nmse"])
        / float(iid_by_name["dynamic_context_strict"]["terminal_nmse"]),
        "outside_support_nmse_gain_dynamic_vs_best_constant": float(
            outside_by_name["best_constant"]["terminal_nmse"]
        )
        / float(outside_by_name["dynamic_context"]["terminal_nmse"]),
        "outside_support_rk4_nmse_over_dynamic_nmse": float(
            outside_by_name["rk4_compute_matched"]["terminal_nmse"]
        )
        / float(outside_by_name["dynamic_context"]["terminal_nmse"]),
        "shuffle_penalty_static_nmse": float(iid_by_name["static_context_shuffled"]["terminal_nmse"])
        / float(iid_by_name["static_context"]["terminal_nmse"]),
    }

    summary = {
        "config": asdict(config),
        "problem": "scalar Riccati family y'=a+y^2 with exact terminal teacher",
        "split_sizes": {
            "train": int(train.a.numel()),
            "iid": int(iid.a.numel()),
            "expanded_box": int(ood.a.numel()),
            "expanded_box_inside_train_support": int((~outside_mask).sum()),
            "outside_train_support": int(outside.a.numel()),
        },
        "rk2_parameterization": {
            "a21": "alpha",
            "c2": "alpha",
            "b1": "1 - 1/(2 alpha)",
            "b2": "1/(2 alpha)",
            "relaxed_alpha_interval": [0.5, 2.0],
            "strict_internal_stage_interval": [0.5, 1.0],
        },
        "headline": headline,
        "paired_comparisons": paired_comparisons,
        "policy_parameters": {
            name: [float(value) for value in policy.parameters() for value in value.detach().reshape(-1)]
            for name, policy in policies.items()
        },
        "training_history": histories,
        "oracle": oracle_summary,
        "order_checks": order_checks,
        "adjoint_check": manual_adjoint_check(),
        "closed_loop_adjoint_check": closed_loop_adjoint_check(),
        "linear_negative_control": linear_negative_control(),
        "convergence": convergence,
        "runtime_seconds": time.perf_counter() - start_time,
    }

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    json_path = args.output_dir / "summary.json"
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    save_figure(metrics, convergence, args.output_dir / "poc_summary.png")

    print(json.dumps(summary["headline"], indent=2, sort_keys=True))
    print(f"results: {args.output_dir}")
    print(f"runtime_seconds: {summary['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
