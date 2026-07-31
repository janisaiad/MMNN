"""Exact recurrent solver cells for prompt-conditioned inverse problems.

The learned component lives outside this module and produces one fixed SPD
inverse preconditioner ``B = P_theta(context)``.  This module contains only
known solver algebra: prompt matrix-vector products, scalar contractions,
safe quotients, and routed state updates.  In particular, it contains no MLP
and no learned arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Protocol, Tuple

import torch

Tensor = torch.Tensor
HVP = Callable[[Tensor], Tensor]


class AppliedPreconditioner(Protocol):
    def apply(self, vector: Tensor) -> Tensor: ...


Preconditioner = Tensor | AppliedPreconditioner | Callable[[Tensor], Tensor]


def fixed_prompt_linear_attention_hvp(
    equations: Tensor,
    vector: Tensor,
    noise_precision: float,
    prior_precision: float,
) -> Tensor:
    """Apply ``(tau G^T G + lambda I) vector`` without materializing ``H``.

    The two contractions are the fixed linear-attention part of both
    decoders: equation tokens first receive ``<g_i, vector>`` and are then
    summed with values ``g_i``.  All projection weights are fixed.
    """

    token_scores = torch.einsum("bmk,bk->bm", equations, vector)
    prompt_value_sum = torch.einsum("bmk,bm->bk", equations, token_scores)
    return noise_precision * prompt_value_sum + prior_precision * vector


def apply_fixed_preconditioner(preconditioner: Preconditioner, vector: Tensor) -> Tensor:
    """Route a vector through the prompt-conditioned but iteration-fixed map."""

    if hasattr(preconditioner, "apply"):
        return preconditioner.apply(vector)
    if callable(preconditioner):
        return preconditioner(vector)
    return torch.einsum("bkl,bl->bk", preconditioner, vector)


def batch_inner(left: Tensor, right: Tensor) -> Tensor:
    """Fixed batchwise scalar-reduction primitive."""

    return torch.einsum("bk,bk->b", left, right)


def _batch_scalar(value: float | Tensor, reference: Tensor) -> Tensor:
    scalar = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if scalar.ndim == 0:
        return scalar.expand(reference.shape[0])
    if scalar.shape != (reference.shape[0],):
        raise ValueError(f"expected scalar or batch vector, got {tuple(scalar.shape)}")
    return scalar


def safe_positive_quotient(
    numerator: Tensor,
    denominator: Tensor,
    active: Tensor,
    eps: float,
) -> Tensor:
    """Exact quotient on active SPD states and zero after convergence.

    In exact arithmetic an unconverged PCG state has a positive denominator.
    The clamp and mask only define finite-precision behavior at convergence;
    they are fixed operations, not learned approximations.
    """

    return torch.where(
        active,
        numerator / denominator.clamp_min(eps),
        torch.zeros_like(numerator),
    )


@dataclass(frozen=True)
class HeavyBallState:
    """Persistent tokens of the tied Heavy-Ball macro-block."""

    x: Tensor
    x_previous: Tensor


@dataclass(frozen=True)
class HeavyBallStep:
    """Persistent next state and inspectable routed work tokens."""

    state: HeavyBallState
    residual: Tensor
    preconditioned_residual: Tensor


def initialize_heavy_ball(rhs: Tensor) -> HeavyBallState:
    zero = torch.zeros_like(rhs)
    return HeavyBallState(x=zero, x_previous=zero.clone())


def heavy_ball_macro_block(
    state: HeavyBallState,
    rhs: Tensor,
    hvp: HVP,
    preconditioner: Preconditioner,
    step_size: float | Tensor,
    momentum: float | Tensor,
) -> HeavyBallStep:
    """One exact tied Heavy-Ball block; one HVP and one application of ``B``."""

    alpha = _batch_scalar(step_size, rhs)
    beta = _batch_scalar(momentum, rhs)
    residual = rhs - hvp(state.x)
    preconditioned_residual = apply_fixed_preconditioner(preconditioner, residual)
    x_next = (
        state.x
        + alpha[:, None] * preconditioned_residual
        + beta[:, None] * (state.x - state.x_previous)
    )
    return HeavyBallStep(
        state=HeavyBallState(x=x_next, x_previous=state.x),
        residual=residual,
        preconditioned_residual=preconditioned_residual,
    )


def run_heavy_ball_state_machine(
    hvp: HVP,
    rhs: Tensor,
    preconditioner: Preconditioner,
    depth: int,
    step_size: float | Tensor,
    momentum: float | Tensor,
    target: Tensor | None = None,
) -> Tuple[Tensor, List[float], List[float]]:
    state = initialize_heavy_ball(rhs)
    mse_history: List[float] = []
    residual_history: List[float] = []
    for _ in range(depth):
        step = heavy_ball_macro_block(
            state,
            rhs,
            hvp,
            preconditioner,
            step_size,
            momentum,
        )
        state = step.state
        residual_history.append(torch.norm(step.residual, dim=-1).mean().item())
        if target is not None:
            mse_history.append(((state.x - target) ** 2).mean().item())
    return state.x, mse_history, residual_history


@dataclass(frozen=True)
class ChebyshevState:
    """Persistent vector and scalar tokens of a tied Chebyshev block."""

    x: Tensor
    x_previous: Tensor
    step_size: Tensor
    momentum: Tensor
    center: Tensor
    quarter_half_width_squared: Tensor


@dataclass(frozen=True)
class ChebyshevStep:
    state: ChebyshevState
    residual: Tensor
    preconditioned_residual: Tensor
    used_step_size: Tensor
    used_momentum: Tensor


def initialize_chebyshev(
    rhs: Tensor,
    spectral_min: float | Tensor,
    spectral_max: float | Tensor,
) -> ChebyshevState:
    """Initialize the exact scalar recurrence from a certified interval."""

    lower = _batch_scalar(spectral_min, rhs)
    upper = _batch_scalar(spectral_max, rhs)
    if torch.any(lower <= 0) or torch.any(upper < lower):
        raise ValueError("Chebyshev requires 0 < spectral_min <= spectral_max")
    center = 0.5 * (upper + lower)
    half_width = 0.5 * (upper - lower)
    zero = torch.zeros_like(rhs)
    return ChebyshevState(
        x=zero,
        x_previous=zero.clone(),
        step_size=center.reciprocal(),
        momentum=torch.zeros_like(center),
        center=center,
        quarter_half_width_squared=0.25 * half_width.pow(2),
    )


def chebyshev_macro_block(
    state: ChebyshevState,
    rhs: Tensor,
    hvp: HVP,
    preconditioner: Preconditioner,
) -> ChebyshevStep:
    """One tied Chebyshev block plus the exact next-coefficient update."""

    residual = rhs - hvp(state.x)
    preconditioned_residual = apply_fixed_preconditioner(preconditioner, residual)
    x_next = (
        state.x
        + state.step_size[:, None] * preconditioned_residual
        + state.momentum[:, None] * (state.x - state.x_previous)
    )
    next_step_size = (
        state.center
        - state.quarter_half_width_squared * state.step_size
    ).reciprocal()
    next_momentum = (
        state.quarter_half_width_squared * state.step_size * next_step_size
    )
    return ChebyshevStep(
        state=ChebyshevState(
            x=x_next,
            x_previous=state.x,
            step_size=next_step_size,
            momentum=next_momentum,
            center=state.center,
            quarter_half_width_squared=state.quarter_half_width_squared,
        ),
        residual=residual,
        preconditioned_residual=preconditioned_residual,
        used_step_size=state.step_size,
        used_momentum=state.momentum,
    )


def run_chebyshev_state_machine(
    hvp: HVP,
    rhs: Tensor,
    preconditioner: Preconditioner,
    depth: int,
    spectral_min: float | Tensor,
    spectral_max: float | Tensor,
    target: Tensor | None = None,
) -> Tuple[Tensor, List[float], List[float]]:
    state = initialize_chebyshev(rhs, spectral_min, spectral_max)
    mse_history: List[float] = []
    residual_history: List[float] = []
    for _ in range(depth):
        step = chebyshev_macro_block(state, rhs, hvp, preconditioner)
        state = step.state
        residual_history.append(torch.norm(step.residual, dim=-1).mean().item())
        if target is not None:
            mse_history.append(((state.x - target) ** 2).mean().item())
    return state.x, mse_history, residual_history


def chebyshev_coefficient_schedule(
    rhs: Tensor,
    depth: int,
    spectral_min: float | Tensor,
    spectral_max: float | Tensor,
) -> Tuple[Tensor, Tensor]:
    """Precompute the exact per-block Chebyshev scalar-token schedule.

    This is algebraically identical to updating the scalar state inside every
    block.  It exposes the intended loop-Transformer interface directly: an
    interval head supplies two endpoints, fixed arithmetic constructs
    ``[alpha_l, beta_l]``, and the vector loop only consumes those weights.
    """

    lower = _batch_scalar(spectral_min, rhs)
    upper = _batch_scalar(spectral_max, rhs)
    if torch.any(lower <= 0) or torch.any(upper < lower):
        raise ValueError("Chebyshev requires 0 < spectral_min <= spectral_max")
    if depth <= 0:
        raise ValueError("Chebyshev depth must be positive")
    center = 0.5 * (upper + lower)
    half_width = 0.5 * (upper - lower)
    quarter_half_width_squared = 0.25 * half_width.pow(2)

    # If p_0=1, p_1=d and p_{l+2}=d p_{l+1}-q p_l, then
    # alpha_l=p_l/p_{l+1}.  Writing the characteristic roots as r+ and r-
    # gives the stable vectorized ratio below.  It is the closed form of the
    # scalar recurrence, not an approximation learned by the MLP.
    discriminant = torch.sqrt(
        (center.square() - 4.0 * quarter_half_width_squared).clamp_min(1e-30)
    )
    root_plus = 0.5 * (center + discriminant)
    root_minus = 0.5 * (center - discriminant)
    root_ratio = root_minus / root_plus
    # ``cumprod`` generates t, t^2, ..., t^depth without invoking the costly
    # general floating-point ``pow`` kernel for every batch/layer pair.
    ratio_powers = torch.cumprod(
        root_ratio[:, None].expand(-1, depth), dim=1
    )
    next_ratio_powers = ratio_powers * root_ratio[:, None]
    step_schedule = root_plus[:, None].reciprocal() * (
        1.0 - ratio_powers
    ) / (1.0 - next_ratio_powers).clamp_min(1e-30)
    momentum_schedule = torch.zeros_like(step_schedule)
    if depth > 1:
        momentum_schedule[:, 1:] = (
            quarter_half_width_squared[:, None]
            * step_schedule[:, :-1]
            * step_schedule[:, 1:]
        )
    return step_schedule, momentum_schedule


def run_precomputed_chebyshev_state_machine(
    hvp: HVP,
    rhs: Tensor,
    preconditioner: Preconditioner,
    step_schedule: Tensor,
    momentum_schedule: Tensor,
    target: Tensor | None = None,
) -> Tuple[Tensor, List[float], List[float]]:
    """Run exact Chebyshev vector blocks from a precomputed scalar schedule."""

    if step_schedule.shape != momentum_schedule.shape:
        raise ValueError("Chebyshev step and momentum schedules must match")
    if step_schedule.ndim != 2 or step_schedule.shape[0] != rhs.shape[0]:
        raise ValueError("Chebyshev schedules must have shape [batch, depth]")
    x = torch.zeros_like(rhs)
    x_previous = torch.zeros_like(rhs)
    mse_history: List[float] = []
    residual_history: List[float] = []
    for layer in range(step_schedule.shape[1]):
        residual = rhs - hvp(x)
        preconditioned_residual = apply_fixed_preconditioner(
            preconditioner, residual
        )
        x_next = (
            x
            + step_schedule[:, layer, None] * preconditioned_residual
            + momentum_schedule[:, layer, None] * (x - x_previous)
        )
        x_previous, x = x, x_next
        residual_history.append(torch.norm(residual, dim=-1).mean().item())
        if target is not None:
            mse_history.append(((x - target) ** 2).mean().item())
    return x, mse_history, residual_history


@dataclass(frozen=True)
class PCGState:
    """Persistent tokens of the tied PCG macro-block: [X,R,S,P,RHO]."""

    x: Tensor
    residual: Tensor
    preconditioned_residual: Tensor
    direction: Tensor
    rho: Tensor
    rho_initial: Tensor


@dataclass(frozen=True)
class PCGStep:
    """Next state plus transient [Q,DELTA,ALPHA,BETA] work tokens."""

    state: PCGState
    operator_direction: Tensor
    delta: Tensor
    alpha: Tensor
    beta: Tensor


def initialize_pcg(rhs: Tensor, preconditioner: Preconditioner) -> PCGState:
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned_residual = apply_fixed_preconditioner(preconditioner, residual)
    direction = preconditioned_residual.clone()
    rho = batch_inner(residual, preconditioned_residual)
    return PCGState(x, residual, preconditioned_residual, direction, rho, rho.abs())


def pcg_macro_block(
    state: PCGState,
    hvp: HVP,
    preconditioner: Preconditioner,
    eps: float | None = None,
) -> PCGStep:
    """One exact fixed-preconditioner PCG block with explicit scalar routing."""

    machine = torch.finfo(state.x.dtype)
    if eps is None:
        # rho is a squared preconditioned-residual scale.  A relative
        # threshold of (100 eps_machine)^2 corresponds to a residual-level
        # tolerance near 100 eps_machine, without prematurely stopping tasks
        # whose right-hand sides happen to be small.
        eps = (100.0 * machine.eps) ** 2
    operator_direction = hvp(state.direction)
    delta = batch_inner(state.direction, operator_direction)
    relative_threshold = eps * state.rho_initial.clamp_min(machine.tiny)
    active = (state.rho.abs() > relative_threshold) & (delta > machine.tiny)
    alpha = safe_positive_quotient(state.rho, delta, active, machine.tiny)
    x_next = state.x + alpha[:, None] * state.direction
    residual_next = state.residual - alpha[:, None] * operator_direction
    preconditioned_residual_next = apply_fixed_preconditioner(
        preconditioner,
        residual_next,
    )
    rho_next = batch_inner(residual_next, preconditioned_residual_next)
    beta = safe_positive_quotient(rho_next, state.rho, active, machine.tiny)
    direction_next = preconditioned_residual_next + beta[:, None] * state.direction
    next_state = PCGState(
        x_next,
        residual_next,
        preconditioned_residual_next,
        direction_next,
        rho_next,
        state.rho_initial,
    )
    return PCGStep(next_state, operator_direction, delta, alpha, beta)


def run_pcg_state_machine(
    hvp: HVP,
    rhs: Tensor,
    preconditioner: Preconditioner,
    depth: int,
    target: Tensor | None = None,
) -> Tuple[Tensor, List[float], List[float]]:
    state = initialize_pcg(rhs, preconditioner)
    mse_history: List[float] = []
    residual_history: List[float] = []
    for _ in range(depth):
        residual_history.append(torch.norm(state.residual, dim=-1).mean().item())
        step = pcg_macro_block(state, hvp, preconditioner)
        state = step.state
        if target is not None:
            mse_history.append(((state.x - target) ** 2).mean().item())
    return state.x, mse_history, residual_history
