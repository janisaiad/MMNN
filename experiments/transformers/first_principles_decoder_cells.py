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

    token_scores = torch.einsum("bmk,bk...->bm...", equations, vector)
    prompt_value_sum = torch.einsum(
        "bmk,bm...->bk...",
        equations,
        token_scores,
    )
    return noise_precision * prompt_value_sum + prior_precision * vector


def apply_fixed_preconditioner(preconditioner: Preconditioner, vector: Tensor) -> Tensor:
    """Route a vector through the prompt-conditioned but iteration-fixed map."""

    if hasattr(preconditioner, "apply"):
        return preconditioner.apply(vector)
    if callable(preconditioner):
        return preconditioner(vector)
    return torch.einsum("bkl,bl...->bk...", preconditioner, vector)


def materialize_preconditioner(preconditioner: Preconditioner) -> Tensor:
    """Materialize only for diagnostics or dense-baseline comparisons."""

    if isinstance(preconditioner, Tensor):
        return preconditioner
    if hasattr(preconditioner, "materialize"):
        return preconditioner.materialize()
    raise TypeError("the preconditioner has no dense materialization")


def batch_inner(left: Tensor, right: Tensor) -> Tensor:
    """Fixed batchwise scalar-reduction primitive."""

    return torch.einsum("bk...,bk...->b...", left, right)


def _batch_scalar(value: float | Tensor, reference: Tensor) -> Tensor:
    scalar = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    target_shape = (reference.shape[0], *reference.shape[2:])
    if scalar.ndim == 0:
        return scalar.expand(target_shape)
    if scalar.shape == target_shape:
        return scalar
    if scalar.shape == (reference.shape[0],):
        view_shape = (reference.shape[0],) + (1,) * (reference.ndim - 2)
        return scalar.reshape(view_shape).expand(target_shape)
    raise ValueError(
        "expected a scalar, batch vector, or one scalar per right-hand side; "
        f"got {tuple(scalar.shape)} for state {tuple(reference.shape)}"
    )


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
        + alpha.unsqueeze(1) * preconditioned_residual
        + beta.unsqueeze(1) * (state.x - state.x_previous)
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
    record_history: bool = False,
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
        if record_history:
            residual_history.append(
                torch.norm(step.residual, dim=1).mean().item()
            )
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
        + state.step_size.unsqueeze(1) * preconditioned_residual
        + state.momentum.unsqueeze(1) * (state.x - state.x_previous)
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
    record_history: bool = False,
) -> Tuple[Tensor, List[float], List[float]]:
    state = initialize_chebyshev(rhs, spectral_min, spectral_max)
    mse_history: List[float] = []
    residual_history: List[float] = []
    for _ in range(depth):
        step = chebyshev_macro_block(state, rhs, hvp, preconditioner)
        state = step.state
        if record_history:
            residual_history.append(
                torch.norm(step.residual, dim=1).mean().item()
            )
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
        root_ratio.unsqueeze(-1).expand(*root_ratio.shape, depth),
        dim=-1,
    )
    next_ratio_powers = ratio_powers * root_ratio.unsqueeze(-1)
    step_schedule = root_plus.unsqueeze(-1).reciprocal() * (
        1.0 - ratio_powers
    ) / (1.0 - next_ratio_powers).clamp_min(1e-30)
    momentum_schedule = torch.zeros_like(step_schedule)
    if depth > 1:
        momentum_schedule[..., 1:] = (
            quarter_half_width_squared.unsqueeze(-1)
            * step_schedule[..., :-1]
            * step_schedule[..., 1:]
        )
    return step_schedule, momentum_schedule


def run_precomputed_chebyshev_state_machine(
    hvp: HVP,
    rhs: Tensor,
    preconditioner: Preconditioner,
    step_schedule: Tensor,
    momentum_schedule: Tensor,
    target: Tensor | None = None,
    record_history: bool = False,
) -> Tuple[Tensor, List[float], List[float]]:
    """Run exact Chebyshev vector blocks from a precomputed scalar schedule."""

    if step_schedule.shape != momentum_schedule.shape:
        raise ValueError("Chebyshev step and momentum schedules must match")
    scalar_shape = _batch_scalar(1.0, rhs).shape
    if step_schedule.shape[:-1] != scalar_shape:
        raise ValueError(
            "Chebyshev schedule leading dimensions must match the batch and "
            "right-hand-side dimensions"
        )
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
            + step_schedule[..., layer].unsqueeze(1) * preconditioned_residual
            + momentum_schedule[..., layer].unsqueeze(1) * (x - x_previous)
        )
        x_previous, x = x, x_next
        if record_history:
            residual_history.append(
                torch.norm(residual, dim=1).mean().item()
            )
        if target is not None:
            mse_history.append(((x - target) ** 2).mean().item())
    return x, mse_history, residual_history


def shifted_chebyshev_basis(
    spectral_nodes: Tensor,
    degree: int,
    spectral_upper: Tensor,
) -> Tensor:
    """Evaluate the first shifted Chebyshev polynomials on the interval."""

    if degree <= 0:
        raise ValueError("polynomial degree must be positive")
    if spectral_nodes.ndim != 2:
        raise ValueError("spectral nodes must have shape [batch, clusters]")
    if spectral_upper.shape != (spectral_nodes.shape[0],):
        raise ValueError("spectral upper endpoint must have shape [batch]")
    if torch.any(spectral_upper <= 0):
        raise ValueError("spectral upper endpoint must be positive")
    normalized = 2.0 * spectral_nodes / spectral_upper[:, None] - 1.0
    basis = [torch.ones_like(normalized)]
    if degree > 1:
        basis.append(normalized)
    for _ in range(2, degree):
        basis.append(2.0 * normalized * basis[-1] - basis[-2])
    return torch.stack(basis, dim=-1)


def risk_optimal_solution_chebyshev_coefficients(
    spectral_nodes: Tensor,
    spectral_weights: Tensor,
    degree: int,
    spectral_upper: Tensor,
    gram_regularization: float = 1e-10,
) -> Tensor:
    """Construct the exact weighted-risk solution polynomial.

    The neural component may predict nodes and masses, but the small weighted
    Gram solve here is deterministic algebra.  A relative diagonal
    regularizer defines finite-precision behavior when the predicted measure
    has fewer effective clusters than coefficients.
    """

    if spectral_weights.shape != spectral_nodes.shape:
        raise ValueError("spectral nodes and weights must have the same shape")
    if torch.any(spectral_nodes <= 0):
        raise ValueError("spectral nodes must be positive")
    if torch.any(spectral_weights < 0):
        raise ValueError("spectral weights must be nonnegative")
    if gram_regularization < 0:
        raise ValueError("gram regularization must be nonnegative")
    output_dtype = spectral_nodes.dtype
    solve_dtype = (
        torch.float64
        if output_dtype in {torch.float16, torch.bfloat16, torch.float32}
        else output_dtype
    )
    nodes = spectral_nodes.to(solve_dtype)
    weights = spectral_weights.to(solve_dtype)
    upper = spectral_upper.to(solve_dtype)
    weights = weights / weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(solve_dtype).tiny)
    basis = shifted_chebyshev_basis(
        nodes,
        degree,
        upper,
    )
    design = nodes.unsqueeze(-1) * basis
    gram = torch.einsum(
        "bjl,bj,bjm->blm",
        design,
        weights,
        design,
    )
    right_hand_side = torch.einsum(
        "bjl,bj->bl",
        design,
        weights,
    )
    if gram_regularization:
        scale = torch.diagonal(
            gram,
            dim1=-2,
            dim2=-1,
        ).mean(dim=-1).clamp_min(torch.finfo(gram.dtype).tiny)
        identity = torch.eye(
            degree,
            device=gram.device,
            dtype=gram.dtype,
        ).expand(gram.shape[0], -1, -1)
        gram = (
            gram
            + gram_regularization * scale[:, None, None] * identity
        )
    coefficients = torch.linalg.solve(
        gram,
        right_hand_side.unsqueeze(-1),
    ).squeeze(-1)
    return coefficients.to(output_dtype)


def block_krylov_energy_measure(
    operator: HVP,
    probes: Tensor,
    steps: int,
    operator_trace: Tensor,
    spectral_upper: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build a promptwise spectral measure by exact block-Krylov algebra.

    The learned head supplies only the covariant starting probes. Repeated
    operator actions, full reorthogonalization, the small Ritz eigensolve, and
    the trace-complement closure are fixed operations. For an isotropic
    coefficient prior, a mode's contribution to expected energy error is
    proportional to its eigenvalue, which fixes the returned weights without
    spectral labels or a learned arithmetic network.

    The final atom represents the orthogonal complement by its exact mean
    eigenvalue. This is a first-moment closure; increasing 'steps' enlarges
    the resolved Krylov subspace while preserving a fixed block-HVP schedule.
    """

    if probes.ndim != 3:
        raise ValueError("block-Krylov probes must have shape [batch, dimension, slots]")
    if steps <= 0:
        raise ValueError("block-Krylov steps must be positive")
    batch, dimension, slots = probes.shape
    resolved_dimension = steps * slots
    if resolved_dimension > dimension:
        raise ValueError("block-Krylov steps times slots cannot exceed dimension")
    if operator_trace.shape != (batch,):
        raise ValueError("operator trace must have shape [batch]")
    if spectral_upper.shape != (batch,):
        raise ValueError("spectral upper endpoint must have shape [batch]")
    if torch.any(operator_trace <= 0) or torch.any(spectral_upper <= 0):
        raise ValueError("operator trace and spectral upper endpoint must be positive")

    block = torch.linalg.qr(probes, mode="reduced").Q
    basis_blocks = []
    action_blocks = []
    for block_index in range(steps):
        action = operator(block)
        basis_blocks.append(block)
        action_blocks.append(action)
        if block_index + 1 == steps:
            continue
        candidate = action
        # Two deterministic reorthogonalization passes keep the compressed
        # operator symmetric and stable at the small depths used here.
        for _ in range(2):
            for previous in basis_blocks:
                overlap = torch.einsum(
                    "bks,bkt->bst",
                    previous,
                    candidate,
                )
                candidate = candidate - torch.einsum(
                    "bks,bst->bkt",
                    previous,
                    overlap,
                )
        block = torch.linalg.qr(candidate, mode="reduced").Q
        # QR of a very small post-deflation residual can lose orthogonality in
        # float32 even when the block is mathematically full rank. Reproject
        # the normalized block, then QR once more; this changes no exact
        # arithmetic and avoids mistaking scale loss for a new Krylov mode.
        for previous in basis_blocks:
            overlap = torch.einsum(
                "bks,bkt->bst",
                previous,
                block,
            )
            block = block - torch.einsum(
                "bks,bst->bkt",
                previous,
                overlap,
            )
        block = torch.linalg.qr(block, mode="reduced").Q

    basis = torch.cat(basis_blocks, dim=-1)
    actions = torch.cat(action_blocks, dim=-1)
    projected = torch.einsum("bki,bkj->bij", basis, actions)
    projected = 0.5 * (projected + projected.transpose(-1, -2))
    ritz_nodes = torch.linalg.eigvalsh(projected)

    dtype_epsilon = torch.finfo(ritz_nodes.dtype).eps
    tolerance = (
        1024.0
        * dtype_epsilon
        * torch.maximum(
            spectral_upper.abs(),
            ritz_nodes[:, -1].abs(),
        ).clamp_min(1.0)
    )
    if torch.any(ritz_nodes[:, -1] > spectral_upper + tolerance):
        raise ValueError("certified upper endpoint does not cover the Ritz spectrum")
    ritz_nodes = torch.minimum(
        ritz_nodes.clamp_min(torch.finfo(ritz_nodes.dtype).tiny),
        spectral_upper[:, None],
    )

    resolved_trace = ritz_nodes.sum(dim=-1)
    trace_tolerance = (
        1024.0
        * dtype_epsilon
        * torch.maximum(operator_trace.abs(), resolved_trace.abs()).clamp_min(1.0)
    )
    if torch.any(resolved_trace > operator_trace + trace_tolerance):
        raise ValueError("operator trace does not cover the resolved Ritz trace")
    complement_trace = (operator_trace - resolved_trace).clamp_min(0.0)
    complement_dimension = dimension - resolved_dimension
    multiplicities = torch.ones_like(ritz_nodes)
    nodes = ritz_nodes
    if complement_dimension:
        complement_node = (
            complement_trace / complement_dimension
        ).clamp_min(torch.finfo(ritz_nodes.dtype).tiny)
        nodes = torch.cat([ritz_nodes, complement_node[:, None]], dim=-1)
        multiplicities = torch.cat(
            [
                multiplicities,
                ritz_nodes.new_full((batch, 1), float(complement_dimension)),
            ],
            dim=-1,
        )
    energy_weights = multiplicities * nodes
    energy_weights = energy_weights / energy_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(energy_weights.dtype).tiny)
    return nodes, energy_weights, projected, basis, complement_trace


def run_precomputed_moment_chebyshev_state_machine(
    hvp: HVP,
    rhs: Tensor,
    preconditioner: Preconditioner,
    solution_coefficients: Tensor,
    spectral_upper: Tensor,
) -> Tuple[Tensor, List[float], List[float]]:
    """Apply a prompt-conditioned solution polynomial by exact Clenshaw.

    Coefficients are fixed before the vector loop.  Every large-vector
    operation is an exact normal HVP, a fixed preconditioner application, or
    a routed linear combination.  No MLP emulates solver arithmetic.
    """

    if solution_coefficients.ndim != 2:
        raise ValueError("solution coefficients must have shape [batch, depth]")
    if solution_coefficients.shape[0] != rhs.shape[0]:
        raise ValueError("coefficient batch dimension must match the right-hand side")
    if solution_coefficients.shape[1] <= 0:
        raise ValueError("at least one polynomial coefficient is required")
    upper = _batch_scalar(spectral_upper, rhs)
    if torch.any(upper <= 0):
        raise ValueError("spectral upper endpoint must be positive")
    base = apply_fixed_preconditioner(preconditioner, rhs)

    def normalized_operator(vector: Tensor) -> Tensor:
        applied = apply_fixed_preconditioner(preconditioner, hvp(vector))
        return 2.0 * applied / upper.unsqueeze(1) - vector

    next_term = torch.zeros_like(rhs)
    following_term = torch.zeros_like(rhs)
    for index in range(solution_coefficients.shape[1] - 1, 0, -1):
        coefficient = _batch_scalar(
            solution_coefficients[:, index],
            rhs,
        )
        current = (
            2.0 * normalized_operator(next_term)
            - following_term
            + coefficient.unsqueeze(1) * base
        )
        following_term, next_term = next_term, current
    leading = _batch_scalar(solution_coefficients[:, 0], rhs)
    solution = (
        normalized_operator(next_term)
        - following_term
        + leading.unsqueeze(1) * base
    )
    return solution, [], []


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
    x_next = state.x + alpha.unsqueeze(1) * state.direction
    residual_next = state.residual - alpha.unsqueeze(1) * operator_direction
    preconditioned_residual_next = apply_fixed_preconditioner(
        preconditioner,
        residual_next,
    )
    rho_next = batch_inner(residual_next, preconditioned_residual_next)
    beta = safe_positive_quotient(rho_next, state.rho, active, machine.tiny)
    direction_next = (
        preconditioned_residual_next + beta.unsqueeze(1) * state.direction
    )
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
    record_history: bool = False,
) -> Tuple[Tensor, List[float], List[float]]:
    state = initialize_pcg(rhs, preconditioner)
    mse_history: List[float] = []
    residual_history: List[float] = []
    for _ in range(depth):
        if record_history:
            residual_history.append(
                torch.norm(state.residual, dim=1).mean().item()
            )
        step = pcg_macro_block(state, hvp, preconditioner)
        state = step.state
        if target is not None:
            mse_history.append(((state.x - target) ** 2).mean().item())
    return state.x, mse_history, residual_history
