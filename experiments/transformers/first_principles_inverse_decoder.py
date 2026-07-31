"""Minimal ICL decoder for a low-rank ridge inverse problem.

The prompt supplies weak equations ``G z ~= b``.  One attention head recovers
an observable subspace, a fixed Ritz formula builds an SPD factorized inverse,
and an exact tied Heavy-Ball or PCG state machine performs the solve.  The
module supports algebraically equivalent primal and dual representations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from .first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from .low_rank_subspace_preconditioner import (
        OneHeadObservableSubspace,
        build_factorized_subspace_inverse,
        exact_observable_directions,
    )
except ImportError:
    from first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from low_rank_subspace_preconditioner import (
        OneHeadObservableSubspace,
        build_factorized_subspace_inverse,
        exact_observable_directions,
    )

Tensor = torch.Tensor


def spectral_interval_features(
    normal_matrix: Tensor,
    directions: Tensor,
    ridge_precision: float,
) -> Tensor:
    """Fixed invariant summaries used only to estimate Chebyshev bounds."""

    dimension = normal_matrix.shape[-1]
    slots = directions.shape[-1]
    scale = torch.as_tensor(
        ridge_precision,
        device=normal_matrix.device,
        dtype=normal_matrix.dtype,
    ).clamp_min(1e-12)
    trace = torch.diagonal(normal_matrix, dim1=-2, dim2=-1).sum(dim=-1)
    frobenius = torch.linalg.matrix_norm(normal_matrix, ord="fro", dim=(-2, -1))
    projected = torch.einsum(
        "bks,bkl,blt->bst",
        directions,
        normal_matrix,
        directions,
    )
    projected_eigenvalues = torch.linalg.eigvalsh(projected).clamp_min(1e-12)
    hq = torch.einsum("bkl,bls->bks", normal_matrix, directions)
    projected_hq = torch.einsum("bks,bst->bkt", directions, projected)
    cross_norm = torch.linalg.matrix_norm(
        hq - projected_hq,
        ord="fro",
        dim=(-2, -1),
    )
    batch_size = normal_matrix.shape[0]
    dimension_feature = normal_matrix.new_full((batch_size,), math.log1p(dimension))
    slot_fraction = normal_matrix.new_full((batch_size,), slots / dimension)
    return torch.stack(
        [
            torch.log((trace / dimension) / scale),
            torch.log((frobenius / math.sqrt(dimension)) / scale),
            torch.log(projected_eigenvalues[:, 0] / scale),
            torch.log(projected_eigenvalues[:, -1] / scale),
            torch.log1p(cross_norm / scale),
            dimension_feature,
            slot_fraction,
        ],
        dim=-1,
    )


class PromptSpectralIntervalMLP(nn.Module):
    """Predict only a positive ordered interval; solver arithmetic stays fixed."""

    def __init__(self, hidden_dimension: int = 16, safety_margin: float = 0.1) -> None:
        super().__init__()
        self.safety_margin = safety_margin
        self.network = nn.Sequential(
            nn.Linear(7, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 2),
        )
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        raw_center, raw_width = self.network(features).unbind(dim=-1)
        half_log_width = torch.nn.functional.softplus(raw_width) + self.safety_margin
        spectral_min = torch.exp(raw_center - half_log_width).clamp_min(1e-8)
        spectral_max = torch.exp(raw_center + half_log_width)
        return spectral_min, spectral_max


class PromptSpectralMeasureMLP(nn.Module):
    """Predict only cluster nodes and masses for an exact polynomial solve."""

    def __init__(
        self,
        input_dimension: int = 7,
        hidden_dimension: int = 32,
        clusters: int = 8,
        minimum_node_fraction: float = 1e-4,
        upper_safety_margin: float = 0.05,
        initial_upper_multiplier: float = 1.5,
    ) -> None:
        super().__init__()
        if clusters < 2:
            raise ValueError("at least two spectral clusters are required")
        if not 0.0 < minimum_node_fraction < 1.0:
            raise ValueError("minimum node fraction must lie in (0,1)")
        if upper_safety_margin < 0:
            raise ValueError("upper safety margin must be nonnegative")
        if initial_upper_multiplier <= 1.0:
            raise ValueError("initial upper multiplier must exceed one")
        self.clusters = clusters
        self.minimum_node_fraction = minimum_node_fraction
        self.upper_safety_margin = upper_safety_margin
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, 2 * clusters + 1),
        )
        fractions = torch.logspace(
            math.log10(0.02),
            math.log10(0.75),
            clusters,
        )
        fractions = fractions.clamp(
            minimum_node_fraction + 1e-6,
            1.0 - 1e-6,
        )
        normalized = (
            fractions - minimum_node_fraction
        ) / (1.0 - minimum_node_fraction)
        node_bias = torch.log(normalized / (1.0 - normalized))
        initial_mass = torch.full((clusters,), 0.25 / (clusters - 1))
        initial_mass[0] = 0.75
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias[:clusters].copy_(node_bias)
            self.network[-1].bias[clusters:].copy_(
                torch.cat(
                    [
                        torch.log(initial_mass),
                        torch.tensor(
                            [
                                math.log(
                                    math.expm1(initial_upper_multiplier - 1.0)
                                )
                            ]
                        ),
                    ]
                )
            )

    def forward(
        self,
        features: Tensor,
        reference_upper: Tensor,
        certified_upper: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if reference_upper.shape != (features.shape[0],):
            raise ValueError("reference upper endpoint must have shape [batch]")
        if certified_upper.shape != (features.shape[0],):
            raise ValueError("certified upper endpoint must have shape [batch]")
        tolerance = (
            10.0
            * torch.finfo(features.dtype).eps
            * torch.maximum(
                certified_upper.abs(),
                reference_upper.abs(),
            ).clamp_min(1.0)
        )
        if torch.any(certified_upper + tolerance < reference_upper):
            raise ValueError("certified upper endpoint must cover the Ritz reference")
        # A proven bound and its algebraically equal Ritz target can differ by
        # one float32 ulp.  Clip only that certified numerical roundoff before
        # constructing the learned expansion.
        reference_upper = torch.minimum(reference_upper, certified_upper)
        raw = self.network(features)
        raw_fractions = raw[:, : self.clusters]
        mass_logits = raw[:, self.clusters : 2 * self.clusters]
        raw_upper_expansion = raw[:, -1]
        support_upper = torch.minimum(
            reference_upper * (1.0 + torch.nn.functional.softplus(raw_upper_expansion)),
            certified_upper,
        )
        fractions = self.minimum_node_fraction + (
            1.0 - self.minimum_node_fraction
        ) * torch.sigmoid(raw_fractions)
        fractions, order = fractions.sort(dim=-1)
        weights = torch.softmax(mass_logits, dim=-1).gather(-1, order)
        nodes = support_upper[:, None] * fractions
        basis_upper = torch.minimum(
            support_upper * (1.0 + self.upper_safety_margin),
            certified_upper,
        )
        return nodes, weights, basis_upper


def spectral_interval_coverage_loss(
    predicted_min: Tensor,
    predicted_max: Tensor,
    true_min: Tensor,
    true_max: Tensor,
) -> Tensor:
    """One-sided log penalty: intervals are useful only when they cover."""

    lower_violation = torch.relu(torch.log(predicted_min) - torch.log(true_min))
    upper_violation = torch.relu(torch.log(true_max) - torch.log(predicted_max))
    width = torch.log(predicted_max) - torch.log(predicted_min)
    return (lower_violation.square() + upper_violation.square() + 1e-3 * width).mean()


@dataclass(frozen=True)
class ActiveRidgeSystem:
    side: str
    normal_matrix: Tensor
    rhs: Tensor
    equations: Tensor
    active_factor: Tensor
    data_precision: float
    ridge_precision: float

    def hvp(self, vector: Tensor) -> Tensor:
        if self.side == "primal":
            scores = torch.einsum("bmk,bk->bm", self.active_factor, vector)
            moment = torch.einsum("bmk,bm->bk", self.active_factor, scores)
        else:
            scores = torch.einsum("bmk,bm->bk", self.active_factor, vector)
            moment = torch.einsum("bmk,bk->bm", self.active_factor, scores)
        return self.data_precision * moment + self.ridge_precision * vector

    def decode(self, active_solution: Tensor) -> Tensor:
        if self.side == "primal":
            return active_solution
        return torch.einsum("bmk,bm->bk", self.active_factor, active_solution)


def build_active_ridge_system(
    equations: Tensor,
    observations: Tensor,
    data_precision: float,
    ridge_precision: float,
    side: str = "auto",
) -> ActiveRidgeSystem:
    """Choose the smaller exact primal/dual representation of ridge regression."""

    batch_size, equation_count, coefficient_dimension = equations.shape
    if side == "auto":
        side = "dual" if equation_count < coefficient_dimension else "primal"
    if side == "primal":
        identity = torch.eye(
            coefficient_dimension,
            device=equations.device,
            dtype=equations.dtype,
        ).expand(batch_size, -1, -1)
        normal_matrix = (
            data_precision * equations.transpose(-1, -2) @ equations
            + ridge_precision * identity
        )
        rhs = data_precision * torch.einsum("bmk,bm->bk", equations, observations)
        active_factor = equations
    elif side == "dual":
        identity = torch.eye(
            equation_count,
            device=equations.device,
            dtype=equations.dtype,
        ).expand(batch_size, -1, -1)
        normal_matrix = (
            data_precision * equations @ equations.transpose(-1, -2)
            + ridge_precision * identity
        )
        rhs = data_precision * observations
        active_factor = equations
    elif side == "compressed_dual":
        left_basis, active_factor = torch.linalg.qr(equations, mode="reduced")
        active_dimension = active_factor.shape[-2]
        identity = torch.eye(
            active_dimension,
            device=equations.device,
            dtype=equations.dtype,
        ).expand(batch_size, -1, -1)
        normal_matrix = (
            data_precision * active_factor @ active_factor.transpose(-1, -2)
            + ridge_precision * identity
        )
        rhs = data_precision * torch.einsum("bmq,bm->bq", left_basis, observations)
    else:
        raise ValueError(f"unknown side {side}")
    return ActiveRidgeSystem(
        side,
        normal_matrix,
        rhs,
        equations,
        active_factor,
        data_precision,
        ridge_precision,
    )


class FirstPrinciplesInverseDecoder(nn.Module):
    """One learned head followed by one of two exact recurrent controllers."""

    def __init__(
        self,
        coefficient_dimension: int,
        head_dimension: int,
        slots: int,
        depth: int,
        solver_cell: str = "pcg",
        representation: str = "auto",
        heavy_ball_step: float = 1.0,
        heavy_ball_momentum: float = 0.0,
        subspace_mode: str = "attention",
    ) -> None:
        super().__init__()
        if solver_cell not in {"heavy_ball", "chebyshev", "pcg"}:
            raise ValueError(f"unknown solver cell {solver_cell}")
        if representation not in {"auto", "primal", "dual", "compressed_dual"}:
            raise ValueError(f"unknown representation {representation}")
        if subspace_mode not in {"attention", "exact_prompt"}:
            raise ValueError(f"unknown subspace mode {subspace_mode}")
        self.subspace_head = OneHeadObservableSubspace(
            coefficient_dimension,
            head_dimension,
            slots,
        )
        self.depth = depth
        self.solver_cell = solver_cell
        self.representation = representation
        self.heavy_ball_step = heavy_ball_step
        self.heavy_ball_momentum = heavy_ball_momentum
        self.subspace_mode = subspace_mode
        self.spectral_interval_head = (
            PromptSpectralIntervalMLP() if solver_cell == "chebyshev" else None
        )

    def forward(
        self,
        equations: Tensor,
        observations: Tensor,
        data_precision: float,
        ridge_precision: float,
        spectral_bounds: Tuple[Tensor | float, Tensor | float] | None = None,
    ) -> Tuple[Tensor, Dict[str, Tensor | str]]:
        system = build_active_ridge_system(
            equations,
            observations,
            data_precision,
            ridge_precision,
            self.representation,
        )
        if self.subspace_mode == "attention":
            preconditioner, head_info = self.subspace_head(
                equations,
                system.normal_matrix,
                ridge_precision,
                side=system.side,
                active_factor=system.active_factor,
            )
        else:
            subspace_factor = (
                equations if system.side in {"primal", "dual"} else system.active_factor
            )
            directions = exact_observable_directions(
                subspace_factor,
                self.subspace_head.slots,
                side="primal" if system.side == "primal" else "dual",
            )
            preconditioner = build_factorized_subspace_inverse(
                system.normal_matrix,
                directions,
                ridge_precision,
            )
            head_info = {
                "directions": directions,
                "projected_cholesky": preconditioner.projected_cholesky,
            }
        if self.solver_cell == "pcg":
            active_solution, _, _ = run_pcg_state_machine(
                system.hvp,
                system.rhs,
                preconditioner,
                self.depth,
            )
        elif self.solver_cell == "heavy_ball":
            active_solution, _, _ = run_heavy_ball_state_machine(
                system.hvp,
                system.rhs,
                preconditioner,
                self.depth,
                self.heavy_ball_step,
                self.heavy_ball_momentum,
            )
        else:
            if spectral_bounds is None:
                assert self.spectral_interval_head is not None
                interval_features = spectral_interval_features(
                    system.normal_matrix,
                    head_info["directions"],
                    ridge_precision,
                )
                spectral_min, spectral_max = self.spectral_interval_head(
                    interval_features
                )
            else:
                spectral_min, spectral_max = spectral_bounds
            active_solution, _, _ = run_chebyshev_state_machine(
                system.hvp,
                system.rhs,
                preconditioner,
                self.depth,
                spectral_min,
                spectral_max,
            )
            head_info = {
                **head_info,
                "spectral_min": torch.as_tensor(
                    spectral_min,
                    device=equations.device,
                    dtype=equations.dtype,
                ),
                "spectral_max": torch.as_tensor(
                    spectral_max,
                    device=equations.device,
                    dtype=equations.dtype,
                ),
            }
        return system.decode(active_solution), {
            **head_info,
            "side": system.side,
            "normal_matrix": system.normal_matrix,
        }
