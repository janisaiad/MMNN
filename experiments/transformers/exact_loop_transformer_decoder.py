"""One learned softmax head followed by exact tied linear-solver algebra.

The attention head learns only a prompt-conditioned spectral subspace.  The
normal-equation product, Ritz preconditioner, Heavy-Ball memory, Chebyshev
recurrence, and PCG state transitions are explicit.  Thus the module is a
loop-Transformer decoder without asking an MLP to emulate multiplication or
division.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from .first_principles_decoder_cells import (
        Preconditioner,
        apply_fixed_preconditioner,
        chebyshev_coefficient_schedule,
        risk_optimal_solution_chebyshev_coefficients,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
        run_precomputed_chebyshev_state_machine,
        run_precomputed_moment_chebyshev_state_machine,
    )
    from .first_principles_inverse_decoder import (
        PromptSpectralIntervalMLP,
        PromptSpectralMeasureMLP,
    )
    from .structured_one_head_heavyball import (
        EquivariantMatrixFreeNystromPreconditioner,
        EquivariantPromptNystromPreconditioner,
        EquivariantRitzSoftmaxPreconditioner,
        OneHeadSpectralPreconditioner,
    )
except ImportError:
    from first_principles_decoder_cells import (
        Preconditioner,
        apply_fixed_preconditioner,
        chebyshev_coefficient_schedule,
        risk_optimal_solution_chebyshev_coefficients,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
        run_precomputed_chebyshev_state_machine,
        run_precomputed_moment_chebyshev_state_machine,
    )
    from first_principles_inverse_decoder import (
        PromptSpectralIntervalMLP,
        PromptSpectralMeasureMLP,
    )
    from structured_one_head_heavyball import (
        EquivariantMatrixFreeNystromPreconditioner,
        EquivariantPromptNystromPreconditioner,
        EquivariantRitzSoftmaxPreconditioner,
        OneHeadSpectralPreconditioner,
    )

Tensor = torch.Tensor


@dataclass(frozen=True)
class PromptGeometryCache:
    """Prompt-only geometry reusable across any number of observations."""

    equations: Tensor
    ridge: float
    ridge_metric: Tensor | None
    preconditioner: Preconditioner
    normal_matrix: Tensor | None
    head_info: Dict[str, Tensor]


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def normal_equations(
    equations: Tensor,
    observations: Tensor,
    ridge: float,
    ridge_metric: Tensor | None = None,
) -> Tuple[Tensor, Tensor]:
    """Hard-coded moment contraction with an optional covariant ridge metric."""

    normal_matrix = normal_matrix_from_equations(
        equations,
        ridge,
        ridge_metric,
    )
    rhs = torch.einsum("bmk,bm...->bk...", equations, observations)
    return normal_matrix, rhs


def normal_matrix_from_equations(
    equations: Tensor,
    ridge: float,
    ridge_metric: Tensor | None = None,
) -> Tensor:
    """Materialize only the observation-independent normal geometry."""

    batch, _, dimension = equations.shape
    identity = torch.eye(
        dimension,
        device=equations.device,
        dtype=equations.dtype,
    ).expand(batch, -1, -1)
    if ridge_metric is None:
        metric = identity
    elif ridge_metric.ndim == 2:
        metric = ridge_metric.expand(batch, -1, -1)
    else:
        metric = ridge_metric
    return equations.transpose(-1, -2) @ equations + ridge * metric


def symmetric_effective_operator(preconditioner: Tensor, normal_matrix: Tensor) -> Tensor:
    """A symmetric matrix similar to ``B H``; used by fixed Chebyshev summaries."""

    dimension = normal_matrix.shape[-1]
    identity = torch.eye(
        dimension,
        device=normal_matrix.device,
        dtype=normal_matrix.dtype,
    )
    factor = torch.linalg.cholesky(preconditioner + 1e-10 * identity)
    operator = factor.transpose(-1, -2) @ normal_matrix @ factor
    return 0.5 * (operator + operator.transpose(-1, -2))


def effective_spectrum_features(operator: Tensor, prompt_length: int) -> Tensor:
    """Seven fixed reductions with no eigenvalue input or learned arithmetic."""

    dimension = operator.shape[-1]
    diagonal = torch.diagonal(operator, dim1=-2, dim2=-1).clamp_min(1e-12)
    frobenius = torch.linalg.matrix_norm(operator, ord="fro", dim=(-2, -1))
    row_bound = operator.abs().sum(dim=-1).amax(dim=-1).clamp_min(1e-12)
    off_diagonal = operator - torch.diag_embed(diagonal)
    off_fraction = torch.linalg.matrix_norm(
        off_diagonal,
        ord="fro",
        dim=(-2, -1),
    ) / frobenius.clamp_min(1e-12)
    prompt_feature = operator.new_full(
        (operator.shape[0],),
        math.log1p(prompt_length),
    )
    return torch.stack(
        [
            torch.log(diagonal.mean(dim=-1)),
            torch.log((frobenius / math.sqrt(dimension)).clamp_min(1e-12)),
            torch.log(diagonal.amin(dim=-1)),
            torch.log(diagonal.amax(dim=-1)),
            torch.log(row_bound),
            torch.log1p(off_fraction),
            prompt_feature,
        ],
        dim=-1,
    )


class ExactLoopTransformerDecoder(nn.Module):
    """One softmax geometry head and an exact tied recurrent solver cell.

    ``controller`` is one of ``richardson``, ``heavy_ball``, ``chebyshev``,
    ``moment_chebyshev``, ``pcg`` or ``certified_hb_pcg``.  The last
    (historically named) choice
    runs Heavy-Ball by default and hard-routes prompts whose final
    preconditioned residual fails a prescribed test to PCG.  The test
    certifies the observed residual, not the energy error unless a positive
    lower spectral bound is also available.
    """

    def __init__(
        self,
        dimension: int,
        depth: int,
        head_dimension: int,
        slots: int,
        controller: str = "heavy_ball",
        spectral_lmax_bound: float = 4.0,
        step_init: float = 0.5,
        momentum_init: float = 0.05,
        chebyshev_hidden_dimension: int = 16,
        spectral_measure_clusters: int = 8,
        spectral_measure_hidden_dimension: int = 32,
        moment_gram_regularization: float = 1e-8,
        base_preconditioner: str = "jacobi",
        correction_mode: str = "ritz",
        preconditioner_head_type: str = "coordinate_ritz",
        prompt_subspace_refinement_steps: int = 2,
        chebyshev_interval_policy: str = "learned",
        adaptive_heavy_ball: bool = False,
        interval_lower_calibration: float = 1.0,
        interval_upper_calibration: float = 1.0,
        hybrid_residual_threshold: float = 1e-8,
    ) -> None:
        super().__init__()
        if controller not in {
            "richardson",
            "heavy_ball",
            "chebyshev",
            "moment_chebyshev",
            "pcg",
            "certified_hb_pcg",
        }:
            raise ValueError(f"unknown controller {controller}")
        if slots > dimension:
            raise ValueError("slots cannot exceed the coefficient dimension")
        if spectral_lmax_bound <= 0:
            raise ValueError("spectral_lmax_bound must be positive")
        if moment_gram_regularization < 0:
            raise ValueError("moment_gram_regularization must be nonnegative")
        if (
            controller == "moment_chebyshev"
            and preconditioner_head_type != "equivariant_matrix_free_nystrom"
        ):
            raise ValueError(
                "moment_chebyshev currently requires the matrix-free Nystrom head"
            )
        if chebyshev_interval_policy not in {"learned", "exact_head_spectrum"}:
            raise ValueError(
                f"unknown Chebyshev interval policy {chebyshev_interval_policy}"
            )
        if (
            chebyshev_interval_policy == "exact_head_spectrum"
            and controller == "chebyshev"
            and preconditioner_head_type != "equivariant_ritz_softmax"
        ):
            raise ValueError(
                "exact_head_spectrum requires the equivariant Ritz head"
            )
        self.dimension = dimension
        self.depth = depth
        self.controller = controller
        self.spectral_lmax_bound = spectral_lmax_bound
        self.chebyshev_interval_policy = chebyshev_interval_policy
        self.moment_gram_regularization = float(moment_gram_regularization)
        self.adaptive_heavy_ball = bool(adaptive_heavy_ball)
        self.interval_lower_calibration = float(interval_lower_calibration)
        self.interval_upper_calibration = float(interval_upper_calibration)
        self.hybrid_residual_threshold = float(hybrid_residual_threshold)
        self.matrix_free_preconditioner = (
            preconditioner_head_type == "equivariant_matrix_free_nystrom"
        )
        if self.hybrid_residual_threshold <= 0:
            raise ValueError("hybrid_residual_threshold must be positive")
        if self.interval_lower_calibration < 1.0 or self.interval_upper_calibration < 1.0:
            raise ValueError("interval calibration factors must be at least one")
        if self.adaptive_heavy_ball and controller not in {
            "heavy_ball",
            "certified_hb_pcg",
        }:
            raise ValueError(
                "adaptive_heavy_ball is defined only for HB-based controllers"
            )
        if preconditioner_head_type == "coordinate_ritz":
            self.preconditioner_head = OneHeadSpectralPreconditioner(
                dimension=dimension,
                head_dimension=head_dimension,
                slots=slots,
                max_strength=0.999,
                strength_init=0.1,
                base_preconditioner=base_preconditioner,
                base_blocks=2,
                strength_scaling="fixed",
                reference_prompt_length=32,
                slot_orthogonalization="qr",
                correction_mode=correction_mode,
                subspace_refinement_steps=0,
            )
        elif preconditioner_head_type == "equivariant_ritz_softmax":
            self.preconditioner_head = EquivariantRitzSoftmaxPreconditioner(
                dimension=dimension,
                head_dimension=head_dimension,
                slots=slots,
                spectral_lmax_bound=spectral_lmax_bound,
            )
        elif preconditioner_head_type == "equivariant_prompt_nystrom":
            self.preconditioner_head = EquivariantPromptNystromPreconditioner(
                dimension=dimension,
                head_dimension=head_dimension,
                slots=slots,
                spectral_lmax_bound=spectral_lmax_bound,
                refinement_steps=prompt_subspace_refinement_steps,
            )
        elif preconditioner_head_type == "equivariant_matrix_free_nystrom":
            self.preconditioner_head = EquivariantMatrixFreeNystromPreconditioner(
                dimension=dimension,
                head_dimension=head_dimension,
                slots=slots,
                spectral_lmax_bound=spectral_lmax_bound,
                refinement_steps=prompt_subspace_refinement_steps,
            )
        else:
            raise ValueError(
                f"unknown preconditioner head type {preconditioner_head_type}"
            )
        if controller in {"richardson", "heavy_ball", "certified_hb_pcg"}:
            if controller == "richardson":
                momentum_init = 0.0
            momentum_fraction = momentum_init / 0.999
            self.raw_momentum = nn.Parameter(
                torch.tensor(_logit(momentum_fraction)),
                requires_grad=controller in {"heavy_ball", "certified_hb_pcg"},
            )
            stable_cap = 2.0 * (1.0 + momentum_init) / spectral_lmax_bound
            if not 0.0 < step_init < 0.999 * stable_cap:
                raise ValueError(
                    f"step_init must lie in (0, {0.999 * stable_cap:.6g})"
                )
            self.raw_step = nn.Parameter(
                torch.tensor(_logit(step_init / (0.999 * stable_cap)))
            )
        else:
            self.register_parameter("raw_momentum", None)
            self.register_parameter("raw_step", None)
        self.interval_head = (
            PromptSpectralIntervalMLP(chebyshev_hidden_dimension)
            if (
                (
                    controller == "chebyshev"
                    and chebyshev_interval_policy == "learned"
                )
                or self.adaptive_heavy_ball
            )
            else None
        )
        self.measure_head = (
            PromptSpectralMeasureMLP(
                hidden_dimension=spectral_measure_hidden_dimension,
                clusters=spectral_measure_clusters,
            )
            if controller == "moment_chebyshev"
            else None
        )

    def heavy_ball_coefficients(self) -> Tuple[Tensor, Tensor]:
        if self.controller not in {
            "richardson",
            "heavy_ball",
            "certified_hb_pcg",
        }:
            raise RuntimeError("the selected controller does not use tied HB coefficients")
        if self.controller == "richardson":
            momentum = self.raw_step.new_zeros(())
        else:
            momentum = 0.999 * torch.sigmoid(self.raw_momentum)
        cap = 2.0 * (1.0 + momentum) / self.spectral_lmax_bound
        step = 0.999 * cap * torch.sigmoid(self.raw_step)
        return step, momentum

    def build_prompt_geometry(
        self,
        equations: Tensor,
        ridge: float,
        ridge_metric: Tensor | None = None,
    ) -> PromptGeometryCache:
        if self.matrix_free_preconditioner:
            normal_matrix = None
            preconditioner, info = self.preconditioner_head(
                equations,
                ridge,
                ridge_metric,
            )
        else:
            normal_matrix = normal_matrix_from_equations(
                equations,
                ridge,
                ridge_metric,
            )
            preconditioner, info = self.preconditioner_head(
                equations,
                normal_matrix,
            )
        return PromptGeometryCache(
            equations=equations,
            ridge=ridge,
            ridge_metric=ridge_metric,
            preconditioner=preconditioner,
            normal_matrix=normal_matrix,
            head_info=info,
        )

    def forward(
        self,
        equations: Tensor,
        observations: Tensor,
        ridge: float,
        ridge_metric: Tensor | None = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        geometry = self.build_prompt_geometry(
            equations,
            ridge,
            ridge_metric,
        )
        return self.solve_with_geometry(geometry, observations)

    def solve_with_geometry(
        self,
        geometry: PromptGeometryCache,
        observations: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        equations = geometry.equations
        ridge = geometry.ridge
        ridge_metric = geometry.ridge_metric
        normal_matrix = geometry.normal_matrix
        preconditioner = geometry.preconditioner
        info = dict(geometry.head_info)

        def hvp(vector: Tensor) -> Tensor:
            scores = torch.einsum("bmk,bk...->bm...", equations, vector)
            moment = torch.einsum("bmk,bm...->bk...", equations, scores)
            if ridge_metric is None:
                ridge_action = vector
            elif ridge_metric.ndim == 2:
                ridge_action = torch.einsum(
                    "kl,bl...->bk...",
                    ridge_metric,
                    vector,
                )
            else:
                ridge_action = torch.einsum(
                    "bkl,bl...->bk...",
                    ridge_metric,
                    vector,
                )
            return moment + ridge * ridge_action

        rhs = torch.einsum("bmk,bm...->bk...", equations, observations)

        if self.controller in {"richardson", "heavy_ball", "certified_hb_pcg"}:
            if self.adaptive_heavy_ball:
                assert self.interval_head is not None
                if "interval_features" in info:
                    features = info["interval_features"]
                else:
                    assert normal_matrix is not None
                    operator = symmetric_effective_operator(
                        preconditioner,
                        normal_matrix,
                    )
                    features = effective_spectrum_features(
                        operator,
                        equations.shape[1],
                    )
                spectral_min, spectral_max = self.interval_head(features)
                spectral_min = spectral_min / self.interval_lower_calibration
                spectral_max = spectral_max * self.interval_upper_calibration
                sqrt_min = torch.sqrt(spectral_min)
                sqrt_max = torch.sqrt(spectral_max)
                step = 4.0 / (sqrt_max + sqrt_min).square()
                momentum = ((sqrt_max - sqrt_min) / (sqrt_max + sqrt_min)).square()
                info.update(
                    {
                        "spectral_min": spectral_min,
                        "spectral_max": spectral_max,
                        "spectral_features": features,
                    }
                )
            else:
                step, momentum = self.heavy_ball_coefficients()
            solution, _, _ = run_heavy_ball_state_machine(
                hvp,
                rhs,
                preconditioner,
                self.depth,
                step,
                momentum,
            )
            info.update({"step": step, "momentum": momentum})
            if self.controller == "certified_hb_pcg":
                final_residual = rhs - hvp(solution)
                preconditioned_final_residual = apply_fixed_preconditioner(
                    preconditioner,
                    final_residual,
                )
                preconditioned_rhs = apply_fixed_preconditioner(
                    preconditioner,
                    rhs,
                )
                residual_ratio = torch.einsum(
                    "bi...,bi...->b...",
                    final_residual,
                    preconditioned_final_residual,
                ) / torch.einsum(
                    "bi...,bi...->b...",
                    rhs,
                    preconditioned_rhs,
                ).clamp_min(1e-30)
                fallback_mask = residual_ratio > self.hybrid_residual_threshold
                if fallback_mask.any():
                    fallback_solution, _, _ = run_pcg_state_machine(
                        hvp,
                        rhs,
                        preconditioner,
                        self.depth,
                    )
                    solution = torch.where(
                        fallback_mask.unsqueeze(1),
                        fallback_solution,
                        solution,
                    )
                info.update(
                    {
                        "hb_final_preconditioned_residual_ratio": residual_ratio,
                        "pcg_fallback_mask": fallback_mask,
                        "pcg_fallback_rate": fallback_mask.float().mean(),
                    }
                )
        elif self.controller == "pcg":
            solution, _, _ = run_pcg_state_machine(
                hvp,
                rhs,
                preconditioner,
                self.depth,
            )
        elif self.controller == "moment_chebyshev":
            assert self.measure_head is not None
            features = info["interval_features"]
            # The matrix-free Ritz correction maps every selected high mode
            # to the smallest selected Ritz value.  The MLP predicts only a
            # compressed measure below that prompt-dependent reference; all
            # coefficient construction and vector arithmetic remain exact.
            reference_upper = info.get(
                "projected_effective_target",
                info["projected_eigenvalues"][:, 0],
            )
            spectral_nodes, spectral_weights, spectral_upper = self.measure_head(
                features,
                reference_upper,
                info["certified_effective_lmax"],
            )
            coefficients = risk_optimal_solution_chebyshev_coefficients(
                spectral_nodes,
                spectral_weights,
                self.depth,
                spectral_upper,
                self.moment_gram_regularization,
            )
            solution, _, _ = run_precomputed_moment_chebyshev_state_machine(
                hvp,
                rhs,
                preconditioner,
                coefficients,
                spectral_upper,
            )
            info.update(
                {
                    "spectral_features": features,
                    "spectral_measure_nodes": spectral_nodes,
                    "spectral_measure_weights": spectral_weights,
                    "spectral_upper": spectral_upper,
                    "moment_solution_coefficients": coefficients,
                }
            )
        else:
            if self.chebyshev_interval_policy == "exact_head_spectrum":
                effective_eigenvalues = info["effective_eigenvalues_predicted"]
                spectral_min = effective_eigenvalues.amin(dim=-1)
                spectral_max = effective_eigenvalues.amax(dim=-1)
                features = effective_eigenvalues
            else:
                assert self.interval_head is not None
                if "interval_features" in info:
                    features = info["interval_features"]
                else:
                    assert normal_matrix is not None
                    operator = symmetric_effective_operator(
                        preconditioner,
                        normal_matrix,
                    )
                    features = effective_spectrum_features(
                        operator,
                        equations.shape[1],
                    )
                spectral_min, spectral_max = self.interval_head(features)
                spectral_min = spectral_min / self.interval_lower_calibration
                spectral_max = spectral_max * self.interval_upper_calibration
            step_schedule, momentum_schedule = chebyshev_coefficient_schedule(
                rhs, self.depth, spectral_min, spectral_max
            )
            solution, _, _ = run_precomputed_chebyshev_state_machine(
                hvp, rhs, preconditioner, step_schedule, momentum_schedule
            )
            info.update(
                {
                    "spectral_min": spectral_min,
                    "spectral_max": spectral_max,
                    "spectral_features": features,
                    "chebyshev_step_schedule": step_schedule,
                    "chebyshev_momentum_schedule": momentum_schedule,
                }
            )
        info.update({"rhs": rhs, "preconditioner": preconditioner})
        if normal_matrix is not None:
            info["normal_matrix"] = normal_matrix
        info["normal_matrix_materialized"] = rhs.new_full(
            (rhs.shape[0],),
            normal_matrix is not None,
            dtype=torch.bool,
        )
        return solution, info
