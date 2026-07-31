"""One learned softmax head followed by exact tied linear-solver algebra.

The attention head learns only a prompt-conditioned spectral subspace.  The
normal-equation product, Ritz preconditioner, Heavy-Ball memory, Chebyshev
recurrence, and PCG state transitions are explicit.  Thus the module is a
loop-Transformer decoder without asking an MLP to emulate multiplication or
division.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

try:
    from .first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from .first_principles_inverse_decoder import PromptSpectralIntervalMLP
    from .structured_one_head_heavyball import OneHeadSpectralPreconditioner
except ImportError:
    from first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from first_principles_inverse_decoder import PromptSpectralIntervalMLP
    from structured_one_head_heavyball import OneHeadSpectralPreconditioner

Tensor = torch.Tensor


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
    normal_matrix = equations.transpose(-1, -2) @ equations + ridge * metric
    rhs = torch.einsum("bmk,bm->bk", equations, observations)
    return normal_matrix, rhs


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
    ``pcg`` or ``certified_hb_pcg``.  The last choice runs Heavy-Ball by
    default and hard-routes only prompts whose final preconditioned residual
    fails a prescribed certificate to PCG.
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
        base_preconditioner: str = "jacobi",
        correction_mode: str = "ritz",
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
            "pcg",
            "certified_hb_pcg",
        }:
            raise ValueError(f"unknown controller {controller}")
        if slots > dimension:
            raise ValueError("slots cannot exceed the coefficient dimension")
        if spectral_lmax_bound <= 0:
            raise ValueError("spectral_lmax_bound must be positive")
        self.dimension = dimension
        self.depth = depth
        self.controller = controller
        self.spectral_lmax_bound = spectral_lmax_bound
        self.adaptive_heavy_ball = bool(adaptive_heavy_ball)
        self.interval_lower_calibration = float(interval_lower_calibration)
        self.interval_upper_calibration = float(interval_upper_calibration)
        self.hybrid_residual_threshold = float(hybrid_residual_threshold)
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
            if controller == "chebyshev" or self.adaptive_heavy_ball
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

    def forward(
        self,
        equations: Tensor,
        observations: Tensor,
        ridge: float,
        ridge_metric: Tensor | None = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        normal_matrix, rhs = normal_equations(
            equations, observations, ridge, ridge_metric
        )
        preconditioner, info = self.preconditioner_head(equations, normal_matrix)

        def hvp(vector: Tensor) -> Tensor:
            scores = torch.einsum("bmk,bk->bm", equations, vector)
            moment = torch.einsum("bmk,bm->bk", equations, scores)
            if ridge_metric is None:
                ridge_action = vector
            elif ridge_metric.ndim == 2:
                ridge_action = torch.einsum("kl,bl->bk", ridge_metric, vector)
            else:
                ridge_action = torch.einsum("bkl,bl->bk", ridge_metric, vector)
            return moment + ridge * ridge_action

        if self.controller in {"richardson", "heavy_ball", "certified_hb_pcg"}:
            if self.adaptive_heavy_ball:
                assert self.interval_head is not None
                operator = symmetric_effective_operator(preconditioner, normal_matrix)
                features = effective_spectrum_features(operator, equations.shape[1])
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
                preconditioned_final_residual = torch.einsum(
                    "bij,bj->bi", preconditioner, final_residual
                )
                preconditioned_rhs = torch.einsum(
                    "bij,bj->bi", preconditioner, rhs
                )
                residual_ratio = torch.einsum(
                    "bi,bi->b", final_residual, preconditioned_final_residual
                ) / torch.einsum(
                    "bi,bi->b", rhs, preconditioned_rhs
                ).clamp_min(1e-30)
                fallback_mask = residual_ratio > self.hybrid_residual_threshold
                fallback_indices = fallback_mask.nonzero(as_tuple=False).flatten()
                if fallback_indices.numel() > 0:
                    fallback_normal = normal_matrix.index_select(0, fallback_indices)
                    fallback_rhs = rhs.index_select(0, fallback_indices)
                    fallback_preconditioner = preconditioner.index_select(
                        0, fallback_indices
                    )

                    def fallback_hvp(vector: Tensor) -> Tensor:
                        return torch.einsum("bij,bj->bi", fallback_normal, vector)

                    fallback_solution, _, _ = run_pcg_state_machine(
                        fallback_hvp,
                        fallback_rhs,
                        fallback_preconditioner,
                        self.depth,
                    )
                    solution = solution.index_copy(
                        0, fallback_indices, fallback_solution
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
        else:
            assert self.interval_head is not None
            operator = symmetric_effective_operator(preconditioner, normal_matrix)
            features = effective_spectrum_features(operator, equations.shape[1])
            spectral_min, spectral_max = self.interval_head(features)
            spectral_min = spectral_min / self.interval_lower_calibration
            spectral_max = spectral_max * self.interval_upper_calibration
            solution, _, _ = run_chebyshev_state_machine(
                hvp,
                rhs,
                preconditioner,
                self.depth,
                spectral_min,
                spectral_max,
            )
            info.update(
                {
                    "spectral_min": spectral_min,
                    "spectral_max": spectral_max,
                    "spectral_features": features,
                }
            )
        info.update(
            {
                "normal_matrix": normal_matrix,
                "rhs": rhs,
                "preconditioner": preconditioner,
            }
        )
        return solution, info
