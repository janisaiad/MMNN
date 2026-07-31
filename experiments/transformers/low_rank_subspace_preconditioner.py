"""One-head recovery of the observable inverse-problem subspace.

Only the routing from weak-equation tokens to a rank-s subspace is learned.
The SPD inverse map on that subspace is the exact Ritz formula implied by the
ridge normal equations; it stays factorized throughout the solver loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass(frozen=True)
class FactorizedSubspaceInverse:
    """SPD map that is exact on a recovered subspace and ridge on its complement."""

    directions: Tensor
    projected_cholesky: Tensor
    ridge_precision: float

    def apply(self, vector: Tensor) -> Tensor:
        coordinates = torch.einsum("bks,bk->bs", self.directions, vector)
        projected_solution = torch.cholesky_solve(
            coordinates.unsqueeze(-1),
            self.projected_cholesky,
        ).squeeze(-1)
        projection = torch.einsum("bks,bs->bk", self.directions, coordinates)
        ritz_part = torch.einsum(
            "bks,bs->bk",
            self.directions,
            projected_solution,
        )
        return (vector - projection) / self.ridge_precision + ritz_part

    def dense(self) -> Tensor:
        """Materialize only for diagnostics and small unit tests."""

        batch_size, dimension, slots = self.directions.shape
        eye = torch.eye(
            dimension,
            device=self.directions.device,
            dtype=self.directions.dtype,
        ).expand(batch_size, -1, -1)
        slot_eye = torch.eye(
            slots,
            device=self.directions.device,
            dtype=self.directions.dtype,
        ).expand(batch_size, -1, -1)
        projected_inverse = torch.cholesky_solve(slot_eye, self.projected_cholesky)
        complement = eye - torch.einsum(
            "bks,bls->bkl",
            self.directions,
            self.directions,
        )
        ritz = torch.einsum(
            "bks,bst,blt->bkl",
            self.directions,
            projected_inverse,
            self.directions,
        )
        return complement / self.ridge_precision + ritz


def build_factorized_subspace_inverse(
    normal_matrix: Tensor,
    directions: Tensor,
    ridge_precision: float,
) -> FactorizedSubspaceInverse:
    """Build the exact fixed Ritz map for an orthonormal batch of directions."""

    projected_operator = torch.einsum(
        "bks,bkl,blt->bst",
        directions,
        normal_matrix,
        directions,
    )
    projected_cholesky = torch.linalg.cholesky(projected_operator)
    return FactorizedSubspaceInverse(
        directions=directions,
        projected_cholesky=projected_cholesky,
        ridge_precision=ridge_precision,
    )


def exact_observable_directions(
    equations: Tensor,
    rank: int,
    side: str = "primal",
) -> Tensor:
    """Deterministic prompt subspace used as an oracle and no-learning control.

    This operation is intentionally exposed: when exact weak features are
    already available and the complete observable rank is retained, learning
    the same subspace in the decoder would be redundant.
    """

    left, _, right_transpose = torch.linalg.svd(equations, full_matrices=False)
    if rank > min(equations.shape[-2:]):
        raise ValueError("rank exceeds the maximum observable rank")
    if side == "primal":
        return right_transpose.transpose(-1, -2)[:, :, :rank]
    if side == "dual":
        return left[:, :, :rank]
    raise ValueError(f"unknown side {side}")


class OneHeadObservableSubspace(nn.Module):
    """Route weak rows into s directions contained in ``range(G^T)``.

    Values are deliberately fixed to the normalized weak rows.  Therefore the
    learned head can select and combine observable directions but cannot
    fabricate a direction outside the prompt's row space.
    """

    def __init__(self, dimension: int, head_dimension: int, slots: int) -> None:
        super().__init__()
        if slots > dimension:
            raise ValueError("slots cannot exceed coefficient dimension")
        self.dimension = dimension
        self.head_dimension = head_dimension
        self.slots = slots
        self.key = nn.Linear(dimension, head_dimension, bias=False)
        self.slot_queries = nn.Parameter(
            torch.randn(slots, head_dimension) / math.sqrt(head_dimension)
        )
        with torch.no_grad():
            self.key.weight.zero_()
            diagonal = min(dimension, head_dimension)
            self.key.weight[:diagonal, :diagonal] = torch.eye(diagonal)

    def forward(
        self,
        equations: Tensor,
        normal_matrix: Tensor,
        ridge_precision: float,
        side: str = "primal",
        active_factor: Tensor | None = None,
    ) -> Tuple[FactorizedSubspaceInverse, Dict[str, Tensor]]:
        if equations.shape[1] < self.slots:
            raise ValueError("the prompt must contain at least as many rows as slots")
        normalized_rows = equations / equations.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        keys = self.key(normalized_rows)
        scores = torch.einsum("sd,bmd->bsm", self.slot_queries, keys)
        attention = torch.softmax(scores / math.sqrt(self.head_dimension), dim=-1)
        routed_rows = torch.einsum("bsm,bmk->bks", attention, normalized_rows)
        if side == "primal":
            raw_directions = routed_rows
        elif side in {"dual", "compressed_dual"}:
            # G q lies in range(G), the observable subspace on the dual side.
            dual_factor = equations if active_factor is None else active_factor
            raw_directions = torch.einsum("bmk,bks->bms", dual_factor, routed_rows)
        else:
            raise ValueError(f"unknown side {side}")
        directions = torch.linalg.qr(raw_directions, mode="reduced").Q
        inverse = build_factorized_subspace_inverse(
            normal_matrix,
            directions,
            ridge_precision,
        )
        return inverse, {
            "attention": attention,
            "directions": directions,
            "projected_cholesky": inverse.projected_cholesky,
        }
