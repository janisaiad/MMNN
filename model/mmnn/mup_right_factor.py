"""Centered maximum-update scaling for a frozen-left/right-factor MMNN."""

from __future__ import annotations

import math

import torch
from torch import nn


class CenteredRightFactorMuP(nn.Module):
    """Three-affine-map, signed low-rank MMNN with only ``V`` trainable.

    The input grid is fixed because the population experiment repeatedly
    evaluates the same quadrature nodes. Centering the output at ``V0`` makes
    lazy/rich comparisons well-conditioned without altering derivatives.
    """

    def __init__(
        self,
        x: torch.Tensor,
        *,
        width: int,
        rank: int,
        gamma: float,
        seed: int,
        bias_scale_1: float,
        bias_scale_2: float,
    ) -> None:
        super().__init__()
        if not 1 <= rank <= width:
            raise ValueError(f"rank must be in [1, width], got {rank=}, {width=}")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

        phase = 2.0 * math.pi * torch.rand(
            width, generator=generator, device=x.device
        )
        bias1 = bias_scale_1 * torch.randn(
            width, generator=generator, device=x.device
        )
        h = math.sqrt(2.0) * torch.relu(
            torch.cos(x[:, None] - phase[None, :]) + bias1[None, :]
        )
        self.register_buffer("h", h)
        self.register_buffer(
            "U", torch.randn(width, rank, generator=generator, device=x.device)
        )
        self.register_buffer(
            "beta",
            bias_scale_2
            * torch.randn(width, generator=generator, device=x.device),
        )
        signs = torch.randint(
            0, 2, (width,), generator=generator, device=x.device
        ).float()
        self.register_buffer("readout", 2.0 * signs - 1.0)
        v0 = torch.randn(width, rank, generator=generator, device=x.device)
        self.V = nn.Parameter(v0.clone())
        self.register_buffer("V0", v0)
        self.width = width
        self.rank = rank
        self.gamma = gamma

        with torch.no_grad():
            self.register_buffer("s0", self._preactivation(self.V0))

    def latent(self, value: torch.Tensor | None = None) -> torch.Tensor:
        matrix = self.V if value is None else value
        return (self.h @ matrix) / math.sqrt(self.width)

    def _preactivation(self, value: torch.Tensor) -> torch.Tensor:
        return self.latent(value) @ self.U.T / math.sqrt(self.rank) + self.beta

    def preactivation(self) -> torch.Tensor:
        return self._preactivation(self.V)

    def forward(self) -> torch.Tensor:
        current = torch.relu(self.preactivation())
        initial = torch.relu(self.s0)
        return ((current - initial) @ self.readout) / (
            self.gamma * self.width
        )

    @property
    def metric_scale(self) -> float:
        """Euclidean-gradient multiplier in the masked muP flow."""
        return self.gamma**2 * self.width
