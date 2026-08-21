"""Low-rank MMNN layer with a frozen left and trainable right factor."""

from __future__ import annotations

import math

import torch
from torch import nn


class RightFactorMMNN(nn.Module):
    """Three-affine-map MMNN with only the right inner factor trainable.

    For row-vector inputs, the forward map is

        h(x) = ReLU(B psi(x) + b_1),
        f_V(x) = c^T ReLU(U V^T h(x) + b_2),

    where psi(x)=(cos x, sin x). In row-vector notation the middle
    multiplication is h V U^T, hence the dense inner matrix is U V^T.
    """

    def __init__(
        self,
        *,
        feature_width: int,
        outer_width: int,
        rank: int,
        seed: int,
        device: torch.device,
        bias_scale_1: float = 0.5,
        bias_scale_2: float = 0.2,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        def normal(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, device=device)

        self.register_buffer("B", normal(feature_width, 2))
        self.register_buffer("b1", bias_scale_1 * normal(feature_width))
        self.register_buffer(
            "U", normal(outer_width, rank) / math.sqrt(float(rank))
        )
        self.register_buffer("b2", bias_scale_2 * normal(outer_width))
        self.register_buffer(
            "readout", normal(outer_width) / math.sqrt(float(outer_width))
        )
        self.V = nn.Parameter(
            init_scale * normal(feature_width, rank) / math.sqrt(feature_width)
        )

    def first_features(self, x: torch.Tensor) -> torch.Tensor:
        periodic_input = torch.stack((torch.cos(x), torch.sin(x)), dim=1)
        return torch.relu(periodic_input @ self.B.T + self.b1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first_features(x)
        inner = (h @ self.V) @ self.U.T + self.b2
        return torch.relu(inner) @ self.readout

