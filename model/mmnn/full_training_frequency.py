"""Fully trained periodic muP networks for controlled Fourier experiments.

The modules in this file use an explicit raw-weight parameterization.  A
hidden dense map is divided by ``sqrt(width)`` in the forward pass and a
factorized map ``U V^T`` is divided by ``sqrt(width * rank)``.  The scalar
readout is divided by ``width``.  Multiplying Euclidean gradients of the raw
parameters by ``width`` therefore gives the maximal-update flow used by the
companion experiment.

Unlike :mod:`mmnn.mup_right_factor`, every parameter block is trainable.
The output is centered at initialization without changing derivatives.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def periodic_embedding(x: torch.Tensor) -> torch.Tensor:
    """Return the unit-circle embedding ``(cos(x), sin(x))``."""
    return torch.stack((torch.cos(x), torch.sin(x)), dim=-1)


class _CenteredPeriodicNetwork(nn.Module):
    """Shared diagnostics for centered periodic feature-learning networks."""

    width: int
    affine_depth: int

    def _uncentered_forward(self) -> torch.Tensor:
        raise NotImplementedError

    def hidden_features(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _finish_initialization(self) -> None:
        with torch.no_grad():
            self.register_buffer("initial_output", self._uncentered_forward().clone())
            initial_features = self.hidden_features()
            for index, feature in enumerate(initial_features):
                self.register_buffer(f"initial_feature_{index}", feature.clone())

    def forward(self) -> torch.Tensor:
        return self._uncentered_forward() - self.initial_output

    def metric_scale(self, name: str) -> float:
        """Block multiplier for the explicit maximal-update gradient flow."""
        if name == "output_bias":
            return 1.0
        return float(self.width)

    def relative_feature_displacements(self) -> tuple[torch.Tensor, ...]:
        displacements = []
        for index, feature in enumerate(self.hidden_features()):
            initial = getattr(self, f"initial_feature_{index}")
            denominator = torch.linalg.vector_norm(initial).clamp_min(1.0e-12)
            displacements.append(
                torch.linalg.vector_norm(feature - initial) / denominator
            )
        return tuple(displacements)


class FullyTrainedPeriodicMLP(_CenteredPeriodicNetwork):
    """A fully trained periodic MLP in an explicit maximal-update scaling.

    ``affine_depth`` counts the input, hidden, and scalar-output affine maps.
    It must be at least two.  Thus depth three has two nonlinear hidden
    layers and one scalar readout.
    """

    def __init__(
        self,
        x: torch.Tensor,
        *,
        width: int,
        affine_depth: int,
        seed: int,
        bias_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least two")
        if affine_depth < 2:
            raise ValueError("affine_depth must be at least two")
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        self.register_buffer("input", periodic_embedding(x))
        self.input_weight = nn.Parameter(
            torch.randn(width, 2, generator=generator, device=x.device)
        )
        self.input_bias = nn.Parameter(
            bias_scale * torch.randn(width, generator=generator, device=x.device)
        )
        hidden_count = affine_depth - 2
        self.hidden_weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        width, width, generator=generator, device=x.device
                    )
                )
                for _ in range(hidden_count)
            ]
        )
        self.hidden_biases = nn.ParameterList(
            [
                nn.Parameter(
                    bias_scale
                    * torch.randn(width, generator=generator, device=x.device)
                )
                for _ in range(hidden_count)
            ]
        )
        self.readout = nn.Parameter(
            torch.randn(width, generator=generator, device=x.device)
        )
        self.output_bias = nn.Parameter(torch.zeros((), device=x.device))
        self.width = width
        self.affine_depth = affine_depth
        self._finish_initialization()

    def hidden_features(self) -> tuple[torch.Tensor, ...]:
        features: list[torch.Tensor] = []
        state = torch.relu(
            self.input @ self.input_weight.T / math.sqrt(2.0) + self.input_bias
        )
        features.append(state)
        for weight, bias in zip(
            self.hidden_weights, self.hidden_biases, strict=True
        ):
            state = torch.relu(state @ weight.T / math.sqrt(self.width) + bias)
            features.append(state)
        return tuple(features)

    def _uncentered_forward(self) -> torch.Tensor:
        state = self.hidden_features()[-1]
        return state @ self.readout / self.width + self.output_bias


class FullyTrainedPeriodicMMNN(_CenteredPeriodicNetwork):
    """A fully trained low-rank periodic MMNN.

    Every square hidden matrix is represented by a signed factorization
    ``U V^T / sqrt(rank)`` and both factors are trained.  The input and
    scalar-output maps are dense.  Consequently this is a full-training
    control, not the frozen-left/right-factor-only model of the first paper.
    """

    def __init__(
        self,
        x: torch.Tensor,
        *,
        width: int,
        affine_depth: int,
        rank: int,
        seed: int,
        bias_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("width must be at least two")
        if affine_depth < 3:
            raise ValueError("an MMNN needs at least one factorized hidden map")
        if not 1 <= rank <= width:
            raise ValueError(f"rank must lie in [1, width], got {rank=}")
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        self.register_buffer("input", periodic_embedding(x))
        self.input_weight = nn.Parameter(
            torch.randn(width, 2, generator=generator, device=x.device)
        )
        self.input_bias = nn.Parameter(
            bias_scale * torch.randn(width, generator=generator, device=x.device)
        )
        factorized_count = affine_depth - 2
        self.left_factors = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(width, rank, generator=generator, device=x.device)
                )
                for _ in range(factorized_count)
            ]
        )
        self.right_factors = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(width, rank, generator=generator, device=x.device)
                )
                for _ in range(factorized_count)
            ]
        )
        self.hidden_biases = nn.ParameterList(
            [
                nn.Parameter(
                    bias_scale
                    * torch.randn(width, generator=generator, device=x.device)
                )
                for _ in range(factorized_count)
            ]
        )
        self.readout = nn.Parameter(
            torch.randn(width, generator=generator, device=x.device)
        )
        self.output_bias = nn.Parameter(torch.zeros((), device=x.device))
        self.width = width
        self.affine_depth = affine_depth
        self.rank = rank
        self._finish_initialization()

    def hidden_features(self) -> tuple[torch.Tensor, ...]:
        features: list[torch.Tensor] = []
        state = torch.relu(
            self.input @ self.input_weight.T / math.sqrt(2.0) + self.input_bias
        )
        features.append(state)
        denominator = math.sqrt(self.width * self.rank)
        for left, right, bias in zip(
            self.left_factors,
            self.right_factors,
            self.hidden_biases,
            strict=True,
        ):
            state = torch.relu((state @ right) @ left.T / denominator + bias)
            features.append(state)
        return tuple(features)

    def _uncentered_forward(self) -> torch.Tensor:
        state = self.hidden_features()[-1]
        return state @ self.readout / self.width + self.output_bias


def trainable_parameter_names(model: nn.Module) -> Sequence[str]:
    """Stable helper used by optimizer code and algebra tests."""
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
