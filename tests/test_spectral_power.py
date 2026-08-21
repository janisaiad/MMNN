from __future__ import annotations

import math

import torch

from mmnn.spectral_power import (
    spectral_descent_pairing,
    spectral_power_direction,
)


def test_spectral_power_has_prescribed_singular_values_and_rms() -> None:
    torch.manual_seed(4)
    gradient = torch.randn(7, 5, dtype=torch.float64)
    for power in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0):
        direction = spectral_power_direction(gradient, power)
        observed = torch.linalg.svdvals(direction)
        source = torch.linalg.svdvals(gradient).pow(power)
        if power == 0.0:
            source = torch.ones_like(source)
        expected = source * (
            math.sqrt(gradient.numel()) / torch.linalg.vector_norm(source)
        )
        torch.testing.assert_close(observed, expected)
        torch.testing.assert_close(
            torch.mean(direction.square()),
            torch.ones((), dtype=direction.dtype),
        )


def test_spectral_power_is_strict_descent_for_nonzero_gradient() -> None:
    torch.manual_seed(9)
    gradient = torch.randn(8, 3, dtype=torch.float64)
    for power in (0.0, 1.0 / 7.0, 1.0 / 3.0, 1.0):
        assert float(spectral_descent_pairing(gradient, power)) > 0.0


def test_rank_deficient_polar_direction_preserves_rank() -> None:
    left = torch.tensor([[1.0], [2.0], [-1.0]])
    right = torch.tensor([[0.5], [-3.0]])
    gradient = left @ right.T
    direction = spectral_power_direction(gradient, 0.0)
    assert int(torch.linalg.matrix_rank(direction)) == 1


def test_float32_polar_direction_resolves_ill_conditioned_sectors() -> None:
    torch.manual_seed(21)
    left, _ = torch.linalg.qr(torch.randn(7, 5))
    right, _ = torch.linalg.qr(torch.randn(5, 5))
    singular_values = torch.tensor([1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8])
    gradient = (left * singular_values.unsqueeze(0)) @ right.T

    direction = spectral_power_direction(gradient, 0.0)
    expected = left[:, :4] @ right[:, :4].T
    expected *= math.sqrt(gradient.numel()) / torch.linalg.vector_norm(expected)

    cosine = torch.nn.functional.cosine_similarity(
        direction.flatten(),
        expected.flatten(),
        dim=0,
    )
    assert float(cosine) > 0.995


def test_orthogonal_sector_velocity_has_predicted_spectral_power() -> None:
    residual = torch.tensor([0.8, -0.2, 0.05], dtype=torch.float64)
    jacobian_amplitude = torch.tensor([0.7, 0.3, 0.1], dtype=torch.float64)
    gradient = torch.diag(residual * jacobian_amplitude)
    for power in (0.0, 1.0 / 3.0, 2.0 / 3.0):
        direction = spectral_power_direction(gradient, power)
        observed_velocity = -jacobian_amplitude * torch.diag(direction)
        unnormalized = (
            -torch.sign(residual)
            * residual.abs().pow(power)
            * jacobian_amplitude.pow(power + 1.0)
        )
        common_scale = observed_velocity / unnormalized
        torch.testing.assert_close(
            common_scale,
            torch.full_like(common_scale, common_scale[0]),
        )
