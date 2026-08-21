"""Exact spectral-power directions used in controlled Muon experiments."""

from __future__ import annotations

import math

import torch


def spectral_power_direction(
    gradient: torch.Tensor,
    power: float,
    *,
    relative_floor: float = 1.0e-7,
) -> torch.Tensor:
    """Return the RMS-normalized exact ``U diag(s**power) V^T`` direction.

    The normalization follows the block map used in the companion Muon/DMFT
    analysis,

    ``Phi_p(G) = sqrt(rows * cols) Psi_p(G) / ||Psi_p(G)||_F``.

    Zero singular values stay zero, including for ``power == 0``.  This makes
    the polar member the compact polar factor and avoids inventing directions
    outside the gradient's row or column space.
    """
    if gradient.ndim != 2:
        raise ValueError("spectral_power_direction expects a matrix")
    if power < 0.0 or power > 1.0:
        raise ValueError("power must lie in [0, 1]")
    if relative_floor < 0.0:
        raise ValueError("relative_floor must be nonnegative")
    if not torch.any(gradient):
        return torch.zeros_like(gradient)

    if power == 1.0:
        transformed = gradient
    else:
        # Do not recover singular vectors from G^T G or G G^T here.  Squaring
        # the condition number makes the small singular sectors used by the
        # polar member (power == 0) numerically unreliable in float32.
        if gradient.is_cuda:
            left, singular_values, right_h = torch.linalg.svd(
                gradient,
                full_matrices=False,
                driver="gesvd",
            )
        else:
            left, singular_values, right_h = torch.linalg.svd(
                gradient,
                full_matrices=False,
            )
        threshold = relative_floor * singular_values.max()
        active = singular_values > threshold
        powered = singular_values[active].pow(power)
        transformed = (left[:, active] * powered.unsqueeze(0)) @ right_h[active]
    norm = torch.linalg.vector_norm(transformed)
    if not torch.isfinite(norm) or float(norm) == 0.0:
        return torch.zeros_like(gradient)
    return transformed * (math.sqrt(gradient.numel()) / norm)


def spectral_descent_pairing(gradient: torch.Tensor, power: float) -> torch.Tensor:
    """Return ``<G, Phi_p(G)>`` for descent and diagnostics."""
    return torch.sum(gradient * spectral_power_direction(gradient, power))
