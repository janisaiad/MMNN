import sys
from pathlib import Path

import torch

TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "experiments" / "transformers"
sys.path.insert(0, str(TRANSFORMER_DIR))

from compare_global_prompt_preconditioners import (  # noqa: E402
    haar_orthogonal,
    rotate_system,
    spectral_measure,
)


def test_haar_rotation_preserves_the_normal_solve() -> None:
    generator = torch.Generator().manual_seed(17)
    factor = torch.randn(6, 5, 5, generator=generator, dtype=torch.float64)
    normal = factor.transpose(-1, -2) @ factor
    normal = normal + 0.2 * torch.eye(5, dtype=torch.float64)
    rhs = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    rotation = haar_orthogonal(
        6,
        5,
        device=torch.device("cpu"),
        dtype=torch.float64,
        generator=generator,
    )
    identity = torch.eye(5, dtype=torch.float64).expand(6, -1, -1)
    torch.testing.assert_close(
        rotation.transpose(-1, -2) @ rotation,
        identity,
        rtol=1e-12,
        atol=1e-12,
    )
    rotated_normal, rotated_rhs = rotate_system(normal, rhs, rotation)
    original_solution = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
    rotated_solution = torch.linalg.solve(
        rotated_normal,
        rotated_rhs.unsqueeze(-1),
    ).squeeze(-1)
    expected_solution = torch.einsum("bji,bj->bi", rotation, original_solution)
    torch.testing.assert_close(
        rotated_solution,
        expected_solution,
        rtol=1e-11,
        atol=1e-11,
    )


def test_direct_inverse_whitens_the_effective_spectrum() -> None:
    generator = torch.Generator().manual_seed(23)
    factor = torch.randn(8, 4, 4, generator=generator, dtype=torch.float64)
    normal = factor.transpose(-1, -2) @ factor
    normal = normal + 0.5 * torch.eye(4, dtype=torch.float64)
    rhs = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64).expand(8, -1, -1)
    inverse = torch.linalg.solve(normal, identity)
    eigenvalues, weights = spectral_measure(normal, rhs, inverse)
    torch.testing.assert_close(
        eigenvalues,
        torch.ones_like(eigenvalues),
        rtol=2e-11,
        atol=2e-11,
    )
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(8, dtype=torch.float64),
    )
