import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "transformers"
    / "structured_one_head_heavyball.py"
)
TRANSFORMER_DIR = MODULE_PATH.parent
if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))
SPEC = importlib.util.spec_from_file_location("structured_one_head_lab", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LAB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAB
SPEC.loader.exec_module(LAB)

OneHeadSpectralPreconditioner = LAB.OneHeadSpectralPreconditioner
StructuredOneHeadHeavyBall = LAB.StructuredOneHeadHeavyBall


def test_one_head_slot_preconditioner_is_symmetric_positive_definite() -> None:
    torch.manual_seed(3)
    batch_size, prompt_length, dimension = 5, 24, 6
    equations = torch.randn(batch_size, prompt_length, dimension, dtype=torch.float64)
    normal_matrix = torch.einsum("bmk,bml->bkl", equations, equations)
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=8,
        slots=2,
        max_strength=0.95,
        strength_init=0.05,
    ).to(dtype=torch.float64)

    preconditioner, info = head(equations, normal_matrix)

    torch.testing.assert_close(preconditioner, preconditioner.transpose(-1, -2))
    assert torch.linalg.eigvalsh(preconditioner).min().item() > 0.0
    assert info["attention"].shape == (batch_size, 2, dimension)
    torch.testing.assert_close(
        info["attention"].sum(dim=-1),
        torch.ones(batch_size, 2, dtype=torch.float64),
    )


def test_structured_decoder_contains_no_mlp_or_multihead_module() -> None:
    model = StructuredOneHeadHeavyBall(
        dimension=8,
        depth=4,
        head_dimension=8,
        slots=2,
        max_strength=0.95,
        strength_init=0.05,
        head_mode="slots",
        spectral_lmax_bound=1.1,
        step_init=1.0,
        momentum_init=0.05,
    )

    assert not any(isinstance(module, torch.nn.Sequential) for module in model.modules())
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
    assert sum(isinstance(module, torch.nn.Linear) for module in model.modules()) == 2


def test_pcg_cell_has_no_learned_solver_coefficients() -> None:
    model = StructuredOneHeadHeavyBall(
        dimension=8,
        depth=4,
        head_dimension=8,
        slots=2,
        max_strength=0.95,
        strength_init=0.05,
        head_mode="slots",
        spectral_lmax_bound=1.1,
        step_init=1.0,
        momentum_init=0.05,
        solver_cell="pcg",
    )

    parameter_names = dict(model.named_parameters())
    assert "raw_step" not in parameter_names
    assert "raw_momentum" not in parameter_names


def test_block_jacobi_base_plus_head_remains_spd() -> None:
    torch.manual_seed(9)
    dimension = 8
    factors = torch.randn(4, 20, dimension, dtype=torch.float64)
    normal_matrix = torch.einsum("bmk,bml->bkl", factors, factors)
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=8,
        slots=2,
        max_strength=0.95,
        strength_init=0.2,
        base_preconditioner="block_jacobi",
        base_blocks=2,
    ).to(dtype=torch.float64)

    preconditioner, info = head(factors, normal_matrix)

    torch.testing.assert_close(preconditioner, preconditioner.transpose(-1, -2))
    assert torch.linalg.eigvalsh(preconditioner).min().item() > 0.0
    base_inverse = info["base_inverse"]
    assert torch.count_nonzero(base_inverse[:, :4, 4:]).item() == 0
    assert torch.count_nonzero(base_inverse[:, 4:, :4]).item() == 0


def test_inverse_prompt_strength_scaling_is_exact() -> None:
    torch.manual_seed(11)
    dimension = 6
    factors = torch.randn(3, 32, dimension, dtype=torch.float64)
    normal_matrix = torch.einsum("bmk,bml->bkl", factors, factors)
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=6,
        slots=2,
        max_strength=0.95,
        strength_init=0.1,
        strength_scaling="inverse_prompt",
        reference_prompt_length=32,
    ).to(dtype=torch.float64)

    _, reference_info = head(factors, normal_matrix)
    _, short_info = head(factors[:, :16], normal_matrix)

    torch.testing.assert_close(
        short_info["strengths"],
        2.0 * reference_info["strengths"],
    )


def test_qr_slots_are_exactly_orthonormal() -> None:
    torch.manual_seed(13)
    dimension, slots = 10, 4
    factors = torch.randn(3, 40, dimension, dtype=torch.float64)
    normal_matrix = torch.einsum("bmk,bml->bkl", factors, factors)
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=10,
        slots=slots,
        max_strength=0.95,
        strength_init=0.1,
        slot_orthogonalization="qr",
    ).to(dtype=torch.float64)

    _, info = head(factors, normal_matrix)
    gram = torch.einsum("bks,bkt->bst", info["directions"], info["directions"])

    torch.testing.assert_close(
        gram,
        torch.eye(slots, dtype=torch.float64).expand(3, -1, -1),
    )


def test_signed_negative_spectral_correction_remains_spd() -> None:
    torch.manual_seed(19)
    dimension = 8
    factors = torch.randn(4, 32, dimension, dtype=torch.float64)
    normal_matrix = torch.einsum("bmk,bml->bkl", factors, factors)
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=8,
        slots=3,
        max_strength=0.95,
        strength_init=-0.8,
        slot_orthogonalization="qr",
        correction_mode="signed",
    ).to(dtype=torch.float64)

    preconditioner, info = head(factors, normal_matrix)

    assert info["strengths"].max().item() < 0.0
    assert torch.linalg.eigvalsh(preconditioner).min().item() > 0.0


def test_ritz_correction_is_spd_without_learned_strengths() -> None:
    torch.manual_seed(23)
    dimension, slots = 8, 3
    factors = torch.randn(5, 32, dimension, dtype=torch.float64)
    normal_matrix = torch.einsum("bmk,bml->bkl", factors, factors)
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=8,
        slots=slots,
        max_strength=0.95,
        strength_init=0.1,
        slot_orthogonalization="qr",
        correction_mode="ritz",
    ).to(dtype=torch.float64)

    preconditioner, info = head(factors, normal_matrix)

    assert head.raw_strength is None
    assert torch.linalg.eigvalsh(preconditioner).min().item() > 0.0
    slot_gram = torch.einsum(
        "bks,bkl,blt->bst",
        info["directions"],
        info["normalized_inverse"],
        info["directions"],
    )
    normalized_operator = torch.einsum(
        "bki,bkl,blj->bij",
        torch.linalg.cholesky(info["base_inverse"]),
        normal_matrix,
        torch.linalg.cholesky(info["base_inverse"]),
    )
    projected_operator = torch.einsum(
        "bks,bkl,blt->bst",
        info["directions"],
        normalized_operator,
        info["directions"],
    )
    torch.testing.assert_close(
        slot_gram,
        torch.linalg.inv(projected_operator),
    )


def test_power_refinement_improves_fast_subspace_overlap() -> None:
    torch.manual_seed(29)
    dimension, slots = 10, 2
    orthogonal = torch.linalg.qr(torch.randn(dimension, dimension, dtype=torch.float64)).Q
    spectrum = torch.logspace(0, 3, dimension, dtype=torch.float64)
    normal_matrix = (orthogonal @ torch.diag(spectrum) @ orthogonal.T).expand(4, -1, -1)
    equations = torch.randn(4, 24, dimension, dtype=torch.float64)
    base_head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=10,
        slots=slots,
        max_strength=0.95,
        strength_init=0.1,
        slot_orthogonalization="qr",
        correction_mode="ritz",
        subspace_refinement_steps=0,
    ).to(dtype=torch.float64)
    refined_head = OneHeadSpectralPreconditioner(
        dimension=dimension,
        head_dimension=10,
        slots=slots,
        max_strength=0.95,
        strength_init=0.1,
        slot_orthogonalization="qr",
        correction_mode="ritz",
        subspace_refinement_steps=3,
    ).to(dtype=torch.float64)
    refined_head.load_state_dict(base_head.state_dict())

    _, base_info = base_head(equations, normal_matrix)
    _, refined_info = refined_head(equations, normal_matrix)
    diagonal = torch.diagonal(normal_matrix, dim1=-2, dim2=-1)
    normalized = (
        diagonal.rsqrt()[:, :, None]
        * normal_matrix
        * diagonal.rsqrt()[:, None, :]
    )
    fast_space = torch.linalg.eigh(normalized).eigenvectors[:, :, -slots:]

    def overlap(directions: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_norm(
            torch.einsum("bki,bkj->bij", fast_space, directions),
            ord="fro",
            dim=(-2, -1),
        ).square().mean() / slots

    assert overlap(refined_info["directions"]) > overlap(base_info["directions"])


def test_effective_condition_loss_is_zero_for_exact_inverse() -> None:
    torch.manual_seed(31)
    factor = torch.randn(4, 7, 7, dtype=torch.float64)
    normal_matrix = factor.transpose(-1, -2) @ factor
    normal_matrix = normal_matrix + torch.eye(7, dtype=torch.float64)
    exact_inverse = torch.linalg.inv(normal_matrix)

    exact_loss = LAB.effective_log_condition_loss(exact_inverse, normal_matrix)
    identity_loss = LAB.effective_log_condition_loss(
        torch.eye(7, dtype=torch.float64).expand(4, -1, -1),
        normal_matrix,
    )

    torch.testing.assert_close(exact_loss, torch.zeros_like(exact_loss), atol=1e-8, rtol=0)
    assert identity_loss > exact_loss
