import importlib.util
import sys
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "transformers"
    / "low_rank_subspace_preconditioner.py"
)
SPEC = importlib.util.spec_from_file_location("low_rank_subspace", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_oracle_observable_subspace_gives_exact_ridge_inverse() -> None:
    torch.manual_seed(211)
    batch_size, rows, dimension = 3, 4, 9
    ridge_precision = 0.7
    equations = torch.randn(batch_size, rows, dimension, dtype=torch.float64)
    normal_matrix = equations.transpose(-1, -2) @ equations
    normal_matrix = normal_matrix + ridge_precision * torch.eye(
        dimension, dtype=torch.float64
    )
    directions = torch.linalg.qr(equations.transpose(-1, -2), mode="reduced").Q
    factorized = MODULE.build_factorized_subspace_inverse(
        normal_matrix,
        directions,
        ridge_precision,
    )
    vector = torch.randn(batch_size, dimension, dtype=torch.float64)

    actual = factorized.apply(vector)
    expected = torch.linalg.solve(normal_matrix, vector)

    torch.testing.assert_close(actual, expected)


def test_factorized_application_matches_dense_diagnostic() -> None:
    torch.manual_seed(223)
    batch_size, dimension, slots = 4, 10, 3
    factor = torch.randn(batch_size, dimension, dimension, dtype=torch.float64)
    normal_matrix = factor.transpose(-1, -2) @ factor
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    directions = torch.linalg.qr(
        torch.randn(batch_size, dimension, slots, dtype=torch.float64),
        mode="reduced",
    ).Q
    factorized = MODULE.build_factorized_subspace_inverse(
        normal_matrix,
        directions,
        ridge_precision=1.0,
    )
    vector = torch.randn(batch_size, dimension, dtype=torch.float64)

    torch.testing.assert_close(
        factorized.apply(vector),
        torch.einsum("bkl,bl->bk", factorized.dense(), vector),
    )
    assert torch.linalg.eigvalsh(factorized.dense()).min().item() > 0.0


def test_one_head_cannot_leave_prompt_observable_subspace() -> None:
    torch.manual_seed(227)
    batch_size, rows, dimension, slots = 5, 6, 11, 3
    equations = torch.randn(batch_size, rows, dimension, dtype=torch.float64)
    normal_matrix = equations.transpose(-1, -2) @ equations
    normal_matrix = normal_matrix + torch.eye(dimension, dtype=torch.float64)
    head = MODULE.OneHeadObservableSubspace(
        dimension=dimension,
        head_dimension=7,
        slots=slots,
    ).to(dtype=torch.float64)

    _, info = head(equations, normal_matrix, ridge_precision=1.0)
    row_space = torch.linalg.qr(equations.transpose(-1, -2), mode="reduced").Q
    projected = torch.einsum(
        "bkr,blr,bls->bks",
        row_space,
        row_space,
        info["directions"],
    )

    torch.testing.assert_close(projected, info["directions"])
    assert sum(isinstance(module, torch.nn.Linear) for module in head.modules()) == 1
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in head.modules())


def test_dual_head_cannot_leave_dual_observable_subspace() -> None:
    torch.manual_seed(229)
    batch_size, rows, dimension, slots = 4, 7, 12, 3
    equations = torch.randn(batch_size, rows, dimension, dtype=torch.float64)
    dual_normal = equations @ equations.transpose(-1, -2)
    dual_normal = dual_normal + torch.eye(rows, dtype=torch.float64)
    head = MODULE.OneHeadObservableSubspace(dimension, 8, slots).to(dtype=torch.float64)

    _, info = head(equations, dual_normal, ridge_precision=1.0, side="dual")
    dual_space = torch.linalg.qr(equations, mode="reduced").Q
    projected = torch.einsum(
        "bmr,bnr,bns->bms",
        dual_space,
        dual_space,
        info["directions"],
    )

    torch.testing.assert_close(projected, info["directions"])


def test_exact_prompt_directions_recover_both_sides() -> None:
    torch.manual_seed(233)
    equations = torch.randn(3, 4, 9, dtype=torch.float64)
    primal = MODULE.exact_observable_directions(equations, rank=4, side="primal")
    dual = MODULE.exact_observable_directions(equations, rank=4, side="dual")

    torch.testing.assert_close(
        torch.einsum("bkr,bks->brs", primal, primal),
        torch.eye(4, dtype=torch.float64).expand(3, -1, -1),
    )
    torch.testing.assert_close(
        torch.einsum("bmr,bms->brs", dual, dual),
        torch.eye(4, dtype=torch.float64).expand(3, -1, -1),
    )
