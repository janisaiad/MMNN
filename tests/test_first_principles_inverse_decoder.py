import importlib.util
import sys
from pathlib import Path

import torch

TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "experiments" / "transformers"
for module_name in [
    "first_principles_decoder_cells",
    "low_rank_subspace_preconditioner",
    "first_principles_inverse_decoder",
]:
    module_path = TRANSFORMER_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

DECODER = sys.modules["first_principles_inverse_decoder"]


def test_primal_and_dual_are_the_same_ridge_estimator() -> None:
    torch.manual_seed(307)
    equations = torch.randn(5, 6, 9, dtype=torch.float64)
    observations = torch.randn(5, 6, dtype=torch.float64)
    data_precision, ridge_precision = 1.7, 0.4
    primal = DECODER.build_active_ridge_system(
        equations, observations, data_precision, ridge_precision, side="primal"
    )
    dual = DECODER.build_active_ridge_system(
        equations, observations, data_precision, ridge_precision, side="dual"
    )

    primal_solution = torch.linalg.solve(primal.normal_matrix, primal.rhs)
    dual_solution = dual.decode(torch.linalg.solve(dual.normal_matrix, dual.rhs))

    torch.testing.assert_close(primal_solution, dual_solution)


def test_compressed_dual_is_exactly_the_primal_estimator() -> None:
    torch.manual_seed(309)
    equations = torch.randn(4, 13, 6, dtype=torch.float64)
    observations = torch.randn(4, 13, dtype=torch.float64)
    primal = DECODER.build_active_ridge_system(
        equations, observations, 1.3, 0.2, side="primal"
    )
    compressed = DECODER.build_active_ridge_system(
        equations, observations, 1.3, 0.2, side="compressed_dual"
    )

    expected = torch.linalg.solve(primal.normal_matrix, primal.rhs)
    actual = compressed.decode(
        torch.linalg.solve(compressed.normal_matrix, compressed.rhs)
    )

    assert compressed.normal_matrix.shape[-1] == equations.shape[-1]
    torch.testing.assert_close(actual, expected)


def test_auto_representation_uses_smaller_active_space() -> None:
    observations = torch.randn(2, 5)
    wide = torch.randn(2, 5, 11)
    tall = torch.randn(2, 9, 4)

    assert DECODER.build_active_ridge_system(wide, observations, 1.0, 0.5).side == "dual"
    assert (
        DECODER.build_active_ridge_system(tall, torch.randn(2, 9), 1.0, 0.5).side
        == "primal"
    )


def test_decoder_has_one_learned_head_and_no_mlp() -> None:
    model = DECODER.FirstPrinciplesInverseDecoder(
        coefficient_dimension=10,
        head_dimension=6,
        slots=3,
        depth=4,
        solver_cell="pcg",
    )

    assert sum(isinstance(module, torch.nn.Linear) for module in model.modules()) == 1
    assert not any(isinstance(module, torch.nn.Sequential) for module in model.modules())
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())


def test_auto_decoder_runs_on_dual_side_and_backpropagates_to_head() -> None:
    torch.manual_seed(311)
    equations = torch.randn(3, 5, 9, dtype=torch.float64)
    observations = torch.randn(3, 5, dtype=torch.float64)
    model = DECODER.FirstPrinciplesInverseDecoder(
        coefficient_dimension=9,
        head_dimension=7,
        slots=3,
        depth=3,
        solver_cell="pcg",
    ).to(dtype=torch.float64)

    prediction, info = model(equations, observations, 1.0, 0.6)
    prediction.square().mean().backward()

    assert prediction.shape == (3, 9)
    assert info["side"] == "dual"
    assert model.subspace_head.slot_queries.grad is not None


def test_full_exact_prompt_subspace_solves_in_one_pcg_block() -> None:
    torch.manual_seed(313)
    equations = torch.randn(4, 3, 8, dtype=torch.float64)
    observations = torch.randn(4, 3, dtype=torch.float64)
    model = DECODER.FirstPrinciplesInverseDecoder(
        coefficient_dimension=8,
        head_dimension=5,
        slots=3,
        depth=1,
        solver_cell="pcg",
        representation="dual",
        subspace_mode="exact_prompt",
    ).to(dtype=torch.float64)

    prediction, _ = model(equations, observations, 1.4, 0.7)
    system = DECODER.build_active_ridge_system(
        equations, observations, 1.4, 0.7, side="primal"
    )
    expected = torch.linalg.solve(system.normal_matrix, system.rhs)

    torch.testing.assert_close(prediction, expected)


def test_chebyshev_interval_mlp_is_positive_ordered_and_trainable() -> None:
    torch.manual_seed(317)
    head = DECODER.PromptSpectralIntervalMLP(hidden_dimension=9).to(dtype=torch.float64)
    features = torch.randn(6, 7, dtype=torch.float64)

    lower, upper = head(features)
    loss = DECODER.spectral_interval_coverage_loss(
        lower,
        upper,
        true_min=torch.full_like(lower, 0.3),
        true_max=torch.full_like(upper, 4.0),
    )
    loss.backward()

    assert torch.all(lower > 0)
    assert torch.all(upper > lower)
    assert head.network[0].weight.grad is not None


def test_exact_prompt_chebyshev_solves_in_one_block_with_unit_interval() -> None:
    torch.manual_seed(319)
    equations = torch.randn(3, 4, 9, dtype=torch.float64)
    observations = torch.randn(3, 4, dtype=torch.float64)
    model = DECODER.FirstPrinciplesInverseDecoder(
        coefficient_dimension=9,
        head_dimension=6,
        slots=4,
        depth=1,
        solver_cell="chebyshev",
        representation="dual",
        subspace_mode="exact_prompt",
    ).to(dtype=torch.float64)

    prediction, info = model(
        equations,
        observations,
        1.2,
        0.5,
        spectral_bounds=(1.0, 1.0),
    )
    primal = DECODER.build_active_ridge_system(
        equations, observations, 1.2, 0.5, side="primal"
    )
    expected = torch.linalg.solve(primal.normal_matrix, primal.rhs)

    torch.testing.assert_close(prediction, expected)
    assert info["spectral_min"].item() == 1.0
    assert info["spectral_max"].item() == 1.0


def test_only_chebyshev_variant_contains_an_mlp() -> None:
    common = dict(
        coefficient_dimension=8,
        head_dimension=6,
        slots=3,
        depth=4,
    )
    heavy_ball = DECODER.FirstPrinciplesInverseDecoder(**common, solver_cell="heavy_ball")
    pcg = DECODER.FirstPrinciplesInverseDecoder(**common, solver_cell="pcg")
    chebyshev = DECODER.FirstPrinciplesInverseDecoder(**common, solver_cell="chebyshev")

    assert not any(isinstance(module, torch.nn.Sequential) for module in heavy_ball.modules())
    assert not any(isinstance(module, torch.nn.Sequential) for module in pcg.modules())
    assert sum(isinstance(module, torch.nn.Sequential) for module in chebyshev.modules()) == 1
