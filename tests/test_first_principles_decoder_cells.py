import importlib.util
import sys
from pathlib import Path

import torch

TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "experiments" / "transformers"
if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))
for module_name in [
    "constructive_weakform_richardson_transformer",
    "first_principles_decoder_cells",
    "predict_heavy_ball_hyperparameters",
]:
    module_path = TRANSFORMER_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

REFERENCE = sys.modules["constructive_weakform_richardson_transformer"]
CELLS = sys.modules["first_principles_decoder_cells"]
HB_PREDICTOR = sys.modules["predict_heavy_ball_hyperparameters"]


def random_spd_problem(batch_size: int = 4, dimension: int = 7):
    torch.manual_seed(101)
    factor = torch.randn(batch_size, dimension, dimension, dtype=torch.float64)
    matrix = factor.transpose(-1, -2) @ factor
    matrix = matrix + 0.5 * torch.eye(dimension, dtype=torch.float64)
    rhs = torch.randn(batch_size, dimension, dtype=torch.float64)
    diagonal_inverse = torch.diag_embed(
        torch.diagonal(matrix, dim1=-2, dim2=-1).reciprocal()
    )
    return matrix, rhs, diagonal_inverse


def test_fixed_linear_attention_hvp_equals_normal_matrix_product() -> None:
    torch.manual_seed(103)
    equations = torch.randn(5, 19, 8, dtype=torch.float64)
    vector = torch.randn(5, 8, dtype=torch.float64)
    noise_precision, prior_precision = 2.5, 0.3
    normal_matrix = (
        noise_precision * equations.transpose(-1, -2) @ equations
        + prior_precision * torch.eye(8, dtype=torch.float64)
    )

    actual = CELLS.fixed_prompt_linear_attention_hvp(
        equations,
        vector,
        noise_precision,
        prior_precision,
    )
    expected = torch.einsum("bkl,bl->bk", normal_matrix, vector)

    torch.testing.assert_close(actual, expected)


def test_heavy_ball_state_machine_matches_reference_solver_exactly() -> None:
    matrix, rhs, preconditioner = random_spd_problem()
    hvp = lambda vector: torch.einsum("bkl,bl->bk", matrix, vector)
    step_size = torch.tensor(0.012, dtype=torch.float64)
    momentum = torch.tensor(0.35, dtype=torch.float64)

    actual = CELLS.run_heavy_ball_state_machine(
        hvp,
        rhs,
        preconditioner,
        depth=6,
        step_size=step_size,
        momentum=momentum,
    )[0]
    expected = REFERENCE._heavy_ball_solve(
        hvp,
        rhs,
        preconditioner,
        depth=6,
        step_size=step_size,
        momentum=momentum,
    )[0]

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_pcg_state_machine_matches_reference_solver_exactly() -> None:
    matrix, rhs, preconditioner = random_spd_problem()
    hvp = lambda vector: torch.einsum("bkl,bl->bk", matrix, vector)

    actual = CELLS.run_pcg_state_machine(
        hvp,
        rhs,
        preconditioner,
        depth=6,
    )[0]
    expected = REFERENCE._pcg_solve(
        hvp,
        rhs,
        preconditioner,
        depth=6,
    )[0]

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_each_macro_block_uses_exactly_one_hvp() -> None:
    matrix, rhs, preconditioner = random_spd_problem()
    calls = {"count": 0}

    def hvp(vector: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return torch.einsum("bkl,bl->bk", matrix, vector)

    CELLS.run_pcg_state_machine(hvp, rhs, preconditioner, depth=5)
    assert calls["count"] == 5

    calls["count"] = 0
    CELLS.run_heavy_ball_state_machine(
        hvp,
        rhs,
        preconditioner,
        depth=5,
        step_size=0.01,
        momentum=0.2,
    )
    assert calls["count"] == 5


def test_pcg_work_tokens_are_exact_contractions_and_quotients() -> None:
    matrix, rhs, preconditioner = random_spd_problem()
    hvp = lambda vector: torch.einsum("bkl,bl->bk", matrix, vector)
    state = CELLS.initialize_pcg(rhs, preconditioner)
    step = CELLS.pcg_macro_block(state, hvp, preconditioner)

    torch.testing.assert_close(step.operator_direction, hvp(state.direction))
    torch.testing.assert_close(
        step.delta,
        torch.einsum("bk,bk->b", state.direction, step.operator_direction),
    )
    torch.testing.assert_close(step.alpha, state.rho / step.delta)
    torch.testing.assert_close(
        step.state.residual,
        rhs - step.alpha[:, None] * step.operator_direction,
    )
    torch.testing.assert_close(step.beta, step.state.rho / state.rho)


def test_tied_chebyshev_state_machine_matches_reference_exactly() -> None:
    matrix, rhs, preconditioner = random_spd_problem()
    hvp = lambda vector: torch.einsum("bkl,bl->bk", matrix, vector)
    chol = torch.linalg.cholesky(preconditioner)
    symmetric = chol.transpose(-1, -2) @ matrix @ chol
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    spectral_min = eigenvalues[:, 0]
    spectral_max = eigenvalues[:, -1]

    actual = CELLS.run_chebyshev_state_machine(
        hvp,
        rhs,
        preconditioner,
        depth=7,
        spectral_min=spectral_min,
        spectral_max=spectral_max,
    )[0]
    expected = REFERENCE._chebyshev_solve(
        hvp,
        rhs,
        preconditioner,
        depth=7,
        lmin=spectral_min,
        lmax=spectral_max,
    )[0]

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_precomputed_chebyshev_schedule_matches_scalar_token_recurrence() -> None:
    matrix, rhs, preconditioner = random_spd_problem()
    hvp = lambda vector: torch.einsum("bkl,bl->bk", matrix, vector)
    chol = torch.linalg.cholesky(preconditioner)
    symmetric = chol.transpose(-1, -2) @ matrix @ chol
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    spectral_min = eigenvalues[:, 0]
    spectral_max = eigenvalues[:, -1]
    expected = CELLS.run_chebyshev_state_machine(
        hvp,
        rhs,
        preconditioner,
        depth=7,
        spectral_min=spectral_min,
        spectral_max=spectral_max,
    )[0]
    step_schedule, momentum_schedule = CELLS.chebyshev_coefficient_schedule(
        rhs,
        depth=7,
        spectral_min=spectral_min,
        spectral_max=spectral_max,
    )
    actual = CELLS.run_precomputed_chebyshev_state_machine(
        hvp,
        rhs,
        preconditioner,
        step_schedule,
        momentum_schedule,
    )[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_tied_chebyshev_uses_one_hvp_per_block() -> None:
    matrix, rhs, preconditioner = random_spd_problem()
    calls = {"count": 0}

    def hvp(vector: torch.Tensor) -> torch.Tensor:
        calls["count"] += 1
        return torch.einsum("bkl,bl->bk", matrix, vector)

    CELLS.run_chebyshev_state_machine(
        hvp,
        rhs,
        preconditioner,
        depth=6,
        spectral_min=0.01,
        spectral_max=10.0,
    )

    assert calls["count"] == 6


def test_pcg_energy_error_is_no_worse_than_fixed_polynomial_controllers() -> None:
    matrix, rhs, preconditioner = random_spd_problem(batch_size=3, dimension=8)
    hvp = lambda vector: torch.einsum("bkl,bl->bk", matrix, vector)
    exact = torch.linalg.solve(matrix, rhs)
    chol = torch.linalg.cholesky(preconditioner)
    spectrum = torch.linalg.eigvalsh(chol.transpose(-1, -2) @ matrix @ chol)
    lower, upper = spectrum[:, 0], spectrum[:, -1]
    condition_sqrt = torch.sqrt(upper / lower)
    momentum = ((condition_sqrt - 1.0) / (condition_sqrt + 1.0)).square()
    step_size = 4.0 / (torch.sqrt(upper) + torch.sqrt(lower)).square()

    def energy_error(prediction: torch.Tensor) -> torch.Tensor:
        error = prediction - exact
        return torch.einsum("bk,bkl,bl->b", error, matrix, error)

    for depth in range(1, 7):
        pcg = CELLS.run_pcg_state_machine(hvp, rhs, preconditioner, depth)[0]
        heavy_ball = CELLS.run_heavy_ball_state_machine(
            hvp, rhs, preconditioner, depth, step_size, momentum
        )[0]
        chebyshev = CELLS.run_chebyshev_state_machine(
            hvp, rhs, preconditioner, depth, lower, upper
        )[0]

        assert torch.all(energy_error(pcg) <= energy_error(heavy_ball) + 1e-10)
        assert torch.all(energy_error(pcg) <= energy_error(chebyshev) + 1e-10)


def test_weighted_spectral_hb_risk_matches_explicit_state_machine() -> None:
    torch.manual_seed(17)
    batch, dimension, depth = 5, 4, 7
    factor_h = torch.randn(batch, dimension, dimension, dtype=torch.float64)
    matrix = (
        factor_h.transpose(-1, -2) @ factor_h
        + 0.5 * torch.eye(dimension, dtype=torch.float64)
    )
    factor_b = torch.randn(batch, dimension, dimension, dtype=torch.float64)
    preconditioner = torch.linalg.inv(
        factor_b.transpose(-1, -2) @ factor_b
        + 2.0 * torch.eye(dimension, dtype=torch.float64)
    )
    rhs = torch.randn(batch, dimension, dtype=torch.float64)
    step = torch.tensor(0.08, dtype=torch.float64)
    momentum = torch.tensor(0.12, dtype=torch.float64)
    hvp = lambda vector: torch.einsum("bij,bj->bi", matrix, vector)

    prediction = CELLS.run_heavy_ball_state_machine(
        hvp, rhs, preconditioner, depth, step, momentum
    )[0]
    target = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    error = prediction - target
    direct = torch.einsum("bi,bij,bj->b", error, matrix, error)
    direct /= torch.einsum("bi,bij,bj->b", target, matrix, target)

    cholesky = torch.linalg.cholesky(preconditioner)
    effective = cholesky.transpose(-1, -2) @ matrix @ cholesky
    eigenvalues, eigenvectors = torch.linalg.eigh(effective)
    transformed_rhs = torch.einsum("bji,bj->bi", cholesky, rhs)
    spectral_rhs = torch.einsum("bji,bj->bi", eigenvectors, transformed_rhs)
    spectral_solution = spectral_rhs / eigenvalues
    energy_weights = eigenvalues * spectral_solution.square()
    normalized_weights = energy_weights / energy_weights.sum(dim=-1, keepdim=True)
    spectral = HB_PREDICTOR.spectral_h_relative_squared(
        eigenvalues,
        normalized_weights,
        depth,
        step,
        momentum,
    )
    torch.testing.assert_close(spectral, direct, rtol=2e-11, atol=2e-12)
