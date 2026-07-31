import math
import sys
from pathlib import Path

import pytest
import torch

TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "experiments" / "transformers"
sys.path.insert(0, str(TRANSFORMER_DIR))

from exact_loop_transformer_decoder import (  # noqa: E402
    ExactLoopTransformerDecoder,
    normal_equations,
)
from evaluate_trained_loop_controllers import solve_all  # noqa: E402
from first_principles_decoder_cells import (  # noqa: E402
    fixed_prompt_linear_attention_hvp,
    materialize_preconditioner,
    risk_optimal_solution_chebyshev_coefficients,
    run_chebyshev_state_machine,
    run_heavy_ball_state_machine,
    run_pcg_state_machine,
    run_precomputed_moment_chebyshev_state_machine,
    shifted_chebyshev_basis,
)
from first_principles_inverse_decoder import (  # noqa: E402
    PromptSpectralMeasureMLP,
)
from pure_icl_parametric_operator_richardson_attention import (  # noqa: E402
    ParametricOperatorICL,
    make_true_family,
    sample_icl_batch,
    solve_z_exact,
)
from predict_pde_law_hyperparameters import (  # noqa: E402
    chebyshev_task_risk,
    conditional_weak_moments,
    exact_prompt_normal_rhs,
)
from structured_one_head_heavyball import (  # noqa: E402
    EquivariantPromptNystromPreconditioner,
    EquivariantRitzSoftmaxPreconditioner,
)


def _problem(dtype=torch.float64):
    torch.manual_seed(7)
    equations = torch.randn(5, 13, 4, dtype=dtype)
    observations = torch.randn(5, 13, dtype=dtype)
    return equations, observations


def test_normal_equations_are_the_exact_prompt_moments():
    equations, observations = _problem()
    ridge = 0.2
    normal, rhs = normal_equations(equations, observations, ridge)
    eye = torch.eye(4, dtype=equations.dtype)
    assert torch.allclose(normal, equations.transpose(-1, -2) @ equations + ridge * eye)
    expected_rhs = (equations.transpose(-1, -2) @ observations.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(rhs, expected_rhs)


def test_pde_conditional_moments_retain_shared_forcing_dependence() -> None:
    torch.manual_seed(31)
    dtype = torch.float64
    dimension, coefficient_dimension, probes_count = 4, 2, 3
    factor = torch.randn(dimension, dimension, dtype=dtype)
    true_operator = (
        factor.transpose(-1, -2) @ factor
        + torch.eye(dimension, dtype=dtype)
    ).unsqueeze(0)
    learned_A0 = 0.7 * torch.eye(dimension, dtype=dtype)
    learned_basis = torch.randn(
        coefficient_dimension,
        dimension,
        dimension,
        dtype=dtype,
    )
    probes = torch.randn(probes_count, dimension, dtype=dtype)
    forcing_std = 0.8
    expected_normal, expected_rhs = conditional_weak_moments(
        true_operator,
        learned_A0,
        learned_basis,
        probes,
        forcing_std**2,
    )
    samples = 120_000
    normal, rhs = exact_prompt_normal_rhs(
        true_operator,
        learned_A0,
        learned_basis,
        probes,
        samples,
        forcing_std,
        torch.Generator().manual_seed(32),
    )
    torch.testing.assert_close(
        normal / samples,
        expected_normal,
        rtol=1.5e-2,
        atol=2e-3,
    )
    torch.testing.assert_close(
        rhs / samples,
        expected_rhs,
        rtol=6e-2,
        atol=4e-3,
    )


def test_chebyshev_spectral_risk_matches_exact_state_machine() -> None:
    eigenvalues = torch.tensor(
        [[0.7, 1.1, 1.8], [0.8, 1.4, 2.0]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.4, -0.7, 0.2], [0.3, 0.5, -0.6]],
        dtype=torch.float64,
    )
    rhs = eigenvalues * target
    lower = eigenvalues[:, 0]
    upper = eigenvalues[:, -1]
    energy = eigenvalues * target.square()
    weights = energy / energy.sum(dim=-1, keepdim=True)
    predicted = chebyshev_task_risk(
        eigenvalues,
        weights,
        depth=5,
        lower=lower,
        upper=upper,
    )
    matrix = torch.diag_embed(eigenvalues)
    identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
    solution = run_chebyshev_state_machine(
        lambda vector: torch.einsum("bij,bj->bi", matrix, vector),
        rhs,
        identity,
        depth=5,
        spectral_min=lower,
        spectral_max=upper,
    )[0]
    error = solution - target
    actual = (eigenvalues * error.square()).sum(dim=-1) / energy.sum(dim=-1)
    torch.testing.assert_close(predicted, actual, rtol=1e-12, atol=1e-12)


def test_moment_chebyshev_gram_solve_and_clenshaw_are_exact() -> None:
    eigenvalues = torch.tensor(
        [[0.25, 0.8, 1.7], [0.35, 1.1, 1.9]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.4, -0.7, 0.2], [0.3, 0.5, -0.6]],
        dtype=torch.float64,
    )
    rhs = eigenvalues * target
    weights = torch.full_like(eigenvalues, 1.0 / eigenvalues.shape[-1])
    upper = eigenvalues[:, -1]
    coefficients = risk_optimal_solution_chebyshev_coefficients(
        eigenvalues,
        weights,
        degree=3,
        spectral_upper=upper,
        gram_regularization=0.0,
    )
    matrix = torch.diag_embed(eigenvalues)
    identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
    solution = run_precomputed_moment_chebyshev_state_machine(
        lambda vector: torch.einsum("bij,bj->bi", matrix, vector),
        rhs,
        identity,
        coefficients,
        upper,
    )[0]
    torch.testing.assert_close(solution, target, rtol=2e-11, atol=2e-11)

    basis = shifted_chebyshev_basis(
        eigenvalues,
        degree=3,
        spectral_upper=upper,
    )
    residual = 1.0 - eigenvalues * torch.einsum(
        "bkl,bl->bk",
        basis,
        coefficients,
    )
    torch.testing.assert_close(
        residual,
        torch.zeros_like(residual),
        rtol=0.0,
        atol=2e-11,
    )


def test_moment_chebyshev_clenshaw_vectorizes_multiple_rhs() -> None:
    eigenvalues = torch.tensor(
        [[0.3, 0.9, 1.6], [0.4, 1.0, 1.8]],
        dtype=torch.float64,
    )
    upper = eigenvalues[:, -1]
    weights = torch.full_like(eigenvalues, 1.0 / 3.0)
    coefficients = risk_optimal_solution_chebyshev_coefficients(
        eigenvalues,
        weights,
        degree=3,
        spectral_upper=upper,
        gram_regularization=0.0,
    )
    target = torch.tensor(
        [
            [[0.4, -0.2], [-0.7, 0.3], [0.2, 0.5]],
            [[0.3, 0.6], [0.5, -0.1], [-0.6, 0.2]],
        ],
        dtype=torch.float64,
    )
    rhs = eigenvalues.unsqueeze(-1) * target
    matrix = torch.diag_embed(eigenvalues)
    identity = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
    solution = run_precomputed_moment_chebyshev_state_machine(
        lambda vector: torch.einsum("bij,bjq->biq", matrix, vector),
        rhs,
        identity,
        coefficients,
        upper,
    )[0]
    torch.testing.assert_close(solution, target, rtol=2e-11, atol=2e-11)


def test_spectral_measure_mlp_learns_only_ordered_nodes_and_masses() -> None:
    torch.manual_seed(37)
    features = torch.randn(6, 7, dtype=torch.float64)
    reference_upper = torch.rand(6, dtype=torch.float64) + 0.5
    certified_upper = 3.0 * reference_upper
    head = PromptSpectralMeasureMLP(
        hidden_dimension=12,
        clusters=5,
    ).double()
    nodes, weights, basis_upper = head(
        features,
        reference_upper,
        certified_upper,
    )
    assert nodes.shape == weights.shape == (6, 5)
    assert torch.all(nodes > 0)
    assert torch.all(nodes[:, 1:] >= nodes[:, :-1])
    assert torch.all(nodes[:, -1] < basis_upper)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(6, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.all(weights > 0)
    assert torch.all(basis_upper > nodes[:, -1])
    assert torch.all(basis_upper <= certified_upper)

    coefficients = risk_optimal_solution_chebyshev_coefficients(
        nodes,
        weights,
        degree=4,
        spectral_upper=basis_upper,
    )
    coefficients.square().mean().backward()
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_moment_chebyshev_loop_is_matrix_free_and_differentiable(
    monkeypatch,
) -> None:
    import exact_loop_transformer_decoder as decoder_module

    equations, observations = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=4,
        head_dimension=12,
        slots=3,
        controller="moment_chebyshev",
        spectral_lmax_bound=2.5,
        spectral_measure_clusters=6,
        spectral_measure_hidden_dimension=10,
        moment_gram_regularization=1e-7,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=2,
    ).double()

    def forbidden_normal_matrix(*args, **kwargs):
        raise AssertionError("moment Chebyshev materialized H")

    monkeypatch.setattr(
        decoder_module,
        "normal_matrix_from_equations",
        forbidden_normal_matrix,
    )
    solution, info = decoder(equations, observations, ridge=0.2)
    assert torch.isfinite(solution).all()
    assert "normal_matrix" not in info
    assert not info["normal_matrix_materialized"].any()
    assert info["matrix_free"].all()
    assert info["spectral_measure_nodes"].shape == (equations.shape[0], 6)
    torch.testing.assert_close(
        info["spectral_measure_weights"].sum(dim=-1),
        torch.ones(equations.shape[0], dtype=equations.dtype),
    )
    assert info["moment_solution_coefficients"].shape == (
        equations.shape[0],
        decoder.depth,
    )

    solution.square().mean().backward()
    assert decoder.measure_head is not None
    assert all(
        parameter.grad is not None
        for parameter in decoder.measure_head.parameters()
    )


def test_moment_chebyshev_loop_reuses_prompt_geometry_for_multiple_rhs() -> None:
    equations, _ = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=4,
        head_dimension=12,
        slots=3,
        controller="moment_chebyshev",
        spectral_lmax_bound=2.5,
        spectral_measure_clusters=6,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=2,
    ).double()
    observations = torch.randn(
        equations.shape[0],
        equations.shape[1],
        3,
        dtype=equations.dtype,
    )
    geometry = decoder.build_prompt_geometry(equations, ridge=0.2)
    multi, _ = decoder.solve_with_geometry(geometry, observations)
    separate = torch.stack(
        [
            decoder.solve_with_geometry(geometry, observations[..., index])[0]
            for index in range(observations.shape[-1])
        ],
        dim=-1,
    )
    torch.testing.assert_close(multi, separate, rtol=2e-10, atol=2e-10)


def test_moment_chebyshev_rejects_non_matrix_free_head() -> None:
    with pytest.raises(ValueError, match="requires the matrix-free Nystrom head"):
        ExactLoopTransformerDecoder(
            dimension=4,
            depth=4,
            head_dimension=8,
            slots=2,
            controller="moment_chebyshev",
            preconditioner_head_type="coordinate_ritz",
        )


def test_equivariant_ritz_softmax_head_is_gauge_covariant_and_spectral() -> None:
    torch.manual_seed(29)
    batch, dimension = 6, 5
    factor = torch.randn(batch, dimension, dimension, dtype=torch.float64)
    normal = (
        factor.transpose(-1, -2) @ factor
        + 0.4 * torch.eye(dimension, dtype=torch.float64)
    )
    equations = torch.zeros(batch, 17, dimension, dtype=torch.float64)
    head = EquivariantRitzSoftmaxPreconditioner(
        dimension=dimension,
        head_dimension=12,
        slots=3,
    ).double()
    preconditioner, info = head(equations, normal)

    gauge = torch.linalg.qr(
        torch.randn(batch, dimension, dimension, dtype=torch.float64)
    ).Q
    rotated_normal = gauge.transpose(-1, -2) @ normal @ gauge
    rotated_preconditioner, _ = head(equations, rotated_normal)
    expected = gauge.transpose(-1, -2) @ preconditioner @ gauge
    torch.testing.assert_close(
        rotated_preconditioner,
        expected,
        rtol=2e-10,
        atol=2e-10,
    )

    cholesky = torch.linalg.cholesky(preconditioner)
    effective = cholesky.transpose(-1, -2) @ normal @ cholesky
    actual_eigenvalues = torch.linalg.eigvalsh(effective)
    predicted_eigenvalues = info["effective_eigenvalues_predicted"].sort(dim=-1).values
    torch.testing.assert_close(
        actual_eigenvalues,
        predicted_eigenvalues,
        rtol=2e-10,
        atol=2e-10,
    )
    assert actual_eigenvalues.max() <= 4.0 + 1e-10
    assert torch.all(info["spectral_gates"].sum(dim=-1) <= 3.0 + 1e-12)

    actual_eigenvalues.square().mean().backward()
    assert head.key.weight.grad is not None
    assert head.slot_queries.grad is not None
    assert head.key.weight.grad[:3, :3].norm() > 0
    assert head.slot_queries.grad[:, :3].norm() > 0


def test_prompt_nystrom_head_is_gauge_covariant_and_certified() -> None:
    equations, _ = _problem()
    normal = equations.transpose(-1, -2) @ equations
    normal = normal + 0.2 * torch.eye(normal.shape[-1], dtype=normal.dtype)
    head = EquivariantPromptNystromPreconditioner(
        dimension=normal.shape[-1],
        head_dimension=12,
        slots=3,
        spectral_lmax_bound=2.5,
        refinement_steps=2,
    ).double()
    preconditioner, info = head(equations, normal)
    generator = torch.Generator().manual_seed(39)
    gauge = torch.linalg.qr(
        torch.randn(4, 4, generator=generator, dtype=torch.float64)
    ).Q
    rotated_equations = equations @ gauge
    rotated_normal = gauge.transpose(-1, -2) @ normal @ gauge
    rotated_preconditioner, rotated_info = head(
        rotated_equations,
        rotated_normal,
    )
    expected = gauge.transpose(-1, -2) @ preconditioner @ gauge
    torch.testing.assert_close(
        rotated_preconditioner,
        expected,
        rtol=3e-9,
        atol=3e-9,
    )
    torch.testing.assert_close(
        rotated_info["interval_features"],
        info["interval_features"],
        rtol=3e-9,
        atol=3e-9,
    )

    permutation = torch.randperm(equations.shape[1], generator=generator)
    permuted_preconditioner, _ = head(equations[:, permutation], normal)
    torch.testing.assert_close(
        permuted_preconditioner,
        preconditioner,
        rtol=3e-9,
        atol=3e-9,
    )
    factor = torch.linalg.cholesky(preconditioner)
    effective = factor.transpose(-1, -2) @ normal @ factor
    assert torch.linalg.eigvalsh(effective).amax() <= 2.5 + 1e-9
    assert info["projected_operator"].shape[-1] == 3
    assert not info["uses_full_eigendecomposition"].any()

    loss = preconditioner.square().sum()
    loss.backward()
    assert head.key.weight.grad is not None
    assert head.slot_queries.grad is not None
    assert head.slot_queries.grad.norm() > 0


def test_prompt_nystrom_loop_decoder_is_finite_without_full_eigenspectrum() -> None:
    equations, observations = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=8,
        head_dimension=12,
        slots=3,
        controller="heavy_ball",
        spectral_lmax_bound=2.5,
        step_init=0.5,
        momentum_init=0.08,
        preconditioner_head_type="equivariant_prompt_nystrom",
        prompt_subspace_refinement_steps=2,
    ).double()
    solution, info = decoder(equations, observations, ridge=0.2)
    assert torch.isfinite(solution).all()
    assert not info["uses_full_eigendecomposition"].any()
    assert (info["certified_effective_lmax"] == 2.5).all()


def test_matrix_free_nystrom_decoder_never_materializes_normal_or_preconditioner(
    monkeypatch,
) -> None:
    import exact_loop_transformer_decoder as decoder_module

    equations, observations = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=8,
        head_dimension=12,
        slots=3,
        controller="heavy_ball",
        spectral_lmax_bound=2.5,
        step_init=0.5,
        momentum_init=0.08,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=3,
    ).double()

    def forbidden_normal_equations(*args, **kwargs):
        raise AssertionError("matrix-free decoder materialized H")

    monkeypatch.setattr(
        decoder_module,
        "normal_equations",
        forbidden_normal_equations,
    )
    solution, info = decoder(equations, observations, ridge=0.2)
    assert torch.isfinite(solution).all()
    assert "normal_matrix" not in info
    assert not info["normal_matrix_materialized"].any()
    assert info["matrix_free"].all()
    assert not info["uses_full_eigendecomposition"].any()

    dense = materialize_preconditioner(info["preconditioner"])
    probe = torch.randn_like(solution)
    torch.testing.assert_close(
        info["preconditioner"].apply(probe),
        torch.einsum("bij,bj->bi", dense, probe),
        rtol=2e-10,
        atol=2e-10,
    )
    normal = equations.transpose(-1, -2) @ equations
    normal = normal + 0.2 * torch.eye(normal.shape[-1], dtype=normal.dtype)
    factor = torch.linalg.cholesky(dense)
    effective = factor.transpose(-1, -2) @ normal @ factor
    effective_lmax = torch.linalg.eigvalsh(effective)[:, -1]
    assert effective_lmax.amax() <= 2.5 + 1e-9
    assert (
        effective_lmax * info["certificate_normalizer"]
        <= info["post_deflation_lmax_bound"] + 1e-9
    ).all()
    torch.testing.assert_close(
        info["post_deflation_lmax_bound"]
        / info["certificate_normalizer"],
        torch.full_like(info["certificate_normalizer"], 2.5),
        rtol=2e-10,
        atol=2e-10,
    )


def test_matrix_free_nystrom_equals_block_moment_ritz_formula() -> None:
    equations, _ = _problem()
    ridge = 0.2
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=4,
        head_dimension=12,
        slots=3,
        controller="heavy_ball",
        spectral_lmax_bound=2.5,
        step_init=0.5,
        momentum_init=0.08,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=3,
    ).double()
    head = decoder.preconditioner_head
    preconditioner, _ = head(equations, ridge)

    row_norm_squared = equations.square().sum(dim=-1).clamp_min(1e-12)
    normalized_rows = equations / row_norm_squared.sqrt().unsqueeze(-1)
    standardized_norm = head._standardize(0.5 * torch.log(row_norm_squared))
    features = torch.stack(
        [
            standardized_norm,
            head._standardize(standardized_norm.square()),
            head._standardize(standardized_norm.pow(3)),
            torch.ones_like(standardized_norm),
        ],
        dim=-1,
    )
    keys = head.key(features)
    scores = torch.einsum("sd,bmd->bsm", head.slot_queries, keys)
    attention = torch.softmax(scores / math.sqrt(head.head_dimension), dim=-1)
    routed_block = torch.einsum(
        "bsm,bmk->bks",
        attention,
        normalized_rows,
    )

    dimension = equations.shape[-1]
    normal = equations.transpose(-1, -2) @ equations
    normal = normal + ridge * torch.eye(dimension, dtype=equations.dtype)
    scale = (
        (row_norm_squared.sum(dim=-1) + ridge * dimension)
        / head.spectral_lmax_bound
    )
    operator = normal / scale[:, None, None]
    power_r = torch.linalg.matrix_power(operator, head.refinement_steps)
    power_2r = torch.linalg.matrix_power(
        operator,
        2 * head.refinement_steps,
    )
    gram = routed_block.transpose(-1, -2) @ power_2r @ routed_block
    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(gram)
    inverse_sqrt = torch.einsum(
        "bsi,bi,bti->bst",
        gram_eigenvectors,
        gram_eigenvalues.rsqrt(),
        gram_eigenvectors,
    )
    directions = power_r @ routed_block @ inverse_sqrt
    projected = directions.transpose(-1, -2) @ operator @ directions
    projected_eigenvalues, projected_eigenvectors = torch.linalg.eigh(projected)
    target = projected_eigenvalues[:, :1]
    slot_map = torch.einsum(
        "bsi,bi,bti->bst",
        projected_eigenvectors,
        target / projected_eigenvalues,
        projected_eigenvectors,
    )
    multiplier_sqrt = torch.einsum(
        "bsi,bi,bti->bst",
        projected_eigenvectors,
        (target / projected_eigenvalues).sqrt(),
        projected_eigenvectors,
    )
    residual = operator @ directions - directions @ projected
    cross_bound = torch.linalg.matrix_norm(
        residual @ multiplier_sqrt,
        ord="fro",
        dim=(-2, -1),
    )
    complement_trace = (
        head.spectral_lmax_bound
        - torch.diagonal(projected, dim1=-2, dim2=-1).sum(dim=-1)
    ).clamp_min(0.0)
    selected_bound = target[:, 0]
    post_deflation_bound = 0.5 * (
        selected_bound
        + complement_trace
        + torch.sqrt(
            (selected_bound - complement_trace).square()
            + 4.0 * cross_bound.square()
        )
    )
    certificate_normalizer = (
        post_deflation_bound / head.spectral_lmax_bound
    ).clamp_min(1e-10)
    identity = torch.eye(dimension, dtype=equations.dtype).expand(
        equations.shape[0],
        -1,
        -1,
    )
    formula = (
        identity
        - directions @ directions.transpose(-1, -2)
        + directions @ slot_map @ directions.transpose(-1, -2)
    ) / (scale * certificate_normalizer)[:, None, None]
    torch.testing.assert_close(
        materialize_preconditioner(preconditioner),
        formula,
        rtol=2e-8,
        atol=2e-8,
    )


def test_matrix_free_nystrom_head_is_gauge_covariant() -> None:
    equations, observations = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=4,
        head_dimension=12,
        slots=3,
        controller="heavy_ball",
        spectral_lmax_bound=2.5,
        step_init=0.5,
        momentum_init=0.08,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=3,
    ).double()
    preconditioner, info = decoder.preconditioner_head(
        equations,
        0.2,
    )
    generator = torch.Generator().manual_seed(49)
    gauge = torch.linalg.qr(
        torch.randn(4, 4, generator=generator, dtype=torch.float64)
    ).Q
    rotated, rotated_info = decoder.preconditioner_head(
        equations @ gauge,
        0.2,
    )
    dense = materialize_preconditioner(preconditioner)
    rotated_dense = materialize_preconditioner(rotated)
    expected = gauge.transpose(-1, -2) @ dense @ gauge
    torch.testing.assert_close(
        rotated_dense,
        expected,
        rtol=2e-9,
        atol=2e-9,
    )
    torch.testing.assert_close(
        rotated_info["interval_features"],
        info["interval_features"],
        rtol=2e-9,
        atol=2e-9,
    )


def test_matrix_free_certified_hb_routes_to_pcg_without_dense_normal(
    monkeypatch,
) -> None:
    import exact_loop_transformer_decoder as decoder_module

    equations, observations = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=5,
        head_dimension=12,
        slots=3,
        controller="certified_hb_pcg",
        spectral_lmax_bound=2.5,
        step_init=0.5,
        momentum_init=0.08,
        hybrid_residual_threshold=1e-30,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=2,
    ).double()

    def forbidden_normal_equations(*args, **kwargs):
        raise AssertionError("certified matrix-free decoder materialized H")

    monkeypatch.setattr(
        decoder_module,
        "normal_equations",
        forbidden_normal_equations,
    )
    solution, info = decoder(equations, observations, ridge=0.2)
    assert torch.isfinite(solution).all()
    assert info["pcg_fallback_mask"].all()
    assert info["pcg_fallback_rate"] == 1.0
    assert "normal_matrix" not in info


def test_matrix_free_cells_vectorize_multiple_right_hand_sides() -> None:
    equations, _ = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=5,
        head_dimension=12,
        slots=3,
        controller="heavy_ball",
        spectral_lmax_bound=2.5,
        step_init=0.5,
        momentum_init=0.08,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=2,
    ).double()
    preconditioner, _ = decoder.preconditioner_head(equations, 0.2)
    rhs = torch.randn(
        equations.shape[0],
        equations.shape[-1],
        5,
        dtype=equations.dtype,
    )

    def hvp(vector):
        return fixed_prompt_linear_attention_hvp(
            equations,
            vector,
            noise_precision=1.0,
            prior_precision=0.2,
        )

    dense = materialize_preconditioner(preconditioner)
    normal = equations.transpose(-1, -2) @ equations
    normal = normal + 0.2 * torch.eye(normal.shape[-1], dtype=normal.dtype)
    factor = torch.linalg.cholesky(dense)
    spectrum = torch.linalg.eigvalsh(
        factor.transpose(-1, -2) @ normal @ factor
    )
    spectral_min, spectral_max = spectrum[:, 0], spectrum[:, -1]

    multi_hb = run_heavy_ball_state_machine(
        hvp,
        rhs,
        preconditioner,
        5,
        0.5,
        0.08,
    )[0]
    multi_chebyshev = run_chebyshev_state_machine(
        hvp,
        rhs,
        preconditioner,
        5,
        spectral_min,
        spectral_max,
    )[0]
    multi_pcg = run_pcg_state_machine(hvp, rhs, preconditioner, 5)[0]

    for multi, solver in [
        (
            multi_hb,
            lambda column: run_heavy_ball_state_machine(
                hvp, column, preconditioner, 5, 0.5, 0.08
            )[0],
        ),
        (
            multi_chebyshev,
            lambda column: run_chebyshev_state_machine(
                hvp,
                column,
                preconditioner,
                5,
                spectral_min,
                spectral_max,
            )[0],
        ),
        (
            multi_pcg,
            lambda column: run_pcg_state_machine(
                hvp, column, preconditioner, 5
            )[0],
        ),
    ]:
        separate = torch.stack(
            [solver(rhs[..., index]) for index in range(rhs.shape[-1])],
            dim=-1,
        )
        torch.testing.assert_close(multi, separate, rtol=2e-10, atol=2e-10)

    observations = torch.randn(
        equations.shape[0],
        equations.shape[1],
        5,
        dtype=equations.dtype,
    )
    geometry = decoder.build_prompt_geometry(equations, ridge=0.2)
    cached_multi, multi_info = decoder.solve_with_geometry(
        geometry,
        observations,
    )
    cached_separate = torch.stack(
        [
            decoder.solve_with_geometry(geometry, observations[..., index])[0]
            for index in range(observations.shape[-1])
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        cached_multi,
        cached_separate,
        rtol=2e-10,
        atol=2e-10,
    )
    assert multi_info["rhs"].shape == cached_multi.shape


def test_exact_head_spectrum_chebyshev_reuses_ritz_eigenvalues() -> None:
    equations, observations = _problem()
    decoder = ExactLoopTransformerDecoder(
        dimension=equations.shape[-1],
        depth=7,
        head_dimension=12,
        slots=3,
        controller="chebyshev",
        spectral_lmax_bound=4.0,
        preconditioner_head_type="equivariant_ritz_softmax",
        chebyshev_interval_policy="exact_head_spectrum",
    ).double()
    solution, info = decoder(equations, observations, ridge=0.2)
    assert decoder.interval_head is None
    predicted = info["effective_eigenvalues_predicted"]
    torch.testing.assert_close(info["spectral_min"], predicted.amin(dim=-1))
    torch.testing.assert_close(info["spectral_max"], predicted.amax(dim=-1))

    normal = info["normal_matrix"]
    preconditioner = info["preconditioner"]
    factor = torch.linalg.cholesky(
        preconditioner
        + 1e-10 * torch.eye(normal.shape[-1], dtype=normal.dtype)
    )
    effective = factor.transpose(-1, -2) @ normal @ factor
    actual = torch.linalg.eigvalsh(effective)
    torch.testing.assert_close(
        predicted.sort(dim=-1).values,
        actual,
        rtol=2e-9,
        atol=2e-9,
    )
    assert torch.isfinite(solution).all()


def test_exact_head_spectrum_rejects_coordinate_ritz_head() -> None:
    with pytest.raises(ValueError, match="requires the equivariant Ritz head"):
        ExactLoopTransformerDecoder(
            dimension=4,
            depth=5,
            head_dimension=8,
            slots=2,
            controller="chebyshev",
            preconditioner_head_type="coordinate_ritz",
            chebyshev_interval_policy="exact_head_spectrum",
        )


def test_richardson_is_zero_momentum_heavy_ball():
    equations, observations = _problem()
    model = ExactLoopTransformerDecoder(
        dimension=4,
        depth=3,
        head_dimension=8,
        slots=2,
        controller="richardson",
        spectral_lmax_bound=20.0,
        step_init=0.05,
    ).double()
    _, info = model(equations, observations, ridge=0.2)
    assert info["momentum"].item() == 0.0
    assert 0.0 < info["step"].item() < 2.0 / 20.0


def test_pcg_is_no_worse_in_h_energy_than_hb_with_same_initial_head():
    equations, observations = _problem()
    hb = ExactLoopTransformerDecoder(
        dimension=4,
        depth=4,
        head_dimension=8,
        slots=2,
        controller="heavy_ball",
        spectral_lmax_bound=40.0,
        step_init=0.02,
        momentum_init=0.05,
    ).double()
    pcg = ExactLoopTransformerDecoder(
        dimension=4,
        depth=4,
        head_dimension=8,
        slots=2,
        controller="pcg",
    ).double()
    pcg.preconditioner_head.load_state_dict(hb.preconditioner_head.state_dict())
    hb_solution, hb_info = hb(equations, observations, ridge=0.2)
    pcg_solution, _ = pcg(equations, observations, ridge=0.2)
    target = torch.linalg.solve(hb_info["normal_matrix"], hb_info["rhs"].unsqueeze(-1)).squeeze(-1)

    def h_error(solution):
        error = solution - target
        return torch.einsum("bk,bkl,bl->b", error, hb_info["normal_matrix"], error)

    assert torch.all(h_error(pcg_solution) <= h_error(hb_solution) + 1e-10)


def test_certified_hb_routes_failed_residuals_to_the_exact_same_pcg_cell():
    equations, observations = _problem()
    hybrid = ExactLoopTransformerDecoder(
        dimension=4,
        depth=4,
        head_dimension=8,
        slots=2,
        controller="certified_hb_pcg",
        spectral_lmax_bound=40.0,
        step_init=0.02,
        momentum_init=0.05,
        hybrid_residual_threshold=1e-30,
    ).double()
    pcg = ExactLoopTransformerDecoder(
        dimension=4,
        depth=4,
        head_dimension=8,
        slots=2,
        controller="pcg",
    ).double()
    pcg.preconditioner_head.load_state_dict(hybrid.preconditioner_head.state_dict())
    hybrid_solution, info = hybrid(equations, observations, ridge=0.2)
    pcg_solution, _ = pcg(equations, observations, ridge=0.2)
    assert info["pcg_fallback_mask"].all()
    assert info["pcg_fallback_rate"].item() == 1.0
    assert torch.allclose(hybrid_solution, pcg_solution)


def test_chebyshev_mlp_receives_end_to_end_gradient_without_emulating_solver_arithmetic():
    equations, observations = _problem()
    model = ExactLoopTransformerDecoder(
        dimension=4,
        depth=3,
        head_dimension=8,
        slots=2,
        controller="chebyshev",
    ).double()
    prediction, info = model(equations, observations, ridge=0.2)
    prediction.square().mean().backward()
    assert info["spectral_min"].shape == (equations.shape[0],)
    assert info["spectral_max"].shape == (equations.shape[0],)
    assert all(parameter.grad is not None for parameter in model.interval_head.parameters())


def test_adaptive_hb_mlp_predicts_only_spectral_scalars_and_receives_gradient():
    equations, observations = _problem()
    model = ExactLoopTransformerDecoder(
        dimension=4,
        depth=3,
        head_dimension=8,
        slots=2,
        controller="heavy_ball",
        spectral_lmax_bound=20.0,
        step_init=0.05,
        adaptive_heavy_ball=True,
    ).double()
    prediction, info = model(equations, observations, ridge=0.2)
    prediction.square().mean().backward()
    assert info["step"].shape == (equations.shape[0],)
    assert info["momentum"].shape == (equations.shape[0],)
    assert torch.all((info["momentum"] >= 0) & (info["momentum"] < 1))
    assert all(parameter.grad is not None for parameter in model.interval_head.parameters())


def test_cross_evaluator_reuses_hb_interval_mlp_for_chebyshev() -> None:
    torch.manual_seed(31)
    device = torch.device("cpu")
    family = make_true_family(8, 4, 0.2, 2.0, device)
    model = ParametricOperatorICL(
        d=8,
        K=4,
        R=4,
        lam_z=1e-2,
        gamma_u=1e-5,
        solver="primal_loop_heavy_ball",
        z_depth=3,
        learn_dictionary=False,
        learn_probes=False,
        true_family=family,
        init="true",
        init_noise=0.0,
        heads=1,
        d_head=8,
        qk_from="g",
        use_safe_scale=True,
        hb_alpha_init=1.0,
        hb_beta_init=0.05,
        subspace_slots=2,
        loop_lmax_bound=4.0,
        loop_step_init=0.2,
        adaptive_heavy_ball=True,
        loop_preconditioner_head="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=1,
    )
    batch = sample_icl_batch(family, 3, 3, 0.5, 1.0, 0.0, device)
    solutions, _, _, diagnostics = solve_all(
        model,
        batch,
        hybrid_residual_threshold=1e-8,
        hb_depth=3,
        pcg_depth=2,
    )
    assert "learned_chebyshev" in solutions
    assert "residual_guarded_chebyshev_pcg" in solutions
    mask = diagnostics["chebyshev_fallback_mask"][:, None]
    expected = torch.where(
        mask,
        solutions["pcg"],
        solutions["learned_chebyshev"],
    )
    torch.testing.assert_close(
        solutions["residual_guarded_chebyshev_pcg"],
        expected,
    )


def test_exact_loop_decoder_is_connected_to_the_full_icl_model():
    torch.manual_seed(11)
    device = torch.device("cpu")
    family = make_true_family(8, 4, 0.2, 2.0, device)
    model = ParametricOperatorICL(
        d=8,
        K=4,
        R=4,
        lam_z=1e-2,
        gamma_u=1e-5,
        solver="primal_loop_heavy_ball",
        z_depth=2,
        learn_dictionary=True,
        learn_probes=False,
        true_family=family,
        init="true_noisy",
        init_noise=0.01,
        heads=1,
        d_head=8,
        qk_from="g",
        use_safe_scale=True,
        hb_alpha_init=1.0,
        hb_beta_init=0.05,
        subspace_slots=2,
        loop_lmax_bound=40.0,
        loop_step_init=0.02,
    )
    batch = sample_icl_batch(family, 3, 3, 0.5, 1.0, 0.0, device)
    prediction, info = model(
        batch.f_prompt,
        batch.u_prompt,
        batch.f_star,
        return_info=True,
    )
    prediction.square().mean().backward()
    assert prediction.shape == batch.u_star.shape
    assert "preconditioner" in info
    assert model.Abasis.grad is not None
    assert model.loop_decoder.preconditioner_head.key.weight.grad is not None


def test_elliptic_operator_family_is_spd_at_baseline_and_low_rank_parametric():
    family = make_true_family(
        16,
        4,
        0.1,
        2.0,
        torch.device("cpu"),
        operator_family="elliptic_1d",
    )
    assert torch.allclose(family.A0, family.A0.transpose(-1, -2))
    assert torch.linalg.eigvalsh(family.A0).min() > 0
    flattened = family.Abasis.flatten(1)
    assert torch.linalg.matrix_rank(flattened) == 4
    assert torch.allclose(family.Abasis, family.Abasis.transpose(-1, -2))


def test_elliptic_dictionary_projection_removes_nonphysical_matrix_components():
    torch.manual_seed(13)
    family = make_true_family(
        8,
        3,
        0.1,
        2.0,
        torch.device("cpu"),
        operator_family="elliptic_1d",
    )
    model = ParametricOperatorICL(
        d=8,
        K=3,
        R=4,
        lam_z=1e-2,
        gamma_u=1e-5,
        solver="exact",
        z_depth=2,
        learn_dictionary=True,
        learn_probes=False,
        true_family=family,
        init="identity_random",
        init_noise=0.0,
        heads=1,
        d_head=8,
        qk_from="g",
        use_safe_scale=True,
        hb_alpha_init=1.0,
        hb_beta_init=0.05,
        dictionary_projection="elliptic_1d",
    )
    with torch.no_grad():
        model.Abasis.add_(torch.randn_like(model.Abasis))
        model.project_dictionary_()
    basis = model.dictionary_projection_basis
    flattened = model.Abasis.flatten(1)
    residual = flattened - (flattened @ basis) @ basis.transpose(0, 1)
    assert residual.norm() < 1e-5


def test_frozen_projected_dictionary_is_not_reprojected_or_drifted():
    torch.manual_seed(23)
    family = make_true_family(
        8,
        3,
        0.1,
        2.0,
        torch.device("cpu"),
        operator_family="elliptic_1d",
    )
    model = ParametricOperatorICL(
        d=8,
        K=3,
        R=4,
        lam_z=1e-2,
        gamma_u=1e-5,
        solver="exact",
        z_depth=2,
        learn_dictionary=False,
        learn_probes=False,
        true_family=family,
        init="true",
        init_noise=0.0,
        heads=1,
        d_head=8,
        qk_from="g",
        use_safe_scale=True,
        hb_alpha_init=1.0,
        hb_beta_init=0.05,
        dictionary_projection="elliptic_1d",
    )
    frozen_basis = model.Abasis.detach().clone()
    for _ in range(100):
        model.project_dictionary_()
    assert torch.equal(model.Abasis, frozen_basis)


def test_covariant_ridge_is_invariant_under_dictionary_change_of_basis():
    torch.manual_seed(19)
    batch, rows, rank, physical_dimension = 4, 11, 3, 7
    equations = torch.randn(batch, rows, rank, dtype=torch.float64)
    observations = torch.randn(batch, rows, dtype=torch.float64)
    dictionary = torch.randn(rank, physical_dimension, dtype=torch.float64)
    transform = torch.randn(rank, rank, dtype=torch.float64) + 2.0 * torch.eye(
        rank, dtype=torch.float64
    )
    transformed_dictionary = transform @ dictionary
    transformed_equations = equations @ transform.transpose(0, 1)
    metric = dictionary @ dictionary.transpose(0, 1)
    transformed_metric = transformed_dictionary @ transformed_dictionary.transpose(0, 1)
    coefficients = solve_z_exact(equations, observations, 0.3, metric)
    transformed_coefficients = solve_z_exact(
        transformed_equations,
        observations,
        0.3,
        transformed_metric,
    )
    physical = coefficients @ dictionary
    transformed_physical = transformed_coefficients @ transformed_dictionary
    assert torch.allclose(physical, transformed_physical, atol=1e-9, rtol=1e-9)
