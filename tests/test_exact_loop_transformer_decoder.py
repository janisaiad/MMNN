import sys
from pathlib import Path

import torch

TRANSFORMER_DIR = Path(__file__).resolve().parents[1] / "experiments" / "transformers"
sys.path.insert(0, str(TRANSFORMER_DIR))

from exact_loop_transformer_decoder import (
    ExactLoopTransformerDecoder,
    normal_equations,
)
from pure_icl_parametric_operator_richardson_attention import (
    ParametricOperatorICL,
    make_true_family,
    sample_icl_batch,
    solve_z_exact,
)
from structured_one_head_heavyball import EquivariantRitzSoftmaxPreconditioner


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
