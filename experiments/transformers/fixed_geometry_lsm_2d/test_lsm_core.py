"""Fast structural tests for the 2D LSM proof of concept."""

from __future__ import annotations

import torch

from .foundation_uq import (
    EquivariantChebyshevPCGLSMLoop,
    FoundationPCGLSMLoop,
    occupancy_probability,
    posterior_covariance,
    posterior_score_moments,
    sample_multi_obstacle_masks,
)
from .lsm_core import (
    FixedGeometryBornLSM,
    PhysicsConfig,
    TiedBayesianLSMLoop,
    TiedKrylovLSMLoop,
    TiedAttentionPCGLSMLoop,
    TiedStationaryLSMLoop,
    angular_kernel,
    exact_bayesian_lsm,
    ranking_loss,
)
from .near_field_lsm import (
    NearFieldConfig,
    NearFieldSoundSoftLSM,
    PosteriorMomentHBLSMLoop,
    PosteriorMomentLSMLoop,
    build_near_field_system,
    exact_near_field_lsm,
)


def small_problem() -> tuple[FixedGeometryBornLSM, torch.Tensor, torch.Tensor]:
    device = torch.device("cpu")
    physics = FixedGeometryBornLSM(
        PhysicsConfig(n_angles=12, grid_size=12, noise_rel=0.01),
        device,
    )
    far_field, mask = physics.sample_batch(3, "ellipse")
    return physics, far_field, mask


def test_angular_kernel_is_fixed_positive_and_row_stochastic() -> None:
    angles = torch.linspace(0.0, 5.0, 11)
    attention, kernel = angular_kernel(angles, gamma=1.2, mix=0.3)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(11), atol=1.0e-6)
    assert torch.allclose(kernel, kernel.T, atol=1.0e-6)
    assert torch.linalg.eigvalsh(kernel).amin() > 0.0


def test_forward_data_and_masks_are_two_dimensional_and_complex() -> None:
    physics, far_field, mask = small_problem()
    assert far_field.shape == (3, 12, 12)
    assert far_field.is_complex()
    assert mask.shape == (3, physics.n_probes)
    assert mask.any(dim=-1).all()
    assert (~mask).any(dim=-1).all()


def test_loop_is_differentiable_and_kernel_is_not_trainable() -> None:
    physics, far_field, mask = small_problem()
    model = TiedBayesianLSMLoop(physics.kernel, 0.01, depth=3, controller_width=8)
    score, info = model(far_field, physics.probe_rhs)
    loss = ranking_loss(score, mask, n_pairs=5) + info["relative_residual"].mean()
    loss.backward()
    assert model.geometry_is_frozen
    assert "kernel" not in dict(model.named_parameters())
    assert all(
        parameter.grad is not None for parameter in model.controller.parameters()
    )


def test_exact_reference_solves_only_at_evaluation() -> None:
    physics, far_field, _ = small_problem()
    score, info = exact_bayesian_lsm(far_field, physics.probe_rhs, physics.kernel, 0.01)
    assert score.shape == (3, physics.n_probes)
    assert info["relative_residual"].amax() < 2.0e-4


def test_subgeometry_is_consistent() -> None:
    physics, far_field, _ = small_problem()
    indices = torch.tensor([0, 2, 5, 7, 9])
    selected, probe, attention, kernel = physics.subset(far_field, indices)
    assert selected.shape == (3, 5, 5)
    assert probe.shape == (5, physics.n_probes)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(5), atol=1.0e-6)
    assert kernel.shape == (5, 5)


def test_krylov_loop_is_tied_differentiable_and_effective() -> None:
    physics, far_field, mask = small_problem()
    model = TiedKrylovLSMLoop(physics.kernel, 0.01, depth=4, controller_width=8)
    score, info = model(far_field, physics.probe_rhs, return_history=True)
    loss = (
        ranking_loss(score, mask, n_pairs=5) + 0.01 * info["relative_residual"].mean()
    )
    loss.backward()
    assert model.geometry_is_frozen
    assert info["residual_history"].shape == (3, 4)
    assert all(
        parameter.grad is not None for parameter in model.controller.parameters()
    )


def test_continuous_design_angles_receive_gradients() -> None:
    physics, _, mask = small_problem()
    angles = torch.linspace(0.0, 5.0, 6, requires_grad=True)
    far_field, probe, _, kernel, _ = physics.acquisition_at_angles(
        mask, angles, noise_rel=0.0
    )
    model = TiedKrylovLSMLoop(kernel.detach(), 0.01, depth=2, controller_width=8)
    score, _ = model(far_field, probe, kernel=kernel, fixed_damping=(1.0, 1.0))
    ranking_loss(score, mask, n_pairs=4).backward()
    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()


def test_attention_pcg_preserves_solver_structure_and_gradients() -> None:
    physics, far_field, mask = small_problem()
    model = TiedAttentionPCGLSMLoop(
        physics.kernel,
        physics.feature_kernel,
        0.01,
        depth=5,
        controller_width=8,
    )
    score, info = model(far_field, physics.probe_rhs, return_history=True)
    loss = ranking_loss(score, mask, n_pairs=5) + info["relative_residual"].mean()
    loss.backward()
    assert model.geometry_is_frozen
    assert info["residual_history"].shape == (3, 5)
    assert (info["preconditioner_coefficients"] >= 0.0).all()
    assert all(
        parameter.grad is not None for parameter in model.controller.parameters()
    )


def test_stationary_variants_share_encoder_decoder_and_are_differentiable() -> None:
    physics, far_field, mask = small_problem()
    parameter_counts = set()
    for method in ("richardson", "heavy_ball", "chebyshev"):
        model = TiedStationaryLSMLoop(
            physics.kernel,
            0.01,
            depth=4,
            method=method,
            controller_width=8,
        )
        score, info = model(far_field, physics.probe_rhs, return_history=True)
        loss = ranking_loss(score, mask, n_pairs=5) + info["relative_residual"].mean()
        loss.backward()
        parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
        assert model.geometry_is_frozen
        assert score.shape == (3, physics.n_probes)
        assert info["residual_history"].shape == (3, 4)
        assert all(
            parameter.grad is not None for parameter in model.controller.parameters()
        )
    assert len(parameter_counts) == 1


def test_foundation_controller_is_a_differentiable_fixed_kernel_solver() -> None:
    physics, far_field, mask = small_problem()
    model = FoundationPCGLSMLoop(
        physics.kernel,
        physics.feature_kernel,
        0.01,
        depth=4,
        width=16,
        n_blocks=2,
        expansion=32,
        polynomial_degree=3,
    )
    score, info = model(far_field, physics.probe_rhs, return_history=True)
    loss = ranking_loss(score, mask, n_pairs=5) + info["relative_residual"].mean()
    loss.backward()
    assert model.geometry_is_frozen
    assert score.shape == (3, physics.n_probes)
    assert info["residual_history"].shape == (3, 4)
    assert info["preconditioner_coefficients"].shape == (3, 3)
    assert (info["preconditioner_coefficients"] >= 0.0).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_multi_obstacle_generator_and_bayesian_uq_are_well_formed() -> None:
    physics, far_field, _ = small_problem()
    counts = torch.tensor([1, 2, 4])
    mask, components, centres = sample_multi_obstacle_masks(
        physics, 3, counts, mode="mixed"
    )
    assert mask.shape == (3, physics.n_probes)
    assert components.shape == (3, 4, physics.n_probes)
    assert centres.shape == (3, 4, 2)
    assert mask.any(dim=-1).all()

    _, mean_info = exact_bayesian_lsm(
        far_field, physics.probe_rhs, physics.kernel, 0.01
    )
    rhs_fk = far_field @ physics.kernel
    _, covariance_info = exact_bayesian_lsm(far_field, rhs_fk, physics.kernel, 0.01)
    covariance = posterior_covariance(
        far_field,
        physics.kernel,
        covariance_info["q"],
    )
    score_mean, score_std = posterior_score_moments(
        mean_info["coefficients"],
        covariance,
        physics.kernel,
    )
    probability = occupancy_probability(score_mean, score_std, threshold=0.0)
    assert torch.linalg.eigvalsh(covariance).amin() > 0.0
    assert torch.isfinite(score_mean).all()
    assert (score_std > 0.0).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_equivariant_chebyshev_factor_is_identifiable_and_commutes_with_context() -> (
    None
):
    physics, far_field, mask = small_problem()
    model = EquivariantChebyshevPCGLSMLoop(
        physics.kernel,
        physics.feature_kernel,
        0.01,
        depth=3,
        polynomial_degree=4,
        moment_degree=4,
        controller_width=16,
    )
    score, info = model(
        far_field,
        physics.probe_rhs,
        identify_witnesses=4,
        certify=True,
    )
    loss = (
        ranking_loss(score, mask, n_pairs=4)
        + info["identification_loss"].mean()
        + 0.01 * info["relative_residual"].mean()
    )
    loss.backward()
    commutator = info["operator"] @ info["factor"] - info["factor"] @ info["operator"]
    assert torch.linalg.matrix_norm(commutator).amax() < 2.0e-4
    assert torch.isfinite(info["certificate_epsilon"]).all()
    assert (info["identification_loss"] >= 0.0).all()
    assert all(
        parameter.grad is not None for parameter in model.controller.parameters()
    )


def test_original_near_field_forward_is_physical_and_two_dimensional() -> None:
    torch.manual_seed(7)
    cfg = NearFieldConfig(
        n_sensors=12,
        grid_size=12,
        wavenumber=6.0,
        boundary_points_per_component=12,
    )
    physics = NearFieldSoundSoftLSM(cfg, torch.device("cpu"))
    near_field, probe, kernel, feature, mask, diagnostics = physics.simulate(
        2, 2, mode="mixed", noise_rel=0.10
    )
    assert near_field.shape == (2, 12, 12)
    assert near_field.is_complex()
    assert probe.shape == (2, 12, 144)
    assert kernel.shape == feature.shape == (12, 12)
    assert mask.shape == (2, 144)
    assert torch.isfinite(near_field).all()
    assert diagnostics["boundary_residual"].amax() < 2.0e-3


def test_near_field_loop_computes_parallel_posterior_moments() -> None:
    torch.manual_seed(11)
    cfg = NearFieldConfig(
        n_sensors=12,
        grid_size=12,
        wavenumber=6.0,
        boundary_points_per_component=12,
    )
    physics = NearFieldSoundSoftLSM(cfg, torch.device("cpu"))
    near_field, probe, kernel, feature, _, _ = physics.simulate(2, 2, noise_rel=0.10)
    model = PosteriorMomentHBLSMLoop(
        kernel,
        feature,
        physics.n_probes,
        depth=3,
        moment_degree=2,
        sketch_size=3,
        controller_width=16,
    )
    score, info = model(near_field, probe, certify=True, return_history=True)
    loss = score.mean() + info["score_std"].mean() + info["relative_residual"].mean()
    loss.backward()
    covariance_eigenvalues = torch.linalg.eigvalsh(info["posterior_covariance"])
    assert score.shape == (2, 144)
    assert info["q_mean"].shape == probe.shape
    assert info["q_covariance"].shape == (2, 12, 12)
    assert info["score_std"].shape == score.shape
    assert info["residual_history"].shape == (2, 3)
    assert covariance_eigenvalues.amin() > -2.0e-6
    assert (info["score_std"] > 0.0).all()
    assert all(parameter.grad is not None for parameter in model.parameters())

    exact_score, exact_info = exact_near_field_lsm(near_field, probe, kernel)
    assert exact_score.shape == score.shape
    assert exact_info["mean_relative_residual"].amax() < 3.0e-4
    assert exact_info["covariance_relative_residual"].amax() < 3.0e-4


def test_near_field_loop_accepts_variable_context_lengths() -> None:
    device = torch.device("cpu")
    base = NearFieldSoundSoftLSM(
        NearFieldConfig(
            n_sensors=12,
            grid_size=10,
            boundary_points_per_component=8,
        ),
        device,
    )
    shorter = NearFieldSoundSoftLSM(
        NearFieldConfig(
            n_sensors=8,
            grid_size=10,
            boundary_points_per_component=8,
        ),
        device,
    )
    base_geometry = base.acquisition_geometry()
    model = PosteriorMomentHBLSMLoop(
        base_geometry["source_kernel"],
        base_geometry["receiver_feature"],
        base.n_probes,
        depth=2,
        moment_degree=2,
        sketch_size=4,
        controller_width=12,
        population_width=8,
        context_adaptive_sketch=True,
    )
    near_field, probe, kernel, feature, mask, _ = shorter.simulate(2, 1, noise_rel=0.05)
    score, info = model(
        near_field,
        probe,
        source_kernel=kernel,
        receiver_feature=feature,
        depth=2,
    )
    loss = ranking_loss(score, mask, n_pairs=4) + info["mean_relative_residual"].mean()
    loss.backward()
    assert score.shape == (2, shorter.n_probes)
    assert info["population_factor"].shape == (2, 8, 8)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_context_conditioned_pcg_factor_is_spd_and_receives_task_gradients() -> None:
    torch.manual_seed(23)
    physics = NearFieldSoundSoftLSM(
        NearFieldConfig(
            n_sensors=10,
            grid_size=9,
            boundary_points_per_component=8,
        ),
        torch.device("cpu"),
    )
    near_field, probe, kernel, feature, _, _ = physics.simulate(
        2, 2, noise_rel=0.08
    )
    model = PosteriorMomentLSMLoop(
        kernel,
        feature,
        physics.n_probes,
        depth=3,
        moment_degree=2,
        sketch_size=3,
        controller_width=12,
        population_width=8,
        task_conditioned_factor=True,
        context_adaptive_sketch=True,
        method="pcg",
    )
    score, info = model(near_field, probe, certify=True)
    loss = score.square().mean() + info["mean_relative_residual"].mean()
    loss.backward()
    factors = info["population_factor"]
    assert factors.shape == (2, 10, 10)
    assert info["population_gains"].shape == (2, 10)
    assert torch.linalg.eigvalsh(factors).amin() > 0.0
    system = build_near_field_system(near_field, kernel, probe)
    physical_residual = probe - system["hessian"] @ info["q_mean"]
    expected_physical_relative = torch.sqrt(
        physical_residual.abs().square().sum(dim=(1, 2))
        / probe.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
    )
    assert torch.allclose(
        info["original_mean_relative_residual"],
        expected_physical_relative,
        atol=1.0e-6,
    )
    assert torch.allclose(
        info["transformed_mean_relative_residual"],
        info["mean_relative_residual"],
    )
    correction = model.population.correction[-1]
    assert correction.weight.grad is not None
    assert correction.weight.grad.norm() > 0.0


def test_pcg_skips_unused_endpoint_controller_only_during_inference() -> None:
    torch.manual_seed(31)
    physics = NearFieldSoundSoftLSM(
        NearFieldConfig(
            n_sensors=8,
            grid_size=7,
            boundary_points_per_component=8,
        ),
        torch.device("cpu"),
    )
    near_field, probe, kernel, feature, _, _ = physics.simulate(
        1, 1, noise_rel=0.05
    )
    model = PosteriorMomentLSMLoop(
        kernel,
        feature,
        physics.n_probes,
        depth=2,
        moment_degree=2,
        sketch_size=2,
        controller_width=8,
        population_width=8,
        context_adaptive_sketch=True,
        method="pcg",
    )
    model.eval()
    _, inference_info = model(near_field, probe)
    assert inference_info["moment_features"].shape == (1, 0)
    model.train()
    _, training_info = model(near_field, probe, certify=True)
    assert training_info["moment_features"].shape[-1] > 0


def test_hybrid_context_factor_starts_from_analytic_angular_jacobi() -> None:
    torch.manual_seed(37)
    physics = NearFieldSoundSoftLSM(
        NearFieldConfig(
            n_sensors=8,
            grid_size=7,
            boundary_points_per_component=8,
        ),
        torch.device("cpu"),
    )
    near_field, probe, kernel, feature, _, _ = physics.simulate(
        2, 2, noise_rel=0.05
    )
    model = PosteriorMomentLSMLoop(
        kernel,
        feature,
        physics.n_probes,
        depth=2,
        moment_degree=2,
        sketch_size=2,
        controller_width=8,
        population_width=8,
        task_conditioned_factor=True,
        analytic_context_factor=True,
        context_adaptive_sketch=True,
        method="pcg",
    )
    model.train()
    _, info = model(near_field, probe, certify=True)
    system = build_near_field_system(near_field, kernel, probe)
    _, eigenvectors_real = torch.linalg.eigh(feature.real)
    eigenvectors = eigenvectors_real.to(near_field.dtype)
    rotated = eigenvectors.mH.unsqueeze(0) @ system["hessian"] @ eigenvectors.unsqueeze(0)
    diagonal = rotated.diagonal(dim1=-2, dim2=-1).real.clamp_min(1.0e-8)
    scale = torch.exp(torch.log(diagonal).mean(dim=-1, keepdim=True))
    expected = (scale / diagonal).clamp(0.02, 50.0)
    assert torch.allclose(info["population_gains"], expected, atol=2.0e-5)
    assert torch.allclose(info["base_population_gains"], expected, atol=2.0e-5)
    assert torch.linalg.eigvalsh(info["population_factor"]).amin() > 0.0
    with torch.no_grad():
        model.population.correction[-1].bias.fill_(10.0)
        _, corrected = model(near_field, probe, certify=True)
    ratio = corrected["population_gains"] / corrected[
        "base_population_gains"
    ].clamp_min(1.0e-8)
    assert ratio.amax() <= torch.exp(torch.tensor(0.5)) + 1.0e-5
