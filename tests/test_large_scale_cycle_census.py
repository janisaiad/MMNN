import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.basin_boundary_step_scan import (
    cluster_signature,
)
from experiments.spectral_self_attention.basin_partition_mismatch_scaling import (
    decode_partition,
    partition_codes,
    signature_codes,
)
from experiments.spectral_self_attention.continuous_ode_audit import (
    ode_lyapunov,
    rk4_step,
)
from experiments.spectral_self_attention.continuous_curl_census import (
    beta0_harmonic_drift,
    beta0_pairwise_fourier_force,
    field_jacobian,
)
from experiments.spectral_self_attention.compile_small_step_final_results import (
    apply_internal_mover_decay_corrections,
    apply_unresolved_mover_corrections,
    correct_long_replay_by_family,
    final_extension_work_scale,
    paired_noise_sensitivity,
    stability_adjusted_continuation_endpoint_agreement,
    threshold_sensitivity,
    unscreened_random_model_work_scale,
)
from experiments.spectral_self_attention.large_scale_cycle_census import (
    classify_history,
    draw_models,
    map_angles,
    serializable_model,
    wrap,
)
from experiments.spectral_self_attention.mlp_equilibrium_taxonomy import random_block
from experiments.spectral_self_attention.period_bifurcation_sweep import (
    contiguous_windows,
)
from experiments.spectral_self_attention.periodic_orbit_audit import (
    block_from_record,
    refine_example,
)
from experiments.spectral_self_attention.random_model_finite_horizon import (
    run as run_random_model_finite_horizon,
)
from experiments.spectral_self_attention.random_state_finite_horizon import (
    feature_errors,
)
from experiments.spectral_self_attention.small_step_continuation import (
    beta0_type1_potential,
    continuous_angular_field,
    evaluate_ratio,
)


def periodic_history(values: list[list[float]], repeats: int = 48) -> np.ndarray:
    cycle = np.asarray(values, dtype=float)
    sequence = np.concatenate([cycle for _ in range(repeats)], axis=0)
    return sequence[:, None, None, :]


def test_classifier_recovers_primitive_periods_one_through_four() -> None:
    cycles = {
        1: [[0.3, -0.7]],
        2: [[0.3, -0.7], [1.1, 0.2]],
        3: [[0.3, -0.7], [1.1, 0.2], [-2.0, 2.4]],
        4: [[0.3, -0.7], [1.1, 0.2], [-2.0, 2.4], [2.7, -1.4]],
    }
    for expected, values in cycles.items():
        periods, residuals, rotations = classify_history(periodic_history(values))
        assert periods.item() == expected
        assert residuals.item() < 1e-12
        assert not rotations.item()


def test_cluster_signature_respects_circle_wraparound() -> None:
    angles = np.array([np.pi - 2e-5, -np.pi + 2e-5, 0.2, 0.20003])
    assert cluster_signature(angles, tolerance=1e-3) == [2, 2]


def test_partition_codes_retain_token_identity_and_circle_wrap() -> None:
    angles = np.array(
        [
            [np.pi - 1e-5, 0.2, -np.pi + 1e-5, 0.2],
            [0.2, np.pi - 1e-5, -np.pi + 1e-5, 0.2],
        ]
    )
    codes = partition_codes(angles)
    assert codes[0] != codes[1]
    assert decode_partition(int(codes[0]), 4) == [[1, 3], [2, 4]]
    assert decode_partition(int(codes[1]), 4) == [[1, 4], [2, 3]]
    assert np.array_equal(signature_codes(codes, 4), [12, 12])


def test_classifier_does_not_mistake_rigid_rotation_for_short_cycle() -> None:
    times = np.arange(48, dtype=float)
    history = np.stack((0.17 * times, 0.17 * times + 1.1), axis=-1)
    periods, _, rotations = classify_history(history[:, None, None, :])
    assert periods.item() == 0
    assert rotations.item()


def test_all_four_model_families_map_batches_to_circle() -> None:
    for family in (1, 2, 3, 4):
        rng = np.random.default_rng(100 + family)
        models = draw_models(rng, family=family, count=5)
        angles = rng.uniform(-np.pi, np.pi, size=(5, 7, 3))
        output = map_angles(angles, models)
        assert output.shape == angles.shape
        assert np.all(np.isfinite(output))
        assert np.all(output >= -np.pi)
        assert np.all(output <= np.pi)


def test_potential_families_have_symmetric_linear_and_tied_quadratic_terms() -> None:
    for family in (1, 3):
        models = draw_models(np.random.default_rng(200 + family), family, count=12)
        np.testing.assert_allclose(
            models["linear"], np.swapaxes(models["linear"], -1, -2), atol=1e-12
        )
        expected = np.swapaxes(models["hidden"], -1, -2)
        nonzero = np.abs(expected) > 1e-10
        ratios = np.divide(
            models["output"],
            expected,
            out=np.zeros_like(models["output"]),
            where=nonzero,
        )
        for model in range(ratios.shape[0]):
            for unit in range(ratios.shape[2]):
                components = ratios[model, :, unit][nonzero[model, :, unit]]
                if components.size:
                    np.testing.assert_allclose(components, components[0], atol=1e-12)


def test_refinement_certifies_known_primitive_period_two() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([1234, 6, 33]))
    block = replace(random_block(rng, "equal", "potential"), step_size=0.6)
    record = refine_example(block, np.array([-0.94, 1.13]), period=2)
    assert record["primitive"]
    assert record["stable"]
    assert record["primitive_residuals"]["p2"] < 1e-10


def test_serial_and_batched_maps_agree() -> None:
    rng = np.random.default_rng(410)
    models = draw_models(rng, family=3, count=1)
    block = block_from_record({"model": serializable_model(models, 0)}, family=3)
    angles = rng.uniform(-np.pi, np.pi, size=3)
    expected = block.map_angles(angles)
    actual = map_angles(angles[None, None, :], models)[0, 0]
    np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_period_windows_follow_adjacent_grid_rows() -> None:
    rows = [
        {"step_size": 0.1, "p3": 0},
        {"step_size": 0.2, "p3": 2},
        {"step_size": 0.5, "p3": 1},
        {"step_size": 0.7, "p3": 0},
        {"step_size": 0.8, "p3": 4},
    ]
    assert contiguous_windows(rows, 3) == [[0.2, 0.5], [0.8, 0.8]]


def test_continuous_field_is_small_step_limit_of_serial_map() -> None:
    rng = np.random.default_rng(510)
    models = draw_models(rng, family=3, count=4)
    angles = rng.uniform(-np.pi, np.pi, size=(4, 3, 2))
    expected = continuous_angular_field(angles, models)
    epsilon = 1e-7
    models["step_size"].fill(epsilon)
    actual = wrap(map_angles(angles, models) - angles) / epsilon
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_small_step_evaluator_returns_finite_metrics() -> None:
    rng = np.random.default_rng(520)
    models = draw_models(rng, family=1, count=3)
    original_steps = models["step_size"].copy()
    angles = rng.uniform(-np.pi, np.pi, size=(3, 1, 2))
    final, metrics = evaluate_ratio(
        angles,
        models,
        original_steps,
        ratio=0.5,
        burn_time=2.0,
        sample_count=12,
        sample_spacing=0.25,
        lyapunov_time=2.0,
        rng=rng,
    )
    assert final.shape == angles.shape
    for values in metrics.values():
        assert np.all(np.isfinite(values))


def test_rk4_step_has_continuous_field_as_derivative() -> None:
    rng = np.random.default_rng(530)
    models = draw_models(rng, family=4, count=3)
    original_steps = models["step_size"].copy()
    angles = rng.uniform(-np.pi, np.pi, size=(3, 1, 3))
    expected = original_steps[:, None, None] * continuous_angular_field(angles, models)
    dt = 1e-6
    actual = wrap(rk4_step(angles, models, original_steps, dt) - angles) / dt
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_random_state_feature_errors_ignore_common_rotation() -> None:
    angles = np.array([[[0.1, 1.2, -2.0]]])
    rotated = wrap(angles + 0.73)
    score = np.broadcast_to(np.eye(2), (1, 2, 2)).copy()
    gram_error, kernel_error = feature_errors(rotated, angles, score)
    np.testing.assert_allclose(gram_error, 0.0, atol=1e-14)
    np.testing.assert_allclose(kernel_error, 0.0, atol=1e-14)


def test_anti_lock_kicks_do_not_create_false_lyapunov_growth() -> None:
    models = draw_models(np.random.default_rng(534), family=4, count=2)
    angles = np.random.default_rng(535).uniform(
        -np.pi, np.pi, size=(2, 1, 4)
    )
    rng = np.random.default_rng(536)
    control_rng = np.random.default_rng(536)
    exponents = ode_lyapunov(
        angles,
        models,
        np.zeros(2),
        rng,
        dt=0.02,
        duration=2.0,
        anti_lock_noise=1e-6,
        anti_lock_interval=0.1,
        anti_lock_rng=np.random.default_rng(537),
    )
    ode_lyapunov(
        angles,
        models,
        np.zeros(2),
        control_rng,
        dt=0.02,
        duration=2.0,
    )
    np.testing.assert_allclose(exponents, 0.0, atol=1e-7)
    np.testing.assert_allclose(rng.normal(size=8), control_rng.normal(size=8))


def test_beta0_type1_continuous_field_climbs_global_potential() -> None:
    rng = np.random.default_rng(540)
    models = draw_models(rng, family=1, count=4)
    models["beta"].fill(0.0)
    angles = rng.uniform(-np.pi, np.pi, size=(4, 2, 3))
    field = continuous_angular_field(angles, models)
    velocity = models["step_size"][:, None, None] * field
    epsilon = 1e-6
    forward = beta0_type1_potential(wrap(angles + epsilon * velocity), models)
    backward = beta0_type1_potential(wrap(angles - epsilon * velocity), models)
    actual = (forward - backward) / (2.0 * epsilon)
    expected = models["step_size"][:, None] * np.sum(field * field, axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_beta0_harmonic_formula_matches_full_torus_average() -> None:
    rng = np.random.default_rng(550)
    models = draw_models(rng, family=4, count=3)
    models["beta"].fill(0.0)
    grid = np.linspace(-np.pi, np.pi, 64, endpoint=False)
    first, second = np.meshgrid(grid, grid, indexing="ij")
    base = np.stack((first.ravel(), second.ravel()), axis=-1)
    angles = np.broadcast_to(base, (3, *base.shape)).copy()
    actual = np.mean(continuous_angular_field(angles, models), axis=1)
    expected = np.repeat(beta0_harmonic_drift(models, n_tokens=2)[:, None], 2, axis=1)
    np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_beta0_pairwise_force_has_exact_first_harmonic_form() -> None:
    rng = np.random.default_rng(555)
    theta_i = rng.uniform(-np.pi, np.pi, size=20)
    theta_j = rng.uniform(-np.pi, np.pi, size=20)
    value = rng.normal(size=(20, 2, 2))
    tangent_i = np.stack((-np.sin(theta_i), np.cos(theta_i)), axis=-1)
    token_j = np.stack((np.cos(theta_j), np.sin(theta_j)), axis=-1)
    direct = np.einsum("md,mde,me->m", tangent_i, value, token_j)
    fourier = beta0_pairwise_fourier_force(theta_i, theta_j, value)
    np.testing.assert_allclose(fourier, direct, atol=2e-15)


def test_row_softmax_creates_curl_despite_tied_symmetric_attention() -> None:
    models = draw_models(np.random.default_rng(560), family=1, count=1)
    value = np.array([[[2.0, 0.4], [0.4, -1.0]]])
    models["score"] = value.copy()
    models["value"] = value.copy()
    models["beta"].fill(1.7)
    models["mlp_bias"].fill(0.0)
    models["linear"].fill(0.0)
    models["output"].fill(0.0)
    angles = np.array([[[0.2, 1.4]]])
    jacobian = field_jacobian(angles, models)[0]
    assert abs(jacobian[0, 1] - jacobian[1, 0]) > 0.1
    models["beta"].fill(0.0)
    beta0_jacobian = field_jacobian(angles, models)[0]
    np.testing.assert_allclose(beta0_jacobian, beta0_jacobian.T, atol=1e-9)


def audit_record(index: int, motion: float, gram: float = 0.0) -> dict:
    return {
        "family": 3,
        "label": "chaos",
        "identity": {
            "n_tokens": 4,
            "subtype_code": 1,
            "source_model_index": index,
        },
        "metrics": {
            "motion_per_normalized_time": motion,
            "gram_variation": gram,
        },
    }


def test_noise_sensitivity_pairs_records_by_identity() -> None:
    baseline = [audit_record(1, 0.0), audit_record(2, 0.2)]
    stronger = [audit_record(2, 0.0), audit_record(1, 0.1)]
    result = paired_noise_sensitivity(baseline, stronger)
    assert result["agreement_fraction"] == 0.0
    assert result["transition_counts"]["fixed_to_moving"] == 1
    assert result["transition_counts"]["moving_to_fixed"] == 1


def test_motion_threshold_sensitivity_is_monotone() -> None:
    records = [
        audit_record(1, 5e-5, 5e-5),
        audit_record(2, 5e-4, 5e-4),
        audit_record(3, 5e-3, 5e-3),
    ]
    counts = [row["moving"] for row in threshold_sensitivity(records)]
    assert counts == sorted(counts, reverse=True)


def test_promoting_unresolved_mover_does_not_remove_a_fixed_record() -> None:
    rows = [
        {
            "family": 4,
            "records": 3,
            "fixed": 1,
            "moving": 1,
            "internal_shape_motion": 1,
        }
    ]
    promoted = apply_unresolved_mover_corrections(rows, {4: 1})[0]
    assert promoted["fixed"] == 1
    assert promoted["moving"] == 2
    assert promoted["internal_shape_motion"] == 2
    decayed = apply_internal_mover_decay_corrections([promoted], {4: 1})[0]
    assert decayed["fixed"] == 2
    assert decayed["moving"] == 1
    assert decayed["internal_shape_motion"] == 1


def test_long_replay_correction_targets_only_requested_family() -> None:
    rows = [
        {"family": 3, "still_moving": 4, "still_internal": 2},
        {"family": 4, "still_moving": 7, "still_internal": 5},
    ]
    corrected = correct_long_replay_by_family(rows, {3: 1})
    assert corrected[0]["still_moving"] == 5
    assert corrected[0]["still_internal"] == 3
    assert corrected[1] == rows[1]


def test_final_extension_work_scale_counts_only_latest_ratios(tmp_path: Path) -> None:
    path = tmp_path / "extension.json"
    path.write_text(
        json.dumps(
            {
                "settings": {
                    "extension": {
                        "ratios": [0.25],
                        "sample_count": 2,
                        "lyapunov_time_normalized": 1.0,
                    }
                },
                "records": [
                    {
                        "identity": {"n_tokens": 3},
                        "trace": [
                            {
                                "ratio": 0.5,
                                "burn_layers": 2,
                                "sample_stride_layers": 2,
                            },
                            {
                                "ratio": 0.25,
                                "burn_layers": 3,
                                "sample_stride_layers": 4,
                            },
                        ],
                    }
                ],
            }
        )
    )
    result = final_extension_work_scale([path])
    assert result["model_layer_updates"] == 19
    assert result["token_layer_updates"] == 57


def test_random_model_convergence_covers_partial_final_batch() -> None:
    result = run_random_model_finite_horizon(
        families=[1],
        token_counts=[1, 2],
        models_per_cell=3,
        batch_size=2,
        ratios=[0.5, 0.25],
        horizon=1.0,
        reference_dt=0.1,
        seed=19,
    )
    assert result["summary"]["models"] == 6
    assert len(result["records"]) == 4
    assert all(row["models"] == 3 for row in result["records"])
    assert all(
        np.isfinite(value)
        for value in result["summary"]["median_error_orders"].values()
    )
    assert result["summary"]["by_token_count"]["1"][
        "median_error_orders"
    ]["gram"] is None


def test_unscreened_work_scale_counts_layers_tokens_and_rk4(
    tmp_path: Path,
) -> None:
    path = tmp_path / "random_model_finite_horizon_fixture.json"
    path.write_text(
        json.dumps(
            {
                "settings": {
                    "families": [1, 4],
                    "token_counts": [1, 3],
                    "models_per_cell": 5,
                    "ratios": [0.5, 0.25],
                    "horizon_normalized": 2.0,
                    "reference_rk4_dt": 0.2,
                }
            }
        )
    )
    result = unscreened_random_model_work_scale(tmp_path)
    assert result == {
        "files": 1,
        "discrete_model_layer_updates": 240,
        "discrete_token_layer_updates": 480,
        "rk4_model_field_evaluations": 800,
        "rk4_token_field_evaluations": 1600,
    }


def test_stability_adjusted_endpoint_agreement_promotes_and_decays(
    tmp_path: Path,
) -> None:
    identities = [
        {"n_tokens": 8, "subtype_code": 0, "source_model_index": index}
        for index in (10, 11)
    ]
    path = tmp_path / "continuation.json"
    path.write_text(
        json.dumps(
            {
                "family": 3,
                "label": "chaos",
                "records": [
                    {
                        "identity": identities[0],
                        "trace": [
                            {
                                "motion_per_normalized_time": 0.0,
                                "lyapunov_per_normalized_time": 0.1,
                            }
                        ],
                    },
                    {
                        "identity": identities[1],
                        "trace": [
                            {
                                "motion_per_normalized_time": 0.2,
                                "lyapunov_per_normalized_time": -0.1,
                            }
                        ],
                    },
                ],
            }
        )
    )
    ode_records = [
        {
            "family": 3,
            "label": "chaos",
            "identity": identity,
            "metrics": {"motion_per_normalized_time": 0.2},
        }
        for identity in identities
    ]
    decay_key = (3, "chaos", 8, 0, 11)
    result = stability_adjusted_continuation_endpoint_agreement(
        [path], ode_records, ode_promote_keys=set(), ode_decay_keys={decay_key}
    )
    assert result["finite_unstable_lock_promotions"] == 1
    assert result["finite_moving"] == 2
    assert result["ode_moving"] == 1
    assert result["transitions"]["finite_moving_to_ode_moving"] == 1
    assert result["transitions"]["finite_moving_to_ode_fixed"] == 1
