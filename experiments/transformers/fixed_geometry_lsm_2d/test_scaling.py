"""Structural checks for the near-field scaling-law pipeline."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .analyze_near_field_scaling import (
    aggregate_depth_audit,
    cg_gain_by_context,
    paired_cg_stress_effects,
    paired_geometry_effects,
    scaling_prediction,
    scenario_cg_comparison,
    simultaneous_geometry_bounds,
    simultaneous_heldout_bounds,
)
from .run_near_field_classical_preconditioners import METHODS, classical_factor
from .run_near_field_cg_stress import stress_scenarios
from .run_near_field_scaling import configuration_order


def test_configuration_order_prioritizes_central_pcg_and_hb() -> None:
    order = configuration_order(
        (32, 64, 128, 256, 512),
        ("pcg", "heavy_ball", "chebyshev", "richardson"),
    )
    assert order[:2] == [(128, "pcg"), (128, "heavy_ball")]
    assert set(order) == {
        (width, method)
        for width in (32, 64, 128, 256, 512)
        for method in ("pcg", "heavy_ball", "chebyshev", "richardson")
    }

    context_order = configuration_order(
        (32, 128, 512),
        ("pcg", "context_pcg", "heavy_ball"),
    )
    assert context_order[:3] == [
        (128, "context_pcg"),
        (128, "pcg"),
        (128, "heavy_ball"),
    ]


def test_simultaneous_bound_uses_heldout_count_not_training_size() -> None:
    rows = []
    for dataset_size in (128, 1024):
        for task in range(20):
            rows.append(
                {
                    "method": "population-PCG",
                    "network_width": 32,
                    "parameter_count": 100,
                    "dataset_size": dataset_size,
                    "context_size": 12,
                    "context_measurements": 144,
                    "regime_class": "ID",
                    "average_precision": 0.7 + 0.01 * (task % 2),
                    "seed": task % 2,
                    "training_seconds": float(dataset_size),
                    "mean_relative_residual": 0.1,
                    "covariance_relative_residual": 0.2,
                }
            )
    bounds = simultaneous_heldout_bounds(pd.DataFrame(rows), delta=0.05)
    selected = bounds[bounds["regime_class"] == "All"].sort_values("dataset_size")
    assert np.all(selected["n_holdout"].to_numpy() == 20)
    assert math.isclose(
        float(selected["hoeffding_slack"].iloc[0]),
        float(selected["hoeffding_slack"].iloc[1]),
    )
    assert (selected["risk_ucb"] >= selected["risk_mean"]).all()


def test_scaling_law_has_expected_monotone_terms() -> None:
    parameters = np.asarray([0.05, 0.20, 0.5, 0.30, 0.8, 0.10, 0.4, 0.0])
    design = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
            [1.0, 4.0, 1.0],
            [1.0, 1.0, 4.0],
        ]
    )
    risks = scaling_prediction(parameters, design)
    assert risks[1] < risks[0]
    assert risks[2] < risks[0]
    assert risks[3] < risks[0]


def test_depth_aggregation_keeps_all_regime_task_count_exact() -> None:
    rows = []
    for regime in ("ID", "OOD noise"):
        for task in range(3):
            rows.append(
                {
                    "seed": 1,
                    "method": "population-PCG",
                    "network_width": 32,
                    "parameter_count": 100,
                    "dataset_size": 128,
                    "depth": 4,
                    "context_size": 12,
                    "context_measurements": 144,
                    "regime": regime,
                    "task": task,
                    "average_precision": 0.75,
                    "mean_relative_residual": 0.1,
                    "covariance_relative_residual": 0.2,
                    "relative_score_error": 0.05,
                    "numerical_coverage_95": 0.9,
                }
            )
    aggregate = aggregate_depth_audit(pd.DataFrame(rows))
    all_row = aggregate[aggregate["regime_class"] == "All"].iloc[0]
    id_row = aggregate[aggregate["regime_class"] == "ID"].iloc[0]
    assert int(all_row["n_tasks"]) == 6
    assert int(id_row["n_tasks"]) == 3


def test_geometry_bound_counts_batches_instead_of_tasks_within_batch() -> None:
    rows = []
    for seed in (1, 2):
        for geometry_draw in range(4):
            rows.append(
                {
                    "seed": seed,
                    "geometry_draw": geometry_draw,
                    "method": "context-PCG",
                    "network_width": 32,
                    "depth": 8,
                    "context_size": 12,
                    "scenario": "ID four obstacles",
                    "regime": "ID",
                    "tasks_per_geometry": 16,
                    "localization_risk": 0.2,
                    "solver_risk": 0.1,
                }
            )
    bounds = simultaneous_geometry_bounds(pd.DataFrame(rows), delta=0.05)
    scenario = bounds[bounds["scenario"] == "ID four obstacles"].iloc[0]
    assert int(scenario["n_geometry_batches"]) == 8
    assert int(scenario["n_unique_geometries"]) == 8
    assert int(scenario["tasks_per_geometry"]) == 16
    assert float(scenario["risk_ucb"]) >= float(scenario["risk_mean"])


def test_paired_geometry_effects_separate_solver_localization_and_uq() -> None:
    rows = []
    for draw in range(4):
        for method, residual, average_precision, coverage in (
            ("identity-CG", 0.4, 0.80, 0.20),
            ("context-PCG", 0.1, 0.81, 0.90),
            ("hybrid-PCG", 0.2, 0.80, 0.80),
            ("angular-Jacobi-PCG", 0.08, 0.81, 0.92),
        ):
            rows.append(
                {
                    "seed": 1,
                    "geometry_draw": draw,
                    "method": method,
                    "context_size": 48,
                    "scenario": "OOD geometry",
                    "mean_relative_residual": residual,
                    "average_precision": average_precision,
                    "numerical_coverage_95": coverage,
                }
            )
    effects = paired_geometry_effects(pd.DataFrame(rows), context_size=48)
    comparison = effects[
        (effects["scenario"] == "All")
        & (effects["comparison"] == "CG / context-PCG")
    ].iloc[0]
    assert math.isclose(float(comparison["geometric_mean_gain"]), 4.0)
    assert math.isclose(
        float(comparison["average_precision_delta_mean"]), 0.01
    )
    assert math.isclose(float(comparison["coverage_delta_mean"]), 0.70)


def test_scenario_comparison_matches_cg_and_context_tasks() -> None:
    rows = []
    for method, dataset_size, network_width, mean_residual in (
        ("identity-CG", 0, 0, 0.4),
        ("context-PCG", 1024, 64, 0.1),
    ):
        for task in range(3):
            rows.append(
                {
                    "method": method,
                    "dataset_size": dataset_size,
                    "network_width": network_width,
                    "context_size": 48,
                    "scenario": "OOD geometry",
                    "task": task,
                    "average_precision": 0.8,
                    "mean_relative_residual": mean_residual,
                    "covariance_relative_residual": mean_residual / 2,
                }
            )
    comparison = scenario_cg_comparison(
        pd.DataFrame(rows), width=64, dataset_size=1024, context_size=48
    )
    assert len(comparison) == 1
    assert math.isclose(
        float(comparison["transformed_mean_residual_gain"].iloc[0]), 4.0
    )
    assert math.isclose(
        float(comparison["transformed_covariance_residual_gain"].iloc[0]), 4.0
    )


def test_context_gain_table_includes_accuracy_and_runtime_ratios() -> None:
    rows = []
    for method, width, dataset, residual, runtime_ms in (
        ("identity-CG", 0, 0, 0.4, 10.0),
        ("context-PCG", 64, 1024, 0.1, 20.0),
    ):
        rows.append(
            {
                "method": method,
                "network_width": width,
                "dataset_size": dataset,
                "context_size": 24,
                "regime_class": "All",
                "average_precision_mean": 0.8,
                "mean_relative_residual_mean": residual,
                "covariance_relative_residual_mean": residual / 2,
                "numerical_coverage_95_mean": 0.9,
            }
        )
    runtime = pd.DataFrame(
        [
            {
                "seed": 1,
                "method": method,
                "network_width": width,
                "context_size": 24,
                "inference_ms": runtime_ms,
            }
            for method, width, _, _, runtime_ms in (
                ("identity-CG", 0, 0, 0.4, 10.0),
                ("context-PCG", 64, 1024, 0.1, 20.0),
            )
        ]
    )
    comparison = cg_gain_by_context(
        pd.DataFrame(rows), runtime, width=64, dataset_size=1024
    )
    assert len(comparison) == 1
    assert math.isclose(float(comparison["mean_residual_gain"].iloc[0]), 4.0)
    assert math.isclose(float(comparison["runtime_ratio"].iloc[0]), 2.0)


def test_stress_grid_and_paired_effects_preserve_geometry_pairing() -> None:
    scenarios = stress_scenarios()
    assert len(scenarios) == 20
    assert {axis for axis, _, _ in scenarios} == {
        "obstacle_count",
        "relative_noise",
        "aperture_degrees",
        "wavenumber",
        "joint_severity",
    }
    joint = [
        scenario
        for axis, level, scenario in scenarios
        if axis == "joint_severity" and level == 3.0
    ][0]
    assert joint.count == 6
    assert math.isclose(joint.noise, 0.50)
    assert math.isclose(joint.aperture, 120.0)
    assert math.isclose(joint.wavenumber, 12.0)
    rows = []
    for seed in (1, 2):
        for geometry_draw in range(3):
            for method, residual, average_precision, coverage in (
                ("identity-CG", 0.4, 0.80, 0.20),
                ("context-PCG", 0.1, 0.81, 0.90),
                ("hybrid-PCG", 0.2, 0.80, 0.80),
                ("population-PCG", 0.25, 0.81, 0.85),
                ("angular-Jacobi-PCG", 0.08, 0.81, 0.92),
                ("looped-HB", 0.3, 0.50, 0.05),
            ):
                rows.append(
                    {
                        "seed": seed,
                        "axis": "relative_noise",
                        "level": 0.3,
                        "geometry_draw": geometry_draw,
                        "method": method,
                        "mean_relative_residual": residual,
                        "average_precision": average_precision,
                        "numerical_coverage_95": coverage,
                    }
                )
    effects = paired_cg_stress_effects(pd.DataFrame(rows))
    context = effects[effects["candidate"] == "context-PCG"].iloc[0]
    assert int(context["n_geometry_batches"]) == 6
    assert math.isclose(float(context["geometric_mean_gain"]), 4.0)
    assert math.isclose(float(context["candidate_win_rate"]), 1.0)
    assert 0.0 < float(context["win_rate_ci95_lower"]) < 1.0
    assert math.isclose(float(context["win_rate_ci95_upper"]), 1.0)
    assert math.isclose(float(context["average_precision_delta_mean"]), 0.01)
    assert math.isclose(float(context["coverage_delta_mean"]), 0.70)


def test_classical_preconditioner_factors_are_hermitian_positive() -> None:
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(2, 8, 8, generator=generator, dtype=torch.complex64)
    hessian = matrix @ matrix.mH + torch.eye(8).unsqueeze(0)
    feature = torch.randn(8, 8, generator=generator)
    feature = feature @ feature.mT + torch.eye(8)
    for method in METHODS:
        factor = classical_factor(hessian, feature, method, block_size=4)
        assert torch.allclose(factor, factor.mH, atol=2.0e-5)
        assert float(torch.linalg.eigvalsh(factor).amin()) > 0.0
