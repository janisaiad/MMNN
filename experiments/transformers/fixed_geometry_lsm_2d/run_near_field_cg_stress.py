#!/usr/bin/env python3
"""Continuous physical stress audit for CG and the final preconditioners."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .lsm_core import set_seed
from .near_field_lsm import exact_near_field_lsm
from .run_near_field_classical_preconditioners import solve_classical_pcg
from .run_near_field_scaling import (
    EvaluationBatch,
    ScalingScenario,
    append_rows,
    build_physics_cache,
    comma_ints,
    existing_keys,
    make_model,
    numerical_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--geometry-draws", type=int, default=24)
    parser.add_argument("--tasks-per-geometry", type=int, default=4)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--moment-degree", type=int, default=6)
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def stress_scenarios() -> tuple[tuple[str, float, ScalingScenario], ...]:
    """One-factor-at-a-time sweep around the four-obstacle ID condition."""
    rows: list[tuple[str, float, ScalingScenario]] = []
    for count in (1, 2, 4, 6):
        rows.append(
            (
                "obstacle_count",
                float(count),
                ScalingScenario(
                    f"count={count}", "stress", count, "mixed", 8.0, 0.15
                ),
            )
        )
    for noise in (0.05, 0.15, 0.30, 0.50):
        rows.append(
            (
                "relative_noise",
                noise,
                ScalingScenario(
                    f"noise={noise:g}", "stress", 4, "mixed", 8.0, noise
                ),
            )
        )
    for aperture in (360.0, 270.0, 180.0, 120.0):
        rows.append(
            (
                "aperture_degrees",
                aperture,
                ScalingScenario(
                    f"aperture={aperture:g}",
                    "stress",
                    4,
                    "mixed",
                    8.0,
                    0.15,
                    aperture,
                    0.10,
                ),
            )
        )
    for wavenumber in (6.0, 8.0, 10.0, 12.0):
        rows.append(
            (
                "wavenumber",
                wavenumber,
                ScalingScenario(
                    f"wavenumber={wavenumber:g}",
                    "stress",
                    4,
                    "mixed",
                    wavenumber,
                    0.15,
                ),
            )
        )
    joint_levels = (
        (0, 4, 0.15, 360.0, 8.0, 0.05),
        (1, 4, 0.30, 270.0, 10.0, 0.10),
        (2, 6, 0.30, 180.0, 12.0, 0.15),
        (3, 6, 0.50, 120.0, 12.0, 0.20),
    )
    for severity, count, noise, aperture, wavenumber, jitter in joint_levels:
        rows.append(
            (
                "joint_severity",
                float(severity),
                ScalingScenario(
                    f"joint={severity}",
                    "joint stress",
                    count,
                    "mixed",
                    wavenumber,
                    noise,
                    aperture,
                    jitter,
                ),
            )
        )
    return tuple(rows)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.geometry_draws < 1 or args.tasks_per_geometry < 1:
        raise ValueError("geometry draws and tasks per geometry must be positive")
    device = torch.device(args.device)
    seeds = comma_ints(args.seeds)
    output_path = args.results_dir / "cg_stress_sweep.csv"
    if not args.resume:
        output_path.unlink(missing_ok=True)
    keys = existing_keys(
        output_path,
        ("seed", "axis", "level", "geometry_draw", "method"),
    )
    physics_cache = build_physics_cache((24, args.context), device)
    base_physics = physics_cache[(24, 8.0)]
    scenarios = stress_scenarios()
    protocol = {
        "seeds": list(seeds),
        "width": args.width,
        "context": args.context,
        "geometry_draws": args.geometry_draws,
        "tasks_per_geometry": args.tasks_per_geometry,
        "depth": args.depth,
        "methods": [
            "identity-CG",
            "population-PCG",
            "context-PCG",
            "hybrid-PCG",
            "looped-HB",
            "angular-Jacobi-PCG",
            "exact",
        ],
        "stress_levels": [
            {"axis": axis, "level": level, "scenario": scenario.name}
            for axis, level, scenario in scenarios
        ],
    }
    (args.results_dir / "cg_stress_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    for seed in seeds:
        identity = make_model(
            base_physics,
            "pcg",
            args.width,
            depth=args.depth,
            moment_degree=args.moment_degree,
            sketch_size=args.sketch_size,
            population_factor=False,
        )
        models = {"identity-CG": identity}
        for method, label in (
            ("pcg", "population-PCG"),
            ("context_pcg", "context-PCG"),
            ("hybrid_pcg", "hybrid-PCG"),
            ("heavy_ball", "looped-HB"),
        ):
            checkpoint_path = (
                args.results_dir
                / "checkpoints"
                / f"{method}_w{args.width}_seed{seed}.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model = make_model(
                base_physics,
                method,
                args.width,
                depth=args.depth,
                moment_degree=args.moment_degree,
                sketch_size=args.sketch_size,
            )
            model.load_state_dict(checkpoint["model"])
            models[label] = model
        for model in models.values():
            model.eval()

        for scenario_index, (axis, level, scenario) in enumerate(scenarios):
            physics = physics_cache[(args.context, scenario.wavenumber)]
            for draw in range(args.geometry_draws):
                expected_methods = (*models, "angular-Jacobi-PCG", "exact")
                missing_methods = {
                    method
                    for method in expected_methods
                    if (
                        str(seed),
                        axis,
                        str(level),
                        str(draw),
                        method,
                    )
                    not in keys
                }
                if not missing_methods:
                    continue
                seed_value = (
                    seed * 100_000_000
                    + scenario_index * 100_000
                    + draw * 101
                    + 71
                ) % (2**32 - 1)
                set_seed(seed_value)
                rotation = 2.0 * math.pi * float(torch.rand((), device=device))
                near_field, probe, kernel, feature, mask, _ = physics.simulate(
                    args.tasks_per_geometry,
                    scenario.count,
                    mode=scenario.mode,
                    noise_rel=scenario.noise,
                    aperture_degrees=scenario.aperture,
                    jitter_fraction=scenario.jitter,
                    rotation=rotation,
                )
                exact_score, exact_info = exact_near_field_lsm(
                    near_field, probe, kernel
                )
                batch = EvaluationBatch(
                    args.context,
                    scenario,
                    near_field,
                    probe,
                    kernel,
                    feature,
                    mask,
                    exact_score,
                    exact_info,
                    physics.grid,
                )
                outputs = {}
                if "exact" in missing_methods:
                    outputs["exact"] = (exact_score, exact_info)
                if "angular-Jacobi-PCG" in missing_methods:
                    outputs["angular-Jacobi-PCG"] = solve_classical_pcg(
                        batch,
                        "angular-Jacobi-PCG",
                        depth=args.depth,
                        block_size=4,
                        condition_diagnostics=False,
                    )
                for label, model in models.items():
                    if label not in missing_methods:
                        continue
                    outputs[label] = model(
                        near_field,
                        probe,
                        source_kernel=kernel,
                        receiver_feature=feature,
                        depth=args.depth,
                    )

                rows = []
                for label, (score, info) in outputs.items():
                    metrics = numerical_metrics(score, info, batch)
                    values = {
                        metric: float(np.mean([row[metric] for row in metrics]))
                        for metric in (
                            "average_precision",
                            "mean_relative_residual",
                            "covariance_relative_residual",
                            "original_mean_relative_residual",
                            "original_covariance_relative_residual",
                            "relative_score_error",
                            "numerical_coverage_95",
                        )
                    }
                    row = {
                        "seed": seed,
                        "axis": axis,
                        "level": level,
                        "level_label": scenario.name,
                        "geometry_draw": draw,
                        "method": label,
                        "network_width": (
                            0
                            if label
                            in ("identity-CG", "angular-Jacobi-PCG", "exact")
                            else args.width
                        ),
                        "depth": 0 if label == "exact" else args.depth,
                        "context_size": args.context,
                        "obstacle_count": scenario.count,
                        "relative_noise": scenario.noise,
                        "aperture_degrees": scenario.aperture,
                        "wavenumber": scenario.wavenumber,
                        "rotation": rotation,
                        "jitter_fraction": scenario.jitter,
                        "tasks_per_geometry": args.tasks_per_geometry,
                        **values,
                    }
                    key = tuple(
                        str(row[column])
                        for column in (
                            "seed",
                            "axis",
                            "level",
                            "geometry_draw",
                            "method",
                        )
                    )
                    if key not in keys:
                        rows.append(row)
                        keys.add(key)
                append_rows(output_path, rows)
            print(
                f"seed={seed} {axis}={level:g}: "
                f"{args.geometry_draws} independent geometries",
                flush=True,
            )


if __name__ == "__main__":
    main()
