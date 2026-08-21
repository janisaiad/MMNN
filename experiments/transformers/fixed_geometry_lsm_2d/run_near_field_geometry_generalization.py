#!/usr/bin/env python3
"""Independent-acquisition generalization audit for final near-field solvers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .lsm_core import set_seed
from .near_field_lsm import exact_near_field_lsm
from .run_near_field_classical_preconditioners import solve_classical_pcg
from .run_near_field_scaling import (
    EvaluationBatch,
    SCENARIOS,
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
    parser.add_argument("--contexts", default="8,24,48")
    parser.add_argument("--geometry-draws", type=int, default=64)
    parser.add_argument("--tasks-per-geometry", type=int, default=4)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--moment-degree", type=int, default=6)
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seeds = comma_ints(args.seeds)
    contexts = comma_ints(args.contexts)
    if args.geometry_draws < 1 or args.tasks_per_geometry < 1:
        raise ValueError("geometry draws and tasks per geometry must be positive")
    scenario_names = (
        "ID four obstacles",
        "OOD six obstacles",
        "OOD half aperture",
        "OOD wavenumber 12",
    )
    scenarios = {scenario.name: scenario for scenario in SCENARIOS}
    output_path = args.results_dir / "geometry_generalization.csv"
    if not args.resume:
        output_path.unlink(missing_ok=True)
    keys = existing_keys(
        output_path,
        ("seed", "geometry_draw", "method", "context_size", "scenario"),
    )
    cache = build_physics_cache(contexts, device)
    base_context = min(contexts, key=lambda value: abs(value - 24))

    for seed in seeds:
        base_physics = cache[(base_context, 8.0)]
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

        for context_index, context in enumerate(contexts):
            for scenario_index, scenario_name in enumerate(scenario_names):
                scenario = scenarios[scenario_name]
                physics = cache[(context, scenario.wavenumber)]
                for draw in range(args.geometry_draws):
                    expected_methods = (*models, "angular-Jacobi-PCG", "exact")
                    if all(
                        (
                            str(seed),
                            str(draw),
                            method,
                            str(context),
                            scenario_name,
                        )
                        in keys
                        for method in expected_methods
                    ):
                        continue
                    seed_value = (
                        seed * 100_000_000
                        + context_index * 1_000_000
                        + scenario_index * 100_000
                        + draw * 101
                        + 37
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
                        context,
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
                    outputs = {
                        "exact": (exact_score, exact_info),
                        "angular-Jacobi-PCG": solve_classical_pcg(
                            batch,
                            "angular-Jacobi-PCG",
                            depth=args.depth,
                            block_size=4,
                            condition_diagnostics=False,
                        ),
                    }
                    for label, model in models.items():
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
                                "auc",
                                "area_matched_iou",
                                "relative_score_error",
                                "mean_relative_residual",
                                "covariance_relative_residual",
                                "original_mean_relative_residual",
                                "original_covariance_relative_residual",
                                "uq_error_correlation",
                                "numerical_coverage_95",
                            )
                        }
                        values["localization_risk"] = 1.0 - values[
                            "average_precision"
                        ]
                        values["solver_risk"] = values[
                            "original_mean_relative_residual"
                        ] / (1.0 + values["original_mean_relative_residual"])
                        row = {
                            "seed": seed,
                            "geometry_draw": draw,
                            "method": label,
                            "network_width": (
                                0
                                if label
                                in ("identity-CG", "angular-Jacobi-PCG", "exact")
                                else args.width
                            ),
                            "depth": 0 if label == "exact" else args.depth,
                            "context_size": context,
                            "context_measurements": context * context,
                            "scenario": scenario_name,
                            "regime": scenario.regime,
                            "rotation": rotation,
                            "jitter_fraction": scenario.jitter,
                            "tasks_per_geometry": args.tasks_per_geometry,
                            **values,
                        }
                        key = tuple(
                            str(row[column])
                            for column in (
                                "seed",
                                "geometry_draw",
                                "method",
                                "context_size",
                                "scenario",
                            )
                        )
                        if key not in keys:
                            rows.append(row)
                            keys.add(key)
                    append_rows(output_path, rows)
                print(
                    f"seed={seed} m={context:2d} {scenario_name}: "
                    f"{args.geometry_draws} independent geometries",
                    flush=True,
                )


if __name__ == "__main__":
    main()
