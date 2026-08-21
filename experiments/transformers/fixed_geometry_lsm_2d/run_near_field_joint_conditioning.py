#!/usr/bin/env python3
"""Spectral audit under the joint near-field distribution shifts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .audit_near_field_conditioning import normalized_commutator
from .lsm_core import set_seed
from .near_field_lsm import build_near_field_system, exact_near_field_lsm
from .run_near_field_cg_stress import stress_scenarios
from .run_near_field_classical_preconditioners import solve_classical_pcg
from .run_near_field_scaling import (
    EvaluationBatch,
    append_rows,
    build_physics_cache,
    comma_ints,
    existing_keys,
    make_model,
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
    output_path = args.results_dir / "joint_conditioning.csv"
    if not args.resume:
        output_path.unlink(missing_ok=True)
    keys = existing_keys(
        output_path,
        ("seed", "joint_severity", "geometry_draw", "method"),
    )
    cache = build_physics_cache((24, args.context), device)
    base_physics = cache[(24, 8.0)]
    all_scenarios = stress_scenarios()
    joint_scenarios = [
        (index, level, scenario)
        for index, (axis, level, scenario) in enumerate(all_scenarios)
        if axis == "joint_severity"
    ]

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
        ):
            checkpoint_path = (
                args.results_dir
                / "checkpoints"
                / f"{method}_w{args.width}_seed{seed}.pt"
            )
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

        for scenario_index, severity, scenario in joint_scenarios:
            physics = cache[(args.context, scenario.wavenumber)]
            for draw in range(args.geometry_draws):
                expected = (*models, "angular-Jacobi-PCG")
                if all(
                    (str(seed), str(severity), str(draw), method) in keys
                    for method in expected
                ):
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
                system = build_near_field_system(near_field, kernel, probe)
                raw_eigenvalues = torch.linalg.eigvalsh(system["operator"])
                raw_condition = raw_eigenvalues.amax(dim=-1) / raw_eigenvalues.amin(
                    dim=-1
                ).clamp_min(1.0e-12)
                commutator = normalized_commutator(system["operator"], feature)
                outputs: dict[str, dict[str, torch.Tensor]] = {}
                for label, model in models.items():
                    _, info = model(
                        near_field,
                        probe,
                        source_kernel=kernel,
                        receiver_feature=feature,
                        depth=args.depth,
                        certify=True,
                    )
                    outputs[label] = info
                _, angular_info = solve_classical_pcg(
                    batch,
                    "angular-Jacobi-PCG",
                    depth=args.depth,
                    block_size=4,
                    condition_diagnostics=True,
                )
                outputs["angular-Jacobi-PCG"] = angular_info

                rows = []
                for method, info in outputs.items():
                    if method == "angular-Jacobi-PCG":
                        transformed = info["transformed_condition"]
                    else:
                        transformed = info["true_upper"] / info[
                            "true_lower"
                        ].clamp_min(1.0e-12)
                    row = {
                        "seed": seed,
                        "joint_severity": severity,
                        "geometry_draw": draw,
                        "method": method,
                        "context_size": args.context,
                        "depth": args.depth,
                        "tasks_per_geometry": args.tasks_per_geometry,
                        "raw_condition_median": float(raw_condition.median()),
                        "transformed_condition_median": float(
                            transformed.median()
                        ),
                        "condition_reduction_median": float(
                            (raw_condition / transformed).median()
                        ),
                        "geometry_commutator_mean": float(commutator.mean()),
                        "mean_relative_residual_mean": float(
                            info["mean_relative_residual"].mean()
                        ),
                        "covariance_relative_residual_mean": float(
                            info["covariance_relative_residual"].mean()
                        ),
                        "original_mean_relative_residual_mean": float(
                            info["original_mean_relative_residual"].mean()
                        ),
                        "original_covariance_relative_residual_mean": float(
                            info["original_covariance_relative_residual"].mean()
                        ),
                    }
                    key = tuple(
                        str(row[column])
                        for column in (
                            "seed",
                            "joint_severity",
                            "geometry_draw",
                            "method",
                        )
                    )
                    if key not in keys:
                        rows.append(row)
                        keys.add(key)
                append_rows(output_path, rows)
            print(
                f"seed={seed} joint={severity:g}: "
                f"{args.geometry_draws} conditioning batches",
                flush=True,
            )


if __name__ == "__main__":
    main()
