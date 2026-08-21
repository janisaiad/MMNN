#!/usr/bin/env python3
"""Render pre-specified reconstructions along the joint-shift stress path."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lsm_core import set_seed
from .near_field_lsm import exact_near_field_lsm
from .run_experiments import task_metrics
from .run_near_field_cg_stress import stress_scenarios
from .run_near_field_classical_preconditioners import solve_classical_pcg
from .run_near_field_scaling import (
    EvaluationBatch,
    build_physics_cache,
    make_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--geometry-draw", type=int, default=0)
    parser.add_argument("--tasks-per-geometry", type=int, default=4)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--moment-degree", type=int, default=6)
    parser.add_argument("--sketch-size", type=int, default=4)
    return parser.parse_args()


def normalized_score(array: np.ndarray) -> np.ndarray:
    lower, upper = np.nanpercentile(array, (1.0, 99.0))
    scale = max(float(upper - lower), 1.0e-12)
    return np.clip((array - lower) / scale, 0.0, 1.0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.task < 0 or args.task >= args.tasks_per_geometry:
        raise ValueError("task must be inside the pre-specified geometry batch")
    device = torch.device(args.device)
    cache = build_physics_cache((24, args.context), device)
    base_physics = cache[(24, 8.0)]
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
            / f"{method}_w{args.width}_seed{args.seed}.pt"
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

    all_scenarios = stress_scenarios()
    joint = [
        (index, level, scenario)
        for index, (axis, level, scenario) in enumerate(all_scenarios)
        if axis == "joint_severity"
    ]
    row_labels = (
        "ground truth",
        "exact posterior score",
        "identity-CG",
        "angular-Jacobi-PCG",
        "population-PCG",
        "context-PCG",
        "hybrid-PCG",
        "looped-HB",
        "context-PCG posterior std.",
    )
    figure, axes = plt.subplots(
        len(row_labels),
        len(joint),
        figsize=(13.2, 18.2),
        constrained_layout=True,
    )
    for column, (scenario_index, severity, scenario) in enumerate(joint):
        physics = cache[(args.context, scenario.wavenumber)]
        seed_value = (
            args.seed * 100_000_000
            + scenario_index * 100_000
            + args.geometry_draw * 101
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
        exact_score, exact_info = exact_near_field_lsm(near_field, probe, kernel)
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
        for label, model in models.items():
            score, info = model(
                near_field,
                probe,
                source_kernel=kernel,
                receiver_feature=feature,
                depth=args.depth,
            )
            outputs[label] = (score, info)
        angular_score, angular_info = solve_classical_pcg(
            batch,
            "angular-Jacobi-PCG",
            depth=args.depth,
            block_size=4,
            condition_diagnostics=False,
        )
        outputs["angular-Jacobi-PCG"] = (angular_score, angular_info)

        task = args.task
        side = int(round(np.sqrt(mask.shape[-1])))
        truth = mask[task].reshape(side, side).cpu().numpy()
        exact = exact_score[task].reshape(side, side).cpu().numpy()
        axes[0, column].imshow(truth, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        axes[1, column].imshow(
            normalized_score(exact), origin="lower", cmap="viridis", vmin=0, vmax=1
        )
        exact_metrics = task_metrics(exact_score, mask, physics.grid)[task]
        axes[1, column].set_xlabel(f"AP={exact_metrics['average_precision']:.3f}")
        for row, label in enumerate(
            (
                "identity-CG",
                "angular-Jacobi-PCG",
                "population-PCG",
                "context-PCG",
                "hybrid-PCG",
                "looped-HB",
            ),
            start=2,
        ):
            score, info = outputs[label]
            array = score[task].reshape(side, side).cpu().numpy()
            metrics = task_metrics(score, mask, physics.grid)[task]
            residual = float(info["original_mean_relative_residual"][task])
            axes[row, column].imshow(
                normalized_score(array),
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            axes[row, column].contour(
                truth, levels=[0.5], colors="white", linewidths=0.65
            )
            axes[row, column].set_xlabel(
                f"AP={metrics['average_precision']:.3f}, res={residual:.2e}"
            )
        context_info = outputs["context-PCG"][1]
        posterior_std = (
            context_info["score_std"][task].reshape(side, side).cpu().numpy()
        )
        axes[-1, column].imshow(posterior_std, origin="lower", cmap="magma")
        axes[-1, column].contour(
            truth, levels=[0.5], colors="white", linewidths=0.65
        )
        axes[0, column].set_title(
            f"joint severity {severity:g}\n"
            f"q={scenario.count}; σ={scenario.noise:.0%}; "
            f"φ={scenario.aperture:.0f}°\n"
            f"k={scenario.wavenumber:g}",
            fontsize=10,
        )
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(
        args.results_dir / "scaling_joint_reconstructions.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
