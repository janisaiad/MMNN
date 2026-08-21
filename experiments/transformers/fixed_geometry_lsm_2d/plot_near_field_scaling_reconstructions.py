#!/usr/bin/env python3
"""Render pre-specified qualitative reconstructions from final scaling models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .run_experiments import task_metrics
from .run_near_field_classical_preconditioners import solve_classical_pcg
from .run_near_field_scaling import (
    SCENARIOS,
    build_physics_cache,
    make_evaluation_cache,
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
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--eval-tasks", type=int, default=8)
    parser.add_argument("--moment-degree", type=int, default=6)
    parser.add_argument("--sketch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.task < 0 or args.task >= args.eval_tasks:
        raise ValueError("the displayed task must be inside the evaluation batch")
    cache = build_physics_cache((args.context, 24), device)
    evaluation = make_evaluation_cache(
        args.seed, (args.context,), cache, args.eval_tasks
    )
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
    ):
        checkpoint_path = (
            args.results_dir
            / "checkpoints"
            / f"{method}_w{args.width}_seed{args.seed}.pt"
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

    scenario_names = (
        "ID four obstacles",
        "OOD six obstacles",
        "OOD stars",
        "OOD wavenumber 12",
    )
    row_labels = (
        "ground truth",
        "exact posterior score",
        "identity-CG",
        "angular-Jacobi-PCG",
        "population-PCG",
        "context-PCG",
        "hybrid-PCG",
        "context-PCG posterior std.",
    )
    figure, axes = plt.subplots(
        len(row_labels),
        len(scenario_names),
        figsize=(11.2, 17.0),
        constrained_layout=True,
    )
    scenario_lookup = {scenario.name: scenario for scenario in SCENARIOS}
    with torch.no_grad():
        for column, scenario_name in enumerate(scenario_names):
            batch = evaluation[(args.context, scenario_name)]
            task = args.task
            side = int(round(np.sqrt(batch.mask.shape[-1])))
            truth = batch.mask[task].reshape(side, side).cpu().numpy()
            exact = batch.exact_score[task].reshape(side, side).cpu().numpy()
            outputs = {}
            for label, model in models.items():
                score, info = model(
                    batch.near_field,
                    batch.probe,
                    source_kernel=batch.kernel,
                    receiver_feature=batch.feature,
                    depth=args.depth,
                )
                outputs[label] = (
                    score[task].reshape(side, side).cpu().numpy(),
                    info,
                    task_metrics(score, batch.mask, batch.grid)[task],
                )
            angular_score, angular_info = solve_classical_pcg(
                batch,
                "angular-Jacobi-PCG",
                depth=args.depth,
                block_size=4,
                condition_diagnostics=False,
            )
            outputs["angular-Jacobi-PCG"] = (
                angular_score[task].reshape(side, side).cpu().numpy(),
                angular_info,
                task_metrics(angular_score, batch.mask, batch.grid)[task],
            )
            score_arrays = [exact, *(value[0] for value in outputs.values())]
            score_min = min(float(np.nanmin(value)) for value in score_arrays)
            score_max = max(float(np.nanmax(value)) for value in score_arrays)
            axes[0, column].imshow(truth, origin="lower", cmap="gray_r", vmin=0, vmax=1)
            axes[1, column].imshow(
                exact, origin="lower", cmap="viridis", vmin=score_min, vmax=score_max
            )
            exact_metrics = task_metrics(
                batch.exact_score, batch.mask, batch.grid
            )[task]
            axes[1, column].set_xlabel(f"AP={exact_metrics['average_precision']:.3f}")
            for row, label in enumerate(
                (
                    "identity-CG",
                    "angular-Jacobi-PCG",
                    "population-PCG",
                    "context-PCG",
                    "hybrid-PCG",
                ),
                start=2,
            ):
                score_array, info, metrics = outputs[label]
                axes[row, column].imshow(
                    score_array,
                    origin="lower",
                    cmap="viridis",
                    vmin=score_min,
                    vmax=score_max,
                )
                residual = float(info["mean_relative_residual"][task])
                axes[row, column].set_xlabel(
                    f"AP={metrics['average_precision']:.3f}, res={residual:.2e}"
                )
                axes[row, column].contour(
                    truth, levels=[0.5], colors="white", linewidths=0.65
                )
            context_info = outputs["context-PCG"][1]
            posterior_std = (
                context_info["score_std"][task].reshape(side, side).cpu().numpy()
            )
            axes[7, column].imshow(posterior_std, origin="lower", cmap="magma")
            axes[7, column].contour(
                truth, levels=[0.5], colors="white", linewidths=0.65
            )
            scenario = scenario_lookup[scenario_name]
            axes[0, column].set_title(
                f"{scenario_name}\n$m={args.context}$, noise={scenario.noise:.0%}"
            )
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(
        args.results_dir / "scaling_reconstructions.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
