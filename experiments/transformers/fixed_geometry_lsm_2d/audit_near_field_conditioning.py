#!/usr/bin/env python3
"""Audit what geometry and prompt conditioning do to the near-field spectrum."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .near_field_lsm import build_near_field_system
from .run_near_field_scaling import (
    METHOD_LABELS,
    append_rows,
    build_physics_cache,
    comma_ints,
    existing_keys,
    make_evaluation_cache,
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
    parser.add_argument("--contexts", default="8,12,16,24,32,48")
    parser.add_argument("--eval-tasks", type=int, default=8)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--moment-degree", type=int, default=6)
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalized_commutator(operator: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    geometry_batch = geometry.unsqueeze(0).expand(operator.shape[0], -1, -1)
    commutator = operator @ geometry_batch - geometry_batch @ operator
    numerator = torch.linalg.matrix_norm(commutator, ord="fro")
    denominator = (
        torch.linalg.matrix_norm(operator, ord="fro")
        * torch.linalg.matrix_norm(geometry_batch, ord="fro")
    ).clamp_min(1.0e-12)
    return numerator / denominator


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seeds = comma_ints(args.seeds)
    contexts = comma_ints(args.contexts)
    output_path = args.results_dir / "preconditioner_conditioning.csv"
    if not args.resume:
        output_path.unlink(missing_ok=True)
    keys = existing_keys(
        output_path,
        ("seed", "method", "context_size", "scenario", "task"),
    )
    cache = build_physics_cache(contexts, device)
    base_context = min(contexts, key=lambda value: abs(value - 24))
    methods = ("identity", "pcg", "context_pcg", "hybrid_pcg")

    for seed in seeds:
        evaluation = make_evaluation_cache(seed, contexts, cache, args.eval_tasks)
        models = {}
        for method in methods:
            if method == "identity":
                model = make_model(
                    cache[(base_context, 8.0)],
                    "pcg",
                    args.width,
                    depth=args.depth,
                    moment_degree=args.moment_degree,
                    sketch_size=args.sketch_size,
                    population_factor=False,
                )
            else:
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
                    cache[(base_context, 8.0)],
                    method,
                    args.width,
                    depth=args.depth,
                    moment_degree=args.moment_degree,
                    sketch_size=args.sketch_size,
                )
                model.load_state_dict(checkpoint["model"])
            model.eval()
            models[method] = model

        for (context, scenario), batch in evaluation.items():
            system = build_near_field_system(
                batch.near_field, batch.kernel, batch.probe
            )
            raw_operator = system["operator"]
            raw_eigenvalues = torch.linalg.eigvalsh(raw_operator)
            raw_condition = raw_eigenvalues.amax(dim=-1) / raw_eigenvalues.amin(
                dim=-1
            ).clamp_min(1.0e-12)
            commutator = normalized_commutator(raw_operator, batch.feature)
            rows = []
            for method, model in models.items():
                label = "identity-CG" if method == "identity" else METHOD_LABELS[method]
                score, info = model(
                    batch.near_field,
                    batch.probe,
                    source_kernel=batch.kernel,
                    receiver_feature=batch.feature,
                    depth=args.depth,
                    certify=True,
                )
                del score
                condition = info["true_upper"] / info["true_lower"].clamp_min(
                    1.0e-12
                )
                contraction = (
                    torch.sqrt(condition) - 1.0
                ) / (torch.sqrt(condition) + 1.0).clamp_min(1.0e-12)
                energy_error_factor = 2.0 * contraction.pow(args.depth)
                gains = info["population_gains"]
                if gains.ndim == 1:
                    gain_spread = (
                        gains.amax() / gains.amin().clamp_min(1.0e-12)
                    ).expand(args.eval_tasks)
                else:
                    gain_spread = gains.amax(dim=-1) / gains.amin(dim=-1).clamp_min(
                        1.0e-12
                    )
                for task in range(args.eval_tasks):
                    row = {
                        "seed": seed,
                        "method": label,
                        "network_width": 0 if method == "identity" else args.width,
                        "depth": args.depth,
                        "context_size": context,
                        "context_measurements": context * context,
                        "scenario": scenario,
                        "regime": batch.scenario.regime,
                        "task": task,
                        "raw_condition": float(raw_condition[task]),
                        "transformed_condition": float(condition[task]),
                        "condition_reduction": float(
                            raw_condition[task] / condition[task]
                        ),
                        "geometry_commutator": float(commutator[task]),
                        "pcg_energy_error_factor": float(energy_error_factor[task]),
                        "mean_relative_residual": float(
                            info["mean_relative_residual"][task]
                        ),
                        "covariance_relative_residual": float(
                            info["covariance_relative_residual"][task]
                        ),
                        "gain_spread": float(gain_spread[task]),
                    }
                    key = tuple(
                        str(row[column])
                        for column in (
                            "seed",
                            "method",
                            "context_size",
                            "scenario",
                            "task",
                        )
                    )
                    if key not in keys:
                        rows.append(row)
                        keys.add(key)
            append_rows(output_path, rows)
            print(
                f"seed={seed} m={context:2d} {scenario:24s}: "
                f"median raw kappa={float(np.median(raw_condition.cpu().numpy())):.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
