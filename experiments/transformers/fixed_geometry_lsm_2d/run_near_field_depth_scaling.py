#!/usr/bin/env python3
"""Evaluate recurrent depth versus context size for trained near-field LSM solvers.

This is a post-training audit: it reloads the final central-width checkpoints,
changes only the number of operator applications, and evaluates every method on
the same held-out physical tasks.  No parameters are updated.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .run_near_field_scaling import (
    LEARNED_METHODS,
    METHOD_LABELS,
    append_rows,
    build_physics_cache,
    comma_ints,
    comma_strings,
    evaluate_baselines,
    evaluate_model,
    exact_near_field_lsm,
    existing_keys,
    make_evaluation_cache,
    make_model,
    parameter_count,
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
    parser.add_argument("--depths", default="1,2,4,8,16,32,48,64")
    parser.add_argument("--methods", default=",".join(LEARNED_METHODS))
    parser.add_argument("--eval-tasks", type=int, default=8)
    parser.add_argument("--moment-degree", type=int, default=6)
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--runtime-repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def timed_forward(
    model,
    batch,
    *,
    depth: int,
    repeats: int,
) -> float:
    for _ in range(2):
        model(
            batch.near_field,
            batch.probe,
            source_kernel=batch.kernel,
            receiver_feature=batch.feature,
            depth=depth,
        )
    synchronize(batch.near_field.device)
    started = time.perf_counter()
    for _ in range(repeats):
        model(
            batch.near_field,
            batch.probe,
            source_kernel=batch.kernel,
            receiver_feature=batch.feature,
            depth=depth,
        )
    synchronize(batch.near_field.device)
    return 1_000.0 * (time.perf_counter() - started) / repeats


@torch.no_grad()
def timed_exact(batch, *, repeats: int) -> float:
    for _ in range(2):
        exact_near_field_lsm(batch.near_field, batch.probe, batch.kernel)
    synchronize(batch.near_field.device)
    started = time.perf_counter()
    for _ in range(repeats):
        exact_near_field_lsm(batch.near_field, batch.probe, batch.kernel)
    synchronize(batch.near_field.device)
    return 1_000.0 * (time.perf_counter() - started) / repeats


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seeds = comma_ints(args.seeds)
    contexts = comma_ints(args.contexts)
    depths = comma_ints(args.depths)
    methods = comma_strings(args.methods)
    unknown = set(methods) - set(LEARNED_METHODS)
    if unknown:
        raise ValueError(f"unknown learned methods: {sorted(unknown)}")
    if min(depths) < 1:
        raise ValueError("all recurrent depths must be positive")
    if args.eval_tasks < 1 or args.runtime_repeats < 1:
        raise ValueError("evaluation tasks and runtime repeats must be positive")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = args.results_dir / "depth_scaling.csv"
    runtime_path = args.results_dir / "depth_runtime.csv"
    if not args.resume:
        evaluation_path.unlink(missing_ok=True)
        runtime_path.unlink(missing_ok=True)

    evaluation_keys = existing_keys(
        evaluation_path,
        (
            "seed",
            "method",
            "network_width",
            "depth",
            "context_size",
            "scenario",
            "task",
        ),
    )
    runtime_keys = existing_keys(
        runtime_path,
        ("seed", "method", "network_width", "depth", "context_size"),
    )
    cache = build_physics_cache(contexts, device)
    base_context = min(contexts, key=lambda value: abs(value - 24))
    final_dataset_size = 0

    for seed in seeds:
        evaluation = make_evaluation_cache(seed, contexts, cache, args.eval_tasks)
        base_physics = cache[(base_context, 8.0)]
        identity_cg = make_model(
            base_physics,
            "pcg",
            args.width,
            depth=max(depths),
            moment_degree=args.moment_degree,
            sketch_size=args.sketch_size,
            population_factor=False,
        )
        identity_cg.eval()
        learned_models = {}
        checkpoints = {}
        for method in methods:
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
                depth=max(depths),
                moment_degree=args.moment_degree,
                sketch_size=args.sketch_size,
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()
            learned_models[method] = model
            checkpoints[method] = checkpoint
            final_dataset_size = max(
                final_dataset_size, int(checkpoint["completed_examples"])
            )

        for depth in depths:
            baseline_rows = evaluate_baselines(
                identity_cg, evaluation, seed=seed, depth=depth
            )
            # Exact does not depend on recurrent depth; retain it only once.
            baseline_rows = [
                row
                for row in baseline_rows
                if row["method"] == "identity-CG"
            ]
            for row in baseline_rows:
                row["depth"] = depth
            pending_baselines = []
            for row in baseline_rows:
                key = tuple(
                    str(row[column])
                    for column in (
                        "seed",
                        "method",
                        "network_width",
                        "depth",
                        "context_size",
                        "scenario",
                        "task",
                    )
                )
                if key not in evaluation_keys:
                    pending_baselines.append(row)
                    evaluation_keys.add(key)
            append_rows(evaluation_path, pending_baselines)

            for method, model in learned_models.items():
                checkpoint = checkpoints[method]
                rows = evaluate_model(
                    model,
                    METHOD_LABELS[method],
                    evaluation,
                    seed=seed,
                    width=args.width,
                    dataset_size=int(checkpoint["completed_examples"]),
                    parameter_count_value=parameter_count(model),
                    training_seconds=float(checkpoint["training_seconds"]),
                    depth=depth,
                )
                for row in rows:
                    row["depth"] = depth
                pending = []
                for row in rows:
                    key = tuple(
                        str(row[column])
                        for column in (
                            "seed",
                            "method",
                            "network_width",
                            "depth",
                            "context_size",
                            "scenario",
                            "task",
                        )
                    )
                    if key not in evaluation_keys:
                        pending.append(row)
                        evaluation_keys.add(key)
                append_rows(evaluation_path, pending)

            runtime_batch = {
                context: evaluation[(context, "ID four obstacles")]
                for context in contexts
            }
            timed_models = {"identity-CG": identity_cg}
            timed_models.update(
                {METHOD_LABELS[name]: model for name, model in learned_models.items()}
            )
            runtime_rows = []
            for label, model in timed_models.items():
                width = 0 if label == "identity-CG" else args.width
                parameters = 0 if width == 0 else parameter_count(model)
                for context, batch in runtime_batch.items():
                    key = tuple(
                        str(value) for value in (seed, label, width, depth, context)
                    )
                    if key in runtime_keys:
                        continue
                    runtime_rows.append(
                        {
                            "seed": seed,
                            "method": label,
                            "network_width": width,
                            "parameter_count": parameters,
                            "dataset_size": (
                                0 if width == 0 else final_dataset_size
                            ),
                            "context_size": context,
                            "context_measurements": context * context,
                            "depth": depth,
                            "batch_size": args.eval_tasks,
                            "inference_ms": timed_forward(
                                model,
                                batch,
                                depth=depth,
                                repeats=args.runtime_repeats,
                            ),
                        }
                    )
                    runtime_keys.add(key)
            append_rows(runtime_path, runtime_rows)
            print(
                f"seed={seed} depth={depth:2d}: "
                f"{len(pending_baselines) + sum(len(evaluation) * args.eval_tasks for _ in methods):5d} task rows",
                flush=True,
            )

        exact_rows = evaluate_baselines(
            identity_cg, evaluation, seed=seed, depth=max(depths)
        )
        exact_rows = [row for row in exact_rows if row["method"] == "exact"]
        for row in exact_rows:
            row["depth"] = 0
        pending_exact = []
        for row in exact_rows:
            key = tuple(
                str(row[column])
                for column in (
                    "seed",
                    "method",
                    "network_width",
                    "depth",
                    "context_size",
                    "scenario",
                    "task",
                )
            )
            if key not in evaluation_keys:
                pending_exact.append(row)
                evaluation_keys.add(key)
        append_rows(evaluation_path, pending_exact)

        exact_runtime = []
        for context in contexts:
            key = tuple(str(value) for value in (seed, "exact", 0, 0, context))
            if key in runtime_keys:
                continue
            batch = evaluation[(context, "ID four obstacles")]
            exact_runtime.append(
                {
                    "seed": seed,
                    "method": "exact",
                    "network_width": 0,
                    "parameter_count": 0,
                    "dataset_size": 0,
                    "context_size": context,
                    "context_measurements": context * context,
                    "depth": 0,
                    "batch_size": args.eval_tasks,
                    "inference_ms": timed_exact(
                        batch, repeats=args.runtime_repeats
                    ),
                }
            )
            runtime_keys.add(key)
        append_rows(runtime_path, exact_runtime)


if __name__ == "__main__":
    main()
