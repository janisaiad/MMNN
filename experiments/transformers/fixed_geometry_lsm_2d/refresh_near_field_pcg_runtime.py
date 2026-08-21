#!/usr/bin/env python3
"""Refresh PCG timings after removing its unused endpoint-controller work."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .run_near_field_depth_scaling import timed_forward
from .run_near_field_scaling import (
    METHOD_LABELS,
    benchmark_model,
    build_physics_cache,
    comma_ints,
    comma_strings,
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
    parser.add_argument("--widths", default="32,64,128,256,512")
    parser.add_argument("--contexts", default="8,12,16,24,32,48")
    parser.add_argument("--depths", default="1,2,4,8,16,32,48,64,96,128")
    parser.add_argument("--methods", default="pcg,context_pcg,hybrid_pcg")
    parser.add_argument("--main-repeats", type=int, default=20)
    parser.add_argument("--depth-repeats", type=int, default=20)
    return parser.parse_args()


def replace_rows(
    path: Path,
    replacement: pd.DataFrame,
    keys: list[str],
) -> None:
    existing = pd.read_csv(path)
    replacement_keys = pd.MultiIndex.from_frame(replacement[keys])
    existing_keys = pd.MultiIndex.from_frame(existing[keys])
    retained = existing[~existing_keys.isin(replacement_keys)]
    output = pd.concat([retained, replacement], ignore_index=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.main_repeats < 1 or args.depth_repeats < 1:
        raise ValueError("timing repeats must be positive")
    device = torch.device(args.device)
    seeds = comma_ints(args.seeds)
    widths = comma_ints(args.widths)
    contexts = comma_ints(args.contexts)
    depths = comma_ints(args.depths)
    methods = comma_strings(args.methods)
    valid_methods = {"pcg", "context_pcg", "hybrid_pcg"}
    if not set(methods).issubset(valid_methods):
        raise ValueError(f"methods must be drawn from {sorted(valid_methods)}")
    cache = build_physics_cache(tuple(sorted(set(contexts) | {24})), device)
    main_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    for seed in seeds:
        evaluation = make_evaluation_cache(seed, contexts, cache, 8)
        base_physics = cache[(24, 8.0)]
        identity = make_model(
            base_physics,
            "pcg",
            min(widths),
            depth=max(depths),
            moment_degree=6,
            sketch_size=4,
            population_factor=False,
        )
        identity.eval()
        main_rows.extend(
            benchmark_model(
                identity,
                "identity-CG",
                evaluation,
                contexts,
                seed=seed,
                width=0,
                parameter_count_value=0,
                depth=32,
                repeats=args.main_repeats,
            )
        )
        depth_models = {"identity-CG": identity}
        for width in widths:
            for method in methods:
                checkpoint = torch.load(
                    args.results_dir
                    / "checkpoints"
                    / f"{method}_w{width}_seed{seed}.pt",
                    map_location=device,
                    weights_only=False,
                )
                model = make_model(
                    base_physics,
                    method,
                    width,
                    depth=max(depths),
                    moment_degree=6,
                    sketch_size=4,
                )
                model.load_state_dict(checkpoint["model"])
                model.eval()
                label = METHOD_LABELS[method]
                main_rows.extend(
                    benchmark_model(
                        model,
                        label,
                        evaluation,
                        contexts,
                        seed=seed,
                        width=width,
                        parameter_count_value=parameter_count(model),
                        depth=32,
                        repeats=args.main_repeats,
                    )
                )
                if width == 128:
                    depth_models[label] = model
        runtime_batches = {
            context: evaluation[(context, "ID four obstacles")]
            for context in contexts
        }
        for label, model in depth_models.items():
            width = 0 if label == "identity-CG" else 128
            parameters = 0 if width == 0 else parameter_count(model)
            for depth in depths:
                for context, batch in runtime_batches.items():
                    depth_rows.append(
                        {
                            "seed": seed,
                            "method": label,
                            "network_width": width,
                            "parameter_count": parameters,
                            "dataset_size": 0 if width == 0 else 32768,
                            "context_size": context,
                            "context_measurements": context * context,
                            "depth": depth,
                            "batch_size": 8,
                            "inference_ms": timed_forward(
                                model,
                                batch,
                                depth=depth,
                                repeats=args.depth_repeats,
                            ),
                        }
                    )
        print(f"seed={seed}: refreshed PCG runtime controls", flush=True)

    replace_rows(
        args.results_dir / "runtime.csv",
        pd.DataFrame(main_rows),
        ["seed", "method", "network_width", "context_size"],
    )
    replace_rows(
        args.results_dir / "depth_runtime.csv",
        pd.DataFrame(depth_rows),
        ["seed", "method", "network_width", "depth", "context_size"],
    )


if __name__ == "__main__":
    main()
