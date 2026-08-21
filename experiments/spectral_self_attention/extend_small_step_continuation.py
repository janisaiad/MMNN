"""Extend an existing small-step continuation to still smaller layer sizes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.small_step_continuation import (
    evaluate_ratio,
    stack_models,
)


def final_counts(records: list[dict[str, object]]) -> dict[str, int]:
    final = [record["trace"][-1] for record in records]
    return {
        "fixed": sum(bool(row["fixed"]) for row in final),
        "map_stationary": sum(bool(row["map_stationary"]) for row in final),
        "moving_recurrent": sum(bool(row["moving_recurrent"]) for row in final),
        "positive_lyapunov": sum(bool(row["positive_lyapunov"]) for row in final),
        "moving_nonrecurrent": sum(
            row["motion_per_normalized_time"] >= 1e-3
            and not row["moving_recurrent"]
            for row in final
        ),
    }


def run(
    source: dict[str, object],
    ratios: list[float],
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    seed: int,
    continuation_noise: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    records = source["records"]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[int(record["identity"]["n_tokens"])].append(index)
    for indices in groups.values():
        selected = [records[index] for index in indices]
        models = stack_models(selected)
        original_steps = models["step_size"].astype(float).copy()
        angles = np.asarray([record["final_angle"] for record in selected])[:, None, :]
        for ratio in ratios:
            angles, metrics = evaluate_ratio(
                angles,
                models,
                original_steps,
                ratio,
                burn_time,
                sample_count,
                sample_spacing,
                lyapunov_time,
                rng,
                continuation_noise,
            )
            for local, global_index in enumerate(indices):
                motion = float(metrics["motion_per_normalized_time"][local])
                field_rms = float(metrics["continuous_field_rms"][local])
                recurrence = float(metrics["recurrence_error"][local])
                lyapunov = float(metrics["lyapunov_per_normalized_time"][local])
                point = {
                    "ratio": ratio,
                    "absolute_step": float(metrics["absolute_step_min"][local]),
                    "motion_per_normalized_time": motion,
                    "continuous_field_rms": field_rms,
                    "relative_local_error": float(metrics["relative_local_error"][local]),
                    "absolute_local_error": float(metrics["absolute_local_error"][local]),
                    "recurrence_error": recurrence,
                    "return_time_normalized": float(
                        metrics["return_time_normalized"][local]
                    ),
                    "gram_variation": float(metrics["gram_variation"][local]),
                    "gram_recurrence_error": float(
                        metrics["gram_recurrence_error"][local]
                    ),
                    "gram_return_time_normalized": float(
                        metrics["gram_return_time_normalized"][local]
                    ),
                    "score_kernel_variation": float(
                        metrics["score_kernel_variation"][local]
                    ),
                    "score_kernel_recurrence_error": float(
                        metrics["score_kernel_recurrence_error"][local]
                    ),
                    "score_kernel_return_time_normalized": float(
                        metrics["score_kernel_return_time_normalized"][local]
                    ),
                    "mean_coherence": float(metrics["mean_coherence"][local]),
                    "mean_winding_speed": float(metrics["mean_winding_speed"][local]),
                    "lyapunov_per_normalized_time": lyapunov,
                    "fixed": field_rms < 1e-3 and motion < 1e-3,
                    "map_stationary": motion < 1e-8,
                    "moving_recurrent": motion >= 1e-3 and recurrence < 3e-2,
                    "positive_lyapunov": lyapunov > 5e-3,
                    "sample_stride_layers": int(metrics["sample_stride_layers"][local]),
                    "burn_layers": int(metrics["burn_layers"][local]),
                }
                records[global_index]["trace"].append(point)
        for local, global_index in enumerate(indices):
            records[global_index]["final_angle"] = angles[local, 0].tolist()
    source["settings"]["ratios"].extend(ratios)
    source["settings"]["extension"] = {
        "ratios": ratios,
        "burn_time_normalized": burn_time,
        "sample_count": sample_count,
        "sample_spacing_normalized": sample_spacing,
        "lyapunov_time_normalized": lyapunov_time,
        "seed": seed,
        "continuation_noise": continuation_noise,
    }
    source["final_counts"] = final_counts(records)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--ratios", default=".001953125,.0009765625")
    parser.add_argument("--burn-time", type=float, default=600.0)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--sample-spacing", type=float, default=0.25)
    parser.add_argument("--lyapunov-time", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=260814401)
    parser.add_argument("--continuation-noise", type=float, default=1e-7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        [float(value) for value in args.ratios.split(",")],
        args.burn_time,
        args.sample_count,
        args.sample_spacing,
        args.lyapunov_time,
        args.seed,
        args.continuation_noise,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["final_counts"], indent=2))


if __name__ == "__main__":
    main()
