"""Fine downward/upward step sweep for one selected attractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.small_step_continuation import (
    evaluate_ratio,
    stack_models,
)


def selected_record(
    payload: dict[str, object], n_tokens: int, source_model_index: int
) -> dict[str, object]:
    records = [
        record
        for record in payload["records"]
        if int(record["identity"]["n_tokens"]) == n_tokens
        and int(record["identity"]["source_model_index"]) == source_model_index
    ]
    if len(records) != 1:
        raise ValueError(f"expected one record, found {len(records)}")
    return records[0]


def point(ratio: float, metrics: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "ratio": ratio,
        "motion_per_normalized_time": float(metrics["motion_per_normalized_time"][0]),
        "continuous_field_rms": float(metrics["continuous_field_rms"][0]),
        "recurrence_error": float(metrics["recurrence_error"][0]),
        "return_time_normalized": float(metrics["return_time_normalized"][0]),
        "gram_variation": float(metrics["gram_variation"][0]),
        "gram_recurrence_error": float(metrics["gram_recurrence_error"][0]),
        "score_kernel_variation": float(metrics["score_kernel_variation"][0]),
        "score_kernel_recurrence_error": float(
            metrics["score_kernel_recurrence_error"][0]
        ),
        "mean_coherence": float(metrics["mean_coherence"][0]),
        "lyapunov_per_normalized_time": float(
            metrics["lyapunov_per_normalized_time"][0]
        ),
        "absolute_local_error": float(metrics["absolute_local_error"][0]),
    }


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    ratios: list[float],
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    continuation_noise: float,
    seed: int,
) -> dict[str, object]:
    record = selected_record(payload, n_tokens, source_model_index)
    models = stack_models([record])
    original_steps = models["step_size"].astype(float).copy()
    rng = np.random.default_rng(seed)
    initial = np.asarray(record["initial_angle"], dtype=float)[None, None, :]

    def sweep(start: np.ndarray, sequence: list[float]) -> tuple[np.ndarray, list[dict]]:
        angles = start.copy()
        output = []
        for ratio in sequence:
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
            output.append(point(ratio, metrics))
        return angles, output

    bottom, downward = sweep(initial, ratios)
    _, upward = sweep(bottom, list(reversed(ratios)))
    return {
        "identity": record["identity"],
        "model": record["model"],
        "settings": {
            "ratios": ratios,
            "burn_time_normalized": burn_time,
            "sample_count": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "lyapunov_time_normalized": lyapunov_time,
            "continuation_noise": continuation_noise,
            "seed": seed,
        },
        "downward": downward,
        "upward": upward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--ratio-min", type=float, default=0.02)
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument("--burn-time", type=float, default=600.0)
    parser.add_argument("--sample-count", type=int, default=160)
    parser.add_argument("--sample-spacing", type=float, default=0.2)
    parser.add_argument("--lyapunov-time", type=float, default=200.0)
    parser.add_argument("--continuation-noise", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=260815201)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ratios = np.geomspace(1.0, args.ratio_min, args.points).tolist()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        ratios,
        args.burn_time,
        args.sample_count,
        args.sample_spacing,
        args.lyapunov_time,
        args.continuation_noise,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["identity"], indent=2))


if __name__ == "__main__":
    main()
