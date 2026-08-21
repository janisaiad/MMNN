"""Resolve a finite-step basin switch against the continuous reference flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.continuous_ode_robustness import (
    select_record,
)
from experiments.spectral_self_attention.large_scale_cycle_census import (
    map_angles,
    wrap,
)
from experiments.spectral_self_attention.small_step_continuation import (
    continuous_angular_field,
    stack_models,
)


def cluster_signature(angles: np.ndarray, tolerance: float = 1e-3) -> list[int]:
    count = angles.size
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(count):
        for second in range(first + 1, count):
            if abs(float(wrap(angles[first] - angles[second]))) < tolerance:
                union(first, second)
    sizes: dict[int, int] = {}
    for index in range(count):
        representative = root(index)
        sizes[representative] = sizes.get(representative, 0) + 1
    return sorted(sizes.values(), reverse=True)


def geometry_error(first: np.ndarray, second: np.ndarray) -> float:
    first_tokens = np.stack((np.cos(first), np.sin(first)), axis=-1)
    second_tokens = np.stack((np.cos(second), np.sin(second)), axis=-1)
    return float(
        np.sqrt(np.mean((first_tokens @ first_tokens.T - second_tokens @ second_tokens.T) ** 2))
    )


def field_rms(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    original_step: np.ndarray,
) -> float:
    field = continuous_angular_field(angles[None, None, :], models)[0, 0]
    return float(np.sqrt(np.mean((original_step[0] * field) ** 2)))


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    initial: np.ndarray,
    denominators: list[int],
    horizon: float,
    reference_dt: float,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    models = stack_models([record])
    original_step = models["step_size"].astype(float).copy()
    start = initial[None, None, :].copy()
    reference = start.copy()
    reference_steps = int(np.ceil(horizon / reference_dt))
    effective_dt = horizon / reference_steps
    for _ in range(reference_steps):
        reference = rk4_step(reference, models, original_step, effective_dt)
    reference_angles = reference[0, 0]
    rows = []
    for denominator in denominators:
        ratio = 1.0 / denominator
        layers = int(round(horizon / ratio))
        models["step_size"] = original_step * ratio
        discrete = start.copy()
        for _ in range(layers):
            discrete = map_angles(discrete, models)
        angles = discrete[0, 0]
        rows.append(
            {
                "step_denominator": denominator,
                "step_ratio": ratio,
                "layers": layers,
                "angle_rms_error": float(
                    np.sqrt(np.mean(wrap(angles - reference_angles) ** 2))
                ),
                "gram_rms_error": geometry_error(angles, reference_angles),
                "cluster_signature": cluster_signature(angles),
                "continuous_field_rms_at_endpoint": field_rms(
                    angles, models, original_step
                ),
                "final_angle": angles.tolist(),
            }
        )
    models["step_size"] = original_step
    reference_signature = cluster_signature(reference_angles)
    matching = [
        row["step_denominator"]
        for row in rows
        if row["cluster_signature"] == reference_signature
    ]
    mismatching = [
        row["step_denominator"]
        for row in rows
        if row["cluster_signature"] != reference_signature
    ]
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "horizon_normalized": horizon,
            "reference_rk4_dt": effective_dt,
            "step_denominators": denominators,
        },
        "initial_angle": initial.tolist(),
        "reference": {
            "cluster_signature": reference_signature,
            "continuous_field_rms_at_endpoint": field_rms(
                reference_angles, models, original_step
            ),
            "final_angle": reference_angles.tolist(),
        },
        "summary": {
            "largest_denominator_with_mismatching_basin": max(
                mismatching, default=None
            ),
            "smallest_denominator_with_matching_basin": min(
                matching, default=None
            ),
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--initial", type=float, nargs="+", required=True)
    parser.add_argument(
        "--denominators",
        default="64,128,256,512,768,896,960,992,1008,1024,1040,1056,1088,"
        "1152,1280,1536,1792,2048,4096,8192",
    )
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--reference-dt", type=float, default=0.001)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        np.asarray(args.initial, dtype=float),
        [int(value) for value in args.denominators.split(",")],
        args.horizon,
        args.reference_dt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
