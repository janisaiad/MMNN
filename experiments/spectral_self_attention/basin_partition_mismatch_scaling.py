"""Measure how asymptotic token-cluster basins move with the layer step."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.continuous_ode_robustness import select_record
from experiments.spectral_self_attention.large_scale_cycle_census import map_angles, wrap
from experiments.spectral_self_attention.small_step_continuation import (
    continuous_angular_field,
    stack_models,
)


def partition_codes(angles: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    """Encode the token-index partition induced by coincident circle angles."""
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 2:
        raise ValueError("angles must have shape (batch, tokens)")
    same = np.abs(wrap(angles[:, :, None] - angles[:, None, :])) < tolerance
    for middle in range(angles.shape[1]):
        same |= same[:, :, middle, None] & same[:, None, middle, :]
    codes = np.zeros(angles.shape[0], dtype=np.uint64)
    bit = 0
    for first in range(angles.shape[1]):
        for second in range(first + 1, angles.shape[1]):
            codes |= same[:, first, second].astype(np.uint64) << np.uint64(bit)
            bit += 1
    return codes


def decode_partition(code: int, n_tokens: int) -> list[list[int]]:
    """Decode a partition code into one-indexed token groups."""
    parent = list(range(n_tokens))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    bit = 0
    for first in range(n_tokens):
        for second in range(first + 1, n_tokens):
            if code & (1 << bit):
                first_root = root(first)
                second_root = root(second)
                if first_root != second_root:
                    parent[second_root] = first_root
            bit += 1
    groups: dict[int, list[int]] = {}
    for index in range(n_tokens):
        groups.setdefault(root(index), []).append(index + 1)
    return sorted(groups.values(), key=lambda group: group[0])


def signature_codes(codes: np.ndarray, n_tokens: int) -> np.ndarray:
    """Encode only sorted cluster sizes, discarding token identities."""
    cache: dict[int, int] = {}
    output = np.empty(len(codes), dtype=np.uint32)
    for index, raw_code in enumerate(codes):
        code = int(raw_code)
        if code not in cache:
            sizes = sorted(
                (len(group) for group in decode_partition(code, n_tokens)),
                reverse=True,
            )
            signature = 0
            for size in sizes:
                signature = signature * (n_tokens + 1) + size
            cache[code] = signature
        output[index] = cache[code]
    return output


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q90": float(np.quantile(values, 0.9)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def distribution_rows(
    counter: Counter[int], total: int, n_tokens: int
) -> list[dict[str, object]]:
    return [
        {
            "code": int(code),
            "token_groups": decode_partition(int(code), n_tokens),
            "count": int(count),
            "fraction": count / total,
        }
        for code, count in counter.most_common()
    ]


def total_variation(first: Counter[int], second: Counter[int], total: int) -> float:
    return 0.5 * sum(
        abs(first.get(key, 0) - second.get(key, 0)) / total
        for key in first.keys() | second.keys()
    )


def fitted_order(ratios: list[float], values: list[float]) -> float | None:
    kept = [(ratio, value) for ratio, value in zip(ratios, values) if value > 0]
    if len(kept) < 2:
        return None
    return float(
        np.polyfit(
            np.log([ratio for ratio, _ in kept]),
            np.log([value for _, value in kept]),
            1,
        )[0]
    )


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    starts: int,
    batch_size: int,
    ratios: list[float],
    horizon: float,
    reference_dt: float,
    tolerance: float,
    seed: int,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    rng = np.random.default_rng(seed)
    reference_counts: Counter[int] = Counter()
    discrete_counts = {ratio: Counter() for ratio in ratios}
    mismatch_counts = {ratio: 0 for ratio in ratios}
    shape_mismatch_counts = {ratio: 0 for ratio in ratios}
    error_chunks = {ratio: [] for ratio in ratios}
    reference_field_chunks: list[np.ndarray] = []

    for first in range(0, starts, batch_size):
        count = min(batch_size, starts - first)
        initial = rng.uniform(-np.pi, np.pi, size=(count, 1, n_tokens))
        models = stack_models([record] * count)
        original_step = models["step_size"].astype(float).copy()

        reference = initial.copy()
        reference_steps = int(np.ceil(horizon / reference_dt))
        effective_dt = horizon / reference_steps
        for _ in range(reference_steps):
            reference = rk4_step(reference, models, original_step, effective_dt)
        reference_angles = reference[:, 0]
        reference_codes = partition_codes(reference_angles, tolerance)
        reference_signatures = signature_codes(reference_codes, n_tokens)
        reference_counts.update(map(int, reference_codes))
        reference_field = continuous_angular_field(reference, models)
        reference_field_chunks.append(
            np.sqrt(
                np.mean(
                    (original_step[:, None, None] * reference_field) ** 2,
                    axis=(1, 2),
                )
            )
        )

        for ratio in ratios:
            layers = int(round(horizon / ratio))
            if not np.isclose(layers * ratio, horizon, atol=1e-12, rtol=0.0):
                raise ValueError("horizon must be divisible by every ratio")
            models["step_size"] = original_step * ratio
            discrete = initial.copy()
            for _ in range(layers):
                discrete = map_angles(discrete, models)
            discrete_angles = discrete[:, 0]
            discrete_codes = partition_codes(discrete_angles, tolerance)
            discrete_signatures = signature_codes(discrete_codes, n_tokens)
            discrete_counts[ratio].update(map(int, discrete_codes))
            mismatch_counts[ratio] += int(np.sum(discrete_codes != reference_codes))
            shape_mismatch_counts[ratio] += int(
                np.sum(discrete_signatures != reference_signatures)
            )
            error_chunks[ratio].append(
                np.sqrt(np.mean(wrap(discrete_angles - reference_angles) ** 2, axis=1))
            )

    rows = []
    mismatch_fractions = []
    shape_mismatch_fractions = []
    distribution_distances = []
    for ratio in ratios:
        mismatch_fraction = mismatch_counts[ratio] / starts
        shape_mismatch_fraction = shape_mismatch_counts[ratio] / starts
        tv = total_variation(reference_counts, discrete_counts[ratio], starts)
        mismatch_fractions.append(mismatch_fraction)
        shape_mismatch_fractions.append(shape_mismatch_fraction)
        distribution_distances.append(tv)
        rows.append(
            {
                "step_ratio": ratio,
                "step_denominator": round(1.0 / ratio),
                "partition_mismatch_count": mismatch_counts[ratio],
                "partition_mismatch_fraction": mismatch_fraction,
                "partition_mismatch_standard_error": float(
                    np.sqrt(mismatch_fraction * (1.0 - mismatch_fraction) / starts)
                ),
                "cluster_size_mismatch_count": shape_mismatch_counts[ratio],
                "cluster_size_mismatch_fraction": shape_mismatch_fraction,
                "partition_distribution_total_variation": tv,
                "angle_error": summarize(np.concatenate(error_chunks[ratio])),
                "partition_distribution": distribution_rows(
                    discrete_counts[ratio], starts, n_tokens
                ),
            }
        )

    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "uniform_random_starts": starts,
            "batch_size": batch_size,
            "ratios": ratios,
            "horizon_normalized": horizon,
            "reference_rk4_dt": reference_dt,
            "cluster_tolerance": tolerance,
            "seed": seed,
        },
        "summary": {
            "partition_mismatch_order": fitted_order(ratios, mismatch_fractions),
            "cluster_size_mismatch_order": fitted_order(
                ratios, shape_mismatch_fractions
            ),
            "partition_distribution_total_variation_order": fitted_order(
                ratios, distribution_distances
            ),
            "reference_endpoint_field_rms": summarize(
                np.concatenate(reference_field_chunks)
            ),
            "deepest_step": rows[-1],
        },
        "reference_partition_distribution": distribution_rows(
            reference_counts, starts, n_tokens
        ),
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--starts", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--ratios",
        default=".03125,.015625,.0078125,.00390625,.001953125,.0009765625",
    )
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--cluster-tolerance", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=260816801)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        args.starts,
        args.batch_size,
        [float(value) for value in args.ratios.split(",")],
        args.horizon,
        args.reference_dt,
        args.cluster_tolerance,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
