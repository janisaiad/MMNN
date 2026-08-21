"""Finite-horizon layer-to-flow convergence from uniform random token states."""

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
    stack_models,
)


def feature_errors(
    discrete: np.ndarray,
    reference: np.ndarray,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    discrete_tokens = np.stack((np.cos(discrete), np.sin(discrete)), axis=-1)
    reference_tokens = np.stack((np.cos(reference), np.sin(reference)), axis=-1)
    discrete_gram = np.einsum(
        "mbid,mbjd->mbij", discrete_tokens, discrete_tokens, optimize=True
    )
    reference_gram = np.einsum(
        "mbid,mbjd->mbij", reference_tokens, reference_tokens, optimize=True
    )
    discrete_kernel = np.einsum(
        "mbid,mde,mbje->mbij",
        discrete_tokens,
        score,
        discrete_tokens,
        optimize=True,
    )
    reference_kernel = np.einsum(
        "mbid,mde,mbje->mbij",
        reference_tokens,
        score,
        reference_tokens,
        optimize=True,
    )
    gram_error = np.sqrt(
        np.mean((discrete_gram - reference_gram)[:, 0] ** 2, axis=(1, 2))
    )
    kernel_error = np.sqrt(
        np.mean(
            (discrete_kernel - reference_kernel)[:, 0] ** 2,
            axis=(1, 2),
        )
    )
    return gram_error, kernel_error


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q90": float(np.quantile(values, 0.9)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    starts: int,
    batch_size: int,
    ratios: list[float],
    horizon: float,
    reference_dt: float,
    seed: int,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    rng = np.random.default_rng(seed)
    error_lists = {
        ratio: {"angle": [], "gram": [], "score_kernel": []}
        for ratio in ratios
    }
    worst = {ratio: (-1.0, -1, []) for ratio in ratios}
    for first in range(0, starts, batch_size):
        count = min(batch_size, starts - first)
        initial = rng.uniform(-np.pi, np.pi, size=(count, 1, n_tokens))
        models = stack_models([record] * count)
        original_steps = models["step_size"].astype(float).copy()
        reference = initial.copy()
        reference_steps = int(np.ceil(horizon / reference_dt))
        effective_dt = horizon / reference_steps
        for _ in range(reference_steps):
            reference = rk4_step(reference, models, original_steps, effective_dt)
        for ratio in ratios:
            layers = int(round(horizon / ratio))
            if not np.isclose(layers * ratio, horizon, atol=1e-12, rtol=0.0):
                raise ValueError("horizon must be an integer multiple of every ratio")
            models["step_size"] = original_steps * ratio
            discrete = initial.copy()
            for _ in range(layers):
                discrete = map_angles(discrete, models)
            angle_error = np.sqrt(
                np.mean(wrap(discrete - reference)[:, 0] ** 2, axis=-1)
            )
            gram_error, kernel_error = feature_errors(
                discrete, reference, models["score"]
            )
            error_lists[ratio]["angle"].append(angle_error)
            error_lists[ratio]["gram"].append(gram_error)
            error_lists[ratio]["score_kernel"].append(kernel_error)
            local = int(np.argmax(angle_error))
            if float(angle_error[local]) > worst[ratio][0]:
                worst[ratio] = (
                    float(angle_error[local]),
                    first + local,
                    initial[local, 0].tolist(),
                )
        models["step_size"] = original_steps

    rows = []
    medians = {name: [] for name in ("angle", "gram", "score_kernel")}
    for ratio in ratios:
        row: dict[str, object] = {
            "step_ratio": ratio,
            "starts": starts,
            "worst_angle_start_index": worst[ratio][1],
            "worst_angle_initial_state": worst[ratio][2],
        }
        for name, chunks in error_lists[ratio].items():
            values = np.concatenate(chunks)
            row[f"{name}_error"] = summarize(values)
            medians[name].append(float(np.median(values)))
        rows.append(row)
    orders = {
        name: float(np.polyfit(np.log(ratios), np.log(values), 1)[0])
        for name, values in medians.items()
    }
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "uniform_random_starts": starts,
            "batch_size": batch_size,
            "ratios": ratios,
            "horizon_normalized": horizon,
            "reference_rk4_dt": reference_dt,
            "seed": seed,
        },
        "summary": {
            "median_error_orders": orders,
            "deepest_step": rows[-1],
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--starts", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--ratios", default=".015625,.00390625,.0009765625"
    )
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--reference-dt", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=260816301)
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
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
