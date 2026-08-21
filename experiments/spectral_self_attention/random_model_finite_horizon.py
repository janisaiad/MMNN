"""Finite-horizon layer-to-flow convergence on fresh, unscreened random models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.large_scale_cycle_census import (
    draw_models,
    map_angles,
    wrap,
)
from experiments.spectral_self_attention.random_state_finite_horizon import (
    feature_errors,
    summarize,
)


MetricChunks = dict[str, list[np.ndarray]]


def fitted_orders(
    ratios: list[float], chunks: dict[float, MetricChunks]
) -> dict[str, float | None]:
    """Fit median error slopes for one aggregate population."""
    result: dict[str, float | None] = {}
    for name in ("angle", "gram", "score_kernel"):
        medians = np.asarray(
            [
                np.median(np.concatenate(chunks[ratio][name]))
                for ratio in ratios
            ]
        )
        result[name] = (
            None
            if np.any(medians <= 0.0)
            else float(np.polyfit(np.log(ratios), np.log(medians), 1)[0])
        )
    return result


def aggregate_summary(
    ratios: list[float], chunks: dict[float, MetricChunks]
) -> dict[str, object]:
    deepest = ratios[-1]
    return {
        "median_error_orders": fitted_orders(ratios, chunks),
        "deepest_step": {
            "step_ratio": deepest,
            **{
                f"{name}_error": summarize(
                    np.concatenate(chunks[deepest][name])
                )
                for name in ("angle", "gram", "score_kernel")
            },
        },
    }


def empty_chunks(ratios: list[float]) -> dict[float, MetricChunks]:
    return {
        ratio: {name: [] for name in ("angle", "gram", "score_kernel")}
        for ratio in ratios
    }


def run(
    families: list[int],
    token_counts: list[int],
    models_per_cell: int,
    batch_size: int,
    ratios: list[float],
    horizon: float,
    reference_dt: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    aggregate = empty_chunks(ratios)
    by_family = {family: empty_chunks(ratios) for family in families}
    by_tokens = {n_tokens: empty_chunks(ratios) for n_tokens in token_counts}
    rows: list[dict[str, object]] = []

    for family in families:
        for n_tokens in token_counts:
            cell = empty_chunks(ratios)
            for first in range(0, models_per_cell, batch_size):
                count = min(batch_size, models_per_cell - first)
                models = draw_models(rng, family, count)
                original_steps = models["step_size"].astype(float).copy()
                initial = rng.uniform(-np.pi, np.pi, size=(count, 1, n_tokens))

                reference = initial.copy()
                reference_steps = int(np.ceil(horizon / reference_dt))
                effective_dt = horizon / reference_steps
                for _ in range(reference_steps):
                    reference = rk4_step(
                        reference, models, original_steps, effective_dt
                    )

                for ratio in ratios:
                    layers = int(round(horizon / ratio))
                    if not np.isclose(
                        layers * ratio, horizon, atol=1e-12, rtol=0.0
                    ):
                        raise ValueError(
                            "horizon must be an integer multiple of every ratio"
                        )
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
                    values = {
                        "angle": angle_error,
                        "gram": gram_error,
                        "score_kernel": kernel_error,
                    }
                    for name, errors in values.items():
                        cell[ratio][name].append(errors)
                        aggregate[ratio][name].append(errors)
                        by_family[family][ratio][name].append(errors)
                        by_tokens[n_tokens][ratio][name].append(errors)
                models["step_size"] = original_steps

            for ratio in ratios:
                rows.append(
                    {
                        "family": family,
                        "n_tokens": n_tokens,
                        "step_ratio": ratio,
                        "models": models_per_cell,
                        **{
                            f"{name}_error": summarize(
                                np.concatenate(cell[ratio][name])
                            )
                            for name in ("angle", "gram", "score_kernel")
                        },
                    }
                )

    return {
        "settings": {
            "families": families,
            "token_counts": token_counts,
            "models_per_cell": models_per_cell,
            "batch_size": batch_size,
            "ratios": ratios,
            "horizon_normalized": horizon,
            "reference_rk4_dt": reference_dt,
            "seed": seed,
        },
        "summary": {
            "models": len(families) * len(token_counts) * models_per_cell,
            **aggregate_summary(ratios, aggregate),
            "by_family": {
                str(family): {
                    "models": len(token_counts) * models_per_cell,
                    **aggregate_summary(ratios, by_family[family]),
                }
                for family in families
            },
            "by_token_count": {
                str(n_tokens): {
                    "models": len(families) * models_per_cell,
                    **aggregate_summary(ratios, by_tokens[n_tokens]),
                }
                for n_tokens in token_counts
            },
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="1,2,3,4")
    parser.add_argument("--token-counts", default="1,2,3,4,8,16")
    parser.add_argument("--models-per-cell", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ratios", default=".015625,.00390625,.0009765625")
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--reference-dt", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=260817201)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.models_per_cell < 1 or args.batch_size < 1:
        parser.error("model and batch counts must be positive")
    result = run(
        [int(value) for value in args.families.split(",")],
        [int(value) for value in args.token_counts.split(",")],
        args.models_per_cell,
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
