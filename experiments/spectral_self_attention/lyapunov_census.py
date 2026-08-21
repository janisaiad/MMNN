"""Classify non-fixed trajectories with a largest-Lyapunov diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import (
    FAMILY_NAMES,
    classify_history,
    draw_models,
    map_angles,
    wrap,
)


def largest_lyapunov(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    rng: np.random.Generator,
    steps: int,
    epsilon: float = 1e-7,
) -> np.ndarray:
    direction = rng.normal(size=angles.shape)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    perturbed = wrap(angles + epsilon * direction)
    log_growth = np.zeros(angles.shape[:2])
    for _ in range(steps):
        angles = map_angles(angles, models)
        perturbed = map_angles(perturbed, models)
        delta = wrap(perturbed - angles)
        norm = np.linalg.norm(delta, axis=-1)
        safe_norm = np.maximum(norm, 1e-15)
        log_growth += np.log(safe_norm / epsilon)
        direction = delta / safe_norm[..., None]
        collapsed = norm < 1e-14
        if np.any(collapsed):
            replacement = rng.normal(size=(int(collapsed.sum()), angles.shape[-1]))
            replacement /= np.maximum(
                np.linalg.norm(replacement, axis=-1, keepdims=True), 1e-12
            )
            direction[collapsed] = replacement
        perturbed = wrap(angles + epsilon * direction)
    return log_growth / steps


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "positive_count": int(np.sum(array > 0.002)),
        "near_zero_count": int(np.sum(np.abs(array) <= 0.002)),
        "negative_count": int(np.sum(array < -0.002)),
    }


def run(
    family: int,
    models_per_token_count: int,
    batch_models: int,
    basins: int,
    burn_steps: int,
    lyapunov_steps: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    categories: dict[str, list[float]] = {}
    token_categories: dict[str, dict[str, list[float]]] = {}
    positive_examples: list[dict[str, object]] = []
    for n_tokens in (1, 2, 3, 4):
        by_category = token_categories.setdefault(str(n_tokens), {})
        for start in range(0, models_per_token_count, batch_models):
            count = min(batch_models, models_per_token_count - start)
            models = draw_models(rng, family, count)
            angles = rng.uniform(-np.pi, np.pi, size=(count, basins, n_tokens))
            for _ in range(burn_steps):
                angles = map_angles(angles, models)
            saved = []
            for _ in range(36):
                angles = map_angles(angles, models)
                saved.append(angles.copy())
            periods, _, rotations = classify_history(np.stack(saved))
            exponents = largest_lyapunov(
                angles, models, rng, steps=lyapunov_steps
            )
            labels = np.full(periods.shape, "unresolved", dtype="U12")
            labels[rotations] = "rotation"
            for period in range(1, 13):
                labels[periods == period] = f"p{period}"
            for label in np.unique(labels):
                values = exponents[labels == label].tolist()
                categories.setdefault(label, []).extend(values)
                by_category.setdefault(label, []).extend(values)
            positive = (exponents > 0.01) & (periods == 0) & ~rotations
            for model_index, basin_index in np.argwhere(positive)[:4]:
                if len(positive_examples) >= 12:
                    break
                positive_examples.append(
                    {
                        "n_tokens": n_tokens,
                        "exponent_per_layer": float(
                            exponents[model_index, basin_index]
                        ),
                        "angles": angles[model_index, basin_index].tolist(),
                        "model": {
                            key: (
                                float(value[model_index])
                                if value.ndim == 1
                                else value[model_index].tolist()
                            )
                            for key, value in models.items()
                        },
                    }
                )
    return {
        "family": family,
        "family_name": FAMILY_NAMES[family],
        "settings": {
            "models_per_token_count": models_per_token_count,
            "basins": basins,
            "burn_steps": burn_steps,
            "lyapunov_steps": lyapunov_steps,
            "seed": seed,
        },
        "by_attractor": {key: summarize(values) for key, values in categories.items()},
        "by_tokens": {
            tokens: {key: summarize(values) for key, values in groups.items()}
            for tokens, groups in token_categories.items()
        },
        "positive_examples": positive_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--models-per-token-count", type=int, default=512)
    parser.add_argument("--batch-models", type=int, default=32)
    parser.add_argument("--basins", type=int, default=16)
    parser.add_argument("--burn-steps", type=int, default=1000)
    parser.add_argument("--lyapunov-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=260813071)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        family=args.family,
        models_per_token_count=args.models_per_token_count,
        batch_models=args.batch_models,
        basins=args.basins,
        burn_steps=args.burn_steps,
        lyapunov_steps=args.lyapunov_steps,
        seed=args.seed + args.family,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"family": result["family_name"], "by_attractor": result["by_attractor"]}, indent=2))


if __name__ == "__main__":
    main()
