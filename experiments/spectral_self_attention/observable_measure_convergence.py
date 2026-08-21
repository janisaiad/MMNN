"""Compare long-run observable measures of fine layers and the ODE limit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

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


def observables(
    history: np.ndarray, score: np.ndarray, beta: float
) -> dict[str, np.ndarray]:
    tokens = np.stack((np.cos(history), np.sin(history)), axis=-1)
    gram = np.einsum("tid,tjd->tij", tokens, tokens, optimize=True)
    kernel = np.einsum("tid,de,tje->tij", tokens, score, tokens, optimize=True)
    logits = beta * kernel
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(np.clip(logits, -80.0, 0.0))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    n_tokens = history.shape[-1]
    upper = np.triu_indices(n_tokens, k=1)
    gram_vector = gram[:, upper[0], upper[1]]
    coherence = np.abs(np.mean(np.exp(1j * history), axis=-1))[:, None]
    return {
        "gram": gram_vector,
        "score_kernel": kernel.reshape(history.shape[0], -1),
        "attention_weights": weights.reshape(history.shape[0], -1),
        "coherence": coherence,
        "joint": np.concatenate(
            (
                gram_vector,
                kernel.reshape(history.shape[0], -1),
                weights.reshape(history.shape[0], -1),
                coherence,
            ),
            axis=-1,
        ),
    }


def integrate_map_history(
    initial: np.ndarray,
    models: dict[str, np.ndarray],
    original_step: np.ndarray,
    ratio: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
) -> np.ndarray:
    models["step_size"] = original_step * ratio
    angles = initial.copy()
    for _ in range(int(np.ceil(burn_time / ratio))):
        angles = map_angles(angles, models)
    stride = max(1, int(np.ceil(sample_spacing / ratio)))
    history = []
    for _ in range(sample_count):
        for _ in range(stride):
            angles = map_angles(angles, models)
        history.append(angles[0, 0].copy())
    return np.asarray(history)


def integrate_ode_history(
    initial: np.ndarray,
    models: dict[str, np.ndarray],
    original_step: np.ndarray,
    dt: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
) -> np.ndarray:
    angles = initial.copy()
    for _ in range(int(np.ceil(burn_time / dt))):
        angles = rk4_step(angles, models, original_step, dt)
    stride = max(1, int(np.ceil(sample_spacing / dt)))
    history = []
    for _ in range(sample_count):
        for _ in range(stride):
            angles = rk4_step(angles, models, original_step, dt)
        history.append(angles[0, 0].copy())
    return np.asarray(history)


def pooled_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    return float(wasserstein_distance(left.ravel(), right.ravel()))


def sliced_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    reference: np.ndarray,
    directions: np.ndarray,
) -> float:
    center = np.mean(reference, axis=0)
    scale = np.maximum(np.std(reference, axis=0), 1e-3)
    left_scaled = (left - center) / scale
    right_scaled = (right - center) / scale
    return float(
        np.mean(
            [
                wasserstein_distance(left_scaled @ direction, right_scaled @ direction)
                for direction in directions
            ]
        )
    )


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    ratios: list[float],
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    ode_dt: float,
    initial_noise: float,
    projections: int,
    seed: int,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    models = stack_models([record])
    original_step = models["step_size"].astype(float).copy()
    rng = np.random.default_rng(seed)
    initial = np.asarray(record["final_angle"], dtype=float)[None, None, :]
    initial = wrap(initial + initial_noise * rng.normal(size=initial.shape))
    ode_history = integrate_ode_history(
        initial,
        models,
        original_step,
        ode_dt,
        burn_time,
        sample_count,
        sample_spacing,
    )
    reference = observables(
        ode_history, models["score"][0], float(models["beta"][0])
    )
    joint_dimension = reference["joint"].shape[-1]
    directions = rng.normal(size=(projections, joint_dimension))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    half = sample_count // 2
    baseline = {
        name: pooled_wasserstein(values[:half], values[-half:])
        for name, values in reference.items()
        if name != "joint"
    }
    baseline["joint_sliced_standardized"] = sliced_wasserstein(
        reference["joint"][:half],
        reference["joint"][-half:],
        reference["joint"],
        directions,
    )
    rows = []
    for ratio in ratios:
        map_history = integrate_map_history(
            initial,
            models,
            original_step,
            ratio,
            burn_time,
            sample_count,
            sample_spacing,
        )
        current = observables(
            map_history, models["score"][0], float(models["beta"][0])
        )
        distances = {
            name: pooled_wasserstein(current[name], reference[name])
            for name in ("gram", "score_kernel", "attention_weights", "coherence")
        }
        distances["joint_sliced_standardized"] = sliced_wasserstein(
            current["joint"], reference["joint"], reference["joint"], directions
        )
        rows.append(
            {
                "step_ratio": ratio,
                "effective_sample_spacing_normalized": max(
                    1, int(np.ceil(sample_spacing / ratio))
                )
                * ratio,
                "distances": distances,
                "distance_over_reference_split": {
                    name: value / max(baseline[name], 1e-12)
                    for name, value in distances.items()
                },
            }
        )
    models["step_size"] = original_step
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "ratios": ratios,
            "burn_time_normalized": burn_time,
            "sample_count": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "ode_dt": ode_dt,
            "initial_noise": initial_noise,
            "sliced_wasserstein_projections": projections,
            "seed": seed,
        },
        "reference_split_half_baseline": baseline,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument(
        "--ratios", default=".015625,.0078125,.00390625,.001953125,.0009765625"
    )
    parser.add_argument("--burn-time", type=float, default=1000.0)
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--sample-spacing", type=float, default=0.2)
    parser.add_argument("--ode-dt", type=float, default=0.02)
    parser.add_argument("--initial-noise", type=float, default=1e-3)
    parser.add_argument("--projections", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260816401)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        [float(value) for value in args.ratios.split(",")],
        args.burn_time,
        args.sample_count,
        args.sample_spacing,
        args.ode_dt,
        args.initial_noise,
        args.projections,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["records"], indent=2))


if __name__ == "__main__":
    main()
