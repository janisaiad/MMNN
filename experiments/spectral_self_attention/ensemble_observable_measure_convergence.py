"""Compare ensemble observable measures of fine layers and the ODE flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.continuous_ode_robustness import (
    select_record,
)
from experiments.spectral_self_attention.large_scale_cycle_census import map_angles
from experiments.spectral_self_attention.observable_measure_convergence import (
    observables,
    pooled_wasserstein,
    sliced_wasserstein,
)
from experiments.spectral_self_attention.small_step_continuation import stack_models


def map_history(
    initial: np.ndarray,
    models: dict[str, np.ndarray],
    original_steps: np.ndarray,
    ratio: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
) -> np.ndarray:
    models["step_size"] = original_steps * ratio
    angles = initial.copy()
    for _ in range(int(np.ceil(burn_time / ratio))):
        angles = map_angles(angles, models)
    stride = max(1, int(np.ceil(sample_spacing / ratio)))
    saved = []
    for _ in range(sample_count):
        for _ in range(stride):
            angles = map_angles(angles, models)
        saved.append(angles[:, 0].copy())
    return np.asarray(saved)


def ode_history(
    initial: np.ndarray,
    models: dict[str, np.ndarray],
    original_steps: np.ndarray,
    dt: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
) -> np.ndarray:
    angles = initial.copy()
    for _ in range(int(np.ceil(burn_time / dt))):
        angles = rk4_step(angles, models, original_steps, dt)
    stride = max(1, int(np.ceil(sample_spacing / dt)))
    saved = []
    for _ in range(sample_count):
        for _ in range(stride):
            angles = rk4_step(angles, models, original_steps, dt)
        saved.append(angles[:, 0].copy())
    return np.asarray(saved)


def flatten_observables(
    history: np.ndarray, score: np.ndarray, beta: float
) -> dict[str, np.ndarray]:
    return observables(history.reshape(-1, history.shape[-1]), score, beta)


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    starts: int,
    ratios: list[float],
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    ode_dt: float,
    projections: int,
    seed: int,
) -> dict[str, object]:
    if starts < 2 or starts % 2:
        raise ValueError("starts must be an even integer of at least two")
    record = select_record(payload, n_tokens, source_model_index)
    models = stack_models([record] * starts)
    original_steps = models["step_size"].astype(float).copy()
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-np.pi, np.pi, size=(starts, 1, n_tokens))
    reference_history = ode_history(
        initial,
        models,
        original_steps,
        ode_dt,
        burn_time,
        sample_count,
        sample_spacing,
    )
    score = models["score"][0]
    beta = float(models["beta"][0])
    reference = flatten_observables(reference_history, score, beta)
    first_half = flatten_observables(reference_history[:, : starts // 2], score, beta)
    second_half = flatten_observables(reference_history[:, starts // 2 :], score, beta)
    dimension = reference["joint"].shape[-1]
    directions = rng.normal(size=(projections, dimension))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    baseline = {
        name: pooled_wasserstein(first_half[name], second_half[name])
        for name in ("gram", "score_kernel", "attention_weights", "coherence")
    }
    baseline["joint_sliced_standardized"] = sliced_wasserstein(
        first_half["joint"],
        second_half["joint"],
        reference["joint"],
        directions,
    )
    rows = []
    for ratio in ratios:
        current_history = map_history(
            initial,
            models,
            original_steps,
            ratio,
            burn_time,
            sample_count,
            sample_spacing,
        )
        current = flatten_observables(current_history, score, beta)
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
                "distance_over_reference_ensemble_split": {
                    name: value / max(baseline[name], 1e-12)
                    for name, value in distances.items()
                },
            }
        )
    models["step_size"] = original_steps
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "uniform_random_starts": starts,
            "ratios": ratios,
            "burn_time_normalized": burn_time,
            "sample_count_per_start": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "ode_dt": ode_dt,
            "sliced_wasserstein_projections": projections,
            "seed": seed,
        },
        "reference_ensemble_split_baseline": baseline,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--starts", type=int, default=128)
    parser.add_argument("--ratios", default=".015625,.00390625,.0009765625")
    parser.add_argument("--burn-time", type=float, default=1000.0)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-spacing", type=float, default=0.5)
    parser.add_argument("--ode-dt", type=float, default=0.02)
    parser.add_argument("--projections", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260816601)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        args.starts,
        [float(value) for value in args.ratios.split(",")],
        args.burn_time,
        args.sample_count,
        args.sample_spacing,
        args.ode_dt,
        args.projections,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["records"], indent=2))


if __name__ == "__main__":
    main()
