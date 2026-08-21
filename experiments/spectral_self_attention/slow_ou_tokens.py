"""General slow-OU experiment for symmetric multi-head token dynamics on S^2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def symmetric_noise(
    rng: np.random.Generator,
    shape_prefix: tuple[int, ...],
    dimension: int,
) -> np.ndarray:
    noise = np.zeros(shape_prefix + (dimension, dimension))
    for i in range(dimension):
        noise[..., i, i] = rng.normal(size=shape_prefix)
        for j in range(i + 1, dimension):
            entry = rng.normal(size=shape_prefix)
            noise[..., i, j] = entry
            noise[..., j, i] = entry
    return noise


def token_field(tokens: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    scores = np.einsum("rid,rhde,rje->rhij", tokens, matrices, tokens)
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    values = np.einsum("rhde,rje->rhjd", matrices, tokens)
    output = np.mean(np.einsum("rhij,rhjd->rhid", weights, values), axis=1)
    radial = np.sum(tokens * output, axis=-1, keepdims=True)
    return output - radial * tokens


def observables(
    tokens: np.ndarray,
    matrices: np.ndarray,
    field: np.ndarray,
    target_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(tokens, axis=1)
    concentration = np.linalg.norm(mean, axis=-1)
    mean_direction = normalize(mean + 1e-12 * target_axis)
    average_matrix = np.mean(matrices, axis=1)
    _, eigenvectors = np.linalg.eigh(average_matrix)
    instantaneous_axis = eigenvectors[..., -1]
    tracking = np.abs(np.sum(mean_direction * instantaneous_axis, axis=-1))
    target_alignment = np.abs(np.sum(mean_direction * target_axis, axis=-1))
    speed = np.sqrt(np.mean(np.sum(field * field, axis=-1), axis=-1))
    return concentration, tracking, target_alignment, speed


def simulate(
    *,
    seed: int,
    rate: float | None,
    trials: int,
    tokens_count: int,
    heads: int,
    sigma: float,
    dt: float,
    steps: int,
    save_every: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    dimension = 3
    target = np.diag([1.2, 1.0, 0.2])
    target_axis = np.array([1.0, 0.0, 0.0])
    tokens = normalize(rng.normal(size=(trials, tokens_count, dimension)))
    matrices = np.broadcast_to(target, (trials, heads, dimension, dimension)).copy()
    if rate is not None:
        matrices += sigma * symmetric_noise(rng, (trials, heads), dimension)

    times: list[float] = []
    concentration_trace: list[float] = []
    tracking_trace: list[float] = []
    target_trace: list[float] = []
    speed_trace: list[float] = []
    direction_trace: list[np.ndarray] = []
    concentration_trials: list[np.ndarray] = []
    tracking_trials: list[np.ndarray] = []
    target_trials: list[np.ndarray] = []
    speed_trials: list[np.ndarray] = []

    for step in range(steps + 1):
        field = token_field(tokens, matrices)
        if step % save_every == 0 or step == steps:
            concentration, tracking, target_alignment, speed = observables(
                tokens, matrices, field, target_axis
            )
            direction = normalize(np.mean(tokens, axis=1) + 1e-12 * target_axis)
            times.append(step * dt)
            concentration_trace.append(float(np.mean(concentration)))
            tracking_trace.append(float(np.mean(tracking)))
            target_trace.append(float(np.mean(target_alignment)))
            speed_trace.append(float(np.mean(speed)))
            direction_trace.append(direction)
            concentration_trials.append(concentration)
            tracking_trials.append(tracking)
            target_trials.append(target_alignment)
            speed_trials.append(speed)
        if step == steps:
            break

        tokens = normalize(tokens + dt * field)
        if rate is not None:
            decay = np.exp(-rate * dt)
            innovation = sigma * np.sqrt(max(0.0, 1.0 - decay**2))
            matrices = (
                target
                + decay * (matrices - target)
                + innovation * symmetric_noise(rng, (trials, heads), dimension)
            )

    saved_directions = np.stack(direction_trace)
    direction_overlap = np.abs(
        np.sum(saved_directions[1:] * saved_directions[:-1], axis=-1)
    )
    angular_steps = np.arccos(np.clip(direction_overlap, -1.0, 1.0))
    late_start = len(times) // 2
    late_by_trial = {
        "concentration": np.mean(np.stack(concentration_trials)[late_start:], axis=0),
        "tracking": np.mean(np.stack(tracking_trials)[late_start:], axis=0),
        "target_alignment": np.mean(np.stack(target_trials)[late_start:], axis=0),
        "speed": np.mean(np.stack(speed_trials)[late_start:], axis=0),
    }
    standard_errors = {
        key: float(np.std(value, ddof=1) / np.sqrt(trials))
        for key, value in late_by_trial.items()
    }
    quantiles = {
        key: np.quantile(value, [0.1, 0.5, 0.9]).tolist()
        for key, value in late_by_trial.items()
    }
    return {
        "rate": rate,
        "correlation_time": None if rate in (None, 0.0) else 1.0 / rate,
        "late_concentration": float(np.mean(concentration_trace[late_start:])),
        "late_tracking": float(np.mean(tracking_trace[late_start:])),
        "late_target_alignment": float(np.mean(target_trace[late_start:])),
        "late_speed": float(np.mean(speed_trace[late_start:])),
        "late_standard_errors": standard_errors,
        "late_quantiles_10_50_90": quantiles,
        "mean_angular_motion_per_sample": float(np.mean(angular_steps)),
        "large_turn_fraction": float(np.mean(angular_steps > np.deg2rad(10.0))),
        "times": times,
        "concentration_trace": concentration_trace,
        "tracking_trace": tracking_trace,
        "target_alignment_trace": target_trace,
        "speed_trace": speed_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=260518872)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.65)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spectral_self_attention/slow_ou_tokens.json"),
    )
    args = parser.parse_args()
    fixed = simulate(
        seed=args.seed,
        rate=None,
        trials=args.trials,
        tokens_count=args.tokens,
        heads=args.heads,
        sigma=args.sigma,
        dt=args.dt,
        steps=args.steps,
        save_every=args.save_every,
    )
    rates = [
        simulate(
            # Coupling every rate to the same initial tokens, initial weights,
            # and Gaussian innovations isolates the effect of persistence.
            seed=args.seed,
            rate=rate,
            trials=args.trials,
            tokens_count=args.tokens,
            heads=args.heads,
            sigma=args.sigma,
            dt=args.dt,
            steps=args.steps,
            save_every=args.save_every,
        )
        for rate in args.rates
    ]
    result = {
        "settings": {
            "seed": args.seed,
            "trials": args.trials,
            "tokens": args.tokens,
            "heads": args.heads,
            "sigma": args.sigma,
            "dt": args.dt,
            "steps": args.steps,
            "target_eigenvalues": [1.2, 1.0, 0.2],
        },
        "fixed": fixed,
        "rates": rates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "fixed": {key: value for key, value in fixed.items() if "trace" not in key and key != "times"},
        "rates": [
            {key: value for key, value in run.items() if "trace" not in key and key != "times"}
            for run in rates
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
