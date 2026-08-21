"""Save a direct continuous-time trajectory for a selected attractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import (
    normalized_field,
    rk4_step,
)
from experiments.spectral_self_attention.large_scale_cycle_census import wrap
from experiments.spectral_self_attention.small_step_continuation import stack_models


def select_record(
    payload: dict[str, object], n_tokens: int, source_model_index: int
) -> dict[str, object]:
    selected = [
        record
        for record in payload["records"]
        if int(record["identity"]["n_tokens"]) == n_tokens
        and int(record["identity"]["source_model_index"]) == source_model_index
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one record, found {len(selected)}")
    return selected[0]


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    dt: float,
    burn_time: float,
    duration: float,
    sample_spacing: float,
    initial_noise: float,
    seed: int,
    post_noise_time: float = 200.0,
    noise_relaxations: int = 2,
    anti_lock_noise: float = 0.0,
    anti_lock_interval: float = 100.0,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    models = stack_models([record])
    original_steps = models["step_size"].astype(float).copy()
    rng = np.random.default_rng(seed)
    angles = np.asarray(record["initial_angle"], dtype=float)[None, None, :]
    angles = wrap(angles + initial_noise * rng.normal(size=angles.shape))
    for _ in range(int(np.ceil(burn_time / dt))):
        angles = rk4_step(angles, models, original_steps, dt)
    if initial_noise > 0.0 and post_noise_time > 0.0:
        for _ in range(noise_relaxations):
            angles = wrap(angles + initial_noise * rng.normal(size=angles.shape))
            for _ in range(int(np.ceil(post_noise_time / dt))):
                angles = rk4_step(angles, models, original_steps, dt)
    stride = max(1, int(np.ceil(sample_spacing / dt)))
    samples = int(np.ceil(duration / (stride * dt)))
    anti_lock_stride = max(1, int(np.ceil(anti_lock_interval / dt)))
    angle_history = []
    speed_history = []
    elapsed_steps = 0
    for _ in range(samples):
        for _ in range(stride):
            if anti_lock_noise > 0.0 and elapsed_steps % anti_lock_stride == 0:
                angles = wrap(
                    angles + anti_lock_noise * rng.normal(size=angles.shape)
                )
            angles = rk4_step(angles, models, original_steps, dt)
            elapsed_steps += 1
        angle_history.append(angles[0, 0].copy())
        speed_history.append(normalized_field(angles, models, original_steps)[0, 0])
    history = np.asarray(angle_history)
    speeds = np.asarray(speed_history)
    tokens = np.stack((np.cos(history), np.sin(history)), axis=-1)
    gram = np.einsum("tid,tjd->tij", tokens, tokens, optimize=True)
    score_kernel = np.einsum(
        "tid,de,tje->tij", tokens, models["score"][0], tokens, optimize=True
    )
    logits = models["beta"][0] * score_kernel
    logits -= np.max(logits, axis=-1, keepdims=True)
    attention_weights = np.exp(np.clip(logits, -80.0, 0.0))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    relative_angles = wrap(history[:, 1:] - history[:, :1])
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "dt": dt,
            "burn_time_normalized": burn_time,
            "duration_normalized": duration,
            "sample_spacing_normalized": stride * dt,
            "initial_noise": initial_noise,
            "post_noise_time_normalized": post_noise_time,
            "noise_relaxations": noise_relaxations,
            "anti_lock_noise": anti_lock_noise,
            "anti_lock_interval_normalized": anti_lock_interval,
            "seed": seed,
        },
        "time": (np.arange(samples) * stride * dt).tolist(),
        "angles": history.tolist(),
        "relative_angles": relative_angles.tolist(),
        "angular_velocity": speeds.tolist(),
        "gram": gram.tolist(),
        "score_kernel": score_kernel.tolist(),
        "attention_weights": attention_weights.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--burn-time", type=float, default=1200.0)
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--sample-spacing", type=float, default=0.02)
    parser.add_argument("--initial-noise", type=float, default=1e-5)
    parser.add_argument("--post-noise-time", type=float, default=200.0)
    parser.add_argument("--noise-relaxations", type=int, default=2)
    parser.add_argument("--anti-lock-noise", type=float, default=0.0)
    parser.add_argument("--anti-lock-interval", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=260814801)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        args.dt,
        args.burn_time,
        args.duration,
        args.sample_spacing,
        args.initial_noise,
        args.seed,
        args.post_noise_time,
        args.noise_relaxations,
        args.anti_lock_noise,
        args.anti_lock_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(
        json.dumps(
            {
                "samples": len(result["time"]),
                "identity": result["identity"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
