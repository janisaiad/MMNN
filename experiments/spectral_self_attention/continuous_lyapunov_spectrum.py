"""Compute the full Lyapunov spectrum of a selected continuous attractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.large_scale_cycle_census import wrap
from experiments.spectral_self_attention.small_step_continuation import stack_models


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


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    dt: float,
    burn_time: float,
    duration: float,
    epsilon: float,
    initial_noise: float,
    seed: int,
    component_mode: str = "full",
    post_noise_time: float = 200.0,
    noise_relaxations: int = 2,
    anti_lock_noise: float = 0.0,
    anti_lock_interval: float = 100.0,
) -> dict[str, object]:
    record = selected_record(payload, n_tokens, source_model_index)
    models = stack_models([record])
    if component_mode == "attention_only":
        models["mlp_bias"].fill(0.0)
        models["linear"].fill(0.0)
        models["output"].fill(0.0)
    elif component_mode == "mlp_only":
        models["value"].fill(0.0)
    original_steps = models["step_size"].astype(float).copy()
    rng = np.random.default_rng(seed)
    anti_lock_rng = np.random.default_rng(
        np.random.SeedSequence([seed, n_tokens, source_model_index, 0xA17])
    )
    base = np.asarray(record["initial_angle"], dtype=float)[None, None, :]
    base = wrap(base + initial_noise * rng.normal(size=base.shape))
    for _ in range(int(np.ceil(burn_time / dt))):
        base = rk4_step(base, models, original_steps, dt)
    if initial_noise > 0.0 and post_noise_time > 0.0:
        for _ in range(noise_relaxations):
            base = wrap(base + initial_noise * rng.normal(size=base.shape))
            for _ in range(int(np.ceil(post_noise_time / dt))):
                base = rk4_step(base, models, original_steps, dt)
    basis, _ = np.linalg.qr(rng.normal(size=(n_tokens, n_tokens)))
    growth = np.zeros(n_tokens)
    steps = int(np.ceil(duration / dt))
    anti_lock_stride = max(1, int(np.ceil(anti_lock_interval / dt)))
    checkpoint_stride = max(1, steps // 100)
    checkpoints = []
    for step in range(steps):
        if anti_lock_noise > 0.0 and step % anti_lock_stride == 0:
            base = wrap(
                base + anti_lock_noise * anti_lock_rng.normal(size=base.shape)
            )
        states = np.empty((1, n_tokens + 1, n_tokens))
        states[0, 0] = base[0, 0]
        for column in range(n_tokens):
            states[0, column + 1] = wrap(base[0, 0] + epsilon * basis[:, column])
        advanced = rk4_step(states, models, original_steps, dt)[0]
        base[0, 0] = advanced[0]
        tangent = np.stack(
            [wrap(advanced[column + 1] - advanced[0]) / epsilon for column in range(n_tokens)],
            axis=-1,
        )
        basis, triangular = np.linalg.qr(tangent)
        growth += np.log(np.maximum(np.abs(np.diag(triangular)), 1e-15))
        if (step + 1) % checkpoint_stride == 0 or step + 1 == steps:
            checkpoints.append(
                {
                    "time": (step + 1) * dt,
                    "spectrum": (growth / ((step + 1) * dt)).tolist(),
                }
            )
    spectrum = np.sort(growth / (steps * dt))[::-1]
    positive_sum = float(np.sum(spectrum[spectrum > 0.0]))
    kaplan_yorke = 0.0
    cumulative = 0.0
    for index, exponent in enumerate(spectrum):
        if cumulative + exponent < 0.0:
            kaplan_yorke = index + cumulative / abs(exponent)
            break
        cumulative += exponent
    else:
        kaplan_yorke = float(n_tokens)
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "dt": dt,
            "burn_time_normalized": burn_time,
            "duration_normalized": steps * dt,
            "epsilon": epsilon,
            "initial_noise": initial_noise,
            "seed": seed,
            "component_mode": component_mode,
            "post_noise_time_normalized": post_noise_time,
            "noise_relaxations": noise_relaxations,
            "anti_lock_noise": anti_lock_noise,
            "anti_lock_interval_normalized": anti_lock_interval,
        },
        "spectrum": spectrum.tolist(),
        "positive_exponent_sum": positive_sum,
        "kaplan_yorke_dimension": kaplan_yorke,
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--burn-time", type=float, default=1200.0)
    parser.add_argument("--duration", type=float, default=2000.0)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--initial-noise", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=260814901)
    parser.add_argument(
        "--component-mode",
        choices=("full", "attention_only", "mlp_only"),
        default="full",
    )
    parser.add_argument("--post-noise-time", type=float, default=200.0)
    parser.add_argument("--noise-relaxations", type=int, default=2)
    parser.add_argument("--anti-lock-noise", type=float, default=0.0)
    parser.add_argument("--anti-lock-interval", type=float, default=100.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        args.dt,
        args.burn_time,
        args.duration,
        args.epsilon,
        args.initial_noise,
        args.seed,
        args.component_mode,
        args.post_noise_time,
        args.noise_relaxations,
        args.anti_lock_noise,
        args.anti_lock_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("spectrum", "kaplan_yorke_dimension")}, indent=2))


if __name__ == "__main__":
    main()
