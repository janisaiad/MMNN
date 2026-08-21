"""Survey global basins of one selected continuous-time transformer model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import evaluate_ode
from experiments.spectral_self_attention.continuous_ode_robustness import select_record
from experiments.spectral_self_attention.small_step_continuation import stack_models


def classify(metrics: dict[str, np.ndarray], index: int) -> str:
    motion = float(metrics["motion_per_normalized_time"][index])
    field = float(metrics["continuous_field_rms"][index])
    gram = float(metrics["gram_variation"][index])
    recurrence = float(metrics["score_kernel_recurrence_error"][index])
    lyapunov = float(metrics["lyapunov_per_normalized_time"][index])
    if motion < 1e-3 and field < 1e-3:
        return "fixed"
    if lyapunov > 3e-2 and gram >= 1e-3:
        return "internal_positive_lyapunov"
    if motion >= 1e-3 and gram >= 1e-3 and recurrence < 3e-2:
        return "internal_recurrent"
    if motion >= 1e-3 and gram < 1e-3:
        return "rigid_motion"
    if motion >= 1e-3 and gram >= 1e-3:
        return "internal_unresolved"
    return "slow_or_unresolved"


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    starts: int,
    batch_size: int,
    dt: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    initial_noise: float,
    post_noise_time: float,
    noise_relaxations: int,
    anti_lock_noise: float,
    anti_lock_interval: float,
    seed: int,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    rng = np.random.default_rng(seed)
    rows = []
    for first in range(0, starts, batch_size):
        count = min(batch_size, starts - first)
        repeated = [record] * count
        models = stack_models(repeated)
        original_steps = models["step_size"].astype(float).copy()
        angles = rng.uniform(-np.pi, np.pi, size=(count, 1, n_tokens))
        anti_lock_rng = np.random.default_rng(
            np.random.SeedSequence([seed, first, n_tokens, 0xA17])
        )
        final, metrics = evaluate_ode(
            angles,
            models,
            original_steps,
            rng,
            dt,
            burn_time,
            sample_count,
            sample_spacing,
            lyapunov_time,
            initial_noise,
            False,
            post_noise_time,
            noise_relaxations,
            anti_lock_noise,
            anti_lock_interval,
            anti_lock_rng,
        )
        for local in range(count):
            rows.append(
                {
                    "start_index": first + local,
                    "classification": classify(metrics, local),
                    "motion_per_normalized_time": float(
                        metrics["motion_per_normalized_time"][local]
                    ),
                    "continuous_field_rms": float(
                        metrics["continuous_field_rms"][local]
                    ),
                    "gram_variation": float(metrics["gram_variation"][local]),
                    "score_kernel_recurrence_error": float(
                        metrics["score_kernel_recurrence_error"][local]
                    ),
                    "lyapunov_per_normalized_time": float(
                        metrics["lyapunov_per_normalized_time"][local]
                    ),
                    "final_angle": final[local, 0].tolist(),
                }
            )
    labels = sorted({row["classification"] for row in rows})
    summary = {
        label: sum(row["classification"] == label for row in rows)
        for label in labels
    }
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "random_starts": starts,
            "batch_size": batch_size,
            "dt": dt,
            "burn_time_normalized": burn_time,
            "sample_count": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "lyapunov_time_normalized": lyapunov_time,
            "initial_noise": initial_noise,
            "post_noise_time_normalized": post_noise_time,
            "noise_relaxations": noise_relaxations,
            "anti_lock_noise": anti_lock_noise,
            "anti_lock_interval_normalized": anti_lock_interval,
            "seed": seed,
        },
        "summary": summary,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--starts", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--burn-time", type=float, default=1200.0)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--sample-spacing", type=float, default=0.2)
    parser.add_argument("--lyapunov-time", type=float, default=600.0)
    parser.add_argument("--initial-noise", type=float, default=1e-3)
    parser.add_argument("--post-noise-time", type=float, default=400.0)
    parser.add_argument("--noise-relaxations", type=int, default=2)
    parser.add_argument("--anti-lock-noise", type=float, default=0.0)
    parser.add_argument("--anti-lock-interval", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=260815401)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        args.starts,
        args.batch_size,
        args.dt,
        args.burn_time,
        args.sample_count,
        args.sample_spacing,
        args.lyapunov_time,
        args.initial_noise,
        args.post_noise_time,
        args.noise_relaxations,
        args.anti_lock_noise,
        args.anti_lock_interval,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
