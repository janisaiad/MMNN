"""Re-run one candidate from every discrete-cycle phase and noise trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import evaluate_ode
from experiments.spectral_self_attention.large_scale_cycle_census import map_angles
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


def cycle_phases(record: dict[str, object], period: int) -> list[np.ndarray]:
    model = stack_models([record])
    angle = np.asarray(record["initial_angle"], dtype=float)[None, None, :]
    phases = []
    for _ in range(period):
        phases.append(angle[0, 0].copy())
        angle = map_angles(angle, model)
    return phases


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    noise_trials: int,
    dt: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    initial_noise: float,
    seed: int,
    post_noise_time: float = 200.0,
    noise_relaxations: int = 2,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    period = int(str(payload["label"])[1:]) if str(payload["label"]).startswith("p") else 1
    phases = cycle_phases(record, period)
    starts = []
    identities = []
    for phase_index, phase in enumerate(phases):
        for trial in range(noise_trials):
            starts.append(phase)
            identities.append((phase_index, trial))
    repeated = [record] * len(starts)
    models = stack_models(repeated)
    original_steps = models["step_size"].astype(float).copy()
    rng = np.random.default_rng(seed)
    final, metrics = evaluate_ode(
        np.asarray(starts)[:, None, :],
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
    )
    rows = []
    for index, (phase_index, trial) in enumerate(identities):
        motion = float(metrics["motion_per_normalized_time"][index])
        field = float(metrics["continuous_field_rms"][index])
        lyapunov = float(metrics["lyapunov_per_normalized_time"][index])
        gram_variation = float(metrics["gram_variation"][index])
        kernel_recurrence = float(metrics["score_kernel_recurrence_error"][index])
        attention_recurrence = float(
            metrics["attention_weight_recurrence_error"][index]
        )
        rows.append(
            {
                "phase_index": phase_index,
                "noise_trial": trial,
                "motion_per_normalized_time": motion,
                "continuous_field_rms": field,
                "recurrence_error": float(metrics["recurrence_error"][index]),
                "gram_variation": gram_variation,
                "gram_recurrence_error": float(
                    metrics["gram_recurrence_error"][index]
                ),
                "score_kernel_variation": float(
                    metrics["score_kernel_variation"][index]
                ),
                "score_kernel_recurrence_error": kernel_recurrence,
                "attention_weight_variation": float(
                    metrics["attention_weight_variation"][index]
                ),
                "attention_weight_recurrence_error": attention_recurrence,
                "lyapunov_per_normalized_time": lyapunov,
                "fixed": field < 1e-3 and motion < 1e-3,
                "internal_kernel_cycle": (
                    motion >= 1e-3
                    and gram_variation >= 1e-3
                    and attention_recurrence < 3e-2
                    and lyapunov < 3e-2
                ),
                "robust_positive_lyapunov": lyapunov > 3e-2,
                "final_angle": final[index, 0].tolist(),
            }
        )
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "cycle_phases": period,
            "noise_trials_per_phase": noise_trials,
            "dt": dt,
            "burn_time_normalized": burn_time,
            "sample_count": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "lyapunov_time_normalized": lyapunov_time,
            "initial_noise": initial_noise,
            "post_noise_time_normalized": post_noise_time,
            "noise_relaxations": noise_relaxations,
            "seed": seed,
        },
        "summary": {
            "runs": len(rows),
            "fixed": sum(row["fixed"] for row in rows),
            "internal_kernel_cycle": sum(
                row["internal_kernel_cycle"] for row in rows
            ),
            "robust_positive_lyapunov": sum(
                row["robust_positive_lyapunov"] for row in rows
            ),
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument("--noise-trials", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--burn-time", type=float, default=1200.0)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-spacing", type=float, default=0.1)
    parser.add_argument("--lyapunov-time", type=float, default=600.0)
    parser.add_argument("--initial-noise", type=float, default=1e-5)
    parser.add_argument("--post-noise-time", type=float, default=200.0)
    parser.add_argument("--noise-relaxations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260814701)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        args.noise_trials,
        args.dt,
        args.burn_time,
        args.sample_count,
        args.sample_spacing,
        args.lyapunov_time,
        args.initial_noise,
        args.seed,
        args.post_noise_time,
        args.noise_relaxations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
