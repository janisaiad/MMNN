"""Continue selected direct-ODE outcomes for a much longer metastability audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import evaluate_ode
from experiments.spectral_self_attention.small_step_continuation import stack_models


def run(
    inputs: list[Path],
    dt: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    motion_threshold: float,
    selection_threshold: float | None,
    anti_lock_noise: float,
    anti_lock_interval: float,
    seed: int,
) -> dict[str, object]:
    effective_selection_threshold = (
        motion_threshold if selection_threshold is None else selection_threshold
    )
    candidates = [
        record
        for path in inputs
        for record in json.loads(path.read_text())["records"]
        if float(record["metrics"]["motion_per_normalized_time"])
        >= effective_selection_threshold
    ]
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in candidates:
        groups[
            (int(record["family"]), int(record["identity"]["n_tokens"]))
        ].append(record)
    rows = []
    for (family, n_tokens), records in groups.items():
        models = stack_models(records)
        original_steps = models["step_size"].astype(float).copy()
        angles = np.asarray([record["final_angle"] for record in records])[:, None, :]
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, family, n_tokens, 0x10A6])
        )
        anti_lock_rng = np.random.default_rng(
            np.random.SeedSequence([seed, family, n_tokens, 0xA17])
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
            0.0,
            False,
            0.0,
            0,
            anti_lock_noise,
            anti_lock_interval,
            anti_lock_rng,
        )
        for index, record in enumerate(records):
            motion = float(metrics["motion_per_normalized_time"][index])
            gram = float(metrics["gram_variation"][index])
            rows.append(
                {
                    "family": family,
                    "label": record["label"],
                    "identity": record["identity"],
                    "model": record["model"],
                    "starting_metrics": record["metrics"],
                    "metrics": {
                        "motion_per_normalized_time": motion,
                        "continuous_field_rms": float(
                            metrics["continuous_field_rms"][index]
                        ),
                        "gram_variation": gram,
                        "score_kernel_variation": float(
                            metrics["score_kernel_variation"][index]
                        ),
                        "attention_weight_variation": float(
                            metrics["attention_weight_variation"][index]
                        ),
                        "recurrence_error": float(
                            metrics["recurrence_error"][index]
                        ),
                        "score_kernel_recurrence_error": float(
                            metrics["score_kernel_recurrence_error"][index]
                        ),
                        "lyapunov_per_normalized_time": float(
                            metrics["lyapunov_per_normalized_time"][index]
                        ),
                        "moving": motion >= motion_threshold,
                        "internal": motion >= motion_threshold and gram >= 1e-3,
                    },
                    "final_angle": final[index, 0].tolist(),
                }
            )
    summary = {
        "selected_starting_movers": len(rows),
        "still_moving": sum(row["metrics"]["moving"] for row in rows),
        "still_internal": sum(row["metrics"]["internal"] for row in rows),
        "positive_lyapunov": sum(
            row["metrics"]["lyapunov_per_normalized_time"] > 5e-3
            for row in rows
        ),
    }
    return {
        "settings": {
            "dt": dt,
            "additional_burn_time_normalized": burn_time,
            "sample_count": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "lyapunov_time_normalized": lyapunov_time,
            "motion_threshold": motion_threshold,
            "selection_threshold": effective_selection_threshold,
            "anti_lock_noise": anti_lock_noise,
            "anti_lock_interval_normalized": anti_lock_interval,
            "seed": seed,
        },
        "summary": summary,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--burn-time", type=float, default=5000.0)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-spacing", type=float, default=0.2)
    parser.add_argument("--lyapunov-time", type=float, default=1000.0)
    parser.add_argument("--motion-threshold", type=float, default=1e-3)
    parser.add_argument("--selection-threshold", type=float)
    parser.add_argument("--anti-lock-noise", type=float, default=1e-12)
    parser.add_argument("--anti-lock-interval", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=260815801)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        args.inputs,
        args.dt,
        args.burn_time,
        args.sample_count,
        args.sample_spacing,
        args.lyapunov_time,
        args.motion_threshold,
        args.selection_threshold,
        args.anti_lock_noise,
        args.anti_lock_interval,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
