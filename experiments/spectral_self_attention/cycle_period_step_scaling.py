"""Measure how a continuous cycle's period in layers grows as the step shrinks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.continuous_ode_audit import rk4_step
from experiments.spectral_self_attention.large_scale_cycle_census import (
    map_angles,
    wrap,
)
from experiments.spectral_self_attention.small_step_continuation import stack_models


def select_record(
    payload: dict[str, object], n_tokens: int, source_model_index: int
) -> dict[str, object]:
    records = [
        record
        for record in payload["records"]
        if int(record["identity"]["n_tokens"]) == n_tokens
        and int(record["identity"]["source_model_index"]) == source_model_index
    ]
    if len(records) != 1:
        raise ValueError(f"expected one selected record, found {len(records)}")
    return records[0]


def recurrence(
    history: np.ndarray, spacing: float, wrap_differences: bool = True
) -> dict[str, float]:
    lower = max(3, int(np.floor(4.0 / spacing)))
    upper = min(history.shape[0] // 2, int(np.ceil(50.0 / spacing)))
    errors = []
    for lag in range(lower, upper + 1):
        difference = history[lag:] - history[:-lag]
        if wrap_differences:
            difference = wrap(difference)
        error = float(np.quantile(np.abs(difference), 0.9))
        errors.append((lag, error))
    significant_local_minima = [
        (lag, error)
        for index, (lag, error) in enumerate(errors)
        if error < 3e-2
        and (index == 0 or error <= errors[index - 1][1])
        and (index == len(errors) - 1 or error <= errors[index + 1][1])
    ]
    if significant_local_minima:
        best_lag, best_error = significant_local_minima[0]
    else:
        best_lag, best_error = min(errors, key=lambda item: item[1])
    return {
        "return_time_normalized": best_lag * spacing,
        "recurrence_error": best_error,
    }


def feature_histories(
    history: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.stack((np.cos(history), np.sin(history)), axis=-1)
    gram = np.einsum("tid,tjd->tij", tokens, tokens, optimize=True)
    kernel = np.einsum("tid,de,tje->tij", tokens, score, tokens, optimize=True)
    return gram, kernel


def run(
    payload: dict[str, object],
    n_tokens: int,
    source_model_index: int,
    ratios: list[float],
    burn_time: float,
    observation_time: float,
    requested_spacing: float,
    ode_dt: float,
    initial_noise: float,
    continuation_noise: float,
    seed: int,
) -> dict[str, object]:
    record = select_record(payload, n_tokens, source_model_index)
    models = stack_models([record])
    original_step = models["step_size"].copy()
    rng = np.random.default_rng(seed)
    angles = np.asarray(record["initial_angle"], dtype=float)[None, None, :]
    angles = wrap(angles + initial_noise * rng.normal(size=angles.shape))
    rows = []
    for ratio in ratios:
        models["step_size"] = original_step * ratio
        angles = wrap(
            angles + continuation_noise * rng.normal(size=angles.shape)
        )
        burn_layers = int(np.ceil(burn_time / ratio))
        for _ in range(burn_layers):
            angles = map_angles(angles, models)
        stride = max(1, int(np.ceil(requested_spacing / ratio)))
        spacing = stride * ratio
        samples = max(1000, int(np.ceil(observation_time / spacing)))
        history = []
        for _ in range(samples):
            for _ in range(stride):
                angles = map_angles(angles, models)
            history.append(angles[0, 0].copy())
        history_array = np.asarray(history)
        gram_history, kernel_history = feature_histories(
            history_array, models["score"][0]
        )
        angle_metrics = recurrence(history_array, spacing)
        gram_metrics = recurrence(gram_history, spacing, False)
        kernel_metrics = recurrence(kernel_history, spacing, False)
        rows.append(
            {
                "step_ratio": ratio,
                "burn_layers": burn_layers,
                "sample_stride_layers": stride,
                "sample_spacing_normalized": spacing,
                "angle_return_time_normalized": angle_metrics[
                    "return_time_normalized"
                ],
                "angle_return_period_layers": angle_metrics[
                    "return_time_normalized"
                ]
                / ratio,
                "angle_recurrence_error": angle_metrics["recurrence_error"],
                "gram_return_time_normalized": gram_metrics[
                    "return_time_normalized"
                ],
                "gram_return_period_layers": gram_metrics[
                    "return_time_normalized"
                ]
                / ratio,
                "gram_recurrence_error": gram_metrics["recurrence_error"],
                "kernel_return_time_normalized": kernel_metrics[
                    "return_time_normalized"
                ],
                "kernel_return_period_layers": kernel_metrics[
                    "return_time_normalized"
                ]
                / ratio,
                "kernel_recurrence_error": kernel_metrics["recurrence_error"],
            }
        )
    models["step_size"] = original_step
    for _ in range(int(np.ceil(burn_time / ode_dt))):
        angles = rk4_step(angles, models, original_step, ode_dt)
    ode_stride = max(1, int(np.ceil(requested_spacing / ode_dt)))
    ode_spacing = ode_stride * ode_dt
    ode_samples = max(1000, int(np.ceil(observation_time / ode_spacing)))
    ode_history = []
    for _ in range(ode_samples):
        for _ in range(ode_stride):
            angles = rk4_step(angles, models, original_step, ode_dt)
        ode_history.append(angles[0, 0].copy())
    ode_history_array = np.asarray(ode_history)
    ode_gram, ode_kernel = feature_histories(
        ode_history_array, models["score"][0]
    )
    ode_reference = {
        "dt": ode_dt,
        "sample_spacing_normalized": ode_spacing,
        "angles": recurrence(ode_history_array, ode_spacing),
        "gram": recurrence(ode_gram, ode_spacing, False),
        "score_kernel": recurrence(ode_kernel, ode_spacing, False),
    }
    log_inverse_step = np.log([1.0 / row["step_ratio"] for row in rows])
    log_layer_period = np.log(
        [row["angle_return_period_layers"] for row in rows]
    )
    slope, intercept = np.polyfit(log_inverse_step, log_layer_period, 1)
    periods = np.asarray(
        [row["angle_return_time_normalized"] for row in rows]
    )
    gram_slope, _ = np.polyfit(
        log_inverse_step,
        np.log([row["gram_return_period_layers"] for row in rows]),
        1,
    )
    asymptotic_count = min(2, len(rows))
    asymptotic_slice = slice(len(rows) - asymptotic_count, None)
    asymptotic_slope, _ = np.polyfit(
        log_inverse_step[asymptotic_slice],
        log_layer_period[asymptotic_slice],
        1,
    )
    asymptotic_periods = periods[asymptotic_slice]
    return {
        "identity": record["identity"],
        "label": payload["label"],
        "settings": {
            "ratios": ratios,
            "burn_time_normalized": burn_time,
            "observation_time_normalized": observation_time,
            "requested_sample_spacing_normalized": requested_spacing,
            "ode_reference_dt": ode_dt,
            "initial_noise": initial_noise,
            "continuation_noise": continuation_noise,
            "seed": seed,
        },
        "summary": {
            "layer_period_power_law_slope": float(slope),
            "layer_period_power_law_intercept": float(intercept),
            "normalized_period_mean": float(np.mean(periods)),
            "normalized_period_relative_std": float(np.std(periods) / np.mean(periods)),
            "gram_layer_period_power_law_slope": float(gram_slope),
            "deepest_two_layer_period_power_law_slope": float(
                asymptotic_slope
            ),
            "deepest_two_normalized_period_mean": float(
                np.mean(asymptotic_periods)
            ),
            "deepest_two_normalized_period_relative_std": float(
                np.std(asymptotic_periods) / np.mean(asymptotic_periods)
            ),
            "deepest_step_vs_ode_period_relative_error": float(
                abs(
                    periods[-1]
                    - ode_reference["angles"]["return_time_normalized"]
                )
                / ode_reference["angles"]["return_time_normalized"]
            ),
            "deepest_step_vs_ode_angle_period_relative_error": float(
                abs(
                    rows[-1]["angle_return_time_normalized"]
                    - ode_reference["angles"]["return_time_normalized"]
                )
                / ode_reference["angles"]["return_time_normalized"]
            ),
            "deepest_step_vs_ode_gram_period_relative_error": float(
                abs(
                    rows[-1]["gram_return_time_normalized"]
                    - ode_reference["gram"]["return_time_normalized"]
                )
                / ode_reference["gram"]["return_time_normalized"]
            ),
            "deepest_step_vs_ode_kernel_period_relative_error": float(
                abs(
                    rows[-1]["kernel_return_time_normalized"]
                    - ode_reference["score_kernel"][
                        "return_time_normalized"
                    ]
                )
                / ode_reference["score_kernel"]["return_time_normalized"]
            ),
        },
        "records": rows,
        "ode_reference": ode_reference,
        "final_angle": angles[0, 0].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--source-model-index", type=int, required=True)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024],
    )
    parser.add_argument("--burn-time", type=float, default=400.0)
    parser.add_argument("--observation-time", type=float, default=40.0)
    parser.add_argument("--sample-spacing", type=float, default=0.02)
    parser.add_argument("--ode-dt", type=float, default=0.01)
    parser.add_argument("--initial-noise", type=float, default=1e-3)
    parser.add_argument("--continuation-noise", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=260816101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        json.loads(args.input.read_text()),
        args.n_tokens,
        args.source_model_index,
        args.ratios,
        args.burn_time,
        args.observation_time,
        args.sample_spacing,
        args.ode_dt,
        args.initial_noise,
        args.continuation_noise,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
