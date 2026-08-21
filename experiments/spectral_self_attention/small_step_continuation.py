"""Continue finite-layer attractors toward the continuous-time limit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import (
    map_angles,
    wrap,
)


def stack_models(records: list[dict[str, object]]) -> dict[str, np.ndarray]:
    keys = records[0]["model"].keys()
    return {
        key: np.stack([np.asarray(record["model"][key]) for record in records])
        for key in keys
    }


def continuous_angular_field(
    angles: np.ndarray, models: dict[str, np.ndarray]
) -> np.ndarray:
    """First-order angular field of the serial Attention->MLP map."""
    tokens = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    scores = np.einsum(
        "mbid,mde,mbje->mbij", tokens, models["score"], tokens, optimize=True
    )
    logits = models["beta"][:, None, None, None] * scores
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(np.clip(logits, -80.0, 0.0))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    values = np.einsum("mde,mbje->mbjd", models["value"], tokens, optimize=True)
    attention = np.einsum("mbij,mbjd->mbid", weights, values, optimize=True)

    hidden = np.einsum(
        "mrd,mbnd->mbnr", models["hidden"], tokens, optimize=True
    )
    hidden += models["hidden_bias"][:, None, None, :]
    mlp = models["mlp_bias"][:, None, None, :]
    mlp = mlp + np.einsum(
        "mde,mbne->mbnd", models["linear"], tokens, optimize=True
    )
    mlp = mlp + np.einsum(
        "mdr,mbnr->mbnd", models["output"], hidden**2, optimize=True
    )
    tangent = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
    return np.einsum("mbnd,mbnd->mbn", tangent, attention + mlp, optimize=True)


def beta0_type1_potential(
    angles: np.ndarray, models: dict[str, np.ndarray]
) -> np.ndarray:
    """Global potential whose spherical gradient is the beta=0 type-1 field."""
    tokens = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    n_tokens = angles.shape[-1]
    mean_token = np.mean(tokens, axis=-2)
    attention = 0.5 * n_tokens * np.einsum(
        "mbd,mde,mbe->mb", mean_token, models["value"], mean_token, optimize=True
    )
    bias = np.einsum("md,mbnd->mbn", models["mlp_bias"], tokens, optimize=True)
    linear = 0.5 * np.einsum(
        "mbnd,mde,mbne->mbn", tokens, models["linear"], tokens, optimize=True
    )
    hidden = np.einsum(
        "mrd,mbnd->mbnr", models["hidden"], tokens, optimize=True
    )
    hidden += models["hidden_bias"][:, None, None, :]
    coefficients = np.einsum(
        "mdr,mrd->mr", models["output"], models["hidden"], optimize=True
    )
    cubic = np.einsum(
        "mr,mbnr->mbn", coefficients / 3.0, hidden**3, optimize=True
    )
    return attention + np.sum(bias + linear + cubic, axis=-1)


def largest_lyapunov_normalized_time(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    rng: np.random.Generator,
    ratio: float,
    normalized_time: float,
    epsilon: float = 1e-7,
) -> np.ndarray:
    steps = max(1, int(np.ceil(normalized_time / ratio)))
    base = angles.copy()
    direction = rng.normal(size=base.shape)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    perturbed = wrap(base + epsilon * direction)
    growth = np.zeros(base.shape[:2])
    for _ in range(steps):
        base = map_angles(base, models)
        perturbed = map_angles(perturbed, models)
        delta = wrap(perturbed - base)
        norm = np.linalg.norm(delta, axis=-1)
        safe = np.maximum(norm, 1e-15)
        growth += np.log(safe / epsilon)
        direction = delta / safe[..., None]
        collapsed = norm < 1e-14
        if np.any(collapsed):
            replacement = rng.normal(size=(int(collapsed.sum()), base.shape[-1]))
            replacement /= np.maximum(
                np.linalg.norm(replacement, axis=-1, keepdims=True), 1e-12
            )
            direction[collapsed] = replacement
        perturbed = wrap(base + epsilon * direction)
    return growth / (steps * ratio)


def history_metrics(
    history: np.ndarray,
    models: dict[str, np.ndarray],
    path_length: np.ndarray,
    local_error_sum: np.ndarray,
    local_reference_sum: np.ndarray,
    field_square_sum: np.ndarray,
    sampled_points: int,
    total_layers: int,
    ratio: float,
    stride: int,
) -> dict[str, np.ndarray]:
    _, models_count, _, n_tokens = history.shape
    motion = path_length / (total_layers * ratio)
    field_rms = np.sqrt(field_square_sum / (sampled_points * n_tokens))
    relative_local_error = local_error_sum / np.maximum(local_reference_sum, 1e-12)
    absolute_local_error = local_error_sum / total_layers

    maximum_lag = min(80, history.shape[0] // 2)
    recurrence = np.full(models_count, np.inf)
    best_lag = np.zeros(models_count, dtype=int)
    for lag in range(3, maximum_lag + 1):
        differences = np.abs(wrap(history[lag:] - history[:-lag]))
        error = np.quantile(differences, 0.9, axis=(0, 2, 3))
        improved = error < recurrence
        recurrence[improved] = error[improved]
        best_lag[improved] = lag

    tokens = np.stack((np.cos(history), np.sin(history)), axis=-1)
    gram = np.einsum("tmbid,tmbjd->tmbij", tokens, tokens, optimize=True)
    gram_variation = np.max(np.std(gram, axis=0), axis=(1, 2, 3))
    score_kernel = np.einsum(
        "tmbid,mde,tmbje->tmbij", tokens, models["score"], tokens, optimize=True
    )
    score_kernel_variation = np.max(
        np.std(score_kernel, axis=0), axis=(1, 2, 3)
    )
    gram_recurrence = np.full(models_count, np.inf)
    score_kernel_recurrence = np.full(models_count, np.inf)
    gram_best_lag = np.zeros(models_count, dtype=int)
    score_kernel_best_lag = np.zeros(models_count, dtype=int)
    for lag in range(3, maximum_lag + 1):
        gram_error = np.quantile(
            np.abs(gram[lag:] - gram[:-lag]), 0.9, axis=(0, 2, 3, 4)
        )
        score_error = np.quantile(
            np.abs(score_kernel[lag:] - score_kernel[:-lag]),
            0.9,
            axis=(0, 2, 3, 4),
        )
        gram_improved = gram_error < gram_recurrence
        score_improved = score_error < score_kernel_recurrence
        gram_recurrence[gram_improved] = gram_error[gram_improved]
        score_kernel_recurrence[score_improved] = score_error[score_improved]
        gram_best_lag[gram_improved] = lag
        score_kernel_best_lag[score_improved] = lag
    coherence = np.mean(np.abs(np.mean(np.exp(1j * history), axis=-1)), axis=0)[:, 0]
    unwrapped = np.unwrap(history[:, :, 0, :], axis=0)
    elapsed = max((history.shape[0] - 1) * stride * ratio, ratio)
    winding_speed = np.mean((unwrapped[-1] - unwrapped[0]) / elapsed, axis=-1)
    return {
        "motion_per_normalized_time": motion[:, 0],
        "continuous_field_rms": field_rms,
        "relative_local_error": relative_local_error[:, 0],
        "absolute_local_error": absolute_local_error[:, 0],
        "recurrence_error": recurrence,
        "return_time_normalized": best_lag * stride * ratio,
        "gram_variation": gram_variation,
        "gram_recurrence_error": gram_recurrence,
        "gram_return_time_normalized": gram_best_lag * stride * ratio,
        "score_kernel_variation": score_kernel_variation,
        "score_kernel_recurrence_error": score_kernel_recurrence,
        "score_kernel_return_time_normalized": score_kernel_best_lag * stride * ratio,
        "mean_coherence": coherence,
        "mean_winding_speed": winding_speed,
    }


def evaluate_ratio(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    original_steps: np.ndarray,
    ratio: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    rng: np.random.Generator,
    continuation_noise: float = 1e-7,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    models["step_size"] = original_steps * ratio
    if continuation_noise > 0.0:
        angles = wrap(angles + continuation_noise * rng.normal(size=angles.shape))
    burn_layers = int(np.ceil(burn_time / ratio))
    for _ in range(burn_layers):
        angles = map_angles(angles, models)

    stride = max(1, int(np.ceil(sample_spacing / ratio)))
    path_length = np.zeros(angles.shape[:2])
    local_error_sum = np.zeros(angles.shape[:2])
    local_reference_sum = np.zeros(angles.shape[:2])
    field_square_sum = np.zeros(angles.shape[0])
    history = []
    total_layers = 0
    for _ in range(sample_count):
        for _ in range(stride):
            before = angles
            field = continuous_angular_field(before, models)
            after = map_angles(before, models)
            delta = wrap(after - before)
            normalized_discrete_velocity = delta / ratio
            normalized_continuous_velocity = original_steps[:, None, None] * field
            error = np.linalg.norm(
                normalized_discrete_velocity - normalized_continuous_velocity, axis=-1
            )
            reference = np.linalg.norm(normalized_continuous_velocity, axis=-1)
            local_error_sum += error
            local_reference_sum += reference
            path_length += np.mean(np.abs(delta), axis=-1)
            angles = after
            total_layers += 1
        field = continuous_angular_field(angles, models)
        normalized_field = original_steps[:, None, None] * field
        field_square_sum += np.sum(normalized_field[:, 0, :] ** 2, axis=-1)
        history.append(angles.copy())
    metrics = history_metrics(
        np.stack(history),
        models,
        path_length,
        local_error_sum,
        local_reference_sum,
        field_square_sum,
        sample_count,
        total_layers,
        ratio,
        stride,
    )
    metrics["lyapunov_per_normalized_time"] = largest_lyapunov_normalized_time(
        angles, models, rng, ratio, lyapunov_time
    )[:, 0]
    metrics["absolute_step_min"] = original_steps * ratio
    metrics["sample_stride_layers"] = np.full(angles.shape[0], stride)
    metrics["burn_layers"] = np.full(angles.shape[0], burn_layers)
    return angles, metrics


def load_records(
    paths: list[Path], family: int, label: str
) -> list[dict[str, object]]:
    records = []
    for path in paths:
        source = json.loads(path.read_text())
        if int(source["family"]) == family:
            records.extend(source["records"][label])
    return records


def stratified_sample(
    records: list[dict[str, object]], maximum: int, rng: np.random.Generator
) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[(int(record["n_tokens"]), int(record["subtype_code"]))].append(record)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    ordered_keys = sorted(groups)
    while len(selected) < maximum and any(groups.values()):
        for key in ordered_keys:
            if groups[key] and len(selected) < maximum:
                selected.append(groups[key].pop())
    return selected


def run(
    inputs: list[Path],
    family: int,
    label: str,
    maximum: int,
    ratios: list[float],
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    seed: int,
    continuation_noise: float = 1e-7,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    records = stratified_sample(load_records(inputs, family, label), maximum, rng)
    output_records: list[dict[str, object] | None] = [None] * len(records)
    for n_tokens in sorted({int(record["n_tokens"]) for record in records}):
        indices = [
            index
            for index, record in enumerate(records)
            if int(record["n_tokens"]) == n_tokens
        ]
        selected = [records[index] for index in indices]
        models = stack_models(selected)
        original_steps = models["step_size"].astype(float).copy()
        angles = np.asarray([record["angle"] for record in selected])[:, None, :]
        traces: list[list[dict[str, float | int | bool]]] = [
            [] for _ in selected
        ]
        for ratio in ratios:
            angles, metrics = evaluate_ratio(
                angles,
                models,
                original_steps,
                ratio,
                burn_time,
                sample_count,
                sample_spacing,
                lyapunov_time,
                rng,
                continuation_noise,
            )
            for model_index, trace in enumerate(traces):
                field_rms = float(metrics["continuous_field_rms"][model_index])
                motion = float(metrics["motion_per_normalized_time"][model_index])
                recurrence = float(metrics["recurrence_error"][model_index])
                lyapunov = float(
                    metrics["lyapunov_per_normalized_time"][model_index]
                )
                trace.append(
                    {
                        "ratio": ratio,
                        "absolute_step": float(
                            metrics["absolute_step_min"][model_index]
                        ),
                        "motion_per_normalized_time": motion,
                        "continuous_field_rms": field_rms,
                        "relative_local_error": float(
                            metrics["relative_local_error"][model_index]
                        ),
                        "absolute_local_error": float(
                            metrics["absolute_local_error"][model_index]
                        ),
                        "recurrence_error": recurrence,
                        "return_time_normalized": float(
                            metrics["return_time_normalized"][model_index]
                        ),
                        "gram_variation": float(
                            metrics["gram_variation"][model_index]
                        ),
                        "gram_recurrence_error": float(
                            metrics["gram_recurrence_error"][model_index]
                        ),
                        "gram_return_time_normalized": float(
                            metrics["gram_return_time_normalized"][model_index]
                        ),
                        "score_kernel_variation": float(
                            metrics["score_kernel_variation"][model_index]
                        ),
                        "score_kernel_recurrence_error": float(
                            metrics["score_kernel_recurrence_error"][model_index]
                        ),
                        "score_kernel_return_time_normalized": float(
                            metrics["score_kernel_return_time_normalized"][model_index]
                        ),
                        "mean_coherence": float(
                            metrics["mean_coherence"][model_index]
                        ),
                        "mean_winding_speed": float(
                            metrics["mean_winding_speed"][model_index]
                        ),
                        "lyapunov_per_normalized_time": lyapunov,
                        "fixed": field_rms < 1e-3 and motion < 1e-3,
                        "map_stationary": motion < 1e-8,
                        "moving_recurrent": (
                            motion >= 1e-3 and recurrence < 3e-2
                        ),
                        "positive_lyapunov": lyapunov > 5e-3,
                        "sample_stride_layers": int(
                            metrics["sample_stride_layers"][model_index]
                        ),
                        "burn_layers": int(metrics["burn_layers"][model_index]),
                    }
                )
        for local_index, global_index in enumerate(indices):
            output_records[global_index] = {
                "identity": {
                    key: records[global_index][key]
                    for key in (
                        "family",
                        "n_tokens",
                        "subtype_code",
                        "screen_period",
                        "screen_periodic_residual",
                        "screen_lyapunov_per_layer",
                        "source_model_index",
                    )
                },
                "model": records[global_index]["model"],
                "initial_angle": records[global_index]["angle"],
                "trace": traces[local_index],
                "final_angle": angles[local_index, 0].tolist(),
            }
    complete_records = [record for record in output_records if record is not None]
    final = [record["trace"][-1] for record in complete_records]
    return {
        "family": family,
        "label": label,
        "settings": {
            "available_records": len(load_records(inputs, family, label)),
            "continued_records": len(complete_records),
            "ratios": ratios,
            "burn_time_normalized": burn_time,
            "sample_count": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "lyapunov_time_normalized": lyapunov_time,
            "seed": seed,
            "continuation_noise": continuation_noise,
        },
        "final_counts": {
            "fixed": sum(bool(row["fixed"]) for row in final),
            "map_stationary": sum(bool(row["map_stationary"]) for row in final),
            "moving_recurrent": sum(bool(row["moving_recurrent"]) for row in final),
            "positive_lyapunov": sum(bool(row["positive_lyapunov"]) for row in final),
            "moving_nonrecurrent": sum(
                row["motion_per_normalized_time"] >= 1e-3
                and not row["moving_recurrent"]
                for row in final
            ),
        },
        "records": complete_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--family", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--label", required=True, choices=("p3", "p4", "chaos"))
    parser.add_argument("--maximum", type=int, default=256)
    parser.add_argument(
        "--ratios",
        default=(
            "1,.7,.5,.35,.25,.18,.125,.088,.0625,.044,.03125,.022,"
            ".015625,.011,.0078125,.0055,.00390625"
        ),
    )
    parser.add_argument("--burn-time", type=float, default=600.0)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--sample-spacing", type=float, default=0.25)
    parser.add_argument("--lyapunov-time", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=260814101)
    parser.add_argument("--continuation-noise", type=float, default=1e-7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        inputs=args.inputs,
        family=args.family,
        label=args.label,
        maximum=args.maximum,
        ratios=[float(value) for value in args.ratios.split(",")],
        burn_time=args.burn_time,
        sample_count=args.sample_count,
        sample_spacing=args.sample_spacing,
        lyapunov_time=args.lyapunov_time,
        seed=args.seed + 10 * args.family + ("p3", "p4", "chaos").index(args.label),
        continuation_noise=args.continuation_noise,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "family": result["family"],
                "label": result["label"],
                "continued": result["settings"]["continued_records"],
                "final_counts": result["final_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
