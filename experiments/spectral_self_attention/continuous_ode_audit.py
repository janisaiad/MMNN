"""Direct RK4 audit of the continuous-time limit of continued attractors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import wrap
from experiments.spectral_self_attention.small_step_continuation import (
    beta0_type1_potential,
    continuous_angular_field,
    stack_models,
)


def normalized_field(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    original_steps: np.ndarray,
) -> np.ndarray:
    return original_steps[:, None, None] * continuous_angular_field(angles, models)


def rk4_step(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    original_steps: np.ndarray,
    dt: float,
) -> np.ndarray:
    k1 = normalized_field(angles, models, original_steps)
    k2 = normalized_field(wrap(angles + 0.5 * dt * k1), models, original_steps)
    k3 = normalized_field(wrap(angles + 0.5 * dt * k2), models, original_steps)
    k4 = normalized_field(wrap(angles + dt * k3), models, original_steps)
    return wrap(angles + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def ode_lyapunov(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    original_steps: np.ndarray,
    rng: np.random.Generator,
    dt: float,
    duration: float,
    epsilon: float = 1e-7,
    anti_lock_noise: float = 0.0,
    anti_lock_interval: float = 100.0,
    anti_lock_rng: np.random.Generator | None = None,
) -> np.ndarray:
    steps = max(1, int(np.ceil(duration / dt)))
    base = angles.copy()
    direction = rng.normal(size=base.shape)
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
    perturbed = wrap(base + epsilon * direction)
    growth = np.zeros(base.shape[:2])
    kick_rng = anti_lock_rng if anti_lock_rng is not None else rng
    anti_lock_stride = max(1, int(np.ceil(anti_lock_interval / dt)))
    for step in range(steps):
        if anti_lock_noise > 0.0 and step % anti_lock_stride == 0:
            kick = anti_lock_noise * kick_rng.normal(size=base.shape)
            base = wrap(base + kick)
            perturbed = wrap(perturbed + kick)
        base = rk4_step(base, models, original_steps, dt)
        perturbed = rk4_step(perturbed, models, original_steps, dt)
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
    return growth / (steps * dt)


def recurrence_metrics(history: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    models_count = history.shape[1]
    recurrence = np.full(models_count, np.inf)
    best_lag = np.zeros(models_count, dtype=int)
    for lag in range(3, min(100, history.shape[0] // 2) + 1):
        error = np.quantile(
            np.abs(wrap(history[lag:] - history[:-lag])),
            0.9,
            axis=(0, 2, 3),
        )
        improved = error < recurrence
        recurrence[improved] = error[improved]
        best_lag[improved] = lag
    return recurrence, best_lag * spacing


def matrix_recurrence_metrics(
    history: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    models_count = history.shape[1]
    recurrence = np.full(models_count, np.inf)
    best_lag = np.zeros(models_count, dtype=int)
    for lag in range(3, min(100, history.shape[0] // 2) + 1):
        error = np.quantile(
            np.abs(history[lag:] - history[:-lag]), 0.9, axis=(0, 2, 3, 4)
        )
        improved = error < recurrence
        recurrence[improved] = error[improved]
        best_lag[improved] = lag
    return recurrence, best_lag * spacing


def evaluate_ode(
    angles: np.ndarray,
    models: dict[str, np.ndarray],
    original_steps: np.ndarray,
    rng: np.random.Generator,
    dt: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    initial_noise: float = 1e-7,
    track_potential: bool = False,
    post_noise_time: float = 200.0,
    noise_relaxations: int = 2,
    anti_lock_noise: float = 0.0,
    anti_lock_interval: float = 100.0,
    anti_lock_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    kick_rng = anti_lock_rng if anti_lock_rng is not None else rng
    if initial_noise > 0.0:
        angles = wrap(angles + initial_noise * rng.normal(size=angles.shape))
    initial_potential = (
        beta0_type1_potential(angles, models)[:, 0] if track_potential else None
    )
    burn_steps = int(np.ceil(burn_time / dt))
    anti_lock_stride = max(1, int(np.ceil(anti_lock_interval / dt)))
    for step in range(burn_steps):
        if anti_lock_noise > 0.0 and step % anti_lock_stride == 0:
            angles = wrap(
                angles + anti_lock_noise * kick_rng.normal(size=angles.shape)
            )
        angles = rk4_step(angles, models, original_steps, dt)
    if initial_noise > 0.0 and post_noise_time > 0.0:
        for _ in range(noise_relaxations):
            angles = wrap(angles + initial_noise * rng.normal(size=angles.shape))
            for step in range(int(np.ceil(post_noise_time / dt))):
                if anti_lock_noise > 0.0 and step % anti_lock_stride == 0:
                    angles = wrap(
                        angles + anti_lock_noise * kick_rng.normal(size=angles.shape)
                    )
                angles = rk4_step(angles, models, original_steps, dt)
    stride = max(1, int(np.ceil(sample_spacing / dt)))
    effective_spacing = stride * dt
    history = []
    path = np.zeros(angles.shape[:2])
    field_square = np.zeros(angles.shape[0])
    potential_history = []
    elapsed_steps = 0
    for _ in range(sample_count):
        for _ in range(stride):
            if (
                anti_lock_noise > 0.0
                and elapsed_steps % anti_lock_stride == 0
            ):
                angles = wrap(
                    angles + anti_lock_noise * kick_rng.normal(size=angles.shape)
                )
            after = rk4_step(angles, models, original_steps, dt)
            path += np.mean(np.abs(wrap(after - angles)), axis=-1)
            angles = after
            elapsed_steps += 1
        field = normalized_field(angles, models, original_steps)
        field_square += np.sum(field[:, 0, :] ** 2, axis=-1)
        history.append(angles.copy())
        if track_potential:
            potential_history.append(beta0_type1_potential(angles, models)[:, 0])
    stacked = np.stack(history)
    recurrence, return_time = recurrence_metrics(stacked, effective_spacing)
    elapsed = sample_count * effective_spacing
    motion = path[:, 0] / elapsed
    field_rms = np.sqrt(field_square / (sample_count * angles.shape[-1]))
    tokens = np.stack((np.cos(stacked), np.sin(stacked)), axis=-1)
    gram = np.einsum("tmbid,tmbjd->tmbij", tokens, tokens, optimize=True)
    gram_variation = np.max(np.std(gram, axis=0), axis=(1, 2, 3))
    gram_recurrence, gram_return_time = matrix_recurrence_metrics(
        gram, effective_spacing
    )
    score_kernel = np.einsum(
        "tmbid,mde,tmbje->tmbij", tokens, models["score"], tokens, optimize=True
    )
    score_kernel_variation = np.max(
        np.std(score_kernel, axis=0), axis=(1, 2, 3)
    )
    score_kernel_recurrence, score_kernel_return_time = matrix_recurrence_metrics(
        score_kernel, effective_spacing
    )
    logits = models["beta"][None, :, None, None, None] * score_kernel
    logits -= np.max(logits, axis=-1, keepdims=True)
    attention_weights = np.exp(np.clip(logits, -80.0, 0.0))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    attention_weight_variation = np.max(
        np.std(attention_weights, axis=0), axis=(1, 2, 3)
    )
    attention_weight_recurrence, attention_weight_return_time = (
        matrix_recurrence_metrics(attention_weights, effective_spacing)
    )
    coherence = np.mean(
        np.abs(np.mean(np.exp(1j * stacked), axis=-1)), axis=0
    )[:, 0]
    lyapunov = ode_lyapunov(
        angles,
        models,
        original_steps,
        rng,
        dt,
        lyapunov_time,
        anti_lock_noise=anti_lock_noise,
        anti_lock_interval=anti_lock_interval,
        anti_lock_rng=kick_rng,
    )[:, 0]
    if potential_history:
        potentials = np.stack(potential_history)
        potential_gain = potentials[-1] - initial_potential
        minimum_potential_increment = np.min(np.diff(potentials, axis=0), axis=0)
    else:
        potential_gain = np.full(angles.shape[0], np.nan)
        minimum_potential_increment = np.full(angles.shape[0], np.nan)
    return angles, {
        "motion_per_normalized_time": motion,
        "continuous_field_rms": field_rms,
        "recurrence_error": recurrence,
        "return_time_normalized": return_time,
        "gram_variation": gram_variation,
        "gram_recurrence_error": gram_recurrence,
        "gram_return_time_normalized": gram_return_time,
        "score_kernel_variation": score_kernel_variation,
        "score_kernel_recurrence_error": score_kernel_recurrence,
        "score_kernel_return_time_normalized": score_kernel_return_time,
        "attention_weight_variation": attention_weight_variation,
        "attention_weight_recurrence_error": attention_weight_recurrence,
        "attention_weight_return_time_normalized": attention_weight_return_time,
        "mean_coherence": coherence,
        "lyapunov_per_normalized_time": lyapunov,
        "potential_gain": potential_gain,
        "minimum_potential_increment": minimum_potential_increment,
        "burn_steps": np.full(angles.shape[0], burn_steps),
        "effective_sample_spacing": np.full(angles.shape[0], effective_spacing),
    }


def load_continued_records(paths: list[Path]) -> list[dict[str, object]]:
    records = []
    for path in paths:
        source = json.loads(path.read_text())
        for record in source["records"]:
            records.append(
                {
                    "family": source["family"],
                    "label": source["label"],
                    **record,
                }
            )
    return records


def load_cohort_angles(paths: list[Path]) -> dict[tuple[int, int, int, str], list[float]]:
    lookup = {}
    for path in paths:
        source = json.loads(path.read_text())
        family = int(source["family"])
        n_tokens = int(source["n_tokens"])
        for label, records in source["records"].items():
            for record in records:
                lookup[
                    (family, n_tokens, int(record["source_model_index"]), label)
                ] = record["angle"]
    return lookup


def run(
    inputs: list[Path],
    dt: float,
    burn_time: float,
    sample_count: int,
    sample_spacing: float,
    lyapunov_time: float,
    seed: int,
    start_mode: str = "final",
    cohort_inputs: list[Path] | None = None,
    initial_noise: float = 1e-7,
    selection: str = "all",
    selected_n_tokens: int | None = None,
    selected_model_index: int | None = None,
    component_mode: str = "full",
    post_noise_time: float = 200.0,
    noise_relaxations: int = 2,
    anti_lock_noise: float = 0.0,
    anti_lock_interval: float = 100.0,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    records = load_continued_records(inputs)
    if selection == "moving":
        records = [
            record
            for record in records
            if float(record["trace"][-1]["motion_per_normalized_time"]) >= 1e-3
        ]
    elif selection == "positive":
        records = [
            record
            for record in records
            if bool(record["trace"][-1]["positive_lyapunov"])
        ]
    elif selection == "internal":
        records = [
            record
            for record in records
            if float(record["trace"][-1]["gram_variation"]) >= 1e-3
        ]
    if selected_n_tokens is not None:
        records = [
            record
            for record in records
            if int(record["identity"]["n_tokens"]) == selected_n_tokens
        ]
    if selected_model_index is not None:
        records = [
            record
            for record in records
            if int(record["identity"]["source_model_index"]) == selected_model_index
        ]
    if not records:
        raise ValueError("record selection is empty")
    cohort_angles = load_cohort_angles(cohort_inputs or [])
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(int(record["family"]), int(record["identity"]["n_tokens"]))].append(index)
    output: list[dict[str, object] | None] = [None] * len(records)
    for indices in groups.values():
        selected = [records[index] for index in indices]
        family = int(selected[0]["family"])
        n_tokens = int(selected[0]["identity"]["n_tokens"])
        anti_lock_rng = np.random.default_rng(
            np.random.SeedSequence([seed, family, n_tokens, 0xA17])
        )
        models = stack_models(selected)
        if component_mode == "attention_only":
            models["mlp_bias"].fill(0.0)
            models["linear"].fill(0.0)
            models["output"].fill(0.0)
        elif component_mode == "mlp_only":
            models["value"].fill(0.0)
        elif component_mode == "symmetric_value":
            models["value"] = 0.5 * (
                models["value"] + np.swapaxes(models["value"], -1, -2)
            )
        elif component_mode == "antisymmetric_value":
            models["value"] = 0.5 * (
                models["value"] - np.swapaxes(models["value"], -1, -2)
            )
        track_potential = int(selected[0]["family"]) == 1 and bool(
            np.all(np.abs(models["beta"]) < 1e-14)
        )
        original_steps = models["step_size"].astype(float).copy()
        if start_mode == "original":
            starts = []
            for record in selected:
                identity = record["identity"]
                key = (
                    int(record["family"]),
                    int(identity["n_tokens"]),
                    int(identity["source_model_index"]),
                    str(record["label"]),
                )
                angle = record.get("initial_angle", cohort_angles.get(key))
                if angle is None:
                    raise ValueError(f"missing original angle for {key}")
                starts.append(angle)
            angles = np.asarray(starts)[:, None, :]
        else:
            angles = np.asarray([record["final_angle"] for record in selected])[:, None, :]
        final_angles, metrics = evaluate_ode(
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
            track_potential,
            post_noise_time,
            noise_relaxations,
            anti_lock_noise,
            anti_lock_interval,
            anti_lock_rng,
        )
        for local, global_index in enumerate(indices):
            motion = float(metrics["motion_per_normalized_time"][local])
            field_rms = float(metrics["continuous_field_rms"][local])
            recurrence = float(metrics["recurrence_error"][local])
            lyapunov = float(metrics["lyapunov_per_normalized_time"][local])
            potential_gain = float(metrics["potential_gain"][local])
            minimum_potential_increment = float(
                metrics["minimum_potential_increment"][local]
            )
            output[global_index] = {
                "family": records[global_index]["family"],
                "label": records[global_index]["label"],
                "identity": records[global_index]["identity"],
                "model": records[global_index]["model"],
                "starting_small_step_trace": records[global_index]["trace"][-1],
                "metrics": {
                    "motion_per_normalized_time": motion,
                    "continuous_field_rms": field_rms,
                    "recurrence_error": recurrence,
                    "return_time_normalized": float(
                        metrics["return_time_normalized"][local]
                    ),
                    "gram_variation": float(metrics["gram_variation"][local]),
                    "gram_recurrence_error": float(
                        metrics["gram_recurrence_error"][local]
                    ),
                    "gram_return_time_normalized": float(
                        metrics["gram_return_time_normalized"][local]
                    ),
                    "score_kernel_variation": float(
                        metrics["score_kernel_variation"][local]
                    ),
                    "score_kernel_recurrence_error": float(
                        metrics["score_kernel_recurrence_error"][local]
                    ),
                    "score_kernel_return_time_normalized": float(
                        metrics["score_kernel_return_time_normalized"][local]
                    ),
                    "attention_weight_variation": float(
                        metrics["attention_weight_variation"][local]
                    ),
                    "attention_weight_recurrence_error": float(
                        metrics["attention_weight_recurrence_error"][local]
                    ),
                    "attention_weight_return_time_normalized": float(
                        metrics["attention_weight_return_time_normalized"][local]
                    ),
                    "mean_coherence": float(metrics["mean_coherence"][local]),
                    "lyapunov_per_normalized_time": lyapunov,
                    "potential_gain": (
                        potential_gain if np.isfinite(potential_gain) else None
                    ),
                    "minimum_potential_increment": (
                        minimum_potential_increment
                        if np.isfinite(minimum_potential_increment)
                        else None
                    ),
                    "fixed": field_rms < 1e-3 and motion < 1e-3,
                    "moving_recurrent": motion >= 1e-3 and recurrence < 3e-2,
                    "positive_lyapunov": lyapunov > 5e-3,
                },
                "final_angle": final_angles[local, 0].tolist(),
            }
    complete = [record for record in output if record is not None]
    summary: dict[str, dict[str, int]] = {}
    for record in complete:
        key = f"type{record['family']}_{record['label']}"
        entry = summary.setdefault(
            key, {"records": 0, "fixed": 0, "moving_recurrent": 0, "positive_lyapunov": 0}
        )
        entry["records"] += 1
        for metric in ("fixed", "moving_recurrent", "positive_lyapunov"):
            entry[metric] += int(bool(record["metrics"][metric]))
    return {
        "settings": {
            "dt": dt,
            "burn_time_normalized": burn_time,
            "sample_count": sample_count,
            "sample_spacing_normalized": sample_spacing,
            "lyapunov_time_normalized": lyapunov_time,
            "seed": seed,
            "start_mode": start_mode,
            "initial_noise": initial_noise,
            "selection": selection,
            "selected_n_tokens": selected_n_tokens,
            "selected_model_index": selected_model_index,
            "component_mode": component_mode,
            "post_noise_time_normalized": post_noise_time,
            "noise_relaxations": noise_relaxations,
            "anti_lock_noise": anti_lock_noise,
            "anti_lock_interval_normalized": anti_lock_interval,
        },
        "summary": summary,
        "records": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--burn-time", type=float, default=600.0)
    parser.add_argument("--sample-count", type=int, default=160)
    parser.add_argument("--sample-spacing", type=float, default=0.25)
    parser.add_argument("--lyapunov-time", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=260814201)
    parser.add_argument("--start-mode", choices=("final", "original"), default="final")
    parser.add_argument("--initial-noise", type=float, default=1e-7)
    parser.add_argument("--post-noise-time", type=float, default=200.0)
    parser.add_argument("--noise-relaxations", type=int, default=2)
    parser.add_argument("--anti-lock-noise", type=float, default=0.0)
    parser.add_argument("--anti-lock-interval", type=float, default=100.0)
    parser.add_argument(
        "--selection", choices=("all", "moving", "positive", "internal"), default="all"
    )
    parser.add_argument("--n-tokens", type=int)
    parser.add_argument("--source-model-index", type=int)
    parser.add_argument(
        "--component-mode",
        choices=(
            "full",
            "attention_only",
            "mlp_only",
            "symmetric_value",
            "antisymmetric_value",
        ),
        default="full",
    )
    parser.add_argument("--cohorts", nargs="*", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        inputs=args.inputs,
        dt=args.dt,
        burn_time=args.burn_time,
        sample_count=args.sample_count,
        sample_spacing=args.sample_spacing,
        lyapunov_time=args.lyapunov_time,
        seed=args.seed,
        start_mode=args.start_mode,
        cohort_inputs=args.cohorts,
        initial_noise=args.initial_noise,
        selection=args.selection,
        selected_n_tokens=args.n_tokens,
        selected_model_index=args.source_model_index,
        component_mode=args.component_mode,
        post_noise_time=args.post_noise_time,
        noise_relaxations=args.noise_relaxations,
        anti_lock_noise=args.anti_lock_noise,
        anti_lock_interval=args.anti_lock_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
