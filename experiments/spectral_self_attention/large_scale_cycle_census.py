"""Large-scale attractor census for serial Attention->quadratic-MLP blocks.

The script samples the four structural families from the MLP taxonomy and
classifies long trajectories as fixed points, primitive cycles of period 2--12,
rigid rotations, or unresolved dynamics.  All models in a batch are advanced in
parallel, as are their random initial conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FAMILY_NAMES = {
    1: "tied_attention_potential_mlp",
    2: "tied_attention_general_mlp",
    3: "untied_attention_potential_mlp",
    4: "untied_attention_general_mlp",
}
UNTIED_SUBTYPES = (
    "untied_symmetric",
    "symmetric_score_general_value",
    "general_score_symmetric_value",
    "fully_general",
)


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def wrap(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def symmetrize(matrices: np.ndarray) -> np.ndarray:
    return 0.5 * (matrices + np.swapaxes(matrices, -1, -2))


def unit_rms(matrices: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(matrices * matrices, axis=(-2, -1), keepdims=True))
    return matrices / np.maximum(rms, 1e-12)


def draw_log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def draw_models(
    rng: np.random.Generator,
    family: int,
    count: int,
    width_max: int = 8,
) -> dict[str, np.ndarray]:
    if family not in FAMILY_NAMES:
        raise ValueError(f"unknown family {family}")
    potential = family in (1, 3)
    tied = family in (1, 2)

    attention_scale = draw_log_uniform(rng, 0.08, 3.0, count)
    mlp_scale = draw_log_uniform(rng, 0.08, 3.0, count)
    step_size = np.empty(count)
    small_step = rng.random(count) < 0.25
    step_size[small_step] = draw_log_uniform(rng, 0.02, 0.30, int(small_step.sum()))
    step_size[~small_step] = rng.uniform(0.30, 1.60, size=int((~small_step).sum()))
    beta = draw_log_uniform(rng, 0.08, 12.0, count)
    beta[rng.random(count) < 0.08] = 0.0

    raw_score = unit_rms(rng.normal(size=(count, 2, 2)))
    raw_value = unit_rms(rng.normal(size=(count, 2, 2)))
    subtype_code = np.full(count, -1, dtype=int)
    if tied:
        value = symmetrize(raw_value)
        value = unit_rms(value) * attention_scale[:, None, None]
        score = value.copy()
    else:
        subtype_code = rng.integers(0, len(UNTIED_SUBTYPES), size=count)
        score = np.empty_like(raw_score)
        value = np.empty_like(raw_value)
        for code, subtype in enumerate(UNTIED_SUBTYPES):
            mask = subtype_code == code
            if not np.any(mask):
                continue
            selected_score = raw_score[mask]
            selected_value = raw_value[mask]
            if subtype in ("untied_symmetric", "symmetric_score_general_value"):
                selected_score = unit_rms(symmetrize(selected_score))
            if subtype in ("untied_symmetric", "general_score_symmetric_value"):
                selected_value = unit_rms(symmetrize(selected_value))
            score[mask] = selected_score * attention_scale[mask, None, None]
            value[mask] = selected_value * attention_scale[mask, None, None]

    widths = rng.choice(np.array([1, 2, 4, 8]), size=count)
    active = np.arange(width_max)[None, :] < widths[:, None]
    hidden = rng.normal(size=(count, width_max, 2))
    hidden /= np.maximum(np.linalg.norm(hidden, axis=-1, keepdims=True), 1e-12)
    hidden_bias = rng.normal(scale=0.4, size=(count, width_max))
    hidden_bias *= active
    mlp_bias = rng.normal(scale=0.35, size=(count, 2))
    raw_linear = rng.normal(size=(count, 2, 2))
    if potential:
        linear = symmetrize(raw_linear)
        coefficients = rng.normal(size=(count, width_max))
        coefficients *= active / np.sqrt(widths[:, None])
        output = np.swapaxes(hidden, -1, -2) * coefficients[:, None, :]
    else:
        linear = raw_linear
        output = rng.normal(size=(count, 2, width_max))
        output *= active[:, None, :] / np.sqrt(widths[:, None, None])

    mlp_bias *= mlp_scale[:, None]
    linear *= mlp_scale[:, None, None]
    output *= mlp_scale[:, None, None]
    return {
        "score": score,
        "value": value,
        "beta": beta,
        "step_size": step_size,
        "mlp_bias": mlp_bias,
        "linear": linear,
        "hidden": hidden,
        "hidden_bias": hidden_bias,
        "output": output,
        "width": widths,
        "subtype_code": subtype_code,
        "attention_scale": attention_scale,
        "mlp_scale": mlp_scale,
    }


def map_angles(angles: np.ndarray, models: dict[str, np.ndarray]) -> np.ndarray:
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

    step = models["step_size"][:, None, None]
    after_attention = normalize(tokens + step[..., None] * attention)
    hidden_values = np.einsum(
        "mrd,mbnd->mbnr", models["hidden"], after_attention, optimize=True
    )
    hidden_values += models["hidden_bias"][:, None, None, :]
    mlp = models["mlp_bias"][:, None, None, :]
    mlp = mlp + np.einsum(
        "mde,mbne->mbnd", models["linear"], after_attention, optimize=True
    )
    mlp = mlp + np.einsum(
        "mdr,mbnr->mbnd", models["output"], hidden_values**2, optimize=True
    )
    output = normalize(after_attention + step[..., None] * mlp)
    return np.arctan2(output[..., 1], output[..., 0])


def classify_history(
    history: np.ndarray,
    max_period: int = 12,
    tolerance: float = 2e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return primitive period, periodic residual, and rigid-rotation mask."""
    _, models, basins, _ = history.shape
    period = np.zeros((models, basins), dtype=np.int16)
    residual = np.full((models, basins), np.inf)
    for candidate in range(1, min(max_period, history.shape[0] - 1) + 1):
        error = np.max(
            np.abs(wrap(history[candidate:] - history[:-candidate])), axis=(0, 3)
        )
        newly_periodic = (period == 0) & (error < tolerance)
        period[newly_periodic] = candidate
        residual[newly_periodic] = error[newly_periodic]

    tokens = np.stack((np.cos(history), np.sin(history)), axis=-1)
    gram = np.einsum("tmbid,tmbjd->tmbij", tokens, tokens, optimize=True)
    shape_change = np.max(np.abs(gram[1:] - gram[:-1]), axis=(0, 3, 4))
    increments = wrap(history[1:] - history[:-1])
    mean_increment = np.mean(increments, axis=(0, 3))
    increment_spread = np.max(
        np.abs(increments - mean_increment[None, :, :, None]), axis=(0, 3)
    )
    rigid_rotation = (
        (period == 0)
        & (shape_change < 2e-5)
        & (increment_spread < 2e-5)
        & (np.abs(mean_increment) > 2e-5)
    )
    return period, residual, rigid_rotation


def parameter_bins(models: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    step = models["step_size"]
    beta = models["beta"]
    ratio = models["attention_scale"] / models["mlp_scale"]
    return {
        "step_bin": np.digitize(step, [0.10, 0.30, 0.60, 1.00]),
        "beta_bin": np.digitize(beta, [0.10, 0.50, 2.00, 6.00]),
        "ratio_bin": np.digitize(ratio, [0.25, 0.75, 1.50, 4.00]),
        "width": models["width"],
        "subtype_code": models["subtype_code"],
    }


def serializable_model(models: dict[str, np.ndarray], index: int) -> dict[str, object]:
    return {
        key: (float(value[index]) if value.ndim == 1 else value[index].tolist())
        for key, value in models.items()
    }


def update_group_counts(
    groups: dict[str, dict[str, dict[str, int]]],
    key: str,
    values: np.ndarray,
    periods: np.ndarray,
    rotations: np.ndarray,
) -> None:
    destination = groups.setdefault(key, {})
    for value in np.unique(values):
        mask = values == value
        label = str(int(value))
        entry = destination.setdefault(label, {"basins": 0})
        entry["basins"] += int(mask.sum() * periods.shape[1])
        for period in range(1, 13):
            name = f"p{period}"
            entry[name] = entry.get(name, 0) + int(np.sum(periods[mask] == period))
        entry["rotation"] = entry.get("rotation", 0) + int(np.sum(rotations[mask]))
        entry["unresolved"] = entry.get("unresolved", 0) + int(
            np.sum((periods[mask] == 0) & ~rotations[mask])
        )


def run_census(
    family: int,
    models_per_token_count: int,
    batch_models: int,
    basins: int,
    burn_steps: int,
    history_steps: int,
    seed: int,
    force_beta_zero: bool = False,
    force_attention_zero: bool = False,
    token_counts: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    totals = {"models": 0, "basins": 0, **{f"p{p}": 0 for p in range(1, 13)}}
    totals.update({"rotation": 0, "unresolved": 0})
    model_incidence = {f"p{p}": 0 for p in range(1, 13)}
    model_incidence.update({"rotation": 0, "unresolved": 0})
    by_tokens: dict[str, dict[str, int]] = {}
    grouped: dict[str, dict[str, dict[str, int]]] = {}
    examples: dict[str, dict[str, object]] = {}

    for n_tokens in token_counts:
        token_result = {"models": 0, "basins": 0, **{f"p{p}": 0 for p in range(1, 13)}}
        token_result.update({"rotation": 0, "unresolved": 0})
        for start in range(0, models_per_token_count, batch_models):
            count = min(batch_models, models_per_token_count - start)
            models = draw_models(rng, family, count)
            if force_beta_zero:
                models["beta"].fill(0.0)
            if force_attention_zero:
                models["score"].fill(0.0)
                models["value"].fill(0.0)
            angles = rng.uniform(-np.pi, np.pi, size=(count, basins, n_tokens))
            for _ in range(burn_steps):
                angles = map_angles(angles, models)
            saved = []
            for _ in range(history_steps):
                angles = map_angles(angles, models)
                saved.append(angles.copy())
            history = np.stack(saved)
            periods, residuals, rotations = classify_history(history)

            totals["models"] += count
            totals["basins"] += count * basins
            token_result["models"] += count
            token_result["basins"] += count * basins
            for period in range(1, 13):
                name = f"p{period}"
                hits = int(np.sum(periods == period))
                totals[name] += hits
                token_result[name] += hits
                model_incidence[name] += int(np.sum(np.any(periods == period, axis=1)))
                if hits and name not in examples:
                    model_index, basin_index = np.argwhere(periods == period)[0]
                    examples[name] = {
                        "n_tokens": n_tokens,
                        "model": serializable_model(models, int(model_index)),
                        "cycle_tail": history[-period:, model_index, basin_index].tolist(),
                        "residual": float(residuals[model_index, basin_index]),
                    }
            rotation_hits = int(np.sum(rotations))
            unresolved_hits = int(np.sum((periods == 0) & ~rotations))
            totals["rotation"] += rotation_hits
            totals["unresolved"] += unresolved_hits
            token_result["rotation"] += rotation_hits
            token_result["unresolved"] += unresolved_hits
            model_incidence["rotation"] += int(np.sum(np.any(rotations, axis=1)))
            model_incidence["unresolved"] += int(
                np.sum(np.any((periods == 0) & ~rotations, axis=1))
            )
            if rotation_hits and "rotation" not in examples:
                model_index, basin_index = np.argwhere(rotations)[0]
                examples["rotation"] = {
                    "n_tokens": n_tokens,
                    "model": serializable_model(models, int(model_index)),
                    "tail": history[-12:, model_index, basin_index].tolist(),
                }

            bins = parameter_bins(models)
            for key, values in bins.items():
                update_group_counts(grouped, key, values, periods, rotations)
        by_tokens[str(n_tokens)] = token_result

    return {
        "family": family,
        "family_name": FAMILY_NAMES[family],
        "settings": {
            "models_per_token_count": models_per_token_count,
            "basins": basins,
            "burn_steps": burn_steps,
            "history_steps": history_steps,
            "seed": seed,
            "force_beta_zero": force_beta_zero,
            "force_attention_zero": force_attention_zero,
            "token_counts": list(token_counts),
        },
        "totals": totals,
        "model_incidence": model_incidence,
        "by_tokens": by_tokens,
        "grouped": grouped,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--models-per-token-count", type=int, default=1024)
    parser.add_argument("--batch-models", type=int, default=32)
    parser.add_argument("--basins", type=int, default=96)
    parser.add_argument("--burn-steps", type=int, default=1400)
    parser.add_argument("--history-steps", type=int, default=36)
    parser.add_argument("--seed", type=int, default=260813001)
    parser.add_argument("--force-beta-zero", action="store_true")
    parser.add_argument("--force-attention-zero", action="store_true")
    parser.add_argument("--token-counts", default="1,2,3,4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_census(
        family=args.family,
        models_per_token_count=args.models_per_token_count,
        batch_models=args.batch_models,
        basins=args.basins,
        burn_steps=args.burn_steps,
        history_steps=args.history_steps,
        seed=args.seed + args.family,
        force_beta_zero=args.force_beta_zero,
        force_attention_zero=args.force_attention_zero,
        token_counts=tuple(int(value) for value in args.token_counts.split(",")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "family": result["family_name"],
        "totals": result["totals"],
        "model_incidence": result["model_incidence"],
    }, indent=2))


if __name__ == "__main__":
    main()
