"""Harvest distinct p3, p4, and chaotic models for small-step continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import (
    FAMILY_NAMES,
    classify_history,
    draw_models,
    map_angles,
    serializable_model,
)
from experiments.spectral_self_attention.lyapunov_census import largest_lyapunov


def reservoir_add(
    reservoir: list[dict[str, object]],
    record: dict[str, object],
    seen: int,
    capacity: int,
    rng: np.random.Generator,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(record)
        return
    replacement = int(rng.integers(0, seen))
    if replacement < capacity:
        reservoir[replacement] = record


def harvest(
    family: int,
    n_tokens: int,
    models_count: int,
    batch_models: int,
    basins: int,
    burn_steps: int,
    lyapunov_steps: int,
    capacity_per_label: int,
    seed: int,
    force_beta_zero: bool = False,
    force_attention_zero: bool = False,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    records: dict[str, list[dict[str, object]]] = {"p3": [], "p4": [], "chaos": []}
    seen = {"p3": 0, "p4": 0, "chaos": 0}
    total_hits = {"p3_basins": 0, "p4_basins": 0, "chaos_basins": 0}
    model_hits = {"p3_models": 0, "p4_models": 0, "chaos_models": 0}
    for start in range(0, models_count, batch_models):
        count = min(batch_models, models_count - start)
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
        for _ in range(48):
            angles = map_angles(angles, models)
            saved.append(angles.copy())
        periods, residuals, rotations = classify_history(np.stack(saved))
        unresolved = (periods == 0) & ~rotations
        exponents = largest_lyapunov(
            angles.copy(), models, rng, steps=lyapunov_steps
        )
        chaos = unresolved & (exponents > 0.01)
        masks = {"p3": periods == 3, "p4": periods == 4, "chaos": chaos}
        for label, mask in masks.items():
            total_hits[f"{label}_basins"] += int(np.sum(mask))
            for model_index in np.flatnonzero(np.any(mask, axis=1)):
                basin_index = int(np.flatnonzero(mask[model_index])[0])
                model_hits[f"{label}_models"] += 1
                seen[label] += 1
                record = {
                    "family": family,
                    "n_tokens": n_tokens,
                    "subtype_code": int(models["subtype_code"][model_index]),
                    "model": serializable_model(models, int(model_index)),
                    "angle": angles[model_index, basin_index].tolist(),
                    "screen_period": int(periods[model_index, basin_index]),
                    "screen_periodic_residual": (
                        None
                        if not np.isfinite(residuals[model_index, basin_index])
                        else float(residuals[model_index, basin_index])
                    ),
                    "screen_lyapunov_per_layer": float(
                        exponents[model_index, basin_index]
                    ),
                    "source_model_index": start + int(model_index),
                }
                reservoir_add(
                    records[label], record, seen[label], capacity_per_label, rng
                )
    return {
        "family": family,
        "family_name": FAMILY_NAMES[family],
        "n_tokens": n_tokens,
        "settings": {
            "models": models_count,
            "basins": basins,
            "burn_steps": burn_steps,
            "lyapunov_steps": lyapunov_steps,
            "capacity_per_label": capacity_per_label,
            "seed": seed,
            "force_beta_zero": force_beta_zero,
            "force_attention_zero": force_attention_zero,
        },
        "total_hits": total_hits,
        "model_hits": model_hits,
        "stored": {label: len(values) for label, values in records.items()},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--n-tokens", type=int, required=True)
    parser.add_argument("--models", type=int, default=8192)
    parser.add_argument("--batch-models", type=int, default=32)
    parser.add_argument("--basins", type=int, default=16)
    parser.add_argument("--burn-steps", type=int, default=1500)
    parser.add_argument("--lyapunov-steps", type=int, default=500)
    parser.add_argument("--capacity-per-label", type=int, default=256)
    parser.add_argument("--seed", type=int, default=260814001)
    parser.add_argument("--force-beta-zero", action="store_true")
    parser.add_argument("--force-attention-zero", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.n_tokens < 1:
        parser.error("--n-tokens must be positive")
    result = harvest(
        family=args.family,
        n_tokens=args.n_tokens,
        models_count=args.models,
        batch_models=args.batch_models,
        basins=args.basins,
        burn_steps=args.burn_steps,
        lyapunov_steps=args.lyapunov_steps,
        capacity_per_label=args.capacity_per_label,
        seed=args.seed + 100 * args.family + args.n_tokens,
        force_beta_zero=args.force_beta_zero,
        force_attention_zero=args.force_attention_zero,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "family": result["family_name"],
                "n_tokens": result["n_tokens"],
                "model_hits": result["model_hits"],
                "stored": result["stored"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
