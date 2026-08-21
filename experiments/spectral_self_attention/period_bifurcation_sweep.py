"""Trace period-3 and period-4 windows around census examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import (
    classify_history,
    map_angles,
    wrap,
)


def batch_from_model(model: dict[str, object], steps: np.ndarray) -> dict[str, np.ndarray]:
    count = steps.size
    batch: dict[str, np.ndarray] = {}
    for key, value in model.items():
        array = np.asarray(value)
        if array.ndim == 0:
            batch[key] = np.full(count, array.item())
        else:
            batch[key] = np.repeat(array[None, ...], count, axis=0)
    batch["step_size"] = steps.copy()
    return batch


def contiguous_windows(rows: list[dict[str, object]], period: int) -> list[list[float]]:
    windows: list[list[float]] = []
    start: float | None = None
    previous_hit: float | None = None
    for row in rows:
        value = float(row["step_size"])
        hit = int(row[f"p{period}"]) > 0
        if hit and start is None:
            start = value
        if hit:
            previous_hit = value
        elif start is not None and previous_hit is not None:
            windows.append([start, previous_hit])
            start = None
            previous_hit = None
    if start is not None and previous_hit is not None:
        windows.append([start, previous_hit])
    return windows


def sweep_example(
    record: dict[str, object],
    period: int,
    rng: np.random.Generator,
    basins: int,
    burn_steps: int,
) -> dict[str, object]:
    model = record["model"]
    assert isinstance(model, dict)
    original_step = float(model["step_size"])
    coarse = np.linspace(0.02, 1.80, 120)
    local = np.linspace(max(0.005, original_step - 0.18), original_step + 0.18, 121)
    steps = np.unique(np.concatenate((coarse, local, [original_step])))
    models = batch_from_model(model, steps)
    n_tokens = int(record["n_tokens"])
    angles = rng.uniform(-np.pi, np.pi, size=(steps.size, basins, n_tokens))
    cycle_seed = np.asarray(record["cycle_tail"][0], dtype=float)
    seeded = min(24, basins)
    angles[:, :seeded, :] = wrap(
        cycle_seed[None, None, :]
        + rng.normal(scale=0.04, size=(steps.size, seeded, n_tokens))
    )
    for _ in range(burn_steps):
        angles = map_angles(angles, models)
    saved = []
    for _ in range(48):
        angles = map_angles(angles, models)
        saved.append(angles.copy())
    periods, _, rotations = classify_history(np.stack(saved))
    rows = []
    for index, step in enumerate(steps):
        row: dict[str, object] = {
            "step_size": float(step),
            "rotation": int(np.sum(rotations[index])),
            "unresolved": int(
                np.sum((periods[index] == 0) & ~rotations[index])
            ),
        }
        for candidate in range(1, 13):
            row[f"p{candidate}"] = int(np.sum(periods[index] == candidate))
        rows.append(row)
    return {
        "target_period": period,
        "n_tokens": n_tokens,
        "original_step_size": original_step,
        "basins_per_step": basins,
        "period3_windows": contiguous_windows(rows, 3),
        "period4_windows": contiguous_windows(rows, 4),
        "rows": rows,
    }


def run(inputs: list[Path], basins: int, burn_steps: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    families: dict[str, object] = {}
    for path in inputs:
        census = json.loads(path.read_text())
        examples: dict[str, object] = {}
        for period in (3, 4):
            name = f"p{period}"
            if name in census["examples"]:
                examples[name] = sweep_example(
                    census["examples"][name], period, rng, basins, burn_steps
                )
        families[str(census["family"])] = {
            "family_name": census["family_name"],
            "examples": examples,
        }
    return {
        "settings": {"basins": basins, "burn_steps": burn_steps, "seed": seed},
        "families": families,
    }


def merge_results(inputs: list[Path]) -> dict[str, object]:
    sources = [json.loads(path.read_text()) for path in inputs]
    families: dict[str, object] = {}
    for source in sources:
        families.update(source["families"])
    for family in families.values():
        for example in family["examples"].values():
            example["period3_windows"] = contiguous_windows(example["rows"], 3)
            example["period4_windows"] = contiguous_windows(example["rows"], 4)
    return {"settings": {"merged_from": [str(path) for path in inputs]}, "families": families}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--basins", type=int, default=192)
    parser.add_argument("--burn-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=260813111)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (
        merge_results(args.merge)
        if args.merge
        else run(args.inputs, args.basins, args.burn_steps, args.seed)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        family: {
            period: {
                "p3_windows": example["period3_windows"],
                "p4_windows": example["period4_windows"],
            }
            for period, example in data["examples"].items()
        }
        for family, data in result["families"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
