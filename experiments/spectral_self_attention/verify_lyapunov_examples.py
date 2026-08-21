"""Long-run confirmation of positive finite-time Lyapunov examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import (
    classify_history,
    map_angles,
)
from experiments.spectral_self_attention.lyapunov_census import largest_lyapunov


def batch_model(model: dict[str, object]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, value in model.items():
        array = np.asarray(value)
        output[key] = array.reshape((1,) + array.shape)
    return output


def run(inputs: list[Path], seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    families: dict[str, object] = {}
    for path in inputs:
        source = json.loads(path.read_text())
        verified = []
        for example in source["positive_examples"]:
            models = batch_model(example["model"])
            angles = np.asarray(example["angles"], dtype=float)[None, None, :]
            for _ in range(1200):
                angles = map_angles(angles, models)
            saved = []
            for _ in range(72):
                angles = map_angles(angles, models)
                saved.append(angles.copy())
            periods, _, rotations = classify_history(np.stack(saved))
            first = float(largest_lyapunov(angles, models, rng, 5000).item())
            second = float(largest_lyapunov(angles, models, rng, 5000).item())
            verified.append(
                {
                    "n_tokens": example["n_tokens"],
                    "initial_screen_exponent": example["exponent_per_layer"],
                    "long_exponents_per_layer": [first, second],
                    "robust_positive": min(first, second) > 0.005,
                    "short_period": int(periods.item()),
                    "rigid_rotation": bool(rotations.item()),
                    "model": example["model"],
                }
            )
        families[str(source["family"])] = {
            "family_name": source["family_name"],
            "tested": len(verified),
            "robust_positive": sum(bool(row["robust_positive"]) for row in verified),
            "examples": verified,
        }
    return {"seed": seed, "families": families}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=260813191)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.inputs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                family: {"tested": row["tested"], "robust": row["robust_positive"]}
                for family, row in result["families"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
