"""Monte-Carlo root census of fixed configurations for the four families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import (
    FAMILY_NAMES,
    draw_models,
    serializable_model,
)
from experiments.spectral_self_attention.mlp_equilibrium_taxonomy import (
    discover_equilibria,
)
from experiments.spectral_self_attention.periodic_orbit_audit import block_from_record


def count_summary(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "models": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": int(np.max(array)),
        "nonzero_models": int(np.sum(array > 0)),
    }


def run(
    family: int,
    models_count: int,
    random_starts: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    models = draw_models(rng, family, models_count)
    by_tokens: dict[str, object] = {}
    for n_tokens in (1, 2, 3):
        total_counts: list[int] = []
        stable_counts: list[int] = []
        irregular_stable_counts: list[int] = []
        spiral_stable_counts: list[int] = []
        best: dict[str, object] | None = None
        for index in range(models_count):
            model = serializable_model(models, index)
            block = block_from_record({"model": model}, family)
            equilibria = discover_equilibria(
                block,
                n_tokens=n_tokens,
                rng=rng,
                random_starts=random_starts,
            )
            stable = [record for record in equilibria if record["stable"]]
            irregular = [record for record in stable if record["geometry"] == "irregular"]
            spirals = [record for record in stable if record["complex_linearization"]]
            total_counts.append(len(equilibria))
            stable_counts.append(len(stable))
            irregular_stable_counts.append(len(irregular))
            spiral_stable_counts.append(len(spirals))
            if best is None or len(stable) > int(best["stable_count"]):
                best = {
                    "stable_count": len(stable),
                    "total_count": len(equilibria),
                    "model": model,
                    "stable_equilibria": stable,
                }
        by_tokens[str(n_tokens)] = {
            "all_roots": count_summary(total_counts),
            "stable_roots": count_summary(stable_counts),
            "stable_irregular_roots": count_summary(irregular_stable_counts),
            "stable_spiral_roots": count_summary(spiral_stable_counts),
            "largest_stable_example": best,
        }
    return {
        "family": family,
        "family_name": FAMILY_NAMES[family],
        "settings": {
            "models": models_count,
            "random_starts_per_model_and_token_count": random_starts,
            "seed": seed,
            "warning": "Root counts for n>=2 are lower bounds from dense-grid plus random starts.",
        },
        "by_tokens": by_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=int, required=True, choices=(1, 2, 3, 4))
    parser.add_argument("--models", type=int, default=128)
    parser.add_argument("--random-starts", type=int, default=180)
    parser.add_argument("--seed", type=int, default=260813151)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.family, args.models, args.random_starts, args.seed + args.family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        tokens: {
            "stable": row["stable_roots"],
            "irregular": row["stable_irregular_roots"],
            "spiral": row["stable_spiral_roots"],
        }
        for tokens, row in result["by_tokens"].items()
    }
    print(json.dumps({"family": result["family_name"], "counts": compact}, indent=2))


if __name__ == "__main__":
    main()
