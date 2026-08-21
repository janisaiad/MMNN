"""Small-system numerical catalogue supporting the exact equilibrium theorem."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .equilibrium_catalogue import find_planar_equilibria


CASES = (
    ("indefinite_n2", (2.0, -3.0), 1.5, 2, 8_000),
    ("indefinite_n3", (2.0, -3.0), 1.5, 3, 12_000),
    ("negative_n2", (-0.4, -4.0), 0.03, 2, 8_000),
    ("negative_n3", (-0.4, -4.0), 0.03, 3, 12_000),
    ("positive_n2", (3.0, 2.0), 1.5, 2, 8_000),
    ("positive_n3", (3.0, 2.0), 1.5, 3, 12_000),
)


def _run(payload: tuple) -> tuple[str, int, list]:
    name, eigenvalues, beta, n_tokens, starts, seed = payload
    equilibria = find_planar_equilibria(
        eigenvalues,
        beta,
        n_tokens,
        random_starts=starts,
        seed=seed,
    )
    return name, seed, equilibria


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spectral_self_attention/equilibria"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    for case_index, case in enumerate(CASES):
        for seed_index in range(args.seeds):
            payloads.append((*case, 260426100 + 1009 * case_index + seed_index))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        outputs = list(executor.map(_run, payloads))

    rows = []
    key_sets: dict[str, list[set[tuple[float, ...]]]] = {}
    for name, seed, equilibria in outputs:
        keys = set()
        for equilibrium in equilibria:
            key = tuple(float(value) for value in equilibrium.angles)
            keys.add(key)
            rows.append(
                {
                    "case": name,
                    "seed": seed,
                    "angles": json.dumps(key),
                    "cluster_count": equilibrium.cluster_count,
                    "max_linear_rate": equilibrium.max_linear_rate,
                    "stable": equilibrium.stable,
                    "residual": equilibrium.residual,
                }
            )
        key_sets.setdefault(name, []).append(keys)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "planar_small_systems.csv", index=False)
    saturation = []
    for name, sets in key_sets.items():
        union = set.union(*sets)
        intersection = set.intersection(*sets)
        saturation.append(
            {
                "case": name,
                "runs": len(sets),
                "union_count": len(union),
                "common_count": len(intersection),
                "all_runs_identical": all(values == sets[0] for values in sets[1:]),
            }
        )
    saturation_frame = pd.DataFrame(saturation)
    saturation_frame.to_csv(args.output_dir / "search_saturation.csv", index=False)
    print(saturation_frame.to_string(index=False))


if __name__ == "__main__":
    main()

