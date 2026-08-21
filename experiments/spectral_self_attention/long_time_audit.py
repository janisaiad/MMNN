"""Long-horizon checks for slow, metastable, and non-hyperbolic regimes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .simulator import diagnostics, integrate, random_sphere
from .taxonomy import SPECTRAL_CASES


@dataclass(frozen=True)
class LongCase:
    spectral_case: str
    beta: float
    n_tokens: int
    horizon: float


LONG_CASES = (
    LongCase("nd_simple", 0.03, 3, 1000.0),
    LongCase("nd_simple", 0.03, 5, 1000.0),
    LongCase("nd_simple", 0.10, 3, 500.0),
    LongCase("nd_flat_bottom", 0.03, 3, 500.0),
    LongCase("nd_simple", 8.0, 20, 500.0),
    LongCase("nd_flat_bottom", 8.0, 20, 500.0),
    LongCase("pd_flat_top", 8.0, 20, 500.0),
    LongCase("mixed_equal_extremes", 3.0, 20, 500.0),
    LongCase("mixed_equal_extremes", 8.0, 20, 500.0),
    LongCase("mixed_two_positive", 8.0, 20, 500.0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spectral_self_attention/long_time"),
    )
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=260426086)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    final_states = {}
    for case_index, long_case in enumerate(LONG_CASES):
        spectral_case = next(
            case for case in SPECTRAL_CASES if case.name == long_case.spectral_case
        )
        eigenvalues = np.asarray(spectral_case.eigenvalues)
        rng = np.random.default_rng(args.seed + 1009 * case_index)
        x0 = random_sphere(
            rng, args.trials, long_case.n_tokens, eigenvalues.size
        )
        checkpoint = 50.0
        save_every = max(1, int(round(checkpoint / args.dt)))
        trajectory = integrate(
            x0,
            eigenvalues,
            long_case.beta,
            t_final=long_case.horizon,
            dt=args.dt,
            save_every=save_every,
        )
        key = (
            f"{long_case.spectral_case}_beta{long_case.beta:g}"
            f"_n{long_case.n_tokens}"
        )
        final_states[key] = trajectory.states[-1]
        for time, state in zip(trajectory.times, trajectory.states, strict=True):
            diag = diagnostics(state, eigenvalues, long_case.beta)
            for trial in range(args.trials):
                rows.append(
                    {
                        "case": long_case.spectral_case,
                        "beta": long_case.beta,
                        "n_tokens": long_case.n_tokens,
                        "time": time,
                        "trial": trial,
                        "geometry": str(diag.geometry[trial]),
                        "speed": float(diag.speed[trial]),
                        "selected_group_mass": float(
                            diag.selected_group_mass[trial]
                        ),
                        "extreme_eigenspace_mass": float(
                            diag.extreme_eigenspace_mass[trial]
                        ),
                        "min_correlation": float(diag.min_correlation[trial]),
                        "max_correlation": float(diag.max_correlation[trial]),
                        "mean_abs_correlation": float(
                            diag.mean_abs_correlation[trial]
                        ),
                        "mean_vector_norm": float(diag.mean_vector_norm[trial]),
                        **{
                            f"mass_{mode}": float(diag.modal_masses[trial, mode])
                            for mode in range(eigenvalues.size)
                        },
                    }
                )

    pd.DataFrame(rows).to_csv(args.output_dir / "long_time_audit.csv", index=False)
    np.savez_compressed(args.output_dir / "final_states.npz", **final_states)
    print(
        {
            "cases": len(LONG_CASES),
            "trials": args.trials * len(LONG_CASES),
            "rows": len(rows),
        }
    )


if __name__ == "__main__":
    main()

