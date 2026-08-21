"""Monte-Carlo atlas over all qualitative eigenvalue patterns.

Usage:
    uv run python -m experiments.spectral_self_attention.run_sweep --profile pilot
    uv run python -m experiments.spectral_self_attention.run_sweep --profile full
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .simulator import (
    diagnostics,
    eigenspace_groups,
    integrate,
    is_linearly_stable_pure_mode,
    pure_mode_linear_rates,
    pure_mode_tangent_jacobian,
    random_sphere,
)
from .taxonomy import SPECTRAL_CASES, SpectralCase


PROFILES = {
    "pilot": {
        "betas": (0.1, 0.5, 1.5),
        "token_counts": (2, 3, 4, 7),
        "trials": 24,
        "t_final": 20.0,
        "dt": 0.025,
    },
    "full": {
        "betas": (0.03, 0.1, 0.3, 0.7, 1.5, 3.0, 8.0),
        "token_counts": (2, 3, 4, 5, 8, 20),
        "trials": 64,
        "t_final": 50.0,
        "dt": 0.04,
    },
}


def stable_catalogue(case: SpectralCase, beta: float, n_tokens: int) -> list[dict]:
    eigenvalues = np.asarray(case.eigenvalues)
    catalogue = []
    for mode, eigenvalue in enumerate(eigenvalues):
        for n_plus in range(n_tokens + 1):
            n_minus = n_tokens - n_plus
            rates = pure_mode_linear_rates(
                eigenvalues, beta, mode, n_plus, n_minus
            )
            max_rate = float(np.max(rates)) if rates.size else 0.0
            stable = is_linearly_stable_pure_mode(
                eigenvalues, beta, mode, n_plus, n_minus
            )
            catalogue.append(
                {
                    "case": case.name,
                    "family": case.family,
                    "beta": beta,
                    "n_tokens": n_tokens,
                    "mode": mode,
                    "eigenvalue": eigenvalue,
                    "n_plus": n_plus,
                    "n_minus": n_minus,
                    "geometry": "homogeneous"
                    if n_plus in (0, n_tokens)
                    else "sign_split",
                    "max_linear_rate": max_rate,
                    "linearly_stable": stable,
                    "linearly_neutral": abs(max_rate) <= 1e-10,
                }
            )
    return catalogue


def random_rows(
    case: SpectralCase,
    beta: float,
    n_tokens: int,
    trials: int,
    t_final: float,
    dt: float,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    eigenvalues = np.asarray(case.eigenvalues)
    x0 = random_sphere(rng, trials, n_tokens, eigenvalues.size)
    trajectory = integrate(
        x0, eigenvalues, beta, t_final=t_final, dt=dt, save_every=None
    )
    final = trajectory.states[-1]
    diag = diagnostics(final, eigenvalues, beta)
    groups = eigenspace_groups(eigenvalues)
    group_values = np.asarray([eigenvalues[group[0]] for group in groups])
    energy_gain = trajectory.energies[-1] - trajectory.energies[0]

    rows = []
    for trial in range(trials):
        rows.append(
            {
                "case": case.name,
                "family": case.family,
                "beta": beta,
                "n_tokens": n_tokens,
                "trial": trial,
                "seed": seed,
                "geometry": str(diag.geometry[trial]),
                "selected_eigenvalue": float(group_values[diag.selected_group[trial]]),
                "selected_group_mass": float(diag.selected_group_mass[trial]),
                "speed": float(diag.speed[trial]),
                "energy_gain": float(energy_gain[trial]),
                "min_correlation": float(diag.min_correlation[trial]),
                "max_correlation": float(diag.max_correlation[trial]),
                "mean_abs_correlation": float(diag.mean_abs_correlation[trial]),
                "mean_vector_norm": float(diag.mean_vector_norm[trial]),
                "extreme_eigenspace_mass": float(
                    diag.extreme_eigenspace_mass[trial]
                ),
                **{
                    f"mass_{mode}": float(diag.modal_masses[trial, mode])
                    for mode in range(eigenvalues.size)
                },
            }
        )
    summary = {
        "case": case.name,
        "beta": beta,
        "n_tokens": n_tokens,
        "geometry_counts": dict(Counter(diag.geometry.tolist())),
        "selected_eigenvalue_counts": {
            str(value): int(np.sum(group_values[diag.selected_group] == value))
            for value in group_values
        },
        "median_speed": float(np.median(diag.speed)),
        "max_speed": float(np.max(diag.speed)),
    }
    return rows, summary


def local_validation_rows(
    case: SpectralCase,
    beta: float,
    n_tokens: int,
    dt: float,
) -> list[dict]:
    """Compare exact block rates with an independent finite-difference Jacobian."""
    eigenvalues = np.asarray(case.eigenvalues)
    rows = []
    for mode, eigenvalue in enumerate(eigenvalues):
        split_sizes = {0, n_tokens, n_tokens // 2, (n_tokens + 1) // 2, 1}
        for n_plus in sorted(size for size in split_sizes if 0 <= size <= n_tokens):
            n_minus = n_tokens - n_plus
            rates = pure_mode_linear_rates(
                eigenvalues, beta, mode, n_plus, n_minus
            )
            predicted_rate = float(np.max(rates)) if rates.size else 0.0
            numeric_jacobian = pure_mode_tangent_jacobian(
                eigenvalues, beta, mode, n_plus, n_minus
            )
            numeric_rates = np.linalg.eigvals(numeric_jacobian).real
            numeric_rate = float(np.max(numeric_rates))
            sorted_rate_error = float(
                np.max(np.abs(np.sort(rates) - np.sort(numeric_rates)))
            )
            rows.append(
                {
                    "case": case.name,
                    "family": case.family,
                    "beta": beta,
                    "n_tokens": n_tokens,
                    "mode": mode,
                    "eigenvalue": eigenvalue,
                    "n_plus": n_plus,
                    "n_minus": n_minus,
                    "max_linear_rate": predicted_rate,
                    "predicted_stable": predicted_rate < -1e-10,
                    "predicted_neutral": abs(predicted_rate) <= 1e-10,
                    "numeric_max_rate": numeric_rate,
                    "numeric_stable": numeric_rate < -1e-8,
                    "max_spectrum_error": sorted_rate_error,
                    "dt_reserved_for_nonlinear_checks": dt,
                }
            )
    return rows


def convergence_audit(output_dir: Path, seed: int) -> list[dict]:
    """Time-step and horizon audit on stiff and slow representative cases."""
    selected = (
        next(case for case in SPECTRAL_CASES if case.name == "mixed_two_positive"),
        next(case for case in SPECTRAL_CASES if case.name == "nd_simple"),
        next(case for case in SPECTRAL_CASES if case.name == "mixed_equal_extremes"),
    )
    rows = []
    for case_index, case in enumerate(selected):
        rng = np.random.default_rng(seed + case_index)
        eigenvalues = np.asarray(case.eigenvalues)
        x0 = random_sphere(rng, 12, 5, eigenvalues.size)
        reference = integrate(
            x0, eigenvalues, 1.5, t_final=80.0, dt=0.005, save_every=None
        ).states[-1]
        reference_gram = np.einsum("bid,bjd->bij", reference, reference)
        for dt in (0.04, 0.02, 0.01):
            for horizon in (20.0, 40.0, 80.0):
                final = integrate(
                    x0,
                    eigenvalues,
                    1.5,
                    t_final=horizon,
                    dt=dt,
                    save_every=None,
                ).states[-1]
                gram = np.einsum("bid,bjd->bij", final, final)
                rows.append(
                    {
                        "case": case.name,
                        "dt": dt,
                        "horizon": horizon,
                        "median_gram_error": float(
                            np.median(np.linalg.norm(gram - reference_gram, axis=(1, 2)))
                        ),
                        "median_speed": float(
                            np.median(diagnostics(final, eigenvalues, 1.5).speed)
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "convergence_audit.csv", index=False)
    return rows


def execute_job(payload: tuple) -> tuple[list[dict], list[dict], dict, list[dict]]:
    """Pickle-friendly unit of work for deterministic process parallelism."""
    case, beta, n_tokens, config, seed = payload
    catalogue = stable_catalogue(case, beta, n_tokens)
    rows, summary = random_rows(
        case,
        beta,
        n_tokens,
        config["trials"],
        config["t_final"],
        config["dt"],
        seed,
    )
    local = local_validation_rows(case, beta, n_tokens, config["dt"])
    return catalogue, rows, summary, local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="pilot")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spectral_self_attention"),
    )
    parser.add_argument("--seed", type=int, default=260426085)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = PROFILES[args.profile]
    output_dir = args.output_dir / args.profile
    output_dir.mkdir(parents=True, exist_ok=True)

    random_results: list[dict] = []
    summaries: list[dict] = []
    catalogue: list[dict] = []
    local_results: list[dict] = []
    jobs = []
    for case in SPECTRAL_CASES:
        for beta in config["betas"]:
            for n_tokens in config["token_counts"]:
                jobs.append(
                    (
                        case,
                        beta,
                        n_tokens,
                        config,
                        args.seed + 1009 * len(jobs),
                    )
                )
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers == 1:
        results = map(execute_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(execute_job, jobs)
    try:
        for job_catalogue, rows, summary, local in results:
            catalogue.extend(job_catalogue)
            random_results.extend(rows)
            summaries.append(summary)
            local_results.extend(local)
    finally:
        if args.workers > 1:
            executor.shutdown()

    pd.DataFrame(random_results).to_csv(output_dir / "random_trials.csv", index=False)
    pd.DataFrame(catalogue).to_csv(output_dir / "pure_mode_stability.csv", index=False)
    pd.DataFrame(local_results).to_csv(output_dir / "local_validation.csv", index=False)
    audit = convergence_audit(output_dir, args.seed + 999_999)

    metadata = {
        "profile": args.profile,
        "config": config,
        "seed": args.seed,
        "spectral_cases": [asdict(case) for case in SPECTRAL_CASES],
        "jobs": len(jobs),
        "random_trajectories": len(random_results),
        "local_validations": len(local_results),
        "summaries": summaries,
        "convergence_audit_rows": len(audit),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                key: metadata[key]
                for key in (
                    "profile",
                    "jobs",
                    "random_trajectories",
                    "local_validations",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
