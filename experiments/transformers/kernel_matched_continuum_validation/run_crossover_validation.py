#!/usr/bin/env python3
"""Long-context crossover against dense Cholesky and exact Woodbury.

This benchmark isolates the regime claimed by the proposed inference method:
one fixed elliptic context, many observation tokens, low effective posterior
rank, and one or a few queries.  It does not hide the regime where Woodbury is
faster.  Raw timing samples and bootstrap median intervals are retained.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import check_record, save_csv, save_json
from .run_pde_validation import (
    assemble_context,
    estimate_spectral_norm,
    kernel_metric_builder,
    ritz_effective_spectrum,
    solve_hb,
    solver_errors,
    sync,
    woodbury_solve,
)


def timed_samples(
    function: Callable[[], Any],
    device: torch.device,
    repeats: int,
    warmups: int,
) -> tuple[Any, list[float]]:
    result = None
    for _ in range(warmups):
        result = function()
    sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        sync(device)
        samples.append((time.perf_counter_ns() - start) * 1e-6)
    return result, samples


def timing_summary(samples: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.median(
        array[rng.integers(0, array.size, size=(5000, array.size))], axis=1
    )
    return {
        "median_ms": float(np.median(array)),
        "median_ci_low_ms": float(np.quantile(bootstrap, 0.025)),
        "median_ci_high_ms": float(np.quantile(bootstrap, 0.975)),
        "q25_ms": float(np.quantile(array, 0.25)),
        "q75_ms": float(np.quantile(array, 0.75)),
        "minimum_ms": float(np.min(array)),
    }


def select_columns(maximum: int, count: int) -> np.ndarray:
    if count == maximum:
        return np.arange(maximum)
    return np.unique(np.linspace(0, maximum - 1, count, dtype=np.int64))


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.profile == "smoke":
        side = 48
        context_sizes = [192, 384, 768]
        repeats = 5
        dense_repeats = 3
    else:
        side = 128
        context_sizes = [256, 512, 1024, 2048, 4096]
        repeats = 11
        dense_repeats = 5
    if args.side:
        side = args.side
    if args.context_sizes:
        context_sizes = sorted(
            {int(value) for value in args.context_sizes.split(",") if value.strip()}
        )
    maximum_context = max(context_sizes)
    latent_rank = 8 if args.profile == "smoke" else 24
    head_rank = latent_rank
    print(
        f"assembling long elliptic context N={side * side}, m_max={maximum_context}",
        flush=True,
    )
    context = assemble_context(
        side=side,
        sensor_count=maximum_context,
        latent_rank=latent_rank,
        seed=args.seed,
        floor_amplitude=0.018,
        target_data_singular=30.0,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 101)
    maximum_queries = 64
    rhs_all = torch.randn(
        context.dimension,
        maximum_queries,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for context_size in context_sizes:
        print(f"benchmarking m={context_size}", flush=True)
        columns = select_columns(maximum_context, context_size)
        sensitivity_np = np.ascontiguousarray(context.sensitivity[:, columns])
        sensitivity_np *= 30.0 / estimate_spectral_norm(
            sensitivity_np, args.seed + context_size
        )
        sensitivity = torch.as_tensor(sensitivity_np, device=device, dtype=dtype)
        sensor_coordinates = torch.as_tensor(
            context.sensor_coordinates[columns], device=device, dtype=dtype
        )
        identity_context = torch.eye(context_size, device=device, dtype=dtype)

        def build_woodbury() -> torch.Tensor:
            return torch.linalg.cholesky(
                identity_context + sensitivity.T @ sensitivity
            )

        woodbury_cholesky, samples = timed_samples(
            build_woodbury, device, repeats=repeats, warmups=2
        )
        woodbury_setup = timing_summary(samples, args.seed + context_size)
        for index, value in enumerate(samples):
            raw_rows.append(
                {
                    "context_size": context_size,
                    "queries": 0,
                    "method": "woodbury_setup",
                    "repeat": index,
                    "milliseconds": value,
                }
            )

        def build_kernel():
            return kernel_metric_builder(
                sensitivity,
                sensor_coordinates,
                head_rank,
                length_scale=0.30,
                refinement_steps=1,
            )

        kernel_metric, samples = timed_samples(
            build_kernel, device, repeats=repeats, warmups=2
        )
        kernel_setup = timing_summary(samples, args.seed + 2 * context_size)
        for index, value in enumerate(samples):
            raw_rows.append(
                {
                    "context_size": context_size,
                    "queries": 0,
                    "method": "kernel_setup",
                    "repeat": index,
                    "milliseconds": value,
                }
            )

        gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(
            sensitivity.T @ sensitivity
        )
        mu, ell, condition = ritz_effective_spectrum(
            gram_eigenvalues,
            gram_eigenvectors,
            sensitivity,
            kernel_metric,
        )
        contraction = (math.sqrt(ell / mu) - 1.0) / (
            math.sqrt(ell / mu) + 1.0
        )
        kernel_depth = 1
        while (
            (1.0 + kernel_depth * (1.0 + contraction))
            * contraction**kernel_depth
            > 0.5 * args.tolerance
        ):
            kernel_depth += 1
        kernel_depth = max(4, kernel_depth)
        spectrum_rows.append(
            {
                "context_size": context_size,
                "minimum": mu,
                "maximum": ell,
                "condition": condition,
                "identity_condition": 1.0 + float(gram_eigenvalues[-1].item()),
                "effective_rank": float(
                    (
                        gram_eigenvalues.sum() ** 2
                        / torch.sum(gram_eigenvalues**2).clamp_min(1e-30)
                    ).item()
                ),
            }
        )
        query_counts = [1, 8, 64]
        for query_count in query_counts:
            rhs = rhs_all[:, :query_count]
            exact = woodbury_solve(sensitivity, woodbury_cholesky, rhs)

            def kernel_solve() -> torch.Tensor:
                return solve_hb(
                    sensitivity,
                    rhs,
                    kernel_metric,
                    depth=kernel_depth,
                    mu=max(1e-8, 0.999 * mu),
                    ell=1.001 * ell,
                )

            kernel_solution, kernel_samples = timed_samples(
                kernel_solve, device, repeats=repeats, warmups=3
            )
            kernel_timing = timing_summary(
                kernel_samples, args.seed + 3 * context_size + query_count
            )

            def woodbury_cached() -> torch.Tensor:
                return woodbury_solve(sensitivity, woodbury_cholesky, rhs)

            woodbury_solution, woodbury_samples = timed_samples(
                woodbury_cached, device, repeats=repeats, warmups=3
            )
            woodbury_timing = timing_summary(
                woodbury_samples, args.seed + 4 * context_size + query_count
            )
            for method, timing, setup, solution in (
                ("kernel_hb", kernel_timing, kernel_setup, kernel_solution),
                (
                    "woodbury_exact",
                    woodbury_timing,
                    woodbury_setup,
                    woodbury_solution,
                ),
            ):
                errors = solver_errors(sensitivity, rhs, solution, exact)
                rows.append(
                    {
                        "dimension": context.dimension,
                        "context_size": context_size,
                        "queries": query_count,
                        "method": method,
                        "depth": kernel_depth if method == "kernel_hb" else 1,
                        "setup_median_ms": setup["median_ms"],
                        "setup_ci_low_ms": setup["median_ci_low_ms"],
                        "setup_ci_high_ms": setup["median_ci_high_ms"],
                        "solve_median_ms": timing["median_ms"],
                        "solve_ci_low_ms": timing["median_ci_low_ms"],
                        "solve_ci_high_ms": timing["median_ci_high_ms"],
                        "total_median_ms": setup["median_ms"] + timing["median_ms"],
                        "total_ci_low_ms": setup["median_ci_low_ms"]
                        + timing["median_ci_low_ms"],
                        "total_ci_high_ms": setup["median_ci_high_ms"]
                        + timing["median_ci_high_ms"],
                        **errors,
                    }
                )
            for method, values in (
                ("kernel_hb_solve", kernel_samples),
                ("woodbury_cached_solve", woodbury_samples),
            ):
                for index, value in enumerate(values):
                    raw_rows.append(
                        {
                            "context_size": context_size,
                            "queries": query_count,
                            "method": method,
                            "repeat": index,
                            "milliseconds": value,
                        }
                    )

    # Dense inversion is measured once at the largest context because its cost
    # is dominated by the N x N posterior matrix, not by query count.
    largest_columns = select_columns(maximum_context, context_sizes[-1])
    sensitivity_np = np.ascontiguousarray(context.sensitivity[:, largest_columns])
    sensitivity_np *= 30.0 / estimate_spectral_norm(
        sensitivity_np, args.seed + 991
    )
    sensitivity = torch.as_tensor(sensitivity_np, device=device, dtype=dtype)
    dense_identity = torch.eye(context.dimension, device=device, dtype=dtype)

    def build_dense() -> torch.Tensor:
        return torch.linalg.cholesky(
            dense_identity + sensitivity @ sensitivity.T
        )

    dense_cholesky, dense_samples = timed_samples(
        build_dense, device, repeats=dense_repeats, warmups=1
    )
    dense_setup = timing_summary(dense_samples, args.seed + 5001)
    for index, value in enumerate(dense_samples):
        raw_rows.append(
            {
                "context_size": context_sizes[-1],
                "queries": 0,
                "method": "dense_setup",
                "repeat": index,
                "milliseconds": value,
            }
        )
    largest_woodbury = torch.linalg.cholesky(
        torch.eye(context_sizes[-1], device=device, dtype=dtype)
        + sensitivity.T @ sensitivity
    )
    for query_count in (1, 8, 64):
        rhs = rhs_all[:, :query_count]
        exact = woodbury_solve(sensitivity, largest_woodbury, rhs)

        def dense_solve() -> torch.Tensor:
            return torch.cholesky_solve(rhs, dense_cholesky)

        dense_solution, dense_solve_samples = timed_samples(
            dense_solve, device, repeats=repeats, warmups=3
        )
        dense_solve_timing = timing_summary(
            dense_solve_samples, args.seed + 6001 + query_count
        )
        rows.append(
            {
                "dimension": context.dimension,
                "context_size": context_sizes[-1],
                "queries": query_count,
                "method": "dense_cholesky",
                "depth": 1,
                "setup_median_ms": dense_setup["median_ms"],
                "setup_ci_low_ms": dense_setup["median_ci_low_ms"],
                "setup_ci_high_ms": dense_setup["median_ci_high_ms"],
                "solve_median_ms": dense_solve_timing["median_ms"],
                "solve_ci_low_ms": dense_solve_timing["median_ci_low_ms"],
                "solve_ci_high_ms": dense_solve_timing["median_ci_high_ms"],
                "total_median_ms": dense_setup["median_ms"]
                + dense_solve_timing["median_ms"],
                "total_ci_low_ms": dense_setup["median_ci_low_ms"]
                + dense_solve_timing["median_ci_low_ms"],
                "total_ci_high_ms": dense_setup["median_ci_high_ms"]
                + dense_solve_timing["median_ci_high_ms"],
                **solver_errors(sensitivity, rhs, dense_solution, exact),
            }
        )
        for index, value in enumerate(dense_solve_samples):
            raw_rows.append(
                {
                    "context_size": context_sizes[-1],
                    "queries": query_count,
                    "method": "dense_cached_solve",
                    "repeat": index,
                    "milliseconds": value,
                }
            )

    checks: list[dict[str, Any]] = []
    maximum_kernel_residual = max(
        row["relative_residual_max"] for row in rows if row["method"] == "kernel_hb"
    )
    checks.append(
        check_record(
            "long_context_kernel_accuracy",
            maximum_kernel_residual <= args.tolerance,
            maximum_kernel_residual,
            "maximum kernel-HB residual <= target tolerance",
            target=args.tolerance,
        )
    )
    q1_rows = [row for row in rows if row["queries"] == 1]
    woodbury_crossovers = []
    for context_size in context_sizes:
        kernel = next(
            row
            for row in q1_rows
            if row["context_size"] == context_size and row["method"] == "kernel_hb"
        )
        woodbury = next(
            row
            for row in q1_rows
            if row["context_size"] == context_size
            and row["method"] == "woodbury_exact"
        )
        if kernel["total_ci_high_ms"] < woodbury["total_ci_low_ms"]:
            woodbury_crossovers.append(context_size)
    checks.append(
        check_record(
            "statistically_separated_woodbury_crossover",
            bool(woodbury_crossovers),
            min(woodbury_crossovers) if woodbury_crossovers else -1,
            "kernel upper 95% bootstrap interval < Woodbury lower interval for one query",
            crossover_contexts=woodbury_crossovers,
        )
    )
    maximum_context = context_sizes[-1]
    kernel_largest = next(
        row
        for row in q1_rows
        if row["context_size"] == maximum_context and row["method"] == "kernel_hb"
    )
    dense_largest = next(
        row
        for row in q1_rows
        if row["context_size"] == maximum_context and row["method"] == "dense_cholesky"
    )
    checks.append(
        check_record(
            "statistically_separated_dense_crossover",
            kernel_largest["total_ci_high_ms"] < dense_largest["total_ci_low_ms"],
            dense_largest["total_median_ms"] / kernel_largest["total_median_ms"],
            "dense lower 95% interval > kernel upper interval at largest context",
            kernel_total=kernel_largest["total_median_ms"],
            dense_total=dense_largest["total_median_ms"],
        )
    )
    checks.append(
        check_record(
            "long_context_low_effective_rank",
            3.0 <= max(row["effective_rank"] for row in spectrum_rows) < 0.5 * head_rank,
            max(row["effective_rank"] for row in spectrum_rows),
            "effective rank is nontrivial and remains below one half of head rank",
            head_rank=head_rank,
        )
    )

    payload = {
        "profile": args.profile,
        "device": str(device),
        "dtype": str(dtype),
        "dimension": context.dimension,
        "side": side,
        "latent_rank": latent_rank,
        "head_rank": head_rank,
        "context_sizes": context_sizes,
        "common_context_timings": context.common_timings,
        "rows": rows,
        "spectrum_rows": spectrum_rows,
        "checks": checks,
        "passed": sum(bool(check["passed"]) for check in checks),
        "total": len(checks),
    }
    save_csv(args.outdir / "crossover_timings.csv", rows)
    save_csv(args.outdir / "crossover_raw_timings.csv", raw_rows)
    save_csv(args.outdir / "crossover_spectra.csv", spectrum_rows)
    save_json(args.outdir / "summary.json", payload)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for method in ("kernel_hb", "woodbury_exact"):
        selected = [
            row for row in rows if row["queries"] == 1 and row["method"] == method
        ]
        axes[0].loglog(
            [row["context_size"] for row in selected],
            [row["total_median_ms"] for row in selected],
            "o-",
            label=method,
        )
        axes[0].fill_between(
            [row["context_size"] for row in selected],
            [row["total_ci_low_ms"] for row in selected],
            [row["total_ci_high_ms"] for row in selected],
            alpha=0.18,
        )
    axes[0].set(title="One-query crossover", xlabel="context tokens", ylabel="setup + solve (ms)")
    for method in ("kernel_hb", "woodbury_exact", "dense_cholesky"):
        selected = [
            row
            for row in rows
            if row["context_size"] == context_sizes[-1] and row["method"] == method
        ]
        axes[1].loglog(
            [row["queries"] for row in selected],
            [row["total_median_ms"] for row in selected],
            "o-",
            label=method,
        )
    axes[1].set(title=f"Amortization at m={context_sizes[-1]}", xlabel="queries", ylabel="setup + solve (ms)")
    for axis in axes:
        axis.grid(which="both", alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.outdir / "long_context_crossover.png", dpi=190, bbox_inches="tight")
    plt.close(figure)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--side", type=int, default=0)
    parser.add_argument("--context-sizes", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=84001)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = run(args)
    payload["elapsed_seconds"] = time.time() - started
    save_json(args.outdir / "summary.json", payload)
    print(
        f"crossover validation complete: {payload['passed']}/{payload['total']} checks passed; "
        f"summary={args.outdir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
