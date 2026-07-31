#!/usr/bin/env python3
"""Fair H100 benchmark for one operator and multiple right-hand sides.

Both learned and classical setup costs are exposed.  The matrix-free head is
built once and reused across all right-hand sides; the dense baseline likewise
forms and factorizes the normal matrix once before its cached triangular solve.
The benchmark measures latency only.  Fixed conservative HB/Chebyshev
endpoints avoid inserting an oracle eigendecomposition into the timed path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from .first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from .structured_one_head_heavyball import (
        EquivariantMatrixFreeNystromPreconditioner,
    )
except ImportError:
    from first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from structured_one_head_heavyball import (
        EquivariantMatrixFreeNystromPreconditioner,
    )


def parse_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value]


def benchmark(function, repeats: int, device: torch.device) -> dict[str, float]:
    for _ in range(5):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(1000.0 * (time.perf_counter() - start))
    ordered = sorted(timings)
    return {
        "median_ms": statistics.median(ordered),
        "q25_ms": ordered[len(ordered) // 4],
        "q75_ms": ordered[(3 * len(ordered)) // 4],
    }


def run(args) -> dict:
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(args.seed)
    rows = []
    for dimension in parse_ints(args.dimensions):
        equation_count = args.equation_ratio * dimension
        slots = min(args.slots, dimension - 1)
        equations = torch.randn(
            1,
            equation_count,
            dimension,
            device=device,
        ) / math.sqrt(equation_count)
        identity = torch.eye(dimension, device=device).unsqueeze(0)
        head = EquivariantMatrixFreeNystromPreconditioner(
            dimension=dimension,
            head_dimension=args.head_dimension,
            slots=slots,
            spectral_lmax_bound=args.spectral_lmax_bound,
            refinement_steps=args.refinement_steps,
        ).to(device)

        def build_dense_factor():
            normal = (
                equations.transpose(-1, -2) @ equations
                + args.ridge * identity
            )
            return torch.linalg.cholesky(normal)

        dense_factor = build_dense_factor()

        def build_head():
            return head(equations, args.ridge)[0]

        cached_preconditioner = build_head()

        def hvp(vector):
            token_scores = torch.einsum(
                "bmk,bkq->bmq",
                equations,
                vector,
            )
            return (
                torch.einsum("bmk,bmq->bkq", equations, token_scores)
                + args.ridge * vector
            )

        setup_functions = {
            "dense_setup": build_dense_factor,
            "matrix_free_head_setup": build_head,
        }
        for method, function in setup_functions.items():
            timing = benchmark(function, args.repeats, device)
            rows.append(
                {
                    "dimension": dimension,
                    "equations": equation_count,
                    "right_hand_sides": 0,
                    "method": method,
                    **timing,
                }
            )

        for rhs_count in parse_ints(args.right_hand_sides):
            if dimension * rhs_count > args.max_state_elements:
                continue
            observations = torch.randn(
                1,
                equation_count,
                rhs_count,
                device=device,
            )
            rhs = torch.einsum("bmk,bmq->bkq", equations, observations)

            def dense_cached_solve():
                return torch.cholesky_solve(rhs, dense_factor)

            def dense_total():
                return torch.cholesky_solve(rhs, build_dense_factor())

            def cached_hb():
                return run_heavy_ball_state_machine(
                    hvp,
                    rhs,
                    cached_preconditioner,
                    args.solver_depth,
                    args.hb_step,
                    args.hb_momentum,
                )[0]

            def head_plus_hb():
                preconditioner = build_head()
                return run_heavy_ball_state_machine(
                    hvp,
                    rhs,
                    preconditioner,
                    args.solver_depth,
                    args.hb_step,
                    args.hb_momentum,
                )[0]

            def cached_chebyshev():
                return run_chebyshev_state_machine(
                    hvp,
                    rhs,
                    cached_preconditioner,
                    args.solver_depth,
                    args.chebyshev_min,
                    args.spectral_lmax_bound,
                )[0]

            def head_plus_chebyshev():
                preconditioner = build_head()
                return run_chebyshev_state_machine(
                    hvp,
                    rhs,
                    preconditioner,
                    args.solver_depth,
                    args.chebyshev_min,
                    args.spectral_lmax_bound,
                )[0]

            def cached_pcg():
                return run_pcg_state_machine(
                    hvp,
                    rhs,
                    cached_preconditioner,
                    args.solver_depth,
                )[0]

            def head_plus_pcg():
                preconditioner = build_head()
                return run_pcg_state_machine(
                    hvp,
                    rhs,
                    preconditioner,
                    args.solver_depth,
                )[0]

            def identity_pcg():
                return run_pcg_state_machine(
                    hvp,
                    rhs,
                    identity,
                    args.solver_depth,
                )[0]

            functions = {
                "dense_total": dense_total,
                "dense_cached_solve": dense_cached_solve,
                "matrix_free_head_plus_hb": head_plus_hb,
                "matrix_free_cached_hb": cached_hb,
                "matrix_free_head_plus_chebyshev": head_plus_chebyshev,
                "matrix_free_cached_chebyshev": cached_chebyshev,
                "matrix_free_head_plus_pcg": head_plus_pcg,
                "matrix_free_cached_pcg": cached_pcg,
                "matrix_free_identity_pcg": identity_pcg,
            }
            for method, function in functions.items():
                timing = benchmark(function, args.repeats, device)
                rows.append(
                    {
                        "dimension": dimension,
                        "equations": equation_count,
                        "right_hand_sides": rhs_count,
                        "method": method,
                        **timing,
                    }
                )

    return {
        "device": str(device),
        "solver_depth": args.solver_depth,
        "refinement_steps": args.refinement_steps,
        "rows": rows,
    }


def write_outputs(result: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    with (outdir / "runtime.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    plot_runtime(result["rows"], outdir / "amortized_runtime.png")


def plot_runtime(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    setup_methods = {
        "dense_setup": "normal + Cholesky",
        "matrix_free_head_setup": "matrix-free head",
    }
    for method, label in setup_methods.items():
        selected = sorted(
            (
                row
                for row in rows
                if row["method"] == method and row["right_hand_sides"] == 0
            ),
            key=lambda row: row["dimension"],
        )
        axes[0].plot(
            [row["dimension"] for row in selected],
            [row["median_ms"] for row in selected],
            marker="o",
            label=label,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coefficient dimension K")
    axes[0].set_ylabel("setup latency (ms)")
    axes[0].set_title("Prompt geometry setup")
    axes[0].grid(which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    total_methods = {
        "dense_total": "dense total",
        "matrix_free_head_plus_hb": "head + HB-10",
        "matrix_free_head_plus_chebyshev": "head + Chebyshev-10",
        "matrix_free_head_plus_pcg": "head + PCG-10",
    }
    for method, label in total_methods.items():
        selected = sorted(
            (
                row
                for row in rows
                if row["method"] == method and row["right_hand_sides"] == 1
            ),
            key=lambda row: row["dimension"],
        )
        axes[1].plot(
            [row["dimension"] for row in selected],
            [row["median_ms"] for row in selected],
            marker="o",
            label=label,
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("coefficient dimension K")
    axes[1].set_ylabel("end-to-end latency (ms)")
    axes[1].set_title("One right-hand side, setup included")
    axes[1].grid(which="both", alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimensions", default="256,512,1024,2048,4096")
    parser.add_argument("--right-hand-sides", default="1,4,16,64")
    parser.add_argument("--equation-ratio", type=int, default=4)
    parser.add_argument("--max-state-elements", type=int, default=32768)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--head-dimension", type=int, default=32)
    parser.add_argument("--refinement-steps", type=int, default=4)
    parser.add_argument("--spectral-lmax-bound", type=float, default=2.5)
    parser.add_argument("--chebyshev-min", type=float, default=0.1)
    parser.add_argument("--solver-depth", type=int, default=10)
    parser.add_argument("--hb-step", type=float, default=0.7)
    parser.add_argument("--hb-momentum", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=92000)
    args = parser.parse_args()
    write_outputs(run(args), Path(args.outdir))


if __name__ == "__main__":
    main()
