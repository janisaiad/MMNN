#!/usr/bin/env python3
"""Benchmark exact-spectrum and eigendecomposition-free prompt heads.

The benchmark isolates preconditioner construction from the recurrent solver.
All methods receive the same dense SPD normal matrix.  The prompt Nystrom head
uses one softmax over ``M`` equation tokens, fixed low-rank slow filtering, QR,
and only ``S x S`` spectral algebra; the exact head diagonalizes ``K x K``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

try:
    from .structured_one_head_heavyball import (
        EquivariantPromptNystromPreconditioner,
        EquivariantRitzSoftmaxPreconditioner,
    )
except ImportError:
    from structured_one_head_heavyball import (
        EquivariantPromptNystromPreconditioner,
        EquivariantRitzSoftmaxPreconditioner,
    )


def parse_ints(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value.strip()]
    if not values or min(values) <= 0:
        raise ValueError("expected positive comma-separated integers")
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def benchmark(function, repeats: int, device: torch.device) -> dict[str, float]:
    for _ in range(min(10, repeats)):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        else:
            start_time = time.perf_counter()
            function()
            samples.append(1000.0 * (time.perf_counter() - start_time))
    return {
        "median_ms": statistics.median(samples),
        "q25_ms": percentile(samples, 0.25),
        "q75_ms": percentile(samples, 0.75),
    }


@torch.inference_mode()
def run(args) -> dict:
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    rows = []
    for batch_size in parse_ints(args.batch_sizes):
        for dimension in parse_ints(args.dimensions):
            equation_count = args.equations_per_dimension * dimension
            slots = min(args.slots, dimension)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1009 * dimension + batch_size
            )
            equations = torch.randn(
                batch_size,
                equation_count,
                dimension,
                generator=generator,
                device=device,
            ) / equation_count**0.5
            identity = torch.eye(
                dimension,
                device=device,
                dtype=equations.dtype,
            ).expand(batch_size, -1, -1)
            normal = equations.transpose(-1, -2) @ equations + args.ridge * identity
            rhs = torch.randn(
                batch_size,
                dimension,
                generator=generator,
                device=device,
            )
            exact_head = EquivariantRitzSoftmaxPreconditioner(
                dimension,
                args.head_dimension,
                slots,
                args.spectral_lmax_bound,
            ).to(device)
            heads = {
                f"prompt_nystrom_r{refinement}": (
                    EquivariantPromptNystromPreconditioner(
                        dimension,
                        args.head_dimension,
                        slots,
                        args.spectral_lmax_bound,
                        refinement,
                    ).to(device)
                )
                for refinement in parse_ints(args.refinement_steps)
            }

            functions = {
                "exact_spectrum_head": lambda: exact_head(equations, normal),
                "cholesky": lambda: torch.linalg.cholesky(normal),
                "cholesky_solve": lambda: torch.cholesky_solve(
                    rhs.unsqueeze(-1),
                    torch.linalg.cholesky(normal),
                ),
                "direct_solve": lambda: torch.linalg.solve(
                    normal,
                    rhs.unsqueeze(-1),
                ),
                "jacobi": lambda: torch.diagonal(
                    normal,
                    dim1=-2,
                    dim2=-1,
                ).reciprocal(),
                **{
                    name: (lambda head=head: head(equations, normal))
                    for name, head in heads.items()
                },
            }
            for method, function in functions.items():
                timing = benchmark(function, args.repeats, device)
                rows.append(
                    {
                        "batch_size": batch_size,
                        "dimension": dimension,
                        "equations": equation_count,
                        "slots": slots,
                        "method": method,
                        **timing,
                    }
                )
    return {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "repeats": args.repeats,
        "scope": (
            "Eager dense-kernel construction latency; excludes recurrent "
            "solver iterations and compilation."
        ),
        "rows": rows,
        "args": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dimensions", default="8,16,32,64,128,256")
    parser.add_argument("--batch-sizes", default="1,64")
    parser.add_argument("--equations-per-dimension", type=int, default=4)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--head-dimension", type=int, default=32)
    parser.add_argument("--spectral-lmax-bound", type=float, default=2.5)
    parser.add_argument("--refinement-steps", default="2,8,12,24")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=91000)
    args = parser.parse_args()
    result = run(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (outdir / "runtime.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
