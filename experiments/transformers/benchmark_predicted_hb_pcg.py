#!/usr/bin/env python3
"""Benchmark the retained HB/PCG Pareto point on the same prompt and head."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

try:
    from .evaluate_trained_loop_controllers import build_model
    from .exact_loop_transformer_decoder import normal_equations
    from .first_principles_decoder_cells import (
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from .pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )
except ImportError:
    from evaluate_trained_loop_controllers import build_model
    from exact_loop_transformer_decoder import normal_equations
    from first_principles_decoder_cells import (
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
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
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    saved = checkpoint["args"]
    set_seed(saved["seed"])
    family = make_true_family(
        saved["d"],
        saved["K"],
        saved["basis_scale"],
        saved["A0_scale"],
        device,
        operator_family=saved.get("operator_family", "dense_spd"),
    )
    model = build_model(saved, family, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ridge_metric = model.coefficient_ridge_metric()
    if ridge_metric is None:
        ridge_metric = torch.eye(
            saved["K"],
            device=device,
            dtype=model.A0.dtype,
        )
    step, momentum = model.loop_decoder.heavy_ball_coefficients()
    rows = []
    for batch_size in parse_ints(args.batch_sizes):
        set_seed(args.seed + batch_size)
        batch = sample_icl_batch(
            family,
            batch_size,
            saved["m"],
            args.z_scale,
            saved["f_std"],
            saved["noise_std"],
            device,
        )
        equations, observations = model.weak_system(
            batch.f_prompt,
            batch.u_prompt,
        )
        normal_matrix, rhs = normal_equations(
            equations,
            observations,
            model.lam_z,
            ridge_metric,
        )
        preconditioner, _ = model.loop_decoder.preconditioner_head(
            equations,
            normal_matrix,
        )

        def hvp(vector):
            scores = torch.einsum("bmk,bk->bm", equations, vector)
            moment = torch.einsum("bmk,bm->bk", equations, scores)
            ridge_action = torch.einsum(
                "kl,bl->bk",
                ridge_metric,
                vector,
            )
            return moment + model.lam_z * ridge_action

        def head():
            return model.loop_decoder.preconditioner_head(
                equations,
                normal_matrix,
            )[0]

        def heavy_ball():
            return run_heavy_ball_state_machine(
                hvp,
                rhs,
                preconditioner,
                args.hb_depth,
                step,
                momentum,
            )[0]

        def pcg():
            return run_pcg_state_machine(
                hvp,
                rhs,
                preconditioner,
                args.pcg_depth,
            )[0]

        head_timing = benchmark(head, args.repeats, device)
        hb_timing = benchmark(heavy_ball, args.repeats, device)
        pcg_timing = benchmark(pcg, args.repeats, device)
        rows.append(
            {
                "batch_size": batch_size,
                "hb_depth": args.hb_depth,
                "pcg_depth": args.pcg_depth,
                "head_median_ms": head_timing["median_ms"],
                "hb_median_ms": hb_timing["median_ms"],
                "pcg_median_ms": pcg_timing["median_ms"],
                "hb_over_pcg": (
                    hb_timing["median_ms"] / pcg_timing["median_ms"]
                ),
                "head_plus_hb_ms": (
                    head_timing["median_ms"] + hb_timing["median_ms"]
                ),
                "head_plus_pcg_ms": (
                    head_timing["median_ms"] + pcg_timing["median_ms"]
                ),
                "hb_q25_ms": hb_timing["q25_ms"],
                "hb_q75_ms": hb_timing["q75_ms"],
                "pcg_q25_ms": pcg_timing["q25_ms"],
                "pcg_q75_ms": pcg_timing["q75_ms"],
            }
        )
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "repeats": args.repeats,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", default="1,64,1024")
    parser.add_argument("--hb-depth", type=int, default=10)
    parser.add_argument("--pcg-depth", type=int, default=8)
    parser.add_argument("--z-scale", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=71000)
    args = parser.parse_args()
    result = run(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (outdir / "runtime.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(result["rows"][0]),
        )
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
