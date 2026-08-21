#!/usr/bin/env python3
"""Post-training checks for the final fixed-geometry LSM experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .lsm_core import (
    FixedGeometryBornLSM,
    PhysicsConfig,
    TiedAttentionPCGLSMLoop,
    exact_bayesian_lsm,
    set_seed,
)
from .run_experiments import aggregate_rows, append_evaluation_rows, task_metrics, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(
    results_dir: Path,
    physics: FixedGeometryBornLSM,
    seed: int,
) -> TiedAttentionPCGLSMLoop:
    checkpoint = torch.load(results_dir / f"loop_seed_{seed}.pt", map_location=physics.device)
    width = checkpoint["loop"]["controller_width"]
    depth = checkpoint["loop"]["depth"]
    model = TiedAttentionPCGLSMLoop(
        physics.kernel,
        physics.feature_kernel,
        physics.cfg.ridge_rel,
        depth,
        width,
    ).to(physics.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def offgrid_evaluation(
    results_dir: Path,
    base: FixedGeometryBornLSM,
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    fine_cfg = PhysicsConfig(**{**base.cfg.__dict__, "grid_size": 64})
    fine = FixedGeometryBornLSM(fine_cfg, base.device)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        model = load_model(results_dir, base, seed)
        set_seed(seed + 700_000)
        for family in ("ellipse", "kite", "two_disks"):
            for task_offset in range(0, 192, 24):
                far_field, fine_mask = fine.sample_batch(24, family, noise_rel=0.15)
                coarse_mask = F.avg_pool2d(
                    fine_mask.float().reshape(-1, 1, 64, 64),
                    kernel_size=2,
                ).squeeze(1).flatten(1) > 0.5
                evaluations = {
                    "trained_pcg_20": model(far_field, base.probe_rhs, depth=20),
                    "exact_tikhonov": exact_bayesian_lsm(
                        far_field,
                        base.probe_rhs,
                        base.kernel,
                        base.cfg.ridge_rel,
                    ),
                }
                for method, (score, info) in evaluations.items():
                    local: list[dict[str, object]] = []
                    append_evaluation_rows(
                        local,
                        score,
                        coarse_mask,
                        base.grid,
                        info["relative_residual"],
                        seed=seed,
                        method=method,
                        family=family,
                        noise=0.15,
                        stage="fine_forward_grid",
                    )
                    for row in local:
                        row["task"] = int(row["task"]) + task_offset
                    rows.extend(local)
    return rows


@torch.no_grad()
def depth_comparison(
    results_dir: Path,
    physics: FixedGeometryBornLSM,
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        model = load_model(results_dir, physics, seed)
        set_seed(seed + 800_000)
        far_field, mask = physics.sample_batch(192, "kite", noise_rel=0.15)
        for depth in (1, 2, 4, 6, 8, 12, 16, 20):
            for method, fixed in (("trained_pcg", None), ("identity_cg", (0.0, 0.0))):
                score, info = model(
                    far_field,
                    physics.probe_rhs,
                    depth=depth,
                    fixed_preconditioner=fixed,
                )
                metrics = task_metrics(score, mask, physics.grid)
                for task_index, (metric, residual) in enumerate(
                    zip(metrics, info["relative_residual"].cpu().numpy(), strict=True)
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "depth": depth,
                            "task": task_index,
                            **metric,
                            "relative_residual": float(residual),
                        }
                    )
    return rows


def read_design_angles(results_dir: Path) -> dict[tuple[int, str], np.ndarray]:
    output: dict[tuple[int, str], list[float]] = {}
    with (results_dir / "design_angles.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.setdefault((int(row["seed"]), row["design"]), []).append(
                math.radians(float(row["angle_degrees"]))
            )
    return {key: np.asarray(value) for key, value in output.items()}


def point_spread_diagnostics(
    results_dir: Path,
    wavenumber: float,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    angles = read_design_angles(results_dir)
    axis = torch.linspace(-0.8, 0.8, 96, device=device)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    center_index = grid.square().sum(dim=-1).argmin()
    distance = torch.linalg.vector_norm(grid - grid[center_index], dim=-1)
    rows: list[dict[str, object]] = []
    maps: dict[str, np.ndarray] = {}
    for (seed, method), angle_values in angles.items():
        angle = torch.tensor(angle_values, device=device, dtype=torch.float32)
        directions = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        wavevectors = (directions[None, :, :] - directions[:, None, :]).reshape(-1, 2)
        dictionary = torch.exp(1j * wavenumber * (wavevectors @ grid.T))
        point_spread = (
            dictionary[:, center_index].conj()[:, None] * dictionary
        ).mean(dim=0).abs()
        sidelobes = point_spread[distance > 0.20]
        rows.append(
            {
                "seed": seed,
                "design": method,
                "max_sidelobe": float(sidelobes.max()),
                "p95_sidelobe": float(torch.quantile(sidelobes, 0.95)),
                "mean_sidelobe": float(sidelobes.mean()),
            }
        )
        if seed == min(key[0] for key in angles):
            maps[method] = point_spread.reshape(96, 96).cpu().numpy()
    return rows, maps


@torch.no_grad()
def benchmark_runtime(
    results_dir: Path,
    physics: FixedGeometryBornLSM,
    seed: int,
) -> dict[str, float]:
    model = load_model(results_dir, physics, seed)
    set_seed(seed + 900_000)
    far_field, _ = physics.sample_batch(24, "kite", noise_rel=0.15)
    methods = {
        "trained_pcg_8": lambda: model(far_field, physics.probe_rhs, depth=8),
        "trained_pcg_20": lambda: model(far_field, physics.probe_rhs, depth=20),
        "exact_tikhonov": lambda: exact_bayesian_lsm(
            far_field,
            physics.probe_rhs,
            physics.kernel,
            physics.cfg.ridge_rel,
        ),
    }
    timings: dict[str, float] = {}
    for name, operation in methods.items():
        for _ in range(10):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        timings[name] = 1000.0 * (time.perf_counter() - start) / 50.0
    return timings


def plot_depth_and_offgrid(
    path: Path,
    depth_rows: list[dict[str, object]],
    offgrid_rows: list[dict[str, object]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    depths = sorted({int(row["depth"]) for row in depth_rows})
    for method, color in (("trained_pcg", "C0"), ("identity_cg", "C1")):
        residual = [
            np.mean(
                [
                    float(row["relative_residual"])
                    for row in depth_rows
                    if row["method"] == method and int(row["depth"]) == depth
                ]
            )
            for depth in depths
        ]
        precision = [
            np.mean(
                [
                    float(row["average_precision"])
                    for row in depth_rows
                    if row["method"] == method and int(row["depth"]) == depth
                ]
            )
            for depth in depths
        ]
        axes[0].plot(depths, residual, marker="o", label=method.replace("_", " "), color=color)
        axes[1].plot(depths, precision, marker="o", label=method.replace("_", " "), color=color)
    axes[0].set_yscale("log")
    axes[0].set(title="Sampling-equation convergence", xlabel="loop depth", ylabel="relative residual")
    axes[1].set(title="OOD kite localization", xlabel="loop depth", ylabel="average precision", ylim=(0.90, 1.005))

    families = ("ellipse", "kite", "two_disks")
    x = np.arange(3)
    width = 0.35
    for index, method in enumerate(("trained_pcg_20", "exact_tikhonov")):
        means = [
            np.mean(
                [
                    float(row["average_precision"])
                    for row in offgrid_rows
                    if row["method"] == method and row["family"] == family
                ]
            )
            for family in families
        ]
        axes[2].bar(x + (index - 0.5) * width, means, width=width, label=method.replace("_", " "))
    axes[2].set_xticks(x, ["ellipse", "kite", "two disks"])
    axes[2].set(title="Fine-grid forward / coarse probing", ylabel="average precision", ylim=(0.80, 1.005))
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_point_spread(
    path: Path,
    maps: dict[str, np.ndarray],
    rows: list[dict[str, object]],
) -> None:
    methods = ("learned", "uniform", "random")
    figure, axes = plt.subplots(1, 4, figsize=(13.5, 3.3), layout="constrained")
    for axis, method in zip(axes[:3], methods, strict=True):
        image = axis.imshow(maps[method], origin="lower", extent=(-0.8, 0.8, -0.8, 0.8), cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(f"{method} design")
        axis.set_xticks([])
        axis.set_yticks([])
    del image
    means = []
    errors = []
    for method in methods:
        values = [float(row["max_sidelobe"]) for row in rows if row["design"] == method]
        means.append(np.mean(values))
        errors.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
    axes[3].bar(methods, means, yerr=errors, color=("C0", "C1", "C2"), capsize=3)
    axes[3].set(title="Maximum sidelobe", ylim=(0.0, 1.05))
    axes[3].grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    protocol = json.loads((results_dir / "protocol.json").read_text(encoding="utf-8"))
    physics_cfg = PhysicsConfig(**protocol["physics"])
    device = torch.device(args.device)
    physics = FixedGeometryBornLSM(physics_cfg, device)
    seeds = tuple(int(seed) for seed in protocol["run"]["seeds"])

    offgrid_rows = offgrid_evaluation(results_dir, physics, seeds)
    depth_rows = depth_comparison(results_dir, physics, seeds)
    point_spread_rows, point_spread_maps = point_spread_diagnostics(
        results_dir,
        physics_cfg.wavenumber,
        device,
    )
    timings = benchmark_runtime(results_dir, physics, seeds[0])

    write_rows(results_dir / "offgrid_tasks.csv", offgrid_rows)
    write_rows(
        results_dir / "offgrid_summary.csv",
        aggregate_rows(offgrid_rows, ("method", "family", "noise_rel")),
    )
    write_rows(results_dir / "depth_comparison.csv", depth_rows)
    write_rows(results_dir / "point_spread_diagnostics.csv", point_spread_rows)
    (results_dir / "runtime_benchmark.json").write_text(
        json.dumps(
            {
                "batch_size": 24,
                "n_probes": physics.n_probes,
                "milliseconds_per_batch": timings,
                "note": "H100 wall-clock; includes system assembly and indicator readout",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_depth_and_offgrid(results_dir / "depth_and_offgrid.png", depth_rows, offgrid_rows)
    plot_point_spread(
        results_dir / "point_spread_diagnostics.png",
        point_spread_maps,
        point_spread_rows,
    )


if __name__ == "__main__":
    main()
