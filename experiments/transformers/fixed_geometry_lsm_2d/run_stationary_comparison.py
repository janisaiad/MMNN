#!/usr/bin/env python3
"""Compare Richardson, heavy-ball, Chebyshev, and PCG with one LSM encoder/decoder."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .lsm_core import (
    FixedGeometryBornLSM,
    PhysicsConfig,
    StationaryMethod,
    TiedAttentionPCGLSMLoop,
    TiedStationaryLSMLoop,
    exact_bayesian_lsm,
    ranking_loss,
    set_seed,
)
from .run_experiments import (
    aggregate_rows,
    append_evaluation_rows,
    normalise_image,
    task_metrics,
    write_rows,
)

METHODS: tuple[StationaryMethod, ...] = ("richardson", "heavy_ball", "chebyshev")
DEPTHS = (4, 8, 12, 20, 32, 48, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pcg-results-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def train_stationary(
    physics: FixedGeometryBornLSM,
    method: StationaryMethod,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    log_every: int,
) -> tuple[TiedStationaryLSMLoop, list[dict[str, object]]]:
    set_seed(seed)
    model = TiedStationaryLSMLoop(
        physics.kernel,
        physics.cfg.ridge_rel,
        depth=20,
        method=method,
        controller_width=32,
    ).to(physics.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for step in range(steps + 1):
        family = "disk" if step % 5 == 0 else "ellipse"
        noise = 0.03 + 0.09 * torch.rand(()).item()
        far_field, mask = physics.sample_batch(batch_size, family, noise_rel=noise)
        score, info = model(far_field, physics.probe_rhs)
        rank = ranking_loss(score, mask, n_pairs=48 if not steps < 100 else 12)
        loss = rank + 0.20 * torch.log1p(info["relative_residual"]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            row: dict[str, object] = {
                "method": method,
                "seed": seed,
                "step": step,
                "loss": float(loss.detach()),
                "ranking_loss": float(rank.detach()),
                "relative_residual": float(info["relative_residual"].mean().detach()),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - start,
            }
            for key in ("eta", "beta", "lower_bound", "upper_bound"):
                row[key] = float(info[key].mean().detach()) if key in info else ""
            rows.append(row)
            print(
                f"{method:11s} seed={seed} step={step:04d} "
                f"loss={float(loss):.4f} residual={float(info['relative_residual'].mean()):.4f}"
            )
    return model, rows


def load_pcg(
    physics: FixedGeometryBornLSM,
    pcg_results_dir: Path,
    seed: int,
) -> TiedAttentionPCGLSMLoop:
    checkpoint = torch.load(pcg_results_dir / f"loop_seed_{seed}.pt", map_location=physics.device)
    model = TiedAttentionPCGLSMLoop(
        physics.kernel,
        physics.feature_kernel,
        physics.cfg.ridge_rel,
        depth=checkpoint["loop"]["depth"],
        controller_width=checkpoint["loop"]["controller_width"],
    ).to(physics.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_methods(
    physics: FixedGeometryBornLSM,
    stationary: dict[str, TiedStationaryLSMLoop],
    pcg: TiedAttentionPCGLSMLoop,
    seed: int,
    eval_tasks: int,
) -> tuple[list[dict[str, object]], dict[str, Tensor]]:
    set_seed(seed + 1_100_000)
    rows: list[dict[str, object]] = []
    showcase: dict[str, Tensor] = {}
    for family in ("ellipse", "kite", "two_disks"):
        task_offset = 0
        while task_offset < eval_tasks:
            batch_size = min(24, eval_tasks - task_offset)
            far_field, mask = physics.sample_batch(batch_size, family, noise_rel=0.15)
            evaluations: dict[str, tuple[Tensor, dict[str, Tensor]]] = {
                method: model(far_field, physics.probe_rhs, depth=20)
                for method, model in stationary.items()
            }
            evaluations["pcg"] = pcg(far_field, physics.probe_rhs, depth=20)
            evaluations["exact_tikhonov"] = exact_bayesian_lsm(
                far_field,
                physics.probe_rhs,
                physics.kernel,
                physics.cfg.ridge_rel,
            )
            for method, (score, info) in evaluations.items():
                local: list[dict[str, object]] = []
                append_evaluation_rows(
                    local,
                    score,
                    mask,
                    physics.grid,
                    info["relative_residual"],
                    seed=seed,
                    method=method,
                    family=family,
                    noise=0.15,
                    stage="solver_comparison",
                )
                for row in local:
                    row["task"] = int(row["task"]) + task_offset
                rows.extend(local)
                if family == "kite" and task_offset == 0:
                    showcase[f"score_{method}"] = score[:3].cpu()
            if family == "kite" and task_offset == 0:
                showcase["mask"] = mask[:3].cpu()
            task_offset += batch_size
    return rows, showcase


@torch.no_grad()
def evaluate_depths(
    physics: FixedGeometryBornLSM,
    stationary: dict[str, TiedStationaryLSMLoop],
    pcg: TiedAttentionPCGLSMLoop,
    seed: int,
    eval_tasks: int,
) -> list[dict[str, object]]:
    set_seed(seed + 1_200_000)
    far_field, mask = physics.sample_batch(eval_tasks, "kite", noise_rel=0.15)
    models: dict[str, object] = {**stationary, "pcg": pcg}
    rows: list[dict[str, object]] = []
    for method, model in models.items():
        for depth in DEPTHS:
            score, info = model(far_field, physics.probe_rhs, depth=depth)
            metrics = task_metrics(score, mask, physics.grid)
            for task_index, (metric, residual) in enumerate(
                zip(metrics, info["relative_residual"].cpu().numpy(), strict=True)
            ):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "depth": depth,
                        "task": task_index,
                        **metric,
                        "relative_residual": float(residual),
                    }
                )
    return rows


@torch.no_grad()
def benchmark(
    physics: FixedGeometryBornLSM,
    stationary: dict[str, TiedStationaryLSMLoop],
    pcg: TiedAttentionPCGLSMLoop,
) -> dict[str, float]:
    far_field, _ = physics.sample_batch(24, "kite", noise_rel=0.15)
    operations = {
        **{
            method: (lambda model=model: model(far_field, physics.probe_rhs, depth=20))
            for method, model in stationary.items()
        },
        "pcg": lambda: pcg(far_field, physics.probe_rhs, depth=20),
        "exact_tikhonov": lambda: exact_bayesian_lsm(
            far_field,
            physics.probe_rhs,
            physics.kernel,
            physics.cfg.ridge_rel,
        ),
    }
    output: dict[str, float] = {}
    for method, operation in operations.items():
        for _ in range(10):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        output[method] = 1000.0 * (time.perf_counter() - start) / 50.0
    return output


def plot_reconstructions(path: Path, showcase: dict[str, Tensor], grid_size: int) -> None:
    methods = (
        ("richardson", "Richardson"),
        ("heavy_ball", "heavy-ball"),
        ("chebyshev", "Chebyshev"),
        ("pcg", "PCG"),
        ("exact_tikhonov", "exact"),
    )
    figure, axes = plt.subplots(3, 6, figsize=(14.0, 7.2))
    for row in range(3):
        axes[row, 0].imshow(showcase["mask"][row].reshape(grid_size, grid_size), origin="lower", cmap="gray_r")
        axes[row, 0].set_title("obstacle" if row == 0 else "")
        for column, (method, title) in enumerate(methods, start=1):
            image = showcase[f"score_{method}"][row].reshape(grid_size, grid_size).numpy()
            axes[row, column].imshow(normalise_image(image), origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
            axes[row, column].set_title(title if row == 0 else "")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(
    path: Path,
    task_rows: list[dict[str, object]],
    depth_rows: list[dict[str, object]],
) -> None:
    methods = ("richardson", "heavy_ball", "chebyshev", "pcg")
    colors = dict(zip(methods, ("C3", "C2", "C1", "C0"), strict=True))
    figure, axes = plt.subplots(1, 3, figsize=(13.3, 3.8))
    families = ("ellipse", "kite", "two_disks")
    width = 0.8 / len(methods)
    x = np.arange(3)
    for index, method in enumerate(methods):
        means = [
            np.mean(
                [
                    float(row["average_precision"])
                    for row in task_rows
                    if row["method"] == method and row["family"] == family
                ]
            )
            for family in families
        ]
        axes[0].bar(x + (index - 1.5) * width, means, width=width, label=method.replace("_", " "), color=colors[method])
    axes[0].set_xticks(x, ["ellipse", "kite", "two disks"])
    axes[0].set(title="Localization at 20 loops", ylabel="average precision", ylim=(0.0, 1.02))

    for method in methods:
        residual = [
            np.mean(
                [
                    float(row["relative_residual"])
                    for row in depth_rows
                    if row["method"] == method and int(row["depth"]) == depth
                ]
            )
            for depth in DEPTHS
        ]
        precision = [
            np.mean(
                [
                    float(row["average_precision"])
                    for row in depth_rows
                    if row["method"] == method and int(row["depth"]) == depth
                ]
            )
            for depth in DEPTHS
        ]
        axes[1].plot(DEPTHS, residual, marker="o", label=method.replace("_", " "), color=colors[method])
        axes[2].plot(DEPTHS, precision, marker="o", label=method.replace("_", " "), color=colors[method])
    axes[1].set_yscale("log")
    axes[1].set(title="OOD kite convergence", xlabel="loop depth", ylabel="relative residual")
    axes[2].set(title="OOD kite localization", xlabel="loop depth", ylabel="average precision", ylim=(0.0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def count_parameters(model: object) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pcg_results_dir = args.pcg_results_dir.resolve()
    device = torch.device(args.device)
    physics_cfg = PhysicsConfig(
        n_angles=24 if args.quick else 32,
        grid_size=20 if args.quick else 32,
        domain_half_width=0.8,
        wavenumber=8.0,
        noise_rel=0.08,
        ridge_rel=0.01,
        kernel_gamma=1.0,
        kernel_mix=0.20,
    )
    if args.quick:
        raise ValueError("quick mode requires matching quick PCG checkpoints and is intentionally disabled")
    physics = FixedGeometryBornLSM(physics_cfg, device)
    seeds = (17, 29, 43)
    steps = 1500
    eval_tasks = 192
    training_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    runtime_by_seed: dict[str, dict[str, float]] = {}
    showcase: dict[str, Tensor] | None = None
    parameter_counts: dict[str, int] = {}
    start = time.perf_counter()

    for seed in seeds:
        stationary: dict[str, TiedStationaryLSMLoop] = {}
        for method in METHODS:
            model, local_training = train_stationary(
                physics,
                method,
                seed,
                steps=steps,
                batch_size=16,
                log_every=50,
            )
            stationary[method] = model
            training_rows.extend(local_training)
            parameter_counts[method] = count_parameters(model)
            torch.save(
                {
                    "model": model.state_dict(),
                    "method": method,
                    "physics": asdict(physics_cfg),
                    "depth": 20,
                    "controller_width": 32,
                    "seed": seed,
                },
                output_dir / f"{method}_seed_{seed}.pt",
            )
        pcg = load_pcg(physics, pcg_results_dir, seed)
        parameter_counts["pcg"] = count_parameters(pcg)
        local_tasks, local_showcase = evaluate_methods(
            physics,
            stationary,
            pcg,
            seed,
            eval_tasks,
        )
        task_rows.extend(local_tasks)
        depth_rows.extend(evaluate_depths(physics, stationary, pcg, seed, eval_tasks))
        runtime_by_seed[str(seed)] = benchmark(physics, stationary, pcg)
        if showcase is None:
            showcase = local_showcase

    assert showcase is not None
    write_rows(output_dir / "training.csv", training_rows)
    write_rows(output_dir / "tasks.csv", task_rows)
    write_rows(output_dir / "depth_tasks.csv", depth_rows)
    write_rows(output_dir / "summary.csv", aggregate_rows(task_rows, ("method", "family", "noise_rel")))
    with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime_by_seed, handle, indent=2)
    protocol = {
        "description": "same fixed LSM encoder/decoder; recurrent solver ablation only",
        "physics": asdict(physics_cfg),
        "seeds": seeds,
        "steps": steps,
        "train_depth": 20,
        "eval_tasks_per_seed_and_family": eval_tasks,
        "trainable_parameter_counts": parameter_counts,
        "loss": "balanced ranking + 0.20 log(1 + relative sampling residual)",
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    plot_reconstructions(output_dir / "reconstructions.png", showcase, physics_cfg.grid_size)
    plot_comparison(output_dir / "solver_comparison.png", task_rows, depth_rows)
    print(f"completed in {protocol['elapsed_seconds']:.1f}s: {output_dir}")


if __name__ == "__main__":
    main()
