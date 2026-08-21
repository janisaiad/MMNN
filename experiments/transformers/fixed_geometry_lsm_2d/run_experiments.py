#!/usr/bin/env python3
"""Train and evaluate fixed-geometry ICL and angular experiment design for LSM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

from .lsm_core import (
    FixedGeometryBornLSM,
    LoopConfig,
    PhysicsConfig,
    ShapeFamily,
    TiedAttentionPCGLSMLoop,
    exact_bayesian_lsm,
    ranking_loss,
    set_seed,
)


@dataclass(frozen=True)
class RunConfig:
    seeds: tuple[int, ...] = (17, 29, 43)
    design_budget: int = 6
    design_steps: int = 900
    design_batch_size: int = 20
    design_lr: float = 2.0e-2
    design_repulsion: float = 1.0e-3
    train_noise_min: float = 0.03
    train_noise_max: float = 0.12
    evaluation_noise: tuple[float, ...] = (0.08, 0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def confidence_summary(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = array.std(ddof=1) / math.sqrt(array.size) if array.size > 1 else 0.0
    return float(array.mean()), float(1.96 * standard_error)


def top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    threshold = np.quantile(score, 1.0 - fraction)
    return score >= threshold


def task_metrics(score: Tensor, mask: Tensor, grid: Tensor) -> list[dict[str, float]]:
    scores = score.detach().cpu().numpy()
    masks = mask.detach().cpu().numpy().astype(bool)
    points = grid.detach().cpu().numpy()
    rows: list[dict[str, float]] = []
    for task_score, task_mask in zip(scores, masks, strict=True):
        prediction = top_fraction_mask(task_score, 0.07)
        intersection = np.logical_and(prediction, task_mask).sum()
        union = np.logical_or(prediction, task_mask).sum()
        predicted_centroid = points[prediction].mean(axis=0)
        target_centroid = points[task_mask].mean(axis=0)
        rows.append(
            {
                "auc": float(roc_auc_score(task_mask, task_score)),
                "average_precision": float(average_precision_score(task_mask, task_score)),
                "iou_top7": float(intersection / max(union, 1)),
                "centroid_error": float(np.linalg.norm(predicted_centroid - target_centroid)),
            }
        )
    return rows


def append_evaluation_rows(
    output: list[dict[str, object]],
    score: Tensor,
    mask: Tensor,
    grid: Tensor,
    residual: Tensor,
    *,
    seed: int,
    method: str,
    family: str,
    noise: float,
    stage: str,
) -> None:
    metrics = task_metrics(score, mask, grid)
    residual_values = residual.detach().cpu().numpy()
    for task_index, (metric, residual_value) in enumerate(zip(metrics, residual_values, strict=True)):
        output.append(
            {
                "stage": stage,
                "seed": seed,
                "method": method,
                "family": family,
                "noise_rel": noise,
                "task": task_index,
                **metric,
                "relative_residual": float(residual_value),
            }
        )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_loop(
    physics: FixedGeometryBornLSM,
    loop_cfg: LoopConfig,
    run_cfg: RunConfig,
    seed: int,
) -> tuple[TiedAttentionPCGLSMLoop, list[dict[str, float]]]:
    set_seed(seed)
    model = TiedAttentionPCGLSMLoop(
        physics.kernel,
        physics.feature_kernel,
        physics.cfg.ridge_rel,
        loop_cfg.depth,
        loop_cfg.controller_width,
    ).to(physics.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_cfg.learning_rate,
        weight_decay=loop_cfg.weight_decay,
    )
    log: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(loop_cfg.steps + 1):
        family: ShapeFamily = "disk" if step % 5 == 0 else "ellipse"
        noise = run_cfg.train_noise_min + (
            run_cfg.train_noise_max - run_cfg.train_noise_min
        ) * torch.rand(()).item()
        far_field, mask = physics.sample_batch(
            loop_cfg.batch_size,
            family,
            noise_rel=noise,
        )
        score, info = model(far_field, physics.probe_rhs)
        rank = ranking_loss(score, mask, loop_cfg.rank_pairs)
        loss = rank + 0.20 * torch.log1p(info["relative_residual"]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if step % loop_cfg.log_every == 0 or step == loop_cfg.steps:
            with torch.no_grad():
                _, history = model(
                    far_field[: min(4, far_field.shape[0])],
                    physics.probe_rhs,
                    return_history=True,
                )
            row = {
                "seed": float(seed),
                "step": float(step),
                "loss": float(loss.detach()),
                "relative_residual": float(info["relative_residual"].mean().detach()),
                "gradient_norm": float(gradient_norm),
                "preconditioner_linear": float(
                    history["preconditioner_coefficients"][:, 0].mean()
                ),
                "preconditioner_quadratic": float(
                    history["preconditioner_coefficients"][:, 1].mean()
                ),
                "elapsed_seconds": time.perf_counter() - start,
            }
            log.append(row)
            print(
                f"loop seed={seed} step={step:04d} loss={row['loss']:.4f} "
                f"preconditioner=({row['preconditioner_linear']:.3f},"
                f"{row['preconditioner_quadratic']:.3f})"
            )
    return model, log


@torch.no_grad()
def evaluate_fixed_geometry(
    physics: FixedGeometryBornLSM,
    model: TiedAttentionPCGLSMLoop,
    loop_cfg: LoopConfig,
    run_cfg: RunConfig,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, Tensor]]:
    set_seed(seed + 100_000)
    rows: list[dict[str, object]] = []
    showcase: dict[str, Tensor] = {}
    families: tuple[ShapeFamily, ...] = ("ellipse", "kite", "two_disks")
    for noise in run_cfg.evaluation_noise:
        for family in families:
            remaining = loop_cfg.eval_tasks
            task_offset = 0
            while remaining > 0:
                batch_size = min(loop_cfg.eval_batch_size, remaining)
                far_field, mask = physics.sample_batch(batch_size, family, noise_rel=noise)
                evaluations: dict[str, tuple[Tensor, dict[str, Tensor]]] = {
                    "identity_cg_8": model(
                        far_field,
                        physics.probe_rhs,
                        depth=8,
                        fixed_preconditioner=(0.0, 0.0),
                    ),
                    "trained_pcg_8": model(far_field, physics.probe_rhs, depth=8),
                    "trained_pcg_20": model(far_field, physics.probe_rhs, depth=20),
                    "identity_cg_20": model(
                        far_field,
                        physics.probe_rhs,
                        depth=20,
                        fixed_preconditioner=(0.0, 0.0),
                    ),
                    "exact_tikhonov": exact_bayesian_lsm(
                        far_field,
                        physics.probe_rhs,
                        physics.kernel,
                        physics.cfg.ridge_rel,
                    ),
                }
                for method, (score, info) in evaluations.items():
                    local_rows: list[dict[str, object]] = []
                    append_evaluation_rows(
                        local_rows,
                        score,
                        mask,
                        physics.grid,
                        info["relative_residual"],
                        seed=seed,
                        method=method,
                        family=family,
                        noise=noise,
                        stage="fixed_geometry",
                    )
                    for local_row in local_rows:
                        local_row["task"] = int(local_row["task"]) + task_offset
                    rows.extend(local_rows)
                    if family == "kite" and noise == run_cfg.evaluation_noise[0] and task_offset == 0:
                        showcase[f"score_{method}"] = score[:3].detach().cpu()
                if family == "kite" and noise == run_cfg.evaluation_noise[0] and task_offset == 0:
                    showcase["mask"] = mask[:3].detach().cpu()
                    showcase["far_field"] = far_field[:3].detach().cpu()
                task_offset += batch_size
                remaining -= batch_size
    return rows, showcase


@torch.no_grad()
def evaluate_depths(
    physics: FixedGeometryBornLSM,
    model: TiedAttentionPCGLSMLoop,
    loop_cfg: LoopConfig,
    seed: int,
) -> list[dict[str, object]]:
    set_seed(seed + 200_000)
    far_field, mask = physics.sample_batch(loop_cfg.eval_tasks, "kite", noise_rel=0.12)
    rows: list[dict[str, object]] = []
    for depth in (1, 2, 4, 6, 8, 12, 16, 20):
        score, info = model(far_field, physics.probe_rhs, depth=depth)
        metrics = task_metrics(score, mask, physics.grid)
        for task_index, (metric, residual) in enumerate(
            zip(metrics, info["relative_residual"].cpu().numpy(), strict=True)
        ):
            rows.append(
                {
                    "seed": seed,
                    "depth": depth,
                    "task": task_index,
                    **metric,
                    "relative_residual": float(residual),
                }
            )
    return rows


def circular_repulsion(angles: Tensor) -> Tensor:
    difference = angles[:, None] - angles[None, :]
    off_diagonal = ~torch.eye(angles.numel(), device=angles.device, dtype=torch.bool)
    return torch.exp(3.0 * torch.cos(difference))[off_diagonal].mean()


class SeparatedAngularDesign(nn.Module):
    """Periodic angle parameterisation with a hard minimum separation."""

    def __init__(self, budget: int, device: torch.device, minimum_gap_degrees: float = 28.0) -> None:
        super().__init__()
        minimum_gap = math.radians(minimum_gap_degrees)
        if budget * minimum_gap >= 2.0 * math.pi:
            raise ValueError("minimum gaps leave no angular design freedom")
        self.budget = int(budget)
        self.minimum_gap = float(minimum_gap)
        self.raw_gaps = nn.Parameter(torch.randn(budget, device=device))
        self.rotation = nn.Parameter(torch.rand((), device=device) * 2.0 * math.pi)

    def forward(self) -> Tensor:
        free_angle = 2.0 * math.pi - self.budget * self.minimum_gap
        gaps = self.minimum_gap + free_angle * torch.softmax(self.raw_gaps, dim=0)
        offsets = torch.cat(
            [torch.zeros(1, device=gaps.device, dtype=gaps.dtype), torch.cumsum(gaps[:-1], dim=0)]
        )
        return self.rotation + offsets


def train_design(
    physics: FixedGeometryBornLSM,
    model: TiedAttentionPCGLSMLoop,
    run_cfg: RunConfig,
    seed: int,
) -> tuple[Tensor, Tensor, list[dict[str, float]]]:
    set_seed(seed + 300_000)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    design = SeparatedAngularDesign(run_cfg.design_budget, physics.device)
    random_angles = design().detach().clone()
    optimizer = torch.optim.Adam(design.parameters(), lr=run_cfg.design_lr)
    log: list[dict[str, float]] = []
    start = time.perf_counter()
    for step in range(run_cfg.design_steps + 1):
        family: ShapeFamily = "disk" if step % 5 == 0 else "ellipse"
        mask = physics.sample_masks(run_cfg.design_batch_size, family)
        noise = run_cfg.train_noise_min + (
            run_cfg.train_noise_max - run_cfg.train_noise_min
        ) * torch.rand(()).item()
        angles = design()
        far_field, probe, _, kernel, feature_kernel = physics.acquisition_at_angles(
            mask,
            angles,
            noise_rel=noise,
        )
        score, info = model(
            far_field,
            probe,
            kernel=kernel,
            feature_kernel=feature_kernel,
        )
        rank = ranking_loss(score, mask, n_pairs=48)
        repulsion = circular_repulsion(angles)
        # Separation is enforced by construction; no analytic design proxy is
        # optimized.  The sole design objective is held-out LSM localization.
        loss = rank
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(design.parameters(), 2.0)
        optimizer.step()
        if step % 50 == 0 or step == run_cfg.design_steps:
            sorted_angles = torch.sort(torch.remainder(angles.detach(), 2.0 * math.pi)).values
            row = {
                "seed": float(seed),
                "step": float(step),
                "loss": float(loss.detach()),
                "ranking_loss": float(rank.detach()),
                "repulsion": float(repulsion.detach()),
                "relative_residual": float(info["relative_residual"].mean().detach()),
                "gradient_norm": float(gradient_norm),
                "minimum_gap_degrees": float(minimum_circular_gap(sorted_angles) * 180.0 / math.pi),
                "elapsed_seconds": time.perf_counter() - start,
            }
            log.append(row)
            print(
                f"design seed={seed} step={step:04d} loss={row['loss']:.4f} "
                f"min-gap={row['minimum_gap_degrees']:.1f}deg"
            )
    return design().detach(), random_angles.detach(), log


def minimum_circular_gap(sorted_angles: Tensor) -> Tensor:
    wrapped = torch.cat([sorted_angles, sorted_angles[:1] + 2.0 * math.pi])
    return torch.diff(wrapped).amin()


@torch.no_grad()
def evaluate_design(
    physics: FixedGeometryBornLSM,
    model: TiedAttentionPCGLSMLoop,
    learned_angles: Tensor,
    random_angles: Tensor,
    loop_cfg: LoopConfig,
    run_cfg: RunConfig,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, Tensor]]:
    set_seed(seed + 400_000)
    uniform_angles = (
        torch.arange(run_cfg.design_budget, device=physics.device)
        * (2.0 * math.pi / run_cfg.design_budget)
    )
    geometries = {
        "learned_design": learned_angles,
        "uniform_design": uniform_angles,
        "random_design": random_angles,
    }
    rows: list[dict[str, object]] = []
    showcase: dict[str, Tensor] = {
        "learned_angles": learned_angles.detach().cpu(),
        "uniform_angles": uniform_angles.detach().cpu(),
        "random_angles": random_angles.detach().cpu(),
    }
    for family in ("ellipse", "kite", "two_disks"):
        remaining = loop_cfg.eval_tasks
        task_offset = 0
        while remaining > 0:
            batch_size = min(loop_cfg.eval_batch_size, remaining)
            mask = physics.sample_masks(batch_size, family)
            for geometry_name, angles in geometries.items():
                far_field, probe, _, kernel, feature_kernel = physics.acquisition_at_angles(
                    mask,
                    angles,
                    noise_rel=0.12,
                )
                score, info = model(
                    far_field,
                    probe,
                    kernel=kernel,
                    feature_kernel=feature_kernel,
                )
                local_rows: list[dict[str, object]] = []
                append_evaluation_rows(
                    local_rows,
                    score,
                    mask,
                    physics.grid,
                    info["relative_residual"],
                    seed=seed,
                    method=f"{geometry_name}_loop",
                    family=family,
                    noise=0.12,
                    stage="experiment_design",
                )
                for local_row in local_rows:
                    local_row["task"] = int(local_row["task"]) + task_offset
                rows.extend(local_rows)
                if family == "kite" and task_offset == 0:
                    showcase[f"score_{geometry_name}"] = score[:3].detach().cpu()

                exact_score, exact_info = exact_bayesian_lsm(
                    far_field,
                    probe,
                    kernel,
                    physics.cfg.ridge_rel,
                )
                exact_rows: list[dict[str, object]] = []
                append_evaluation_rows(
                    exact_rows,
                    exact_score,
                    mask,
                    physics.grid,
                    exact_info["relative_residual"],
                    seed=seed,
                    method=f"{geometry_name}_exact",
                    family=family,
                    noise=0.12,
                    stage="experiment_design",
                )
                for local_row in exact_rows:
                    local_row["task"] = int(local_row["task"]) + task_offset
                rows.extend(exact_rows)
            if family == "kite" and task_offset == 0:
                showcase["mask"] = mask[:3].detach().cpu()
            task_offset += batch_size
            remaining -= batch_size
    return rows, showcase


def aggregate_rows(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    metrics = ("auc", "average_precision", "iou_top7", "centroid_error", "relative_residual")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        result = dict(zip(group_keys, key, strict=True))
        result["n_tasks"] = len(group)
        for metric in metrics:
            mean, ci95 = confidence_summary([float(row[metric]) for row in group])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci95"] = ci95
        summary.append(result)
    return summary


def normalise_image(score: np.ndarray) -> np.ndarray:
    lower, upper = np.quantile(score, [0.02, 0.98])
    return np.clip((score - lower) / max(upper - lower, 1.0e-8), 0.0, 1.0)


def plot_reconstructions(
    path: Path,
    showcase: dict[str, Tensor],
    grid_size: int,
    methods: tuple[tuple[str, str], ...],
) -> None:
    n_examples = showcase["mask"].shape[0]
    figure, axes = plt.subplots(n_examples, len(methods) + 1, figsize=(3.0 * (len(methods) + 1), 2.8 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]
    for row in range(n_examples):
        axes[row, 0].imshow(showcase["mask"][row].reshape(grid_size, grid_size), origin="lower", cmap="gray_r")
        axes[row, 0].set_title("Obstacle" if row == 0 else "")
        for column, (key, title) in enumerate(methods, start=1):
            image = showcase[key][row].reshape(grid_size, grid_size).numpy()
            axes[row, column].imshow(normalise_image(image), origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
            axes[row, column].set_title(title if row == 0 else "")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_and_depth(
    path: Path,
    training_rows: list[dict[str, float]],
    depth_rows: list[dict[str, object]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    seeds = sorted({int(row["seed"]) for row in training_rows})
    for seed in seeds:
        local = [row for row in training_rows if int(row["seed"]) == seed]
        axes[0].plot([row["step"] for row in local], [row["loss"] for row in local], alpha=0.8, label=f"seed {seed}")
        axes[1].plot(
            [row["step"] for row in local],
            [row["preconditioner_quadratic"] for row in local],
            alpha=0.8,
        )
    axes[0].set(title="Training loss", xlabel="gradient steps", ylabel="ranking + residual loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1, label="identity PCG")
    axes[1].set(
        title="Learned SPD preconditioner",
        xlabel="gradient steps",
        ylabel="quadratic Laplacian coefficient",
    )

    depths = sorted({int(row["depth"]) for row in depth_rows})
    means = []
    ci = []
    for depth in depths:
        values = [float(row["average_precision"]) for row in depth_rows if int(row["depth"]) == depth]
        mean, interval = confidence_summary(values)
        means.append(mean)
        ci.append(interval)
    axes[2].errorbar(depths, means, yerr=ci, marker="o", capsize=3)
    axes[2].set(title="Tied-depth scaling (OOD kites)", xlabel="loop depth", ylabel="average precision", ylim=(0.0, 1.03))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_metric_bars(
    path: Path,
    rows: list[dict[str, object]],
    *,
    methods: tuple[str, ...],
    noise: float,
    metric: str,
    title: str,
) -> None:
    families = ("ellipse", "kite", "two_disks")
    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    width = 0.8 / len(methods)
    x = np.arange(len(families))
    for method_index, method in enumerate(methods):
        means = []
        intervals = []
        for family in families:
            values = [
                float(row[metric])
                for row in rows
                if row["method"] == method
                and row["family"] == family
                and abs(float(row["noise_rel"]) - noise) < 1.0e-8
            ]
            mean, interval = confidence_summary(values)
            means.append(mean)
            intervals.append(interval)
        offset = (method_index - (len(methods) - 1) / 2) * width
        axis.bar(x + offset, means, width=width, yerr=intervals, capsize=2, label=method.replace("_", " "))
    axis.set_xticks(x, ["ellipse (ID)", "kite (OOD)", "two obstacles (OOD)"])
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_ylim(0.0, 1.03)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_design(
    path: Path,
    design_rows: list[dict[str, object]],
    showcase: dict[str, Tensor],
) -> None:
    figure = plt.figure(figsize=(12.5, 4.0))
    polar = figure.add_subplot(1, 3, 1, projection="polar")
    styles = (
        ("learned_angles", "learned", "C0"),
        ("uniform_angles", "uniform", "C1"),
        ("random_angles", "random", "C2"),
    )
    for key, label, color in styles:
        angles = torch.remainder(showcase[key], 2.0 * math.pi).numpy()
        polar.scatter(angles, np.ones_like(angles), s=45, label=label, color=color)
    polar.set_yticklabels([])
    polar.set_title("Six source/receiver angles")
    polar.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.20), frameon=False, fontsize=8)

    for panel, metric in enumerate(("average_precision", "iou_top7"), start=2):
        axis = figure.add_subplot(1, 3, panel)
        methods = ("learned_design_loop", "uniform_design_loop", "random_design_loop")
        families = ("ellipse", "kite", "two_disks")
        width = 0.24
        x = np.arange(len(families))
        for method_index, method in enumerate(methods):
            means = []
            intervals = []
            for family in families:
                values = [float(row[metric]) for row in design_rows if row["method"] == method and row["family"] == family]
                mean, interval = confidence_summary(values)
                means.append(mean)
                intervals.append(interval)
            axis.bar(x + (method_index - 1) * width, means, width=width, yerr=intervals, capsize=2, label=method.split("_")[0])
        axis.set_xticks(x, ["ellipse", "kite", "two disks"], rotation=12)
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_angle_rows(path: Path, angles_by_seed: dict[int, dict[str, Tensor]]) -> None:
    rows: list[dict[str, object]] = []
    for seed, methods in angles_by_seed.items():
        for method, angles in methods.items():
            degrees = torch.sort(torch.remainder(angles, 2.0 * math.pi)).values * 180.0 / math.pi
            for index, value in enumerate(degrees.cpu().numpy()):
                rows.append({"seed": seed, "design": method, "index": index, "angle_degrees": float(value)})
    write_rows(path, rows)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    physics_cfg = PhysicsConfig(
        n_angles=24 if args.quick else 32,
        grid_size=20 if args.quick else 32,
        noise_rel=0.08,
        ridge_rel=0.01,
        wavenumber=8.0,
        kernel_gamma=1.0,
        kernel_mix=0.20,
    )
    loop_cfg = LoopConfig(
        depth=8,
        controller_width=24 if args.quick else 32,
        steps=40 if args.quick else 1500,
        batch_size=6 if args.quick else 16,
        learning_rate=1.0e-3,
        log_every=10 if args.quick else 50,
        eval_tasks=12 if args.quick else 192,
        eval_batch_size=6 if args.quick else 24,
        rank_pairs=16 if args.quick else 48,
    )
    supplied_seeds = tuple(args.seeds) if args.seeds else None
    run_cfg = RunConfig(
        seeds=supplied_seeds or ((17,) if args.quick else (17, 29, 43)),
        design_steps=30 if args.quick else 900,
        design_batch_size=6 if args.quick else 20,
    )
    physics = FixedGeometryBornLSM(physics_cfg, device)
    protocol = {
        "description": "2D active deterministic multistatic LSM; no random-source correlation",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "physics": asdict(physics_cfg),
        "loop": asdict(loop_cfg),
        "run": asdict(run_cfg),
        "training_target": "obstacle ranking plus sampling-equation residual; no direct-solve imitation",
        "learned_parameters": "tied SPD attention preconditioner, followed by six acquisition angles",
        "fixed_objects": "Helmholtz/Born operator, angular softmax kernel, ridge model, probe grid",
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    training_rows: list[dict[str, float]] = []
    fixed_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    design_training_rows: list[dict[str, float]] = []
    design_rows: list[dict[str, object]] = []
    angles_by_seed: dict[int, dict[str, Tensor]] = {}
    fixed_showcase: dict[str, Tensor] | None = None
    design_showcase: dict[str, Tensor] | None = None
    total_start = time.perf_counter()

    for seed in run_cfg.seeds:
        model, local_training = train_loop(physics, loop_cfg, run_cfg, seed)
        training_rows.extend(local_training)
        torch.save(
            {
                "model": model.state_dict(),
                "physics": asdict(physics_cfg),
                "loop": asdict(loop_cfg),
                "seed": seed,
            },
            output_dir / f"loop_seed_{seed}.pt",
        )
        local_fixed, local_showcase = evaluate_fixed_geometry(physics, model, loop_cfg, run_cfg, seed)
        fixed_rows.extend(local_fixed)
        depth_rows.extend(evaluate_depths(physics, model, loop_cfg, seed))
        if fixed_showcase is None:
            fixed_showcase = local_showcase

        learned_angles, random_angles, local_design_training = train_design(physics, model, run_cfg, seed)
        design_training_rows.extend(local_design_training)
        local_design, local_design_showcase = evaluate_design(
            physics,
            model,
            learned_angles,
            random_angles,
            loop_cfg,
            run_cfg,
            seed,
        )
        design_rows.extend(local_design)
        uniform_angles = torch.arange(run_cfg.design_budget, device=device) * (2.0 * math.pi / run_cfg.design_budget)
        angles_by_seed[seed] = {
            "learned": learned_angles,
            "uniform": uniform_angles,
            "random": random_angles,
        }
        if design_showcase is None:
            design_showcase = local_design_showcase

    assert fixed_showcase is not None and design_showcase is not None
    write_rows(output_dir / "loop_training.csv", training_rows)
    write_rows(output_dir / "fixed_geometry_tasks.csv", fixed_rows)
    write_rows(output_dir / "depth_tasks.csv", depth_rows)
    write_rows(output_dir / "design_training.csv", design_training_rows)
    write_rows(output_dir / "design_tasks.csv", design_rows)
    write_rows(
        output_dir / "fixed_geometry_summary.csv",
        aggregate_rows(fixed_rows, ("method", "family", "noise_rel")),
    )
    write_rows(
        output_dir / "design_summary.csv",
        aggregate_rows(design_rows, ("method", "family", "noise_rel")),
    )
    save_angle_rows(output_dir / "design_angles.csv", angles_by_seed)

    plot_reconstructions(
        output_dir / "fixed_geometry_reconstructions.png",
        fixed_showcase,
        physics_cfg.grid_size,
        (
            ("score_identity_cg_8", "identity CG, 8 loops"),
            ("score_trained_pcg_8", "trained PCG, 8 loops"),
            ("score_trained_pcg_20", "trained PCG, 20 loops"),
            ("score_exact_tikhonov", "exact Tikhonov"),
        ),
    )
    plot_training_and_depth(output_dir / "training_and_depth.png", training_rows, depth_rows)
    plot_metric_bars(
        output_dir / "fixed_geometry_average_precision.png",
        fixed_rows,
        methods=("identity_cg_8", "trained_pcg_8", "trained_pcg_20", "exact_tikhonov"),
        noise=run_cfg.evaluation_noise[-1],
        metric="average_precision",
        title=f"Fixed geometry at {100 * run_cfg.evaluation_noise[-1]:.0f}% relative noise",
    )
    plot_design(output_dir / "experiment_design.png", design_rows, design_showcase)
    plot_reconstructions(
        output_dir / "design_reconstructions.png",
        design_showcase,
        physics_cfg.grid_size,
        (
            ("score_learned_design", "learned angles"),
            ("score_uniform_design", "uniform angles"),
            ("score_random_design", "random angles"),
        ),
    )
    protocol["elapsed_seconds"] = time.perf_counter() - total_start
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"completed in {protocol['elapsed_seconds']:.1f}s: {output_dir}")


if __name__ == "__main__":
    main()
