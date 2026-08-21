#!/usr/bin/env python3
"""Train and audit a larger multi-obstacle Bayesian LSM foundation controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

from .foundation_uq import (
    FoundationPCGLSMLoop,
    balanced_brier,
    balanced_nll,
    occupancy_probability,
    posterior_covariance,
    posterior_score_moments,
    sample_multi_obstacle_masks,
)
from .lsm_core import (
    FixedGeometryBornLSM,
    PhysicsConfig,
    TiedAttentionPCGLSMLoop,
    exact_bayesian_lsm,
    ranking_loss,
    set_seed,
)
from .run_experiments import normalise_image, task_metrics, write_rows


@dataclass(frozen=True)
class Scenario:
    name: str
    count: int
    mode: str
    wavenumber: float
    noise: float
    aperture_degrees: float = 360.0
    heteroscedastic: bool = False
    category: str = "ID"


SCENARIOS = (
    Scenario("one obstacle", 1, "mixed", 8.0, 0.05),
    Scenario("two obstacles", 2, "mixed", 8.0, 0.15),
    Scenario("three obstacles", 3, "mixed", 8.0, 0.15),
    Scenario("four obstacles", 4, "mixed", 8.0, 0.15),
    Scenario("six obstacles", 6, "mixed", 8.0, 0.15, category="OOD count"),
    Scenario("three stars", 3, "star", 8.0, 0.15, category="OOD shape"),
    Scenario("two crescents", 2, "crescent", 8.0, 0.15, category="OOD shape"),
    Scenario("30% noise", 4, "mixed", 8.0, 0.30, category="OOD noise"),
    Scenario(
        "180-degree aperture",
        3,
        "mixed",
        8.0,
        0.15,
        aperture_degrees=180.0,
        category="OOD aperture",
    ),
    Scenario(
        "heteroscedastic noise",
        3,
        "mixed",
        8.0,
        0.20,
        heteroscedastic=True,
        category="OOD noise",
    ),
    Scenario("wavenumber 12", 3, "mixed", 12.0, 0.15, category="OOD frequency"),
    Scenario(
        "mirrored kites",
        2,
        "mirrored_kites",
        8.0,
        0.15,
        category="lsmlab-inspired",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument("--eval-tasks", type=int, default=128)
    parser.add_argument("--seeds", default="17,29,43")
    return parser.parse_args()


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def physics_cache(device: torch.device) -> dict[float, FixedGeometryBornLSM]:
    return {
        wavenumber: FixedGeometryBornLSM(
            PhysicsConfig(
                n_angles=32,
                grid_size=32,
                domain_half_width=0.8,
                wavenumber=wavenumber,
                noise_rel=0.08,
                ridge_rel=0.01,
                kernel_gamma=1.0,
                kernel_mix=0.20,
            ),
            device,
        )
        for wavenumber in (6.0, 8.0, 10.0, 12.0)
    }


def aperture_angles(
    physics: FixedGeometryBornLSM,
    aperture_degrees: float,
    *,
    rotation: float = 0.0,
) -> Tensor:
    if aperture_degrees >= 359.9:
        return physics.angles + float(rotation)
    half = math.radians(aperture_degrees) / 2.0
    return torch.linspace(
        -half,
        half,
        physics.cfg.n_angles,
        device=physics.device,
        dtype=physics.real_dtype,
    ) + float(rotation)


def acquire(
    physics: FixedGeometryBornLSM,
    mask: Tensor,
    *,
    noise: float,
    aperture_degrees: float,
    heteroscedastic: bool = False,
    rotation: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if aperture_degrees >= 359.9 and not heteroscedastic and rotation == 0.0:
        far_field = physics.far_field_from_mask(mask, noise_rel=noise)
        return far_field, physics.probe_rhs, physics.kernel, physics.feature_kernel
    if aperture_degrees >= 359.9 and heteroscedastic:
        profile = torch.linspace(
            0.25,
            2.0,
            physics.cfg.n_angles,
            device=physics.device,
        )
        far_field = physics.far_field_from_mask(
            mask,
            noise_rel=noise,
            receiver_noise_profile=profile,
        )
        return far_field, physics.probe_rhs, physics.kernel, physics.feature_kernel
    angles = aperture_angles(physics, aperture_degrees, rotation=rotation)
    far_field, probe, _, kernel, feature = physics.acquisition_at_angles(
        mask,
        angles,
        noise_rel=noise,
    )
    return far_field, probe, kernel, feature


def make_model(
    kind: str,
    physics: FixedGeometryBornLSM,
    *,
    depth: int,
) -> nn.Module:
    if kind == "small":
        return TiedAttentionPCGLSMLoop(
            physics.kernel,
            physics.feature_kernel,
            physics.cfg.ridge_rel,
            depth=depth,
            controller_width=32,
        ).to(physics.device)
    if kind == "foundation":
        return FoundationPCGLSMLoop(
            physics.kernel,
            physics.feature_kernel,
            physics.cfg.ridge_rel,
            depth=depth,
            width=192,
            n_blocks=6,
            expansion=384,
            polynomial_degree=8,
        ).to(physics.device)
    raise ValueError(f"unknown model kind: {kind}")


def draw_training_batch(
    cache: dict[float, FixedGeometryBornLSM],
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, object]]:
    wavenumber = random.choice((6.0, 8.0, 10.0))
    physics = cache[wavenumber]
    counts = torch.randint(1, 5, (batch_size,), device=physics.device)
    mask, _, _ = sample_multi_obstacle_masks(physics, batch_size, counts, mode="mixed")
    noise = 0.02 + 0.18 * random.random()
    aperture = 240.0 if random.random() < 0.25 else 360.0
    rotation = 2.0 * math.pi * random.random() if aperture < 360.0 else 0.0
    far_field, probe, kernel, feature = acquire(
        physics,
        mask,
        noise=noise,
        aperture_degrees=aperture,
        rotation=rotation,
    )
    metadata: dict[str, object] = {
        "wavenumber": wavenumber,
        "noise": noise,
        "aperture_degrees": aperture,
        "mean_count": float(counts.float().mean()),
    }
    return far_field, probe, kernel, feature, mask, metadata


def train_model(
    kind: str,
    seed: int,
    cache: dict[float, FixedGeometryBornLSM],
    *,
    steps: int,
    batch_size: int,
    depth: int,
    log_every: int,
) -> tuple[nn.Module, list[dict[str, object]]]:
    set_seed(seed + (0 if kind == "small" else 50_000))
    model = make_model(kind, cache[8.0], depth=depth)
    learning_rate = 1.0e-3 if kind == "small" else 3.0e-4
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=1.0e-5
    )
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for step in range(steps + 1):
        # The data seed is independent of model size, giving a controlled
        # same-curriculum comparison.
        set_seed(seed * 1_000_000 + step + 91_000)
        far_field, probe, kernel, feature, mask, metadata = draw_training_batch(
            cache, batch_size
        )
        score, info = model(
            far_field,
            probe,
            kernel=kernel,
            feature_kernel=feature,
            depth=depth,
        )
        rank = ranking_loss(score, mask, n_pairs=64 if steps >= 200 else 16)
        residual_penalty = torch.log1p(info["relative_residual"]).mean()
        coefficient_penalty = info["preconditioner_coefficients"].square().mean()
        loss = rank + 0.25 * residual_penalty + 1.0e-5 * coefficient_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            row: dict[str, object] = {
                "model": kind,
                "seed": seed,
                "step": step,
                "loss": float(loss.detach()),
                "ranking_loss": float(rank.detach()),
                "relative_residual": float(info["relative_residual"].mean().detach()),
                "coefficient_mean": float(
                    info["preconditioner_coefficients"].mean().detach()
                ),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - start,
                **metadata,
            }
            rows.append(row)
            print(
                f"{kind:10s} seed={seed} step={step:04d} "
                f"loss={float(loss):.4f} residual="
                f"{float(info['relative_residual'].mean()):.4f}"
            )
    model.eval()
    return model, rows


def area_matched_iou(score: Tensor, mask: Tensor) -> np.ndarray:
    output = []
    for task_score, task_mask in zip(score, mask, strict=True):
        area = int(task_mask.sum().item())
        indices = torch.topk(task_score, max(area, 1)).indices
        prediction = torch.zeros_like(task_mask)
        prediction[indices] = True
        intersection = (prediction & task_mask).sum().item()
        union = (prediction | task_mask).sum().item()
        output.append(intersection / max(union, 1))
    return np.asarray(output)


def score_correlation(score: Tensor, exact_score: Tensor) -> np.ndarray:
    score = score.double()
    exact_score = exact_score.double()
    score = score - score.mean(dim=-1, keepdim=True)
    exact_score = exact_score - exact_score.mean(dim=-1, keepdim=True)
    numerator = (score * exact_score).sum(dim=-1)
    denominator = torch.sqrt(
        score.square().sum(dim=-1) * exact_score.square().sum(dim=-1)
    ).clamp_min(1.0e-12)
    return (numerator / denominator).cpu().numpy()


def balanced_ece(probability: Tensor, target: Tensor, bins: int = 12) -> np.ndarray:
    output = []
    for task_probability, task_target in zip(probability, target, strict=True):
        target_float = task_target.float()
        positive_count = target_float.sum().clamp_min(1.0)
        negative_count = (1.0 - target_float).sum().clamp_min(1.0)
        weights = torch.where(
            task_target,
            0.5 / positive_count,
            0.5 / negative_count,
        )
        value = torch.zeros((), device=probability.device)
        for index in range(bins):
            lower = index / bins
            upper = (index + 1) / bins
            selected = (task_probability >= lower) & (
                task_probability < upper if index + 1 < bins else task_probability <= upper
            )
            selected_weight = weights[selected].sum()
            if selected_weight > 0:
                confidence = (weights[selected] * task_probability[selected]).sum()
                confidence = confidence / selected_weight
                accuracy = (weights[selected] * target_float[selected]).sum()
                accuracy = accuracy / selected_weight
                value = value + selected_weight * torch.abs(accuracy - confidence)
        output.append(float(value))
    return np.asarray(output)


def uncertainty_error_auc(probability: Tensor, target: Tensor) -> np.ndarray:
    output = []
    prediction = probability >= 0.5
    entropy = -(
        probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        * torch.log(probability.clamp(1.0e-6, 1.0 - 1.0e-6))
        + (1.0 - probability).clamp(1.0e-6, 1.0 - 1.0e-6)
        * torch.log((1.0 - probability).clamp(1.0e-6, 1.0 - 1.0e-6))
    )
    for task_prediction, task_entropy, task_target in zip(
        prediction, entropy, target, strict=True
    ):
        error = (task_prediction != task_target).cpu().numpy().astype(np.int64)
        if error.min() == error.max():
            output.append(0.5)
        else:
            output.append(roc_auc_score(error, task_entropy.cpu().numpy()))
    return np.asarray(output)


@torch.no_grad()
def uq_outputs(
    model: nn.Module | None,
    far_field: Tensor,
    probe: Tensor,
    kernel: Tensor,
    feature: Tensor,
    *,
    depth: int,
    ridge_rel: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if kernel.ndim == 2:
        kernel_batch = kernel.unsqueeze(0).expand(far_field.shape[0], -1, -1)
    else:
        kernel_batch = kernel
    rhs_fk = far_field @ kernel_batch
    if model is None:
        score, mean_info = exact_bayesian_lsm(
            far_field, probe, kernel, ridge_rel
        )
        _, covariance_info = exact_bayesian_lsm(
            far_field, rhs_fk, kernel, ridge_rel
        )
    else:
        score, mean_info = model(
            far_field,
            probe,
            kernel=kernel,
            feature_kernel=feature,
            depth=depth,
        )
        _, covariance_info = model(
            far_field,
            rhs_fk,
            kernel=kernel,
            feature_kernel=feature,
            depth=depth,
        )
    covariance = posterior_covariance(
        far_field,
        kernel,
        covariance_info["q"],
    )
    score_mean, score_std = posterior_score_moments(
        mean_info["coefficients"], covariance, kernel
    )
    return score, score_mean, score_std, mean_info["relative_residual"]


@torch.no_grad()
def calibrate_threshold(
    model: nn.Module | None,
    cache: dict[float, FixedGeometryBornLSM],
    seed: int,
    *,
    depth: int,
    tasks: int,
) -> tuple[float, dict[str, float]]:
    set_seed(seed + 2_000_000)
    means = []
    standard_deviations = []
    masks = []
    completed = 0
    while completed < tasks:
        batch_size = min(12, tasks - completed)
        far_field, probe, kernel, feature, mask, _ = draw_training_batch(
            cache, batch_size
        )
        _, score_mean, score_std, _ = uq_outputs(
            model,
            far_field,
            probe,
            kernel,
            feature,
            depth=depth,
            ridge_rel=cache[8.0].cfg.ridge_rel,
        )
        means.append(score_mean)
        standard_deviations.append(score_std)
        masks.append(mask)
        completed += batch_size
    mean = torch.cat(means)
    standard_deviation = torch.cat(standard_deviations)
    target = torch.cat(masks)
    lower = torch.quantile(mean, 0.02).item()
    upper = torch.quantile(mean, 0.98).item()
    candidates = torch.linspace(lower, upper, 81, device=mean.device)
    risks = []
    for candidate in candidates:
        probability = occupancy_probability(mean, standard_deviation, float(candidate))
        risks.append(balanced_brier(probability, target).mean())
    risk_tensor = torch.stack(risks)
    best = int(risk_tensor.argmin().item())
    threshold = float(candidates[best])
    probability = occupancy_probability(mean, standard_deviation, threshold)
    return threshold, {
        "calibration_brier": float(balanced_brier(probability, target).mean()),
        "calibration_nll": float(balanced_nll(probability, target).mean()),
        "calibration_ece": float(balanced_ece(probability, target).mean()),
    }


@torch.no_grad()
def evaluate_seed(
    small: nn.Module,
    foundation: nn.Module,
    cache: dict[float, FixedGeometryBornLSM],
    seed: int,
    *,
    depth: int,
    eval_tasks: int,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Tensor]]:
    localization_rows: list[dict[str, object]] = []
    uq_rows: list[dict[str, object]] = []
    showcase: dict[str, Tensor] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        set_seed(seed + 3_000_000 + 10_000 * scenario_index)
        physics = cache[scenario.wavenumber]
        completed = 0
        while completed < eval_tasks:
            batch_size = min(16, eval_tasks - completed)
            mask, _, _ = sample_multi_obstacle_masks(
                physics,
                batch_size,
                scenario.count,
                mode=scenario.mode,
            )
            far_field, probe, kernel, feature = acquire(
                physics,
                mask,
                noise=scenario.noise,
                aperture_degrees=scenario.aperture_degrees,
                heteroscedastic=scenario.heteroscedastic,
            )
            small_score, small_info = small(
                far_field,
                probe,
                kernel=kernel,
                feature_kernel=feature,
                depth=depth,
            )
            foundation_score, score_mean, score_std, foundation_residual = uq_outputs(
                foundation,
                far_field,
                probe,
                kernel,
                feature,
                depth=depth,
                ridge_rel=physics.cfg.ridge_rel,
            )
            exact_score, exact_mean, exact_std, exact_residual = uq_outputs(
                None,
                far_field,
                probe,
                kernel,
                feature,
                depth=depth,
                ridge_rel=physics.cfg.ridge_rel,
            )
            scores = {
                "small": (small_score, small_info["relative_residual"]),
                "foundation": (foundation_score, foundation_residual),
                "exact": (exact_score, exact_residual),
            }
            for model_name, (score, residual) in scores.items():
                metrics = task_metrics(score, mask, physics.grid)
                area_iou = area_matched_iou(score, mask)
                correlation = score_correlation(score, exact_score)
                for task_index, metric in enumerate(metrics):
                    localization_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "scenario": scenario.name,
                            "category": scenario.category,
                            "count": scenario.count,
                            "wavenumber": scenario.wavenumber,
                            "noise": scenario.noise,
                            "aperture_degrees": scenario.aperture_degrees,
                            "task": completed + task_index,
                            **metric,
                            "area_matched_iou": float(area_iou[task_index]),
                            "score_correlation_exact": float(correlation[task_index]),
                            "relative_residual": float(residual[task_index]),
                            "success_ap80": float(metric["average_precision"] >= 0.80),
                        }
                    )

            for model_name, mean, standard_deviation in (
                ("foundation", score_mean, score_std),
                ("exact", exact_mean, exact_std),
            ):
                threshold = thresholds[model_name]
                probability = occupancy_probability(mean, standard_deviation, threshold)
                brier = balanced_brier(probability, mask).cpu().numpy()
                nll = balanced_nll(probability, mask).cpu().numpy()
                ece = balanced_ece(probability, mask)
                error_auc = uncertainty_error_auc(probability, mask)
                lower = mean - NormalDist().inv_cdf(0.95) * standard_deviation
                upper = mean + NormalDist().inv_cdf(0.95) * standard_deviation
                confident = (lower > threshold) | (upper < threshold)
                prediction = mean > threshold
                for task_index in range(batch_size):
                    selected = confident[task_index]
                    selective_accuracy = (
                        (prediction[task_index, selected] == mask[task_index, selected])
                        .float()
                        .mean()
                        if selected.any()
                        else torch.tensor(1.0, device=mask.device)
                    )
                    uq_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "scenario": scenario.name,
                            "category": scenario.category,
                            "task": completed + task_index,
                            "threshold": threshold,
                            "balanced_brier": float(brier[task_index]),
                            "balanced_nll": float(nll[task_index]),
                            "balanced_ece": float(ece[task_index]),
                            "uncertainty_error_auc": float(error_auc[task_index]),
                            "credible_confident_fraction": float(selected.float().mean()),
                            "credible_selective_accuracy": float(selective_accuracy),
                            "mean_score_std": float(standard_deviation[task_index].mean()),
                        }
                    )
                if seed == 17 and scenario.name in (
                    "four obstacles",
                    "six obstacles",
                    "three stars",
                    "180-degree aperture",
                ) and completed == 0:
                    key = scenario.name.replace(" ", "_")
                    showcase[f"{key}_mask"] = mask[:1].cpu()
                    showcase[f"{key}_small"] = small_score[:1].cpu()
                    showcase[f"{key}_foundation"] = foundation_score[:1].cpu()
                    showcase[f"{key}_exact"] = exact_score[:1].cpu()
                    showcase[f"{key}_probability"] = probability[:1].cpu()
                    showcase[f"{key}_std"] = standard_deviation[:1].cpu()
            completed += batch_size
    return localization_rows, uq_rows, showcase


def aggregate(
    rows: list[dict[str, object]],
    value_columns: tuple[str, ...],
    *,
    keys: tuple[str, ...] = ("model", "scenario", "category"),
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        groups.setdefault(group_key, []).append(row)
    output = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        result = {key: value for key, value in zip(keys, group_key, strict=True)}
        result["n_tasks"] = len(group_rows)
        for column in value_columns:
            values = np.asarray([float(row[column]) for row in group_rows])
            result[f"{column}_mean"] = float(values.mean())
            result[f"{column}_ci95"] = float(
                1.96 * values.std(ddof=1) / math.sqrt(len(values))
            )
            result[f"{column}_q10"] = float(np.quantile(values, 0.10))
        output.append(result)
    return output


@torch.no_grad()
def benchmark(
    models: dict[str, nn.Module],
    cache: dict[float, FixedGeometryBornLSM],
) -> dict[str, float]:
    physics = cache[8.0]
    mask, _, _ = sample_multi_obstacle_masks(physics, 16, 4, mode="mixed")
    far_field, probe, kernel, feature = acquire(
        physics,
        mask,
        noise=0.15,
        aperture_degrees=360.0,
    )
    operations = {
        **{
            name: (
                lambda model=model: model(
                    far_field,
                    probe,
                    kernel=kernel,
                    feature_kernel=feature,
                    depth=16,
                )
            )
            for name, model in models.items()
        },
        "exact": lambda: exact_bayesian_lsm(
            far_field,
            probe,
            kernel,
            physics.cfg.ridge_rel,
        ),
    }
    result = {}
    for name, operation in operations.items():
        for _ in range(10):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        result[name] = 1000.0 * (time.perf_counter() - start) / 50.0
    return result


def plot_scenario_summary(path: Path, summary: list[dict[str, object]]) -> None:
    lookup = {
        (str(row["model"]), str(row["scenario"])): row
        for row in summary
    }
    names = [scenario.name for scenario in SCENARIOS]
    positions = np.arange(len(names))
    colors = {"small": "#59a14f", "foundation": "#f28e2b", "exact": "#4e79a7"}
    figure, axes = plt.subplots(2, 1, figsize=(14.5, 8.2), sharex=True)
    width = 0.25
    for offset, model in zip((-width, 0.0, width), colors, strict=True):
        ap = [lookup[(model, name)]["average_precision_mean"] for name in names]
        axes[0].bar(positions + offset, ap, width, label=model, color=colors[model])
        residual = [lookup[(model, name)]["relative_residual_mean"] for name in names]
        axes[1].plot(
            positions,
            residual,
            marker="o",
            linewidth=2,
            label=model,
            color=colors[model],
        )
    axes[0].set_ylabel("average precision")
    axes[0].set_ylim(0.0, 1.03)
    axes[0].legend(ncol=3)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("relative residual")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(positions, names, rotation=28, ha="right")
    figure.suptitle("Multi-obstacle effectiveness, accuracy, and robustness")
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_uq_summary(path: Path, summary: list[dict[str, object]]) -> None:
    lookup = {
        (str(row["model"]), str(row["scenario"])): row
        for row in summary
    }
    names = [scenario.name for scenario in SCENARIOS]
    positions = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    for model, color in (("foundation", "#f28e2b"), ("exact", "#4e79a7")):
        axes[0].plot(
            positions,
            [lookup[(model, name)]["balanced_brier_mean"] for name in names],
            marker="o",
            label=model,
            color=color,
        )
        axes[1].plot(
            positions,
            [lookup[(model, name)]["balanced_ece_mean"] for name in names],
            marker="o",
            label=model,
            color=color,
        )
        axes[2].plot(
            positions,
            [
                lookup[(model, name)]["uncertainty_error_auc_mean"]
                for name in names
            ],
            marker="o",
            label=model,
            color=color,
        )
    axes[0].set_title("balanced Brier (lower is better)")
    axes[1].set_title("balanced ECE (lower is better)")
    axes[2].set_title("uncertainty detects errors (AUROC)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xticks(positions, [str(index + 1) for index in positions])
    axes[0].legend()
    figure.supxlabel("scenario index in the main results table")
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_showcase(path: Path, showcase: dict[str, Tensor], grid_size: int) -> None:
    scenarios = (
        "four_obstacles",
        "six_obstacles",
        "three_stars",
        "180-degree_aperture",
    )
    columns = ("mask", "small", "foundation", "exact", "probability", "std")
    titles = ("obstacle", "small", "foundation", "exact", "occupancy P", "score std")
    figure, axes = plt.subplots(len(scenarios), len(columns), figsize=(14.8, 10.0))
    for row, scenario in enumerate(scenarios):
        for column, (key, title) in enumerate(zip(columns, titles, strict=True)):
            value = showcase[f"{scenario}_{key}"][0].reshape(grid_size, grid_size)
            if key in ("small", "foundation", "exact"):
                value = normalise_image(value[None])[0]
            image = axes[row, column].imshow(
                value,
                origin="lower",
                cmap="magma" if key != "mask" else "gray_r",
                vmin=0.0 if key in ("mask", "probability") else None,
                vmax=1.0 if key in ("mask", "probability", "small", "foundation", "exact") else None,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(title)
            if column == 0:
                axes[row, column].set_ylabel(scenario.replace("_", " "))
            if key in ("probability", "std"):
                figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.02)
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_readable_scenario_map(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", *asdict(SCENARIOS[0]).keys()])
        writer.writeheader()
        for index, scenario in enumerate(SCENARIOS, start=1):
            writer.writerow({"index": index, **asdict(scenario)})


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = physics_cache(device)
    requested_seeds = tuple(int(value) for value in args.seeds.split(","))
    seeds = requested_seeds[:1] if args.quick else requested_seeds
    steps = min(args.steps, 80) if args.quick else args.steps
    eval_tasks = min(args.eval_tasks, 16) if args.quick else args.eval_tasks
    calibration_tasks = 16 if args.quick else 96
    batch_size = 4 if args.quick else 12
    depth = 8 if args.quick else 16
    log_every = 20 if args.quick else 100

    all_training_rows: list[dict[str, object]] = []
    all_localization_rows: list[dict[str, object]] = []
    all_uq_rows: list[dict[str, object]] = []
    calibration: dict[str, dict[str, object]] = {}
    runtime_by_seed: dict[str, dict[str, float]] = {}
    showcase: dict[str, Tensor] = {}
    parameter_counts: dict[str, int] = {}
    total_start = time.perf_counter()

    for seed in seeds:
        models = {}
        for kind in ("small", "foundation"):
            model, training_rows = train_model(
                kind,
                seed,
                cache,
                steps=steps,
                batch_size=batch_size,
                depth=depth,
                log_every=log_every,
            )
            models[kind] = model
            all_training_rows.extend(training_rows)
            parameter_counts[kind] = model_parameter_count(model)
            torch.save(
                {
                    "model": model.state_dict(),
                    "kind": kind,
                    "seed": seed,
                    "depth": depth,
                    "steps": steps,
                },
                output_dir / f"{kind}_seed_{seed}.pt",
            )

        thresholds = {}
        for model_name, model in (("foundation", models["foundation"]), ("exact", None)):
            threshold, calibration_metrics = calibrate_threshold(
                model,
                cache,
                seed,
                depth=depth,
                tasks=calibration_tasks,
            )
            thresholds[model_name] = threshold
            calibration[f"{seed}_{model_name}"] = {
                "threshold": threshold,
                **calibration_metrics,
            }

        localization_rows, uq_rows, seed_showcase = evaluate_seed(
            models["small"],
            models["foundation"],
            cache,
            seed,
            depth=depth,
            eval_tasks=eval_tasks,
            thresholds=thresholds,
        )
        all_localization_rows.extend(localization_rows)
        all_uq_rows.extend(uq_rows)
        showcase.update(seed_showcase)
        runtime_by_seed[str(seed)] = benchmark(models, cache)

    localization_summary = aggregate(
        all_localization_rows,
        (
            "average_precision",
            "auc",
            "area_matched_iou",
            "score_correlation_exact",
            "relative_residual",
            "success_ap80",
        ),
    )
    uq_summary = aggregate(
        all_uq_rows,
        (
            "balanced_brier",
            "balanced_nll",
            "balanced_ece",
            "uncertainty_error_auc",
            "credible_confident_fraction",
            "credible_selective_accuracy",
            "mean_score_std",
        ),
    )
    write_rows(output_dir / "training.csv", all_training_rows)
    write_rows(output_dir / "localization_tasks.csv", all_localization_rows)
    write_rows(output_dir / "uq_tasks.csv", all_uq_rows)
    write_rows(output_dir / "summary_localization.csv", localization_summary)
    write_rows(output_dir / "summary_uq.csv", uq_summary)
    write_readable_scenario_map(output_dir / "scenarios.csv")
    with (output_dir / "calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
    with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime_by_seed, handle, indent=2)
    protocol = {
        "description": "larger fixed-kernel PCG foundation controller with analytic GP UQ",
        "physics": asdict(cache[8.0].cfg),
        "training_wavenumbers": [6.0, 8.0, 10.0],
        "training_obstacle_counts": [1, 2, 3, 4],
        "training_noise_range": [0.02, 0.20],
        "training_apertures_degrees": [240.0, 360.0],
        "seeds": list(seeds),
        "steps": steps,
        "depth": depth,
        "batch_size": batch_size,
        "eval_tasks_per_seed_and_scenario": eval_tasks,
        "calibration_tasks_per_seed": calibration_tasks,
        "trainable_parameter_counts": parameter_counts,
        "fixed_kernel": "von Mises softmax, gamma=1, mix=0.2; no learned temperature",
        "uq": "proper-complex GP posterior quadratic moments plus log-normal score match",
        "elapsed_seconds": time.perf_counter() - total_start,
    }
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    plot_scenario_summary(output_dir / "foundation_scenarios.png", localization_summary)
    plot_uq_summary(output_dir / "uq_summary.png", uq_summary)
    plot_showcase(output_dir / "multi_obstacle_uq.png", showcase, cache[8.0].cfg.grid_size)
    print(f"completed in {protocol['elapsed_seconds']:.1f}s: {output_dir}")


if __name__ == "__main__":
    main()
