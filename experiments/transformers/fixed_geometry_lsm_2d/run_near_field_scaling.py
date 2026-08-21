#!/usr/bin/env python3
"""Multi-axis scaling laws for original 2D near-field Bayesian LSM.

The experiment varies the number of unique training inverse problems,
the controller/preconditioner width, and the number of source/receiver tokens.
Every learned solver is evaluated on the same held-out physical MFS tasks.  CG
and exact solves are computed once on those tasks as training-free controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .lsm_core import ranking_loss, set_seed
from .near_field_lsm import (
    NearFieldConfig,
    NearFieldSoundSoftLSM,
    PosteriorMomentLSMLoop,
    exact_near_field_lsm,
)
from .run_experiments import task_metrics
from .run_foundation_uq import area_matched_iou, score_correlation


@dataclass(frozen=True)
class ScalingScenario:
    name: str
    regime: str
    count: int
    mode: str
    wavenumber: float
    noise: float
    aperture: float = 360.0
    jitter: float = 0.05


SCENARIOS = (
    ScalingScenario("ID one obstacle", "ID", 1, "mixed", 8.0, 0.10),
    ScalingScenario("ID four obstacles", "ID", 4, "mixed", 8.0, 0.15),
    ScalingScenario("OOD six obstacles", "OOD count", 6, "mixed", 8.0, 0.15),
    ScalingScenario("OOD stars", "OOD shape", 3, "star", 8.0, 0.15),
    ScalingScenario("OOD 30 percent noise", "OOD noise", 4, "mixed", 8.0, 0.30),
    ScalingScenario(
        "OOD half aperture", "OOD aperture", 3, "mixed", 8.0, 0.15, 180.0, 0.20
    ),
    ScalingScenario("OOD wavenumber 12", "OOD frequency", 3, "mixed", 12.0, 0.15),
)

LEARNED_METHODS = (
    "pcg",
    "heavy_ball",
    "chebyshev",
    "richardson",
    "context_pcg",
    "hybrid_pcg",
)
METHOD_LABELS = {
    "pcg": "population-PCG",
    "context_pcg": "context-PCG",
    "hybrid_pcg": "hybrid-PCG",
    "heavy_ball": "looped-HB",
    "chebyshev": "looped-Chebyshev",
    "richardson": "looped-Richardson",
}


@dataclass
class EvaluationBatch:
    context_size: int
    scenario: ScalingScenario
    near_field: Tensor
    probe: Tensor
    kernel: Tensor
    feature: Tensor
    mask: Tensor
    exact_score: Tensor
    exact_info: dict[str, Tensor]
    grid: Tensor


def comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--widths", default="32,64,128,256,512")
    parser.add_argument("--contexts", default="8,12,16,24,32,48")
    parser.add_argument("--train-contexts", default="12,16,24,32")
    parser.add_argument(
        "--dataset-sizes",
        default="128,256,512,1024,2048,4096,8192,16384,32768",
    )
    parser.add_argument("--methods", default=",".join(LEARNED_METHODS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-tasks", type=int, default=8)
    parser.add_argument("--train-depth", type=int, default=16)
    parser.add_argument("--eval-depth", type=int, default=32)
    parser.add_argument("--moment-degree", type=int, default=6)
    parser.add_argument("--sketch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.08)
    parser.add_argument("--deadline-utc", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def deadline_timestamp(value: str) -> float | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def existing_keys(path: Path, columns: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            tuple(row[column] for column in columns) for row in csv.DictReader(handle)
        }


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_physics_cache(
    contexts: Iterable[int], device: torch.device
) -> dict[tuple[int, float], NearFieldSoundSoftLSM]:
    return {
        (context, wavenumber): NearFieldSoundSoftLSM(
            NearFieldConfig(
                n_sensors=context,
                grid_size=28,
                domain_half_width=0.8,
                receiver_radius=2.2,
                source_radius=2.4,
                wavenumber=wavenumber,
                boundary_points_per_component=16,
                kernel_gamma=1.0,
                kernel_mix=0.20,
                receiver_noise_correlation=0.18,
            ),
            device,
        )
        for context in contexts
        for wavenumber in (6.0, 8.0, 10.0, 12.0)
    }


def make_model(
    physics: NearFieldSoundSoftLSM,
    method: str,
    width: int,
    *,
    depth: int,
    moment_degree: int,
    sketch_size: int,
    population_factor: bool = True,
) -> PosteriorMomentLSMLoop:
    geometry = physics.acquisition_geometry()
    task_conditioned_factor = method in ("context_pcg", "hybrid_pcg")
    analytic_context_factor = method == "hybrid_pcg"
    solver_method = "pcg" if task_conditioned_factor else method
    return PosteriorMomentLSMLoop(
        geometry["source_kernel"],
        geometry["receiver_feature"],
        physics.n_probes,
        depth=depth,
        moment_degree=moment_degree,
        sketch_size=sketch_size,
        controller_width=width,
        population_width=width,
        use_population_factor=population_factor,
        task_conditioned_factor=task_conditioned_factor,
        analytic_context_factor=analytic_context_factor,
        context_adaptive_sketch=True,
        method=solver_method,
    ).to(physics.device)


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def parameter_norm(model: nn.Module) -> float:
    return math.sqrt(
        sum(
            float(parameter.detach().double().square().sum())
            for parameter in model.parameters()
        )
    )


def parameter_drift(model: nn.Module, initial: dict[str, Tensor]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        total += float(
            (parameter.detach().double() - initial[name].to(parameter.device).double())
            .square()
            .sum()
        )
    return math.sqrt(total)


def network_complexity(model: nn.Module) -> dict[str, float]:
    """Observable norm factors for a Bartlett-style capacity diagnostic."""
    spectral_product = 1.0
    stable_rank_sum = 0.0
    linear_layers = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach().float()
        spectral = float(torch.linalg.matrix_norm(weight, ord=2).clamp_min(1.0e-12))
        frobenius = float(torch.linalg.matrix_norm(weight, ord="fro"))
        spectral_product *= spectral
        stable_rank_sum += (frobenius / spectral) ** (2.0 / 3.0)
        linear_layers += 1
    bartlett_factor = spectral_product * stable_rank_sum**1.5
    return {
        "linear_layers": float(linear_layers),
        "spectral_product": spectral_product,
        "bartlett_factor": bartlett_factor,
    }


def draw_training_batch(
    cache: dict[tuple[int, float], NearFieldSoundSoftLSM],
    train_contexts: tuple[int, ...],
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, float]]:
    context = random.choice(train_contexts)
    wavenumber = random.choice((6.0, 8.0, 10.0))
    physics = cache[(context, wavenumber)]
    count = random.randint(1, 4)
    noise = 0.03 + 0.17 * random.random()
    aperture = 240.0 if random.random() < 0.25 else 360.0
    jitter = 0.15 * random.random()
    rotation = 2.0 * math.pi * random.random()
    near_field, probe, kernel, feature, mask, diagnostics = physics.simulate(
        batch_size,
        count,
        mode="mixed",
        noise_rel=noise,
        aperture_degrees=aperture,
        jitter_fraction=jitter,
        rotation=rotation,
    )
    return (
        near_field,
        probe,
        kernel,
        feature,
        mask,
        {
            "context_size": float(context),
            "wavenumber": wavenumber,
            "obstacle_count": float(count),
            "noise": noise,
            "aperture": aperture,
            "boundary_residual": float(diagnostics["boundary_residual"].mean()),
        },
    )


def training_objective(
    model: PosteriorMomentLSMLoop,
    near_field: Tensor,
    probe: Tensor,
    kernel: Tensor,
    feature: Tensor,
    mask: Tensor,
    depth: int,
    ranking_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    score, info = model(
        near_field,
        probe,
        source_kernel=kernel,
        receiver_feature=feature,
        depth=depth,
        certify=True,
    )
    mean_residual = torch.log1p(info["mean_relative_residual"]).mean()
    covariance_residual = torch.log1p(info["covariance_relative_residual"]).mean()
    localization = ranking_loss(info["plug_in_score"], mask, n_pairs=32)
    endpoint = (
        (
            torch.log(info["predicted_lower"].clamp_min(1.0e-8))
            - torch.log(info["true_lower"].clamp_min(1.0e-8))
        ).square()
        + (
            torch.log(info["predicted_upper"].clamp_min(1.0e-8))
            - torch.log(info["true_upper"].clamp_min(1.0e-8))
        ).square()
    ).mean()
    log_condition = torch.log(
        info["true_upper"] / info["true_lower"].clamp_min(1.0e-8)
    ).mean()
    if model.analytic_context_factor:
        gain_ratio = info["population_gains"] / info[
            "base_population_gains"
        ].clamp_min(1.0e-6)
        gain_regularizer = torch.log(gain_ratio.clamp_min(1.0e-6)).square().mean()
    else:
        gain_regularizer = (
            torch.log(info["population_gains"].clamp_min(1.0e-6)).square().mean()
        )
    identification = torch.zeros((), device=mean_residual.device)
    if model.task_conditioned_factor:
        context_size = info["operator"].shape[-1]
        identity = torch.eye(
            context_size,
            device=info["operator"].device,
            dtype=info["operator"].dtype,
        )
        identification = (
            info["operator"] - identity.unsqueeze(0)
        ).abs().square().sum(dim=(-2, -1)).div(context_size).mean()
    loss = (
        mean_residual
        + 0.50 * covariance_residual
        + float(ranking_weight) * localization
        + 0.01 * endpoint
        + 0.03 * log_condition
        + 0.001 * gain_regularizer
        + 0.10 * identification
    )
    info["training_ranking_loss"] = localization
    info["training_endpoint_loss"] = endpoint
    info["training_log_condition"] = log_condition
    info["training_identification_loss"] = identification
    return loss, info


@torch.no_grad()
def make_evaluation_cache(
    seed: int,
    contexts: tuple[int, ...],
    cache: dict[tuple[int, float], NearFieldSoundSoftLSM],
    tasks: int,
) -> dict[tuple[int, str], EvaluationBatch]:
    output: dict[tuple[int, str], EvaluationBatch] = {}
    for context_index, context in enumerate(contexts):
        for scenario_index, scenario in enumerate(SCENARIOS):
            set_seed(
                seed * 10_000_000
                + context_index * 100_000
                + scenario_index * 1_000
                + 991
            )
            physics = cache[(context, scenario.wavenumber)]
            near_field, probe, kernel, feature, mask, _ = physics.simulate(
                tasks,
                scenario.count,
                mode=scenario.mode,
                noise_rel=scenario.noise,
                aperture_degrees=scenario.aperture,
                jitter_fraction=scenario.jitter,
                rotation=0.19 * scenario_index,
            )
            exact_score, exact_info = exact_near_field_lsm(near_field, probe, kernel)
            output[(context, scenario.name)] = EvaluationBatch(
                context,
                scenario,
                near_field,
                probe,
                kernel,
                feature,
                mask,
                exact_score,
                exact_info,
                physics.grid,
            )
    return output


def error_uncertainty_correlation(
    score: Tensor, score_std: Tensor, exact_score: Tensor
) -> np.ndarray:
    error = (score - exact_score).abs().double()
    uncertainty = score_std.double()
    error = error - error.mean(dim=-1, keepdim=True)
    uncertainty = uncertainty - uncertainty.mean(dim=-1, keepdim=True)
    numerator = (error * uncertainty).sum(dim=-1)
    denominator = torch.sqrt(
        error.square().sum(dim=-1) * uncertainty.square().sum(dim=-1)
    ).clamp_min(1.0e-12)
    return (numerator / denominator).cpu().numpy()


def numerical_metrics(
    score: Tensor,
    info: dict[str, Tensor],
    batch: EvaluationBatch,
) -> list[dict[str, float]]:
    safe_score = torch.nan_to_num(score, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    score_std = torch.nan_to_num(
        info["score_std"], nan=1.0e3, posinf=1.0e3, neginf=1.0e3
    ).clamp_min(1.0e-8)
    localization = task_metrics(safe_score, batch.mask, batch.grid)
    matched_iou = area_matched_iou(safe_score, batch.mask)
    correlation = score_correlation(safe_score, batch.exact_score)
    uq_correlation = error_uncertainty_correlation(
        safe_score, score_std, batch.exact_score
    )
    relative_score_error = torch.linalg.vector_norm(
        safe_score - batch.exact_score, dim=-1
    ) / torch.linalg.vector_norm(batch.exact_score, dim=-1).clamp_min(1.0e-12)
    numerical_coverage = (
        ((safe_score - batch.exact_score).abs() <= 1.96 * score_std)
        .float()
        .mean(dim=-1)
    )
    relative_score_error_values = relative_score_error.cpu().numpy()
    mean_residual_values = (
        torch.nan_to_num(info["mean_relative_residual"], nan=1.0e6, posinf=1.0e6)
        .cpu()
        .numpy()
    )
    covariance_residual_values = (
        torch.nan_to_num(info["covariance_relative_residual"], nan=1.0e6, posinf=1.0e6)
        .cpu()
        .numpy()
    )
    original_mean_residual_values = (
        torch.nan_to_num(
            info.get("original_mean_relative_residual", info["mean_relative_residual"]),
            nan=1.0e6,
            posinf=1.0e6,
        )
        .cpu()
        .numpy()
    )
    original_covariance_residual_values = (
        torch.nan_to_num(
            info.get(
                "original_covariance_relative_residual",
                info["covariance_relative_residual"],
            ),
            nan=1.0e6,
            posinf=1.0e6,
        )
        .cpu()
        .numpy()
    )
    posterior_std_values = score_std.mean(dim=-1).cpu().numpy()
    coverage_values = numerical_coverage.cpu().numpy()
    output: list[dict[str, float]] = []
    for index, metric in enumerate(localization):
        output.append(
            {
                **metric,
                "area_matched_iou": float(matched_iou[index]),
                "exact_score_correlation": float(correlation[index]),
                "relative_score_error": float(relative_score_error_values[index]),
                "mean_relative_residual": float(mean_residual_values[index]),
                "covariance_relative_residual": float(
                    covariance_residual_values[index]
                ),
                "original_mean_relative_residual": float(
                    original_mean_residual_values[index]
                ),
                "original_covariance_relative_residual": float(
                    original_covariance_residual_values[index]
                ),
                "posterior_std_mean": float(posterior_std_values[index]),
                "uq_error_correlation": float(uq_correlation[index]),
                "numerical_coverage_95": float(coverage_values[index]),
            }
        )
    return output


@torch.no_grad()
def evaluate_model(
    model: PosteriorMomentLSMLoop,
    method_label: str,
    evaluation: dict[tuple[int, str], EvaluationBatch],
    *,
    seed: int,
    width: int,
    dataset_size: int,
    parameter_count_value: int,
    training_seconds: float,
    depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model.eval()
    for (context, scenario_name), batch in evaluation.items():
        score, info = model(
            batch.near_field,
            batch.probe,
            source_kernel=batch.kernel,
            receiver_feature=batch.feature,
            depth=depth,
        )
        for task_index, metrics in enumerate(numerical_metrics(score, info, batch)):
            rows.append(
                {
                    "seed": seed,
                    "method": method_label,
                    "network_width": width,
                    "parameter_count": parameter_count_value,
                    "dataset_size": dataset_size,
                    "training_seconds": training_seconds,
                    "context_size": context,
                    "context_measurements": context * context,
                    "scenario": scenario_name,
                    "regime": batch.scenario.regime,
                    "task": task_index,
                    **metrics,
                }
            )
    return rows


@torch.no_grad()
def evaluate_baselines(
    identity_cg: PosteriorMomentLSMLoop,
    evaluation: dict[tuple[int, str], EvaluationBatch],
    *,
    seed: int,
    depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (context, scenario_name), batch in evaluation.items():
        for name in ("identity-CG", "exact"):
            if name == "identity-CG":
                score, info = identity_cg(
                    batch.near_field,
                    batch.probe,
                    source_kernel=batch.kernel,
                    receiver_feature=batch.feature,
                    depth=depth,
                )
            else:
                score, info = batch.exact_score, batch.exact_info
            for task_index, metrics in enumerate(numerical_metrics(score, info, batch)):
                rows.append(
                    {
                        "seed": seed,
                        "method": name,
                        "network_width": 0,
                        "parameter_count": 0,
                        "dataset_size": 0,
                        "training_seconds": 0.0,
                        "context_size": context,
                        "context_measurements": context * context,
                        "scenario": scenario_name,
                        "regime": batch.scenario.regime,
                        "task": task_index,
                        **metrics,
                    }
                )
    return rows


@torch.no_grad()
def benchmark_model(
    model: PosteriorMomentLSMLoop | None,
    method_label: str,
    evaluation: dict[tuple[int, str], EvaluationBatch],
    contexts: tuple[int, ...],
    *,
    seed: int,
    width: int,
    parameter_count_value: int,
    depth: int,
    repeats: int = 20,
) -> list[dict[str, object]]:
    rows = []
    for context in contexts:
        batch = evaluation[(context, "ID four obstacles")]

        def operation() -> None:
            if method_label == "exact":
                exact_near_field_lsm(batch.near_field, batch.probe, batch.kernel)
            else:
                assert model is not None
                model(
                    batch.near_field,
                    batch.probe,
                    source_kernel=batch.kernel,
                    receiver_feature=batch.feature,
                    depth=depth,
                )

        for _ in range(3):
            operation()
        if batch.near_field.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(repeats):
            operation()
        if batch.near_field.device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 2**20
        else:
            peak_memory = float("nan")
        elapsed_ms = 1000.0 * (time.perf_counter() - started) / repeats
        rows.append(
            {
                "seed": seed,
                "method": method_label,
                "network_width": width,
                "parameter_count": parameter_count_value,
                "context_size": context,
                "context_measurements": context * context,
                "depth": depth,
                "batch_size": batch.near_field.shape[0],
                "inference_ms": elapsed_ms,
                "peak_memory_mib": peak_memory,
                "matvec_proxy": depth
                * context
                * context
                * (batch.grid.shape[0] + context),
            }
        )
    return rows


def configuration_order(
    widths: tuple[int, ...], methods: tuple[str, ...]
) -> list[tuple[int, str]]:
    central = min(widths, key=lambda value: abs(value - 128))
    method_priority = {
        "hybrid_pcg": 0,
        "context_pcg": 1,
        "pcg": 2,
        "heavy_ball": 3,
        "chebyshev": 4,
        "richardson": 5,
    }
    return sorted(
        [(width, method) for width in widths for method in methods],
        key=lambda item: (
            0 if item[0] == central else 1,
            method_priority.get(item[1], 99),
            abs(math.log2(item[0] / central)),
            item[0],
        ),
    )


def train_configuration(
    args: argparse.Namespace,
    seed: int,
    width: int,
    method: str,
    dataset_sizes: tuple[int, ...],
    train_contexts: tuple[int, ...],
    cache: dict[tuple[int, float], NearFieldSoundSoftLSM],
    evaluation: dict[tuple[int, str], EvaluationBatch],
    completed_evaluations: set[tuple[str, ...]],
    deadline: float | None,
) -> bool:
    run_name = f"{method}_w{width}_seed{seed}"
    model_seed = seed * 100_000 + width * 10 + LEARNED_METHODS.index(method)
    set_seed(model_seed)
    base_physics = cache[(24 if (24, 8.0) in cache else train_contexts[0], 8.0)]
    model = make_model(
        base_physics,
        method,
        width,
        depth=args.train_depth,
        moment_degree=args.moment_degree,
        sketch_size=args.sketch_size,
    )
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
    )
    checkpoint_path = args.output_dir / "checkpoints" / f"{run_name}.pt"
    completed_examples = 0
    training_seconds = 0.0
    skipped_updates = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=base_physics.device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        completed_examples = int(checkpoint["completed_examples"])
        training_seconds = float(checkpoint["training_seconds"])
        skipped_updates = int(checkpoint.get("skipped_updates", 0))
    parameter_count_value = parameter_count(model)
    training_keys = existing_keys(
        args.output_dir / "training.csv",
        ("seed", "method", "network_width", "dataset_size"),
    )
    window: list[dict[str, float]] = []
    for target_size in dataset_sizes:
        if target_size < completed_examples:
            continue
        if deadline is not None and time.time() >= deadline:
            return False
        if completed_examples < target_size:
            model.train()
            segment_started = time.perf_counter()
            while completed_examples < target_size:
                step = completed_examples // args.batch_size
                set_seed((model_seed * 10_000_000 + step) % 4_294_967_291)
                near_field, probe, kernel, feature, mask, metadata = (
                    draw_training_batch(cache, train_contexts, args.batch_size)
                )
                loss, info = training_objective(
                    model,
                    near_field,
                    probe,
                    kernel,
                    feature,
                    mask,
                    args.train_depth,
                    args.ranking_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 5.0
                    )
                    if torch.isfinite(gradient_norm):
                        optimizer.step()
                    else:
                        skipped_updates += 1
                else:
                    gradient_norm = torch.tensor(float("nan"), device=loss.device)
                    skipped_updates += 1
                completed_examples += args.batch_size
                window.append(
                    {
                        "loss": float(torch.nan_to_num(loss.detach(), nan=1.0e6)),
                        "ranking_loss": float(info["training_ranking_loss"].detach()),
                        "mean_relative_residual": float(
                            info["mean_relative_residual"].mean().detach()
                        ),
                        "covariance_relative_residual": float(
                            info["covariance_relative_residual"].mean().detach()
                        ),
                        "spectral_condition": float(
                            torch.exp(info["training_log_condition"]).detach()
                        ),
                        "gradient_norm": float(
                            torch.nan_to_num(gradient_norm.detach(), nan=1.0e6)
                        ),
                        **metadata,
                    }
                )
            if base_physics.device.type == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.perf_counter() - segment_started

        complexity = network_complexity(model)
        training_row: dict[str, object] = {
            "seed": seed,
            "method": METHOD_LABELS[method],
            "network_width": width,
            "parameter_count": parameter_count_value,
            "dataset_size": target_size,
            "optimizer_updates": target_size // args.batch_size,
            "training_seconds": training_seconds,
            "parameter_l2": parameter_norm(model),
            "parameter_drift_l2": parameter_drift(model, initial),
            "skipped_updates": skipped_updates,
            **complexity,
        }
        if window:
            for key in (
                "loss",
                "ranking_loss",
                "mean_relative_residual",
                "covariance_relative_residual",
                "spectral_condition",
                "gradient_norm",
                "boundary_residual",
            ):
                training_row[f"window_{key}"] = float(
                    np.mean([item[key] for item in window[-min(len(window), 32) :]])
                )
        training_key = (
            str(seed),
            METHOD_LABELS[method],
            str(width),
            str(target_size),
        )
        if training_key not in training_keys:
            append_rows(args.output_dir / "training.csv", [training_row])
            training_keys.add(training_key)

        evaluation_prefix = (
            str(seed),
            METHOD_LABELS[method],
            str(width),
            str(target_size),
        )
        expected = len(evaluation) * args.eval_tasks
        already = sum(key[:4] == evaluation_prefix for key in completed_evaluations)
        if already < expected:
            evaluation_rows = evaluate_model(
                model,
                METHOD_LABELS[method],
                evaluation,
                seed=seed,
                width=width,
                dataset_size=target_size,
                parameter_count_value=parameter_count_value,
                training_seconds=training_seconds,
                depth=args.eval_depth,
            )
            append_rows(args.output_dir / "evaluation.csv", evaluation_rows)
            completed_evaluations.update(
                (
                    str(row["seed"]),
                    str(row["method"]),
                    str(row["network_width"]),
                    str(row["dataset_size"]),
                    str(row["context_size"]),
                    str(row["scenario"]),
                    str(row["task"]),
                )
                for row in evaluation_rows
            )
        payload: dict[str, object] = {
            "model": model.state_dict(),
            "completed_examples": target_size,
            "training_seconds": training_seconds,
            "skipped_updates": skipped_updates,
            "seed": seed,
            "method": method,
            "width": width,
        }
        if target_size != dataset_sizes[-1]:
            payload["optimizer"] = optimizer.state_dict()
        atomic_torch_save(payload, checkpoint_path)
        completed_examples = target_size
        window.clear()
        print(
            f"{run_name:29s} n={target_size:6d} "
            f"train={training_seconds:8.1f}s drift={training_row['parameter_drift_l2']:.3g}",
            flush=True,
        )

    runtime_key = (str(seed), METHOD_LABELS[method], str(width))
    runtime_keys = existing_keys(
        args.output_dir / "runtime.csv", ("seed", "method", "network_width")
    )
    if runtime_key not in runtime_keys:
        append_rows(
            args.output_dir / "runtime.csv",
            benchmark_model(
                model,
                METHOD_LABELS[method],
                evaluation,
                tuple(sorted({key[0] for key in evaluation})),
                seed=seed,
                width=width,
                parameter_count_value=parameter_count_value,
                depth=args.eval_depth,
            ),
        )
    return True


def main() -> None:
    args = parse_args()
    seeds = comma_ints(args.seeds)
    widths = comma_ints(args.widths)
    contexts = comma_ints(args.contexts)
    train_contexts = comma_ints(args.train_contexts)
    dataset_sizes = comma_ints(args.dataset_sizes)
    methods = comma_strings(args.methods)
    if args.quick:
        seeds = seeds[:1]
        widths = (min(widths, key=lambda value: abs(value - 128)),)
        contexts = (
            tuple(value for value in contexts if value in (12, 24)) or contexts[:2]
        )
        train_contexts = contexts
        dataset_sizes = (16, 32)
        methods = tuple(item for item in methods if item in ("pcg", "heavy_ball"))
        args.batch_size = min(args.batch_size, 4)
        args.eval_tasks = min(args.eval_tasks, 2)
        args.train_depth = min(args.train_depth, 4)
        args.eval_depth = min(args.eval_depth, 6)
        args.moment_degree = min(args.moment_degree, 2)
    if not set(methods).issubset(LEARNED_METHODS):
        raise ValueError(f"methods must be drawn from {LEARNED_METHODS}")
    if min(contexts + train_contexts) < args.sketch_size:
        raise ValueError("every context must be at least as long as the sketch")
    if any(size % args.batch_size for size in dataset_sizes):
        raise ValueError("dataset sizes must be divisible by batch size")
    missing_training_contexts = set(train_contexts) - set(contexts)
    all_contexts = tuple(sorted(set(contexts) | set(train_contexts) | {24}))
    if missing_training_contexts:
        print(f"adding training-only contexts {sorted(missing_training_contexts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    previous_protocol_path = args.output_dir / "protocol.json"
    previous_protocol = (
        json.loads(previous_protocol_path.read_text(encoding="utf-8"))
        if args.resume and previous_protocol_path.exists()
        else {}
    )
    device = torch.device(args.device)
    cache = build_physics_cache(all_contexts, device)
    deadline = deadline_timestamp(args.deadline_utc)
    completed_evaluations = existing_keys(
        args.output_dir / "evaluation.csv",
        (
            "seed",
            "method",
            "network_width",
            "dataset_size",
            "context_size",
            "scenario",
            "task",
        ),
    )
    baseline_keys = existing_keys(
        args.output_dir / "baselines.csv",
        ("seed", "method", "context_size", "scenario", "task"),
    )
    total_started = time.perf_counter()
    finished = True
    evaluations: dict[int, dict[tuple[int, str], EvaluationBatch]] = {}
    for seed in seeds:
        evaluation = make_evaluation_cache(seed, contexts, cache, args.eval_tasks)
        evaluations[seed] = evaluation
        set_seed(seed + 88_000_000)
        identity = make_model(
            cache[(24, 8.0)],
            "pcg",
            min(widths),
            depth=args.eval_depth,
            moment_degree=args.moment_degree,
            sketch_size=args.sketch_size,
            population_factor=False,
        )
        expected_baselines = len(evaluation) * args.eval_tasks * 2
        present_baselines = sum(key[0] == str(seed) for key in baseline_keys)
        if present_baselines < expected_baselines:
            rows = evaluate_baselines(
                identity, evaluation, seed=seed, depth=args.eval_depth
            )
            append_rows(args.output_dir / "baselines.csv", rows)
            baseline_keys.update(
                (
                    str(row["seed"]),
                    str(row["method"]),
                    str(row["context_size"]),
                    str(row["scenario"]),
                    str(row["task"]),
                )
                for row in rows
            )
        runtime_keys = existing_keys(
            args.output_dir / "runtime.csv", ("seed", "method", "network_width")
        )
        for baseline_name, baseline_model in (
            ("identity-CG", identity),
            ("exact", None),
        ):
            key = (str(seed), baseline_name, "0")
            if key not in runtime_keys:
                append_rows(
                    args.output_dir / "runtime.csv",
                    benchmark_model(
                        baseline_model,
                        baseline_name,
                        evaluation,
                        contexts,
                        seed=seed,
                        width=0,
                        parameter_count_value=0,
                        depth=args.eval_depth,
                    ),
                )

    for width, method in configuration_order(widths, methods):
        for seed in seeds:
            if deadline is not None and time.time() >= deadline:
                finished = False
                break
            completed = train_configuration(
                args,
                seed,
                width,
                method,
                dataset_sizes,
                train_contexts,
                cache,
                evaluations[seed],
                completed_evaluations,
                deadline,
            )
            if not completed:
                finished = False
                break
        if not finished:
            break
    del evaluations
    if device.type == "cuda":
        torch.cuda.empty_cache()

    previous_tasks = int(
        previous_protocol.get(
            "intermediate_eval_tasks_per_seed_scenario",
            previous_protocol.get("eval_tasks_per_scenario", args.eval_tasks),
        )
    )
    intermediate_tasks = min(previous_tasks, args.eval_tasks)
    final_tasks = max(
        int(
            previous_protocol.get(
                "final_eval_tasks_per_seed_scenario",
                previous_protocol.get("eval_tasks_per_scenario", args.eval_tasks),
            )
        ),
        args.eval_tasks,
    )

    def merged_values(key: str, current: Iterable[object]) -> list[object]:
        return list(dict.fromkeys([*previous_protocol.get(key, []), *current]))

    protocol = {
        "experiment": "original near-field Bayesian LSM multi-axis scaling",
        "physics": asdict(cache[(24, 8.0)].cfg),
        "seeds": merged_values("seeds", seeds),
        "network_widths": merged_values("network_widths", widths),
        "evaluation_context_sizes": merged_values(
            "evaluation_context_sizes", contexts
        ),
        "training_context_sizes": merged_values(
            "training_context_sizes", train_contexts
        ),
        "dataset_sizes": merged_values("dataset_sizes", dataset_sizes),
        "learned_methods": merged_values(
            "learned_methods", [METHOD_LABELS[item] for item in methods]
        ),
        "baselines": merged_values("baselines", ["identity-CG", "exact"]),
        "batch_size": args.batch_size,
        "eval_tasks_per_scenario": final_tasks,
        "requested_eval_tasks_this_invocation": args.eval_tasks,
        "intermediate_eval_tasks_per_seed_scenario": intermediate_tasks,
        "final_eval_tasks_per_seed_scenario": final_tasks,
        "extended_final_evaluation": final_tasks > intermediate_tasks,
        "train_depth": args.train_depth,
        "eval_depth": args.eval_depth,
        "moment_degree": args.moment_degree,
        "sketch_size": args.sketch_size,
        "learning_rate": args.learning_rate,
        "ranking_weight": args.ranking_weight,
        "context_factor_identification_weight": 0.10,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "context_definition": "number of source tokens = number of receiver tokens; measurements are context_size squared",
        "context_pcg_definition": (
            "a permutation-equivariant prompt encoder predicts positive modal "
            "gains from the current Hessian in the fixed receiver angular-kernel "
            "basis; the SPD factor is held fixed throughout standard PCG"
        ),
        "dataset_definition": (
            "number of unique physical inverse-problem examples presented once "
            "in the online stream; examples within an optimizer batch share "
            "the sampled acquisition configuration"
        ),
        "matched_budget": "CG, PCG, and looped methods use eval_depth operator applications for both posterior right-hand-side blocks",
        "deadline_utc": args.deadline_utc or None,
        "finished_requested_grid": finished,
        "elapsed_seconds_this_invocation": time.perf_counter() - total_started,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output_dir}; finished={finished}", flush=True)


if __name__ == "__main__":
    main()
