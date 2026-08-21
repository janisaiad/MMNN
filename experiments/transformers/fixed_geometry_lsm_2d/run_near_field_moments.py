#!/usr/bin/env python3
"""Train and audit posterior-moment loops for original 2D near-field LSM."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from .foundation_uq import (
    balanced_brier,
    balanced_nll,
    occupancy_probability,
)
from .lsm_core import ranking_loss, set_seed
from .near_field_lsm import (
    NearFieldConfig,
    NearFieldSoundSoftLSM,
    PosteriorMomentLSMLoop,
    exact_near_field_lsm,
)
from .run_experiments import task_metrics, write_rows
from .run_foundation_uq import (
    area_matched_iou,
    balanced_ece,
    score_correlation,
    uncertainty_error_auc,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    count: int
    mode: str
    wavenumber: float
    noise: float
    aperture: float = 360.0
    jitter: float = 0.05
    category: str = "ID"


SCENARIOS = (
    Scenario("one obstacle", 1, "mixed", 8.0, 0.10),
    Scenario("two obstacles", 2, "mixed", 8.0, 0.10),
    Scenario("four obstacles", 4, "mixed", 8.0, 0.15),
    Scenario("six obstacles", 6, "mixed", 8.0, 0.15, category="OOD count"),
    Scenario("three stars", 3, "star", 8.0, 0.15, category="OOD shape"),
    Scenario("30 percent noise", 4, "mixed", 8.0, 0.30, category="OOD noise"),
    Scenario(
        "180 degree aperture",
        3,
        "mixed",
        8.0,
        0.15,
        aperture=180.0,
        jitter=0.20,
        category="OOD aperture",
    ),
    Scenario("wavenumber 12", 3, "mixed", 12.0, 0.15, category="OOD frequency"),
)

METHODS = ("richardson", "heavy_ball", "chebyshev")
DISPLAY_METHODS = (
    "learned-Richardson",
    "learned-HB",
    "learned-Chebyshev",
    "global-safe-HB",
    "population-safe-HB",
    "spectrum-HB",
    "identity-CG",
    "population-PCG",
    "exact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-tasks", type=int, default=64)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def physics_cache(device: torch.device) -> dict[float, NearFieldSoundSoftLSM]:
    return {
        wavenumber: NearFieldSoundSoftLSM(
            NearFieldConfig(
                n_sensors=24,
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
        for wavenumber in (6.0, 8.0, 10.0, 12.0)
    }


def make_model(
    physics: NearFieldSoundSoftLSM,
    method: str,
    depth: int,
    *,
    use_population_factor: bool = True,
) -> PosteriorMomentLSMLoop:
    geometry = physics.acquisition_geometry()
    return PosteriorMomentLSMLoop(
        geometry["source_kernel"],
        geometry["receiver_feature"],
        physics.n_probes,
        depth=depth,
        moment_degree=8,
        sketch_size=6,
        controller_width=256,
        use_population_factor=use_population_factor,
        method=method,
    ).to(physics.device)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def draw_training_batch(
    cache: dict[float, NearFieldSoundSoftLSM],
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, object]]:
    wavenumber = random.choice((6.0, 8.0, 10.0))
    physics = cache[wavenumber]
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
    metadata: dict[str, object] = {
        "wavenumber": wavenumber,
        "count": count,
        "noise": noise,
        "aperture": aperture,
        "jitter": jitter,
        "boundary_residual": float(diagnostics["boundary_residual"].mean()),
    }
    return near_field, probe, kernel, feature, mask, metadata


def train_model(
    seed: int,
    method: str,
    cache: dict[float, NearFieldSoundSoftLSM],
    *,
    steps: int,
    depth: int,
    batch_size: int,
    log_every: int,
) -> tuple[PosteriorMomentLSMLoop, list[dict[str, object]]]:
    set_seed(seed + {"richardson": 0, "heavy_ball": 10_000, "chebyshev": 20_000}[method])
    model = make_model(cache[8.0], method, depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-6)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    stable_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    skipped_updates = 0
    for step in range(steps + 1):
        set_seed(seed * 1_000_000 + step + 100_000 * METHODS.index(method))
        near_field, probe, kernel, feature, mask, metadata = draw_training_batch(
            cache, batch_size
        )
        score, info = model(
            near_field,
            probe,
            source_kernel=kernel,
            receiver_feature=feature,
            depth=depth,
            certify=True,
        )
        mean_residual = torch.log1p(info["mean_relative_residual"]).mean()
        covariance_residual = torch.log1p(
            info["covariance_relative_residual"]
        ).mean()
        # Localization supervision is applied to the posterior mean (the
        # plug-in LSM score).  The covariance recurrence is still trained by
        # its own residual, avoiding fragile eigenvalue gradients through a
        # moment-matched UQ score.
        rank = ranking_loss(info["plug_in_score"], mask, n_pairs=32)
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
        spectral_condition = torch.log(
            info["true_upper"] / info["true_lower"].clamp_min(1.0e-8)
        ).mean()
        gain_regularizer = torch.log(
            info["population_gains"].clamp_min(1.0e-6)
        ).square().mean()
        loss = (
            mean_residual
            + 0.50 * covariance_residual
            + 0.08 * rank
            + 0.01 * endpoint
            + 0.02 * spectral_condition
            + 0.001 * gain_regularizer
        )
        if not torch.isfinite(loss):
            model.load_state_dict(stable_state)
            optimizer.state.clear()
            skipped_updates += 1
            print(f"{method:11s} seed={seed} step={step:04d} skipped nonfinite loss")
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(gradient_norm):
            optimizer.step()
        else:
            model.load_state_dict(stable_state)
            optimizer.state.clear()
            skipped_updates += 1
            print(
                f"{method:11s} seed={seed} step={step:04d} "
                "skipped nonfinite gradient"
            )
            continue
        parameters_are_finite = all(
            torch.isfinite(parameter).all() for parameter in model.parameters()
        )
        if not parameters_are_finite:
            model.load_state_dict(stable_state)
            optimizer.state.clear()
            skipped_updates += 1
            print(f"{method:11s} seed={seed} step={step:04d} rolled back parameters")
            continue
        stable_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        if step % log_every == 0 or step == steps:
            row = {
                "seed": seed,
                "method": method,
                "step": step,
                "loss": float(loss.detach()),
                "ranking_loss": float(rank.detach()),
                "mean_relative_residual": float(
                    info["mean_relative_residual"].mean().detach()
                ),
                "covariance_relative_residual": float(
                    info["covariance_relative_residual"].mean().detach()
                ),
                "endpoint_loss": float(endpoint.detach()),
                "spectral_condition": float(torch.exp(spectral_condition).detach()),
                "certificate_rate": float(info["certified"].float().mean().detach()),
                "gradient_norm": float(gradient_norm),
                "skipped_updates": skipped_updates,
                "elapsed_seconds": time.perf_counter() - started,
                **metadata,
            }
            rows.append(row)
            print(
                f"{method:11s} seed={seed} step={step:04d} "
                f"loss={float(loss):.4f} mean-res="
                f"{float(info['mean_relative_residual'].mean()):.4f} cov-res="
                f"{float(info['covariance_relative_residual'].mean()):.4f}"
            )
    model.eval()
    return model, rows


def derived_models(
    trained: dict[str, PosteriorMomentLSMLoop],
    physics: NearFieldSoundSoftLSM,
    depth: int,
) -> dict[str, PosteriorMomentLSMLoop]:
    global_hb = make_model(
        physics, "heavy_ball", depth, use_population_factor=False
    )
    global_hb.load_state_dict(trained["heavy_ball"].state_dict())
    pcg = make_model(physics, "pcg", depth)
    pcg.load_state_dict(trained["heavy_ball"].state_dict())
    identity_cg = make_model(
        physics, "pcg", depth, use_population_factor=False
    )
    identity_cg.load_state_dict(trained["heavy_ball"].state_dict())
    global_hb.eval()
    pcg.eval()
    identity_cg.eval()
    return {"global_hb": global_hb, "identity_cg": identity_cg, "pcg": pcg}


@torch.no_grad()
def solve_variant(
    name: str,
    trained: dict[str, PosteriorMomentLSMLoop],
    derived: dict[str, PosteriorMomentLSMLoop],
    near_field: Tensor,
    probe: Tensor,
    kernel: Tensor,
    feature: Tensor,
    depth: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    common = {
        "source_kernel": kernel,
        "receiver_feature": feature,
        "depth": depth,
        "certify": True,
    }
    if name == "learned-Richardson":
        return trained["richardson"](near_field, probe, **common)
    if name == "learned-HB":
        return trained["heavy_ball"](near_field, probe, **common)
    if name == "learned-Chebyshev":
        return trained["chebyshev"](near_field, probe, **common)
    if name == "global-safe-HB":
        return derived["global_hb"](near_field, probe, force_safe=True, **common)
    if name == "population-safe-HB":
        return trained["heavy_ball"](near_field, probe, force_safe=True, **common)
    if name == "spectrum-HB":
        return trained["heavy_ball"](near_field, probe, force_oracle=True, **common)
    if name == "identity-CG":
        return derived["identity_cg"](near_field, probe, **common)
    if name == "population-PCG":
        return derived["pcg"](near_field, probe, **common)
    if name == "exact":
        return exact_near_field_lsm(near_field, probe, kernel)
    raise ValueError(f"unknown evaluation variant: {name}")


@torch.no_grad()
def calibrate_thresholds(
    seed: int,
    trained: dict[str, PosteriorMomentLSMLoop],
    derived: dict[str, PosteriorMomentLSMLoop],
    cache: dict[float, NearFieldSoundSoftLSM],
    *,
    depth: int,
    tasks: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    set_seed(seed + 2_000_000)
    scores: dict[str, list[Tensor]] = {name: [] for name in DISPLAY_METHODS}
    standard_deviations: dict[str, list[Tensor]] = {
        name: [] for name in DISPLAY_METHODS
    }
    masks: list[Tensor] = []
    completed = 0
    while completed < tasks:
        batch_size = min(8, tasks - completed)
        near_field, probe, kernel, feature, mask, _ = draw_training_batch(
            cache, batch_size
        )
        masks.append(mask)
        for name in DISPLAY_METHODS:
            score, info = solve_variant(
                name,
                trained,
                derived,
                near_field,
                probe,
                kernel,
                feature,
                depth,
            )
            scores[name].append(score)
            standard_deviations[name].append(info["score_std"])
        completed += batch_size
    target = torch.cat(masks)
    thresholds: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for name in DISPLAY_METHODS:
        mean = torch.nan_to_num(
            torch.cat(scores[name]), nan=0.0, posinf=1.0e3, neginf=-1.0e3
        )
        std = torch.nan_to_num(
            torch.cat(standard_deviations[name]),
            nan=1.0e3,
            posinf=1.0e3,
            neginf=1.0e3,
        ).clamp_min(1.0e-6)
        candidates = torch.linspace(
            torch.quantile(mean, 0.02),
            torch.quantile(mean, 0.98),
            61,
            device=mean.device,
        )
        risks = torch.stack(
            [
                balanced_brier(
                    occupancy_probability(mean, std, float(candidate)), target
                ).mean()
                for candidate in candidates
            ]
        )
        best = int(risks.argmin())
        threshold = float(candidates[best])
        thresholds[name] = threshold
        probability = occupancy_probability(mean, std, threshold)
        rows.append(
            {
                "seed": seed,
                "method": name,
                "threshold": threshold,
                "calibration_brier": float(balanced_brier(probability, target).mean()),
                "calibration_nll": float(balanced_nll(probability, target).mean()),
                "calibration_ece": float(balanced_ece(probability, target).mean()),
            }
        )
    return thresholds, rows


@torch.no_grad()
def evaluate_seed(
    seed: int,
    trained: dict[str, PosteriorMomentLSMLoop],
    derived: dict[str, PosteriorMomentLSMLoop],
    cache: dict[float, NearFieldSoundSoftLSM],
    thresholds: dict[str, float],
    *,
    depth: int,
    eval_tasks: int,
) -> tuple[list[dict[str, object]], dict[str, Tensor]]:
    rows: list[dict[str, object]] = []
    showcase: dict[str, Tensor] = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        set_seed(seed + 3_000_000 + 10_000 * scenario_index)
        physics = cache[scenario.wavenumber]
        completed = 0
        while completed < eval_tasks:
            batch_size = min(8, eval_tasks - completed)
            rotation = 0.37 * scenario_index
            near_field, probe, kernel, feature, mask, diagnostics = physics.simulate(
                batch_size,
                scenario.count,
                mode=scenario.mode,
                noise_rel=scenario.noise,
                aperture_degrees=scenario.aperture,
                jitter_fraction=scenario.jitter,
                rotation=rotation,
            )
            exact_score, _ = exact_near_field_lsm(near_field, probe, kernel)
            for name in DISPLAY_METHODS:
                score, info = solve_variant(
                    name,
                    trained,
                    derived,
                    near_field,
                    probe,
                    kernel,
                    feature,
                    depth,
                )
                numerical_failure = ~(
                    torch.isfinite(score).all(dim=-1)
                    & torch.isfinite(info["score_std"]).all(dim=-1)
                    & torch.isfinite(info["mean_relative_residual"])
                    & torch.isfinite(info["covariance_relative_residual"])
                )
                safe_score = torch.nan_to_num(
                    score, nan=0.0, posinf=1.0e3, neginf=-1.0e3
                )
                safe_score = torch.where(
                    numerical_failure[:, None],
                    torch.zeros_like(safe_score),
                    safe_score,
                )
                safe_std = torch.nan_to_num(
                    info["score_std"],
                    nan=1.0e3,
                    posinf=1.0e3,
                    neginf=1.0e3,
                ).clamp_min(1.0e-6)
                localization = task_metrics(safe_score, mask, physics.grid)
                matched_iou = area_matched_iou(safe_score, mask)
                correlation = score_correlation(safe_score, exact_score)
                probability = occupancy_probability(
                    safe_score, safe_std, thresholds[name]
                )
                probability = torch.where(
                    numerical_failure[:, None],
                    torch.full_like(probability, 0.5),
                    probability,
                )
                brier = balanced_brier(probability, mask).cpu().numpy()
                nll = balanced_nll(probability, mask).cpu().numpy()
                ece = balanced_ece(probability, mask)
                error_auc = uncertainty_error_auc(probability, mask)
                for task_index, metric in enumerate(localization):
                    true_lower = info.get("true_lower")
                    true_upper = info.get("true_upper")
                    predicted_lower = info.get("predicted_lower")
                    predicted_upper = info.get("predicted_upper")
                    certified = info.get("certified")
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario.name,
                            "category": scenario.category,
                            "method": name,
                            "task": completed + task_index,
                            **metric,
                            "area_matched_iou": float(matched_iou[task_index]),
                            "score_correlation": float(correlation[task_index]),
                            "mean_relative_residual": float(
                                torch.nan_to_num(
                                    info["mean_relative_residual"][task_index],
                                    nan=1.0e6,
                                    posinf=1.0e6,
                                    neginf=1.0e6,
                                )
                            ),
                            "covariance_relative_residual": float(
                                torch.nan_to_num(
                                    info["covariance_relative_residual"][task_index],
                                    nan=1.0e6,
                                    posinf=1.0e6,
                                    neginf=1.0e6,
                                )
                            ),
                            "balanced_brier": float(brier[task_index]),
                            "balanced_nll": float(nll[task_index]),
                            "balanced_ece": float(ece[task_index]),
                            "uncertainty_error_auc": float(error_auc[task_index]),
                            "numerical_failure": float(numerical_failure[task_index]),
                            "certificate": (
                                float(certified[task_index])
                                if certified is not None
                                else float("nan")
                            ),
                            "predicted_lower": (
                                float(predicted_lower[task_index])
                                if predicted_lower is not None
                                else float("nan")
                            ),
                            "true_lower": (
                                float(true_lower[task_index])
                                if true_lower is not None
                                else float("nan")
                            ),
                            "predicted_upper": (
                                float(predicted_upper[task_index])
                                if predicted_upper is not None
                                else float("nan")
                            ),
                            "true_upper": (
                                float(true_upper[task_index])
                                if true_upper is not None
                                else float("nan")
                            ),
                            "boundary_residual": float(
                                diagnostics["boundary_residual"][task_index]
                            ),
                        }
                    )
                if (
                    scenario.name == "four obstacles"
                    and name in ("learned-HB", "population-PCG", "exact")
                    and name not in showcase
                ):
                    if "mask" not in showcase:
                        showcase["mask"] = mask[0].detach().cpu()
                    showcase[name] = safe_score[0].detach().cpu()
                    showcase[f"{name}-std"] = safe_std[0].detach().cpu()
            completed += batch_size
    return rows, showcase


@torch.no_grad()
def depth_audit(
    seed: int,
    trained: dict[str, PosteriorMomentLSMLoop],
    derived: dict[str, PosteriorMomentLSMLoop],
    physics: NearFieldSoundSoftLSM,
) -> list[dict[str, object]]:
    set_seed(seed + 4_000_000)
    near_field, probe, kernel, feature, _, _ = physics.simulate(
        16, 4, noise_rel=0.15, jitter_fraction=0.05
    )
    rows = []
    methods = (
        "learned-Richardson",
        "learned-HB",
        "learned-Chebyshev",
        "global-safe-HB",
        "population-safe-HB",
        "spectrum-HB",
        "identity-CG",
        "population-PCG",
    )
    for depth in (4, 8, 16, 24, 32, 48, 64):
        for name in methods:
            _, info = solve_variant(
                name,
                trained,
                derived,
                near_field,
                probe,
                kernel,
                feature,
                depth,
            )
            rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "depth": depth,
                    "mean_relative_residual": float(
                        torch.nan_to_num(
                            info["mean_relative_residual"],
                            nan=1.0e6,
                            posinf=1.0e6,
                            neginf=1.0e6,
                        ).mean()
                    ),
                    "covariance_relative_residual": float(
                        torch.nan_to_num(
                            info["covariance_relative_residual"],
                            nan=1.0e6,
                            posinf=1.0e6,
                            neginf=1.0e6,
                        ).mean()
                    ),
                }
            )
    return rows


@torch.no_grad()
def benchmark(
    trained: dict[str, PosteriorMomentLSMLoop],
    derived: dict[str, PosteriorMomentLSMLoop],
    physics: NearFieldSoundSoftLSM,
    depth: int,
) -> dict[str, float]:
    near_field, probe, kernel, feature, _, _ = physics.simulate(
        8, 4, noise_rel=0.15
    )
    result = {}
    for name in DISPLAY_METHODS:
        def operation() -> tuple[Tensor, dict[str, Tensor]]:
            return solve_variant(
                name,
                trained,
                derived,
                near_field,
                probe,
                kernel,
                feature,
                depth,
            )

        for _ in range(5):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            operation()
        if physics.device.type == "cuda":
            torch.cuda.synchronize()
        result[name] = 1000.0 * (time.perf_counter() - started) / 20.0
    return result


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("method", "scenario", "category")
    columns = (
        "average_precision",
        "auc",
        "area_matched_iou",
        "score_correlation",
        "mean_relative_residual",
        "covariance_relative_residual",
        "balanced_brier",
        "balanced_nll",
        "balanced_ece",
        "uncertainty_error_auc",
        "numerical_failure",
        "certificate",
        "boundary_residual",
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {name: value for name, value in zip(keys, key, strict=True)}
        summary["n_tasks"] = len(group)
        for column in columns:
            values = np.asarray([float(row[column]) for row in group])
            finite = values[np.isfinite(values)]
            summary[f"{column}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
            summary[f"{column}_ci95"] = (
                float(1.96 * finite.std(ddof=1) / math.sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
        output.append(summary)
    return output


def save_showcase(path: Path, showcase: dict[str, Tensor], grid_size: int) -> None:
    if not showcase:
        return
    columns = ("mask", "learned-HB", "population-PCG", "exact")
    figure, axes = plt.subplots(2, 4, figsize=(11.0, 5.3), constrained_layout=True)
    for column, name in enumerate(columns):
        image = showcase[name].reshape(grid_size, grid_size).numpy()
        axes[0, column].imshow(image, origin="lower", cmap="magma")
        axes[0, column].set_title(name)
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])
        if name == "mask":
            uncertainty = np.zeros_like(image)
        else:
            uncertainty = showcase[f"{name}-std"].reshape(grid_size, grid_size).numpy()
        axes[1, column].imshow(uncertainty, origin="lower", cmap="viridis")
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])
    axes[0, 0].set_ylabel("score / truth")
    axes[1, 0].set_ylabel("posterior std")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.steps = min(args.steps, 30)
        args.eval_tasks = min(args.eval_tasks, 4)
        args.seeds = args.seeds.split(",")[0]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    cache = physics_cache(device)
    all_training: list[dict[str, object]] = []
    all_tasks: list[dict[str, object]] = []
    all_calibration: list[dict[str, object]] = []
    all_depth: list[dict[str, object]] = []
    runtimes: dict[str, dict[str, float]] = {}
    parameter_counts: dict[str, int] = {}
    showcase: dict[str, Tensor] = {}
    total_started = time.perf_counter()
    for seed in seeds:
        trained: dict[str, PosteriorMomentLSMLoop] = {}
        for method in METHODS:
            checkpoint_path = output_dir / f"near_field_{method}_seed_{seed}.pt"
            if args.resume and checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model = make_model(cache[8.0], method, args.depth)
                model.load_state_dict(checkpoint["model"])
                model.eval()
                training_rows = [
                    {
                        "seed": seed,
                        "method": method,
                        "step": args.steps,
                        "resumed_checkpoint": 1,
                    }
                ]
            else:
                model, training_rows = train_model(
                    seed,
                    method,
                    cache,
                    steps=args.steps,
                    depth=args.depth,
                    batch_size=8 if not args.quick else 3,
                    log_every=max(args.steps // 10, 1),
                )
            trained[method] = model
            all_training.extend(training_rows)
            parameter_counts[method] = parameter_count(model)
            torch.save(
                {
                    "model": model.state_dict(),
                    "depth": args.depth,
                    "seed": seed,
                    "method": method,
                },
                checkpoint_path,
            )
        derived = derived_models(trained, cache[8.0], args.depth)
        thresholds, calibration_rows = calibrate_thresholds(
            seed,
            trained,
            derived,
            cache,
            depth=args.depth,
            tasks=12 if args.quick else 48,
        )
        all_calibration.extend(calibration_rows)
        task_rows, seed_showcase = evaluate_seed(
            seed,
            trained,
            derived,
            cache,
            thresholds,
            depth=args.depth,
            eval_tasks=args.eval_tasks,
        )
        all_tasks.extend(task_rows)
        if not showcase:
            showcase = seed_showcase
        all_depth.extend(depth_audit(seed, trained, derived, cache[8.0]))
        runtimes[str(seed)] = benchmark(trained, derived, cache[8.0], args.depth)

    summary = aggregate(all_tasks)
    write_rows(output_dir / "training.csv", all_training)
    write_rows(output_dir / "calibration.csv", all_calibration)
    write_rows(output_dir / "tasks.csv", all_tasks)
    write_rows(output_dir / "summary.csv", summary)
    write_rows(output_dir / "depth.csv", all_depth)
    save_showcase(output_dir / "near_field_reconstructions.png", showcase, 28)
    protocol = {
        "architecture": (
            "parallel posterior-mean and posterior-covariance recurrences with "
            "shared fixed softmax GP kernel, population factor, and spectral controller"
        ),
        "physics": asdict(cache[8.0].cfg),
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "seeds": seeds,
        "steps": args.steps,
        "depth": args.depth,
        "eval_tasks_per_seed_scenario": args.eval_tasks,
        "parameter_counts": parameter_counts,
        "runtime_ms_batch8": runtimes,
        "elapsed_seconds": time.perf_counter() - total_started,
        "resumed_checkpoints": bool(args.resume),
        "optimization_safeguard": (
            "rollback to the last finite state and clear optimizer state after "
            "any nonfinite loss, gradient, or parameter update"
        ),
        "paper_alignment": {
            "mean_rhs": "Phi",
            "covariance_rhs": "N K",
            "shared_system": "N K N* + I",
            "posterior_covariance": "K - K N* (N K N* + I)^-1 N K",
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
