"""Full-training Fourier hierarchy experiments for MLPs, MMNNs, and Muon^p.

This companion campaign answers questions that the right-factor-only study
cannot answer: whether the hierarchy survives when every layer is trained,
how it changes with affine depth, and whether an exact spectral-power update
can move high-frequency sector clocks.  It writes raw paired trajectories,
run summaries, calibration records, and vector figures.

The optimizer comparison is deliberately full batch.  ``mup_gd`` applies the
explicit layer metric of the model.  ``muon_p*`` applies a direct compact SVD
map, with a declared numerical-rank floor, to every matrix block and the same
maximal-update metric to vectors and scalars.  The spectral maps are
memoryless and use no weight decay, which isolates singular-value reshaping
from momentum and regularization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mmnn.full_training_frequency import (
    FullyTrainedPeriodicMLP,
    FullyTrainedPeriodicMMNN,
)
from mmnn.spectral_power import spectral_power_direction


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "full_training_results"
FIGURES = ROOT / "figures"

HIERARCHY_FREQUENCIES = (1, 4, 8, 16)
HIERARCHY_AMPLITUDES = (1.0, 0.65, 0.45, 0.30)
OPTIMIZER_POWERS: dict[str, float | None] = {
    "mup_gd": None,
    "muon_p0": 0.0,
    "muon_p1_3": 1.0 / 3.0,
    "muon_p2_3": 2.0 / 3.0,
}
SPECTRAL_RELATIVE_FLOOR = 1.0e-7
SPECTRAL_BACKEND = "direct_torch_svd"
SPECTRAL_CUDA_DRIVER = "gesvd"


@dataclass(frozen=True)
class FullTrainingConfig:
    architecture: str = "fc"
    affine_depth: int = 5
    width: int = 128
    rank_ratio: float = 0.25
    grid_size: int = 128
    target_kind: str = "hierarchy"
    powerlaw_alpha: float = 1.25
    powerlaw_max_frequency: int = 24
    optimizer: str = "mup_gd"
    learning_rate: float = 0.1
    steps: int = 2_500
    record_every: int = 25
    seed: int = 0
    bias_scale: float = 0.1


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def periodic_grid(size: int, device: torch.device) -> torch.Tensor:
    return 2.0 * math.pi * torch.arange(size, device=device) / size


def target_components(
    config: FullTrainingConfig,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if config.target_kind == "hierarchy":
        return HIERARCHY_FREQUENCIES, HIERARCHY_AMPLITUDES
    if config.target_kind == "powerlaw":
        frequencies = tuple(range(1, config.powerlaw_max_frequency + 1))
        amplitudes = tuple(
            float(frequency ** (-config.powerlaw_alpha)) for frequency in frequencies
        )
        return frequencies, amplitudes
    raise ValueError(f"unknown target kind: {config.target_kind}")


def cosine_target(
    x: torch.Tensor,
    frequencies: tuple[int, ...],
    amplitudes: tuple[float, ...],
) -> torch.Tensor:
    result = torch.zeros_like(x)
    for frequency, amplitude in zip(frequencies, amplitudes, strict=True):
        result.add_(torch.cos(frequency * x), alpha=amplitude)
    return result


def cosine_coefficient(values: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    return 2.0 * torch.mean(values * torch.cos(k * x))


def make_model(
    config: FullTrainingConfig, x: torch.Tensor
) -> FullyTrainedPeriodicMLP | FullyTrainedPeriodicMMNN:
    if config.architecture == "fc":
        return FullyTrainedPeriodicMLP(
            x,
            width=config.width,
            affine_depth=config.affine_depth,
            seed=config.seed,
            bias_scale=config.bias_scale,
        )
    if config.architecture == "mmnn":
        rank = max(1, int(round(config.rank_ratio * config.width)))
        return FullyTrainedPeriodicMMNN(
            x,
            width=config.width,
            affine_depth=config.affine_depth,
            rank=rank,
            seed=config.seed,
            bias_scale=config.bias_scale,
        )
    raise ValueError(f"unknown architecture: {config.architecture}")


def optimizer_directions(
    model: FullyTrainedPeriodicMLP | FullyTrainedPeriodicMMNN,
    names: tuple[str, ...],
    parameters: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
    optimizer: str,
) -> tuple[torch.Tensor, ...]:
    power = OPTIMIZER_POWERS[optimizer]
    directions: list[torch.Tensor] = []
    for name, parameter, gradient in zip(names, parameters, gradients, strict=True):
        if power is not None and parameter.ndim == 2:
            directions.append(
                spectral_power_direction(
                    gradient,
                    power,
                    relative_floor=SPECTRAL_RELATIVE_FLOOR,
                )
            )
        else:
            directions.append(gradient * model.metric_scale(name))
    return tuple(directions)


def tangent_curvature_and_velocity(
    prediction: torch.Tensor,
    x: torch.Tensor,
    frequency: int,
    model: FullyTrainedPeriodicMLP | FullyTrainedPeriodicMMNN,
    names: tuple[str, ...],
    parameters: tuple[torch.Tensor, ...],
    directions: tuple[torch.Tensor, ...],
    *,
    retain_graph: bool,
) -> tuple[float, float]:
    # Divide by sqrt(2) to use the normalized basis e_q=sqrt(2)cos(qx).
    normalized_coefficient = cosine_coefficient(prediction, x, frequency) / math.sqrt(
        2.0
    )
    coefficient_gradients = torch.autograd.grad(
        normalized_coefficient,
        parameters,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    curvature = sum(
        model.metric_scale(name) * torch.sum(gradient.square())
        for name, gradient in zip(names, coefficient_gradients, strict=True)
    )
    # The parameter update is -eta * direction, so this is d<f,e_q>/d eta.
    velocity = -sum(
        torch.sum(gradient * direction)
        for gradient, direction in zip(coefficient_gradients, directions, strict=True)
    )
    return float(curvature.detach()), float(velocity.detach())


def first_half_error_time(trace: list[dict[str, Any]], frequency: int) -> float | None:
    key = f"relative_error_{frequency}"
    for row in trace:
        if float(row[key]) <= 0.5:
            return float(row["step"])
    return None


def _parameter_displacement(
    parameters: tuple[torch.Tensor, ...],
    initial_parameters: tuple[torch.Tensor, ...],
) -> float:
    numerator = torch.sqrt(
        sum(
            torch.sum((parameter.detach() - initial).square())
            for parameter, initial in zip(parameters, initial_parameters, strict=True)
        )
    )
    denominator = torch.sqrt(
        sum(torch.sum(initial.square()) for initial in initial_parameters)
    )
    return float(numerator / denominator.clamp_min(1.0e-12))


def train_case(
    config: FullTrainingConfig,
    *,
    tag: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config.optimizer not in OPTIMIZER_POWERS:
        raise ValueError(f"unknown optimizer: {config.optimizer}")
    x = periodic_grid(config.grid_size, device)
    frequencies, amplitudes = target_components(config)
    target = cosine_target(x, frequencies, amplitudes)
    model = make_model(config, x)
    named_parameters = tuple(model.named_parameters())
    names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    initial_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    checkpoints = set(range(0, config.steps + 1, config.record_every))
    checkpoints.add(config.steps)
    diagnostic_checkpoints = {
        0,
        config.steps // 4,
        config.steps // 2,
        3 * config.steps // 4,
        config.steps,
    }
    diagnostic_frequencies = (
        HIERARCHY_FREQUENCIES
        if config.target_kind == "hierarchy"
        else (1, 4, 8, 16, config.powerlaw_max_frequency)
    )
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    stable = True

    for step in range(config.steps + 1):
        prediction = model()
        residual = prediction - target
        loss = 0.5 * torch.mean(residual.square())
        diagnostic = step in diagnostic_checkpoints
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=diagnostic,
            allow_unused=False,
        )
        raw_finite = bool(torch.isfinite(loss)) and all(
            bool(torch.all(torch.isfinite(gradient))) for gradient in gradients
        )
        if not raw_finite or float(loss.detach()) > 1.0e6:
            stable = False
            break
        directions = optimizer_directions(
            model, names, parameters, gradients, config.optimizer
        )

        if step in checkpoints:
            with torch.no_grad():
                feature_displacements = model.relative_feature_displacements()
                row: dict[str, Any] = {
                    "tag": tag,
                    "architecture": config.architecture,
                    "affine_depth": config.affine_depth,
                    "width": config.width,
                    "rank_ratio": config.rank_ratio,
                    "rank": (getattr(model, "rank", config.width)),
                    "target_kind": config.target_kind,
                    "powerlaw_alpha": config.powerlaw_alpha,
                    "optimizer": config.optimizer,
                    "spectral_backend": (
                        ""
                        if OPTIMIZER_POWERS[config.optimizer] is None
                        else SPECTRAL_BACKEND
                    ),
                    "spectral_cuda_driver": (
                        ""
                        if OPTIMIZER_POWERS[config.optimizer] is None
                        else SPECTRAL_CUDA_DRIVER
                    ),
                    "spectral_relative_floor": (
                        ""
                        if OPTIMIZER_POWERS[config.optimizer] is None
                        else SPECTRAL_RELATIVE_FLOOR
                    ),
                    "spectral_power": (
                        ""
                        if OPTIMIZER_POWERS[config.optimizer] is None
                        else OPTIMIZER_POWERS[config.optimizer]
                    ),
                    "learning_rate": config.learning_rate,
                    "seed": config.seed,
                    "step": step,
                    "loss": float(loss.detach()),
                    "relative_parameter_displacement": _parameter_displacement(
                        parameters, initial_parameters
                    ),
                    "max_relative_feature_displacement": max(
                        float(value) for value in feature_displacements
                    ),
                }
                for index, value in enumerate(feature_displacements):
                    row[f"feature_displacement_{index}"] = float(value)
                for frequency, amplitude in zip(frequencies, amplitudes, strict=True):
                    coefficient = float(
                        cosine_coefficient(prediction, x, frequency).detach()
                    )
                    row[f"coefficient_{frequency}"] = coefficient
                    row[f"relative_error_{frequency}"] = abs(
                        coefficient - amplitude
                    ) / abs(amplitude)

            if diagnostic:
                for index, frequency in enumerate(diagnostic_frequencies):
                    curvature, velocity = tangent_curvature_and_velocity(
                        prediction,
                        x,
                        frequency,
                        model,
                        names,
                        parameters,
                        directions,
                        retain_graph=index + 1 < len(diagnostic_frequencies),
                    )
                    row[f"lambda_{frequency}"] = curvature
                    row[f"velocity_per_lr_{frequency}"] = velocity
            trace.append(row)

        finite = all(
            bool(torch.all(torch.isfinite(direction))) for direction in directions
        )
        if not finite:
            stable = False
            break
        if step == config.steps:
            break
        with torch.no_grad():
            for parameter, direction in zip(parameters, directions, strict=True):
                parameter.add_(direction, alpha=-config.learning_rate)

    elapsed = time.perf_counter() - started
    final_row = trace[-1]
    summary: dict[str, Any] = {
        **asdict(config),
        "tag": tag,
        "rank": getattr(model, "rank", config.width),
        "spectral_backend": (
            "" if OPTIMIZER_POWERS[config.optimizer] is None else SPECTRAL_BACKEND
        ),
        "spectral_cuda_driver": (
            "" if OPTIMIZER_POWERS[config.optimizer] is None else SPECTRAL_CUDA_DRIVER
        ),
        "spectral_relative_floor": (
            ""
            if OPTIMIZER_POWERS[config.optimizer] is None
            else SPECTRAL_RELATIVE_FLOOR
        ),
        "stable": stable and int(final_row["step"]) == config.steps,
        "elapsed_seconds": elapsed,
        "final_loss": float(final_row["loss"]),
        "final_parameter_displacement": float(
            final_row["relative_parameter_displacement"]
        ),
        "final_feature_displacement": float(
            final_row["max_relative_feature_displacement"]
        ),
    }
    for frequency in frequencies:
        summary[f"t50_{frequency}"] = first_half_error_time(trace, frequency)
        summary[f"final_relative_error_{frequency}"] = float(
            final_row[f"relative_error_{frequency}"]
        )
    return trace, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def calibration_score(summary: dict[str, Any]) -> float:
    if not bool(summary["stable"]):
        return float("inf")
    frequencies, _ = target_components(
        FullTrainingConfig(
            **{
                key: summary[key]
                for key in FullTrainingConfig.__dataclass_fields__
                if key in summary
            }
        )
    )
    errors = np.array(
        [
            float(summary[f"final_relative_error_{frequency}"])
            for frequency in frequencies
        ]
    )
    # All target sectors contribute on a logarithmic scale; endpoint risk is
    # a light tie breaker rather than the sole selection criterion.
    return float(
        np.mean(np.log10(np.maximum(errors, 1.0e-5)))
        + 0.15 * np.log10(max(float(summary["final_loss"]), 1.0e-12))
    )


def calibration_grid(optimizer: str) -> tuple[float, ...]:
    if optimizer == "mup_gd":
        return (0.1, 0.3, 1.0)
    return (3.0e-3, 1.0e-2, 3.0e-2)


def run_calibration(device: torch.device, quick: bool) -> dict[str, float]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    keys: list[tuple[str, int, str]] = []
    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            keys.append((architecture, depth, "mup_gd"))
        for optimizer in ("muon_p0", "muon_p1_3", "muon_p2_3"):
            keys.append((architecture, 5, optimizer))

    repeats = (101,) if quick else (101, 102)
    steps = 200 if quick else 600
    for architecture, depth, optimizer in keys:
        candidates: dict[float, list[float]] = {}
        for learning_rate in calibration_grid(optimizer):
            for seed in repeats:
                config = FullTrainingConfig(
                    architecture=architecture,
                    affine_depth=depth,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    steps=steps,
                    record_every=max(10, steps // 20),
                    seed=seed,
                    width=64 if quick else 128,
                )
                tag = f"cal_{architecture}_d{depth}_{optimizer}_lr{learning_rate:g}_s{seed}"
                _, summary = train_case(config, tag=tag, device=device)
                score = calibration_score(summary)
                row = {**summary, "calibration_score": score}
                rows.append(row)
                candidates.setdefault(learning_rate, []).append(score)
                print(
                    f"[calibration] {tag}: loss={summary['final_loss']:.3e}, score={score:.3f}",
                    flush=True,
                )
        finite_candidates = {
            learning_rate: float(np.median(scores))
            for learning_rate, scores in candidates.items()
            if np.all(np.isfinite(scores))
        }
        if not finite_candidates:
            raise RuntimeError(
                f"no stable calibration candidate for {(architecture, depth, optimizer)}"
            )
        best = min(finite_candidates, key=finite_candidates.get)
        selected[f"{architecture}|{depth}|{optimizer}"] = best

    write_csv(RESULTS / "calibration_runs.csv", rows)
    (RESULTS / "selected_learning_rates.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True)
    )
    return selected


def load_learning_rates() -> dict[str, float]:
    path = RESULTS / "selected_learning_rates.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run with --mode calibrate first"
        )
    return {key: float(value) for key, value in json.loads(path.read_text()).items()}


def campaign_configs(
    learning_rates: dict[str, float], quick: bool
) -> list[tuple[str, FullTrainingConfig]]:
    configs: dict[str, FullTrainingConfig] = {}
    confirm_seeds = range(2) if quick else range(8)
    optimizer_seeds = range(2) if quick else range(8)
    steps_hierarchy = 300 if quick else 2_500
    steps_powerlaw = 350 if quick else 1_500
    width = 64 if quick else 128

    # Full-training depth confirmation under maximal-update gradient descent.
    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            optimizer = "mup_gd"
            learning_rate = learning_rates[f"{architecture}|{depth}|{optimizer}"]
            for seed in confirm_seeds:
                tag = f"hierarchy_{architecture}_d{depth}_{optimizer}_s{seed}"
                configs[tag] = FullTrainingConfig(
                    architecture=architecture,
                    affine_depth=depth,
                    width=width,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    steps=steps_hierarchy,
                    record_every=max(10, steps_hierarchy // 100),
                    seed=seed,
                )

    # Paired Muon^p comparison on the representative five-affine-map model.
    for architecture in ("fc", "mmnn"):
        depth = 5
        for optimizer in OPTIMIZER_POWERS:
            learning_rate = learning_rates[f"{architecture}|{depth}|{optimizer}"]
            for seed in optimizer_seeds:
                tag = f"hierarchy_{architecture}_d{depth}_{optimizer}_s{seed}"
                configs[tag] = FullTrainingConfig(
                    architecture=architecture,
                    affine_depth=depth,
                    width=width,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    steps=steps_hierarchy,
                    record_every=max(10, steps_hierarchy // 100),
                    seed=seed,
                )
                power_tag = f"powerlaw_{architecture}_d{depth}_{optimizer}_s{seed}"
                configs[power_tag] = FullTrainingConfig(
                    architecture=architecture,
                    affine_depth=depth,
                    width=width,
                    target_kind="powerlaw",
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    steps=steps_powerlaw,
                    record_every=max(10, steps_powerlaw // 120),
                    seed=seed,
                )

    # Width transfer is the empirical check that the selected rates have the
    # intended maximal-update interpretation.  No rates are retuned by width.
    width_values = (64, 128) if quick else (64, 128, 256)
    width_seeds = range(1) if quick else range(20, 22)
    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            learning_rate = learning_rates[f"{architecture}|5|{optimizer}"]
            for current_width in width_values:
                for seed in width_seeds:
                    tag = f"width_{architecture}_{optimizer}_m{current_width}_s{seed}"
                    configs[tag] = FullTrainingConfig(
                        architecture=architecture,
                        affine_depth=5,
                        width=current_width,
                        target_kind="powerlaw",
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        steps=steps_hierarchy,
                        record_every=max(10, steps_hierarchy // 100),
                        seed=seed,
                    )
    return sorted(configs.items())


def run_campaign(device: torch.device, quick: bool) -> None:
    learning_rates = load_learning_rates()
    configs = campaign_configs(learning_rates, quick)
    all_trace: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    started = time.time()
    for index, (tag, config) in enumerate(configs, start=1):
        trace, summary = train_case(config, tag=tag, device=device)
        all_trace.extend(trace)
        all_summary.append(summary)
        print(
            f"[{index:03d}/{len(configs):03d}] {tag}: "
            f"loss={summary['final_loss']:.3e}, stable={summary['stable']}",
            flush=True,
        )
    write_csv(RESULTS / "full_training_traces.csv", all_trace)
    write_csv(RESULTS / "full_training_summaries.csv", all_summary)
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "quick": quick,
        "number_of_runs": len(configs),
        "elapsed_seconds": time.time() - started,
        "optimizer_note": (
            "Direct torch.linalg.svd memoryless compact spectral-power maps "
            f"on matrices with relative numerical-rank floor {SPECTRAL_RELATIVE_FLOOR:g}; "
            "maximal-update gradient directions on vectors; no weight decay."
        ),
        "spectral_backend": SPECTRAL_BACKEND,
        "spectral_cuda_driver": SPECTRAL_CUDA_DRIVER,
        "spectral_relative_floor": SPECTRAL_RELATIVE_FLOOR,
        "learning_rates": learning_rates,
    }
    (RESULTS / "full_training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def run_discretization(device: torch.device, quick: bool) -> None:
    """Halve the learning rate and double the horizon at fixed optimizer time."""
    learning_rates = load_learning_rates()
    traces: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seeds = range(1) if quick else range(30, 33)
    base_steps = 250 if quick else 1_000
    configs: list[tuple[str, FullTrainingConfig]] = []
    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            base_rate = learning_rates[f"{architecture}|5|{optimizer}"]
            for multiplier, label in ((1.0, "base"), (0.5, "half")):
                steps = int(round(base_steps / multiplier))
                for seed in seeds:
                    tag = f"dt_{architecture}_{optimizer}_{label}_s{seed}"
                    configs.append(
                        (
                            tag,
                            FullTrainingConfig(
                                architecture=architecture,
                                affine_depth=5,
                                target_kind="powerlaw",
                                optimizer=optimizer,
                                learning_rate=base_rate * multiplier,
                                steps=steps,
                                record_every=max(5, steps // 100),
                                seed=seed,
                            ),
                        )
                    )
            for seed in seeds:
                tag = f"grid_{architecture}_{optimizer}_g256_s{seed}"
                configs.append(
                    (
                        tag,
                        FullTrainingConfig(
                            architecture=architecture,
                            affine_depth=5,
                            grid_size=256,
                            target_kind="powerlaw",
                            optimizer=optimizer,
                            learning_rate=base_rate,
                            steps=base_steps,
                            record_every=max(5, base_steps // 100),
                            seed=seed,
                        ),
                    )
                )
    for index, (tag, config) in enumerate(configs, start=1):
        trace, summary = train_case(config, tag=tag, device=device)
        traces.extend(trace)
        summaries.append(summary)
        print(
            f"[dt {index:02d}/{len(configs):02d}] {tag}: "
            f"loss={summary['final_loss']:.3e}, stable={summary['stable']}",
            flush=True,
        )
    write_csv(RESULTS / "discretization_traces.csv", traces)
    write_csv(RESULTS / "discretization_summaries.csv", summaries)


def run_discretization_quarter(device: torch.device, quick: bool) -> None:
    """Append a quarter-step refinement after the base/half-step campaign."""
    if quick:
        raise ValueError("quarter-step refinement is defined for the full campaign")
    trace_path = RESULTS / "discretization_traces.csv"
    summary_path = RESULTS / "discretization_summaries.csv"
    if not trace_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run --mode discretization before the quarter-step refinement")
    learning_rates = load_learning_rates()
    traces: list[dict[str, Any]] = [
        row for row in read_csv(trace_path) if "_quarter_" not in row["tag"]
    ]
    summaries: list[dict[str, Any]] = [
        row for row in read_csv(summary_path) if "_quarter_" not in row["tag"]
    ]
    configs: list[tuple[str, FullTrainingConfig]] = []
    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            base_rate = learning_rates[f"{architecture}|5|{optimizer}"]
            for seed in range(30, 33):
                tag = f"dt_{architecture}_{optimizer}_quarter_s{seed}"
                configs.append(
                    (
                        tag,
                        FullTrainingConfig(
                            architecture=architecture,
                            affine_depth=5,
                            target_kind="powerlaw",
                            optimizer=optimizer,
                            learning_rate=0.25 * base_rate,
                            steps=4_000,
                            record_every=40,
                            seed=seed,
                        ),
                    )
                )
    for index, (tag, config) in enumerate(configs, start=1):
        trace, summary = train_case(config, tag=tag, device=device)
        traces.extend(trace)
        summaries.append(summary)
        print(
            f"[quarter {index:02d}/{len(configs):02d}] {tag}: "
            f"loss={summary['final_loss']:.3e}, stable={summary['stable']}",
            flush=True,
        )
    write_csv(trace_path, traces)
    write_csv(summary_path, summaries)


def run_discretization_eighth(device: torch.device, quick: bool) -> None:
    """Append an eighth-step refinement when quarter-step drift remains."""
    if quick:
        raise ValueError("eighth-step refinement is defined for the full campaign")
    trace_path = RESULTS / "discretization_traces.csv"
    summary_path = RESULTS / "discretization_summaries.csv"
    if not trace_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run the preceding discretization stages first")
    learning_rates = load_learning_rates()
    traces: list[dict[str, Any]] = [
        row for row in read_csv(trace_path) if "_eighth_" not in row["tag"]
    ]
    summaries: list[dict[str, Any]] = [
        row for row in read_csv(summary_path) if "_eighth_" not in row["tag"]
    ]
    configs: list[tuple[str, FullTrainingConfig]] = []
    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            base_rate = learning_rates[f"{architecture}|5|{optimizer}"]
            for seed in range(30, 33):
                tag = f"dt_{architecture}_{optimizer}_eighth_s{seed}"
                configs.append(
                    (
                        tag,
                        FullTrainingConfig(
                            architecture=architecture,
                            affine_depth=5,
                            target_kind="powerlaw",
                            optimizer=optimizer,
                            learning_rate=0.125 * base_rate,
                            steps=8_000,
                            record_every=80,
                            seed=seed,
                        ),
                    )
                )
    for index, (tag, config) in enumerate(configs, start=1):
        trace, summary = train_case(config, tag=tag, device=device)
        traces.extend(trace)
        summaries.append(summary)
        print(
            f"[eighth {index:02d}/{len(configs):02d}] {tag}: "
            f"loss={summary['final_loss']:.3e}, stable={summary['stable']}",
            flush=True,
        )
    write_csv(trace_path, traces)
    write_csv(summary_path, summaries)


def plot_discretization() -> None:
    rows = read_csv(RESULTS / "discretization_traces.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.25), constrained_layout=True)
    colors = {"mup_gd": "#222222", "muon_p1_3": "#1b7837"}
    for axis, architecture in zip(axes[0], ("fc", "mmnn"), strict=True):
        for optimizer in ("mup_gd", "muon_p1_3"):
            for label, linestyle in (
                ("base", "-"),
                ("half", "--"),
                ("quarter", ":"),
                ("eighth", "-."),
            ):
                tags = sorted(
                    {
                        row["tag"]
                        for row in rows
                        if row["tag"].startswith(
                            f"dt_{architecture}_{optimizer}_{label}_s"
                        )
                    }
                )
                runs = [
                    sorted(
                        [row for row in rows if row["tag"] == tag],
                        key=lambda item: float(item["step"]),
                    )
                    for tag in tags
                ]
                optimizer_time = np.array(
                    [
                        float(row["step"]) * float(row["learning_rate"])
                        for row in runs[0]
                    ]
                )
                losses = np.array([[float(row["loss"]) for row in run] for run in runs])
                axis.semilogy(
                    optimizer_time,
                    np.median(losses, axis=0),
                    color=colors[optimizer],
                    linestyle=linestyle,
                    label=(
                        (r"$\mu$P--GD" if optimizer == "mup_gd" else r"Muon$^{1/3}$")
                        + {
                            "base": ", base step",
                            "half": ", half step",
                            "quarter": ", quarter step",
                            "eighth": ", eighth step",
                        }[label]
                    ),
                )
        axis.set(
            xlabel=r"optimizer time $\eta\,t$",
            ylabel="population loss",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.legend(frameon=False, fontsize=6.8, ncol=2)
    for axis, architecture in zip(axes[1], ("fc", "mmnn"), strict=True):
        for optimizer in ("mup_gd", "muon_p1_3"):
            base_tags = sorted(
                {
                    row["tag"]
                    for row in rows
                    if row["tag"].startswith(f"dt_{architecture}_{optimizer}_base_s")
                }
            )
            fine_tags = sorted(
                {
                    row["tag"]
                    for row in rows
                    if row["tag"].startswith(f"grid_{architecture}_{optimizer}_g256_s")
                }
            )
            for tags, linestyle, grid_size in (
                (base_tags, "-", 128),
                (fine_tags, ":", 256),
            ):
                runs = [
                    sorted(
                        [row for row in rows if row["tag"] == tag],
                        key=lambda item: float(item["step"]),
                    )
                    for tag in tags
                ]
                optimizer_time = np.array(
                    [
                        float(row["step"]) * float(row["learning_rate"])
                        for row in runs[0]
                    ]
                )
                losses = np.array([[float(row["loss"]) for row in run] for run in runs])
                axis.semilogy(
                    optimizer_time,
                    np.median(losses, axis=0),
                    color=colors[optimizer],
                    linestyle=linestyle,
                    label=(
                        (r"$\mu$P--GD" if optimizer == "mup_gd" else r"Muon$^{1/3}$")
                        + rf", grid {grid_size}"
                    ),
                )
        axis.set(
            xlabel=r"optimizer time $\eta\,t$",
            ylabel="population loss",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.legend(frameon=False, fontsize=7.0)
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURES / "full_training_step_convergence.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "full_training_step_convergence.png", bbox_inches="tight")
    plt.close(fig)


def _float_or_none(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value not in ("", "None") else None


def _median_event(summary: list[dict[str, str]], frequency: int) -> float | None:
    events = [_float_or_none(row, f"t50_{frequency}") for row in summary]
    observed = sorted(value for value in events if value is not None)
    if len(observed) < math.ceil(len(events) / 2):
        return None
    return observed[math.ceil(len(events) / 2) - 1]


def _scatter_observed_events(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    frequencies: tuple[int, ...],
    *,
    color: str,
) -> None:
    if not rows:
        return
    offsets = np.linspace(-0.035, 0.035, len(rows))
    for offset, row in zip(offsets, rows, strict=True):
        for frequency in frequencies:
            event = _float_or_none(row, f"t50_{frequency}")
            if event is not None:
                axis.scatter(
                    frequency * math.exp(float(offset)),
                    event,
                    color=color,
                    alpha=0.16,
                    s=8,
                    linewidths=0.0,
                    zorder=1,
                )


def plot_depth_confirmation(summary: list[dict[str, str]]) -> None:
    selected = [
        row
        for row in summary
        if row["target_kind"] == "hierarchy" and row["optimizer"] == "mup_gd"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)
    markers = {3: "o", 5: "s", 7: "^"}
    colors = {3: "#2166ac", 5: "#1b7837", 7: "#b2182b"}
    for axis, architecture in zip(axes, ("fc", "mmnn"), strict=True):
        for depth in (3, 5, 7):
            rows = [
                row
                for row in selected
                if row["architecture"] == architecture
                and int(row["affine_depth"]) == depth
            ]
            _scatter_observed_events(
                axis,
                rows,
                HIERARCHY_FREQUENCIES,
                color=colors[depth],
            )
            medians = [
                _median_event(rows, frequency) for frequency in HIERARCHY_FREQUENCIES
            ]
            observed_x = [
                frequency
                for frequency, value in zip(HIERARCHY_FREQUENCIES, medians, strict=True)
                if value is not None
            ]
            observed_y = [value for value in medians if value is not None]
            axis.plot(
                observed_x,
                observed_y,
                marker=markers[depth],
                color=colors[depth],
                label=rf"depth ${depth}$",
            )
            censored = [
                frequency
                for frequency, value in zip(HIERARCHY_FREQUENCIES, medians, strict=True)
                if value is None
            ]
            if censored:
                horizon = max(float(row["steps"]) for row in rows)
                axis.scatter(
                    censored,
                    [horizon] * len(censored),
                    marker="v",
                    facecolors="none",
                    edgecolors=colors[depth],
                )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel=r"frequency $q$",
            ylabel="median half-error step",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.set_xticks(HIERARCHY_FREQUENCIES, labels=HIERARCHY_FREQUENCIES)
        axis.legend(frameon=False)
    for label, axis in zip("ab", axes, strict=True):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURES / "full_training_depth_hierarchy.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "full_training_depth_hierarchy.png", bbox_inches="tight")
    plt.close(fig)


def plot_muon_hierarchy(summary: list[dict[str, str]]) -> None:
    selected = [
        row
        for row in summary
        if row["target_kind"] == "hierarchy" and int(row["affine_depth"]) == 5
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)
    colors = {
        "mup_gd": "#222222",
        "muon_p0": "#2166ac",
        "muon_p1_3": "#1b7837",
        "muon_p2_3": "#b2182b",
    }
    markers = {
        "mup_gd": "o",
        "muon_p0": "s",
        "muon_p1_3": "^",
        "muon_p2_3": "D",
    }
    labels = {
        "mup_gd": r"$\mu$P--GD",
        "muon_p0": r"Muon $p=0$",
        "muon_p1_3": r"Muon$^p$, $p=1/3$",
        "muon_p2_3": r"Muon$^p$, $p=2/3$",
    }
    for axis, architecture in zip(axes, ("fc", "mmnn"), strict=True):
        for optimizer in OPTIMIZER_POWERS:
            rows = [
                row
                for row in selected
                if row["architecture"] == architecture and row["optimizer"] == optimizer
            ]
            _scatter_observed_events(
                axis,
                rows,
                HIERARCHY_FREQUENCIES,
                color=colors[optimizer],
            )
            medians = [
                _median_event(rows, frequency) for frequency in HIERARCHY_FREQUENCIES
            ]
            observed = [
                (frequency, value)
                for frequency, value in zip(HIERARCHY_FREQUENCIES, medians, strict=True)
                if value is not None
            ]
            if observed:
                axis.plot(
                    [item[0] for item in observed],
                    [item[1] for item in observed],
                    marker=markers[optimizer],
                    color=colors[optimizer],
                    label=labels[optimizer],
                )
            censored = [
                frequency
                for frequency, value in zip(HIERARCHY_FREQUENCIES, medians, strict=True)
                if value is None
            ]
            if censored:
                horizon = max(float(row["steps"]) for row in rows)
                axis.scatter(
                    censored,
                    [horizon] * len(censored),
                    marker="v",
                    facecolors="none",
                    edgecolors=colors[optimizer],
                    s=24,
                )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel=r"frequency $q$",
            ylabel="median half-error step",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.set_xticks(HIERARCHY_FREQUENCIES, labels=HIERARCHY_FREQUENCIES)
        axis.legend(frameon=False, fontsize=7.6)
    for label, axis in zip("ab", axes, strict=True):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURES / "muon_hierarchy_clocks.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "muon_hierarchy_clocks.png", bbox_inches="tight")
    plt.close(fig)


def plot_powerlaw(summary: list[dict[str, str]]) -> None:
    selected = [
        row
        for row in summary
        if row["target_kind"] == "powerlaw"
        and int(row["affine_depth"]) == 5
        and row["tag"].startswith("powerlaw_")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)
    colors = {
        "mup_gd": "#222222",
        "muon_p0": "#2166ac",
        "muon_p1_3": "#1b7837",
        "muon_p2_3": "#b2182b",
    }
    markers = {
        "mup_gd": "o",
        "muon_p0": "s",
        "muon_p1_3": "^",
        "muon_p2_3": "D",
    }
    labels = {
        "mup_gd": r"$\mu$P--GD",
        "muon_p0": r"$p=0$",
        "muon_p1_3": r"$p=1/3$",
        "muon_p2_3": r"$p=2/3$",
    }
    frequencies = tuple(range(1, 25))
    for axis, architecture in zip(axes, ("fc", "mmnn"), strict=True):
        for optimizer in OPTIMIZER_POWERS:
            rows = [
                row
                for row in selected
                if row["architecture"] == architecture and row["optimizer"] == optimizer
            ]
            medians = [_median_event(rows, frequency) for frequency in frequencies]
            observed = [
                (frequency, value)
                for frequency, value in zip(frequencies, medians, strict=True)
                if value is not None
            ]
            if observed:
                axis.plot(
                    [item[0] for item in observed],
                    [item[1] for item in observed],
                    color=colors[optimizer],
                    marker=markers[optimizer],
                    markersize=2.8,
                    label=labels[optimizer],
                )
            censored = [
                frequency
                for frequency, value in zip(frequencies, medians, strict=True)
                if value is None
            ]
            if censored:
                horizon = max(float(row["steps"]) for row in rows)
                axis.scatter(
                    censored,
                    [horizon] * len(censored),
                    marker="v",
                    facecolors="none",
                    edgecolors=colors[optimizer],
                    s=12,
                )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel=r"Fourier frequency $q$",
            ylabel="median half-error step",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.legend(frameon=False, fontsize=7.6)
    for label, axis in zip("ab", axes, strict=True):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURES / "muon_powerlaw_front.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "muon_powerlaw_front.png", bbox_inches="tight")
    plt.close(fig)


def plot_paired_endpoint(summary: list[dict[str, str]]) -> None:
    optimizers = tuple(OPTIMIZER_POWERS)
    colors = {
        "mup_gd": "#222222",
        "muon_p0": "#2166ac",
        "muon_p1_3": "#1b7837",
        "muon_p2_3": "#b2182b",
    }
    markers = {
        "mup_gd": "o",
        "muon_p0": "s",
        "muon_p1_3": "^",
        "muon_p2_3": "D",
    }
    tick_labels = (r"$\mu$P--GD", r"$p=0$", r"$p=1/3$", r"$p=2/3$")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), constrained_layout=True)
    for row_axes, target_kind in zip(
        axes,
        ("hierarchy", "powerlaw"),
        strict=True,
    ):
        for axis, architecture in zip(
            row_axes,
            ("fc", "mmnn"),
            strict=True,
        ):
            selected = [
                row
                for row in summary
                if row["tag"].startswith(f"{target_kind}_{architecture}_d5_")
            ]
            by_optimizer = {
                optimizer: {
                    int(row["seed"]): float(row["final_loss"])
                    for row in selected
                    if row["optimizer"] == optimizer
                }
                for optimizer in optimizers
            }
            seeds = sorted(
                set.intersection(
                    *(set(by_optimizer[optimizer]) for optimizer in optimizers)
                )
            )
            for seed in seeds:
                values = [
                    by_optimizer[optimizer][seed] for optimizer in optimizers
                ]
                axis.plot(
                    range(len(optimizers)),
                    values,
                    color="0.72",
                    alpha=0.48,
                    linewidth=0.65,
                    zorder=1,
                )
            for index, optimizer in enumerate(optimizers):
                values = np.array(
                    [by_optimizer[optimizer][seed] for seed in seeds]
                )
                axis.scatter(
                    np.full(values.shape, index),
                    values,
                    marker=markers[optimizer],
                    color=colors[optimizer],
                    alpha=0.78,
                    s=20,
                    linewidths=0.0,
                    zorder=2,
                )
                median = float(np.median(values))
                axis.plot(
                    [index - 0.25, index + 0.25],
                    [median, median],
                    color="black",
                    linewidth=1.5,
                    zorder=3,
                )
            target_label = "sparse target" if target_kind == "hierarchy" else "power law"
            architecture_label = (
                "fully connected" if architecture == "fc" else "full-training MMNN"
            )
            axis.set(
                yscale="log",
                ylabel="terminal population loss",
                title=f"{architecture_label}, {target_label}",
                xticks=range(len(optimizers)),
                xticklabels=tick_labels,
            )
            axis.tick_params(axis="x", labelrotation=18)
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURES / "muon_paired_endpoints.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "muon_paired_endpoints.png", bbox_inches="tight")
    plt.close(fig)


def plot_width_transfer(traces: list[dict[str, str]]) -> None:
    selected = [row for row in traces if row["tag"].startswith("width_")]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)
    linestyles = {"mup_gd": "-", "muon_p1_3": "--"}
    palette = {64: "#2166ac", 128: "#1b7837", 256: "#b2182b"}
    for axis, architecture in zip(axes, ("fc", "mmnn"), strict=True):
        groups: dict[tuple[str, int], list[list[dict[str, str]]]] = {}
        tags = sorted(
            {row["tag"] for row in selected if row["architecture"] == architecture}
        )
        for tag in tags:
            rows = sorted(
                [row for row in selected if row["tag"] == tag],
                key=lambda item: float(item["step"]),
            )
            key = (rows[0]["optimizer"], int(rows[0]["width"]))
            groups.setdefault(key, []).append(rows)
        for (optimizer, width), runs in groups.items():
            steps = np.array([float(row["step"]) for row in runs[0]])
            losses = np.array([[float(row["loss"]) for row in run] for run in runs])
            axis.semilogy(
                steps,
                np.median(losses, axis=0),
                linestyle=linestyles[optimizer],
                color=palette[width],
                label=(
                    (r"Muon$^{1/3}$" if optimizer == "muon_p1_3" else r"$\mu$P--GD")
                    + rf", $m={width}$"
                ),
            )
        axis.set(
            xlabel="training step",
            ylabel="population loss",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.legend(frameon=False, fontsize=6.8, ncol=2)
    for label, axis in zip("ab", axes, strict=True):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURES / "muon_mup_width_transfer.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "muon_mup_width_transfer.png", bbox_inches="tight")
    plt.close(fig)


def plot_full_training_diagnostics(traces: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.25), constrained_layout=True)
    frequencies = np.array(HIERARCHY_FREQUENCIES)
    colors = {"initial": "#777777", "final": "#2166ac"}
    for axis, architecture in zip(axes[0], ("fc", "mmnn"), strict=True):
        selected = [
            row
            for row in traces
            if row["tag"].startswith(f"hierarchy_{architecture}_d5_mup_gd")
        ]
        tags = sorted({row["tag"] for row in selected})
        for stage in ("initial", "final"):
            values: list[list[float]] = []
            for tag in tags:
                rows = sorted(
                    [row for row in selected if row["tag"] == tag],
                    key=lambda item: float(item["step"]),
                )
                row = rows[0] if stage == "initial" else rows[-1]
                if all(row.get(f"lambda_{frequency}", "") for frequency in frequencies):
                    values.append(
                        [float(row[f"lambda_{frequency}"]) for frequency in frequencies]
                    )
            if values:
                mean = np.mean(values, axis=0)
                slope = np.polyfit(np.log(frequencies[1:]), np.log(mean[1:]), 1)[0]
                axis.loglog(
                    frequencies,
                    mean,
                    marker="o" if stage == "initial" else "s",
                    color=colors[stage],
                    label=f"{stage}, slope {slope:.2f}",
                )
        axis.set(
            xlabel=r"frequency $q$",
            ylabel=r"metric curvature $\Lambda_{qq}$",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.set_xticks(frequencies, labels=frequencies)
        axis.legend(frameon=False, fontsize=7.4)

    optimizer_colors = {
        "mup_gd": "#222222",
        "muon_p0": "#2166ac",
        "muon_p1_3": "#1b7837",
        "muon_p2_3": "#b2182b",
    }
    optimizer_markers = {
        "mup_gd": "o",
        "muon_p0": "s",
        "muon_p1_3": "^",
        "muon_p2_3": "D",
    }
    optimizer_labels = {
        "mup_gd": r"$\mu$P--GD",
        "muon_p0": r"$p=0$",
        "muon_p1_3": r"$p=1/3$",
        "muon_p2_3": r"$p=2/3$",
    }
    diagnostic_frequencies = np.array((1, 4, 8, 16, 24))
    for axis, architecture in zip(axes[1], ("fc", "mmnn"), strict=True):
        for optimizer in OPTIMIZER_POWERS:
            initial_rows = [
                row
                for row in traces
                if row["tag"].startswith(f"powerlaw_{architecture}_d5_{optimizer}_s")
                and int(float(row["step"])) == 0
            ]
            values = []
            for row in initial_rows:
                if all(
                    row.get(f"velocity_per_lr_{frequency}", "")
                    for frequency in diagnostic_frequencies
                ):
                    values.append(
                        [
                            float(row[f"velocity_per_lr_{frequency}"])
                            * float(row["learning_rate"])
                            for frequency in diagnostic_frequencies
                        ]
                    )
            if values:
                values_array = np.asarray(values)
                axis.scatter(
                    np.tile(diagnostic_frequencies, values_array.shape[0]),
                    values_array.reshape(-1),
                    color=optimizer_colors[optimizer],
                    alpha=0.22,
                    s=8,
                    linewidths=0.0,
                )
                axis.semilogx(
                    diagnostic_frequencies,
                    np.median(values_array, axis=0),
                    marker=optimizer_markers[optimizer],
                    color=optimizer_colors[optimizer],
                    label=optimizer_labels[optimizer],
                )
        axis.set(
            xlabel=r"frequency $q$",
            ylabel="signed initial coefficient gain / step",
            title="fully connected" if architecture == "fc" else "full-training MMNN",
        )
        axis.set_yscale("symlog", linthresh=1.0e-5)
        axis.axhline(0.0, color="0.75", linewidth=0.8)
        axis.set_xticks(diagnostic_frequencies, labels=diagnostic_frequencies)
        axis.legend(frameon=False, fontsize=7.3)
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(-0.15, 1.05, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(FIGURES / "full_training_dynamic_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "full_training_dynamic_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def _paired_bootstrap_median(
    values: np.ndarray, *, seed: int = 0, repetitions: int = 10_000
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        sample = generator.integers(0, values.size, size=values.size)
        estimates[index] = np.median(values[sample])
    return (
        float(np.median(values)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def analyze_results(
    traces: list[dict[str, str]], summary: list[dict[str, str]]
) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "paired_endpoint": {},
        "sector_clocks": {},
        "depth_hierarchy": {},
        "width_transfer": {},
        "initial_velocity": {},
        "wall_clock": {},
        "curvature": {},
    }
    for target_kind in ("hierarchy", "powerlaw"):
        for architecture in ("fc", "mmnn"):
            base_rows = {
                int(row["seed"]): row
                for row in summary
                if row["tag"].startswith(f"{target_kind}_{architecture}_d5_mup_gd_s")
            }
            for optimizer in ("muon_p0", "muon_p1_3", "muon_p2_3"):
                compared_rows = {
                    int(row["seed"]): row
                    for row in summary
                    if row["tag"].startswith(
                        f"{target_kind}_{architecture}_d5_{optimizer}_s"
                    )
                }
                seeds = sorted(set(base_rows) & set(compared_rows))
                differences = np.array(
                    [
                        math.log(
                            max(
                                float(compared_rows[seed]["final_loss"]),
                                1.0e-30,
                            )
                        )
                        - math.log(max(float(base_rows[seed]["final_loss"]), 1.0e-30))
                        for seed in seeds
                    ]
                )
                key = f"{target_kind}|{architecture}|{optimizer}"
                median, low, high = _paired_bootstrap_median(differences)
                analysis["paired_endpoint"][key] = {
                    "pairs": len(seeds),
                    "median_log_loss_difference": median,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "wins": int(np.sum(differences < 0.0)),
                }

    for target_kind, frequencies in (
        ("hierarchy", HIERARCHY_FREQUENCIES),
        ("powerlaw", tuple(range(1, 25))),
    ):
        for architecture in ("fc", "mmnn"):
            for optimizer in OPTIMIZER_POWERS:
                rows = [
                    row
                    for row in summary
                    if row["tag"].startswith(
                        f"{target_kind}_{architecture}_d5_{optimizer}_s"
                    )
                ]
                medians = {
                    str(frequency): _median_event(rows, frequency)
                    for frequency in frequencies
                }
                coverage = {
                    str(frequency): sum(
                        _float_or_none(row, f"t50_{frequency}") is not None
                        for row in rows
                    )
                    / max(len(rows), 1)
                    for frequency in frequencies
                }
                fit = [
                    (frequency, medians[str(frequency)])
                    for frequency in frequencies
                    if frequency >= 3 and medians[str(frequency)] is not None
                ]
                slope = None
                if len(fit) >= 3:
                    slope = float(
                        np.polyfit(
                            np.log([item[0] for item in fit]),
                            np.log([item[1] for item in fit]),
                            1,
                        )[0]
                    )
                analysis["sector_clocks"][
                    f"{target_kind}|{architecture}|{optimizer}"
                ] = {
                    "runs": len(rows),
                    "median_half_error_step": medians,
                    "event_coverage": coverage,
                    "loglog_slope": slope,
                    "median_final_feature_displacement": float(
                        np.median(
                            [float(row["final_feature_displacement"]) for row in rows]
                        )
                    ),
                }
                analysis["wall_clock"][f"{target_kind}|{architecture}|{optimizer}"] = {
                    "median_elapsed_seconds": float(
                        np.median([float(row["elapsed_seconds"]) for row in rows])
                    ),
                    "median_seconds_per_step": float(
                        np.median(
                            [
                                float(row["elapsed_seconds"])
                                / max(float(row["steps"]), 1.0)
                                for row in rows
                            ]
                        )
                    ),
                }

    for architecture in ("fc", "mmnn"):
        for depth in (3, 5, 7):
            rows = [
                row
                for row in summary
                if row["tag"].startswith(f"hierarchy_{architecture}_d{depth}_mup_gd_s")
            ]
            analysis["depth_hierarchy"][f"{architecture}|{depth}"] = {
                "runs": len(rows),
                "median_final_feature_displacement": float(
                    np.median(
                        [float(row["final_feature_displacement"]) for row in rows]
                    )
                ),
                "median_final_parameter_displacement": float(
                    np.median(
                        [float(row["final_parameter_displacement"]) for row in rows]
                    )
                ),
                "median_half_error_step": {
                    str(frequency): _median_event(rows, frequency)
                    for frequency in HIERARCHY_FREQUENCIES
                },
                "event_coverage": {
                    str(frequency): sum(
                        _float_or_none(row, f"t50_{frequency}") is not None
                        for row in rows
                    )
                    / max(len(rows), 1)
                    for frequency in HIERARCHY_FREQUENCIES
                },
            }

    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            for width in (64, 128, 256):
                rows = [
                    row
                    for row in summary
                    if row["tag"].startswith(
                        f"width_{architecture}_{optimizer}_m{width}_s"
                    )
                ]
                analysis["width_transfer"][f"{architecture}|{optimizer}|{width}"] = {
                    "runs": len(rows),
                    "median_final_loss": float(
                        np.median([float(row["final_loss"]) for row in rows])
                    ),
                    "median_feature_displacement": float(
                        np.median(
                            [float(row["final_feature_displacement"]) for row in rows]
                        )
                    ),
                }

    diagnostic_frequencies = (1, 4, 8, 16, 24)
    for architecture in ("fc", "mmnn"):
        for optimizer in OPTIMIZER_POWERS:
            initial_rows = [
                row
                for row in traces
                if row["tag"].startswith(f"powerlaw_{architecture}_d5_{optimizer}_s")
                and int(float(row["step"])) == 0
            ]
            analysis["initial_velocity"][f"{architecture}|{optimizer}"] = {
                str(frequency): {
                    "median_per_step": float(
                        np.median(
                            [
                                float(row[f"velocity_per_lr_{frequency}"])
                                * float(row["learning_rate"])
                                for row in initial_rows
                            ]
                        )
                    ),
                    "negative_fraction": sum(
                        float(row[f"velocity_per_lr_{frequency}"]) < 0.0
                        for row in initial_rows
                    )
                    / max(len(initial_rows), 1),
                }
                for frequency in diagnostic_frequencies
            }

    for architecture in ("fc", "mmnn"):
        selected = [
            row
            for row in traces
            if row["tag"].startswith(f"hierarchy_{architecture}_d5_mup_gd_s")
        ]
        tags = sorted({row["tag"] for row in selected})
        for stage in ("initial", "final"):
            values = []
            for tag in tags:
                rows = sorted(
                    [row for row in selected if row["tag"] == tag],
                    key=lambda item: float(item["step"]),
                )
                row = rows[0] if stage == "initial" else rows[-1]
                if all(row.get(f"lambda_{q}", "") for q in HIERARCHY_FREQUENCIES):
                    values.append(
                        [float(row[f"lambda_{q}"]) for q in HIERARCHY_FREQUENCIES]
                    )
            mean = np.mean(values, axis=0)
            analysis["curvature"][f"{architecture}|{stage}"] = {
                "mean": {
                    str(q): float(value)
                    for q, value in zip(HIERARCHY_FREQUENCIES, mean, strict=True)
                },
                "slope_q_ge_4": float(
                    np.polyfit(
                        np.log(HIERARCHY_FREQUENCIES[1:]),
                        np.log(mean[1:]),
                        1,
                    )[0]
                ),
            }
    (RESULTS / "full_training_analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True)
    )
    return analysis


def refresh_environment_metadata() -> None:
    path = RESULTS / "full_training_metadata.json"
    if not path.exists():
        return
    metadata = json.loads(path.read_text())
    metadata.update(
        {
            "dtype": str(torch.get_default_dtype()),
            "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }
    )
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def make_plots() -> None:
    set_plot_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    traces = read_csv(RESULTS / "full_training_traces.csv")
    summary = read_csv(RESULTS / "full_training_summaries.csv")
    plot_depth_confirmation(summary)
    plot_muon_hierarchy(summary)
    plot_powerlaw(summary)
    plot_paired_endpoint(summary)
    plot_width_transfer(traces)
    plot_full_training_diagnostics(traces)
    analyze_results(traces, summary)
    refresh_environment_metadata()


def run_pilot(args: argparse.Namespace, device: torch.device) -> None:
    config = FullTrainingConfig(
        architecture=args.architecture,
        affine_depth=args.depth,
        width=args.width,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        steps=args.steps,
        record_every=max(1, args.steps // 20),
        seed=args.seed,
        target_kind=args.target_kind,
    )
    trace, summary = train_case(config, tag="pilot", device=device)
    frequencies, _ = target_components(config)
    compact = {
        "stable": summary["stable"],
        "final_loss": summary["final_loss"],
        "feature_displacement": summary["final_feature_displacement"],
        "t50": {str(q): summary[f"t50_{q}"] for q in frequencies},
        "final_relative_error": {
            str(q): summary[f"final_relative_error_{q}"] for q in frequencies
        },
    }
    print(json.dumps(compact, indent=2))
    if args.pilot_output:
        output = Path(args.pilot_output)
        write_csv(output.with_suffix(".trace.csv"), trace)
        write_csv(output.with_suffix(".summary.csv"), [summary])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "pilot",
            "calibrate",
            "campaign",
            "discretization",
            "discretization-quarter",
            "discretization-eighth",
            "plots-only",
            "all",
        ),
        default="pilot",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--architecture", choices=("fc", "mmnn"), default="fc")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--optimizer", choices=tuple(OPTIMIZER_POWERS), default="mup_gd"
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-kind", choices=("hierarchy", "powerlaw"), default="hierarchy"
    )
    parser.add_argument("--pilot-output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "pilot":
        run_pilot(args, device)
        return
    if args.mode in ("calibrate", "all"):
        run_calibration(device, args.quick)
    if args.mode in ("campaign", "all"):
        run_campaign(device, args.quick)
    if args.mode == "discretization":
        run_discretization(device, args.quick)
        set_plot_style()
        FIGURES.mkdir(parents=True, exist_ok=True)
        plot_discretization()
    if args.mode == "discretization-quarter":
        run_discretization_quarter(device, args.quick)
        set_plot_style()
        FIGURES.mkdir(parents=True, exist_ok=True)
        plot_discretization()
    if args.mode == "discretization-eighth":
        run_discretization_eighth(device, args.quick)
        set_plot_style()
        FIGURES.mkdir(parents=True, exist_ok=True)
        plot_discretization()
    if args.mode in ("campaign", "plots-only", "all"):
        make_plots()


if __name__ == "__main__":
    main()
