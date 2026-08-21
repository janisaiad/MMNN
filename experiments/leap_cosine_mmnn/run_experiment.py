"""Numerical tests of frequency-indexed approximate saddles in a low-rank MMNN.

The network has two hidden ReLU layers (three affine maps including the scalar
readout).  Its inner matrix is W_2 = U V^T.  The left factor U and every other
weight are frozen; only the right factor V is trained.  This is the
fixed-left/right-trainable factorization used by the MMNN interpretation in
the accompanying note.

The script produces three complementary diagnostics:

1. mode-by-mode recovery for a sum of four cosines;
2. escape time from the low-frequency plateau as the next frequency is moved
   farther away;
3. the Fourier spectrum of the right-factor tangent kernel after fitting the
   first mode, and the associated inverse-curvature (saddle) index.

Run from the repository root with

    uv run python experiments/leap_cosine_mmnn/run_experiment.py

Use --quick for a short smoke run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mmnn.right_factor import RightFactorMMNN


@dataclass(frozen=True)
class ExperimentConfig:
    grid_size: int = 512
    feature_width: int = 192
    outer_width: int = 192
    rank: int = 8
    bias_scale_1: float = 0.5
    bias_scale_2: float = 0.2
    init_scale: float = 0.02
    learning_rate: float = 2.0e-3
    steps: int = 20_000
    record_every: int = 10
    recovery_fraction: float = 0.90
    high_amplitude: float = 0.5
    high_frequencies: tuple[int, ...] = (2, 4, 6, 8, 12, 16)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hierarchy_frequencies: tuple[int, ...] = (1, 4, 8, 16)
    hierarchy_amplitudes: tuple[float, ...] = (1.0, 0.5, 0.35, 0.25)
    kernel_max_frequency: int = 32
    kernel_pretrain_steps: int = 3_000


def periodic_grid(size: int, device: torch.device) -> torch.Tensor:
    return torch.arange(size, device=device) * (2.0 * math.pi / size)


def cosine_target(
    x: torch.Tensor, frequencies: Iterable[int], amplitudes: Iterable[float]
) -> torch.Tensor:
    terms = [
        amplitude * torch.cos(frequency * x)
        for frequency, amplitude in zip(frequencies, amplitudes, strict=True)
    ]
    return torch.stack(terms, dim=0).sum(dim=0)


def cosine_coefficient(
    values: torch.Tensor, x: torch.Tensor, frequency: int
) -> torch.Tensor:
    return 2.0 * torch.mean(values * torch.cos(frequency * x))


def make_model(
    config: ExperimentConfig, seed: int, device: torch.device, rank: int | None = None
) -> RightFactorMMNN:
    return RightFactorMMNN(
        feature_width=config.feature_width,
        outer_width=config.outer_width,
        rank=config.rank if rank is None else rank,
        seed=seed,
        device=device,
        bias_scale_1=config.bias_scale_1,
        bias_scale_2=config.bias_scale_2,
        init_scale=config.init_scale,
    )


def first_recovery_time(
    coefficient: float, target: float, fraction: float
) -> bool:
    if target == 0.0:
        return abs(coefficient) <= 1.0 - fraction
    return abs(coefficient - target) <= (1.0 - fraction) * abs(target)


def train_gap_case(
    config: ExperimentConfig,
    *,
    high_frequency: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float | int | bool | None], list[dict[str, float | int]]]:
    x = periodic_grid(config.grid_size, device)
    target = cosine_target(x, (1, high_frequency), (1.0, config.high_amplitude))
    model = make_model(config, seed, device)
    optimizer = torch.optim.Adam((model.V,), lr=config.learning_rate)

    low_time: int | None = None
    high_time: int | None = None
    trace: list[dict[str, float | int]] = []

    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = 0.5 * torch.mean((prediction - target) ** 2)

        with torch.no_grad():
            low_coefficient = float(cosine_coefficient(prediction, x, 1))
            high_coefficient = float(
                cosine_coefficient(prediction, x, high_frequency)
            )
            if low_time is None and first_recovery_time(
                low_coefficient, 1.0, config.recovery_fraction
            ):
                low_time = step
            if high_time is None and first_recovery_time(
                high_coefficient,
                config.high_amplitude,
                config.recovery_fraction,
            ):
                high_time = step
            if step % config.record_every == 0 or step == config.steps:
                trace.append(
                    {
                        "seed": seed,
                        "high_frequency": high_frequency,
                        "frequency_gap": high_frequency - 1,
                        "step": step,
                        "loss": float(loss),
                        "low_coefficient": low_coefficient,
                        "high_coefficient": high_coefficient,
                    }
                )

        if step == config.steps:
            break
        loss.backward()
        optimizer.step()

    plateau_steps = None
    if low_time is not None and high_time is not None:
        plateau_steps = max(0, high_time - low_time)
    summary: dict[str, float | int | bool | None] = {
        "seed": seed,
        "high_frequency": high_frequency,
        "frequency_gap": high_frequency - 1,
        "low_recovery_step": low_time,
        "high_recovery_step": high_time,
        "plateau_steps": plateau_steps,
        "censored": high_time is None,
        "final_loss": trace[-1]["loss"],
        "final_low_coefficient": trace[-1]["low_coefficient"],
        "final_high_coefficient": trace[-1]["high_coefficient"],
    }
    return summary, trace


def logarithmic_checkpoints(steps: int) -> set[int]:
    positive = np.unique(
        np.geomspace(1, max(1, steps), num=min(320, steps), dtype=int)
    )
    return {0, steps, *positive.tolist()}


def train_hierarchy(
    config: ExperimentConfig, device: torch.device
) -> list[dict[str, float | int]]:
    x = periodic_grid(config.grid_size, device)
    target = cosine_target(
        x, config.hierarchy_frequencies, config.hierarchy_amplitudes
    )
    model = make_model(config, seed=0, device=device)
    optimizer = torch.optim.Adam((model.V,), lr=config.learning_rate)
    checkpoints = logarithmic_checkpoints(config.steps)
    rows: list[dict[str, float | int]] = []

    for step in range(config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = 0.5 * torch.mean((prediction - target) ** 2)
        if step in checkpoints:
            row: dict[str, float | int] = {"step": step, "loss": float(loss)}
            with torch.no_grad():
                for frequency, amplitude in zip(
                    config.hierarchy_frequencies,
                    config.hierarchy_amplitudes,
                    strict=True,
                ):
                    coefficient = float(
                        cosine_coefficient(prediction, x, frequency)
                    )
                    row[f"coefficient_{frequency}"] = coefficient
                    row[f"recovery_{frequency}"] = coefficient / amplitude
            rows.append(row)
        if step == config.steps:
            break
        loss.backward()
        optimizer.step()
    return rows


def tangent_spectrum_after_low_mode(
    config: ExperimentConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], float]:
    x = periodic_grid(config.grid_size, device)
    target = torch.cos(x)
    model = make_model(config, seed, device)
    optimizer = torch.optim.Adam((model.V,), lr=config.learning_rate)
    low_loss = math.inf
    for _ in range(config.kernel_pretrain_steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = 0.5 * torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
        low_loss = float(loss)

    rows: list[dict[str, float | int]] = []
    for frequency in range(1, config.kernel_max_frequency + 1):
        prediction = model(x)
        coefficient = cosine_coefficient(prediction, x, frequency)
        coefficient_gradient = torch.autograd.grad(coefficient, model.V)[0]
        coefficient_gradient_sq = float(torch.sum(coefficient_gradient**2))
        # With e_k=sqrt(2) cos(kx), lambda_k=||grad <f,e_k>||^2=G_k/2.
        eigenvalue = 0.5 * coefficient_gradient_sq
        rows.append(
            {
                "seed": seed,
                "frequency": frequency,
                "frequency_gap": max(0, frequency - 1),
                "coefficient_gradient_sq": coefficient_gradient_sq,
                "tangent_eigenvalue": eigenvalue,
                "saddle_index": 1.0 / max(eigenvalue, 1.0e-30),
            }
        )
    return rows, low_loss


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loglog_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x_valid = x[valid]
    y_valid = y[valid]
    if x_valid.size < 2:
        return math.nan, math.nan, math.nan
    slope, intercept = np.polyfit(np.log(x_valid), np.log(y_valid), 1)
    fitted = intercept + slope * np.log(x_valid)
    residual = np.log(y_valid) - fitted
    total = np.log(y_valid) - np.mean(np.log(y_valid))
    r_squared = 1.0 - float(np.sum(residual**2) / max(np.sum(total**2), 1e-30))
    return float(slope), float(intercept), r_squared


def aggregate_gap_rows(
    rows: list[dict[str, float | int | bool | None]], max_steps: int
) -> list[dict[str, float | int]]:
    aggregates: list[dict[str, float | int]] = []
    for frequency in sorted({int(row["high_frequency"]) for row in rows}):
        selected = [row for row in rows if int(row["high_frequency"]) == frequency]
        uncensored = [
            float(row["plateau_steps"])
            for row in selected
            if row["plateau_steps"] is not None
        ]
        # Restricted mean with right-censored runs placed at the observation horizon.
        restricted = [
            float(row["plateau_steps"])
            if row["plateau_steps"] is not None
            else float(max_steps)
            for row in selected
        ]
        aggregates.append(
            {
                "high_frequency": frequency,
                "frequency_gap": frequency - 1,
                "runs": len(selected),
                "uncensored_runs": len(uncensored),
                "censored_runs": len(selected) - len(uncensored),
                "median_plateau_steps": float(np.median(restricted)),
                "q25_plateau_steps": float(np.quantile(restricted, 0.25)),
                "q75_plateau_steps": float(np.quantile(restricted, 0.75)),
            }
        )
    return aggregates


def aggregate_kernel_rows(
    rows: list[dict[str, float | int]]
) -> list[dict[str, float | int]]:
    aggregates: list[dict[str, float | int]] = []
    frequencies = sorted({int(row["frequency"]) for row in rows})
    for frequency in frequencies:
        selected = [row for row in rows if int(row["frequency"]) == frequency]
        eigenvalues = np.asarray(
            [float(row["tangent_eigenvalue"]) for row in selected]
        )
        aggregates.append(
            {
                "frequency": frequency,
                "frequency_gap": max(0, frequency - 1),
                "median_tangent_eigenvalue": float(np.median(eigenvalues)),
                "q25_tangent_eigenvalue": float(np.quantile(eigenvalues, 0.25)),
                "q75_tangent_eigenvalue": float(np.quantile(eigenvalues, 0.75)),
                "median_saddle_index": float(np.median(1.0 / eigenvalues)),
            }
        )
    return aggregates


def plot_hierarchy(rows: list[dict[str, float | int]], config: ExperimentConfig, path: Path) -> None:
    steps = np.asarray([int(row["step"]) for row in rows])
    positive_steps = np.maximum(steps, 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for frequency in config.hierarchy_frequencies:
        recovery = np.asarray([float(row[f"recovery_{frequency}"]) for row in rows])
        axes[0].plot(positive_steps, recovery, label=rf"$k={frequency}$")
    axes[0].axhline(config.recovery_fraction, color="0.45", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.08, 1.35)
    axes[0].set_xlabel("optimization step")
    axes[0].set_ylabel("Fourier coefficient / target coefficient")
    axes[0].set_title("Sequential Fourier-mode recovery")
    axes[0].legend(frameon=False, ncol=2)

    losses = np.asarray([float(row["loss"]) for row in rows])
    axes[1].loglog(positive_steps, losses, color="#3B6FB6")
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel(r"$\frac{1}{2}\|f-y\|_{L^2}^2$")
    axes[1].set_title("Loss stages induced by unresolved modes")
    axes[1].grid(alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_gap_scaling(
    aggregates: list[dict[str, float | int]], path: Path
) -> tuple[float, float]:
    gaps = np.asarray([float(row["frequency_gap"]) for row in aggregates])
    medians = np.asarray([float(row["median_plateau_steps"]) for row in aggregates])
    q25 = np.asarray([float(row["q25_plateau_steps"]) for row in aggregates])
    q75 = np.asarray([float(row["q75_plateau_steps"]) for row in aggregates])
    slope, intercept, r_squared = loglog_fit(gaps, medians)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.fill_between(gaps, q25, q75, alpha=0.22, color="#D4663A", label="seed IQR")
    ax.plot(gaps, medians, "o-", color="#B84A24", label="median escape time")
    if math.isfinite(slope):
        fit_x = np.geomspace(gaps.min(), gaps.max(), 200)
        fit_y = np.exp(intercept) * fit_x**slope
        ax.plot(
            fit_x,
            fit_y,
            "--",
            color="0.2",
            label=rf"fit $\Delta k^{{{slope:.2f}}}$, $R^2={r_squared:.2f}$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"frequency leap $\Delta k=k_{\rm next}-1$")
    ax.set_ylabel("steps between 90% recovery events")
    ax.set_title("Larger frequency leaps produce longer plateaux")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return slope, r_squared


def plot_kernel_scaling(
    aggregates: list[dict[str, float | int]], path: Path
) -> tuple[float, float]:
    frequencies = np.asarray([float(row["frequency"]) for row in aggregates])
    median = np.asarray(
        [float(row["median_tangent_eigenvalue"]) for row in aggregates]
    )
    q25 = np.asarray([float(row["q25_tangent_eigenvalue"]) for row in aggregates])
    q75 = np.asarray([float(row["q75_tangent_eigenvalue"]) for row in aggregates])
    fit_mask = frequencies >= 8
    slope, intercept, r_squared = loglog_fit(frequencies[fit_mask], median[fit_mask])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].fill_between(frequencies, q25, q75, alpha=0.22, color="#3B6FB6")
    axes[0].loglog(frequencies, median, "o-", color="#2C5C99", markersize=3)
    if math.isfinite(slope):
        fit_x = np.geomspace(frequencies[fit_mask].min(), frequencies.max(), 200)
        axes[0].loglog(
            fit_x,
            np.exp(intercept) * fit_x**slope,
            "--",
            color="0.2",
            label=rf"tail $k^{{{slope:.2f}}}$, $R^2={r_squared:.2f}$",
        )
    axes[0].set_xlabel("frequency k")
    axes[0].set_ylabel(r"right-factor tangent eigenvalue $\lambda_k$")
    axes[0].set_title("High-frequency curvature collapses")
    axes[0].grid(alpha=0.22, which="both")
    axes[0].legend(frameon=False)

    saddle_index = 1.0 / median
    axes[1].loglog(frequencies, saddle_index, "o-", color="#B84A24", markersize=3)
    axes[1].set_xlabel("frequency k")
    axes[1].set_ylabel(r"saddle index $S_k=1/\lambda_k$")
    axes[1].set_title("Approximate saddles grow with frequency")
    axes[1].grid(alpha=0.22, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return slope, r_squared


def quick_config() -> ExperimentConfig:
    return ExperimentConfig(
        grid_size=256,
        feature_width=96,
        outer_width=96,
        rank=6,
        steps=2_000,
        record_every=20,
        high_frequencies=(2, 4, 8),
        seeds=(0, 1),
        hierarchy_frequencies=(1, 3, 6, 10),
        hierarchy_amplitudes=(1.0, 0.5, 0.35, 0.25),
        kernel_max_frequency=16,
        kernel_pretrain_steps=600,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run a short smoke experiment")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/leap_cosine_mmnn/results"),
        help="directory for CSV, JSON, and figures",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = quick_config() if args.quick else ExperimentConfig()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_default_dtype(torch.float32)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(f"device={device}; output={output}")

    hierarchy_rows = train_hierarchy(config, device)
    write_csv(output / "hierarchy_trajectory.csv", hierarchy_rows)
    plot_hierarchy(hierarchy_rows, config, output / "hierarchy_trajectory.png")

    gap_rows: list[dict[str, float | int | bool | None]] = []
    gap_traces: list[dict[str, float | int]] = []
    for high_frequency in config.high_frequencies:
        for seed in config.seeds:
            summary, trace = train_gap_case(
                config,
                high_frequency=high_frequency,
                seed=seed,
                device=device,
            )
            gap_rows.append(summary)
            gap_traces.extend(trace)
            print(
                f"gap={high_frequency - 1:2d}, seed={seed}, "
                f"plateau={summary['plateau_steps']}, censored={summary['censored']}"
            )
    write_csv(output / "gap_runs.csv", gap_rows)
    write_csv(output / "gap_traces.csv", gap_traces)
    gap_aggregates = aggregate_gap_rows(gap_rows, config.steps)
    write_csv(output / "gap_summary.csv", gap_aggregates)
    escape_slope, escape_r_squared = plot_gap_scaling(
        gap_aggregates, output / "escape_time_vs_frequency_leap.png"
    )

    kernel_rows: list[dict[str, float | int]] = []
    pretrain_losses: dict[str, float] = {}
    for seed in config.seeds:
        rows, low_loss = tangent_spectrum_after_low_mode(
            config, seed=seed, device=device
        )
        kernel_rows.extend(rows)
        pretrain_losses[str(seed)] = low_loss
    write_csv(output / "tangent_spectrum_runs.csv", kernel_rows)
    kernel_aggregates = aggregate_kernel_rows(kernel_rows)
    write_csv(output / "tangent_spectrum_summary.csv", kernel_aggregates)
    kernel_slope, kernel_r_squared = plot_kernel_scaling(
        kernel_aggregates, output / "tangent_spectrum_and_saddle_index.png"
    )

    elapsed = time.perf_counter() - start
    summary = {
        "config": asdict(config),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "escape_time_loglog_slope": escape_slope,
        "escape_time_loglog_r_squared": escape_r_squared,
        "kernel_tail_loglog_slope_k_ge_8": kernel_slope,
        "kernel_tail_loglog_r_squared_k_ge_8": kernel_r_squared,
        "predicted_relu_kernel_tail_slope": -2.0,
        "low_mode_pretrain_losses": pretrain_losses,
        "interpretation": (
            "S_k=1/lambda_k is twice the loss gap divided by squared gradient "
            "norm at the ideal state that has fitted all lower modes."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
