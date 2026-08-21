"""MuP/DMFT diagnostics for frequency learning in a right-factor MMNN.

The experiment uses the centered mean-field parameterization

    z_a(x) = m^{-1/2} sum_j V_{ja} h_j(x),
    s_i(x) = r^{-1/2} sum_a U_{ia} z_a(x) + beta_i,
    f(x)   = (gamma n)^{-1} sum_i c_i [relu(s_i(x))-relu(s_i(x;V0))].

Only V is trained, with gradient-flow metric gamma^2 n I.  Thus gamma=O(1)
is the feature-learning (muP) limit, while gamma -> 0 freezes the features.
The centered output makes comparisons across gamma well conditioned at finite
width without changing any parameter derivatives.

The script writes tidy CSV/JSON data and publication-quality PDF figures.
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

from mmnn.mup_right_factor import CenteredRightFactorMuP


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


@dataclass(frozen=True)
class Config:
    grid_size: int = 256
    width: int = 256
    rank_ratio: float = 0.25
    gamma: float = 1.0
    dt: float = 5.0
    steps: int = 5_000
    record_every: int = 50
    seed: int = 0
    frequencies: tuple[int, ...] = (1, 4, 8, 16)
    amplitudes: tuple[float, ...] = (1.0, 0.65, 0.45, 0.30)
    bias_scale_1: float = 0.35
    bias_scale_2: float = 0.15


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
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


def cosine_target(
    x: torch.Tensor, frequencies: Iterable[int], amplitudes: Iterable[float]
) -> torch.Tensor:
    return sum(
        amplitude * torch.cos(frequency * x)
        for frequency, amplitude in zip(frequencies, amplitudes, strict=True)
    )


def cosine_coefficient(values: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    return 2.0 * torch.mean(values * torch.cos(k * x))


def tangent_fourier_matrix(
    model: CenteredRightFactorMuP,
    x: torch.Tensor,
    frequencies: tuple[int, ...],
) -> np.ndarray:
    prediction = model()
    grads: list[torch.Tensor] = []
    for index, frequency in enumerate(frequencies):
        coefficient = cosine_coefficient(prediction, x, frequency)
        gradient = torch.autograd.grad(
            coefficient,
            model.V,
            retain_graph=index + 1 < len(frequencies),
            create_graph=False,
        )[0]
        # e_k=sqrt(2)cos(kx), hence <f,e_k>=coefficient/sqrt(2).
        grads.append(gradient.detach().flatten() / math.sqrt(2.0))
    stacked = torch.stack(grads)
    matrix = model.metric_scale * (stacked @ stacked.T)
    return matrix.detach().cpu().double().numpy()


def recovery_time(
    trace: list[dict[str, float | int | str]], frequency: int, fraction: float
) -> float | None:
    key = f"recovery_{frequency}"
    for row in trace:
        if float(row[key]) >= fraction:
            return float(row["time"])
    return None


def train_case(
    config: Config,
    *,
    diagnostics: bool,
    tag: str,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    x = periodic_grid(config.grid_size, device)
    rank = max(1, int(round(config.width * config.rank_ratio)))
    target = cosine_target(x, config.frequencies, config.amplitudes)
    model = CenteredRightFactorMuP(
        x,
        width=config.width,
        rank=rank,
        gamma=config.gamma,
        seed=config.seed,
        bias_scale_1=config.bias_scale_1,
        bias_scale_2=config.bias_scale_2,
    )
    checkpoints = set(range(0, config.steps + 1, config.record_every))
    checkpoints.add(config.steps)
    kernel_stride = max(config.record_every, config.steps // 24)
    kernel_checkpoints = set(range(0, config.steps + 1, kernel_stride))
    kernel_checkpoints.add(config.steps)
    v0_norm = torch.linalg.vector_norm(model.V0)
    z0 = model.latent(model.V0).detach()
    initial_gates = model.s0 > 0
    initial_fourier: np.ndarray | None = None
    trace: list[dict[str, float | int | str]] = []

    for step in range(config.steps + 1):
        prediction = model()
        residual = prediction - target
        loss = 0.5 * torch.mean(residual**2)

        if step in checkpoints:
            with torch.no_grad():
                row: dict[str, float | int | str] = {
                    "tag": tag,
                    "parameterization": "muP" if config.gamma >= 0.5 else "lazy",
                    "seed": config.seed,
                    "width": config.width,
                    "rank": rank,
                    "rank_ratio": config.rank_ratio,
                    "gamma": config.gamma,
                    "step": step,
                    "time": config.dt * step,
                    "loss": float(loss),
                    "relative_v_displacement": float(
                        torch.linalg.vector_norm(model.V - model.V0) / v0_norm
                    ),
                    "relative_latent_displacement": float(
                        torch.linalg.vector_norm(model.latent() - z0)
                        / torch.linalg.vector_norm(z0)
                    ),
                    "gate_flip_fraction": float(
                        torch.mean(
                            ((model.preactivation() > 0) != initial_gates).float()
                        )
                    ),
                }
                for frequency, amplitude in zip(
                    config.frequencies, config.amplitudes, strict=True
                ):
                    coefficient = float(
                        cosine_coefficient(prediction, x, frequency)
                    )
                    row[f"coefficient_{frequency}"] = coefficient
                    row[f"recovery_{frequency}"] = coefficient / amplitude

            if diagnostics and step in kernel_checkpoints:
                fourier = tangent_fourier_matrix(model, x, config.frequencies)
                if initial_fourier is None:
                    initial_fourier = fourier.copy()
                diagonal = np.maximum(np.diag(fourier), 1.0e-30)
                off_diagonal = fourier - np.diag(np.diag(fourier))
                row["fourier_offdiag_ratio"] = float(
                    np.linalg.norm(off_diagonal) / np.linalg.norm(fourier)
                )
                for index, frequency in enumerate(config.frequencies):
                    row[f"lambda_{frequency}"] = float(diagonal[index])
                    row[f"amplification_{frequency}"] = float(
                        diagonal[index]
                        / max(initial_fourier[index, index], 1.0e-30)
                    )
                # The full matrix is retained in a compact JSON field for
                # auditing frequency coupling without proliferating columns.
                row["fourier_matrix"] = json.dumps(fourier.tolist())
            trace.append(row)

        if step == config.steps:
            break
        gradient = torch.autograd.grad(loss, model.V)[0]
        with torch.no_grad():
            model.V.add_(gradient, alpha=-config.dt * model.metric_scale)

    summary: dict[str, object] = {
        **asdict(config),
        "rank": rank,
        "tag": tag,
        "final_loss": float(trace[-1]["loss"]),
        "final_relative_latent_displacement": float(
            trace[-1]["relative_latent_displacement"]
        ),
        "final_gate_flip_fraction": float(trace[-1]["gate_flip_fraction"]),
    }
    for frequency in config.frequencies:
        summary[f"t20_{frequency}"] = recovery_time(trace, frequency, 0.20)
        summary[f"t50_{frequency}"] = recovery_time(trace, frequency, 0.50)
        summary[f"t90_{frequency}"] = recovery_time(trace, frequency, 0.90)
    return trace, summary


def saddle_spectrum_case(
    config: Config,
    *,
    max_frequency: int,
    tag: str,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Fit the first mode, then measure the dynamic Fourier tangent spectrum."""
    x = periodic_grid(config.grid_size, device)
    rank = max(1, int(round(config.width * config.rank_ratio)))
    target = torch.cos(x)
    model = CenteredRightFactorMuP(
        x,
        width=config.width,
        rank=rank,
        gamma=config.gamma,
        seed=config.seed,
        bias_scale_1=config.bias_scale_1,
        bias_scale_2=config.bias_scale_2,
    )
    frequencies = tuple(range(1, max_frequency + 1))
    initial = tangent_fourier_matrix(model, x, frequencies)
    z0 = model.latent(model.V0).detach()
    initial_gates = model.s0 > 0
    for _ in range(config.steps):
        prediction = model()
        loss = 0.5 * torch.mean((prediction - target) ** 2)
        gradient = torch.autograd.grad(loss, model.V)[0]
        with torch.no_grad():
            model.V.add_(gradient, alpha=-config.dt * model.metric_scale)
    final_prediction = model()
    final_loss = 0.5 * torch.mean((final_prediction - target) ** 2)
    final = tangent_fourier_matrix(model, x, frequencies)
    rows: list[dict[str, object]] = []
    for index, frequency in enumerate(frequencies):
        initial_value = max(float(initial[index, index]), 1.0e-30)
        final_value = max(float(final[index, index]), 1.0e-30)
        rows.append(
            {
                "tag": tag,
                "parameterization": "muP" if config.gamma >= 0.5 else "lazy",
                "seed": config.seed,
                "width": config.width,
                "rank": rank,
                "rank_ratio": config.rank_ratio,
                "gamma": config.gamma,
                "frequency": frequency,
                "lambda_initial": initial_value,
                "lambda_plateau": final_value,
                "saddle_index": 1.0 / final_value,
                "dynamic_amplification": final_value / initial_value,
                "scaled_bv_proxy": frequency**2 * final_value,
            }
        )
    summary: dict[str, object] = {
        "tag": tag,
        "parameterization": "muP" if config.gamma >= 0.5 else "lazy",
        "seed": config.seed,
        "width": config.width,
        "rank": rank,
        "rank_ratio": config.rank_ratio,
        "gamma": config.gamma,
        "final_loss": float(final_loss),
        "relative_latent_displacement": float(
            torch.linalg.vector_norm(model.latent() - z0)
            / torch.linalg.vector_norm(z0)
        ),
        "gate_flip_fraction": float(
            torch.mean(((model.preactivation() > 0) != initial_gates).float())
        ),
        "offdiag_ratio": float(
            np.linalg.norm(final - np.diag(np.diag(final)))
            / np.linalg.norm(final)
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_primary(device: torch.device, quick: bool) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    configs: list[tuple[str, Config, bool]] = []
    steps = 600 if quick else 5_000
    record_every = 20 if quick else 50
    for gamma, name in ((1.0, "muP"), (1.0e-4, "lazy")):
        for seed in ((0,) if quick else (0, 1, 2)):
            configs.append(
                (
                    f"hierarchy_{name}_s{seed}",
                    Config(
                        gamma=gamma,
                        seed=seed,
                        steps=steps,
                        record_every=record_every,
                        width=128 if quick else 256,
                        grid_size=128 if quick else 256,
                        dt=5.0,
                    ),
                    True,
                )
            )

    # Frequency-gap and absolute-frequency controls.  The preceding mode is
    # either fixed at 1 or moved with q, separating gap from absolute q.
    q_values = (2, 4, 8) if quick else (2, 4, 6, 8, 12, 16, 24)
    for gamma, name in ((1.0, "muP"), (1.0e-4, "lazy")):
        for q in q_values:
            for seed in ((0,) if quick else (0, 1, 2)):
                configs.append(
                    (
                        f"gap_{name}_q{q}_s{seed}",
                        Config(
                            grid_size=128,
                            width=128,
                            rank_ratio=0.25,
                            gamma=gamma,
                            dt=5.0,
                            steps=steps,
                            record_every=record_every,
                            seed=seed,
                            frequencies=(1, q),
                            amplitudes=(1.0, 0.5),
                        ),
                        False,
                    )
                )
        # Hold the raw gap fixed at three while translating both frequencies.
        # This is the direct test that absolute q, rather than q-p alone, sets
        # the geometric Fourier difficulty.
        for q in ((4, 8) if quick else (4, 8, 12, 16, 24)):
            p = q - 3
            configs.append(
                (
                    f"fixed_gap_{name}_p{p}_q{q}",
                    Config(
                        grid_size=128,
                        width=128,
                        rank_ratio=0.25,
                        gamma=gamma,
                        dt=5.0,
                        steps=steps,
                        record_every=record_every,
                        seed=17,
                        frequencies=(p, q),
                        amplitudes=(1.0, 0.5),
                    ),
                    False,
                )
            )
            if q >= 4:
                p = q // 2
                configs.append(
                    (
                        f"gap_control_{name}_p{p}_q{q}",
                        Config(
                            grid_size=128,
                            width=128,
                            rank_ratio=0.25,
                            gamma=gamma,
                            dt=5.0,
                            steps=steps,
                            record_every=record_every,
                            seed=11,
                            frequencies=(p, q),
                            amplitudes=(1.0, 0.5),
                        ),
                        False,
                    )
                )

    # Width collapse and rank-ratio ablations use one representative gap.
    widths = (64, 128) if quick else (64, 128, 256, 512)
    for width in widths:
        configs.append(
            (
                f"width_muP_n{width}",
                Config(
                    grid_size=128,
                    width=width,
                    rank_ratio=0.25,
                    gamma=1.0,
                    steps=steps,
                    record_every=record_every,
                    frequencies=(1, 8),
                    amplitudes=(1.0, 0.5),
                    seed=7,
                ),
                False,
            )
        )
    ratios = (0.125, 0.5) if quick else (0.0625, 0.125, 0.25, 0.5, 1.0)
    for ratio in ratios:
        for seed in ((0,) if quick else (0, 1, 2)):
            configs.append(
                (
                    f"rank_muP_rho{ratio:g}_s{seed}",
                    Config(
                        grid_size=128,
                        width=256,
                        rank_ratio=ratio,
                        gamma=1.0,
                        steps=steps,
                        record_every=record_every,
                        frequencies=(1, 8, 16),
                        amplitudes=(1.0, 0.5, 0.3),
                        seed=seed,
                    ),
                    False,
                )
            )

    all_trace: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    started = time.time()
    for index, (tag, config, diagnostics) in enumerate(configs, start=1):
        trace, summary = train_case(
            config, diagnostics=diagnostics, tag=tag, device=device
        )
        all_trace.extend(trace)
        all_summary.append(summary)
        print(
            f"[{index:03d}/{len(configs):03d}] {tag}: "
            f"loss={float(summary['final_loss']):.3e}",
            flush=True,
        )

    spectrum_rows: list[dict[str, object]] = []
    spectrum_summary: list[dict[str, object]] = []
    for gamma, name in ((1.0, "muP"), (1.0e-4, "lazy")):
        for seed in ((0,) if quick else (0, 1, 2)):
            spectrum_config = Config(
                grid_size=128 if quick else 256,
                width=128 if quick else 256,
                rank_ratio=0.25,
                gamma=gamma,
                dt=5.0,
                steps=600 if quick else 1_000,
                seed=seed,
                frequencies=(1,),
                amplitudes=(1.0,),
            )
            tag = f"spectrum_{name}_s{seed}"
            rows, summary = saddle_spectrum_case(
                spectrum_config,
                max_frequency=12 if quick else 32,
                tag=tag,
                device=device,
            )
            spectrum_rows.extend(rows)
            spectrum_summary.append(summary)
            print(
                f"[spectrum] {tag}: loss={float(summary['final_loss']):.3e}",
                flush=True,
            )

    write_csv(RESULTS / "training_traces.csv", all_trace)
    write_csv(RESULTS / "run_summaries.csv", all_summary)
    write_csv(RESULTS / "saddle_spectra.csv", spectrum_rows)
    write_csv(RESULTS / "saddle_spectrum_summaries.csv", spectrum_summary)
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.time() - started,
        "quick": quick,
        "number_of_runs": len(configs) + len(spectrum_summary),
        "parameterization_note": (
            "Centered output; only V trained; gradient metric gamma^2 * width."
        ),
    }
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2))


def group_rows(rows: list[dict[str, str]], prefix: str) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["tag"].startswith(prefix):
            result.setdefault(row["tag"], []).append(row)
    for values in result.values():
        values.sort(key=lambda item: float(item["time"]))
    return result


def plot_hierarchy(rows: list[dict[str, str]]) -> None:
    groups = group_rows(rows, "hierarchy_")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.35), constrained_layout=True)
    colors = {1: "#2166ac", 4: "#67a9cf", 8: "#ef8a62", 16: "#b2182b"}
    for parameterization, linestyle in (("muP", "-"), ("lazy", "--")):
        tags = [tag for tag in groups if f"hierarchy_{parameterization}" in tag]
        for frequency in (1, 4, 8, 16):
            curves = []
            for tag in tags:
                values = groups[tag]
                curves.append(
                    np.array(
                        [float(row[f"recovery_{frequency}"]) for row in values]
                    )
                )
            if not curves:
                continue
            time_axis = np.array([float(row["time"]) for row in groups[tags[0]]])
            matrix = np.stack(curves)
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            label = rf"$k={frequency}$, {parameterization}"
            axes[0, 0].plot(
                time_axis, mean, linestyle, color=colors[frequency], label=label
            )
            axes[0, 0].fill_between(
                time_axis,
                mean - std,
                mean + std,
                color=colors[frequency],
                alpha=0.10,
            )
        loss_curves = np.stack(
            [[float(row["loss"]) for row in groups[tag]] for tag in tags]
        )
        time_axis = np.array([float(row["time"]) for row in groups[tags[0]]])
        axes[0, 1].semilogy(
            time_axis,
            loss_curves.mean(axis=0),
            linestyle,
            linewidth=2,
            label=parameterization,
        )

    diagnostic_tags = [tag for tag in groups if "hierarchy_muP" in tag]
    for frequency in (1, 4, 8, 16):
        curves: list[tuple[np.ndarray, np.ndarray]] = []
        for tag in diagnostic_tags:
            selected = [row for row in groups[tag] if row.get(f"lambda_{frequency}")]
            curves.append(
                (
                    np.array([float(row["time"]) for row in selected]),
                    np.array(
                        [float(row[f"amplification_{frequency}"]) for row in selected]
                    ),
                )
            )
        if curves:
            axes[1, 0].semilogy(
                curves[0][0],
                np.stack([curve[1] for curve in curves]).mean(axis=0),
                color=colors[frequency],
                label=rf"$k={frequency}$",
            )
    latent_axis = axes[1, 1]
    gate_axis = latent_axis.twinx()
    for parameterization, linestyle in (("muP", "-"), ("lazy", "--")):
        tags = [tag for tag in groups if f"hierarchy_{parameterization}" in tag]
        for quantity, color, label, axis in (
            (
                "relative_latent_displacement",
                "#542788",
                r"latent drift",
                latent_axis,
            ),
            ("gate_flip_fraction", "#1b7837", r"gate flips", gate_axis),
        ):
            curves = np.stack(
                [[float(row[quantity]) for row in groups[tag]] for tag in tags]
            )
            time_axis = np.array([float(row["time"]) for row in groups[tags[0]]])
            axis.plot(
                time_axis,
                curves.mean(axis=0),
                linestyle,
                color=color,
                label=f"{label}, {parameterization}",
            )

    axes[0, 0].axhline(0.9, color="0.65", linewidth=0.8)
    axes[0, 0].set(ylabel="target-mode recovery", xlabel="gradient-flow time")
    axes[0, 0].set_ylim(-0.08, 1.15)
    axes[0, 0].legend(ncol=2, frameon=False, fontsize=7.2)
    axes[0, 1].set(ylabel="population loss", xlabel="gradient-flow time")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set(
        ylabel=r"dynamic amplification $\Lambda_{kk}(t)/\Lambda_{kk}(0)$",
        xlabel="gradient-flow time",
    )
    axes[1, 0].legend(frameon=False, ncol=2)
    latent_axis.set(
        ylabel="relative latent displacement",
        xlabel="gradient-flow time",
    )
    gate_axis.set_ylabel("gate-flip fraction")
    latent_axis.tick_params(axis="y", colors="#542788")
    gate_axis.tick_params(axis="y", colors="#1b7837")
    latent_axis.spines["left"].set_color("#542788")
    gate_axis.spines["right"].set_visible(True)
    gate_axis.spines["right"].set_color("#1b7837")
    handles_left, labels_left = latent_axis.get_legend_handles_labels()
    handles_right, labels_right = gate_axis.get_legend_handles_labels()
    latent_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        fontsize=7.2,
        loc="center right",
    )
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.savefig(FIGURES / "hierarchy_and_kernel_evolution.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "hierarchy_and_kernel_evolution.png", bbox_inches="tight")
    plt.close(fig)


def plot_gap_and_rank(summary: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55), constrained_layout=True)
    for parameterization, marker, color in (
        ("muP", "o", "#2166ac"),
        ("lazy", "s", "#b2182b"),
    ):
        grouped: dict[int, list[float]] = {}
        censor_limits: dict[int, float] = {}
        for row in summary:
            if not row["tag"].startswith(f"gap_{parameterization}_q"):
                continue
            q = int(row["frequencies"].strip("()").split(",")[-1])
            value = row.get(f"t20_{q}", "")
            if value:
                grouped.setdefault(q, []).append(float(value))
            else:
                censor_limits[q] = float(row["steps"]) * float(row["dt"])
        q_values = np.array(sorted(grouped))
        means = np.array([np.mean(grouped[q]) for q in q_values])
        stds = np.array([np.std(grouped[q]) for q in q_values])
        axes[0].errorbar(
            q_values,
            means,
            yerr=stds,
            marker=marker,
            color=color,
            capsize=2,
            label=parameterization,
        )
        if censor_limits:
            censored = sorted(censor_limits)
            axes[0].scatter(
                censored,
                [censor_limits[q] for q in censored],
                marker="^",
                facecolors="none",
                edgecolors=color,
            )
    axes[0].set(
        xlabel=r"next frequency $q$",
        ylabel=r"20\% recovery time $T_q$",
        xscale="log",
        yscale="log",
    )
    axes[0].legend(frameon=False)

    # The raw gap is exactly three in every run on this panel.
    for parameterization, marker, color in (
        ("muP", "o", "#2166ac"),
        ("lazy", "s", "#b2182b"),
    ):
        controls = [
            row
            for row in summary
            if row["tag"].startswith(f"fixed_gap_{parameterization}")
        ]
        q_values: list[int] = []
        times: list[float] = []
        censored_q: list[int] = []
        for row in controls:
            frequencies = [
                int(value) for value in row["frequencies"].strip("()").split(",")
            ]
            q = frequencies[-1]
            value = row.get(f"t20_{q}", "")
            if value:
                q_values.append(q)
                times.append(float(value))
            else:
                censored_q.append(q)
        axes[1].plot(
            q_values,
            times,
            marker=marker,
            color=color,
            label=parameterization,
        )
        if censored_q:
            censor_height = max(times, default=1.0) * 1.12
            axes[1].scatter(
                censored_q,
                [censor_height] * len(censored_q),
                marker="^",
                facecolors="none",
                edgecolors=color,
            )
    axes[1].set(
        xlabel=r"next $q$ (fixed $q-p=3$)",
        ylabel=r"20\% recovery time $T_q$",
        yscale="log",
    )
    axes[1].legend(frameon=False, fontsize=7.7)

    rank_rows = [row for row in summary if row["tag"].startswith("rank_muP")]
    grouped_rank: dict[float, list[float]] = {}
    for row in rank_rows:
        value = row.get("t20_8", "")
        if value:
            grouped_rank.setdefault(float(row["rank_ratio"]), []).append(float(value))
    ratios = np.array(sorted(grouped_rank))
    if ratios.size:
        means = np.array([np.mean(grouped_rank[ratio]) for ratio in ratios])
        stds = np.array([np.std(grouped_rank[ratio]) for ratio in ratios])
        axes[2].errorbar(
            ratios,
            means,
            yerr=stds,
            marker="o",
            color="#1b7837",
            capsize=2,
        )
    axes[2].set(
        xlabel=r"rank ratio $\rho=r/m$",
        ylabel=r"$k=8$ 20\% recovery time",
        xscale="log",
    )
    if not grouped_rank:
        axes[2].set_xscale("linear")
    for label, axis in zip("abc", axes, strict=True):
        axis.text(
            -0.17,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.savefig(FIGURES / "frequency_rank_controls.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "frequency_rank_controls.png", bbox_inches="tight")
    plt.close(fig)


def plot_width_collapse(rows: list[dict[str, str]]) -> None:
    groups = group_rows(rows, "width_muP")
    fig, axes = plt.subplots(1, 2, figsize=(5.05, 2.35), constrained_layout=True)
    palette = plt.cm.viridis(np.linspace(0.12, 0.90, max(1, len(groups))))
    final_curves: list[tuple[int, np.ndarray]] = []
    for color, (tag, values) in zip(palette, sorted(groups.items()), strict=True):
        width = int(values[0]["width"])
        time_axis = np.array([float(row["time"]) for row in values])
        loss = np.array([float(row["loss"]) for row in values])
        recovery = np.array([float(row["recovery_8"]) for row in values])
        axes[0].semilogy(time_axis, loss, color=color, label=rf"$m={width}$")
        axes[1].plot(time_axis, recovery, color=color, label=rf"$m={width}$")
        final_curves.append((width, loss))
    axes[0].set(xlabel="gradient-flow time", ylabel="population loss")
    axes[1].set(xlabel="gradient-flow time", ylabel=r"$k=8$ recovery")
    axes[1].axhline(0.9, color="0.7", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[0].text(-0.18, 1.05, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.18, 1.05, "b", transform=axes[1].transAxes, fontweight="bold")
    fig.savefig(FIGURES / "mup_width_collapse.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "mup_width_collapse.png", bbox_inches="tight")
    plt.close(fig)


def plot_saddle_spectrum(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.5), constrained_layout=True)
    palette = {"muP": "#2166ac", "lazy": "#b2182b"}
    for parameterization in ("muP", "lazy"):
        selected = [row for row in rows if row["parameterization"] == parameterization]
        frequencies = sorted({int(row["frequency"]) for row in selected})
        by_seed: dict[int, dict[int, dict[str, str]]] = {}
        for row in selected:
            by_seed.setdefault(int(row["seed"]), {})[int(row["frequency"])] = row
        lambdas = np.array(
            [
                [float(by_seed[seed][frequency]["lambda_plateau"]) for frequency in frequencies]
                for seed in sorted(by_seed)
            ]
        )
        amplification = np.array(
            [
                [float(by_seed[seed][frequency]["dynamic_amplification"]) for frequency in frequencies]
                for seed in sorted(by_seed)
            ]
        )
        mean_lambda = lambdas.mean(axis=0)
        mean_amp = amplification.mean(axis=0)
        fit_mask = np.array(frequencies) >= 3
        slope = np.polyfit(
            np.log(np.array(frequencies)[fit_mask]),
            np.log(mean_lambda[fit_mask]),
            1,
        )[0]
        axes[0].loglog(
            frequencies,
            mean_lambda,
            marker="o" if parameterization == "muP" else "s",
            color=palette[parameterization],
            label=rf"{parameterization}, slope {slope:.2f}",
        )
        axes[1].loglog(
            frequencies,
            1.0 / mean_lambda,
            marker="o" if parameterization == "muP" else "s",
            color=palette[parameterization],
            label=parameterization,
        )
        axes[2].semilogx(
            frequencies,
            mean_amp,
            marker="o" if parameterization == "muP" else "s",
            color=palette[parameterization],
            label=parameterization,
        )
    axes[0].set(xlabel=r"frequency $q$", ylabel=r"plateau curvature $\Lambda_{qq}$")
    axes[1].set(xlabel=r"frequency $q$", ylabel=r"saddle index $1/\Lambda_{qq}$")
    axes[2].axhline(1.0, color="0.65", linewidth=0.8)
    axes[2].set(
        xlabel=r"frequency $q$",
        ylabel=r"kernel change $\Lambda_{qq}(t)/\Lambda_{qq}(0)$",
    )
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].legend(frameon=False)
    axes[2].legend(frameon=False)
    for label, axis in zip("abc", axes, strict=True):
        axis.text(
            -0.17,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    fig.savefig(FIGURES / "dynamic_saddle_spectrum.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "dynamic_saddle_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def make_plots() -> None:
    set_plot_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RESULTS / "training_traces.csv")
    summary = read_csv(RESULTS / "run_summaries.csv")
    spectra = read_csv(RESULTS / "saddle_spectra.csv")
    plot_hierarchy(rows)
    plot_gap_and_rank(summary)
    plot_width_collapse(rows)
    plot_saddle_spectrum(spectra)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.plots_only:
        run_primary(device, quick=args.quick)
    make_plots()


if __name__ == "__main__":
    main()
