#!/usr/bin/env python3
"""Power-law scaling and early stopping under a PSD NTK deformation.

To keep the finite-rank correction positive semidefinite, use

    K_sigma = (diag(sqrt(eta)) + sigma W)^2,

with GOE-normalized W.  The eigenvectors are those of the deformed-Wigner
matrix B=diag(sqrt(eta))+sigma W and the kernel eigenvalues are its squared
eigenvalues.  The Dyson local spectral density rho_i^B(s) therefore gives the
deterministic-equivalent learning curve

    L_sigma(t) = sum_i eta_i integral c_t(s^2) rho_i^B(s) ds.

For sigma=0 this reduces exactly to the aligned power-law model of
Kramp--Lindner--Helias.  The script computes minimum-risk and optimal-time
scaling without replacing the unresolved eigenbasis by an ad hoc Haar block.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deformed_wigner_dyson import solve_dyson


def spectral_cost(rate: np.ndarray, sample_size: float, time: float, beta: float):
    rate = np.asarray(rate, dtype=float)
    exponent = np.exp(-2.0 * sample_size * time * rate)
    variance = np.empty_like(rate)
    positive = rate > 1e-14
    variance[positive] = (1.0 - exponent[positive]) / (
        2.0 * sample_size * beta * rate[positive]
    )
    variance[~positive] = time / beta
    return 0.5 * exponent + variance


def normalized_local_density(grid: np.ndarray, diagonal_resolvent: np.ndarray):
    local = np.imag(diagonal_resolvent) / np.pi
    masses = np.trapezoid(local, grid, axis=0)
    if np.any(masses <= 0):
        raise FloatingPointError("local spectral measure lost positive mass")
    return local / masses[None, :]


def dyson_risk_curve(
    eta: np.ndarray,
    singular_grid: np.ndarray,
    local_density: np.ndarray,
    sample_size: int,
    beta: float,
    times: np.ndarray,
) -> np.ndarray:
    rates = singular_grid**2
    weighted_local = local_density @ eta
    return np.asarray(
        [
            np.trapezoid(
                spectral_cost(rates, sample_size, time, beta) * weighted_local,
                singular_grid,
            )
            for time in times
        ]
    )


def aligned_risk_curve(
    eta: np.ndarray, sample_size: int, beta: float, times: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [np.sum(eta * spectral_cost(eta, sample_size, time, beta)) for time in times]
    )


def run_experiment(
    output_dir: Path,
    *,
    dimension: int,
    alpha: float,
    beta: float,
    sigmas: tuple[float, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eta = np.arange(1, dimension + 1, dtype=float) ** (-(1.0 + alpha))
    signal = np.sqrt(eta)
    sample_sizes = np.unique(np.geomspace(24, 4096, 18).astype(int))
    theoretical_time = 0.5 * beta * alpha / (1.0 + alpha)
    times = np.geomspace(theoretical_time / 80.0, theoretical_time * 12.0, 360)
    dyson_measures: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for sigma in sigmas:
        if sigma == 0.0:
            continue
        grid = np.linspace(-2.25 * sigma, 1.0 + 2.25 * sigma, 1500)
        _, diagonal = solve_dyson(
            signal,
            grid,
            sigma,
            imaginary_part=max(7e-4, sigma / 28.0),
        )
        dyson_measures[sigma] = (grid, normalized_local_density(grid, diagonal))

    rows = []
    curves: dict[tuple[float, int], np.ndarray] = {}
    for sigma in sigmas:
        for sample_size in sample_sizes:
            if sigma == 0.0:
                risk = aligned_risk_curve(eta, sample_size, beta, times)
            else:
                grid, local = dyson_measures[sigma]
                risk = dyson_risk_curve(
                    eta, grid, local, sample_size, beta, times
                )
            optimum = int(np.argmin(risk))
            curves[(sigma, int(sample_size))] = risk
            rows.append(
                {
                    "sigma": sigma,
                    "sample_size": int(sample_size),
                    "minimum_risk": float(risk[optimum]),
                    "optimal_time": float(times[optimum]),
                    "aligned_asymptotic_time": theoretical_time,
                    "time_ratio": float(times[optimum] / theoretical_time),
                }
            )
    with (output_dir / "dyson_powerlaw_early_stopping.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(sigmas)))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 3.8), constrained_layout=True)
    for color, sigma in zip(colors, sigmas):
        selected = [row for row in rows if row["sigma"] == sigma]
        p = np.asarray([row["sample_size"] for row in selected])
        minimum = np.asarray([row["minimum_risk"] for row in selected])
        optimum = np.asarray([row["optimal_time"] for row in selected])
        label = rf"$\sigma={sigma:g}$"
        axes[0].loglog(p, minimum, marker="o", markersize=3.5, color=color, label=label)
        axes[1].semilogx(p, optimum, marker="o", markersize=3.5, color=color, label=label)
    reference_p = np.asarray([sample_sizes[0], sample_sizes[-1]])
    aligned_first = next(row for row in rows if row["sigma"] == 0.0)
    reference = float(aligned_first["minimum_risk"]) * (
        reference_p / reference_p[0]
    ) ** (-alpha / (1.0 + alpha))
    axes[0].loglog(
        reference_p,
        reference,
        "k--",
        linewidth=1.3,
        label=rf"$P^{{-{alpha/(1+alpha):.2f}}}$",
    )
    axes[1].axhline(
        theoretical_time,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=rf"$t_0^*={theoretical_time:.3g}$",
    )

    representative_p = int(sample_sizes[len(sample_sizes) // 2])
    for color, sigma in zip(colors, sigmas):
        risk = curves[(sigma, representative_p)]
        axes[2].loglog(times, risk, color=color, linewidth=1.8, label=rf"$\sigma={sigma:g}$")
    axes[2].axvline(theoretical_time, color="black", linestyle="--", linewidth=1.2)

    axes[0].set_title("minimum test risk")
    axes[0].set_xlabel("number of samples $P$")
    axes[0].set_ylabel(r"$\min_t\,\mathcal{L}(t)$")
    axes[1].set_title("optimal stopping time")
    axes[1].set_xlabel("number of samples $P$")
    axes[1].set_ylabel(r"$t^*$")
    axes[2].set_title(rf"learning curves at $P={representative_p}$")
    axes[2].set_xlabel("training time $t$")
    axes[2].set_ylabel(r"$\mathcal{L}(t)$")
    for axis in axes:
        axis.grid(alpha=0.22, which="both")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "dyson_powerlaw_early_stopping.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "dyson_powerlaw_early_stopping.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sigmas", type=float, nargs="+", default=(0.0, 0.01, 0.025, 0.05))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/dyson_powerlaw_early_stopping"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args.output_dir,
        dimension=args.dimension,
        alpha=args.alpha,
        beta=args.beta,
        sigmas=tuple(args.sigmas),
    )
    print(f"wrote Dyson early-stopping results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
