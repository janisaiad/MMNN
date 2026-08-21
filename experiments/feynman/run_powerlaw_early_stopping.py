#!/usr/bin/env python3
"""Power-law data, finite-NTK spectral mixing, and early stopping.

This experiment connects the low-rank finite-width NTK scale to the
teacher--student power-law model of Kramp, Lindner, and Helias
(arXiv:2602.23039).  It has two complementary parts.

1.  A block-Haar model in which the well-separated spectral head stays
    aligned with the population modes and the near-degenerate tail is Haar
    mixed.  Orthogonal Weingarten calculus gives the exact mean and variance
    of the bias-plus-variance risk over the Haar eigenvectors.

2.  A positive-semidefinite Wigner deformation

        K_hat = (Lambda^{1/2} + tau W)(Lambda^{1/2} + tau W)^T,

    where W is GOE-normalized.  This realizes a Wigner-like correction while
    keeping the training kernel positive semidefinite.  The deformation is
    used to measure the loss of population-eigenvector alignment, changes in
    the data-scaling exponent, and shifts of the optimal stopping time.

For an arbitrary positive-semidefinite training operator K_hat, the plotted
risk is the exact teacher-averaged linear/Langevin expression

  L(t) = 1/2 Tr[Lambda exp(-2 P t K_hat)]
       + 1/(2 P beta) Tr[Lambda K_hat^dagger
                          (I - exp(-2 P t K_hat))].

When K_hat=Lambda this reduces to the power-law model's Ornstein--Uhlenbeck
bias and high-temperature variance approximations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (6.4, 4.6),
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def powerlaw_spectrum(num_modes: int, alpha: float, eta_one: float = 1.0) -> np.ndarray:
    indices = np.arange(1, num_modes + 1, dtype=float)
    return eta_one * indices ** (-(1.0 + alpha))


def gamma_defect(width: int, rank: int) -> float:
    if not (1 <= rank <= width):
        raise ValueError("rank must lie in [1,width]")
    if rank == width:
        return 0.0
    return width * (width - rank) / (rank * (width - 1) * (width + 2))


def epsilon_width_rank(width: int, rank: int) -> float:
    return 1.0 / width + gamma_defect(width, rank)


def kramp_stopping_time(alpha: float, beta: float) -> float:
    return 0.5 * beta * alpha / (1.0 + alpha)


def stable_response(eigenvalues: np.ndarray, sample_size: float, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exp(-2Pt kappa) and (1-exp(-2Pt kappa))/kappa.

    The second function is continued continuously at kappa=0, where it is
    equal to 2Pt.
    """
    kappa = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    t = np.asarray(times, dtype=float)
    argument = 2.0 * sample_size * t[:, None] * kappa[None, :]
    exponential = np.exp(-argument)
    numerator = -np.expm1(-argument)
    response = np.empty_like(numerator)
    positive = kappa > 1e-14
    response[:, positive] = numerator[:, positive] / kappa[None, positive]
    response[:, ~positive] = 2.0 * sample_size * t[:, None]
    return exponential, response


def aligned_risk_curve(
    population_eigenvalues: np.ndarray,
    sample_size: float,
    beta: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exponential, response = stable_response(population_eigenvalues, sample_size, times)
    eta = population_eigenvalues[None, :]
    bias = 0.5 * np.sum(eta * exponential, axis=1)
    variance = np.sum(eta * response, axis=1) / (2.0 * sample_size * beta)
    return bias + variance, bias, variance


def kernel_risk_curve(
    kernel: np.ndarray,
    population_eigenvalues: np.ndarray,
    sample_size: float,
    beta: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact teacher-averaged risk for a possibly misaligned PSD kernel."""
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    spectral_population_weight = np.sum(
        population_eigenvalues[:, None] * eigenvectors**2, axis=0
    )
    exponential, response = stable_response(eigenvalues, sample_size, times)
    bias = 0.5 * (exponential @ spectral_population_weight)
    variance = (response @ spectral_population_weight) / (2.0 * sample_size * beta)
    return bias + variance, bias, variance, eigenvalues, eigenvectors


def haar_orthogonal(size: int, rng: np.random.Generator) -> np.ndarray:
    gaussian = rng.normal(size=(size, size))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def orthogonal_twirl_moments(left_spectrum: np.ndarray, right_spectrum: np.ndarray) -> tuple[float, float]:
    """Exact O(N) mean/variance of Tr(A Q B Q^T) for diagonal A,B."""
    a = np.asarray(left_spectrum, dtype=float)
    b = np.asarray(right_spectrum, dtype=float)
    if a.shape != b.shape:
        raise ValueError("the two spectra must have the same shape")
    size = a.size
    if size < 2:
        return float(a[0] * b[0]), 0.0
    mean = float(np.sum(a) * np.sum(b) / size)
    a_centered_sq = float(np.sum(a**2) - np.sum(a) ** 2 / size)
    b_centered_sq = float(np.sum(b**2) - np.sum(b) ** 2 / size)
    variance = 2.0 * a_centered_sq * b_centered_sq / ((size - 1) * (size + 2))
    return mean, max(variance, 0.0)


def block_haar_risk_moments(
    population_eigenvalues: np.ndarray,
    training_eigenvalues: np.ndarray,
    aligned_head: int,
    sample_size: float,
    beta: float,
    time: float,
) -> tuple[float, float]:
    """Exact mean/variance when only the spectral tail is Haar mixed."""
    eta = np.asarray(population_eigenvalues, dtype=float)
    kappa = np.asarray(training_eigenvalues, dtype=float)
    if eta.shape != kappa.shape:
        raise ValueError("population and training spectra must have equal length")
    if not (0 <= aligned_head < eta.size):
        raise ValueError("aligned_head must leave a nonempty Haar tail")
    exponential, response = stable_response(kappa, sample_size, np.array([time]))
    spectral_cost = 0.5 * exponential[0] + response[0] / (2.0 * sample_size * beta)
    head = float(np.sum(eta[:aligned_head] * spectral_cost[:aligned_head]))
    tail_mean, tail_variance = orthogonal_twirl_moments(
        eta[aligned_head:], spectral_cost[aligned_head:]
    )
    return head + tail_mean, tail_variance


def sample_block_haar_risk(
    population_eigenvalues: np.ndarray,
    training_eigenvalues: np.ndarray,
    aligned_head: int,
    sample_size: float,
    beta: float,
    time: float,
    rng: np.random.Generator,
) -> float:
    eta = np.asarray(population_eigenvalues, dtype=float)
    kappa = np.asarray(training_eigenvalues, dtype=float)
    exponential, response = stable_response(kappa, sample_size, np.array([time]))
    spectral_cost = 0.5 * exponential[0] + response[0] / (2.0 * sample_size * beta)
    head = float(np.sum(eta[:aligned_head] * spectral_cost[:aligned_head]))
    q_tail = haar_orthogonal(eta.size - aligned_head, rng)
    mixed_cost = q_tail @ np.diag(spectral_cost[aligned_head:]) @ q_tail.T
    tail = float(np.sum(eta[aligned_head:] * np.diag(mixed_cost)))
    return head + tail


def goe_matrix(size: int, rng: np.random.Generator) -> np.ndarray:
    gaussian = rng.normal(size=(size, size))
    return (gaussian + gaussian.T) / math.sqrt(2.0 * size)


def psd_wigner_deformation(
    population_eigenvalues: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """PSD kernel whose leading correction has Wigner-like eigenvectors."""
    root = np.diag(np.sqrt(population_eigenvalues))
    amplitude = root + tau * goe_matrix(population_eigenvalues.size, rng)
    return amplitude @ amplitude.T


def optimal_point(times: np.ndarray, risks: np.ndarray) -> tuple[float, float]:
    index = int(np.nanargmin(risks))
    return float(times[index]), float(risks[index])


def gap_mixing_index(population_eigenvalues: np.ndarray, perturbation_norm: float) -> int:
    """Number of leading adjacent gaps larger than twice the perturbation."""
    gaps = population_eigenvalues[:-1] - population_eigenvalues[1:]
    good = np.flatnonzero(gaps > 2.0 * perturbation_norm)
    if good.size == 0:
        return 1
    contiguous = 0
    for index in good:
        if index != contiguous:
            break
        contiguous += 1
    return max(1, contiguous + 1)


def asymptotic_mixing_index(alpha: float, eta_one: float, perturbation_norm: float, num_modes: int) -> int:
    if perturbation_norm <= 0.0:
        return num_modes
    value = ((1.0 + alpha) * eta_one / (2.0 * perturbation_norm)) ** (1.0 / (2.0 + alpha))
    return int(np.clip(round(value), 1, num_modes))


def run_weingarten_check(
    outdir: Path,
    eta: np.ndarray,
    sample_size: float,
    beta: float,
    time: float,
    aligned_head: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    exact_mean, exact_variance = block_haar_risk_moments(
        eta, eta, aligned_head, sample_size, beta, time
    )
    samples = np.array(
        [
            sample_block_haar_risk(
                eta, eta, aligned_head, sample_size, beta, time, rng
            )
            for _ in range(repetitions)
        ]
    )
    row = {
        "num_modes": eta.size,
        "aligned_head": aligned_head,
        "tail_size": eta.size - aligned_head,
        "repetitions": repetitions,
        "time": time,
        "exact_mean": exact_mean,
        "empirical_mean": float(np.mean(samples)),
        "exact_std": math.sqrt(exact_variance),
        "empirical_std": float(np.std(samples, ddof=1)),
        "mean_standardized_error": float(
            (np.mean(samples) - exact_mean)
            / max(math.sqrt(exact_variance / repetitions), 1e-15)
        ),
    }
    write_csv(outdir / "weingarten_risk_check.csv", [row])

    fig, ax = plt.subplots()
    ax.hist(samples, bins=30, density=True, alpha=0.55, color="#0077BB", label="Haar samples")
    ax.axvline(exact_mean, color="#CC3311", linewidth=2.0, label="exact mean")
    ax.axvspan(
        exact_mean - math.sqrt(exact_variance),
        exact_mean + math.sqrt(exact_variance),
        color="#CC3311",
        alpha=0.13,
        label=r"exact $\pm1$ std.",
    )
    ax.set_xlabel(r"test risk $\mathcal{L}(t)$")
    ax.set_ylabel("density")
    ax.set_title("Block-Haar risk: exact Weingarten moments")
    ax.legend(frameon=False)
    savefig(fig, outdir, "weingarten_risk_distribution")
    return row


def run_deformation_scaling(
    outdir: Path,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    sample_sizes: Iterable[int],
    taus: Iterable[float],
    times: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[list[dict], dict[float, list[np.ndarray]]]:
    kernels: dict[float, list[np.ndarray]] = {}
    for tau in taus:
        kernels[float(tau)] = [psd_wigner_deformation(eta, float(tau), rng) for _ in range(repetitions)]

    rows: list[dict] = []
    for sample_size in sample_sizes:
        base_total, _, _ = aligned_risk_curve(eta, sample_size, beta, times)
        base_time, base_risk = optimal_point(times, base_total)
        rows.append(
            {
                "sample_size": sample_size,
                "tau": 0.0,
                "repetitions": 0,
                "optimal_time_mean": base_time,
                "optimal_time_std": 0.0,
                "minimum_risk_mean": base_risk,
                "minimum_risk_std": 0.0,
                "kramp_time": kramp_stopping_time(alpha, beta),
            }
        )
        for tau in taus:
            optimum_times = []
            minimum_risks = []
            for kernel in kernels[float(tau)]:
                total, _, _, _, _ = kernel_risk_curve(kernel, eta, sample_size, beta, times)
                optimum_time, minimum_risk = optimal_point(times, total)
                optimum_times.append(optimum_time)
                minimum_risks.append(minimum_risk)
            rows.append(
                {
                    "sample_size": sample_size,
                    "tau": float(tau),
                    "repetitions": repetitions,
                    "optimal_time_mean": float(np.mean(optimum_times)),
                    "optimal_time_std": float(np.std(optimum_times, ddof=1)),
                    "minimum_risk_mean": float(np.mean(minimum_risks)),
                    "minimum_risk_std": float(np.std(minimum_risks, ddof=1)),
                    "kramp_time": kramp_stopping_time(alpha, beta),
                }
            )
    write_csv(outdir / "wigner_scaling_and_stopping.csv", rows)

    fig, ax = plt.subplots()
    sample_array = np.array(sorted(set(int(value) for value in sample_sizes)), dtype=float)
    for tau in [0.0] + [float(value) for value in taus]:
        subset = sorted((row for row in rows if row["tau"] == tau), key=lambda row: row["sample_size"])
        values = np.array([row["minimum_risk_mean"] for row in subset])
        label = "aligned kernel" if tau == 0.0 else fr"$\tau={tau:g}$"
        ax.loglog(sample_array, values, marker="o", label=label)
    reference_exponent = -alpha / (1.0 + alpha)
    reference = np.array([row["minimum_risk_mean"] for row in rows if row["tau"] == 0.0])
    anchor = reference[len(reference) // 2]
    reference_line = anchor * (sample_array / sample_array[len(sample_array) // 2]) ** reference_exponent
    ax.loglog(
        sample_array,
        reference_line,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=fr"$P^{{{reference_exponent:.2f}}}$",
    )
    ax.set_xlabel(r"training samples $P$")
    ax.set_ylabel(r"minimum test risk $\mathcal{L}(t^*)$")
    ax.set_title("PSD Wigner deformation bends the data scaling law")
    ax.legend(frameon=False, ncol=2)
    savefig(fig, outdir, "wigner_minimum_risk_data_scaling")

    fig, ax = plt.subplots()
    reference_time = kramp_stopping_time(alpha, beta)
    for tau in [0.0] + [float(value) for value in taus]:
        subset = sorted((row for row in rows if row["tau"] == tau), key=lambda row: row["sample_size"])
        values = np.array([row["optimal_time_mean"] for row in subset])
        label = "aligned kernel" if tau == 0.0 else fr"$\tau={tau:g}$"
        ax.semilogx(sample_array, values / reference_time, marker="o", label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label=r"$t_0^*$")
    ax.set_xlabel(r"training samples $P$")
    ax.set_ylabel(r"$t^*/t_0^*$")
    ax.set_title("PSD Wigner deformation shifts early stopping")
    ax.legend(frameon=False, ncol=2)
    savefig(fig, outdir, "wigner_optimal_stopping_data_scaling")

    return rows, kernels


def run_block_haar_data_scaling(
    outdir: Path,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    sample_sizes: Iterable[int],
    perturbation_scales: Iterable[float],
    times: np.ndarray,
) -> list[dict]:
    """Exact scaling curves for a stable head and Haar-mixed tail.

    The mixing boundary is set by the power-law adjacent-gap estimate
    J_mix ~ s^{-1/(2+alpha)}.  No Monte Carlo is used: every risk value is
    the orthogonal-Weingarten expectation.
    """
    rows: list[dict] = []
    sample_values = sorted(set(int(value) for value in sample_sizes))
    scale_values = [float(value) for value in perturbation_scales]
    conditions: list[tuple[float, int]] = [(0.0, eta.size)]
    conditions.extend(
        (
            scale,
            asymptotic_mixing_index(alpha, eta[0], scale, eta.size),
        )
        for scale in scale_values
    )

    for sample_size in sample_values:
        for scale, aligned_head in conditions:
            if aligned_head >= eta.size:
                curve, _, _ = aligned_risk_curve(eta, sample_size, beta, times)
                variances = np.zeros_like(curve)
            else:
                moments = [
                    block_haar_risk_moments(
                        eta,
                        eta,
                        aligned_head,
                        sample_size,
                        beta,
                        float(time),
                    )
                    for time in times
                ]
                curve = np.array([moment[0] for moment in moments])
                variances = np.array([moment[1] for moment in moments])
            optimum_time, minimum_risk = optimal_point(times, curve)
            optimum_index = int(np.nanargmin(curve))
            rows.append(
                {
                    "sample_size": sample_size,
                    "perturbation_scale": scale,
                    "aligned_head": aligned_head,
                    "haar_tail": eta.size - aligned_head,
                    "optimal_time": optimum_time,
                    "minimum_risk_mean": minimum_risk,
                    "minimum_risk_std": math.sqrt(max(float(variances[optimum_index]), 0.0)),
                    "kramp_time": kramp_stopping_time(alpha, beta),
                }
            )

    write_csv(outdir / "powerlaw_scaling_and_stopping.csv", rows)

    fig, ax = plt.subplots()
    sample_array = np.array(sample_values, dtype=float)
    for scale, aligned_head in conditions:
        subset = sorted(
            (row for row in rows if row["perturbation_scale"] == scale),
            key=lambda row: row["sample_size"],
        )
        values = np.array([row["minimum_risk_mean"] for row in subset])
        if scale == 0.0:
            label = "aligned kernel"
        else:
            label = fr"$s={scale:g}$, $J_{{\rm mix}}={aligned_head}$"
        ax.loglog(sample_array, values, marker="o", label=label)
    reference_exponent = -alpha / (1.0 + alpha)
    base_rows = sorted(
        (row for row in rows if row["perturbation_scale"] == 0.0),
        key=lambda row: row["sample_size"],
    )
    base_values = np.array([row["minimum_risk_mean"] for row in base_rows])
    anchor_index = len(sample_array) // 2
    reference_line = base_values[anchor_index] * (
        sample_array / sample_array[anchor_index]
    ) ** reference_exponent
    ax.loglog(
        sample_array,
        reference_line,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=fr"$P^{{{reference_exponent:.2f}}}$",
    )
    ax.set_xlabel(r"training samples $P$")
    ax.set_ylabel(r"minimum test risk $\mathcal{L}(t^*)$")
    ax.set_title("Haar tail mixing bends the power-law frontier")
    ax.legend(frameon=False, ncol=2)
    savefig(fig, outdir, "minimum_risk_data_scaling")

    fig, ax = plt.subplots()
    reference_time = kramp_stopping_time(alpha, beta)
    for scale, aligned_head in conditions:
        subset = sorted(
            (row for row in rows if row["perturbation_scale"] == scale),
            key=lambda row: row["sample_size"],
        )
        values = np.array([row["optimal_time"] for row in subset])
        if scale == 0.0:
            label = "aligned kernel"
        else:
            label = fr"$s={scale:g}$, $J_{{\rm mix}}={aligned_head}$"
        ax.semilogx(sample_array, values / reference_time, marker="o", label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label=r"$t_0^*$")
    ax.set_xlabel(r"training samples $P$")
    ax.set_ylabel(r"$t^*/t_0^*$")
    ax.set_title("Haar tail mixing advances early stopping")
    ax.legend(frameon=False, ncol=2)
    savefig(fig, outdir, "optimal_stopping_data_scaling")
    return rows


def run_risk_curves(
    outdir: Path,
    eta: np.ndarray,
    alpha: float,
    sample_size: int,
    beta: float,
    taus: Iterable[float],
    times: np.ndarray,
    kernels: dict[float, list[np.ndarray]],
) -> None:
    base_total, base_bias, base_variance = aligned_risk_curve(eta, sample_size, beta, times)
    fig, ax = plt.subplots()
    ax.loglog(times, base_total, color="black", linewidth=2.2, label="aligned total")
    ax.loglog(times, base_bias, color="black", linestyle="--", alpha=0.55, label="aligned bias")
    ax.loglog(times, base_variance, color="black", linestyle=":", alpha=0.75, label="aligned variance")
    colors = ["#0077BB", "#EE7733", "#009988", "#CC3311"]
    for color, tau in zip(colors, taus):
        curves = []
        for kernel in kernels[float(tau)]:
            total, _, _, _, _ = kernel_risk_curve(kernel, eta, sample_size, beta, times)
            curves.append(total)
        stack = np.stack(curves)
        mean = np.mean(stack, axis=0)
        sem = np.std(stack, axis=0, ddof=1) / math.sqrt(stack.shape[0])
        ax.loglog(times, mean, color=color, linewidth=1.8, label=fr"$\tau={float(tau):g}$")
        ax.fill_between(times, np.maximum(mean - 2.0 * sem, 1e-16), mean + 2.0 * sem, color=color, alpha=0.13)
    ax.axvline(
        kramp_stopping_time(alpha, beta),
        color="gray",
        linewidth=1.0,
        linestyle="-.",
    )
    ax.set_xlabel(r"training time $t$")
    ax.set_ylabel(r"test risk $\mathcal{L}(t)$")
    ax.set_title(fr"Power-law dynamics with finite-NTK mixing, $P={sample_size}$")
    ax.legend(frameon=False, ncol=2)
    savefig(fig, outdir, "powerlaw_risk_curves")


def run_spectral_mixing_plot(
    outdir: Path,
    eta: np.ndarray,
    alpha: float,
    tau: float,
    kernel: np.ndarray,
) -> dict:
    _, _, _, eigenvalues, eigenvectors = kernel_risk_curve(
        kernel, eta, sample_size=1.0, beta=1.0, times=np.array([1.0])
    )
    perturbation_norm = float(np.linalg.norm(kernel - np.diag(eta), ord=2))
    exact_index = gap_mixing_index(eta, perturbation_norm)
    asymptotic_index = asymptotic_mixing_index(alpha, eta[0], perturbation_norm, eta.size)
    diagonal_overlap = np.diag(eigenvectors) ** 2
    indices = np.arange(1, eta.size + 1)

    fig, ax = plt.subplots()
    ax.semilogy(indices, np.maximum(diagonal_overlap, 1e-8), color="#0077BB", linewidth=1.2)
    ax.axvline(exact_index, color="#CC3311", linestyle="--", linewidth=1.5, label=fr"gap bound $J={exact_index}$")
    ax.axvline(asymptotic_index, color="#EE7733", linestyle=":", linewidth=1.8, label=fr"power-law estimate $J={asymptotic_index}$")
    ax.axhline(1.0 / eta.size, color="black", linestyle="-.", linewidth=1.1, label="Haar scale")
    ax.set_xlabel("population mode index")
    ax.set_ylabel(r"matched overlap $|\langle e_i,\hat e_i\rangle|^2$")
    ax.set_title(fr"Head stability and Wigner-like tail mixing, $\tau={tau:g}$")
    ax.legend(frameon=False)
    savefig(fig, outdir, "spectral_eigenvector_mixing")

    row = {
        "tau": tau,
        "perturbation_operator_norm": perturbation_norm,
        "gap_mixing_index": exact_index,
        "asymptotic_mixing_index": asymptotic_index,
        "minimum_kernel_eigenvalue": float(np.min(eigenvalues)),
        "maximum_kernel_eigenvalue": float(np.max(eigenvalues)),
    }
    write_csv(outdir / "spectral_mixing_diagnostic.csv", [row])
    return row


def run_rank_depth_phase(
    outdir: Path,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    sample_size: int,
    width: int,
    ranks: list[int],
    depths: list[int],
    ntk_strength: float,
    times: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> list[dict]:
    base_total, _, _ = aligned_risk_curve(eta, sample_size, beta, times)
    base_time, base_risk = optimal_point(times, base_total)
    rows: list[dict] = []
    time_ratio = np.empty((len(depths), len(ranks)))
    risk_ratio = np.empty_like(time_ratio)

    for depth_index, depth in enumerate(depths):
        for rank_index, rank in enumerate(ranks):
            epsilon = epsilon_width_rank(width, rank)
            fluctuation_scale = depth ** 1.5 * math.sqrt(eta.size * epsilon)
            tau = ntk_strength * fluctuation_scale
            optimum_times = []
            minimum_risks = []
            operator_norms = []
            mixing_indices = []
            for _ in range(repetitions):
                kernel = psd_wigner_deformation(eta, tau, rng)
                total, _, _, _, _ = kernel_risk_curve(kernel, eta, sample_size, beta, times)
                optimum_time, minimum_risk = optimal_point(times, total)
                perturbation_norm = float(np.linalg.norm(kernel - np.diag(eta), ord=2))
                optimum_times.append(optimum_time)
                minimum_risks.append(minimum_risk)
                operator_norms.append(perturbation_norm)
                mixing_indices.append(gap_mixing_index(eta, perturbation_norm))
            mean_time = float(np.mean(optimum_times))
            mean_risk = float(np.mean(minimum_risks))
            mean_operator = float(np.mean(operator_norms))
            mean_mixing_index = float(np.mean(mixing_indices))
            mixing_time = mean_mixing_index ** (1.0 + alpha) / sample_size
            row = {
                "width": width,
                "rank": rank,
                "depth": depth,
                "epsilon": epsilon,
                "theory_fluctuation_scale": fluctuation_scale,
                "deformation_tau": tau,
                "mean_operator_deviation": mean_operator,
                "mean_gap_mixing_index": mean_mixing_index,
                "predicted_mixing_time": mixing_time,
                "optimal_time_mean": mean_time,
                "optimal_time_std": float(np.std(optimum_times, ddof=1)),
                "optimal_time_over_aligned": mean_time / base_time,
                "minimum_risk_mean": mean_risk,
                "minimum_risk_std": float(np.std(minimum_risks, ddof=1)),
                "minimum_risk_over_aligned": mean_risk / base_risk,
                "kramp_time": kramp_stopping_time(alpha, beta),
            }
            rows.append(row)
            time_ratio[depth_index, rank_index] = row["optimal_time_over_aligned"]
            risk_ratio[depth_index, rank_index] = row["minimum_risk_over_aligned"]

    write_csv(outdir / "rank_depth_early_stopping_phase.csv", rows)

    def heatmap(values: np.ndarray, name: str, label: str, title: str, center_one: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        if center_one:
            limit = max(float(np.max(np.abs(values - 1.0))), 0.05)
            image = ax.imshow(values, origin="lower", aspect="auto", cmap="coolwarm", vmin=1.0 - limit, vmax=1.0 + limit)
        else:
            image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(ranks)), [str(rank) for rank in ranks])
        ax.set_yticks(np.arange(len(depths)), [str(depth) for depth in depths])
        ax.set_xlabel(r"rank $r$")
        ax.set_ylabel(r"depth $L$")
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                color = "white" if values[i, j] > np.median(values) else "black"
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(label)
        savefig(fig, outdir, name)

    heatmap(
        time_ratio,
        "rank_depth_optimal_time",
        r"$t^*/t^*_{\rm aligned}$",
        "Rank and depth shift the early-stopping time",
        center_one=True,
    )
    heatmap(
        risk_ratio,
        "rank_depth_minimum_risk",
        r"$\mathcal{L}^*/\mathcal{L}^*_{\rm aligned}$",
        "Rank and depth bend the scaling-law floor",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="data/feynman/powerlaw_early_stopping")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--num-modes", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--weingarten-repetitions", type=int, default=500)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--ntk-strength", type=float, default=0.02)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.alpha <= 0.0 or args.beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if args.num_modes < 8:
        raise ValueError("num-modes must be at least 8")

    if args.quick:
        args.num_modes = min(args.num_modes, 64)
        args.repetitions = min(args.repetitions, 4)
        args.weingarten_repetitions = min(args.weingarten_repetitions, 80)

    set_paper_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    eta = powerlaw_spectrum(args.num_modes, args.alpha)
    stopping_time = kramp_stopping_time(args.alpha, args.beta)
    times = np.geomspace(max(stopping_time / 100.0, 1e-5), stopping_time * 30.0, 180)

    taus = [0.0025, 0.0075, 0.02]
    sample_sizes = [16, 32, 64, 128, 256]
    if args.quick:
        taus = [0.005, 0.02]
        sample_sizes = [16, 64, 256]

    weingarten_row = run_weingarten_check(
        outdir=outdir,
        eta=eta,
        sample_size=args.sample_size,
        beta=args.beta,
        time=stopping_time,
        aligned_head=max(2, args.num_modes // 4),
        repetitions=args.weingarten_repetitions,
        rng=rng,
    )
    scaling_rows, kernels = run_deformation_scaling(
        outdir=outdir,
        eta=eta,
        alpha=args.alpha,
        beta=args.beta,
        sample_sizes=sample_sizes,
        taus=taus,
        times=times,
        repetitions=args.repetitions,
        rng=rng,
    )
    run_risk_curves(
        outdir=outdir,
        eta=eta,
        alpha=args.alpha,
        sample_size=args.sample_size,
        beta=args.beta,
        taus=taus,
        times=times,
        kernels=kernels,
    )

    scaling_num_modes = 8192 if not args.quick else 1024
    scaling_eta = powerlaw_spectrum(scaling_num_modes, args.alpha)
    block_sample_sizes = [32, 64, 128, 256, 512, 1024]
    if args.quick:
        block_sample_sizes = [32, 128, 512]
    block_times = np.geomspace(max(stopping_time / 100.0, 1e-5), stopping_time * 12.0, 140)
    block_rows = run_block_haar_data_scaling(
        outdir=outdir,
        eta=scaling_eta,
        alpha=args.alpha,
        beta=args.beta,
        sample_sizes=block_sample_sizes,
        perturbation_scales=taus,
        times=block_times,
    )
    spectral_row = run_spectral_mixing_plot(
        outdir=outdir,
        eta=eta,
        alpha=args.alpha,
        tau=float(taus[-1]),
        kernel=kernels[float(taus[-1])][0],
    )

    ranks = [args.width // 16, args.width // 8, args.width // 4, args.width // 2, args.width]
    depths = [1, 2, 3, 4, 6]
    if args.quick:
        ranks = [args.width // 8, args.width // 2, args.width]
        depths = [1, 3, 6]
    phase_rows = run_rank_depth_phase(
        outdir=outdir,
        eta=eta,
        alpha=args.alpha,
        beta=args.beta,
        sample_size=args.sample_size,
        width=args.width,
        ranks=ranks,
        depths=depths,
        ntk_strength=args.ntk_strength,
        times=times,
        repetitions=args.repetitions,
        rng=rng,
    )

    summary = {
        "seed": args.seed,
        "num_modes": args.num_modes,
        "alpha": args.alpha,
        "beta": args.beta,
        "sample_size": args.sample_size,
        "kramp_stopping_time": stopping_time,
        "repetitions": args.repetitions,
        "weingarten_repetitions": args.weingarten_repetitions,
        "width": args.width,
        "ntk_strength": args.ntk_strength,
        "taus": taus,
        "sample_sizes": sample_sizes,
        "weingarten": weingarten_row,
        "spectral_mixing": spectral_row,
        "num_scaling_rows": len(scaling_rows),
        "num_block_haar_scaling_rows": len(block_rows),
        "num_phase_rows": len(phase_rows),
        "outputs": sorted(path.name for path in outdir.iterdir()),
    }
    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
