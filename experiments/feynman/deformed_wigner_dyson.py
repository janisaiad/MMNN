#!/usr/bin/env python3
"""Dyson law, moments, and overlaps for a Wigner-like NTK correction.

For a deterministic kernel spectrum ``diag(eta)`` and a leading Gaussian
finite-width correction

    H = diag(eta) + sigma W,      E W_ij^2 = 1/m  (i != j),

the large-m diagonal resolvent is determined by the scalar matrix Dyson
equation

    m(z) = m^{-1} sum_i [eta_i - z - sigma^2 m(z)]^{-1}.

The same denominators give the basis-resolved spectral measures and hence
the mean eigenvector overlaps.  Spectral moments are computed independently
from free cumulants: the correction adds only ``sigma^2`` to the second free
cumulant.  This is the appropriate resummation after the Feynman tensor
calculus has supplied ``sigma^2``; Weingarten calculus alone determines Haar
eigenvector moments but does not determine this deformed spectrum.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@lru_cache(maxsize=None)
def noncrossing_partitions(n: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate noncrossing set partitions of ``range(n)`` exactly."""
    if n == 0:
        return ((),)

    def set_partitions(items: tuple[int, ...]):
        if not items:
            return {()}
        first, rest = items[0], items[1:]
        result = set()
        for partition in set_partitions(rest):
            result.add(tuple(sorted(((first,), *partition))))
            for block_index in range(len(partition)):
                blocks = list(partition)
                blocks[block_index] = tuple(sorted((first, *blocks[block_index])))
                result.add(tuple(sorted(blocks)))
        return result

    def is_noncrossing(partition: tuple[tuple[int, ...], ...]) -> bool:
        block_of = {}
        for block_index, block in enumerate(partition):
            for item in block:
                block_of[item] = block_index
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if (
                            block_of[a] == block_of[c]
                            and block_of[b] == block_of[d]
                            and block_of[a] != block_of[b]
                        ):
                            return False
        return True

    return tuple(sorted(p for p in set_partitions(tuple(range(n))) if is_noncrossing(p)))


def moments_to_free_cumulants(moments: np.ndarray) -> np.ndarray:
    """Möbius inversion of the noncrossing moment-cumulant relation."""
    moments = np.asarray(moments, dtype=float)
    order = len(moments) - 1
    cumulants = np.zeros(order + 1, dtype=float)
    for n in range(1, order + 1):
        total_without_single_block = 0.0
        for partition in noncrossing_partitions(n):
            if len(partition) == 1:
                continue
            term = 1.0
            for block in partition:
                term *= cumulants[len(block)]
            total_without_single_block += term
        cumulants[n] = moments[n] - total_without_single_block
    return cumulants


def free_cumulants_to_moments(cumulants: np.ndarray) -> np.ndarray:
    cumulants = np.asarray(cumulants, dtype=float)
    order = len(cumulants) - 1
    moments = np.ones(order + 1, dtype=float)
    for n in range(1, order + 1):
        total = 0.0
        for partition in noncrossing_partitions(n):
            term = 1.0
            for block in partition:
                term *= cumulants[len(block)]
            total += term
        moments[n] = total
    return moments


def deformed_wigner_moments(
    eigenvalues: np.ndarray, sigma: float, max_order: int
) -> np.ndarray:
    """Limiting trace moments using free additive convolution."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    signal_moments = np.asarray(
        [1.0] + [float(np.mean(eigenvalues**k)) for k in range(1, max_order + 1)]
    )
    cumulants = moments_to_free_cumulants(signal_moments)
    if max_order >= 2:
        cumulants[2] += sigma**2
    return free_cumulants_to_moments(cumulants)


def solve_dyson(
    eigenvalues: np.ndarray,
    spectral_grid: np.ndarray,
    sigma: float,
    imaginary_part: float,
    *,
    tolerance: float = 2e-13,
    max_iterations: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the scalar Dyson equation and return m(z), G_ii(z)."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    spectral_grid = np.asarray(spectral_grid, dtype=float)
    z_values = spectral_grid + 1j * imaginary_part
    m_values = np.empty_like(z_values)
    diagonal_resolvent = np.empty((len(spectral_grid), len(eigenvalues)), complex)
    previous = None
    for index, z in enumerate(z_values):
        m = previous if previous is not None else np.mean(1.0 / (eigenvalues - z))
        for _ in range(max_iterations):
            update = np.mean(1.0 / (eigenvalues - z - sigma**2 * m))
            candidate = 0.45 * m + 0.55 * update
            if abs(candidate - m) <= tolerance * max(1.0, abs(candidate)):
                m = candidate
                break
            m = candidate
        else:
            raise RuntimeError(f"Dyson iteration failed at z={z}")
        previous = m
        m_values[index] = m
        diagonal_resolvent[index] = 1.0 / (
            eigenvalues - z - sigma**2 * m
        )
    return m_values, diagonal_resolvent


def dyson_density_and_overlaps(
    m_values: np.ndarray, diagonal_resolvent: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global density, local measures, and conditional overlaps."""
    density = np.imag(m_values) / np.pi
    local_density = np.imag(diagonal_resolvent) / np.pi
    denominator = diagonal_resolvent.shape[1] * density[:, None]
    overlaps = np.divide(
        local_density,
        denominator,
        out=np.zeros_like(local_density),
        where=denominator > 1e-15,
    )
    return density, local_density, overlaps


def sample_goe(size: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(size, size))
    return (raw + raw.T) / np.sqrt(2.0 * size)


def monte_carlo(
    eigenvalues: np.ndarray,
    sigma: float,
    samples: int,
    max_moment: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_eigenvalues = []
    moment_samples = []
    for _ in range(samples):
        matrix = np.diag(eigenvalues) + sigma * sample_goe(len(eigenvalues), rng)
        values = np.linalg.eigvalsh(matrix)
        all_eigenvalues.append(values)
        moment_samples.append([np.mean(values**k) for k in range(1, max_moment + 1)])
    return (
        np.concatenate(all_eigenvalues),
        np.asarray(moment_samples),
    )


def save_experiment(
    output_dir: Path,
    *,
    size: int,
    alpha: float,
    sigma: float,
    samples: int,
    seed: int,
    max_moment: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = np.arange(1, size + 1)
    eigenvalues = indices ** (-(1.0 + alpha))
    edge = max(abs(eigenvalues.min()), abs(eigenvalues.max())) + 2.3 * sigma
    grid = np.linspace(-2.3 * sigma, edge, 1000)
    imaginary_part = max(2.0e-3, 0.018 * sigma)
    m_values, diagonal_resolvent = solve_dyson(
        eigenvalues, grid, sigma, imaginary_part
    )
    density, local_density, overlaps = dyson_density_and_overlaps(
        m_values, diagonal_resolvent
    )
    theoretical_moments = deformed_wigner_moments(
        eigenvalues, sigma, max_moment
    )
    sampled_eigenvalues, moment_samples = monte_carlo(
        eigenvalues, sigma, samples, max_moment, seed
    )

    rows = []
    for order in range(1, max_moment + 1):
        rows.append(
            {
                "order": order,
                "dyson_free_moment": theoretical_moments[order],
                "monte_carlo_mean": float(moment_samples[:, order - 1].mean()),
                "monte_carlo_standard_error": float(
                    moment_samples[:, order - 1].std(ddof=1) / np.sqrt(samples)
                ),
            }
        )
    with (output_dir / "spectral_moments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "dyson_solution.npz",
        signal_eigenvalues=eigenvalues,
        spectral_grid=grid,
        stieltjes=m_values,
        density=density,
        local_density=local_density,
        conditional_overlaps=overlaps,
    )

    selected = [0, min(4, size - 1), min(19, size - 1), min(99, size - 1)]
    selected = list(dict.fromkeys(selected))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.75), constrained_layout=True)
    axes[0].hist(
        sampled_eigenvalues,
        bins=110,
        density=True,
        color="#A7C7E7",
        edgecolor="none",
        label="GOE simulations",
    )
    axes[0].plot(grid, density, color="#174A7E", linewidth=2.0, label="Dyson equation")
    axes[0].set_xlabel("NTK eigenvalue")
    axes[0].set_ylabel("density")
    axes[0].set_title("deformed-Wigner spectrum")
    axes[0].legend(frameon=False)

    colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(selected)))
    for color, signal_index in zip(colors, selected):
        axes[1].plot(
            grid,
            local_density[:, signal_index],
            color=color,
            linewidth=1.8,
            label=rf"population mode {signal_index + 1}",
        )
    axes[1].set_xlabel("NTK eigenvalue")
    axes[1].set_ylabel(r"local spectral density $\rho_i(\lambda)$")
    axes[1].set_title("eigenvector mixing, mode by mode")
    axes[1].legend(frameon=False, fontsize=8)

    orders = np.arange(1, max_moment + 1)
    empirical = moment_samples.mean(axis=0)
    errors = moment_samples.std(axis=0, ddof=1) / np.sqrt(samples)
    axes[2].errorbar(
        orders - 0.08,
        empirical,
        yerr=errors,
        fmt="o",
        color="#C44E52",
        capsize=2,
        label="GOE simulations",
    )
    axes[2].scatter(
        orders + 0.08,
        theoretical_moments[1:],
        marker="D",
        facecolor="white",
        edgecolor="#222222",
        label="free-cumulant moments",
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks(orders)
    axes[2].set_xlabel("moment order $k$")
    axes[2].set_ylabel(r"$m^{-1}\operatorname{Tr}K^k$")
    axes[2].set_title("independent moment check")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output_dir / "deformed_wigner_dyson.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "deformed_wigner_dyson.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.035)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--max-moment", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/deformed_wigner_dyson"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_experiment(
        args.output_dir,
        size=args.size,
        alpha=args.alpha,
        sigma=args.sigma,
        samples=args.samples,
        seed=args.seed,
        max_moment=args.max_moment,
    )
    print(f"wrote Dyson-law outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
