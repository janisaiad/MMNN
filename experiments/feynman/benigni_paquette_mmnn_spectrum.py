"""Quadratic-scaling spectrum for concatenated Benigni--Paquette blocks.

The implemented law is the conditional homogeneous formula

    MP_{gamma_1 / L} boxtimes (L .)_# mu_BP

or, after normalizing the summed NTK by depth, simply

    MP_{gamma_1 / L} boxtimes mu_BP.

For the explicit case nu=delta_1 and a linear derivative feature, mu_BP is
the chi law in Corollary 2 of Benigni--Paquette (arXiv:2508.20036).  The
script also evaluates the exact Frechet derivative of the MP fixed point with
respect to a signed perturbation of the population law.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class DiscreteMeasure:
    locations: np.ndarray
    weights: np.ndarray

    def normalized(self) -> "DiscreteMeasure":
        total = float(np.sum(self.weights))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("measure must have positive finite mass")
        return DiscreteMeasure(self.locations.copy(), self.weights / total)

    @property
    def mean(self) -> float:
        return float(np.sum(self.locations * self.weights) / np.sum(self.weights))


def _histogram_measure(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    lower: float,
    upper: float,
    bins: int,
) -> DiscreteMeasure:
    edges = np.linspace(lower, upper, bins + 1)
    mass, _ = np.histogram(values, bins=edges, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = mass > 1e-15
    return DiscreteMeasure(centers[keep], mass[keep]).normalized()


def marchenko_pastur_measure(gamma: float, points: int = 1200) -> DiscreteMeasure:
    """Deterministic quadrature for the MP law of shape gamma.

    The convention has support (1 +/- sqrt(gamma))^2, mean one, and an atom
    1-1/gamma at zero when gamma > 1.
    """

    if gamma <= 0:
        raise ValueError("gamma must be positive")
    edge_lo = (1.0 - np.sqrt(gamma)) ** 2
    edge_hi = (1.0 + np.sqrt(gamma)) ** 2
    dx = (edge_hi - edge_lo) / points
    x = edge_lo + (np.arange(points) + 0.5) * dx
    density = np.sqrt(np.maximum((edge_hi - x) * (x - edge_lo), 0.0))
    density /= 2.0 * np.pi * gamma * np.maximum(x, 1e-15)
    continuous_mass = min(1.0, 1.0 / gamma)
    w = density * dx
    w *= continuous_mass / np.sum(w)
    if gamma > 1.0:
        x = np.concatenate(([0.0], x))
        w = np.concatenate(([1.0 - 1.0 / gamma], w))
    return DiscreteMeasure(x, w).normalized()


def bp_explicit_population(
    gamma_2: float,
    *,
    alpha: float = 1.0,
    beta_squared: float = 0.0,
    points: int = 700,
    bins: int = 1400,
) -> DiscreteMeasure:
    """Explicit BP population law for nu=delta_1 and 0 < gamma_2 <= 1.

    chi = (gamma_2/2) (MP * MP)
          + (1-gamma_2) MP
          + (gamma_2/2) delta_0,
    followed by t -> alpha^2 t + beta_squared.
    """

    if not 0 < gamma_2 <= 1:
        raise ValueError("this compact chi discretization assumes 0 < gamma_2 <= 1")
    mp = marchenko_pastur_measure(gamma_2, points=points)
    sums = (mp.locations[:, None] + mp.locations[None, :]).ravel()
    sum_weights = (mp.weights[:, None] * mp.weights[None, :]).ravel()
    upper = 2.0 * (1.0 + np.sqrt(gamma_2)) ** 2
    conv = _histogram_measure(
        sums,
        sum_weights,
        lower=0.0,
        upper=upper,
        bins=bins,
    )

    locations = np.concatenate((conv.locations, mp.locations, np.array([0.0])))
    weights = np.concatenate(
        (
            (gamma_2 / 2.0) * conv.weights,
            (1.0 - gamma_2) * mp.weights,
            np.array([gamma_2 / 2.0]),
        )
    )
    pushed = alpha**2 * locations + beta_squared
    return _histogram_measure(
        pushed,
        weights,
        lower=max(0.0, beta_squared),
        upper=alpha**2 * upper + beta_squared + 1e-12,
        bins=bins,
    )


def homogeneous_concatenated_population(
    one_block: DiscreteMeasure,
    depth: int,
    *,
    normalize_by_depth: bool = False,
) -> DiscreteMeasure:
    if depth < 1:
        raise ValueError("depth must be positive")
    scale = 1.0 if normalize_by_depth else float(depth)
    return DiscreteMeasure(scale * one_block.locations, one_block.weights.copy())


def mp_stieltjes(
    z: complex,
    population: DiscreteMeasure,
    gamma: float,
    *,
    initial: complex | None = None,
) -> complex:
    """Solve the Benigni--Paquette MP fixed-point equation."""

    t = population.locations
    w = population.weights

    def map_value(s: complex) -> complex:
        denominator = t * (1.0 - gamma * (1.0 + z * s)) - z
        return complex(np.sum(w / denominator))

    def residual(v: np.ndarray) -> np.ndarray:
        s = complex(v[0], v[1])
        value = s - map_value(s)
        return np.array([value.real, value.imag])

    guess = initial if initial is not None else -1.0 / z
    solved = root(residual, np.array([guess.real, guess.imag]), method="hybr")
    candidate = complex(solved.x[0], solved.x[1])
    if solved.success and candidate.imag > 0 and np.linalg.norm(residual(solved.x)) < 1e-8:
        return candidate

    s = guess
    for _ in range(5000):
        new_s = 0.35 * map_value(s) + 0.65 * s
        if abs(new_s - s) < 1e-12:
            return new_s
        s = new_s
    raise RuntimeError(f"MP fixed point did not converge at z={z}")


def mp_density(
    grid: np.ndarray,
    population: DiscreteMeasure,
    gamma: float,
    *,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    stieltjes = np.empty(grid.size, dtype=np.complex128)
    previous: complex | None = None
    for idx, x in enumerate(grid):
        previous = mp_stieltjes(complex(x, eta), population, gamma, initial=previous)
        stieltjes[idx] = previous
    return stieltjes.imag / np.pi, stieltjes


def mp_population_linear_response(
    z: complex,
    population: DiscreteMeasure,
    gamma: float,
    signed_perturbation: DiscreteMeasure,
    *,
    stieltjes: complex | None = None,
) -> complex:
    """Exact derivative of the MP map in a signed population direction."""

    s = stieltjes if stieltjes is not None else mp_stieltjes(z, population, gamma)
    a = 1.0 - gamma * (1.0 + z * s)
    denominator = population.locations * a - z
    stability = 1.0 - gamma * z * np.sum(
        population.weights * population.locations / denominator**2
    )
    perturbation_denominator = signed_perturbation.locations * a - z
    forcing = np.sum(signed_perturbation.weights / perturbation_denominator)
    return complex(forcing / stability)


def scale_mixture_perturbation(
    population: DiscreteMeasure, scale: float
) -> DiscreteMeasure:
    """Signed law (scale .)_# population - population."""

    return DiscreteMeasure(
        np.concatenate((scale * population.locations, population.locations)),
        np.concatenate((population.weights, -population.weights)),
    )


def mixed_population(
    population: DiscreteMeasure, scale: float, epsilon: float
) -> DiscreteMeasure:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    return DiscreteMeasure(
        np.concatenate((population.locations, scale * population.locations)),
        np.concatenate(
            ((1.0 - epsilon) * population.weights, epsilon * population.weights)
        ),
    ).normalized()


def run_experiment(
    output_dir: Path,
    *,
    gamma_1: float = 0.8,
    gamma_2: float = 0.5,
    eta: float = 0.035,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    one_block = bp_explicit_population(gamma_2)
    depths = (1, 2, 4, 8)
    grid = np.linspace(0.0, 7.0, 520)

    densities: dict[int, np.ndarray] = {}
    transforms: dict[int, np.ndarray] = {}
    for depth in depths:
        population = homogeneous_concatenated_population(
            one_block, depth, normalize_by_depth=True
        )
        density, transform = mp_density(
            grid, population, gamma_1 / depth, eta=eta
        )
        densities[depth] = density
        transforms[depth] = transform

    response_depth = 4
    response_gamma = gamma_1 / response_depth
    response_population = homogeneous_concatenated_population(
        one_block, response_depth, normalize_by_depth=True
    )
    scale = 1.35
    epsilon = 0.025
    perturbation = scale_mixture_perturbation(response_population, scale)
    perturbed_population = mixed_population(
        response_population, scale=scale, epsilon=epsilon
    )
    perturbed_density, _ = mp_density(
        grid, perturbed_population, response_gamma, eta=eta
    )
    analytic_response = np.array(
        [
            mp_population_linear_response(
                complex(x, eta),
                response_population,
                response_gamma,
                perturbation,
                stieltjes=transforms[response_depth][idx],
            )
            for idx, x in enumerate(grid)
        ]
    ).imag / np.pi
    finite_difference = (
        perturbed_density - densities[response_depth]
    ) / epsilon

    stability = np.empty_like(grid)
    for idx, x in enumerate(grid):
        z = complex(x, eta)
        s = transforms[response_depth][idx]
        a = 1.0 - response_gamma * (1.0 + z * s)
        denominator = response_population.locations * a - z
        stability[idx] = abs(
            1.0
            - response_gamma
            * z
            * np.sum(
                response_population.weights
                * response_population.locations
                / denominator**2
            )
        )

    with (output_dir / "benigni_paquette_mmnn_spectrum.csv").open(
        "w", newline=""
    ) as handle:
        fieldnames = [
            "lambda_over_depth",
            *[f"density_L{depth}" for depth in depths],
            "linear_response_L4",
            "finite_difference_L4",
            "stability_denominator_L4",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, x in enumerate(grid):
            row = {
                "lambda_over_depth": float(x),
                **{
                    f"density_L{depth}": float(densities[depth][idx])
                    for depth in depths
                },
                "linear_response_L4": float(analytic_response[idx]),
                "finite_difference_L4": float(finite_difference[idx]),
                "stability_denominator_L4": float(stability[idx]),
            }
            writer.writerow(row)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.45))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(depths)))
    for color, depth in zip(colors, depths, strict=True):
        axes[0].plot(
            grid,
            densities[depth],
            lw=2.0,
            color=color,
            label=rf"$L={depth}$",
        )
    axes[0].set_title("Concatenated BP--MMNN bulk")
    axes[0].set_xlabel(r"normalized eigenvalue $\lambda/L$")
    axes[0].set_ylabel(r"density")
    axes[0].legend(frameon=False)

    axes[1].plot(
        grid,
        analytic_response,
        color="#D1495B",
        lw=2.2,
        label="exact linear response",
    )
    axes[1].plot(
        grid,
        finite_difference,
        color="#276FBF",
        lw=1.6,
        ls="--",
        label=rf"finite difference, $\epsilon={epsilon}$",
    )
    axes[1].axhline(0.0, color="0.45", lw=0.8)
    axes[1].set_title(r"Population-law deviation, $L=4$")
    axes[1].set_xlabel(r"normalized eigenvalue $\lambda/L$")
    axes[1].set_ylabel(r"$\partial_\epsilon\rho$")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(grid, 1.0 / stability, color="#6A4C93", lw=2.0)
    axes[2].set_yscale("log")
    axes[2].set_title("MP stability factor")
    axes[2].set_xlabel(r"normalized eigenvalue $\lambda/L$")
    axes[2].set_ylabel(r"$|1-\gamma z I_1(z)|^{-1}$")

    for axis in axes:
        axis.grid(alpha=0.22, lw=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        rf"$\gamma_1={gamma_1}$, $\gamma_2={gamma_2}$, "
        r"$\nu=\delta_1$, linear derivative feature",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "benigni_paquette_mmnn_spectrum.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "benigni_paquette_mmnn_spectrum.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/benigni_paquette_mmnn_spectrum"),
    )
    parser.add_argument("--gamma-1", type=float, default=0.8)
    parser.add_argument("--gamma-2", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.035)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args.output_dir,
        gamma_1=args.gamma_1,
        gamma_2=args.gamma_2,
        eta=args.eta,
    )


if __name__ == "__main__":
    main()
