"""Exact neural-quadratic-form NTK spectra and finite-rank ignition clocks.

This module implements the ZGZ neural quadratic form

    f_mu(W) = Tr(A_mu W W^T)

under squared-loss gradient flow.  It is deliberately separated from the
local Taylor theorem: every identity below is exact *for the quadratic
model*.  Applying it to a nonlinear MMNN additionally requires control of
the Taylor remainder (or defining a deep NQF as the model itself).

In the commuting, orthogonal-feature sector the complete sample NTK is

    Theta(t) = 4 Lambda diag(z(t)) Lambda^T,

so its nonzero eigenvalues and the logistic ignition of every mode are
available in closed form.  Gaussian and Haar--Stiefel initializations give
exact chi-square and beta laws for the small-initialization clock.  These are
the radial/Wishart and orientation/Weingarten sectors of the MMNN Gram
calculus, respectively.
"""

from __future__ import annotations

import argparse
import csv
from itertools import permutations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.special import digamma, polygamma


def nqf_outputs(structure: np.ndarray, order_parameter: np.ndarray) -> np.ndarray:
    """Return f_mu=Tr(A_mu M) for symmetric A_mu and M."""

    return np.einsum("mij,ji->m", structure, order_parameter)


def nqf_ntk(structure: np.ndarray, order_parameter: np.ndarray) -> np.ndarray:
    """Exact empirical NTK of the pure ZGZ NQF.

    The manifestly symmetric implementation is equivalent to
    4 Tr(A_mu A_nu M); symmetry follows because A_mu, A_nu and M are
    symmetric.
    """

    left = np.einsum("mij,jk->mik", structure, order_parameter)
    kernel = 4.0 * np.einsum("mik,nki->mn", left, structure)
    return 0.5 * (kernel + kernel.T)


def explicit_parameter_ntk(structure: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Parameter-gradient Gram matrix, used as an independent certificate."""

    gradients = 2.0 * np.einsum("mij,jd->mid", structure, weights)
    flat = gradients.reshape(gradients.shape[0], -1)
    return flat @ flat.T


def gaussian_trace_cumulant(
    test_matrices: list[np.ndarray], *, epsilon: float, component_count: int
) -> float:
    """Exact joint cumulant of Tr(C_j M) for a scaled Gaussian Wishart M."""

    if not test_matrices:
        raise ValueError("at least one test matrix is required")
    if component_count < 1 or epsilon < 0:
        raise ValueError("component_count must be positive and epsilon nonnegative")
    symmetric = [0.5 * (matrix + matrix.T) for matrix in test_matrices]
    order = len(symmetric)
    if order == 1:
        return float(epsilon * np.trace(symmetric[0]))
    trace_sum = 0.0
    for tail in permutations(range(1, order)):
        product = symmetric[0]
        for index in tail:
            product = product @ symmetric[index]
        trace_sum += float(np.trace(product))
    return (
        2.0 ** (order - 1)
        * epsilon**order
        * trace_sum
        / component_count ** (order - 1)
    )


def order_parameter_rhs(
    structure: np.ndarray,
    targets: np.ndarray,
    order_parameter: np.ndarray,
) -> np.ndarray:
    """Exact M-dot for mean squared loss m^{-1} sum_mu(f_mu-y_mu)^2."""

    residual = nqf_outputs(structure, order_parameter) - targets
    hessian = 4.0 * np.einsum("m,mij->ij", residual, structure) / len(targets)
    return -(hessian @ order_parameter + order_parameter @ hessian)


def orthogonal_logistic(
    rates: np.ndarray,
    competition: np.ndarray,
    initial_modes: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Closed solution z_k(t) in the orthogonal-feature sector."""

    rates = np.asarray(rates, dtype=float)
    competition = np.asarray(competition, dtype=float)
    initial_modes = np.asarray(initial_modes, dtype=float)
    times = np.asarray(times, dtype=float)
    if np.any(competition <= 0) or np.any(initial_modes < 0):
        raise ValueError("competition must be positive and initial modes nonnegative")

    z = np.empty((times.size, rates.size), dtype=float)
    nonzero = np.abs(rates) > 1e-14
    denominator = (
        competition[None, nonzero] * initial_modes[None, nonzero]
        + (
            rates[None, nonzero]
            - competition[None, nonzero] * initial_modes[None, nonzero]
        )
        * np.exp(-times[:, None] * rates[None, nonzero])
    )
    z[:, nonzero] = (
        rates[None, nonzero] * initial_modes[None, nonzero] / denominator
    )
    z[:, ~nonzero] = initial_modes[None, ~nonzero] / (
        1.0
        + times[:, None]
        * competition[None, ~nonzero]
        * initial_modes[None, ~nonzero]
    )
    return z


def orthogonal_ntk(
    feature_eigenvalues: np.ndarray, modes: np.ndarray
) -> np.ndarray:
    """Theta=4 Lambda diag(z) Lambda^T for one or many mode vectors."""

    feature_eigenvalues = np.asarray(feature_eigenvalues, dtype=float)
    modes = np.asarray(modes, dtype=float)
    if modes.ndim == 1:
        return 4.0 * (feature_eigenvalues * modes[None, :]) @ feature_eigenvalues.T
    return 4.0 * np.einsum(
        "mk,tk,nk->tmn", feature_eigenvalues, modes, feature_eigenvalues
    )


def orthogonal_ntk_eigenvalues(
    feature_eigenvalues: np.ndarray, modes: np.ndarray
) -> np.ndarray:
    """Labeled nonzero eigenvalues when Lambda's columns are orthogonal."""

    column_norm_squared = np.sum(np.asarray(feature_eigenvalues) ** 2, axis=0)
    return 4.0 * np.asarray(modes) * column_norm_squared


def isotropic_solution(
    target_matrix: np.ndarray,
    initial_order_parameter: np.ndarray,
    isotropy: float,
    time: float,
) -> np.ndarray:
    """Exact Riccati resummation M(t) in the isotropic NQF sector."""

    if isotropy <= 0 or time < 0:
        raise ValueError("isotropy must be positive and time nonnegative")
    eigenvalues, eigenvectors = np.linalg.eigh(target_matrix)
    scaled = 2.0 * eigenvalues * time
    phi_eigenvalues = np.empty_like(eigenvalues)
    nonzero = np.abs(eigenvalues) > 1e-13
    phi_eigenvalues[nonzero] = np.expm1(scaled[nonzero]) / (
        2.0 * eigenvalues[nonzero]
    )
    phi_eigenvalues[~nonzero] = time
    phi = (eigenvectors * phi_eigenvalues) @ eigenvectors.T
    propagator = expm(target_matrix * time)
    inverse_factor = np.linalg.solve(
        np.eye(target_matrix.shape[0])
        + 8.0 * isotropy * phi @ initial_order_parameter,
        propagator,
    )
    result = propagator @ initial_order_parameter @ inverse_factor
    return 0.5 * (result + result.T)


def isotropic_rhs(
    target_matrix: np.ndarray, order_parameter: np.ndarray, isotropy: float
) -> np.ndarray:
    return (
        target_matrix @ order_parameter
        + order_parameter @ target_matrix
        - 8.0 * isotropy * order_parameter @ order_parameter
    )


def half_ignition_time(
    rates: np.ndarray, competition: np.ndarray, initial_modes: np.ndarray
) -> np.ndarray:
    """Time at which z_k reaches half of its positive fixed point."""

    rates = np.asarray(rates, dtype=float)
    competition = np.asarray(competition, dtype=float)
    initial_modes = np.asarray(initial_modes, dtype=float)
    if np.any(rates <= 0):
        raise ValueError("half-ignition requires positive rates")
    ratio = competition * initial_modes / rates
    if np.any((ratio <= 0) | (ratio >= 0.5)):
        raise ValueError("initial modes must lie below half their fixed points")
    return np.log((1.0 - ratio) / ratio) / rates


def gaussian_log_seed_moments(component_count: int) -> tuple[float, float]:
    """Mean and variance of log(z/epsilon), z=epsilon*chi2_d/d."""

    if component_count < 1:
        raise ValueError("component_count must be positive")
    d = float(component_count)
    mean = float(digamma(d / 2.0) + np.log(2.0) - np.log(d))
    variance = float(polygamma(1, d / 2.0))
    return mean, variance


def stiefel_log_seed_moments(
    ambient_dimension: int, component_count: int
) -> tuple[float, float]:
    """Moments for z=epsilon*(p/d)*Beta(d/2,(p-d)/2).

    At full rank d=p the projector is the identity and the clock disorder
    vanishes pathwise.
    """

    p = int(ambient_dimension)
    d = int(component_count)
    if not 1 <= d <= p:
        raise ValueError("require 1 <= component_count <= ambient_dimension")
    if d == p:
        return 0.0, 0.0
    mean = float(np.log(p / d) + digamma(d / 2.0) - digamma(p / 2.0))
    variance = float(polygamma(1, d / 2.0) - polygamma(1, p / 2.0))
    return mean, variance


def asymptotic_clock_moments(
    rate: float,
    competition: float,
    epsilon: float,
    log_seed_mean: float,
    log_seed_variance: float,
) -> tuple[float, float]:
    """Small-epsilon half-ignition mean and variance."""

    mean = (np.log(rate / (competition * epsilon)) - log_seed_mean) / rate
    variance = log_seed_variance / rate**2
    return float(mean), float(variance)


def sample_initial_modes(
    ensemble: str,
    *,
    epsilon: float,
    component_count: int,
    samples: int,
    rng: np.random.Generator,
    ambient_dimension: int | None = None,
) -> np.ndarray:
    """Sample the exact diagonal Wishart or Haar-projector seed law."""

    if ensemble == "gaussian":
        return epsilon * rng.chisquare(component_count, size=samples) / component_count
    if ensemble != "stiefel":
        raise ValueError("ensemble must be 'gaussian' or 'stiefel'")
    if ambient_dimension is None:
        raise ValueError("ambient_dimension is required for Stiefel seeds")
    if component_count == ambient_dimension:
        return np.full(samples, epsilon)
    beta = rng.beta(
        component_count / 2.0,
        (ambient_dimension - component_count) / 2.0,
        size=samples,
    )
    return epsilon * ambient_dimension * beta / component_count


def run_experiment(
    output_dir: Path,
    *,
    ambient_dimension: int = 64,
    epsilon: float = 1e-7,
    samples: int = 120_000,
    seed: int = 17,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    mode_count = 6
    rates = np.geomspace(1.0, 0.16, mode_count)
    competition = np.ones(mode_count)
    initial = epsilon * np.ones(mode_count)
    times = np.linspace(0.0, 125.0, 700)
    modes = orthogonal_logistic(rates, competition, initial, times)
    feature_eigenvalues = np.eye(mode_count)
    labeled_eigenvalues = orthogonal_ntk_eigenvalues(feature_eigenvalues, modes)

    numerical_eigenvalues = np.array(
        [np.linalg.eigvalsh(kernel)[::-1] for kernel in orthogonal_ntk(feature_eigenvalues, modes)]
    )
    sorted_labeled = np.sort(labeled_eigenvalues, axis=1)[:, ::-1]
    spectrum_error = float(np.max(np.abs(numerical_eigenvalues - sorted_labeled)))

    with (output_dir / "nqf_dynamic_ntk_modes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "mode", "rate", "z", "ntk_eigenvalue"])
        for time_index, time in enumerate(times):
            for mode in range(mode_count):
                writer.writerow(
                    [
                        float(time),
                        mode,
                        float(rates[mode]),
                        float(modes[time_index, mode]),
                        float(labeled_eigenvalues[time_index, mode]),
                    ]
                )

    component_counts = np.array([2, 4, 8, 16, 32, ambient_dimension])
    clock_rows: list[dict[str, float | int | str]] = []
    max_mean_zscore = 0.0
    max_std_relative_error = 0.0
    rate = 0.55
    coefficient = 1.0
    for ensemble in ("gaussian", "stiefel"):
        for component_count in component_counts:
            seeds = sample_initial_modes(
                ensemble,
                epsilon=epsilon,
                component_count=int(component_count),
                ambient_dimension=ambient_dimension,
                samples=samples,
                rng=rng,
            )
            clocks = half_ignition_time(
                np.full(samples, rate),
                np.full(samples, coefficient),
                seeds,
            )
            if ensemble == "gaussian":
                log_mean, log_variance = gaussian_log_seed_moments(
                    int(component_count)
                )
            else:
                log_mean, log_variance = stiefel_log_seed_moments(
                    ambient_dimension, int(component_count)
                )
            theory_mean, theory_variance = asymptotic_clock_moments(
                rate, coefficient, epsilon, log_mean, log_variance
            )
            empirical_mean = float(np.mean(clocks))
            empirical_std = float(np.std(clocks, ddof=1))
            theory_std = float(np.sqrt(theory_variance))
            standard_error = empirical_std / np.sqrt(samples)
            if theory_std > 1e-14 and standard_error > 0:
                max_mean_zscore = max(
                    max_mean_zscore,
                    abs(empirical_mean - theory_mean) / standard_error,
                )
            if theory_std > 1e-14:
                max_std_relative_error = max(
                    max_std_relative_error,
                    abs(empirical_std - theory_std) / theory_std,
                )
            clock_rows.append(
                {
                    "ensemble": ensemble,
                    "ambient_dimension": ambient_dimension,
                    "component_count": int(component_count),
                    "empirical_mean": empirical_mean,
                    "theory_mean_small_epsilon": theory_mean,
                    "empirical_std": empirical_std,
                    "theory_std_small_epsilon": theory_std,
                }
            )

    with (output_dir / "finite_rank_ignition_clocks.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clock_rows[0]))
        writer.writeheader()
        writer.writerows(clock_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.25), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, mode_count))
    fixed_points = rates / competition
    for mode, color in enumerate(colors):
        axes[0].plot(
            times,
            labeled_eigenvalues[:, mode] / (4.0 * fixed_points[mode]),
            color=color,
            linewidth=2.2,
            label=rf"$r_{{{mode + 1}}}={rates[mode]:.2f}$",
        )
    axes[0].set(xlabel="gradient-flow time", ylabel="NTK eigenvalue / plateau")
    axes[0].set_title("Exact spectral mode ignition", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    markers = {"gaussian": "o", "stiefel": "s"}
    palette = {"gaussian": "#D1495B", "stiefel": "#276FBF"}
    for ensemble in ("gaussian", "stiefel"):
        selected = [row for row in clock_rows if row["ensemble"] == ensemble]
        counts = np.array([row["component_count"] for row in selected])
        empirical_mean = np.array([row["empirical_mean"] for row in selected])
        theory_mean = np.array([row["theory_mean_small_epsilon"] for row in selected])
        empirical_std = np.array([row["empirical_std"] for row in selected])
        theory_std = np.array([row["theory_std_small_epsilon"] for row in selected])
        baseline = np.log(rate / (coefficient * epsilon)) / rate
        axes[1].plot(
            counts,
            theory_mean - baseline,
            color=palette[ensemble],
            linewidth=2.2,
            label=f"{ensemble}: exact log law",
        )
        axes[1].scatter(
            counts,
            empirical_mean - baseline,
            color=palette[ensemble],
            marker=markers[ensemble],
            s=38,
            zorder=3,
        )
        axes[2].plot(
            counts,
            theory_std,
            color=palette[ensemble],
            linewidth=2.2,
            label=f"{ensemble}: exact log law",
        )
        axes[2].scatter(
            counts,
            empirical_std,
            color=palette[ensemble],
            marker=markers[ensemble],
            s=38,
            zorder=3,
        )

    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axes[1].set_xscale("log", base=2)
    axes[1].set(
        xlabel="bottleneck / component count $d$",
        ylabel="mean clock shift",
    )
    axes[1].set_title("Wishart vs. Weingarten clock", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("symlog", linthresh=1e-3, linscale=0.8)
    axes[2].set(
        xlabel="bottleneck / component count $d$",
        ylabel="ignition-time standard deviation",
    )
    axes[2].set_title("Full-rank Stiefel cancellation", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].annotate(
        "exactly zero",
        xy=(ambient_dimension, 0.0),
        xytext=(ambient_dimension / 2.6, 0.018),
        arrowprops={"arrowstyle": "->", "color": palette["stiefel"]},
        color=palette["stiefel"],
        fontsize=8,
    )
    figure.savefig(output_dir / "nqf_mmnn_mode_ignition.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / "nqf_mmnn_mode_ignition.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)

    summary = {
        "max_exact_ntk_spectrum_error": spectrum_error,
        "max_monte_carlo_mean_zscore": float(max_mean_zscore),
        "max_monte_carlo_std_relative_error": float(max_std_relative_error),
        "ambient_dimension": ambient_dimension,
        "epsilon": epsilon,
        "samples": samples,
        "interpretation": (
            "Exact within the commuting orthogonal-feature NQF; local when "
            "used as the quadratic normal form of a nonlinear MMNN."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/nqf_mmnn_mode_ignition"),
    )
    parser.add_argument("--ambient-dimension", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    summary = run_experiment(
        args.output_dir,
        ambient_dimension=args.ambient_dimension,
        epsilon=args.epsilon,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
