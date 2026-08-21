"""Exact three-token equilibria mixing the extreme eigendirections."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from .simulator import normalize, state_tangent_jacobian, vector_field


def mixed_balanced_equation(
    q: ArrayLike,
    center_eigenvalue: float,
    polar_eigenvalue: float,
    beta: float,
    n_center: int = 1,
    n_each_polar: int = 1,
):
    """Equilibrium equation for a center plus two balanced polar groups.

    The configuration contains ``n_center`` copies of ``u`` and
    ``n_each_polar`` copies of each of ``q*u ± sqrt(1-q²)*v``, where ``u`` and
    ``v`` are eigenvectors with the two supplied eigenvalues.
    """
    if n_center < 1 or n_each_polar < 1:
        raise ValueError("both group sizes must be positive")
    q = np.asarray(q)
    q2 = q * q
    self_score = polar_eigenvalue + (
        center_eigenvalue - polar_eigenvalue
    ) * q2
    opposite_score = -polar_eigenvalue + (
        center_eigenvalue + polar_eigenvalue
    ) * q2
    center_score = center_eigenvalue * q
    return (
        q
        * (
            (center_eigenvalue - polar_eigenvalue)
            * np.exp(beta * self_score)
            + (center_eigenvalue + polar_eigenvalue)
            * np.exp(beta * opposite_score)
        )
        + center_eigenvalue
        * (n_center / n_each_polar)
        * np.exp(beta * center_score)
    )


def mixed_three_equation(q: ArrayLike, positive: float, magnitude: float, beta: float):
    """Scalar equilibrium equation for ``(u, q u ± sqrt(1-q²) v)``.

    Here ``V u = positive*u`` and ``V v = -magnitude*v``.
    """
    return mixed_balanced_equation(
        q,
        center_eigenvalue=positive,
        polar_eigenvalue=-magnitude,
        beta=beta,
    )


def mixed_balanced_roots(
    center_eigenvalue: float,
    polar_eigenvalue: float,
    beta: float,
    n_center: int = 1,
    n_each_polar: int = 1,
    *,
    grid_size: int = 2001,
) -> list[float]:
    """All sign-certified interior roots of the balanced mixed equation."""
    grid = np.linspace(-1.0 + 1e-8, 1.0 - 1e-8, grid_size)
    values = mixed_balanced_equation(
        grid,
        center_eigenvalue,
        polar_eigenvalue,
        beta,
        n_center,
        n_each_polar,
    )
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:], strict=True
    ):
        if f_left * f_right < 0:
            root = brentq(
                mixed_balanced_equation,
                left,
                right,
                args=(
                    center_eigenvalue,
                    polar_eigenvalue,
                    beta,
                    n_center,
                    n_each_polar,
                ),
                xtol=1e-14,
            )
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(float(root))
    return roots


def mixed_three_roots(
    positive: float,
    magnitude: float,
    beta: float,
    *,
    grid_size: int = 2001,
) -> list[float]:
    """All interior roots found by a sign-certified one-dimensional scan."""
    if positive <= 0 or magnitude <= 0 or beta < 0:
        raise ValueError("positive, magnitude must be positive and beta nonnegative")
    return mixed_balanced_roots(
        positive,
        -magnitude,
        beta,
        grid_size=grid_size,
    )


def mixed_balanced_state(
    q: float,
    n_center: int = 1,
    n_each_polar: int = 1,
    dimension: int = 2,
) -> np.ndarray:
    """Construct the center/upper/lower configuration in its eigenplane."""
    if dimension < 2 or abs(q) >= 1:
        raise ValueError("dimension >= 2 and |q| < 1 are required")
    if n_center < 1 or n_each_polar < 1:
        raise ValueError("both group sizes must be positive")
    s = np.sqrt(1.0 - q * q)
    center = np.zeros((n_center, dimension))
    upper = np.zeros((n_each_polar, dimension))
    lower = np.zeros((n_each_polar, dimension))
    center[:, 0] = 1.0
    upper[:, [0, -1]] = (q, s)
    lower[:, [0, -1]] = (q, -s)
    return normalize(np.concatenate((center, upper, lower), axis=0))


def embed_mixed_balanced_state(
    q: float,
    dimension: int,
    center_mode: int,
    polar_mode: int,
    n_center: int = 1,
    n_each_polar: int = 1,
) -> np.ndarray:
    """Embed the balanced mixed family in two chosen spectral coordinates."""
    if center_mode == polar_mode or not (
        0 <= center_mode < dimension and 0 <= polar_mode < dimension
    ):
        raise ValueError("center_mode and polar_mode must be distinct valid indices")
    planar = mixed_balanced_state(q, n_center, n_each_polar)
    state = np.zeros((planar.shape[0], dimension))
    state[:, center_mode] = planar[:, 0]
    state[:, polar_mode] = planar[:, 1]
    return state


def mixed_three_state(q: float, dimension: int = 2) -> np.ndarray:
    """State ordered as center, upper polar token, lower polar token."""
    return mixed_balanced_state(q, dimension=dimension)


def root_diagnostics(
    q: float, positive: float, magnitude: float, beta: float
) -> dict[str, float | bool]:
    eigenvalues = np.array([positive, -magnitude])
    state = mixed_three_state(q)
    jacobian = state_tangent_jacobian(state, eigenvalues, beta)
    rates = np.linalg.eigvals(jacobian).real
    residual = np.linalg.norm(vector_field(state, eigenvalues, beta), axis=1).max()
    return {
        "q": q,
        "max_linear_rate": float(np.max(rates)),
        "min_linear_rate": float(np.min(rates)),
        "linearly_stable": bool(np.max(rates) < -1e-7),
        "residual": float(residual),
    }


def generate_phase_atlas(output_dir: Path) -> pd.DataFrame:
    """Map existence and full in-plane stability in dimensionless parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = np.linspace(0.5, 4.0, 71)
    scaled_betas = np.linspace(0.05, 6.0, 120)
    rows = []
    stable_q = np.full((scaled_betas.size, ratios.size), np.nan)
    root_count = np.zeros_like(stable_q)
    for beta_index, scaled_beta in enumerate(scaled_betas):
        for ratio_index, ratio in enumerate(ratios):
            roots = mixed_three_roots(1.0, ratio, scaled_beta, grid_size=801)
            root_count[beta_index, ratio_index] = len(roots)
            for root_index, q in enumerate(roots):
                diag = root_diagnostics(q, 1.0, ratio, scaled_beta)
                rows.append(
                    {
                        "magnitude_over_positive": ratio,
                        "beta_times_positive": scaled_beta,
                        "root_index": root_index,
                        **diag,
                    }
                )
                if diag["linearly_stable"]:
                    stable_q[beta_index, ratio_index] = q

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "mixed_three_phase.csv", index=False)

    extent = [ratios[0], ratios[-1], scaled_betas[0], scaled_betas[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    image0 = axes[0].imshow(
        root_count,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=max(2, np.nanmax(root_count)),
    )
    axes[0].set_title("Number of mixed roots")
    axes[0].set_xlabel(r"$|\lambda_-|/\lambda_+$")
    axes[0].set_ylabel(r"$\beta\lambda_+$")
    fig.colorbar(image0, ax=axes[0])

    image1 = axes[1].imshow(
        stable_q,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[1].set_title("Stable root q (blank: none)")
    axes[1].set_xlabel(r"$|\lambda_-|/\lambda_+$")
    axes[1].set_ylabel(r"$\beta\lambda_+$")
    fig.colorbar(image1, ax=axes[1])
    fig.savefig(output_dir / "mixed_three_phase.png", dpi=180)
    plt.close(fig)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spectral_self_attention/theory"),
    )
    args = parser.parse_args()
    frame = generate_phase_atlas(args.output_dir)
    print(
        {
            "roots": len(frame),
            "stable_roots": int(frame.linearly_stable.sum()),
            "max_residual": float(frame.residual.max()),
        }
    )


if __name__ == "__main__":
    main()
