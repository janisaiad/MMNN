"""Mean-field, kernel, normalization, training, and oscillation experiments.

The experiments in this module extend the symmetric spherical attention model in
five controlled directions:

1. Monte Carlo convergence of the finite attention sum to its continuum integral;
2. persistence and stability of the three-atom mixed polygon for other kernels;
3. row-normalized versus unnormalized attention on the sphere;
4. a trained terminal control acting on the polygonal attractor; and
5. a two-head rotating cluster once score/output alignment is broken.

The code is deliberately low dimensional.  Its purpose is to isolate mechanisms,
not to imitate the full architecture of a trained Transformer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import exp1


FloatArray = NDArray[np.float64]


MATRIX = np.diag([2.0, -3.0])
SHARPNESS = 1.5
MIXED_EXP_ROOT = 0.023905191282460915


def kernel_values(name: str, scores: FloatArray) -> FloatArray:
    """Positive kernels normalized to equal one at score zero."""
    if name == "exponential":
        return np.exp(SHARPNESS * scores)
    if name == "sigmoid":
        return 2.0 / (1.0 + np.exp(-np.clip(SHARPNESS * scores, -700.0, 700.0)))
    if name == "softplus":
        return np.logaddexp(0.0, SHARPNESS * scores) / np.log(2.0)
    if name == "polynomial4":
        # Scores lie in [-3, 3] on the sphere for MATRIX.
        return np.maximum(1.0 + scores / 3.01, 1e-12) ** 4
    raise ValueError(f"unknown kernel: {name}")


def points_from_angles(angles: FloatArray) -> FloatArray:
    return np.column_stack([np.cos(angles), np.sin(angles)])


def angular_tangents(angles: FloatArray) -> FloatArray:
    return np.column_stack([-np.sin(angles), np.cos(angles)])


def atomic_angle_field(
    angles: FloatArray,
    masses: FloatArray,
    kernel: str,
    *,
    row_normalized: bool,
) -> FloatArray:
    """Mean-field velocity restricted to a finite atomic measure on S1."""
    points = points_from_angles(angles)
    transformed = points @ MATRIX
    scores = points @ MATRIX @ points.T
    weights = kernel_values(kernel, scores) * masses[None, :]
    force = weights @ transformed
    if row_normalized:
        force /= np.sum(weights, axis=1, keepdims=True)
    return np.sum(force * angular_tangents(angles), axis=1)


def mixed_equation(kernel: str, q: FloatArray | float) -> FloatArray:
    """Three-atom existence equation with equal masses."""
    a, b = 2.0, -3.0
    q = np.asarray(q)
    score_same = b + (a - b) * q * q
    score_mirror = -b + (a + b) * q * q
    score_central = a * q
    return q * (
        (a - b) * kernel_values(kernel, score_same)
        + (a + b) * kernel_values(kernel, score_mirror)
    ) + a * kernel_values(kernel, score_central)


def roots_for_kernel(kernel: str) -> list[float]:
    grid = np.linspace(-0.999, 0.999, 10_000)
    values = mixed_equation(kernel, grid)
    roots: list[float] = []
    for index in range(len(grid) - 1):
        if values[index] * values[index + 1] >= 0.0:
            continue
        left, right = float(grid[index]), float(grid[index + 1])
        for _ in range(60):
            middle = 0.5 * (left + right)
            if mixed_equation(kernel, left) * mixed_equation(kernel, middle) <= 0.0:
                right = middle
            else:
                left = middle
        roots.append(0.5 * (left + right))
    return roots


def atomic_jacobian(
    q: float, kernel: str, *, row_normalized: bool, epsilon: float = 1e-6
) -> FloatArray:
    angles = np.array([0.0, np.arccos(q), -np.arccos(q)])
    masses = np.ones(3) / 3.0
    jacobian = np.empty((3, 3))
    for column in range(3):
        perturbation = np.zeros(3)
        perturbation[column] = epsilon
        plus = atomic_angle_field(
            angles + perturbation,
            masses,
            kernel,
            row_normalized=row_normalized,
        )
        minus = atomic_angle_field(
            angles - perturbation,
            masses,
            kernel,
            row_normalized=row_normalized,
        )
        jacobian[:, column] = (plus - minus) / (2.0 * epsilon)
    return jacobian


def polygon_kernel_rows() -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for kernel in ["exponential", "sigmoid", "softplus", "polynomial4"]:
        for root_index, q in enumerate(roots_for_kernel(kernel), start=1):
            for normalized in [True, False]:
                eigenvalues = np.linalg.eigvals(
                    atomic_jacobian(q, kernel, row_normalized=normalized)
                )
                max_real = float(np.max(eigenvalues.real))
                rows.append(
                    {
                        "kernel": kernel,
                        "root_index": root_index,
                        "q": q,
                        "row_normalized": normalized,
                        "max_jacobian_real_part": max_real,
                        "stable": max_real < -1e-5,
                        "min_jacobian_real_part": float(np.min(eigenvalues.real)),
                    }
                )
    return rows


def field_from_support(
    probe_angles: FloatArray,
    support_angles: FloatArray,
    support_weights: FloatArray,
) -> FloatArray:
    probes = points_from_angles(probe_angles)
    support = points_from_angles(support_angles)
    scores = probes @ MATRIX @ support.T
    weights = np.exp(SHARPNESS * scores) * support_weights[None, :]
    force = weights @ (support @ MATRIX)
    force /= np.sum(weights, axis=1, keepdims=True)
    return np.sum(force * angular_tangents(probe_angles), axis=1)


def continuum_convergence_rows(
    seed: int,
) -> tuple[list[dict[str, float | int]], float]:
    """Audit the n^-1/2 replacement of a sum by an integral."""
    rng = np.random.default_rng(seed)
    q = MIXED_EXP_ROOT
    centres = np.array([0.0, np.arccos(q), -np.arccos(q)])
    concentration = 35.0
    grid = np.linspace(-np.pi, np.pi, 8192, endpoint=False)
    density = np.mean(
        np.exp(concentration * np.cos(grid[:, None] - centres[None, :])), axis=1
    ) / (2.0 * np.pi * np.i0(concentration))
    quadrature_weights = density * (2.0 * np.pi / len(grid))
    quadrature_weights /= np.sum(quadrature_weights)
    probes = np.linspace(-np.pi, np.pi, 96, endpoint=False)
    reference = field_from_support(probes, grid, quadrature_weights)

    rows: list[dict[str, float | int]] = []
    sample_sizes = [16, 32, 64, 128, 256, 512, 1024]
    trials = 64
    for size in sample_sizes:
        errors = []
        for _ in range(trials):
            labels = rng.integers(0, len(centres), size=size)
            samples = rng.vonmises(centres[labels], concentration)
            estimate = field_from_support(probes, samples, np.ones(size) / size)
            errors.append(float(np.sqrt(np.mean((estimate - reference) ** 2))))
        error_array = np.asarray(errors)
        rows.append(
            {
                "n_tokens": size,
                "trials": trials,
                "velocity_rmse": float(np.mean(error_array)),
                "velocity_rmse_se": float(np.std(error_array, ddof=1) / np.sqrt(trials)),
            }
        )
    slope = float(
        np.polyfit(
            np.log([row["n_tokens"] for row in rows]),
            np.log([row["velocity_rmse"] for row in rows]),
            1,
        )[0]
    )
    return rows, slope


def rk4_angles(
    angles: FloatArray,
    field,
    *,
    t_final: float,
    dt: float,
    save_every: int,
) -> tuple[FloatArray, FloatArray]:
    current = np.asarray(angles, dtype=float).copy()
    steps = int(np.ceil(t_final / dt))
    actual_dt = t_final / steps
    times = [0.0]
    states = [current.copy()]
    for step in range(1, steps + 1):
        k1 = field(current)
        k2 = field(current + 0.5 * actual_dt * k1)
        k3 = field(current + 0.5 * actual_dt * k2)
        k4 = field(current + actual_dt * k3)
        current += actual_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if step % save_every == 0 or step == steps:
            times.append(step * actual_dt)
            states.append(current.copy())
    return np.asarray(times), np.asarray(states)


def oscillation_rows(seed: int) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Two heads: one clusters, the other rotates the attended value."""
    rng = np.random.default_rng(seed + 1)
    initial = rng.vonmises(0.0, 1.0, size=64)
    beta = 1.0
    omega = 0.8
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])

    def field(angles: FloatArray) -> FloatArray:
        points = points_from_angles(angles)
        logits = beta * (points @ points.T)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        attended = weights @ points
        # Head 1 has output I; head 2 has output omega * J.
        force = attended + omega * (attended @ rotation.T)
        return np.sum(force * angular_tangents(angles), axis=1)

    times, states = rk4_angles(initial, field, t_final=30.0, dt=0.02, save_every=10)
    complex_means = np.mean(np.exp(1j * states), axis=1)
    order = np.abs(complex_means)
    phase = np.unwrap(np.angle(complex_means))
    tail = max(8, len(times) // 3)
    angular_velocity = float(np.polyfit(times[-tail:], phase[-tail:], 1)[0])
    rows = [
        {
            "time": float(time),
            "synchronization": float(sync),
            "unwrapped_mean_angle": float(angle),
        }
        for time, sync, angle in zip(times, order, phase, strict=True)
    ]
    summary = {
        "omega": omega,
        "initial_synchronization": float(order[0]),
        "final_synchronization": float(order[-1]),
        "measured_angular_velocity": angular_velocity,
        "final_max_instantaneous_speed": float(np.max(np.abs(field(states[-1])))),
    }
    return rows, summary


def train_polygon_turnpike() -> tuple[list[dict[str, float]], dict[str, float]]:
    """Train a small Fourier FFN control against the stable mixed polygon."""
    import torch

    torch.set_default_dtype(torch.float64)
    base = torch.tensor(
        [0.0, np.arccos(MIXED_EXP_ROOT), -np.arccos(MIXED_EXP_ROOT)]
    )
    initial = base + torch.tensor([0.08, -0.05, 0.04])
    target = base + 0.9
    matrix = torch.diag(torch.tensor([2.0, -3.0]))
    bins = 40
    substeps = 4
    horizon = 8.0
    dt = horizon / (bins * substeps)
    regularization = 0.03
    control = torch.nn.Parameter(torch.zeros(bins, 5))
    optimizer = torch.optim.Adam([control], lr=0.05)

    def attention(angles):
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        transformed = points @ matrix
        scores = points @ matrix @ points.T
        weights = torch.exp(SHARPNESS * scores) / 3.0
        force = weights @ transformed
        force = force / weights.sum(dim=1, keepdim=True)
        tangents = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=1)
        return (force * tangents).sum(dim=1)

    def basis(angles):
        return torch.stack(
            [
                torch.ones_like(angles),
                torch.cos(angles),
                torch.sin(angles),
                torch.cos(2.0 * angles),
                torch.sin(2.0 * angles),
            ],
            dim=1,
        )

    for _ in range(300):
        angles = initial
        for bin_index in range(bins):
            for _ in range(substeps):
                angles = angles + dt * (
                    attention(angles) + basis(angles) @ control[bin_index]
                )
        terminal_loss = (1.0 - torch.cos(angles - target)).mean()
        penalty = (
            regularization
            * (horizon / bins)
            * torch.sum(control * control)
            / 2.0
        )
        objective = terminal_loss + penalty
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    angles = initial.detach()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for bin_index in range(bins):
            rows.append(
                {
                    "time": bin_index * horizon / bins,
                    "control_norm": float(torch.linalg.vector_norm(control[bin_index])),
                    "distance_from_polygon": float(
                        (1.0 - torch.cos(angles - base)).mean()
                    ),
                    "distance_from_target": float(
                        (1.0 - torch.cos(angles - target)).mean()
                    ),
                    "angle_0": float(angles[0]),
                    "angle_1": float(angles[1]),
                    "angle_2": float(angles[2]),
                }
            )
            for _ in range(substeps):
                angles = angles + dt * (
                    attention(angles) + basis(angles) @ control[bin_index]
                )
        final_target_loss = float((1.0 - torch.cos(angles - target)).mean())
        final_polygon_distance = float((1.0 - torch.cos(angles - base)).mean())
        control_norms = torch.linalg.vector_norm(control, dim=1)
    summary = {
        "horizon": horizon,
        "target_rotation_radians": 0.9,
        "regularization": regularization,
        "training_steps": 300,
        "final_target_loss": final_target_loss,
        "final_polygon_distance": final_polygon_distance,
        "early_control_max": float(torch.max(control_norms[:20])),
        "terminal_control_max": float(torch.max(control_norms[-5:])),
    }
    return rows, summary


def no_sphere_summary() -> dict[str, float]:
    """Exact one-token radial behavior along the positive eigenvector."""
    positive_eigenvalue = 2.0
    target_norm = 10.0
    initial_norm = 1.0
    unprojected_softmax_time = np.log(target_norm / initial_norm) / positive_eigenvalue
    initial_argument = SHARPNESS * positive_eigenvalue * initial_norm**2
    target_argument = SHARPNESS * positive_eigenvalue * target_norm**2
    unnormalized_time = float(
        (exp1(initial_argument) - exp1(target_argument))
        / (2.0 * positive_eigenvalue)
    )
    blowup_time = float(exp1(initial_argument) / (2.0 * positive_eigenvalue))
    return {
        "target_norm": target_norm,
        "projected_norm": 1.0,
        "unprojected_row_normalized_time_to_target": float(unprojected_softmax_time),
        "unprojected_unnormalized_time_to_target": unnormalized_time,
        "unprojected_unnormalized_finite_blowup_time": blowup_time,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    polygon_rows = polygon_kernel_rows()
    continuum_rows, continuum_slope = continuum_convergence_rows(seed)
    oscillation_data, oscillation_summary = oscillation_rows(seed)
    training_data, training_summary = train_polygon_turnpike()

    write_csv(output / "kernel_polygon_roots.csv", polygon_rows)
    write_csv(output / "continuum_convergence.csv", continuum_rows)
    write_csv(output / "oscillatory_multihead.csv", oscillation_data)
    write_csv(output / "trained_polygon_turnpike.csv", training_data)
    summary = {
        "seed": seed,
        "continuum_monte_carlo_loglog_slope": continuum_slope,
        "oscillation": oscillation_summary,
        "trained_polygon": training_summary,
        "without_sphere": no_sphere_summary(),
        "kernel_protocol": {
            "matrix": [2.0, -3.0],
            "exponential_sigmoid_softplus_sharpness": SHARPNESS,
            "polynomial": "(1 + score / 3.01)^4",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spectral_self_attention/mean_field_extensions"),
    )
    parser.add_argument("--seed", type=int, default=260507772)
    args = parser.parse_args()
    run(args.output, args.seed)


if __name__ == "__main__":
    main()
