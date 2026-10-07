"""Multiscale settling-time audit for a three-token transformer ODE.

The experiment follows the type-2 candidate with source model index 1051 in
``continuous_ode_f2_chaos_antilock_matched.json``.  The first two token angles
form a two-dimensional initial-condition slice and the third angle is fixed.
All terms in the MLP vector field are multiplied by ``gamma``; the attention
field and the hidden affine features are unchanged.

The code distinguishes two questions that a coloured settling-time image can
otherwise conflate:

* finite-horizon endpoint clustering, which is retained only as an audit and is
  not interpreted as attractor identity; and
* the scalar settling time, defined on the sampled trajectory as one time step
  after the last observation at which ``max_i |d theta_i / dt|`` is at least a
  prescribed threshold.

The latter is accepted only if the trajectory stays below threshold for a
validation tail at the end of the integration.  A postprocessing audit compares
the finite-horizon endpoints with the numerically resolved synchronized saddle
and stable roots; it does not relabel endpoint clusters as attractors.  The
reported uncertainty fraction is the fraction of coordinate-neighbour pairs
separated by epsilon whose settling times differ by at least ``delta_time``.
It is a finite-scale diagnostic, not a dimension estimate or a proof of a
fractal boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TWO_PI = 2.0 * np.pi
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def wrap(angles: np.ndarray) -> np.ndarray:
    """Map angles to [-pi, pi)."""

    return (angles + np.pi) % TWO_PI - np.pi


def wrapped_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return wrap(left - right)


@dataclass(frozen=True)
class PlanarTransformerField:
    """Fixed coefficients of the continuous normalized-depth angular field."""

    score: np.ndarray
    value: np.ndarray
    beta: float
    depth_scale: float
    mlp_bias: np.ndarray
    linear: np.ndarray
    hidden: np.ndarray
    hidden_bias: np.ndarray
    output: np.ndarray
    gamma: float

    @classmethod
    def from_record(
        cls, record: dict[str, Any], gamma: float
    ) -> "PlanarTransformerField":
        model = record["model"]
        return cls(
            score=np.asarray(model["score"], dtype=np.float64),
            value=np.asarray(model["value"], dtype=np.float64),
            beta=float(model["beta"]),
            depth_scale=float(model["step_size"]),
            mlp_bias=np.asarray(model["mlp_bias"], dtype=np.float64),
            linear=np.asarray(model["linear"], dtype=np.float64),
            hidden=np.asarray(model["hidden"], dtype=np.float64),
            hidden_bias=np.asarray(model["hidden_bias"], dtype=np.float64),
            output=np.asarray(model["output"], dtype=np.float64),
            gamma=float(gamma),
        )

    def rhs(self, angles: np.ndarray) -> np.ndarray:
        """Evaluate the angular ODE on an array with final dimension 3."""

        tokens = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        score_left = tokens @ self.score
        scores = score_left @ np.swapaxes(tokens, -1, -2)
        logits = self.beta * scores
        logits -= np.max(logits, axis=-1, keepdims=True)
        weights = np.exp(np.clip(logits, -80.0, 0.0))
        weights /= np.sum(weights, axis=-1, keepdims=True)

        values = tokens @ self.value.T
        attention = weights @ values

        hidden_values = tokens @ self.hidden.T
        hidden_values += self.hidden_bias
        mlp = self.mlp_bias + tokens @ self.linear.T
        mlp += hidden_values**2 @ self.output.T

        tangent = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
        vector_field = attention + self.gamma * mlp
        return self.depth_scale * np.sum(tangent * vector_field, axis=-1)

    def rk4_step(self, angles: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(angles)
        k2 = self.rhs(angles + 0.5 * dt * k1)
        k3 = self.rhs(angles + 0.5 * dt * k2)
        k4 = self.rhs(angles + dt * k3)
        return wrap(angles + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    def rk4_step_reusing_rhs(
        self, angles: np.ndarray, k1: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one step and return the field reused by the next step."""

        k2 = self.rhs(angles + 0.5 * dt * k1)
        k3 = self.rhs(angles + 0.5 * dt * k2)
        k4 = self.rhs(angles + dt * k3)
        advanced = wrap(angles + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        return advanced, self.rhs(advanced)


@dataclass(frozen=True)
class Window:
    index: int
    center_theta1: float
    center_theta2: float
    width: float

    @property
    def magnification(self) -> float:
        return TWO_PI / self.width


@dataclass
class GridRun:
    name: str
    window: Window
    resolution: int
    dt: float
    horizon: float
    theta1: np.ndarray
    theta2: np.ndarray
    initial: np.ndarray
    final: np.ndarray
    final_speed: np.ndarray
    last_fast_time: np.ndarray
    settling_time: np.ndarray
    resolved: np.ndarray
    runtime_seconds: float
    labels: np.ndarray | None = None

    @property
    def spacing(self) -> float:
        return self.window.width / self.resolution


def load_record(path: Path, source_model_index: int, n_tokens: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    selected = [
        record
        for record in payload["records"]
        if int(record["identity"]["source_model_index"]) == source_model_index
        and int(record["identity"]["n_tokens"]) == n_tokens
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one source record, found {len(selected)}")
    return selected[0]


def initial_grid(
    window: Window, resolution: int, fixed_theta3: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = window.width / resolution
    offsets = -0.5 * window.width + (np.arange(resolution) + 0.5) * spacing
    theta1_axis = window.center_theta1 + offsets
    theta2_axis = window.center_theta2 + offsets
    theta1, theta2 = np.meshgrid(theta1_axis, theta2_axis, indexing="xy")
    angles = np.stack(
        (
            wrap(theta1.ravel()),
            wrap(theta2.ravel()),
            np.full(resolution * resolution, wrap(np.asarray(fixed_theta3))),
        ),
        axis=-1,
    )
    return theta1, theta2, angles


def integrate_grid(
    name: str,
    field: PlanarTransformerField,
    window: Window,
    resolution: int,
    fixed_theta3: float,
    dt: float,
    horizon: float,
    speed_tolerance: float,
    validation_tail: float,
) -> GridRun:
    theta1, theta2, initial = initial_grid(window, resolution, fixed_theta3)
    angles = initial.copy()
    last_fast = np.zeros(angles.shape[0], dtype=np.float64)
    steps = int(np.ceil(horizon / dt))
    actual_horizon = steps * dt
    start = time.perf_counter()
    velocity = field.rhs(angles)
    for step in range(steps):
        angles, velocity = field.rk4_step_reusing_rhs(angles, velocity, dt)
        speed = np.max(np.abs(velocity), axis=-1)
        last_fast[speed >= speed_tolerance] = (step + 1) * dt
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(speed)):
            raise FloatingPointError(f"non-finite state in {name} at step {step + 1}")
    runtime_seconds = time.perf_counter() - start
    final_speed = np.max(np.abs(field.rhs(angles)), axis=-1)
    resolved = (final_speed < speed_tolerance) & (
        actual_horizon - last_fast >= validation_tail
    )
    settling_time = np.where(resolved, last_fast + dt, np.nan)
    print(
        f"{name}: {int(np.sum(resolved))}/{len(resolved)} resolved; "
        f"runtime={runtime_seconds:.1f}s",
        flush=True,
    )
    return GridRun(
        name=name,
        window=window,
        resolution=resolution,
        dt=dt,
        horizon=actual_horizon,
        theta1=theta1,
        theta2=theta2,
        initial=initial,
        final=angles,
        final_speed=final_speed,
        last_fast_time=last_fast,
        settling_time=settling_time,
        resolved=resolved,
        runtime_seconds=runtime_seconds,
    )


def roughest_cell_center(run: GridRun) -> tuple[float, float, dict[str, float]]:
    """Select the interior cell with the largest four-corner time range."""

    values = run.settling_time.reshape(run.resolution, run.resolution)
    valid = run.resolved.reshape(run.resolution, run.resolution)
    cell_valid = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    stack = np.stack(
        (values[:-1, :-1], values[1:, :-1], values[:-1, 1:], values[1:, 1:])
    )
    cell_range = np.max(stack, axis=0) - np.min(stack, axis=0)
    cell_range[~cell_valid] = -np.inf
    if not np.any(np.isfinite(cell_range)):
        raise RuntimeError(f"no resolved interior cell in {run.name}")
    flat = int(np.argmax(cell_range))
    row, column = np.unravel_index(flat, cell_range.shape)
    center1 = 0.25 * (
        run.theta1[row, column]
        + run.theta1[row + 1, column]
        + run.theta1[row, column + 1]
        + run.theta1[row + 1, column + 1]
    )
    center2 = 0.25 * (
        run.theta2[row, column]
        + run.theta2[row + 1, column]
        + run.theta2[row, column + 1]
        + run.theta2[row + 1, column + 1]
    )
    return (
        float(center1),
        float(center2),
        {
            "row": int(row),
            "column": int(column),
            "corner_range": float(cell_range[row, column]),
        },
    )


def fit_attractor_prototypes(
    finals: np.ndarray, resolved: np.ndarray, tolerance: float
) -> np.ndarray:
    """Greedy deterministic clustering followed by circular-mean refinement."""

    points = wrap(finals[resolved])
    if points.size == 0:
        raise RuntimeError("cannot fit attractor identities without resolved points")
    order = np.lexsort(
        tuple(points[:, index] for index in reversed(range(points.shape[1])))
    )
    prototypes: list[np.ndarray] = []
    members: list[list[np.ndarray]] = []
    for point in points[order]:
        if prototypes:
            distance = np.asarray(
                [np.max(np.abs(wrapped_difference(point, p))) for p in prototypes]
            )
            nearest = int(np.argmin(distance))
        else:
            distance = np.asarray([np.inf])
            nearest = 0
        if not prototypes or distance[nearest] > tolerance:
            prototypes.append(point.copy())
            members.append([point])
        else:
            members[nearest].append(point)
    refined = []
    for cluster in members:
        values = np.asarray(cluster)
        refined.append(np.angle(np.mean(np.exp(1j * values), axis=0)))
    return np.asarray(refined)


def assign_attractor_labels(
    finals: np.ndarray,
    resolved: np.ndarray,
    prototypes: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(finals.shape[0], -1, dtype=np.int16)
    distances = np.full(finals.shape[0], np.nan)
    for prototype_index, prototype in enumerate(prototypes):
        distance = np.max(np.abs(wrapped_difference(finals, prototype)), axis=-1)
        take = resolved & ((labels < 0) | (distance < distances))
        labels[take] = prototype_index
        distances[take] = distance[take]
    unmatched = resolved & (distances > tolerance)
    labels[unmatched] = -2
    return labels, distances


def pair_uncertainty(
    run: GridRun,
    offsets: list[int],
    delta_times: list[float],
    periodic: bool,
) -> list[dict[str, Any]]:
    times = run.settling_time.reshape(run.resolution, run.resolution)
    valid = run.resolved.reshape(run.resolution, run.resolution)
    if run.labels is None:
        raise RuntimeError(
            "attractor labels must be assigned before uncertainty analysis"
        )
    labels = run.labels.reshape(run.resolution, run.resolution)
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        if offset <= 0 or offset >= run.resolution:
            continue
        if periodic:
            time_differences = []
            label_differences = []
            pair_validity = []
            for axis in (0, 1):
                shifted_times = np.roll(times, -offset, axis=axis)
                shifted_labels = np.roll(labels, -offset, axis=axis)
                shifted_valid = np.roll(valid, -offset, axis=axis)
                pair_validity.append(valid & shifted_valid)
                time_differences.append(np.abs(times - shifted_times))
                label_differences.append(labels != shifted_labels)
        else:
            pair_validity = [
                valid[:, :-offset] & valid[:, offset:],
                valid[:-offset, :] & valid[offset:, :],
            ]
            time_differences = [
                np.abs(times[:, :-offset] - times[:, offset:]),
                np.abs(times[:-offset, :] - times[offset:, :]),
            ]
            label_differences = [
                labels[:, :-offset] != labels[:, offset:],
                labels[:-offset, :] != labels[offset:, :],
            ]
        valid_count = int(sum(np.sum(mask) for mask in pair_validity))
        total_count = int(sum(mask.size for mask in pair_validity))
        if valid_count == 0:
            continue
        basin_uncertain = int(
            sum(
                np.sum(diff[mask])
                for diff, mask in zip(label_differences, pair_validity)
            )
        )
        for delta_time in delta_times:
            time_uncertain = int(
                sum(
                    np.sum(diff[mask] >= delta_time)
                    for diff, mask in zip(time_differences, pair_validity)
                )
            )
            rows.append(
                {
                    "run": run.name,
                    "window_index": run.window.index,
                    "magnification": run.window.magnification,
                    "resolution": run.resolution,
                    "dt": run.dt,
                    "horizon": run.horizon,
                    "offset_pixels": int(offset),
                    "epsilon": float(offset * run.spacing),
                    "delta_time": float(delta_time),
                    "valid_pairs": valid_count,
                    "total_pairs": total_count,
                    "resolved_pair_fraction": valid_count / total_count,
                    "time_uncertain_pairs": time_uncertain,
                    "time_uncertainty_fraction": time_uncertain / valid_count,
                    "basin_uncertain_pairs": basin_uncertain,
                    "basin_uncertainty_fraction": basin_uncertain / valid_count,
                    "periodic_pairing": bool(periodic),
                }
            )
    return rows


def power_law_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row for row in rows if 0.0 < float(row["time_uncertainty_fraction"]) < 1.0
    ]
    if len(usable) < 3:
        return {
            "status": "insufficient_nonzero_finite_scale_points",
            "points": len(usable),
            "alpha": None,
            "intercept": None,
            "r_squared": None,
            "standard_error": None,
        }
    x = np.log([float(row["epsilon"]) for row in usable])
    y = np.log([float(row["time_uncertainty_fraction"]) for row in usable])
    design = np.column_stack((x, np.ones_like(x)))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    alpha, intercept = coefficients
    fitted = design @ coefficients
    residual = y - fitted
    total = y - np.mean(y)
    r_squared = 1.0 - float(residual @ residual) / max(float(total @ total), 1e-30)
    standard_error = math.sqrt(float(residual @ residual) / (len(x) - 2))
    standard_error /= math.sqrt(float(np.sum((x - np.mean(x)) ** 2)))
    return {
        "status": "finite_scale_ols",
        "points": len(usable),
        "alpha": float(alpha),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "standard_error": float(standard_error),
        "epsilon_min": float(np.min(np.exp(x))),
        "epsilon_max": float(np.max(np.exp(x))),
    }


def summarize_run(run: GridRun, distances: np.ndarray) -> dict[str, Any]:
    finite_times = run.settling_time[run.resolved]
    finite_distances = distances[np.isfinite(distances)]
    unique, counts = np.unique(run.labels[run.labels >= 0], return_counts=True)
    return {
        "name": run.name,
        "window_index": run.window.index,
        "center_theta1": run.window.center_theta1,
        "center_theta2": run.window.center_theta2,
        "width": run.window.width,
        "magnification": run.window.magnification,
        "resolution": run.resolution,
        "spacing": run.spacing,
        "dt": run.dt,
        "horizon": run.horizon,
        "runtime_seconds": run.runtime_seconds,
        "points": int(run.initial.shape[0]),
        "resolved": int(np.sum(run.resolved)),
        "right_censored": int(np.sum(~run.resolved)),
        "resolved_fraction": float(np.mean(run.resolved)),
        "settling_time_min": float(np.min(finite_times)) if finite_times.size else None,
        "settling_time_median": float(np.median(finite_times))
        if finite_times.size
        else None,
        "settling_time_max": float(np.max(finite_times)) if finite_times.size else None,
        "settling_time_q05": float(np.quantile(finite_times, 0.05))
        if finite_times.size
        else None,
        "settling_time_q95": float(np.quantile(finite_times, 0.95))
        if finite_times.size
        else None,
        "final_speed_max": float(np.max(run.final_speed)),
        "attractor_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "unmatched_attractor_points": int(np.sum(run.labels == -2)),
        "terminal_distance_max": (
            float(np.max(finite_distances)) if finite_distances.size else None
        ),
        "terminal_distance_q99": (
            float(np.quantile(finite_distances, 0.99))
            if finite_distances.size
            else None
        ),
    }


def compare_same_grid(reference: GridRun, control: GridRun) -> dict[str, Any]:
    if reference.initial.shape != control.initial.shape:
        raise ValueError("same-grid comparison received different array shapes")
    initial_error = float(
        np.max(np.abs(wrapped_difference(reference.initial, control.initial)))
    )
    valid = reference.resolved & control.resolved
    time_error = np.abs(reference.settling_time[valid] - control.settling_time[valid])
    terminal_error = np.max(
        np.abs(wrapped_difference(reference.final[valid], control.final[valid])),
        axis=-1,
    )
    label_disagreement = reference.labels[valid] != control.labels[valid]
    resolution_disagreement = reference.resolved != control.resolved
    return {
        "reference": reference.name,
        "control": control.name,
        "initial_grid_max_error": initial_error,
        "jointly_resolved": int(np.sum(valid)),
        "jointly_resolved_fraction": float(np.mean(valid)),
        "resolution_status_disagreements": int(np.sum(resolution_disagreement)),
        "resolved_only_in_reference": int(
            np.sum(reference.resolved & ~control.resolved)
        ),
        "resolved_only_in_control": int(np.sum(~reference.resolved & control.resolved)),
        "settling_time_abs_error_median": float(np.median(time_error)),
        "settling_time_abs_error_q95": float(np.quantile(time_error, 0.95)),
        "settling_time_abs_error_max": float(np.max(time_error)),
        "terminal_angle_error_median": float(np.median(terminal_error)),
        "terminal_angle_error_q95": float(np.quantile(terminal_error, 0.95)),
        "terminal_label_disagreements": int(np.sum(label_disagreement)),
    }


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def gamma_scan_summary(run: GridRun, delta_time: float) -> dict[str, Any]:
    """Finite-horizon transition statistics on periodic nearest-neighbour pairs."""

    resolution = run.resolution
    resolved = run.resolved.reshape(resolution, resolution)
    settling = run.settling_time.reshape(resolution, resolution)
    censored = np.where(resolved, settling, run.horizon)
    pair_uncertain = 0
    status_boundary = 0
    pair_count = 0
    for axis in (0, 1):
        shifted_resolved = np.roll(resolved, -1, axis=axis)
        shifted_time = np.roll(censored, -1, axis=axis)
        status_change = resolved != shifted_resolved
        time_change = np.abs(censored - shifted_time) >= delta_time
        pair_uncertain += int(np.sum(status_change | time_change))
        status_boundary += int(np.sum(status_change))
        pair_count += resolved.size

    resolved_times = run.settling_time[run.resolved]
    p_resolved = float(np.mean(run.resolved))
    status_entropy = entropy(np.asarray([p_resolved, 1.0 - p_resolved]))
    bins = np.floor(censored.ravel() / delta_time).astype(np.int64)
    _, counts = np.unique(bins, return_counts=True)
    time_bin_entropy = entropy(counts / np.sum(counts))
    if run.labels is None:
        attractor_entropy = None
    else:
        labels = run.labels[run.labels >= 0]
        if labels.size:
            _, label_counts = np.unique(labels, return_counts=True)
            attractor_entropy = entropy(label_counts / np.sum(label_counts))
        else:
            attractor_entropy = None
    return {
        "gamma": run.name.removeprefix("gamma_"),
        "resolution": resolution,
        "dt": run.dt,
        "horizon": run.horizon,
        "points": int(run.initial.shape[0]),
        "resolved": int(np.sum(run.resolved)),
        "resolved_fraction": p_resolved,
        "settling_time_min_resolved": (
            float(np.min(resolved_times)) if resolved_times.size else None
        ),
        "settling_time_median_resolved": (
            float(np.median(resolved_times)) if resolved_times.size else None
        ),
        "settling_time_max_resolved": (
            float(np.max(resolved_times)) if resolved_times.size else None
        ),
        "mean_right_censored_time": float(np.mean(censored)),
        "delta_time": delta_time,
        "nearest_pair_count": pair_count,
        "nearest_pair_uncertainty_fraction": pair_uncertain / pair_count,
        "nearest_pair_status_boundary_fraction": status_boundary / pair_count,
        "resolved_status_entropy_nats": status_entropy,
        "right_censored_time_bin_entropy_nats": time_bin_entropy,
        "resolved_attractor_entropy_nats": attractor_entropy,
        "final_speed_median": float(np.median(run.final_speed)),
        "final_speed_max": float(np.max(run.final_speed)),
        "runtime_seconds": run.runtime_seconds,
    }


def finite_difference_jacobian(
    field: PlanarTransformerField, point: np.ndarray, step: float
) -> np.ndarray:
    dimension = point.size
    jacobian = np.empty((dimension, dimension), dtype=np.float64)
    for column in range(dimension):
        direction = np.zeros(dimension, dtype=np.float64)
        direction[column] = step
        plus = field.rhs((point + direction)[None, :])[0]
        minus = field.rhs((point - direction)[None, :])[0]
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return jacobian


def gamma_jacobian_spectra(
    record: dict[str, Any], gamma_runs: list[GridRun], difference_step: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in gamma_runs:
        gamma = float(run.name.removeprefix("gamma_"))
        field = PlanarTransformerField.from_record(record, gamma)
        for label in sorted(np.unique(run.labels[run.labels >= 0])):
            points = run.final[run.labels == label]
            prototype = np.angle(np.mean(np.exp(1j * points), axis=0))
            jacobian = finite_difference_jacobian(field, prototype, difference_step)
            eigenvalues = np.linalg.eigvals(jacobian)
            order = np.lexsort((eigenvalues.imag, eigenvalues.real))[::-1]
            eigenvalues = eigenvalues[order]
            residual = float(np.max(np.abs(field.rhs(prototype[None, :]))))
            for eigenvalue_index, eigenvalue in enumerate(eigenvalues):
                rows.append(
                    {
                        "gamma": gamma,
                        "terminal_attractor": int(label),
                        "members": int(points.shape[0]),
                        "prototype_theta1": float(prototype[0]),
                        "prototype_theta2": float(prototype[1]),
                        "prototype_theta3": float(prototype[2]),
                        "prototype_consensus_spread": float(
                            np.max(np.abs(wrapped_difference(prototype, prototype[0])))
                        ),
                        "prototype_speed": residual,
                        "finite_difference_step": difference_step,
                        "eigenvalue_index_descending_real_part": eigenvalue_index,
                        "eigenvalue_real": float(eigenvalue.real),
                        "eigenvalue_imaginary": float(eigenvalue.imag),
                        "spectral_abscissa": float(np.max(eigenvalues.real)),
                    }
                )
    return rows


def synchronized_branch_spectra(
    record: dict[str, Any],
    gamma_values: list[float],
    difference_step: float,
    theta_samples: int = 32769,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve synchronized roots and bracket their saddle-node creation."""

    from scipy.optimize import brentq, minimize_scalar

    theta_grid = np.linspace(-np.pi, np.pi, theta_samples)
    roots_by_gamma: dict[float, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for gamma in sorted(set(gamma_values)):
        field = PlanarTransformerField.from_record(record, gamma)
        synchronized = np.repeat(theta_grid[:, None], 3, axis=1)
        values = field.rhs(synchronized)[:, 0]
        roots: list[float] = []
        for index in np.flatnonzero(values[:-1] * values[1:] < 0.0):
            root = brentq(
                lambda theta: float(
                    field.rhs(np.full((1, 3), theta, dtype=np.float64))[0, 0]
                ),
                float(theta_grid[index]),
                float(theta_grid[index + 1]),
                xtol=1e-14,
                rtol=4.0 * np.finfo(float).eps,
            )
            if not roots or abs(root - roots[-1]) > 1e-9:
                roots.append(float(root))
        roots_by_gamma[gamma] = roots
        for root_index, root in enumerate(roots):
            point = np.full(3, root, dtype=np.float64)
            jacobian = finite_difference_jacobian(field, point, difference_step)
            eigenvalues = np.linalg.eigvals(jacobian)
            order = np.lexsort((eigenvalues.imag, eigenvalues.real))[::-1]
            eigenvalues = eigenvalues[order]
            residual = float(np.max(np.abs(field.rhs(point[None, :]))))
            for eigenvalue_index, eigenvalue in enumerate(eigenvalues):
                rows.append(
                    {
                        "gamma": gamma,
                        "synchronized_root_index": root_index,
                        "theta": root,
                        "root_residual": residual,
                        "finite_difference_step": difference_step,
                        "eigenvalue_index_descending_real_part": eigenvalue_index,
                        "eigenvalue_real": float(eigenvalue.real),
                        "eigenvalue_imaginary": float(eigenvalue.imag),
                        "spectral_abscissa": float(np.max(eigenvalues.real)),
                    }
                )

    ordered_gamma = sorted(roots_by_gamma)
    transition_pair = None
    for low, high in zip(ordered_gamma[:-1], ordered_gamma[1:]):
        if len(roots_by_gamma[low]) == 0 and len(roots_by_gamma[high]) >= 2:
            transition_pair = (low, high)
    if transition_pair is None:
        return rows, {
            "status": "not_bracketed_by_gamma_scan",
            "root_counts": {
                f"{gamma:.10g}": len(roots) for gamma, roots in roots_by_gamma.items()
            },
        }

    lower, upper = transition_pair
    high_roots = roots_by_gamma[upper]
    closest_pair = min(
        zip(high_roots[:-1], high_roots[1:]), key=lambda pair: pair[1] - pair[0]
    )
    separation = closest_pair[1] - closest_pair[0]
    theta_lower = closest_pair[0] - 2.0 * separation
    theta_upper = closest_pair[1] + 2.0 * separation

    def local_peak(gamma: float) -> tuple[float, float]:
        field = PlanarTransformerField.from_record(record, gamma)
        result = minimize_scalar(
            lambda theta: -float(
                field.rhs(np.full((1, 3), theta, dtype=np.float64))[0, 0]
            ),
            bounds=(theta_lower, theta_upper),
            method="bounded",
            options={"xatol": 1e-14, "maxiter": 500},
        )
        return float(result.x), float(-result.fun)

    initial_lower = lower
    initial_upper = upper
    theta_at_lower, peak_at_lower = local_peak(lower)
    theta_at_upper, peak_at_upper = local_peak(upper)
    if not (peak_at_lower <= 0.0 < peak_at_upper):
        return rows, {
            "status": "root_count_transition_found_but_peak_sign_not_bracketed",
            "initial_gamma_bracket": [initial_lower, initial_upper],
            "peak_at_lower": peak_at_lower,
            "peak_at_upper": peak_at_upper,
            "root_counts": {
                f"{gamma:.10g}": len(roots) for gamma, roots in roots_by_gamma.items()
            },
        }
    while upper - lower > 1e-12:
        midpoint = 0.5 * (lower + upper)
        _, peak = local_peak(midpoint)
        if peak > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    theta_lower_peak, final_lower_peak = local_peak(lower)
    theta_upper_peak, final_upper_peak = local_peak(upper)
    return rows, {
        "status": "numerically_bracketed_synchronized_saddle_node",
        "initial_gamma_bracket": [initial_lower, initial_upper],
        "refined_numerical_gamma_bracket": [lower, upper],
        "bracket_width": upper - lower,
        "critical_gamma_midpoint": 0.5 * (lower + upper),
        "critical_theta_midpoint": 0.5 * (theta_lower_peak + theta_upper_peak),
        "local_peak_at_lower": final_lower_peak,
        "local_peak_at_upper": final_upper_peak,
        "local_theta_search_interval": [theta_lower, theta_upper],
        "root_scan_theta_samples": theta_samples,
        "root_solver": "sign-change scan followed by Brent's method",
        "saddle_node_bracket_method": (
            "bisection on the sign of the local synchronized-field maximum; "
            "the maximum is computed by bounded scalar minimization"
        ),
        "gamma_bisection_stopping_width": 1e-12,
        "bracket_caveat": (
            "The bracket is numerical; scalar maximization and floating-point field "
            "evaluation are not interval-certified."
        ),
        "finite_difference_step_for_full_jacobian": difference_step,
        "root_counts": {
            f"{gamma:.10g}": len(roots) for gamma, roots in roots_by_gamma.items()
        },
    }


def saddle_node_scaling(
    synchronized_rows: list[dict[str, Any]], saddle_node: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit the local square-root laws implied by a generic saddle-node."""

    critical = saddle_node.get("critical_gamma_midpoint")
    if critical is None:
        return [], {"status": "no_saddle_node_location"}
    grouped: dict[float, dict[int, list[dict[str, Any]]]] = {}
    for row in synchronized_rows:
        grouped.setdefault(float(row["gamma"]), {}).setdefault(
            int(row["synchronized_root_index"]), []
        ).append(row)
    scaling_rows: list[dict[str, Any]] = []
    for gamma in sorted(grouped):
        roots = grouped[gamma]
        if gamma <= critical or len(roots) != 2:
            continue
        root_data = []
        for root_index, eigen_rows in roots.items():
            root_data.append(
                (
                    float(eigen_rows[0]["theta"]),
                    float(eigen_rows[0]["spectral_abscissa"]),
                    root_index,
                )
            )
        root_data.sort()
        lower, upper = root_data
        scaling_rows.append(
            {
                "gamma": gamma,
                "gamma_minus_critical": gamma - critical,
                "sqrt_gamma_minus_critical": math.sqrt(gamma - critical),
                "lower_root_theta": lower[0],
                "upper_root_theta": upper[0],
                "root_separation": upper[0] - lower[0],
                "lower_root_spectral_abscissa": lower[1],
                "upper_root_spectral_abscissa": upper[1],
                "upper_root_stable_rate_magnitude": (
                    -upper[1] if upper[1] < 0.0 else None
                ),
                "upper_root_relaxation_time": (
                    -1.0 / upper[1] if upper[1] < 0.0 else None
                ),
            }
        )

    def through_origin_fit(key: str) -> dict[str, Any]:
        x = np.asarray(
            [row["sqrt_gamma_minus_critical"] for row in scaling_rows], dtype=float
        )
        y = np.asarray([row[key] for row in scaling_rows], dtype=float)
        if len(x) < 3:
            return {"status": "insufficient_points", "points": len(x)}
        slope = float((x @ y) / (x @ x))
        residual = y - slope * x
        return {
            "status": "finite_range_through_origin_ols",
            "points": len(x),
            "slope": slope,
            "relative_l2_residual": float(np.linalg.norm(residual) / np.linalg.norm(y)),
            "gamma_min": float(min(row["gamma"] for row in scaling_rows)),
            "gamma_max": float(max(row["gamma"] for row in scaling_rows)),
        }

    return scaling_rows, {
        "status": "finite_range_square_root_check",
        "prediction": (
            "for a generic saddle-node, root separation and the magnitudes of the "
            "common-mode eigenvalues scale as sqrt(gamma-gamma_c)"
        ),
        "root_separation_fit": through_origin_fit("root_separation"),
        "unstable_rate_fit": through_origin_fit("lower_root_spectral_abscissa"),
        "stable_rate_magnitude_fit": through_origin_fit(
            "upper_root_stable_rate_magnitude"
        ),
        "caveat": (
            "These are four finite-distance points and a numerically estimated critical "
            "parameter, not an asymptotic theorem for the full three-angle field."
        ),
    }


def synchronized_normal_form_coefficients(
    record: dict[str, Any], decimal_digits: int = 60
) -> dict[str, Any]:
    """Evaluate the synchronized saddle-node derivatives at high precision."""

    import mpmath as mp

    mp.mp.dps = decimal_digits
    model = record["model"]

    def vector(values: Any) -> list[mp.mpf]:
        return [mp.mpf(str(value)) for value in values]

    def matrix(values: Any) -> list[list[mp.mpf]]:
        return [vector(row) for row in values]

    value = matrix(model["value"])
    bias = vector(model["mlp_bias"])
    linear = matrix(model["linear"])
    hidden = matrix(model["hidden"])
    hidden_bias = vector(model["hidden_bias"])
    output = matrix(model["output"])
    depth_scale = mp.mpf(str(model["step_size"]))

    def dot(left: list[mp.mpf], right: list[mp.mpf]) -> mp.mpf:
        return mp.fsum(a * b for a, b in zip(left, right))

    def synchronized_field(theta: mp.mpf, gamma: mp.mpf) -> mp.mpf:
        token = [mp.cos(theta), mp.sin(theta)]
        tangent = [-mp.sin(theta), mp.cos(theta)]
        attention = [dot(row, token) for row in value]
        hidden_values = [
            dot(row, token) + hidden_bias[index] for index, row in enumerate(hidden)
        ]
        mlp = []
        for coordinate in range(2):
            mlp.append(
                bias[coordinate]
                + dot(linear[coordinate], token)
                + mp.fsum(
                    output[coordinate][index] * hidden_values[index] ** 2
                    for index in range(len(hidden_values))
                )
            )
        return depth_scale * dot(
            tangent,
            [attention[index] + gamma * mlp[index] for index in range(2)],
        )

    def theta_derivative(theta: mp.mpf, gamma: mp.mpf) -> mp.mpf:
        return mp.diff(lambda argument: synchronized_field(argument, gamma), theta)

    theta_critical, gamma_critical = mp.findroot(
        (synchronized_field, theta_derivative),
        (mp.mpf("1.1095854"), mp.mpf("1.43478848")),
        tol=mp.mpf(10) ** (-(decimal_digits - 10)),
        maxsteps=100,
    )
    derivative_gamma = mp.diff(
        lambda argument: synchronized_field(theta_critical, argument), gamma_critical
    )
    derivative_theta_theta = mp.diff(
        lambda argument: synchronized_field(argument, gamma_critical),
        theta_critical,
        2,
    )
    root_separation_coefficient = 2 * mp.sqrt(
        -2 * derivative_gamma / derivative_theta_theta
    )
    collective_rate_coefficient = mp.sqrt(
        -2 * derivative_gamma * derivative_theta_theta
    )

    def decimal(value: mp.mpf) -> str:
        return mp.nstr(value, decimal_digits - 5)

    return {
        "status": "high_precision_numerical_evaluation",
        "decimal_digits": decimal_digits,
        "coefficient_source": (
            "decimal coefficients stored in the source JSON, evaluated on the exactly "
            "synchronized scalar field"
        ),
        "theta_critical": decimal(theta_critical),
        "gamma_critical": decimal(gamma_critical),
        "field_derivative_gamma": decimal(derivative_gamma),
        "field_second_derivative_theta": decimal(derivative_theta_theta),
        "predicted_root_separation_coefficient": decimal(root_separation_coefficient),
        "predicted_collective_rate_coefficient": decimal(collective_rate_coefficient),
        "local_expansion": (
            "g(theta,gamma) = g_gamma*(gamma-gamma_c) + "
            "0.5*g_thetatheta*(theta-theta_c)^2 + higher-order terms"
        ),
        "epistemic_status": (
            "numerical high-precision evaluation of stored decimal coefficients; not a "
            "symbolic or interval-certified result"
        ),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_settling_maps(
    path_stem: Path,
    primary_runs: list[GridRun],
    speed_tolerance: float,
) -> None:
    all_times = np.concatenate(
        [run.settling_time[run.resolved] for run in primary_runs]
    )
    lower, upper = np.quantile(all_times, [0.01, 0.99])
    figure, axes = plt.subplots(
        1,
        len(primary_runs),
        figsize=(3.2 * len(primary_runs), 3.05),
        constrained_layout=True,
    )
    if len(primary_runs) == 1:
        axes = np.asarray([axes])
    image = None
    for axis, run in zip(axes, primary_runs):
        values = run.settling_time.reshape(run.resolution, run.resolution)
        image = axis.imshow(
            values,
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap="viridis",
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(
            rf"$\times {run.window.magnification:g}$; width {run.window.width:.3g}"
        )
        axis.set_xlabel(r"$(\theta_1-c_1)/(w/2)$")
        if axis is axes[0]:
            axis.set_ylabel(r"$(\theta_2-c_2)/(w/2)$")
        else:
            axis.set_yticklabels([])
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.015)
    colorbar.set_label(rf"settling time, $\max_i|\dot\theta_i|<{speed_tolerance:g}$")
    for suffix, dpi in ((".pdf", None), (".png", 220)):
        figure.savefig(path_stem.with_suffix(suffix), dpi=dpi)
    plt.close(figure)


def plot_uncertainty(
    path_stem: Path,
    uncertainty_rows: list[dict[str, Any]],
    fits: list[dict[str, Any]],
    primary_prefix: str,
    delta_time: float,
) -> None:
    figure, axis = plt.subplots(figsize=(5.0, 3.65), constrained_layout=True)
    selected_fits = {
        int(item["window_index"]): item
        for item in fits
        if item["run"].startswith(primary_prefix)
        and math.isclose(float(item["delta_time"]), delta_time)
    }
    for window_index in sorted(selected_fits):
        rows = sorted(
            [
                row
                for row in uncertainty_rows
                if row["run"].startswith(primary_prefix)
                and int(row["window_index"]) == window_index
                and math.isclose(float(row["delta_time"]), delta_time)
            ],
            key=lambda row: float(row["epsilon"]),
        )
        x = np.asarray([row["epsilon"] for row in rows], dtype=float)
        y = np.asarray([row["time_uncertainty_fraction"] for row in rows], dtype=float)
        fit = selected_fits[window_index]
        alpha = fit["alpha"]
        label = rf"zoom {window_index}"
        if alpha is not None:
            label += rf", $\widehat\alpha={alpha:.2f}$"
        axis.loglog(x, y, "o-", linewidth=1.2, markersize=3.8, label=label)
    axis.set_xlabel(r"initial-angle separation $\varepsilon$")
    axis.set_ylabel(rf"$\Pr(|T(x)-T(y)|\geq {delta_time:g})$")
    axis.grid(True, which="both", linewidth=0.45, alpha=0.35)
    axis.legend(frameon=False, fontsize=8)
    for suffix, dpi in ((".pdf", None), (".png", 220)):
        figure.savefig(path_stem.with_suffix(suffix), dpi=dpi)
    plt.close(figure)


def plot_gamma_trend(path_stem: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["gamma"]))
    gamma = np.asarray([float(row["gamma"]) for row in ordered])
    resolved = np.asarray([float(row["resolved_fraction"]) for row in ordered])
    uncertain = np.asarray(
        [float(row["nearest_pair_uncertainty_fraction"]) for row in ordered]
    )
    mean_censored = np.asarray(
        [float(row["mean_right_censored_time"]) for row in ordered]
    )
    median_resolved = np.asarray(
        [
            np.nan
            if row["settling_time_median_resolved"] is None
            else float(row["settling_time_median_resolved"])
            for row in ordered
        ]
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(5.2, 5.1), sharex=True, constrained_layout=True
    )
    axes[0].plot(gamma, resolved, "o-", label="resolved by horizon")
    axes[0].plot(gamma, uncertain, "s-", label="nearest-pair uncertainty")
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, linewidth=0.45, alpha=0.35)
    axes[1].plot(gamma, mean_censored, "o-", label="mean, right-censored")
    axes[1].plot(gamma, median_resolved, "s-", label="median among resolved")
    axes[1].set_xlabel(r"MLP multiplier $\gamma$")
    axes[1].set_ylabel("settling time")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, linewidth=0.45, alpha=0.35)
    for suffix, dpi in ((".pdf", None), (".png", 220)):
        figure.savefig(path_stem.with_suffix(suffix), dpi=dpi)
    plt.close(figure)


def plot_gamma_spectrum(path_stem: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(5.0, 3.55), constrained_layout=True)
    attractors = sorted({int(row["terminal_attractor"]) for row in rows})
    for attractor in attractors:
        selected = [
            row
            for row in rows
            if int(row["terminal_attractor"]) == attractor
            and int(row["eigenvalue_index_descending_real_part"]) == 0
        ]
        selected.sort(key=lambda row: float(row["gamma"]))
        axis.plot(
            [float(row["gamma"]) for row in selected],
            [float(row["spectral_abscissa"]) for row in selected],
            "o-",
            label=f"terminal branch {attractor}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel(r"MLP multiplier $\gamma$")
    axis.set_ylabel("largest Jacobian real part")
    axis.grid(True, linewidth=0.45, alpha=0.35)
    if attractors:
        axis.legend(frameon=False, fontsize=8)
    for suffix, dpi in ((".pdf", None), (".png", 220)):
        figure.savefig(path_stem.with_suffix(suffix), dpi=dpi)
    plt.close(figure)


def smooth_saddle_null(
    rate: float,
    offsets: list[int],
    delta_times: list[float],
    atlas_resolution: int,
    window_widths: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Finite-grid uncertainty for T=lambda^-1 log(1/|x|)."""

    rows: list[dict[str, Any]] = []

    def evaluate(resolution: int, width: float, experiment: str) -> None:
        spacing = width / resolution
        normal = -0.5 * width + (np.arange(resolution) + 0.5) * spacing
        settling = (-np.log(np.abs(normal) / width)) / rate
        for offset in offsets:
            if offset >= resolution:
                continue
            normal_difference = np.abs(settling[offset:] - settling[:-offset])
            # The field is constant in the tangential grid direction.  There are
            # resolution copies of both the normal and tangential pair families.
            pair_count = 2 * resolution * (resolution - offset)
            for delta_time in delta_times:
                uncertain = int(np.sum(normal_difference >= delta_time) * resolution)
                rows.append(
                    {
                        "experiment": experiment,
                        "resolution": resolution,
                        "width": width,
                        "offset_pixels": offset,
                        "epsilon": offset * spacing,
                        "delta_time": delta_time,
                        "valid_pairs": pair_count,
                        "time_uncertain_pairs": uncertain,
                        "time_uncertainty_fraction": uncertain / pair_count,
                    }
                )

    for window_index, width in enumerate(window_widths):
        evaluate(atlas_resolution, width, f"comoving_zoom_z{window_index}")
    refinement_resolutions = [64, 128, 256, 512, 1024, 2048, 4096]
    for resolution in refinement_resolutions:
        evaluate(resolution, TWO_PI, f"fixed_region_n{resolution}")

    fits: list[dict[str, Any]] = []
    experiments = sorted({str(row["experiment"]) for row in rows})
    for experiment in experiments:
        for delta_time in delta_times:
            selected = [
                row
                for row in rows
                if row["experiment"] == experiment
                and math.isclose(float(row["delta_time"]), delta_time)
            ]
            exemplar = selected[0]
            fits.append(
                {
                    "experiment": experiment,
                    "resolution": exemplar["resolution"],
                    "width": exemplar["width"],
                    "delta_time": delta_time,
                    **power_law_fit(selected),
                }
            )
    return (
        rows,
        fits,
        {
            "model": "T(x)=lambda^{-1} log(1/|x|) for a smooth codimension-one saddle",
            "lambda": rate,
            "pairing": (
                "nonperiodic horizontal and vertical grid pairs are pooled; the singular line "
                "is horizontal, so horizontal pairs contribute zero uncertainty"
            ),
            "comoving_test": (
                "the resolution is fixed while the window shrinks; additive shifts in T cancel, "
                "so its finite-scale uncertainty fractions need not change under zoom"
            ),
            "fixed_region_refinement_resolutions": refinement_resolutions,
            "interpretation": (
                "a low exponent on accessible scales, even when stable under co-moving zoom, is "
                "compatible with a smooth saddle. Fixed-region refinement is required to expose "
                "the asymptotic codimension-one exponent."
            ),
        },
    )


def plot_smooth_saddle_null(
    path_stem: Path, fits: list[dict[str, Any]], delta_times: list[float]
) -> None:
    figure, axis = plt.subplots(figsize=(5.0, 3.55), constrained_layout=True)
    for delta_time in delta_times:
        selected = [
            row
            for row in fits
            if str(row["experiment"]).startswith("fixed_region_n")
            and math.isclose(float(row["delta_time"]), delta_time)
            and row["alpha"] is not None
        ]
        selected.sort(key=lambda row: int(row["resolution"]))
        axis.semilogx(
            [int(row["resolution"]) for row in selected],
            [float(row["alpha"]) for row in selected],
            "o-",
            label=rf"$\Delta T={delta_time:g}$",
        )
    axis.axhline(1.0, color="black", linewidth=0.8, label="smooth asymptote")
    axis.set_xlabel("fixed-region grid resolution")
    axis.set_ylabel(r"finite-scale fitted $\widehat\alpha$")
    axis.grid(True, linewidth=0.45, alpha=0.35)
    axis.legend(frameon=False, fontsize=8)
    for suffix, dpi in ((".pdf", None), (".png", 220)):
        figure.savefig(path_stem.with_suffix(suffix), dpi=dpi)
    plt.close(figure)


def fold_normal_form_null(
    quadratic_coefficient: float,
    offsets: list[int],
    delta_times: list[float],
    atlas_resolution: int,
    window_widths: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Finite-grid uncertainty for the critical fold law T=1/(a|x|)."""

    if quadratic_coefficient <= 0.0:
        raise ValueError("the fold coefficient must be positive")
    rows: list[dict[str, Any]] = []

    def evaluate(resolution: int, width: float, experiment: str) -> None:
        spacing = width / resolution
        normal = -0.5 * width + (np.arange(resolution) + 0.5) * spacing
        settling = 1.0 / (quadratic_coefficient * np.abs(normal))
        for offset in offsets:
            if offset >= resolution:
                continue
            normal_difference = np.abs(settling[offset:] - settling[:-offset])
            pair_count = 2 * resolution * (resolution - offset)
            for delta_time in delta_times:
                uncertain = int(np.sum(normal_difference >= delta_time) * resolution)
                rows.append(
                    {
                        "experiment": experiment,
                        "resolution": resolution,
                        "width": width,
                        "offset_pixels": offset,
                        "epsilon": offset * spacing,
                        "delta_time": delta_time,
                        "valid_pairs": pair_count,
                        "time_uncertain_pairs": uncertain,
                        "time_uncertainty_fraction": uncertain / pair_count,
                    }
                )

    for window_index, width in enumerate(window_widths):
        evaluate(atlas_resolution, width, f"comoving_zoom_z{window_index}")
    refinement_resolutions = [64, 128, 256, 512, 1024, 2048, 4096]
    for resolution in refinement_resolutions:
        evaluate(resolution, TWO_PI, f"fixed_region_n{resolution}")

    fits: list[dict[str, Any]] = []
    for experiment in sorted({str(row["experiment"]) for row in rows}):
        for delta_time in delta_times:
            selected = [
                row
                for row in rows
                if row["experiment"] == experiment
                and math.isclose(float(row["delta_time"]), delta_time)
            ]
            exemplar = selected[0]
            fits.append(
                {
                    "experiment": experiment,
                    "resolution": exemplar["resolution"],
                    "width": exemplar["width"],
                    "delta_time": delta_time,
                    **power_law_fit(selected),
                }
            )
    return (
        rows,
        fits,
        {
            "model": "du/dt=-a*u^2 at the fold, giving T(u)=1/(a*|u|)",
            "quadratic_coefficient_a": quadratic_coefficient,
            "pairing": (
                "nonperiodic horizontal and vertical grid pairs are pooled; the singular "
                "line is horizontal, so horizontal pairs contribute zero uncertainty"
            ),
            "asymptotic_prediction": (
                "for fixed delta_time and fixed physical region, the uncertain width is "
                "proportional to sqrt(epsilon), hence alpha=1/2 although the singular set "
                "is a smooth line"
            ),
            "fixed_region_refinement_resolutions": refinement_resolutions,
            "scope": (
                "analytic finite-grid null without the atlas censoring and adaptive-zoom "
                "rules; it tests whether a low exponent alone identifies fractality"
            ),
        },
    )


def plot_fold_normal_form_null(
    path_stem: Path, fits: list[dict[str, Any]], delta_times: list[float]
) -> None:
    figure, axis = plt.subplots(figsize=(5.0, 3.55), constrained_layout=True)
    for delta_time in delta_times:
        selected = [
            row
            for row in fits
            if str(row["experiment"]).startswith("fixed_region_n")
            and math.isclose(float(row["delta_time"]), delta_time)
            and row["alpha"] is not None
        ]
        selected.sort(key=lambda row: int(row["resolution"]))
        axis.semilogx(
            [int(row["resolution"]) for row in selected],
            [float(row["alpha"]) for row in selected],
            "o-",
            label=rf"$\Delta T={delta_time:g}$",
        )
    axis.axhline(0.5, color="black", linewidth=0.8, label=r"smooth-fold limit $1/2$")
    axis.set_xlabel("fixed-region grid resolution")
    axis.set_ylabel(r"finite-scale fitted $\widehat\alpha$")
    axis.grid(True, linewidth=0.45, alpha=0.35)
    axis.legend(frameon=False, fontsize=8)
    for suffix, dpi in ((".pdf", None), (".png", 220)):
        figure.savefig(path_stem.with_suffix(suffix), dpi=dpi)
    plt.close(figure)


def affine_strip_geometry(
    run: GridRun,
    speed_only_resolved: np.ndarray,
    speed_only_settling_time: np.ndarray,
) -> list[dict[str, Any]]:
    """Measure whether right-censored pixels form a smooth one-pixel-scale strip."""

    rows: list[dict[str, Any]] = []
    for definition, resolved, settling_flat in (
        ("speed_only", speed_only_resolved, speed_only_settling_time),
        ("endpoint_closer_to_stable_root", run.resolved, run.settling_time),
    ):
        censored = (~resolved).reshape(run.resolution, run.resolution)
        columns = np.flatnonzero(np.any(censored, axis=0))
        components: list[int] = []
        transitions: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        for column in columns:
            values = censored[:, column].astype(np.int8)
            changes = np.diff(np.pad(values, (1, 1)))
            components.append(int(np.sum(changes == 1)))
            transitions.append(int(np.sum(changes != 0)))
            occupied = np.flatnonzero(values)
            lower.append(float(occupied[0]))
            upper.append(float(occupied[-1]))

        def affine_residual(y: list[float]) -> tuple[float | None, float | None]:
            if len(y) < 2:
                return None, None
            x = columns.astype(float)
            design = np.column_stack((x, np.ones_like(x)))
            fitted = design @ np.linalg.lstsq(design, np.asarray(y), rcond=None)[0]
            residual = np.asarray(y) - fitted
            return (
                float(np.sqrt(np.mean(residual**2))),
                float(np.max(np.abs(residual))),
            )

        lower_rmse, lower_max = affine_residual(lower)
        upper_rmse, upper_max = affine_residual(upper)
        settling = settling_flat.reshape(run.resolution, run.resolution)
        ridge_columns: list[int] = []
        ridge_rows: list[float] = []
        for column in range(run.resolution):
            finite = np.flatnonzero(np.isfinite(settling[:, column]))
            if finite.size:
                ridge_columns.append(column)
                ridge_rows.append(
                    float(finite[np.argmin(settling[finite, column])])
                )
        if len(ridge_rows) >= 2:
            x_ridge = np.asarray(ridge_columns, dtype=float)
            design = np.column_stack((x_ridge, np.ones_like(x_ridge)))
            fitted = design @ np.linalg.lstsq(
                design, np.asarray(ridge_rows), rcond=None
            )[0]
            ridge_residual = np.asarray(ridge_rows) - fitted
            ridge_rmse = float(np.sqrt(np.mean(ridge_residual**2)))
            ridge_max = float(np.max(np.abs(ridge_residual)))
        else:
            ridge_rmse = None
            ridge_max = None
        rows.append(
            {
                "run": run.name,
                "censoring_definition": definition,
                "resolution": run.resolution,
                "censored_pixels": int(np.sum(censored)),
                "columns_with_censoring": int(columns.size),
                "max_components_in_nonempty_column": (
                    int(max(components)) if components else 0
                ),
                "max_transitions_in_nonempty_column": (
                    int(max(transitions)) if transitions else 0
                ),
                "all_nonempty_columns_single_interval": bool(
                    components and max(components) == 1
                ),
                "lower_edge_affine_rmse_pixels": lower_rmse,
                "lower_edge_affine_max_abs_pixels": lower_max,
                "upper_edge_affine_rmse_pixels": upper_rmse,
                "upper_edge_affine_max_abs_pixels": upper_max,
                "minimum_time_ridge_affine_rmse_pixels": ridge_rmse,
                "minimum_time_ridge_affine_max_abs_pixels": ridge_max,
            }
        )
    return rows


def validate_rhs(
    record: dict[str, Any], field: PlanarTransformerField, seed: int
) -> float:
    """Compare the specialized field with the repository reference implementation."""

    from experiments.spectral_self_attention.small_step_continuation import (
        continuous_angular_field,
        stack_models,
    )

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=(1, 17, 3))
    models = stack_models([record])
    for key in ("mlp_bias", "linear", "output"):
        models[key] *= field.gamma
    reference = models["step_size"][:, None, None] * continuous_angular_field(
        angles, models
    )
    error = float(np.max(np.abs(field.rhs(angles[0]) - reference[0])))
    if error > 5e-13:
        raise AssertionError(f"specialized RHS mismatch: {error}")
    return error


def benettin_qr_before_settling(
    field: PlanarTransformerField,
    initial: np.ndarray,
    duration: float,
    dt: float,
    epsilon: float,
    renormalization_interval: float,
    seed: int,
) -> dict[str, Any]:
    """Finite-time tangent spectrum from three paired perturbations."""

    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    base = initial.copy()
    perturbed = wrap(base[None, :] + epsilon * basis.T)
    accumulated = np.zeros(3, dtype=np.float64)
    interval_rates: list[float] = []
    steps = int(np.ceil(duration / dt))
    stride = max(1, int(round(renormalization_interval / dt)))
    elapsed = 0.0
    interval_elapsed = 0.0
    for step in range(steps):
        states = np.concatenate((base[None, :], perturbed), axis=0)
        states = field.rk4_step(states, dt)
        base = states[0]
        perturbed = states[1:]
        elapsed += dt
        interval_elapsed += dt
        if (step + 1) % stride != 0 and step + 1 != steps:
            continue
        differences = wrapped_difference(perturbed, base[None, :]).T
        orthogonal, triangular = np.linalg.qr(differences)
        diagonal = np.maximum(np.abs(np.diag(triangular)), 1e-300)
        log_stretch = np.log(diagonal / epsilon)
        accumulated += log_stretch
        interval_rates.append(float(np.max(log_stretch) / interval_elapsed))
        perturbed = wrap(base[None, :] + epsilon * orthogonal.T)
        interval_elapsed = 0.0
    exponents = accumulated / elapsed
    exponents.sort()
    exponents = exponents[::-1]
    return {
        "duration_requested": duration,
        "duration_integrated": elapsed,
        "dt": dt,
        "epsilon": epsilon,
        "renormalization_interval_requested": renormalization_interval,
        "renormalization_steps": stride,
        "seed": seed,
        "finite_time_exponent_1": float(exponents[0]),
        "finite_time_exponent_2": float(exponents[1]),
        "finite_time_exponent_3": float(exponents[2]),
        "maximum_interval_log_stretch_rate": float(max(interval_rates)),
        "positive_interval_fraction": float(np.mean(np.asarray(interval_rates) > 0.0)),
        "scope": (
            "Benettin QR estimate along one finite pre-settling trajectory; it is not an "
            "asymptotic Lyapunov exponent of an invariant set"
        ),
    }


def transient_tangent_postprocess(
    record: dict[str, Any],
    arrays: Any,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for window_index in range(args.zoom_levels):
        name = f"primary_z{window_index}"
        settling = arrays[f"{name}_settling_time"]
        finite = np.flatnonzero(np.isfinite(settling))
        for point in finite:
            candidates.append(
                {
                    "run": name,
                    "point": int(point),
                    "settling_time": float(settling[point]),
                }
            )
    longest = max(candidates, key=lambda row: row["settling_time"])
    base_times = arrays["primary_z0_settling_time"]
    base_finite = np.flatnonzero(np.isfinite(base_times))
    median = float(np.median(base_times[base_finite]))
    typical_point = int(
        base_finite[np.argmin(np.abs(base_times[base_finite] - median))]
    )
    selections = [
        ("longest_resolved", longest | {"gamma": args.gamma}),
        (
            "typical_base",
            {
                "run": "primary_z0",
                "point": typical_point,
                "settling_time": float(base_times[typical_point]),
                "gamma": args.gamma,
            },
        ),
    ]
    below_name = "gamma_1.4347500"
    below_initial = arrays[f"{below_name}_initial"]
    source_state = np.asarray(record["final_angle"], dtype=float)
    below_point = int(
        np.argmin(
            np.linalg.norm(wrapped_difference(below_initial, source_state), axis=-1)
        )
    )
    selections.append(
        (
            "below_saddle_node_nearest_source_state",
            {
                "run": below_name,
                "point": below_point,
                "settling_time": args.gamma_scan_horizon,
                "gamma": 1.43475,
            },
        )
    )
    rows: list[dict[str, Any]] = []
    controls = [(args.dt, 1e-7), (0.5 * args.dt, 1e-7), (args.dt, 1e-8)]
    for selection_index, (selection_name, selection) in enumerate(selections):
        initial = arrays[f"{selection['run']}_initial"][selection["point"]]
        field = PlanarTransformerField.from_record(record, selection["gamma"])
        for dt, epsilon in controls:
            audit = benettin_qr_before_settling(
                field,
                initial,
                selection["settling_time"],
                dt,
                epsilon,
                0.2,
                args.seed + 1000 + selection_index,
            )
            rows.append(
                {
                    "selection": selection_name,
                    "source_run": selection["run"],
                    "source_point": selection["point"],
                    "row": selection["point"]
                    // int(math.sqrt(arrays[f"{selection['run']}_initial"].shape[0])),
                    "column": selection["point"]
                    % int(math.sqrt(arrays[f"{selection['run']}_initial"].shape[0])),
                    "recorded_settling_time": selection["settling_time"],
                    "gamma": selection["gamma"],
                    "initial_theta1": float(initial[0]),
                    "initial_theta2": float(initial[1]),
                    "initial_theta3": float(initial[2]),
                    **audit,
                }
            )
    return {
        "method": (
            "three perturbations evolved with the base trajectory and QR-renormalized every "
            "0.2 normalized time units"
        ),
        "selection_rule": (
            "longest resolved point over all primary zooms and the base-grid point nearest "
            "the median resolved settling time, both at gamma=1.4348; plus the gamma=1.43475 "
            "scan point nearest the source record's terminal state, followed for the full "
            "right-censoring horizon"
        ),
        "source_gamma_one_reported_finite_time_rate": record["metrics"][
            "lyapunov_per_normalized_time"
        ],
        "source_gamma_one_scope": (
            "value copied from the source record's finite-horizon continuous ODE audit"
        ),
        "rows": rows,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    record = load_record(args.input, args.source_model_index, 3)
    field = PlanarTransformerField.from_record(record, args.gamma)
    rhs_error = validate_rhs(record, field, args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fixed_theta3 = float(args.fixed_theta3)
    windows: list[Window] = [Window(0, 0.0, 0.0, TWO_PI)]
    primary_runs: list[GridRun] = []
    zoom_selections: list[dict[str, Any]] = []
    for window_index in range(args.zoom_levels):
        window = windows[window_index]
        run = integrate_grid(
            f"primary_z{window_index}",
            field,
            window,
            args.resolution,
            fixed_theta3,
            args.dt,
            args.horizon,
            args.speed_tolerance,
            args.validation_tail,
        )
        primary_runs.append(run)
        if window_index + 1 < args.zoom_levels:
            center1, center2, selection = roughest_cell_center(run)
            zoom_selections.append({"source_window_index": window_index, **selection})
            windows.append(
                Window(
                    window_index + 1,
                    center1,
                    center2,
                    window.width / args.zoom_factor,
                )
            )

    control_runs: list[GridRun] = []
    for window in windows:
        control_runs.append(
            integrate_grid(
                f"spatial_z{window.index}",
                field,
                window,
                args.control_resolution,
                fixed_theta3,
                args.dt,
                args.horizon,
                args.speed_tolerance,
                args.validation_tail,
            )
        )
        control_runs.append(
            integrate_grid(
                f"half_dt_z{window.index}",
                field,
                window,
                args.control_resolution,
                fixed_theta3,
                0.5 * args.dt,
                args.horizon,
                args.speed_tolerance,
                args.validation_tail,
            )
        )
    for window_index in sorted(set((0, len(windows) - 1))):
        window = windows[window_index]
        control_runs.append(
            integrate_grid(
                f"long_horizon_z{window.index}",
                field,
                window,
                args.control_resolution,
                fixed_theta3,
                args.dt,
                args.control_horizon,
                args.speed_tolerance,
                args.validation_tail,
            )
        )

    gamma_runs: list[GridRun] = []
    for gamma in args.gamma_scan_values:
        gamma_field = PlanarTransformerField.from_record(record, gamma)
        gamma_runs.append(
            integrate_grid(
                f"gamma_{gamma:.7f}",
                gamma_field,
                windows[0],
                args.gamma_scan_resolution,
                fixed_theta3,
                args.dt,
                args.gamma_scan_horizon,
                args.speed_tolerance,
                args.validation_tail,
            )
        )

    main_runs = primary_runs + control_runs
    checkpoint_arrays: dict[str, np.ndarray] = {}
    for run in main_runs + gamma_runs:
        checkpoint_arrays[f"{run.name}_initial"] = run.initial
        checkpoint_arrays[f"{run.name}_final"] = run.final
        checkpoint_arrays[f"{run.name}_settling_time"] = run.settling_time
        checkpoint_arrays[f"{run.name}_resolved"] = run.resolved
        checkpoint_arrays[f"{run.name}_terminal_speed"] = run.final_speed
    np.savez_compressed(output / "precheck_trajectories.npz", **checkpoint_arrays)
    (output / "precheck_metadata.json").write_text(
        json.dumps(
            {
                "status": "trajectories_complete_analysis_pending",
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "primary_resolution": args.resolution,
                "control_resolution": args.control_resolution,
                "runs": [
                    {
                        "name": run.name,
                        "resolved": int(np.sum(run.resolved)),
                        "right_censored": int(np.sum(~run.resolved)),
                        "points": int(run.initial.shape[0]),
                        "runtime_seconds": run.runtime_seconds,
                    }
                    for run in main_runs + gamma_runs
                ],
            },
            indent=2,
        )
        + "\n"
    )

    primary_finals = np.concatenate([run.final for run in main_runs])
    primary_resolved = np.concatenate([run.resolved for run in main_runs])
    prototypes = fit_attractor_prototypes(
        primary_finals, primary_resolved, args.attractor_tolerance
    )
    all_runs = main_runs + gamma_runs
    distances_by_run: dict[str, np.ndarray] = {}
    for run in main_runs:
        run.labels, distances = assign_attractor_labels(
            run.final, run.resolved, prototypes, args.attractor_tolerance
        )
        distances_by_run[run.name] = distances
    gamma_attractor_counts: dict[str, int] = {}
    for run in gamma_runs:
        if np.any(run.resolved):
            local_prototypes = fit_attractor_prototypes(
                run.final, run.resolved, args.attractor_tolerance
            )
            run.labels, distances = assign_attractor_labels(
                run.final, run.resolved, local_prototypes, args.attractor_tolerance
            )
            gamma_attractor_counts[run.name] = int(len(local_prototypes))
        else:
            run.labels = np.full(run.final.shape[0], -1, dtype=np.int16)
            distances = np.full(run.final.shape[0], np.nan)
            gamma_attractor_counts[run.name] = 0
        distances_by_run[run.name] = distances

    uncertainty_rows: list[dict[str, Any]] = []
    for run in all_runs:
        uncertainty_rows.extend(
            pair_uncertainty(
                run,
                args.uncertainty_offsets,
                args.delta_times,
                periodic=run.window.index == 0,
            )
        )
    fit_rows: list[dict[str, Any]] = []
    for run in all_runs:
        for delta_time in args.delta_times:
            selected = [
                row
                for row in uncertainty_rows
                if row["run"] == run.name
                and math.isclose(float(row["delta_time"]), delta_time)
            ]
            fit_rows.append(
                {
                    "run": run.name,
                    "window_index": run.window.index,
                    "resolution": run.resolution,
                    "dt": run.dt,
                    "horizon": run.horizon,
                    "delta_time": delta_time,
                    **power_law_fit(selected),
                }
            )

    summaries = [summarize_run(run, distances_by_run[run.name]) for run in all_runs]
    by_name = {run.name: run for run in all_runs}
    comparisons = []
    for window in windows:
        spatial = by_name[f"spatial_z{window.index}"]
        comparisons.append(
            compare_same_grid(spatial, by_name[f"half_dt_z{window.index}"])
        )
        long_name = f"long_horizon_z{window.index}"
        if long_name in by_name:
            comparisons.append(compare_same_grid(spatial, by_name[long_name]))

    # Structural failures remain fatal. Interpretive diagnostics are serialized.
    if not np.all(np.isfinite(prototypes)):
        raise AssertionError("non-finite attractor prototype")
    horizon_comparisons = [
        item for item in comparisons if item["control"].startswith("long_horizon")
    ]
    prototype_speed = np.max(np.abs(field.rhs(prototypes)), axis=-1)
    validation_flags = {
        "every_primary_zoom_has_a_resolved_point": bool(
            all(np.any(run.resolved) for run in primary_runs)
        ),
        "every_control_grid_has_a_resolved_point": bool(
            all(np.any(run.resolved) for run in control_runs)
        ),
        "unmatched_resolved_terminal_points": {
            run.name: int(np.sum(run.labels == -2)) for run in main_runs
        },
        "terminal_label_disagreements_across_controls": int(
            sum(item["terminal_label_disagreements"] for item in comparisons)
        ),
        "horizon_control_settling_time_exceeds_one_dt": bool(
            any(
                item["settling_time_abs_error_max"] > args.dt * 1.01
                for item in horizon_comparisons
            )
        ),
        "prototype_speed_max": float(np.max(prototype_speed)),
        "all_prototypes_below_speed_tolerance": bool(
            float(np.max(prototype_speed)) <= args.speed_tolerance
        ),
    }

    gamma_rows = [
        gamma_scan_summary(run, args.primary_delta_time) for run in gamma_runs
    ]
    gamma_spectrum_rows = gamma_jacobian_spectra(
        record, gamma_runs, args.jacobian_difference_step
    )
    synchronized_rows, saddle_node = synchronized_branch_spectra(
        record, args.gamma_scan_values, args.jacobian_difference_step
    )
    saddle_scaling_rows, saddle_scaling_fit = saddle_node_scaling(
        synchronized_rows, saddle_node
    )
    normal_form = synchronized_normal_form_coefficients(record)

    save_csv(output / "uncertainty_scaling.csv", uncertainty_rows)
    save_csv(output / "uncertainty_fits.csv", fit_rows)
    save_csv(output / "grid_summary.csv", summaries)
    save_csv(output / "numerical_controls.csv", comparisons)
    save_csv(output / "gamma_trend.csv", gamma_rows)
    if gamma_spectrum_rows:
        save_csv(output / "gamma_jacobian_spectra.csv", gamma_spectrum_rows)
    if synchronized_rows:
        save_csv(output / "synchronized_branch_spectra.csv", synchronized_rows)
    if saddle_scaling_rows:
        save_csv(output / "saddle_node_scaling.csv", saddle_scaling_rows)
    save_csv(output / "saddle_node_normal_form.csv", [normal_form])

    raw_rows: list[dict[str, Any]] = []
    for run in primary_runs:
        for point in range(run.initial.shape[0]):
            raw_rows.append(
                {
                    "window_index": run.window.index,
                    "row": point // run.resolution,
                    "column": point % run.resolution,
                    "theta1": float(run.theta1.ravel()[point]),
                    "theta2": float(run.theta2.ravel()[point]),
                    "fixed_theta3": fixed_theta3,
                    "settling_time": float(run.settling_time[point]),
                    "terminal_attractor": int(run.labels[point]),
                    "terminal_theta1": float(run.final[point, 0]),
                    "terminal_theta2": float(run.final[point, 1]),
                    "terminal_theta3": float(run.final[point, 2]),
                    "terminal_speed": float(run.final_speed[point]),
                }
            )
    save_csv(output / "primary_grid.csv", raw_rows)

    arrays: dict[str, np.ndarray] = {"attractor_prototypes": prototypes}
    for run in all_runs:
        prefix = run.name
        arrays[f"{prefix}_initial"] = run.initial
        arrays[f"{prefix}_final"] = run.final
        arrays[f"{prefix}_settling_time"] = run.settling_time
        arrays[f"{prefix}_terminal_label"] = run.labels
        arrays[f"{prefix}_terminal_speed"] = run.final_speed
    np.savez_compressed(output / "grids.npz", **arrays)

    plot_settling_maps(
        output / "fractal_settling_atlas", primary_runs, args.speed_tolerance
    )
    plot_uncertainty(
        output / "fractal_settling_uncertainty",
        uncertainty_rows,
        fit_rows,
        "primary_",
        args.primary_delta_time,
    )
    plot_gamma_trend(output / "fractal_settling_gamma_trend", gamma_rows)
    if gamma_spectrum_rows:
        plot_gamma_spectrum(
            output / "fractal_settling_gamma_spectrum", gamma_spectrum_rows
        )

    total_runtime = time.perf_counter() - started
    primary_fits = [
        row
        for row in fit_rows
        if row["run"].startswith("primary_")
        and math.isclose(float(row["delta_time"]), args.primary_delta_time)
    ]
    result = {
        "experiment": "finite-scale settling-time uncertainty on a 2D initial-state slice",
        "epistemic_status": (
            "A bounded multiscale numerical diagnostic.  A fitted alpha below one "
            "over the sampled scales is not a proof of a fractal set or an "
            "asymptotic uncertainty exponent."
        ),
        "source": {
            "path": str(args.input),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "source_model_index": args.source_model_index,
            "family": int(record["identity"]["family"]),
            "n_tokens": int(record["identity"]["n_tokens"]),
            "original_final_angle": record["final_angle"],
            "model": record["model"],
            "source_record_continuous_metrics": record["metrics"],
        },
        "field_modification": {
            "gamma": args.gamma,
            "definition": (
                "attention + gamma * (MLP bias + MLP linear term + MLP quadratic-output term); "
                "hidden weights and hidden biases are unchanged"
            ),
        },
        "slice": {
            "varied_coordinates": ["theta1", "theta2"],
            "fixed_theta3": fixed_theta3,
            "base_domain": "[-pi, pi) x [-pi, pi), sampled at cell centres",
            "zoom_rule": (
                "choose the interior grid cell with the largest four-corner settling-time range, "
                "then divide the window width by the fixed zoom factor"
            ),
            "windows": [
                window.__dict__ | {"magnification": window.magnification}
                for window in windows
            ],
            "zoom_selections": zoom_selections,
        },
        "settling_definition": {
            "speed": "max_i |d theta_i / dt|",
            "speed_tolerance": args.speed_tolerance,
            "settling_time": (
                "dt plus the last sampled time at which speed >= tolerance; censored unless the "
                "remaining validation tail stays below tolerance"
            ),
            "validation_tail": args.validation_tail,
        },
        "uncertainty_definition": {
            "time_pair_uncertain": "|T(x)-T(y)| >= delta_time",
            "basin_pair_uncertain": "terminal attractor labels differ",
            "delta_times": args.delta_times,
            "primary_delta_time": args.primary_delta_time,
            "pixel_offsets": args.uncertainty_offsets,
            "base_window_pairing": "periodic coordinate neighbours",
            "zoom_window_pairing": "interior coordinate neighbours",
            "fit": "unweighted OLS of log f on log epsilon using only 0 < f < 1",
            "censoring": (
                "multiscale uncertainty fractions are conditional on both endpoints being "
                "resolved; gamma-trend uncertainty treats an unresolved endpoint as "
                "right-censored at the gamma-scan horizon"
            ),
        },
        "numerics": {
            "integrator": (
                "classical explicit RK4; trigonometric periodicity is used within a step and "
                "angles are wrapped after the completed step"
            ),
            "primary_resolution": args.resolution,
            "control_resolution": args.control_resolution,
            "primary_dt": args.dt,
            "half_dt_control": 0.5 * args.dt,
            "primary_horizon": args.horizon,
            "long_horizon_control": args.control_horizon,
            "deterministic_seed": args.seed,
            "specialized_vs_repository_rhs_max_error": rhs_error,
            "execution_note": args.execution_note,
            "command": [sys.executable, *sys.argv],
        },
        "attractors": {
            "clustering_tolerance_wrapped_linf": args.attractor_tolerance,
            "count": int(len(prototypes)),
            "prototypes": prototypes.tolist(),
            "prototype_speeds": prototype_speed.tolist(),
        },
        "run_summaries": summaries,
        "primary_uncertainty_fits": primary_fits,
        "numerical_controls": comparisons,
        "validation_flags": validation_flags,
        "gamma_trend": {
            "status": (
                "Finite-horizon coarse-grid scan. Unresolved trajectories are right-censored "
                "at the gamma-scan horizon."
            ),
            "resolution": args.gamma_scan_resolution,
            "horizon": args.gamma_scan_horizon,
            "rows": gamma_rows,
            "local_attractor_counts": gamma_attractor_counts,
            "terminal_prototype_jacobian_spectra": gamma_spectrum_rows,
            "synchronized_branch_spectra": synchronized_rows,
            "synchronized_saddle_node": saddle_node,
            "saddle_node_scaling_rows": saddle_scaling_rows,
            "saddle_node_scaling_fit": saddle_scaling_fit,
            "saddle_node_normal_form": normal_form,
        },
        "runtime": {
            "total_seconds": total_runtime,
            "integration_seconds": float(sum(run.runtime_seconds for run in all_runs)),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "summary_json": "summary.json",
            "grid_summary_csv": "grid_summary.csv",
            "uncertainty_csv": "uncertainty_scaling.csv",
            "fits_csv": "uncertainty_fits.csv",
            "controls_csv": "numerical_controls.csv",
            "gamma_trend_csv": "gamma_trend.csv",
            "gamma_jacobian_spectra_csv": "gamma_jacobian_spectra.csv",
            "synchronized_branch_spectra_csv": "synchronized_branch_spectra.csv",
            "saddle_node_scaling_csv": "saddle_node_scaling.csv",
            "saddle_node_normal_form_csv": "saddle_node_normal_form.csv",
            "primary_grid_csv": "primary_grid.csv",
            "arrays_npz": "grids.npz",
            "precheck_arrays_npz": "precheck_trajectories.npz",
            "precheck_metadata_json": "precheck_metadata.json",
            "maps_pdf": "fractal_settling_atlas.pdf",
            "maps_png": "fractal_settling_atlas.png",
            "uncertainty_pdf": "fractal_settling_uncertainty.pdf",
            "uncertainty_png": "fractal_settling_uncertainty.png",
            "gamma_trend_pdf": "fractal_settling_gamma_trend.pdf",
            "gamma_trend_png": "fractal_settling_gamma_trend.png",
            "gamma_spectrum_pdf": "fractal_settling_gamma_spectrum.pdf",
            "gamma_spectrum_png": "fractal_settling_gamma_spectrum.png",
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def postprocess_gamma_spectrum(args: argparse.Namespace) -> dict[str, Any]:
    """Add Jacobian and synchronized-branch data without rerunning trajectories."""

    output = args.output_dir
    summary_path = output / "summary.json"
    arrays_path = output / "grids.npz"
    if not summary_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(
            "the trajectory run must finish before spectral postprocessing"
        )
    summary = json.loads(summary_path.read_text())
    arrays = np.load(arrays_path)
    record = load_record(args.input, args.source_model_index, 3)
    summary["source"]["input_sha256"] = hashlib.sha256(
        args.input.read_bytes()
    ).hexdigest()
    summary["source"]["model"] = record["model"]
    summary["source"]["source_record_continuous_metrics"] = record["metrics"]
    base = summary["slice"]["windows"][0]
    window = Window(
        0,
        float(base["center_theta1"]),
        float(base["center_theta2"]),
        float(base["width"]),
    )
    trend_by_gamma = {
        float(row["gamma"]): row for row in summary["gamma_trend"]["rows"]
    }
    gamma_runs: list[GridRun] = []
    for gamma in args.gamma_scan_values:
        name = f"gamma_{gamma:.7f}"
        initial = arrays[f"{name}_initial"]
        final = arrays[f"{name}_final"]
        settling = arrays[f"{name}_settling_time"]
        labels = arrays[f"{name}_terminal_label"]
        final_speed = arrays[f"{name}_terminal_speed"]
        row = trend_by_gamma[gamma]
        resolution = int(row["resolution"])
        gamma_runs.append(
            GridRun(
                name=name,
                window=window,
                resolution=resolution,
                dt=float(row["dt"]),
                horizon=float(row["horizon"]),
                theta1=initial[:, 0].reshape(resolution, resolution),
                theta2=initial[:, 1].reshape(resolution, resolution),
                initial=initial,
                final=final,
                final_speed=final_speed,
                last_fast_time=np.where(
                    np.isfinite(settling), settling - float(row["dt"]), row["horizon"]
                ),
                settling_time=settling,
                resolved=np.isfinite(settling),
                runtime_seconds=float(row["runtime_seconds"]),
                labels=labels,
            )
        )
    gamma_spectrum_rows = gamma_jacobian_spectra(
        record, gamma_runs, args.jacobian_difference_step
    )
    synchronized_rows, saddle_node = synchronized_branch_spectra(
        record, args.gamma_scan_values, args.jacobian_difference_step
    )
    saddle_scaling_rows, saddle_scaling_fit = saddle_node_scaling(
        synchronized_rows, saddle_node
    )
    normal_form = synchronized_normal_form_coefficients(record)
    if gamma_spectrum_rows:
        save_csv(output / "gamma_jacobian_spectra.csv", gamma_spectrum_rows)
        plot_gamma_spectrum(
            output / "fractal_settling_gamma_spectrum", gamma_spectrum_rows
        )
    if synchronized_rows:
        save_csv(output / "synchronized_branch_spectra.csv", synchronized_rows)
    if saddle_scaling_rows:
        save_csv(output / "saddle_node_scaling.csv", saddle_scaling_rows)
    save_csv(output / "saddle_node_normal_form.csv", [normal_form])

    for gamma, trend in trend_by_gamma.items():
        selected = [
            row
            for row in gamma_spectrum_rows
            if math.isclose(float(row["gamma"]), gamma)
        ]
        trend["terminal_spectral_abscissa"] = (
            max(float(row["spectral_abscissa"]) for row in selected)
            if selected
            else None
        )
    summary["gamma_trend"]["terminal_prototype_jacobian_spectra"] = gamma_spectrum_rows
    summary["gamma_trend"]["synchronized_branch_spectra"] = synchronized_rows
    summary["gamma_trend"]["synchronized_saddle_node"] = saddle_node
    summary["gamma_trend"]["saddle_node_scaling_rows"] = saddle_scaling_rows
    summary["gamma_trend"]["saddle_node_scaling_fit"] = saddle_scaling_fit
    summary["gamma_trend"]["saddle_node_normal_form"] = normal_form
    summary["gamma_trend"]["jacobian_method"] = {
        "coordinates": "three token angles",
        "method": "central finite differences of the full angular vector field",
        "difference_step": args.jacobian_difference_step,
    }
    summary["artifacts"]["gamma_jacobian_spectra_csv"] = "gamma_jacobian_spectra.csv"
    summary["artifacts"]["synchronized_branch_spectra_csv"] = (
        "synchronized_branch_spectra.csv"
    )
    summary["artifacts"]["saddle_node_scaling_csv"] = "saddle_node_scaling.csv"
    summary["artifacts"]["saddle_node_normal_form_csv"] = "saddle_node_normal_form.csv"
    summary["artifacts"]["gamma_spectrum_pdf"] = "fractal_settling_gamma_spectrum.pdf"
    summary["artifacts"]["gamma_spectrum_png"] = "fractal_settling_gamma_spectrum.png"
    stable_candidates = [
        row
        for row in synchronized_rows
        if math.isclose(float(row["gamma"]), args.gamma)
        and int(row["eigenvalue_index_descending_real_part"]) == 0
        and float(row["spectral_abscissa"]) < 0.0
    ]
    unstable_candidates = [
        row
        for row in synchronized_rows
        if math.isclose(float(row["gamma"]), args.gamma)
        and int(row["eigenvalue_index_descending_real_part"]) == 0
        and float(row["spectral_abscissa"]) > 0.0
    ]
    if len(stable_candidates) != 1 or len(unstable_candidates) != 1:
        raise RuntimeError(
            "expected one stable root and one saddle root at selected gamma"
        )
    stable_theta = float(stable_candidates[0]["theta"])
    saddle_theta = float(unstable_candidates[0]["theta"])

    # Endpoint clusters at a finite horizon are not attractor identities near a
    # weak saddle.  Retain the clustering only as a numerical audit and compare
    # every quiet endpoint with the two resolved synchronized roots.
    endpoint_clusters = summary.pop("attractors", {})
    endpoint_clusters["interpretation"] = (
        "Greedy clusters of finite-horizon endpoints. They are not distinct "
        "attractors: the weak synchronized direction splits one slow branch."
    )
    endpoint_clusters["root_theta"] = {
        "companion_saddle": saddle_theta,
        "stable_consensus": stable_theta,
    }
    summary["finite_horizon_endpoint_clusters"] = endpoint_clusters
    for run_summary in summary["run_summaries"]:
        if "attractor_counts" in run_summary:
            run_summary["endpoint_cluster_counts"] = run_summary.pop(
                "attractor_counts"
            )
        if "unmatched_attractor_points" in run_summary:
            run_summary["unmatched_endpoint_cluster_points"] = run_summary.pop(
                "unmatched_attractor_points"
            )

    primary_by_name = {
        row["name"]: row
        for row in summary["run_summaries"]
        if str(row["name"]).startswith("primary_z")
    }
    root_audit_rows: list[dict[str, Any]] = []
    root_uncertainty_rows: list[dict[str, Any]] = []
    strip_rows: list[dict[str, Any]] = []
    for window_data in summary["slice"]["windows"]:
        index = int(window_data["index"])
        name = f"primary_z{index}"
        meta = primary_by_name[name]
        resolution = int(meta["resolution"])
        initial = arrays[f"{name}_initial"]
        final = arrays[f"{name}_final"]
        speed_settling = arrays[f"{name}_settling_time"]
        speed_resolved = np.isfinite(speed_settling)
        stable_distance = np.max(
            np.abs(wrapped_difference(final, np.full((1, 3), stable_theta))), axis=-1
        )
        saddle_distance = np.max(
            np.abs(wrapped_difference(final, np.full((1, 3), saddle_theta))), axis=-1
        )
        closer_to_stable = stable_distance < saddle_distance
        audited_resolved = speed_resolved & closer_to_stable
        audited_settling = np.where(audited_resolved, speed_settling, np.nan)
        audited_labels = np.where(audited_resolved, 0, -1).astype(np.int16)
        audited_run = GridRun(
            name=f"root_audited_z{index}",
            window=Window(
                index,
                float(window_data["center_theta1"]),
                float(window_data["center_theta2"]),
                float(window_data["width"]),
            ),
            resolution=resolution,
            dt=float(meta["dt"]),
            horizon=float(meta["horizon"]),
            theta1=initial[:, 0].reshape(resolution, resolution),
            theta2=initial[:, 1].reshape(resolution, resolution),
            initial=initial,
            final=final,
            final_speed=arrays[f"{name}_terminal_speed"],
            last_fast_time=np.where(
                speed_resolved,
                speed_settling - float(meta["dt"]),
                float(meta["horizon"]),
            ),
            settling_time=audited_settling,
            resolved=audited_resolved,
            runtime_seconds=float(meta["runtime_seconds"]),
            labels=audited_labels,
        )
        root_audit_rows.append(
            {
                "run": name,
                "speed_quiet_points": int(np.sum(speed_resolved)),
                "closer_to_stable_root": int(np.sum(audited_resolved)),
                "quiet_but_closer_to_saddle": int(
                    np.sum(speed_resolved & ~closer_to_stable)
                ),
                "stable_root_theta": stable_theta,
                "saddle_root_theta": saddle_theta,
                "minimum_stable_root_distance": float(np.min(stable_distance)),
                "minimum_saddle_root_distance": float(np.min(saddle_distance)),
            }
        )
        root_uncertainty_rows.extend(
            pair_uncertainty(
                audited_run,
                args.uncertainty_offsets,
                args.delta_times,
                periodic=index == 0,
            )
        )
        strip_rows.extend(
            affine_strip_geometry(audited_run, speed_resolved, speed_settling)
        )
    root_fit_rows: list[dict[str, Any]] = []
    for index in range(args.zoom_levels):
        run_name = f"root_audited_z{index}"
        for delta_time in args.delta_times:
            selected = [
                row
                for row in root_uncertainty_rows
                if row["run"] == run_name
                and math.isclose(float(row["delta_time"]), delta_time)
            ]
            root_fit_rows.append(
                {
                    "run": run_name,
                    "window_index": index,
                    "delta_time": delta_time,
                    **power_law_fit(selected),
                }
            )
    save_csv(output / "root_distance_endpoint_audit.csv", root_audit_rows)
    save_csv(output / "root_audited_uncertainty_scaling.csv", root_uncertainty_rows)
    save_csv(output / "root_audited_uncertainty_fits.csv", root_fit_rows)
    save_csv(output / "censored_strip_geometry.csv", strip_rows)
    summary["root_distance_endpoint_audit"] = {
        "status": (
            "Post-hoc finite-horizon check. A quiet endpoint closer to the stable "
            "synchronized root than to the companion saddle is retained; this is "
            "not a proof of its asymptotic destination."
        ),
        "rows": root_audit_rows,
        "uncertainty_fits": root_fit_rows,
    }
    summary["censored_strip_geometry"] = {
        "status": (
            "Pixel-scale affine fits to the censor-strip edges and the minimum-time "
            "ridge; these are finite-resolution smoothness diagnostics."
        ),
        "rows": strip_rows,
    }
    summary["artifacts"]["root_distance_endpoint_audit_csv"] = (
        "root_distance_endpoint_audit.csv"
    )
    summary["artifacts"]["root_audited_uncertainty_scaling_csv"] = (
        "root_audited_uncertainty_scaling.csv"
    )
    summary["artifacts"]["root_audited_uncertainty_fits_csv"] = (
        "root_audited_uncertainty_fits.csv"
    )
    summary["artifacts"]["censored_strip_geometry_csv"] = (
        "censored_strip_geometry.csv"
    )

    null_rate = float(unstable_candidates[0]["spectral_abscissa"])
    null_rows, null_fits, null_description = smooth_saddle_null(
        null_rate,
        args.uncertainty_offsets,
        args.delta_times,
        args.resolution,
        [float(window["width"]) for window in summary["slice"]["windows"]],
    )
    save_csv(output / "smooth_saddle_null.csv", null_rows)
    save_csv(output / "smooth_saddle_null_fits.csv", null_fits)
    plot_smooth_saddle_null(output / "smooth_saddle_null", null_fits, args.delta_times)
    summary["smooth_saddle_null"] = {
        **null_description,
        "fits": null_fits,
    }
    summary["artifacts"]["smooth_saddle_null_csv"] = "smooth_saddle_null.csv"
    summary["artifacts"]["smooth_saddle_null_fits_csv"] = "smooth_saddle_null_fits.csv"
    summary["artifacts"]["smooth_saddle_null_pdf"] = "smooth_saddle_null.pdf"
    summary["artifacts"]["smooth_saddle_null_png"] = "smooth_saddle_null.png"
    fold_coefficient = -0.5 * float(normal_form["field_second_derivative_theta"])
    fold_rows, fold_fits, fold_description = fold_normal_form_null(
        fold_coefficient,
        args.uncertainty_offsets,
        args.delta_times,
        args.resolution,
        [float(window["width"]) for window in summary["slice"]["windows"]],
    )
    save_csv(output / "fold_normal_form_null.csv", fold_rows)
    save_csv(output / "fold_normal_form_null_fits.csv", fold_fits)
    plot_fold_normal_form_null(
        output / "fold_normal_form_null", fold_fits, args.delta_times
    )
    summary["fold_normal_form_null"] = {
        **fold_description,
        "fits": fold_fits,
    }
    summary["artifacts"]["fold_normal_form_null_csv"] = "fold_normal_form_null.csv"
    summary["artifacts"]["fold_normal_form_null_fits_csv"] = (
        "fold_normal_form_null_fits.csv"
    )
    summary["artifacts"]["fold_normal_form_null_pdf"] = "fold_normal_form_null.pdf"
    summary["artifacts"]["fold_normal_form_null_png"] = "fold_normal_form_null.png"

    critical_theta = float(normal_form["theta_critical"])
    critical_gamma = float(normal_form["gamma_critical"])
    critical_field = PlanarTransformerField.from_record(record, critical_gamma)
    critical_spectrum_rows = []
    for difference_step in (1e-4, 1e-5, 1e-6, 5e-7, 1e-7):
        eigenvalues = np.linalg.eigvals(
            finite_difference_jacobian(
                critical_field, np.full(3, critical_theta), difference_step
            )
        )
        eigenvalues = eigenvalues[np.argsort(eigenvalues.real)[::-1]]
        critical_spectrum_rows.append(
            {
                "difference_step": difference_step,
                **{
                    f"eigenvalue_{index + 1}_real": float(value.real)
                    for index, value in enumerate(eigenvalues)
                },
                **{
                    f"eigenvalue_{index + 1}_imag": float(value.imag)
                    for index, value in enumerate(eigenvalues)
                },
            }
        )
    save_csv(output / "critical_full_jacobian_spectrum.csv", critical_spectrum_rows)
    summary["critical_full_jacobian_spectrum"] = {
        "status": "central finite differences in the full three-angle field",
        "rows": critical_spectrum_rows,
    }
    summary["artifacts"]["critical_full_jacobian_spectrum_csv"] = (
        "critical_full_jacobian_spectrum.csv"
    )
    tangent_audit = transient_tangent_postprocess(record, arrays, summary, args)
    save_csv(output / "transient_tangent_audit.csv", tangent_audit["rows"])
    summary["transient_tangent_audit"] = tangent_audit
    summary["artifacts"]["transient_tangent_audit_csv"] = "transient_tangent_audit.csv"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "gamma_terminal_spectrum_rows": len(gamma_spectrum_rows),
        "synchronized_branch_spectrum_rows": len(synchronized_rows),
        "saddle_node": saddle_node,
        "saddle_node_scaling": saddle_scaling_fit,
        "saddle_node_normal_form": normal_form,
        "smooth_saddle_null_fits": len(null_fits),
        "fold_normal_form_null_fits": len(fold_fits),
        "root_distance_endpoint_audit_rows": len(root_audit_rows),
        "censored_strip_geometry_rows": len(strip_rows),
        "transient_tangent_rows": len(tangent_audit["rows"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/spectral_self_attention/continuous_ode_f2_chaos_antilock_matched.json"
        ),
    )
    parser.add_argument("--source-model-index", type=int, default=1051)
    parser.add_argument("--gamma", type=float, default=1.4348)
    parser.add_argument("--fixed-theta3", type=float, default=1.4202138116039)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--control-resolution", type=int, default=32)
    parser.add_argument("--zoom-levels", type=int, default=4)
    parser.add_argument("--zoom-factor", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--horizon", type=float, default=220.0)
    parser.add_argument("--control-horizon", type=float, default=300.0)
    parser.add_argument("--speed-tolerance", type=float, default=1e-5)
    parser.add_argument("--validation-tail", type=float, default=20.0)
    parser.add_argument("--attractor-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--uncertainty-offsets", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument(
        "--delta-times", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0]
    )
    parser.add_argument("--primary-delta-time", type=float, default=2.0)
    parser.add_argument(
        "--gamma-scan-values",
        type=float,
        nargs="+",
        default=[1.4346, 1.4347, 1.43475, 1.4348, 1.4349, 1.4350, 1.4352],
    )
    parser.add_argument("--gamma-scan-resolution", type=int, default=24)
    parser.add_argument("--gamma-scan-horizon", type=float, default=220.0)
    parser.add_argument("--jacobian-difference-step", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2609121051)
    parser.add_argument(
        "--execution-note",
        type=str,
        default=(
            "Persistent job fractal-settling-atlas-260912. Launch 1 was stopped after 142 "
            "seconds to make attractor clustering local to each gamma. Launch 2 was stopped "
            "after its first 96^2 panel (397 seconds), because 2 of 9216 trajectories were "
            "properly right-censored and the then-loaded code would have rejected censoring. "
            "Launch 2 subsequently completed all trajectories but stopped before writing when "
            "an interpretive terminal-label diagnostic was still fatal. The final run uses "
            "64^2 primary and 32^2 control grids and checkpoints trajectories before analysis; "
            "the four zoom levels, time steps, horizons, speed threshold, and deterministic "
            "zoom rule are unchanged."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spectral_self_attention/fractal_settling_atlas"),
    )
    parser.add_argument("--postprocess-gamma-spectrum", action="store_true")
    args = parser.parse_args()
    if args.resolution < 24 or args.control_resolution < 16:
        parser.error("resolutions are too small for five uncertainty offsets")
    if args.zoom_levels < 2:
        parser.error("at least two zoom levels are required")
    if args.zoom_factor <= 1.0:
        parser.error("zoom factor must exceed one")
    if args.control_horizon <= args.horizon:
        parser.error("control horizon must exceed primary horizon")
    if args.primary_delta_time not in args.delta_times:
        parser.error("primary delta time must be included in delta-times")
    return args


def main() -> None:
    args = parse_args()
    if args.postprocess_gamma_spectrum:
        print(json.dumps(postprocess_gamma_spectrum(args), indent=2))
        return
    result = run_experiment(args)
    print(
        json.dumps(
            {
                "attractors": result["attractors"]["count"],
                "primary_fits": result["primary_uncertainty_fits"],
                "runtime_seconds": result["runtime"]["total_seconds"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
