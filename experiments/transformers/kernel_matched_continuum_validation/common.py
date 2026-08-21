"""Shared numerical and reporting utilities for the validation campaign."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        return super().default(value)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, cls=NumpyJSONEncoder)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_ints(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("expected a nonempty comma-separated integer list")
    return values


def parse_floats(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("expected a nonempty comma-separated float list")
    return values


def loglog_fit(x: Iterable[float], y: Iterable[float]) -> dict[str, float]:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(x_array) & np.isfinite(y_array) & (x_array > 0) & (y_array > 0)
    lx = np.log(x_array[mask])
    ly = np.log(y_array[mask])
    if lx.size < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "slope_se": float("nan"),
            "r2": float("nan"),
            "points": int(lx.size),
        }
    design = np.column_stack([np.ones_like(lx), lx])
    coefficients, *_ = np.linalg.lstsq(design, ly, rcond=None)
    fitted = design @ coefficients
    residual = ly - fitted
    ss_residual = float(residual @ residual)
    centered = ly - ly.mean()
    ss_total = float(centered @ centered)
    dof = max(1, lx.size - 2)
    covariance = (ss_residual / dof) * np.linalg.inv(design.T @ design)
    return {
        "slope": float(coefficients[1]),
        "intercept": float(coefficients[0]),
        "slope_se": float(math.sqrt(max(0.0, covariance[1, 1]))),
        "r2": float(1.0 - ss_residual / max(ss_total, np.finfo(float).tiny)),
        "points": int(lx.size),
    }


def bootstrap_mean_ci(
    values: Iterable[float],
    rng: np.random.Generator,
    replicates: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if array.size == 1:
        value = float(array[0])
        return {"mean": value, "ci_low": value, "ci_high": value}
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }


def orthonormalize(matrix: np.ndarray, rank: int | None = None) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(matrix, dtype=np.float64), mode="reduced")
    if rank is None:
        rank = q.shape[1]
    diagonal = np.abs(np.diag(r))
    numerical_rank = int(np.count_nonzero(diagonal > 1e-12 * max(1.0, diagonal.max(initial=0.0))))
    if numerical_rank < rank:
        raise np.linalg.LinAlgError(
            f"feature matrix has rank {numerical_rank}, smaller than requested {rank}"
        )
    return q[:, :rank]


def projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left @ left.T - right @ right.T, ord=2))


def ritz_inverse_metric(hessian: np.ndarray, basis: np.ndarray) -> np.ndarray:
    dimension = hessian.shape[0]
    reduced = basis.T @ hessian @ basis
    return (
        np.eye(dimension)
        - basis @ basis.T
        + basis @ np.linalg.solve(reduced, basis.T)
    )


def symmetric_preconditioned_spectrum(
    hessian: np.ndarray, metric: np.ndarray
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    return np.linalg.eigvalsh(root @ hessian @ root)


def hb_parameters(mu: float, ell: float) -> tuple[float, float, float]:
    if not (0 < mu <= ell):
        raise ValueError(f"invalid interval [{mu}, {ell}]")
    root_mu = math.sqrt(mu)
    root_ell = math.sqrt(ell)
    q = (root_ell - root_mu) / (root_ell + root_mu)
    return 4.0 / (root_ell + root_mu) ** 2, q * q, q


def hb_residual(eigenvalues: np.ndarray, depth: int, mu: float, ell: float) -> np.ndarray:
    alpha, beta, _ = hb_parameters(mu, ell)
    previous = np.ones_like(eigenvalues, dtype=np.float64)
    current = np.ones_like(eigenvalues, dtype=np.float64)
    for _ in range(depth):
        following = (1.0 + beta - alpha * eigenvalues) * current - beta * previous
        previous, current = current, following
    return current


def chebyshev_residual(
    eigenvalues: np.ndarray, depth: int, mu: float, ell: float
) -> np.ndarray:
    if depth == 0:
        return np.ones_like(eigenvalues, dtype=np.float64)
    center = 0.5 * (ell + mu)
    radius = 0.5 * (ell - mu)
    if radius == 0:
        return np.zeros_like(eigenvalues, dtype=np.float64)
    mapped = (center - eigenvalues) / radius
    denominator_argument = center / radius
    # arccosh evaluation avoids overflow at the positive denominator argument.
    denominator = math.cosh(depth * math.acosh(denominator_argument))
    numerator = np.polynomial.chebyshev.chebval(
        mapped, [0.0] * depth + [1.0]
    )
    return numerator / denominator


def timed(
    function: Callable[[], Any],
    repeats: int = 10,
    warmups: int = 3,
) -> tuple[Any, dict[str, float]]:
    result = None
    for _ in range(warmups):
        result = function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - start) * 1e-6)
    ordered = sorted(samples)
    return result, {
        "median_ms": float(statistics.median(ordered)),
        "q25_ms": float(np.quantile(ordered, 0.25)),
        "q75_ms": float(np.quantile(ordered, 0.75)),
        "min_ms": float(min(ordered)),
    }


def check_record(
    name: str,
    passed: bool,
    value: float | int | str,
    criterion: str,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "criterion": criterion,
        **metadata,
    }


def geometric_midpoint_grid(size: int, warp: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Nonuniform midpoint rule on [0,1] with exact cell weights."""
    edges_u = np.linspace(0.0, 1.0, size + 1)

    def map_coordinate(u: np.ndarray) -> np.ndarray:
        return u + warp * np.sin(2.0 * np.pi * u) / (2.0 * np.pi)

    edges = map_coordinate(edges_u)
    nodes = map_coordinate(0.5 * (edges_u[:-1] + edges_u[1:]))
    weights = np.diff(edges)
    return nodes, weights


def rbf_kernel(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    squared_distance = (left[:, None] - right[None, :]) ** 2
    return np.exp(-0.5 * squared_distance / (length_scale * length_scale))
