#!/usr/bin/env python3
"""Variable-coefficient elliptic Bayesian inverse problem and solver benchmark.

The physical model is

    -div(a_z grad u) + reaction*u = m,  u|boundary = 0,

with point observations of u and a low-rank-plus-floor latent source
covariance.  In prior-whitened coordinates the Gauss--Newton/posterior system
is H = I + U U^T.  U is assembled from actual sparse elliptic solves.  The
kernel head uses a prescribed spatial RBF softmax over sensor tokens, followed
by exact QR, optional block-power refinement, and exact Ritz algebra.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pyamg
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .common import check_record, save_csv, save_json


@dataclass
class EllipticContext:
    side: int
    dimension: int
    sensor_count: int
    latent_rank: int
    matrix: sp.csr_matrix
    coefficient: np.ndarray
    coordinates: np.ndarray
    sensor_indices: np.ndarray
    sensor_coordinates: np.ndarray
    latent_basis: np.ndarray
    latent_amplitudes: np.ndarray
    sensitivity: np.ndarray
    common_timings: dict[str, float]
    data_scale: float


@dataclass
class TorchMetric:
    name: str
    basis: torch.Tensor | None = None
    reduced_cholesky: torch.Tensor | None = None
    diagonal_inverse: torch.Tensor | None = None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def torch_timed(
    function: Callable[[], Any],
    device: torch.device,
    repeats: int,
    warmups: int = 3,
) -> tuple[Any, dict[str, float]]:
    result = None
    for _ in range(warmups):
        result = function()
    sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        sync(device)
        samples.append((time.perf_counter_ns() - start) * 1e-6)
    return result, {
        "median_ms": float(np.median(samples)),
        "q25_ms": float(np.quantile(samples, 0.25)),
        "q75_ms": float(np.quantile(samples, 0.75)),
        "minimum_ms": float(np.min(samples)),
    }


def grid_coordinates(side: int) -> np.ndarray:
    axis = (np.arange(side, dtype=np.float64) + 1.0) / (side + 1.0)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def coefficient_field(side: int, seed: int) -> np.ndarray:
    coordinates = grid_coordinates(side)
    x, y = coordinates[:, 0], coordinates[:, 1]
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(scale=[0.34, 0.24, 0.18, 0.12])
    log_field = (
        coefficients[0] * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
        + coefficients[1] * np.cos(np.pi * x) * np.sin(2.0 * np.pi * y)
        + coefficients[2] * np.sin(3.0 * np.pi * x + 0.3) * np.sin(2.0 * np.pi * y)
        + coefficients[3] * np.cos(2.0 * np.pi * x - 0.2) * np.cos(3.0 * np.pi * y)
    )
    return np.exp(log_field).reshape(side, side)


def elliptic_matrix(coefficient: np.ndarray, reaction: float = 0.2) -> sp.csr_matrix:
    side = coefficient.shape[0]
    h = 1.0 / (side + 1.0)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    def harmonic(left: float, right: float) -> float:
        return 2.0 * left * right / (left + right)

    for i in range(side):
        for j in range(side):
            index = i * side + j
            center = coefficient[i, j]
            diagonal = reaction
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < side and 0 <= nj < side:
                    conductance = harmonic(center, coefficient[ni, nj]) / (h * h)
                    rows.append(index)
                    columns.append(ni * side + nj)
                    values.append(-conductance)
                    diagonal += conductance
                else:
                    # Dirichlet boundary with the center-to-boundary conductance.
                    diagonal += center / (h * h)
            rows.append(index)
            columns.append(index)
            values.append(diagonal)
    return sp.csr_matrix((values, (rows, columns)), shape=(side * side, side * side))


def task_latent_basis(side: int, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = grid_coordinates(side)
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.08, 0.92, size=(rank, 2))
    widths = rng.uniform(0.10, 0.24, size=rank)
    fields = []
    for index in range(rank):
        displacement = coordinates - centers[index]
        bump = np.exp(-0.5 * np.sum(displacement * displacement, axis=1) / widths[index] ** 2)
        oscillation = 1.0 + 0.25 * np.cos(
            2.0 * np.pi * (index % 4 + 1) * coordinates[:, index % 2]
        )
        fields.append(bump * oscillation)
    raw = np.column_stack(fields)
    basis, _ = np.linalg.qr(raw, mode="reduced")
    # A deliberately nontrivial latent covariance: several directions carry
    # comparable energy, followed by a mild decay.  A nearly rank-one prior
    # would make every low-rank preconditioner look artificially successful.
    amplitudes = np.geomspace(1.0, 0.45, rank)
    return basis, amplitudes


def select_sensors(side: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dimension = side * side
    strata = int(math.ceil(math.sqrt(count)))
    indices = []
    for i in range(strata):
        for j in range(strata):
            if len(indices) >= count:
                break
            low_i = int(i * side / strata)
            high_i = max(low_i + 1, int((i + 1) * side / strata))
            low_j = int(j * side / strata)
            high_j = max(low_j + 1, int((j + 1) * side / strata))
            ii = int(rng.integers(low_i, min(side, high_i)))
            jj = int(rng.integers(low_j, min(side, high_j)))
            indices.append(ii * side + jj)
    if len(indices) < count:
        remaining = np.setdiff1d(np.arange(dimension), np.asarray(indices), assume_unique=False)
        extra = rng.choice(remaining, size=count - len(indices), replace=False)
        indices.extend(extra.tolist())
    return np.asarray(indices[:count], dtype=np.int64)


def estimate_spectral_norm(matrix: np.ndarray, seed: int, steps: int = 12) -> float:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=matrix.shape[1])
    vector /= np.linalg.norm(vector)
    for _ in range(steps):
        vector = matrix.T @ (matrix @ vector)
        vector /= max(np.linalg.norm(vector), np.finfo(float).tiny)
    return float(np.linalg.norm(matrix @ vector))


def assemble_context(
    side: int,
    sensor_count: int,
    latent_rank: int,
    seed: int,
    floor_amplitude: float,
    target_data_singular: float,
) -> EllipticContext:
    dimension = side * side
    coordinates = grid_coordinates(side)
    coefficient = coefficient_field(side, seed + 17)
    matrix = elliptic_matrix(coefficient)
    sensor_indices = select_sensors(side, sensor_count, seed + 31)
    sensor_coordinates = coordinates[sensor_indices]
    latent_basis, latent_amplitudes = task_latent_basis(side, latent_rank, seed + 47)

    start = time.perf_counter()
    factor = spla.splu(matrix.tocsc())
    sparse_lu_setup_ms = 1000.0 * (time.perf_counter() - start)
    sensor_rhs = np.zeros((dimension, sensor_count), dtype=np.float64)
    sensor_rhs[sensor_indices, np.arange(sensor_count)] = 1.0
    start = time.perf_counter()
    green = factor.solve(sensor_rhs)
    green_solve_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    projected = latent_basis.T @ green
    response_norms = np.linalg.norm(projected, axis=1)
    response_median = float(np.median(response_norms))
    information_balance = response_median / np.maximum(
        response_norms, 0.05 * response_median
    )
    latent_amplitudes = latent_amplitudes * np.clip(information_balance, 0.2, 8.0)
    latent_amplitudes /= latent_amplitudes.max()
    sensitivity = (
        floor_amplitude * green
        + latent_basis @ (latent_amplitudes[:, None] * projected)
    )
    raw_norm = estimate_spectral_norm(sensitivity, seed + 59)
    data_scale = target_data_singular / max(raw_norm, np.finfo(float).tiny)
    sensitivity *= data_scale
    covariance_apply_ms = 1000.0 * (time.perf_counter() - start)
    return EllipticContext(
        side=side,
        dimension=dimension,
        sensor_count=sensor_count,
        latent_rank=latent_rank,
        matrix=matrix,
        coefficient=coefficient,
        coordinates=coordinates,
        sensor_indices=sensor_indices,
        sensor_coordinates=sensor_coordinates,
        latent_basis=latent_basis,
        latent_amplitudes=latent_amplitudes,
        sensitivity=sensitivity,
        common_timings={
            "sparse_lu_setup_ms": sparse_lu_setup_ms,
            "sensor_green_solves_ms": green_solve_ms,
            "latent_covariance_apply_and_scaling_ms": covariance_apply_ms,
            "context_assembly_total_ms": sparse_lu_setup_ms
            + green_solve_ms
            + covariance_apply_ms,
        },
        data_scale=data_scale,
    )


def farthest_landmarks(points: torch.Tensor, count: int) -> torch.Tensor:
    selected = [int(torch.argmin(points[:, 0] + points[:, 1]).item())]
    minimum_distance = torch.full(
        (points.shape[0],), float("inf"), device=points.device, dtype=points.dtype
    )
    for _ in range(1, count):
        displacement = points - points[selected[-1]]
        distance = torch.sum(displacement * displacement, dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        selected.append(int(torch.argmax(minimum_distance).item()))
    return torch.tensor(selected, device=points.device, dtype=torch.long)


def orth_torch(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(matrix, mode="reduced").Q


def make_ritz_metric(name: str, sensitivity: torch.Tensor, basis: torch.Tensor) -> TorchMetric:
    basis = orth_torch(basis)
    projected = sensitivity.T @ basis
    reduced = torch.eye(
        basis.shape[1], device=basis.device, dtype=basis.dtype
    ) + projected.T @ projected
    return TorchMetric(
        name=name,
        basis=basis,
        reduced_cholesky=torch.linalg.cholesky(reduced),
    )


def kernel_metric_builder(
    sensitivity: torch.Tensor,
    sensor_coordinates: torch.Tensor,
    rank: int,
    length_scale: float,
    refinement_steps: int,
) -> TorchMetric:
    landmark_indices = farthest_landmarks(sensor_coordinates, rank)
    landmarks = sensor_coordinates[landmark_indices]
    distance_squared = torch.sum(
        (landmarks[:, None, :] - sensor_coordinates[None, :, :]) ** 2,
        dim=-1,
    )
    # Fixed model-chosen RBF score.  There is no learned temperature.
    attention = torch.softmax(-0.5 * distance_squared / length_scale**2, dim=1)
    basis = orth_torch(sensitivity @ attention.T)
    for _ in range(refinement_steps):
        basis = orth_torch(sensitivity @ (sensitivity.T @ basis))
    return make_ritz_metric("kernel_ritz", sensitivity, basis)


def random_metric_builder(
    sensitivity: torch.Tensor,
    rank: int,
    refinement_steps: int,
    generator: torch.Generator,
) -> TorchMetric:
    sketch = torch.randn(
        sensitivity.shape[1],
        rank,
        device=sensitivity.device,
        dtype=sensitivity.dtype,
        generator=generator,
    )
    basis = orth_torch(sensitivity @ sketch)
    for _ in range(refinement_steps):
        basis = orth_torch(sensitivity @ (sensitivity.T @ basis))
    return make_ritz_metric("randomized_ritz", sensitivity, basis)


def oracle_metric_builder(
    sensitivity: torch.Tensor,
    gram_eigenvalues: torch.Tensor,
    gram_eigenvectors: torch.Tensor,
    rank: int,
) -> TorchMetric:
    selected_values = gram_eigenvalues[-rank:].clamp_min(1e-20)
    selected_vectors = gram_eigenvectors[:, -rank:]
    basis = sensitivity @ (selected_vectors / torch.sqrt(selected_values)[None, :])
    return make_ritz_metric("oracle_ritz", sensitivity, basis)


def global_metric_builder(
    sensitivity: torch.Tensor,
    side: int,
    rank: int,
) -> TorchMetric:
    reference, _ = task_latent_basis(side, rank, seed=49001)
    basis = torch.as_tensor(reference, device=sensitivity.device, dtype=sensitivity.dtype)
    return make_ritz_metric("global_ritz", sensitivity, basis)


def apply_hessian(sensitivity: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return vector + sensitivity @ (sensitivity.T @ vector)


def apply_metric(metric: TorchMetric, vector: torch.Tensor) -> torch.Tensor:
    if metric.diagonal_inverse is not None:
        return metric.diagonal_inverse[:, None] * vector
    if metric.basis is None:
        return vector
    coefficients = metric.basis.T @ vector
    solved = torch.cholesky_solve(coefficients, metric.reduced_cholesky)
    return vector - metric.basis @ coefficients + metric.basis @ solved


def solve_hb(
    sensitivity: torch.Tensor,
    rhs: torch.Tensor,
    metric: TorchMetric,
    depth: int,
    mu: float,
    ell: float,
) -> torch.Tensor:
    root_mu, root_ell = math.sqrt(mu), math.sqrt(ell)
    alpha = 4.0 / (root_mu + root_ell) ** 2
    beta = ((root_ell - root_mu) / (root_ell + root_mu)) ** 2
    previous = torch.zeros_like(rhs)
    current = torch.zeros_like(rhs)
    for _ in range(depth):
        residual = rhs - apply_hessian(sensitivity, current)
        following = current + alpha * apply_metric(metric, residual) + beta * (current - previous)
        previous, current = current, following
    return current


def solve_chebyshev(
    sensitivity: torch.Tensor,
    rhs: torch.Tensor,
    metric: TorchMetric,
    depth: int,
    mu: float,
    ell: float,
) -> torch.Tensor:
    center = 0.5 * (ell + mu)
    radius = 0.5 * (ell - mu)
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = torch.zeros_like(rhs)
    alpha_previous = 0.0
    for step in range(depth):
        if step == 0:
            alpha = 1.0 / center
            beta = 0.0
        else:
            beta = (0.5 * radius * alpha_previous) ** 2
            alpha = 1.0 / (center - beta / alpha_previous)
        direction = apply_metric(metric, residual) + beta * direction
        solution = solution + alpha * direction
        residual = residual - alpha * apply_hessian(sensitivity, direction)
        alpha_previous = alpha
    return solution


def solve_pcg(
    sensitivity: torch.Tensor,
    rhs: torch.Tensor,
    metric: TorchMetric,
    depth: int,
) -> torch.Tensor:
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = apply_metric(metric, residual)
    direction = preconditioned.clone()
    rz = torch.sum(residual * preconditioned, dim=0)
    for _ in range(depth):
        image = apply_hessian(sensitivity, direction)
        denominator = torch.sum(direction * image, dim=0).clamp_min(1e-30)
        alpha = rz / denominator
        solution = solution + direction * alpha[None, :]
        residual = residual - image * alpha[None, :]
        preconditioned = apply_metric(metric, residual)
        following_rz = torch.sum(residual * preconditioned, dim=0)
        beta = following_rz / rz.clamp_min(1e-30)
        direction = preconditioned + direction * beta[None, :]
        rz = following_rz
    return solution


def woodbury_solve(
    sensitivity: torch.Tensor,
    cholesky: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    coefficients = sensitivity.T @ rhs
    return rhs - sensitivity @ torch.cholesky_solve(coefficients, cholesky)


def solver_errors(
    sensitivity: torch.Tensor,
    rhs: torch.Tensor,
    solution: torch.Tensor,
    exact: torch.Tensor,
) -> dict[str, float]:
    residual = rhs - apply_hessian(sensitivity, solution)
    relative_residual = torch.linalg.vector_norm(residual, dim=0) / torch.linalg.vector_norm(rhs, dim=0)
    error = solution - exact
    h_error = torch.sum(error * apply_hessian(sensitivity, error), dim=0)
    h_exact = torch.sum(exact * apply_hessian(sensitivity, exact), dim=0).clamp_min(1e-30)
    relative_energy = torch.sqrt((h_error / h_exact).clamp_min(0.0))
    return {
        "relative_residual_mean": float(relative_residual.mean().item()),
        "relative_residual_max": float(relative_residual.max().item()),
        "relative_energy_mean": float(relative_energy.mean().item()),
        "relative_energy_max": float(relative_energy.max().item()),
    }


def ritz_effective_spectrum(
    gram_eigenvalues: torch.Tensor,
    gram_eigenvectors: torch.Tensor,
    sensitivity: torch.Tensor,
    metric: TorchMetric,
) -> tuple[float, float, float]:
    if metric.basis is None:
        minimum = 1.0
        maximum = 1.0 + float(gram_eigenvalues[-1].item())
        return minimum, maximum, maximum / minimum
    positive = gram_eigenvalues > max(1e-12, 1e-10 * float(gram_eigenvalues[-1].item()))
    values = gram_eigenvalues[positive]
    vectors = gram_eigenvectors[:, positive]
    coordinates = (
        vectors.T @ (sensitivity.T @ metric.basis)
    ) / torch.sqrt(values)[:, None]
    hessian_reduced = torch.diag(1.0 + values)
    inverse_reduced = torch.cholesky_inverse(metric.reduced_cholesky)
    metric_reduced = (
        torch.eye(values.numel(), device=values.device, dtype=values.dtype)
        - coordinates @ coordinates.T
        + coordinates @ inverse_reduced @ coordinates.T
    )
    eigenvalues = torch.linalg.eigvals(metric_reduced @ hessian_reduced).real
    minimum = min(1.0, float(eigenvalues.min().item()))
    maximum = max(1.0, float(eigenvalues.max().item()))
    return minimum, maximum, maximum / minimum


def pde_inner_baselines(
    context: EllipticContext,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rhs = rng.normal(size=context.dimension)
    rhs_norm = np.linalg.norm(rhs)
    rows: list[dict[str, Any]] = []

    start = time.perf_counter()
    lu = spla.splu(context.matrix.tocsc())
    setup_ms = 1000.0 * (time.perf_counter() - start)
    start = time.perf_counter()
    solution = lu.solve(rhs)
    solve_ms = 1000.0 * (time.perf_counter() - start)
    rows.append(
        {
            "method": "sparse_lu",
            "setup_ms": setup_ms,
            "solve_ms": solve_ms,
            "iterations": 1,
            "relative_residual": float(np.linalg.norm(context.matrix @ solution - rhs) / rhs_norm),
        }
    )

    diagonal = context.matrix.diagonal()
    jacobi = spla.LinearOperator(context.matrix.shape, matvec=lambda value: value / diagonal)
    jacobi_iterations = 0

    def jacobi_callback(_: np.ndarray) -> None:
        nonlocal jacobi_iterations
        jacobi_iterations += 1

    start = time.perf_counter()
    jacobi_solution, jacobi_info = spla.cg(
        context.matrix,
        rhs,
        M=jacobi,
        rtol=1e-10,
        atol=0.0,
        maxiter=5000,
        callback=jacobi_callback,
    )
    jacobi_solve_ms = 1000.0 * (time.perf_counter() - start)
    rows.append(
        {
            "method": "jacobi_cg",
            "setup_ms": 0.0,
            "solve_ms": jacobi_solve_ms,
            "iterations": jacobi_iterations,
            "info": jacobi_info,
            "relative_residual": float(
                np.linalg.norm(context.matrix @ jacobi_solution - rhs) / rhs_norm
            ),
        }
    )

    start = time.perf_counter()
    hierarchy = pyamg.smoothed_aggregation_solver(context.matrix, symmetry="symmetric")
    amg_setup_ms = 1000.0 * (time.perf_counter() - start)
    amg_iterations = 0

    def amg_callback(_: np.ndarray) -> None:
        nonlocal amg_iterations
        amg_iterations += 1

    start = time.perf_counter()
    amg_solution, amg_info = spla.cg(
        context.matrix,
        rhs,
        M=hierarchy.aspreconditioner(cycle="V"),
        rtol=1e-10,
        atol=0.0,
        maxiter=500,
        callback=amg_callback,
    )
    amg_solve_ms = 1000.0 * (time.perf_counter() - start)
    rows.append(
        {
            "method": "amg_pcg",
            "setup_ms": amg_setup_ms,
            "solve_ms": amg_solve_ms,
            "iterations": amg_iterations,
            "info": amg_info,
            "relative_residual": float(
                np.linalg.norm(context.matrix @ amg_solution - rhs) / rhs_norm
            ),
        }
    )
    return rows


def run_grid(
    side: int,
    profile: str,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    repeats: int,
    tolerance: float,
) -> dict[str, Any]:
    latent_rank = 8 if profile == "smoke" else 24
    head_rank = latent_rank
    sensor_count = min(side * side // 2, 4 * side)
    context = assemble_context(
        side=side,
        sensor_count=sensor_count,
        latent_rank=latent_rank,
        seed=seed,
        floor_amplitude=0.018,
        target_data_singular=30.0,
    )
    sensitivity = torch.as_tensor(context.sensitivity, device=device, dtype=dtype)
    sensor_coordinates = torch.as_tensor(context.sensor_coordinates, device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 101)
    identity = torch.eye(sensor_count, device=device, dtype=dtype)

    def woodbury_setup() -> torch.Tensor:
        return torch.linalg.cholesky(identity + sensitivity.T @ sensitivity)

    woodbury_cholesky, woodbury_setup_timing = torch_timed(
        woodbury_setup, device, repeats=repeats
    )
    gram = sensitivity.T @ sensitivity
    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(gram)
    effective_rank = float(
        (gram_eigenvalues.sum() ** 2 / (torch.sum(gram_eigenvalues**2).clamp_min(1e-30))).item()
    )

    def kernel_builder() -> TorchMetric:
        return kernel_metric_builder(
            sensitivity,
            sensor_coordinates,
            head_rank,
            length_scale=0.30,
            refinement_steps=1,
        )

    kernel_metric, kernel_setup_timing = torch_timed(kernel_builder, device, repeats=repeats)

    def random_builder() -> TorchMetric:
        return random_metric_builder(sensitivity, head_rank, 1, generator)

    random_metric, random_setup_timing = torch_timed(random_builder, device, repeats=repeats)

    def oracle_builder() -> TorchMetric:
        return oracle_metric_builder(
            sensitivity, gram_eigenvalues, gram_eigenvectors, head_rank
        )

    oracle_metric, oracle_setup_timing = torch_timed(oracle_builder, device, repeats=repeats)
    global_metric = global_metric_builder(sensitivity, side, head_rank)
    identity_metric = TorchMetric(name="identity")
    jacobi_metric = TorchMetric(
        name="jacobi",
        diagonal_inverse=1.0 / (1.0 + torch.sum(sensitivity * sensitivity, dim=1)),
    )
    metrics = {
        "identity": identity_metric,
        "jacobi": jacobi_metric,
        "kernel_ritz": kernel_metric,
        "randomized_ritz": random_metric,
        "oracle_ritz": oracle_metric,
        "global_ritz": global_metric,
    }
    spectra = {}
    for name, metric in metrics.items():
        if name in {"jacobi", "global_ritz"}:
            continue
        spectra[name] = ritz_effective_spectrum(
            gram_eigenvalues, gram_eigenvectors, sensitivity, metric
        )

    head_ablation_rows: list[dict[str, Any]] = []
    for length_scale in (0.04, 0.12, 0.30, 0.60):
        for refinements in (0, 1):
            ablation_metric = kernel_metric_builder(
                sensitivity,
                sensor_coordinates,
                head_rank,
                length_scale=length_scale,
                refinement_steps=refinements,
            )
            minimum, maximum, condition = ritz_effective_spectrum(
                gram_eigenvalues,
                gram_eigenvectors,
                sensitivity,
                ablation_metric,
            )
            head_ablation_rows.append(
                {
                    "side": side,
                    "dimension": context.dimension,
                    "family": "rbf_softmax",
                    "length_scale": length_scale,
                    "refinements": refinements,
                    "minimum": minimum,
                    "maximum": maximum,
                    "condition": condition,
                }
            )
    ablation_generator = torch.Generator(device=device)
    ablation_generator.manual_seed(seed + 707)
    for refinements in (0, 1):
        ablation_metric = random_metric_builder(
            sensitivity, head_rank, refinements, ablation_generator
        )
        minimum, maximum, condition = ritz_effective_spectrum(
            gram_eigenvalues,
            gram_eigenvectors,
            sensitivity,
            ablation_metric,
        )
        head_ablation_rows.append(
            {
                "side": side,
                "dimension": context.dimension,
                "family": "gaussian_random",
                "length_scale": float("nan"),
                "refinements": refinements,
                "minimum": minimum,
                "maximum": maximum,
                "condition": condition,
            }
        )

    maximum_queries = 32 if profile == "smoke" else 64
    rhs = torch.randn(
        context.dimension,
        maximum_queries,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    # Put half of the right-hand-side energy in data-informed directions.
    rhs = rhs + 0.5 * sensitivity @ torch.randn(
        sensor_count,
        maximum_queries,
        generator=generator,
        device=device,
        dtype=dtype,
    ) / math.sqrt(sensor_count)
    exact = woodbury_solve(sensitivity, woodbury_cholesky, rhs)

    depths = [4, 8, 16, 32, 64] if profile == "smoke" else [4, 8, 12, 16, 24, 32, 48, 64, 96]
    solvers: dict[str, tuple[Callable[..., torch.Tensor], TorchMetric, tuple[float, float] | None]] = {
        "identity_pcg": (solve_pcg, identity_metric, None),
        "jacobi_pcg": (solve_pcg, jacobi_metric, None),
        "kernel_hb": (solve_hb, kernel_metric, spectra["kernel_ritz"][:2]),
        "kernel_chebyshev": (solve_chebyshev, kernel_metric, spectra["kernel_ritz"][:2]),
        "kernel_pcg": (solve_pcg, kernel_metric, None),
        "randomized_pcg": (solve_pcg, random_metric, None),
        "oracle_pcg": (solve_pcg, oracle_metric, None),
        "global_pcg": (solve_pcg, global_metric, None),
    }
    accuracy_rows: list[dict[str, Any]] = []
    selected_depths: dict[str, int] = {}
    audit_rhs = rhs[:, : min(8, maximum_queries)]
    audit_exact = exact[:, : min(8, maximum_queries)]
    for name, (solver, metric, interval) in solvers.items():
        selected = depths[-1]
        for depth in depths:
            if interval is None:
                solution = solver(sensitivity, audit_rhs, metric, depth)
            else:
                mu, ell = interval
                solution = solver(
                    sensitivity,
                    audit_rhs,
                    metric,
                    depth,
                    max(1e-6, 0.999 * mu),
                    1.001 * ell,
                )
            errors = solver_errors(sensitivity, audit_rhs, solution, audit_exact)
            accuracy_rows.append(
                {
                    "side": side,
                    "dimension": context.dimension,
                    "method": name,
                    "depth": depth,
                    **errors,
                }
            )
            if errors["relative_residual_max"] <= tolerance and selected == depths[-1]:
                selected = depth
        selected_depths[name] = selected

    setup_rows = [
        {"method": "kernel_ritz", **kernel_setup_timing},
        {"method": "randomized_ritz", **random_setup_timing},
        {"method": "oracle_ritz", **oracle_setup_timing},
        {"method": "woodbury", **woodbury_setup_timing},
    ]
    setup_lookup = {row["method"]: row["median_ms"] for row in setup_rows}
    setup_lookup.update({"identity": 0.0, "jacobi": 0.0, "global_ritz": 0.0})

    runtime_rows: list[dict[str, Any]] = []
    query_counts = [1, 8, 32] if profile == "smoke" else [1, 8, 32, 64]
    for query_count in query_counts:
        selected_rhs = rhs[:, :query_count]
        for name, (solver, metric, interval) in solvers.items():
            depth = selected_depths[name]

            def solve_selected() -> torch.Tensor:
                if interval is None:
                    return solver(sensitivity, selected_rhs, metric, depth)
                mu, ell = interval
                return solver(
                    sensitivity,
                    selected_rhs,
                    metric,
                    depth,
                    max(1e-6, 0.999 * mu),
                    1.001 * ell,
                )

            solution, timing = torch_timed(solve_selected, device, repeats=repeats)
            errors = solver_errors(
                sensitivity, selected_rhs, solution, exact[:, :query_count]
            )
            setup_key = (
                "kernel_ritz"
                if name.startswith("kernel_")
                else "randomized_ritz"
                if name.startswith("randomized_")
                else "oracle_ritz"
                if name.startswith("oracle_")
                else "global_ritz"
                if name.startswith("global_")
                else "jacobi"
                if name.startswith("jacobi_")
                else "identity"
            )
            setup_ms = setup_lookup[setup_key]
            runtime_rows.append(
                {
                    "side": side,
                    "dimension": context.dimension,
                    "sensors": sensor_count,
                    "queries": query_count,
                    "method": name,
                    "depth": depth,
                    "hvp_count": depth,
                    "setup_ms": setup_ms,
                    "cached_solve_ms": timing["median_ms"],
                    "total_ms": setup_ms + timing["median_ms"],
                    **errors,
                }
            )

        def cached_woodbury() -> torch.Tensor:
            return woodbury_solve(sensitivity, woodbury_cholesky, selected_rhs)

        woodbury_solution, timing = torch_timed(cached_woodbury, device, repeats=repeats)
        runtime_rows.append(
            {
                "side": side,
                "dimension": context.dimension,
                "sensors": sensor_count,
                "queries": query_count,
                "method": "woodbury_exact",
                "depth": 1,
                "hvp_count": 0,
                "setup_ms": woodbury_setup_timing["median_ms"],
                "cached_solve_ms": timing["median_ms"],
                "total_ms": woodbury_setup_timing["median_ms"] + timing["median_ms"],
                **solver_errors(
                    sensitivity,
                    selected_rhs,
                    woodbury_solution,
                    exact[:, :query_count],
                ),
            }
        )

    dense_rows: list[dict[str, Any]] = []
    dense_memory_bytes = context.dimension * context.dimension * torch.finfo(dtype).bits // 8
    if context.dimension <= (1024 if profile == "smoke" else 2048):
        dense_identity = torch.eye(context.dimension, device=device, dtype=dtype)

        def dense_setup() -> torch.Tensor:
            return torch.linalg.cholesky(dense_identity + sensitivity @ sensitivity.T)

        dense_cholesky, dense_setup_timing = torch_timed(dense_setup, device, repeats=repeats)
        for query_count in query_counts:
            selected_rhs = rhs[:, :query_count]

            def dense_solve() -> torch.Tensor:
                return torch.cholesky_solve(selected_rhs, dense_cholesky)

            dense_solution, dense_solve_timing = torch_timed(
                dense_solve, device, repeats=repeats
            )
            row = {
                "side": side,
                "dimension": context.dimension,
                "sensors": sensor_count,
                "queries": query_count,
                "method": "dense_cholesky",
                "depth": 1,
                "hvp_count": 0,
                "setup_ms": dense_setup_timing["median_ms"],
                "cached_solve_ms": dense_solve_timing["median_ms"],
                "total_ms": dense_setup_timing["median_ms"] + dense_solve_timing["median_ms"],
                **solver_errors(
                    sensitivity,
                    selected_rhs,
                    dense_solution,
                    exact[:, :query_count],
                ),
            }
            dense_rows.append(row)
            runtime_rows.append(row)

    pde_rows = pde_inner_baselines(context, seed + 211)
    for row in pde_rows:
        row.update({"side": side, "dimension": context.dimension})

    spectral_rows = []
    for name, (minimum, maximum, condition) in spectra.items():
        spectral_rows.append(
            {
                "side": side,
                "dimension": context.dimension,
                "sensors": sensor_count,
                "latent_rank": latent_rank,
                "method": name,
                "minimum": minimum,
                "maximum": maximum,
                "condition": condition,
            }
        )

    memory = {
        "dense_hessian_bytes": dense_memory_bytes,
        "sensitivity_bytes": context.sensitivity.nbytes,
        "kernel_basis_bytes": context.dimension * head_rank * torch.finfo(dtype).bits // 8,
        "woodbury_gram_bytes": sensor_count * sensor_count * torch.finfo(dtype).bits // 8,
    }
    return {
        "context": context,
        "effective_rank": effective_rank,
        "gram_eigenvalues": gram_eigenvalues.detach().cpu().numpy(),
        "accuracy_rows": accuracy_rows,
        "runtime_rows": runtime_rows,
        "setup_rows": [
            {"side": side, "dimension": context.dimension, **row} for row in setup_rows
        ],
        "pde_rows": pde_rows,
        "spectral_rows": spectral_rows,
        "head_ablation_rows": head_ablation_rows,
        "memory": memory,
        "selected_depths": selected_depths,
    }


def serializable_grid(result: dict[str, Any]) -> dict[str, Any]:
    context: EllipticContext = result["context"]
    return {
        "side": context.side,
        "dimension": context.dimension,
        "sensor_count": context.sensor_count,
        "latent_rank": context.latent_rank,
        "coefficient_min": float(context.coefficient.min()),
        "coefficient_max": float(context.coefficient.max()),
        "data_scale": context.data_scale,
        "common_timings": context.common_timings,
        "effective_rank": result["effective_rank"],
        "gram_eigenvalues": result["gram_eigenvalues"],
        "memory": result["memory"],
        "selected_depths": result["selected_depths"],
    }


def evaluate_checks(
    grids: list[dict[str, Any]],
    profile: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    all_spectral = [row for grid in grids for row in grid["spectral_rows"]]
    largest = grids[-1]
    largest_spectral = {row["method"]: row for row in largest["spectral_rows"]}
    identity_condition = largest_spectral["identity"]["condition"]
    kernel_condition = largest_spectral["kernel_ritz"]["condition"]
    checks.extend(
        [
            check_record(
                "elliptic_uniform_positivity",
                min(grid["context"].coefficient.min() for grid in grids) > 0.25,
                min(grid["context"].coefficient.min() for grid in grids),
                "minimum diffusion coefficient > 0.25",
            ),
            check_record(
                "latent_covariance_nontrivial_effective_rank",
                min(grid["effective_rank"] for grid in grids) >= 3.0,
                min(grid["effective_rank"] for grid in grids),
                "posterior data update effective rank >= 3 on every mesh",
            ),
            check_record(
                "kernel_ritz_condition_reduction",
                identity_condition / kernel_condition >= 5.0,
                identity_condition / kernel_condition,
                "identity condition / kernel-Ritz condition >= 5 on largest mesh",
                identity_condition=identity_condition,
                kernel_condition=kernel_condition,
            ),
            check_record(
                "mesh_uniform_kernel_ritz_condition",
                max(row["condition"] for row in all_spectral if row["method"] == "kernel_ritz")
                / min(row["condition"] for row in all_spectral if row["method"] == "kernel_ritz")
                < 2.0,
                max(row["condition"] for row in all_spectral if row["method"] == "kernel_ritz")
                / min(row["condition"] for row in all_spectral if row["method"] == "kernel_ritz"),
                "max/min kernel-Ritz condition across meshes < 2",
            ),
        ]
    )
    kernel_accuracy = [
        row
        for grid in grids
        for row in grid["accuracy_rows"]
        if row["method"] in {"kernel_hb", "kernel_chebyshev", "kernel_pcg"}
        and row["depth"] == grid["selected_depths"][row["method"]]
    ]
    checks.append(
        check_record(
            "kernel_loop_equal_accuracy",
            max(row["relative_residual_max"] for row in kernel_accuracy) <= 1.25 * tolerance,
            max(row["relative_residual_max"] for row in kernel_accuracy),
            "all selected kernel loops achieve <= 1.25 times target residual",
            target=tolerance,
        )
    )
    pde_rows = [row for grid in grids for row in grid["pde_rows"]]
    checks.append(
        check_record(
            "amg_inner_pde_baseline",
            max(row["relative_residual"] for row in pde_rows if row["method"] == "amg_pcg") < 2e-9,
            max(row["relative_residual"] for row in pde_rows if row["method"] == "amg_pcg"),
            "AMG-PCG inner elliptic residual < 2e-9",
        )
    )

    largest_ablations = largest["head_ablation_rows"]
    matched_zero = next(
        row
        for row in largest_ablations
        if row["family"] == "rbf_softmax"
        and row["length_scale"] == 0.30
        and row["refinements"] == 0
    )
    short_zero = next(
        row
        for row in largest_ablations
        if row["family"] == "rbf_softmax"
        and row["length_scale"] == 0.04
        and row["refinements"] == 0
    )
    matched_one = next(
        row
        for row in largest_ablations
        if row["family"] == "rbf_softmax"
        and row["length_scale"] == 0.30
        and row["refinements"] == 1
    )
    oracle_condition = largest_spectral["oracle_ritz"]["condition"]
    checks.extend(
        [
            check_record(
                "model_matched_rbf_feature_advantage",
                short_zero["condition"] / matched_zero["condition"] >= 3.0,
                short_zero["condition"] / matched_zero["condition"],
                "too-short-kernel / model-matched zero-refinement condition >= 3",
                matched_length_scale=0.30,
            ),
            check_record(
                "one_refinement_reaches_oracle_spectrum",
                matched_one["condition"] <= 1.2 * oracle_condition,
                matched_one["condition"] / oracle_condition,
                "matched one-refinement condition / oracle condition <= 1.2",
            ),
        ]
    )

    largest_accuracy = largest["accuracy_rows"]
    kernel_depth_eight = next(
        row
        for row in largest_accuracy
        if row["method"] == "kernel_pcg" and row["depth"] == 8
    )
    global_depth_eight = next(
        row
        for row in largest_accuracy
        if row["method"] == "global_pcg" and row["depth"] == 8
    )
    checks.append(
        check_record(
            "contextual_vs_global_rotated_geometry",
            global_depth_eight["relative_residual_max"]
            > 5.0 * kernel_depth_eight["relative_residual_max"],
            global_depth_eight["relative_residual_max"]
            / max(kernel_depth_eight["relative_residual_max"], 1e-30),
            "global / contextual PCG residual at 8 HVP > 5",
        )
    )

    largest_runtime = largest["runtime_rows"]
    one_query = {row["method"]: row for row in largest_runtime if row["queries"] == 1}
    checks.append(
        check_record(
            "kernel_vs_woodbury_one_query_total_time",
            one_query["kernel_hb"]["total_ms"] < one_query["woodbury_exact"]["total_ms"],
            one_query["woodbury_exact"]["total_ms"] / one_query["kernel_hb"]["total_ms"],
            "measured Woodbury total / kernel-HB total > 1 on largest mesh",
            kernel_total_ms=one_query["kernel_hb"]["total_ms"],
            woodbury_total_ms=one_query["woodbury_exact"]["total_ms"],
        )
    )
    largest_memory = largest["memory"]
    checks.append(
        check_record(
            "dense_system_memory_avoidance",
            largest_memory["dense_hessian_bytes"]
            > 8.0 * largest_memory["sensitivity_bytes"],
            largest_memory["dense_hessian_bytes"]
            / largest_memory["sensitivity_bytes"],
            "dense Hessian memory / matrix-free sensitivity memory > 8",
            dense_bytes=largest_memory["dense_hessian_bytes"],
            sensitivity_bytes=largest_memory["sensitivity_bytes"],
        )
    )
    return checks


def make_plots(outdir: Path, grids: list[dict[str, Any]]) -> None:
    spectral_rows = [row for grid in grids for row in grid["spectral_rows"]]
    runtime_rows = [row for grid in grids for row in grid["runtime_rows"]]
    accuracy_rows = [row for grid in grids for row in grid["accuracy_rows"]]
    pde_rows = [row for grid in grids for row in grid["pde_rows"]]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    for method in ("identity", "kernel_ritz", "randomized_ritz", "oracle_ritz"):
        selected = [row for row in spectral_rows if row["method"] == method]
        axes[0, 0].loglog(
            [row["dimension"] for row in selected],
            [row["condition"] for row in selected],
            "o-",
            label=method,
        )
    axes[0, 0].set(title="Posterior effective spectrum", xlabel="state dimension", ylabel="condition number")

    largest_dimension = max(row["dimension"] for row in accuracy_rows)
    for method in ("identity_pcg", "kernel_hb", "kernel_chebyshev", "kernel_pcg", "oracle_pcg"):
        selected = [
            row for row in accuracy_rows if row["dimension"] == largest_dimension and row["method"] == method
        ]
        axes[0, 1].semilogy(
            [row["depth"] for row in selected],
            [row["relative_residual_max"] for row in selected],
            "o-",
            label=method,
        )
    axes[0, 1].set(title="Equal-HVP convergence", xlabel="Hessian-vector products", ylabel="max relative residual")

    for method in ("kernel_hb", "kernel_pcg", "woodbury_exact", "dense_cholesky"):
        selected = [
            row
            for row in runtime_rows
            if row["dimension"] == largest_dimension and row["method"] == method
        ]
        if selected:
            axes[1, 0].loglog(
                [row["queries"] for row in selected],
                [row["total_ms"] for row in selected],
                "o-",
                label=method,
            )
    axes[1, 0].set(title="Setup + multi-query inference", xlabel="right-hand sides", ylabel="total latency (ms)")

    for method in ("sparse_lu", "jacobi_cg", "amg_pcg"):
        selected = [row for row in pde_rows if row["method"] == method]
        axes[1, 1].loglog(
            [row["dimension"] for row in selected],
            [row["setup_ms"] + row["solve_ms"] for row in selected],
            "o-",
            label=method,
        )
    axes[1, 1].set(title="Inner elliptic solve baselines", xlabel="state dimension", ylabel="setup + solve (ms)")
    for axis in axes.flat:
        axis.grid(which="both", alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outdir / "pde_validation_overview.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_report(outdir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Elliptic Bayesian inverse-problem validation",
        "",
        f"Profile: `{payload['profile']}`. Passed: **{payload['passed']}/{payload['total']}**.",
        "",
        "The RBF score scale is prescribed by the spatial covariance model and is never trained. "
        "All posterior solvers use the same assembled sensitivities and right-hand sides. "
        "Common PDE/context assembly is reported separately from solver-specific setup.",
        "",
        "| Check | Status | Value | Criterion |",
        "|---|---:|---:|---|",
    ]
    for check in payload["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"| {check['name']} | {status} | {float(check['value']):.6g} | {check['criterion']} |"
        )
    lines.extend(
        [
            "",
            "A failed speed check is not hidden: it means that the classical Woodbury or dense "
            "baseline is faster in that measured regime.  Any paper claim must be restricted to "
            "the measured crossover where setup plus solve is actually lower.",
        ]
    )
    (outdir / "PDE_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.profile == "smoke":
        sides = [16, 24]
        repeats = 3
    else:
        sides = [32, 64, 96, 128]
        repeats = 9
    if args.sides:
        sides = sorted({int(value) for value in args.sides.split(",") if value.strip()})
    grids = []
    for index, side in enumerate(sides):
        print(f"assembling elliptic context side={side} ({side * side} unknowns)", flush=True)
        grids.append(
            run_grid(
                side=side,
                profile=args.profile,
                seed=args.seed + 1000 * index,
                device=device,
                dtype=dtype,
                repeats=repeats,
                tolerance=args.tolerance,
            )
        )
    checks = evaluate_checks(grids, args.profile, args.tolerance)
    payload = {
        "profile": args.profile,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "tolerance": args.tolerance,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "pyamg": pyamg.__version__,
            "cpu_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "grids": [serializable_grid(grid) for grid in grids],
        "accuracy_rows": [row for grid in grids for row in grid["accuracy_rows"]],
        "runtime_rows": [row for grid in grids for row in grid["runtime_rows"]],
        "setup_rows": [row for grid in grids for row in grid["setup_rows"]],
        "pde_rows": [row for grid in grids for row in grid["pde_rows"]],
        "spectral_rows": [row for grid in grids for row in grid["spectral_rows"]],
        "head_ablation_rows": [
            row for grid in grids for row in grid["head_ablation_rows"]
        ],
        "checks": checks,
        "passed": sum(bool(check["passed"]) for check in checks),
        "total": len(checks),
    }
    save_csv(args.outdir / "posterior_accuracy.csv", payload["accuracy_rows"])
    save_csv(args.outdir / "posterior_runtime.csv", payload["runtime_rows"])
    save_csv(args.outdir / "posterior_setup.csv", payload["setup_rows"])
    save_csv(args.outdir / "pde_inner_solvers.csv", payload["pde_rows"])
    save_csv(args.outdir / "posterior_spectra.csv", payload["spectral_rows"])
    save_csv(args.outdir / "head_nonlinearity_ablation.csv", payload["head_ablation_rows"])
    save_json(args.outdir / "summary.json", payload)
    make_plots(args.outdir, grids)
    write_report(args.outdir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--sides", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=80001)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = run(args)
    payload["elapsed_seconds"] = time.time() - started
    save_json(args.outdir / "summary.json", payload)
    print(
        f"PDE validation complete: {payload['passed']}/{payload['total']} checks passed; "
        f"summary={args.outdir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
