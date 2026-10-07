"""Spectral atlas for neural quadratic-form training dynamics.

The experiment keeps four objects separate:

* the label-weighted structure generator Y = 4 m^{-1} sum_mu y_mu A_mu;
* structure-basis energies z_k = p_k^T M p_k, where M = W W^T;
* the eigenvalues and eigenvectors of M itself;
* a spiked-Wishart spectrum, for which a BBP statement is meaningful.

The signed and degenerate spectra are tested in the exactly solvable isotropic
NQF flow

    M_dot = Y M + M Y - 8 c M^2.

All categorical inertia/multiplicity classes through a requested dimension are
enumerated.  This is finite categorical completeness, not an enumeration of
the continuous eigenvalue magnitudes.  The BBP section is an external
random-matrix control: BBP is not a theorem in the NQF reference.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq, linear_sum_assignment


@dataclass(frozen=True)
class Config:
    seed: int = 260813335
    max_pattern_dimension: int = 6
    width: int = 32
    replicates: int = 64
    epsilon_min_power: int = 2
    epsilon_max_power: int = 8
    isotropy: float = 0.125
    ignition_fraction: float = 0.5
    spectrum_replicates: int = 24
    spectrum_time_count: int = 25
    attention_samples: int = 48
    attention_dimension: int = 5
    attention_sequence_length: int = 11
    attention_key_dimension: int = 3
    attention_value_dimension: int = 4
    tail_mode_count: int = 16384
    tail_epsilon: float = 1e-12
    bbp_dimensions: tuple[int, ...] = (64, 128, 256)
    bbp_aspect: float = 0.5
    bbp_delta: float = 0.25
    bbp_replicates: tuple[int, ...] = (96, 64, 32)


def integer_partitions(number: int, maximum: int | None = None) -> Iterable[tuple[int, ...]]:
    """Yield nonincreasing integer partitions, including one empty partition of 0."""

    if number == 0:
        yield ()
        return
    if maximum is None or maximum > number:
        maximum = number
    for first in range(maximum, 0, -1):
        for remainder in integer_partitions(number - first, first):
            yield (first,) + remainder


def _block_magnitudes(group_count: int, positive: bool) -> np.ndarray:
    if group_count == 0:
        return np.empty(0)
    if positive:
        return np.geomspace(1.0, 0.25, group_count)
    return -np.geomspace(0.35, 1.0, group_count)


def categorical_patterns(maximum_dimension: int) -> list[dict[str, object]]:
    """Enumerate inertia and within-sign eigenvalue-multiplicity classes."""

    patterns: list[dict[str, object]] = []
    pattern_index = 0
    for dimension in range(1, maximum_dimension + 1):
        for positive_count in range(dimension + 1):
            for negative_count in range(dimension - positive_count + 1):
                zero_count = dimension - positive_count - negative_count
                for positive_partition in integer_partitions(positive_count):
                    for negative_partition in integer_partitions(negative_count):
                        positive_values = _block_magnitudes(
                            len(positive_partition), positive=True
                        )
                        negative_values = _block_magnitudes(
                            len(negative_partition), positive=False
                        )
                        eigenvalues: list[float] = []
                        for multiplicity, value in zip(
                            positive_partition, positive_values
                        ):
                            eigenvalues.extend([float(value)] * multiplicity)
                        eigenvalues.extend([0.0] * zero_count)
                        for multiplicity, value in zip(
                            negative_partition, negative_values
                        ):
                            eigenvalues.extend([float(value)] * multiplicity)
                        pattern_index += 1
                        patterns.append(
                            {
                                "pattern_id": f"p{dimension}_{pattern_index:04d}",
                                "dimension": dimension,
                                "positive_count": positive_count,
                                "zero_count": zero_count,
                                "negative_count": negative_count,
                                "positive_partition": positive_partition,
                                "negative_partition": negative_partition,
                                "eigenvalues": tuple(eigenvalues),
                            }
                        )
    return patterns


def symmetric_basis(dimension: int) -> np.ndarray:
    """Frobenius-orthonormal basis of real symmetric matrices."""

    matrices: list[np.ndarray] = []
    for row in range(dimension):
        matrix = np.zeros((dimension, dimension))
        matrix[row, row] = 1.0
        matrices.append(matrix)
    for row in range(dimension):
        for column in range(row + 1, dimension):
            matrix = np.zeros((dimension, dimension))
            matrix[row, column] = 1.0 / math.sqrt(2.0)
            matrix[column, row] = 1.0 / math.sqrt(2.0)
            matrices.append(matrix)
    return np.asarray(matrices)


def isotropic_dataset(target_generator: np.ndarray, isotropy: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct A_mu,y_mu satisfying the exact isotropic NQF assumption."""

    dimension = target_generator.shape[0]
    basis = symmetric_basis(dimension)
    sample_count = len(basis)
    scale = math.sqrt(isotropy * sample_count)
    structure = scale * basis
    coordinates = np.einsum("aij,ij->a", basis, target_generator)
    targets = sample_count * coordinates / (4.0 * scale)
    return structure, targets


def isotropy_residual(structure: np.ndarray, isotropy: float) -> float:
    dimension = structure.shape[1]
    observed = np.einsum("aij,akl->ijkl", structure, structure) / len(structure)
    identity = np.eye(dimension)
    expected = 0.5 * isotropy * (
        np.einsum("ik,jl->ijkl", identity, identity)
        + np.einsum("il,jk->ijkl", identity, identity)
    )
    return float(np.max(np.abs(observed - expected)))


def generator_residual(
    structure: np.ndarray, targets: np.ndarray, target_generator: np.ndarray
) -> float:
    reconstructed = 4.0 * np.einsum("a,aij->ij", targets, structure) / len(targets)
    return float(np.max(np.abs(reconstructed - target_generator)))


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix, triangular = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    return matrix * signs


def isotropic_solution(
    generator_eigenvalues: np.ndarray,
    generator_eigenvectors: np.ndarray,
    initial_order_parameter: np.ndarray,
    isotropy: float,
    time_value: float,
) -> np.ndarray:
    """Stable evaluation of the exact full-rank isotropic NQF solution."""

    basis_initial = (
        generator_eigenvectors.T
        @ initial_order_parameter
        @ generator_eigenvectors
    )
    inverse_initial = np.linalg.inv(basis_initial)
    decay = np.exp(-generator_eigenvalues * time_value)
    twice = 2.0 * generator_eigenvalues * time_value
    resummed = np.empty_like(generator_eigenvalues)
    nonzero = np.abs(generator_eigenvalues) > 1e-13
    resummed[nonzero] = -np.expm1(-twice[nonzero]) / (
        2.0 * generator_eigenvalues[nonzero]
    )
    resummed[~nonzero] = time_value
    inverse_at_time = (
        decay[:, None] * inverse_initial * decay[None, :]
        + 8.0 * isotropy * np.diag(resummed)
    )
    basis_result = np.linalg.solve(
        inverse_at_time, np.eye(len(generator_eigenvalues))
    )
    result = generator_eigenvectors @ basis_result @ generator_eigenvectors.T
    return 0.5 * (result + result.T)


def commuting_mode(
    generator_eigenvalue: float,
    initial_value: float,
    isotropy: float,
    times: np.ndarray,
) -> np.ndarray:
    """Exact scalar mode for M(0) commuting with Y."""

    gamma = float(generator_eigenvalue)
    times = np.asarray(times, dtype=float)
    if abs(gamma) < 1e-13:
        return initial_value / (1.0 + 8.0 * isotropy * initial_value * times)
    denominator = (
        4.0 * isotropy * initial_value
        + (gamma - 4.0 * isotropy * initial_value)
        * np.exp(-2.0 * gamma * times)
    )
    return gamma * initial_value / denominator


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows supplied for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_pattern_atlas(
    config: Config, output_directory: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    patterns = categorical_patterns(config.max_pattern_dimension)
    catalog_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    maximum_isotropy_error = 0.0
    maximum_generator_error = 0.0
    epsilon = 1e-7
    fit_times = np.linspace(0.0, 0.2, 9)

    for pattern in patterns:
        eigenvalues = np.asarray(pattern["eigenvalues"], dtype=float)
        dimension = int(pattern["dimension"])
        catalog_rows.append(
            {
                "pattern_id": pattern["pattern_id"],
                "dimension": dimension,
                "positive_count": pattern["positive_count"],
                "zero_count": pattern["zero_count"],
                "negative_count": pattern["negative_count"],
                "positive_partition": json.dumps(pattern["positive_partition"]),
                "negative_partition": json.dumps(pattern["negative_partition"]),
                "generator_eigenvalues": json.dumps(eigenvalues.tolist()),
            }
        )

        # One exact dataset check per categorical class.  Orthogonal rotations are
        # unnecessary for this algebraic identity and would only add roundoff.
        generator = np.diag(eigenvalues)
        structure, targets = isotropic_dataset(generator, config.isotropy)
        maximum_isotropy_error = max(
            maximum_isotropy_error,
            isotropy_residual(structure, config.isotropy),
        )
        maximum_generator_error = max(
            maximum_generator_error,
            generator_residual(structure, targets, generator),
        )

        for mode, gamma in enumerate(eigenvalues):
            values = commuting_mode(gamma, epsilon, config.isotropy, fit_times)
            fitted_rate = float(
                np.polyfit(fit_times, np.log(np.maximum(values, 1e-300)), 1)[0]
            )
            if gamma > 0:
                asymptotic_state = gamma / (4.0 * config.isotropy)
                fate = "active"
            elif gamma < 0:
                asymptotic_state = 0.0
                fate = "exponentially_suppressed"
            else:
                asymptotic_state = 0.0
                fate = "algebraic_zero_rate"
            diagnostic_rows.append(
                {
                    "pattern_id": pattern["pattern_id"],
                    "dimension": dimension,
                    "mode": mode,
                    "generator_eigenvalue": float(gamma),
                    "predicted_initial_log_growth": float(2.0 * gamma),
                    "fitted_initial_log_growth": fitted_rate,
                    "absolute_growth_error": abs(fitted_rate - 2.0 * gamma),
                    "initial_M_eigenvalue": epsilon,
                    "initial_mode_overlap": 1.0,
                    "predicted_asymptotic_M_eigenvalue": float(asymptotic_state),
                    "fate": fate,
                }
            )

    write_csv(output_directory / "pattern_catalog.csv", catalog_rows)
    write_csv(output_directory / "pattern_mode_diagnostics.csv", diagnostic_rows)
    checks = {
        "maximum_isotropy_identity_error": maximum_isotropy_error,
        "maximum_generator_reconstruction_error": maximum_generator_error,
        "maximum_initial_growth_error": max(
            float(row["absolute_growth_error"]) for row in diagnostic_rows
        ),
    }
    return catalog_rows, diagnostic_rows, checks


def initialization_bases(
    gaussian: np.ndarray,
) -> dict[str, np.ndarray]:
    """Matched Gaussian, row-norm-fixed, and row-Stiefel directions."""

    dimension, width = gaussian.shape
    gaussian_base = gaussian / math.sqrt(width)
    row_norm = np.linalg.norm(gaussian, axis=1, keepdims=True)
    row_fixed = gaussian / np.maximum(row_norm, 1e-300)
    q_matrix, _ = np.linalg.qr(gaussian.T, mode="reduced")
    row_stiefel = q_matrix.T
    return {
        "gaussian": gaussian_base,
        "row_fixed": row_fixed,
        "row_stiefel": row_stiefel,
    }


def projected_mode(
    time_value: float,
    mode: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    initial_order_parameter: np.ndarray,
    isotropy: float,
) -> float:
    result = isotropic_solution(
        eigenvalues,
        eigenvectors,
        initial_order_parameter,
        isotropy,
        time_value,
    )
    direction = eigenvectors[:, mode]
    return float(direction @ result @ direction)


def ignition_time(
    mode: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    initial_order_parameter: np.ndarray,
    isotropy: float,
    fraction: float,
) -> float:
    gamma = float(eigenvalues[mode])
    if gamma <= 0:
        return math.nan
    threshold = fraction * gamma / (4.0 * isotropy)

    def residual(time_value: float) -> float:
        return projected_mode(
            time_value,
            mode,
            eigenvalues,
            eigenvectors,
            initial_order_parameter,
            isotropy,
        ) - threshold

    upper = max(2.0, (math.log(1.0 / np.trace(initial_order_parameter)) + 12.0) / (2.0 * gamma))
    upper = max(upper, 2.0)
    while residual(upper) < 0.0:
        upper *= 1.5
        if upper > 1e4:
            raise RuntimeError("failed to bracket an ignition time")
    return float(brentq(residual, 0.0, upper, xtol=2e-10, rtol=2e-12))


def run_ignition_regression(
    config: Config, output_directory: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    eigenvalues = np.asarray([1.0, 0.7, 0.7, 0.25, 0.0, -0.4])
    dimension = len(eigenvalues)
    eigenvectors = np.eye(dimension)
    epsilons = 10.0 ** (
        -np.arange(config.epsilon_min_power, config.epsilon_max_power + 1)
    )
    raw_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    clocks: dict[tuple[str, int, int], list[float]] = {}

    for replicate in range(config.replicates):
        rng = np.random.default_rng(config.seed + 100_003 + 1009 * replicate)
        raw = rng.normal(size=(dimension, config.width))
        bases = initialization_bases(raw)
        for initialization, base in bases.items():
            for epsilon in epsilons:
                weights = math.sqrt(float(epsilon)) * base
                initial = weights @ weights.T
                for mode, gamma in enumerate(eigenvalues):
                    if gamma <= 0:
                        continue
                    clock = ignition_time(
                        mode,
                        eigenvalues,
                        eigenvectors,
                        initial,
                        config.isotropy,
                        config.ignition_fraction,
                    )
                    epsilon_index = int(round(-math.log10(float(epsilon))))
                    clocks.setdefault((initialization, replicate, mode), []).append(clock)
                    raw_rows.append(
                        {
                            "initialization": initialization,
                            "replicate": replicate,
                            "epsilon": float(epsilon),
                            "epsilon_power": epsilon_index,
                            "weight_entry_scale_gaussian": math.sqrt(
                                float(epsilon) / config.width
                            ),
                            "mode": mode,
                            "generator_eigenvalue": float(gamma),
                            "predicted_log_epsilon_slope": 1.0 / (2.0 * gamma),
                            "ignition_time": clock,
                        }
                    )

    logarithmic_scale = np.log(1.0 / epsilons)
    for initialization in ("gaussian", "row_fixed", "row_stiefel"):
        for mode, gamma in enumerate(eigenvalues):
            if gamma <= 0:
                continue
            slopes: list[float] = []
            intercepts: list[float] = []
            for replicate in range(config.replicates):
                values = np.asarray(clocks[(initialization, replicate, mode)])
                slope, intercept = np.polyfit(logarithmic_scale, values, 1)
                slopes.append(float(slope))
                intercepts.append(float(intercept))
            theory = 1.0 / (2.0 * gamma)
            regression_rows.append(
                {
                    "initialization": initialization,
                    "mode": mode,
                    "generator_eigenvalue": float(gamma),
                    "multiplicity": int(np.sum(np.isclose(eigenvalues, gamma))),
                    "replicates": config.replicates,
                    "mean_fitted_slope": float(np.mean(slopes)),
                    "slope_standard_deviation": float(np.std(slopes, ddof=1)),
                    "predicted_slope": theory,
                    "relative_slope_error": abs(float(np.mean(slopes)) - theory) / theory,
                    "mean_intercept": float(np.mean(intercepts)),
                    "intercept_standard_deviation": float(np.std(intercepts, ddof=1)),
                }
            )

        # The two gamma=0.7 modes test a repeated eigenvalue.  Their clock gap
        # may have an O(1) random intercept but no log(1/epsilon) slope.
        gap_slopes: list[float] = []
        for replicate in range(config.replicates):
            first = np.asarray(clocks[(initialization, replicate, 1)])
            second = np.asarray(clocks[(initialization, replicate, 2)])
            slope = np.polyfit(logarithmic_scale, first - second, 1)[0]
            gap_slopes.append(float(slope))
        gap_rows.append(
            {
                "initialization": initialization,
                "first_mode": 1,
                "second_mode": 2,
                "shared_generator_eigenvalue": 0.7,
                "mean_gap_slope": float(np.mean(gap_slopes)),
                "gap_slope_standard_deviation": float(np.std(gap_slopes, ddof=1)),
                "predicted_gap_slope": 0.0,
            }
        )

    write_csv(output_directory / "ignition_times.csv", raw_rows)
    write_csv(output_directory / "ignition_regressions.csv", regression_rows)
    write_csv(output_directory / "degenerate_clock_gaps.csv", gap_rows)
    return raw_rows, regression_rows, gap_rows


def assignment_by_overlap(
    order_parameter: np.ndarray, generator_eigenvectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(order_parameter)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    overlaps = np.abs(generator_eigenvectors.T @ eigenvectors) ** 2
    rows, columns = linear_sum_assignment(-overlaps)
    assigned_mode = np.full(len(eigenvalues), -1, dtype=int)
    assigned_overlap = np.zeros(len(eigenvalues))
    for mode, rank in zip(rows, columns):
        assigned_mode[rank] = mode
        assigned_overlap[rank] = overlaps[mode, rank]
    return eigenvalues, assigned_mode, assigned_overlap


def run_spectrum_trajectories(
    config: Config, output_directory: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    examples = {
        "mixed_distinct": np.asarray([1.0, 0.65, 0.25, 0.0, -0.35, -0.7]),
        "signed_repeated": np.asarray([0.8, 0.8, 0.0, 0.0, -0.5, -0.5]),
    }
    spectrum_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    epsilon = 1e-5

    for example_index, (example, eigenvalues) in enumerate(examples.items()):
        dimension = len(eigenvalues)
        rotation_rng = np.random.default_rng(config.seed + 200_003 + example_index)
        eigenvectors = random_orthogonal(dimension, rotation_rng)
        slowest_positive = float(np.min(eigenvalues[eigenvalues > 0]))
        maximum_time = (math.log(1.0 / epsilon) + 3.0) / (2.0 * slowest_positive)
        times = np.linspace(0.0, maximum_time, config.spectrum_time_count)

        for replicate in range(config.spectrum_replicates):
            rng = np.random.default_rng(
                config.seed + 210_003 + 10_007 * example_index + 1009 * replicate
            )
            raw = rng.normal(size=(dimension, config.width))
            for initialization, basis_weights in initialization_bases(raw).items():
                # The bases are defined in the eigenbasis of Y, then rotated.
                weights = math.sqrt(epsilon) * eigenvectors @ basis_weights
                initial = weights @ weights.T
                for time_value in times:
                    result = isotropic_solution(
                        eigenvalues,
                        eigenvectors,
                        initial,
                        config.isotropy,
                        float(time_value),
                    )
                    actual, assigned_modes, assigned_overlaps = assignment_by_overlap(
                        result, eigenvectors
                    )
                    eigenvalue_tolerance = 1e-10 * max(
                        1.0, float(np.max(np.abs(actual)))
                    )
                    for rank, actual_value in enumerate(actual):
                        assigned = int(assigned_modes[rank])
                        M_eigenvector_identifiable = bool(
                            np.sum(
                                np.abs(actual - actual_value)
                                <= eigenvalue_tolerance
                            )
                            == 1
                        )
                        generator_mode_identifiable = bool(
                            np.sum(
                                np.isclose(
                                    eigenvalues,
                                    eigenvalues[assigned],
                                    rtol=1e-12,
                                    atol=1e-12,
                                )
                            )
                            == 1
                        )
                        spectrum_rows.append(
                            {
                                "example": example,
                                "initialization": initialization,
                                "replicate": replicate,
                                "epsilon": epsilon,
                                "time": float(time_value),
                                "M_eigenvalue_rank": rank,
                                "M_eigenvalue": float(actual_value),
                                "assigned_generator_mode": assigned,
                                "assigned_generator_eigenvalue": float(
                                    eigenvalues[assigned]
                                ),
                                "squared_eigenvector_overlap": float(
                                    assigned_overlaps[rank]
                                ),
                                "M_eigenvector_identifiable": M_eigenvector_identifiable,
                                "generator_mode_identifiable": generator_mode_identifiable,
                            }
                        )

                    for gamma in np.unique(eigenvalues):
                        mode_indices = np.flatnonzero(np.isclose(eigenvalues, gamma))
                        ascending_values, actual_vectors = np.linalg.eigh(result)
                        weights_by_vector = np.sum(
                            np.abs(eigenvectors[:, mode_indices].T @ actual_vectors) ** 2,
                            axis=0,
                        )
                        selected = np.argsort(weights_by_vector)[-len(mode_indices) :]
                        selected_vectors = actual_vectors[:, selected]
                        unselected = np.setdiff1d(
                            np.arange(dimension), selected, assume_unique=False
                        )
                        if len(unselected) == 0:
                            subspace_identifiable = True
                        else:
                            separating_gap = float(
                                np.min(
                                    np.abs(
                                        ascending_values[selected, None]
                                        - ascending_values[unselected][None, :]
                                    )
                                )
                            )
                            subspace_identifiable = separating_gap > eigenvalue_tolerance
                        projector_overlap = np.linalg.norm(
                            eigenvectors[:, mode_indices].T @ selected_vectors,
                            ord="fro",
                        ) ** 2 / len(mode_indices)
                        overlap_rows.append(
                            {
                                "example": example,
                                "initialization": initialization,
                                "replicate": replicate,
                                "epsilon": epsilon,
                                "time": float(time_value),
                                "generator_eigenvalue": float(gamma),
                                "multiplicity": len(mode_indices),
                                "normalized_subspace_overlap": float(projector_overlap),
                                "subspace_overlap_identifiable": bool(
                                    subspace_identifiable
                                ),
                            }
                        )

    write_csv(output_directory / "M_spectrum_trajectories.csv", spectrum_rows)
    write_csv(output_directory / "generator_subspace_overlaps.csv", overlap_rows)
    return spectrum_rows, overlap_rows


def bipartite_structure(block: np.ndarray, left_zeros: int = 0) -> np.ndarray:
    left, right = block.shape
    matrix = np.zeros((left_zeros + left + right, left_zeros + left + right))
    start = left_zeros
    matrix[start : start + left, start + left :] = 0.5 * block
    matrix[start + left :, start : start + left] = 0.5 * block.T
    return matrix


def full_attention_structure(
    context: np.ndarray,
    key_dimension: int,
    value_dimension: int,
    normalized: bool,
) -> np.ndarray:
    dimension, sequence_length = context.shape
    context_average = np.mean(context, axis=1)
    vector = context_average if normalized else sequence_length * context_average
    block = np.kron(vector[:, None], np.eye(value_dimension))
    query_key_zeros = 2 * key_dimension * dimension
    return bipartite_structure(block, left_zeros=query_key_zeros)


def qk_attention_structure(
    context: np.ndarray,
    query: np.ndarray,
    fixed_value_readout: np.ndarray,
    key_dimension: int,
    normalized: bool,
) -> np.ndarray:
    dimension, sequence_length = context.shape
    average = np.mean(context, axis=1)
    second_moment = context @ context.T / sequence_length
    if normalized:
        covariance = second_moment - np.outer(average, average)
        transported = covariance @ fixed_value_readout
    else:
        transported = sequence_length * second_moment @ fixed_value_readout
    elementary = np.outer(query, transported)
    block = np.kron(elementary, np.eye(key_dimension)) / math.sqrt(key_dimension)
    return bipartite_structure(block)


def spectral_rows(
    matrix: np.ndarray,
    architecture: str,
    normalization: str,
    object_kind: str,
    sample: int,
) -> list[dict[str, object]]:
    eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    rows: list[dict[str, object]] = []
    for rank, value in enumerate(eigenvalues):
        if value > tolerance:
            sign = "positive"
        elif value < -tolerance:
            sign = "negative"
        else:
            sign = "zero"
        rows.append(
            {
                "architecture": architecture,
                "normalization": normalization,
                "object": object_kind,
                "sample": sample,
                "eigenvalue_rank": rank,
                "eigenvalue": float(value),
                "sign": sign,
            }
        )
    return rows


def run_attention_spectra(
    config: Config, output_directory: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(config.seed + 300_003)
    dimension = config.attention_dimension
    sample_count = config.attention_samples
    contexts = rng.normal(
        size=(sample_count, dimension, config.attention_sequence_length)
    )
    queries = rng.normal(size=(sample_count, dimension))
    readout = rng.normal(size=dimension)
    readout /= np.linalg.norm(readout)
    labels = rng.normal(size=sample_count)
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for architecture in ("full_mha", "qk_only"):
        for normalized in (True, False):
            normalization = "softmax" if normalized else "unnormalized_exp"
            structures: list[np.ndarray] = []
            for sample in range(sample_count):
                if architecture == "full_mha":
                    matrix = full_attention_structure(
                        contexts[sample],
                        config.attention_key_dimension,
                        config.attention_value_dimension,
                        normalized,
                    )
                else:
                    matrix = qk_attention_structure(
                        contexts[sample],
                        queries[sample],
                        readout,
                        config.attention_key_dimension,
                        normalized,
                    )
                structures.append(matrix)
                if sample < 4:
                    rows.extend(
                        spectral_rows(
                            matrix,
                            architecture,
                            normalization,
                            "sample_structure_A",
                            sample,
                        )
                    )
            structure_array = np.asarray(structures)
            generator = 4.0 * np.einsum("a,aij->ij", labels, structure_array) / sample_count
            generator_rows = spectral_rows(
                generator,
                architecture,
                normalization,
                "label_weighted_generator_Y",
                -1,
            )
            rows.extend(generator_rows)
            values = np.linalg.eigvalsh(generator)
            tolerance = 1e-10 * max(1.0, float(np.max(np.abs(values))))
            positive = values[values > tolerance]
            negative = values[values < -tolerance]
            zero_count = int(np.sum(np.abs(values) <= tolerance))
            pair_count = min(len(positive), len(negative))
            pairing_error = 0.0
            if pair_count:
                pairing_error = float(
                    np.max(
                        np.abs(
                            np.sort(positive)[-pair_count:]
                            + np.sort(negative)[:pair_count][::-1]
                        )
                    )
                )
            summary_rows.append(
                {
                    "architecture": architecture,
                    "normalization": normalization,
                    "matrix_dimension": len(values),
                    "positive_count": len(positive),
                    "zero_count": zero_count,
                    "negative_count": len(negative),
                    "spectral_radius": float(np.max(np.abs(values))),
                    "positive_negative_pairing_error": pairing_error,
                }
            )

    write_csv(output_directory / "attention_structure_spectra.csv", rows)
    write_csv(output_directory / "attention_spectrum_summary.csv", summary_rows)
    return rows, summary_rows


def logistic_fraction(
    rates: np.ndarray,
    plateaus: np.ndarray,
    initial: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    log_ratio = np.log(np.maximum(plateaus / initial - 1.0, 1e-300))
    argument = log_ratio[None, :] - times[:, None] * rates[None, :]
    return 1.0 / (1.0 + np.exp(np.clip(argument, -700.0, 700.0)))


def run_tail_tests(
    config: Config, output_directory: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mode = np.arange(1, config.tail_mode_count + 1, dtype=float)
    logarithm = math.log(1.0 / config.tail_epsilon)
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cases = [
        ("power_law", 2.0, 0.8),
        ("exponential_rate_control", 2.0, math.nan),
        ("flat_rate_control", 2.0, 0.0),
    ]

    for case, alpha_one, alpha_two in cases:
        target_strength = mode ** (-alpha_one)
        if case == "power_law":
            rates = mode ** (-alpha_two)
            beta = alpha_one - alpha_two
            tau = np.geomspace(16.0**alpha_two, 256.0**alpha_two, 100)
            predicted_slope = -(alpha_one - 1.0) / alpha_two
        elif case == "exponential_rate_control":
            rates = np.exp(-(mode - 1.0) / 128.0)
            beta = alpha_one
            tau = np.exp(np.linspace(0.5, 5.0, 100))
            predicted_slope = math.nan
        else:
            rates = np.ones_like(mode)
            beta = alpha_one
            tau = np.geomspace(0.5, 2.0, 100)
            predicted_slope = math.nan
        plateaus = target_strength / rates
        initial = config.tail_epsilon * mode ** (-beta)
        fractions = logistic_fraction(rates, plateaus, initial, tau * logarithm)
        excess = np.sum(target_strength[None, :] * (1.0 - fractions) ** 2, axis=1)
        local_slope = np.gradient(np.log(excess), np.log(tau))
        fitted_slope = float(np.polyfit(np.log(tau), np.log(excess), 1)[0])
        midpoint = len(tau) // 2
        first_slope = float(
            np.polyfit(np.log(tau[:midpoint]), np.log(excess[:midpoint]), 1)[0]
        )
        second_slope = float(
            np.polyfit(np.log(tau[midpoint:]), np.log(excess[midpoint:]), 1)[0]
        )
        summary_rows.append(
            {
                "case": case,
                "mode_count": config.tail_mode_count,
                "epsilon": config.tail_epsilon,
                "alpha_1": alpha_one,
                "alpha_2": (
                    float(alpha_two) if np.isfinite(alpha_two) else None
                ),
                "fitted_log_log_slope": fitted_slope,
                "predicted_log_log_slope": (
                    float(predicted_slope)
                    if np.isfinite(predicted_slope)
                    else None
                ),
                "first_half_slope": first_slope,
                "second_half_slope": second_slope,
                "slope_drift": second_slope - first_slope,
            }
        )
        for index, tau_value in enumerate(tau):
            curve_rows.append(
                {
                    "case": case,
                    "tau": float(tau_value),
                    "excess_loss": float(excess[index]),
                    "local_log_log_slope": float(local_slope[index]),
                }
            )

    write_csv(output_directory / "tail_scaling_curves.csv", curve_rows)
    write_csv(output_directory / "tail_scaling_summary.csv", summary_rows)
    return curve_rows, summary_rows


def bbp_predictions(aspect: float, theta: float) -> tuple[float, float]:
    edge = (1.0 + math.sqrt(aspect)) ** 2
    if theta <= math.sqrt(aspect):
        return edge, 0.0
    outlier = (1.0 + theta) * (1.0 + aspect / theta)
    overlap = (1.0 - aspect / theta**2) / (1.0 + aspect / theta)
    return outlier, overlap


def run_bbp(
    config: Config, output_directory: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    relative_times = np.asarray([0.0, 0.4, 0.7, 0.9, 1.0, 1.1, 1.4, 2.0])

    for size_index, dimension in enumerate(config.bbp_dimensions):
        width = int(round(dimension / config.bbp_aspect))
        aspect = dimension / width
        repetitions = config.bbp_replicates[
            min(size_index, len(config.bbp_replicates) - 1)
        ]
        critical_time = math.log1p(math.sqrt(aspect)) / (2.0 * config.bbp_delta)
        times = relative_times * critical_time
        rng = np.random.default_rng(config.seed + 400_003 + dimension)

        # Reuse each Gaussian matrix across time: this is one linearized NQF
        # trajectory rather than unrelated covariance samples.
        for replicate in range(repetitions):
            raw = rng.normal(size=(dimension, width)) / math.sqrt(width)
            for time_value, relative_time in zip(times, relative_times):
                population_spike = math.exp(2.0 * config.bbp_delta * time_value)
                theta = population_spike - 1.0
                evolved = raw.copy()
                evolved[0] *= math.sqrt(population_spike)
                covariance = evolved @ evolved.T
                values, vectors = eigh(
                    covariance,
                    subset_by_index=[dimension - 2, dimension - 1],
                )
                top = float(values[-1])
                second = float(values[-2])
                top_vector = vectors[:, -1]
                predicted_top, predicted_overlap = bbp_predictions(aspect, theta)
                rows.append(
                    {
                        "ensemble": "gaussian_wishart",
                        "dimension": dimension,
                        "width": width,
                        "aspect_p_over_d": aspect,
                        "replicate": replicate,
                        "time": float(time_value),
                        "time_over_critical": float(relative_time),
                        "population_spike_excess_theta": theta,
                        "bbp_threshold_sqrt_aspect": math.sqrt(aspect),
                        "top_M_eigenvalue_normalized": top,
                        "second_M_eigenvalue_normalized": second,
                        "predicted_top_branch": predicted_top,
                        "squared_spike_overlap": float(top_vector[0] ** 2),
                        "predicted_squared_overlap": predicted_overlap,
                    }
                )

        # A row-Stiefel initialization has M(0)=epsilon I.  Its normalized
        # spectrum is the population spectrum itself and has no BBP threshold.
        for time_value, relative_time in zip(times, relative_times):
            population_spike = math.exp(2.0 * config.bbp_delta * time_value)
            theta = population_spike - 1.0
            predicted_top, predicted_overlap = bbp_predictions(aspect, theta)
            rows.append(
                {
                    "ensemble": "row_stiefel",
                    "dimension": dimension,
                    "width": width,
                    "aspect_p_over_d": aspect,
                    "replicate": 0,
                    "time": float(time_value),
                    "time_over_critical": float(relative_time),
                    "population_spike_excess_theta": theta,
                    "bbp_threshold_sqrt_aspect": math.sqrt(aspect),
                    "top_M_eigenvalue_normalized": population_spike,
                    "second_M_eigenvalue_normalized": 1.0,
                    "predicted_top_branch": predicted_top,
                    "squared_spike_overlap": 1.0 if theta > 0 else 1.0 / dimension,
                    "predicted_squared_overlap": predicted_overlap,
                }
            )

        dimension_rows = [
            row
            for row in rows
            if row["dimension"] == dimension
            and row["ensemble"] == "gaussian_wishart"
        ]
        for relative_time in relative_times:
            selected = [
                row
                for row in dimension_rows
                if np.isclose(row["time_over_critical"], relative_time)
            ]
            summary_rows.append(
                {
                    "dimension": dimension,
                    "width": width,
                    "aspect_p_over_d": aspect,
                    "replicates": repetitions,
                    "time_over_critical": float(relative_time),
                    "mean_top_eigenvalue": float(
                        np.mean([row["top_M_eigenvalue_normalized"] for row in selected])
                    ),
                    "standard_error_top_eigenvalue": float(
                        np.std(
                            [row["top_M_eigenvalue_normalized"] for row in selected],
                            ddof=1,
                        )
                        / math.sqrt(repetitions)
                    ),
                    "predicted_top_branch": float(selected[0]["predicted_top_branch"]),
                    "mean_squared_spike_overlap": float(
                        np.mean([row["squared_spike_overlap"] for row in selected])
                    ),
                    "standard_error_squared_overlap": float(
                        np.std(
                            [row["squared_spike_overlap"] for row in selected],
                            ddof=1,
                        )
                        / math.sqrt(repetitions)
                    ),
                    "predicted_squared_overlap": float(
                        selected[0]["predicted_squared_overlap"]
                    ),
                }
            )

    write_csv(output_directory / "bbp_dynamic_samples.csv", rows)
    write_csv(output_directory / "bbp_dynamic_summary.csv", summary_rows)
    return rows, summary_rows


def make_figure(
    output_directory: Path,
    pattern_rows: Sequence[dict[str, object]],
    ignition_rows: Sequence[dict[str, object]],
    overlap_rows: Sequence[dict[str, object]],
    tail_rows: Sequence[dict[str, object]],
    bbp_summary: Sequence[dict[str, object]],
    config: Config,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 8.2), constrained_layout=True)
    palette = {1.0: "#1B6CA8", 0.7: "#6A4C93", 0.25: "#C44E52"}

    predicted = np.asarray(
        [float(row["predicted_initial_log_growth"]) for row in pattern_rows]
    )
    fitted = np.asarray(
        [float(row["fitted_initial_log_growth"]) for row in pattern_rows]
    )
    axes[0, 0].scatter(predicted, fitted, s=8, alpha=0.25, color="#2A6F97")
    limit = max(np.max(np.abs(predicted)), np.max(np.abs(fitted))) * 1.05
    axes[0, 0].plot([-limit, limit], [-limit, limit], color="black", linewidth=1.0)
    axes[0, 0].set(
        xlabel="predicted rate $2\\gamma_k$",
        ylabel="fitted rate of $\\log z_k$",
        title="All sign and multiplicity classes",
    )

    for initialization, linestyle, marker in (
        ("gaussian", "-", "o"),
        ("row_stiefel", "--", "s"),
    ):
        for gamma in (1.0, 0.7, 0.25):
            selected = [
                row
                for row in ignition_rows
                if row["initialization"] == initialization
                and np.isclose(float(row["generator_eigenvalue"]), gamma)
            ]
            by_epsilon: dict[float, list[float]] = {}
            for row in selected:
                by_epsilon.setdefault(float(row["epsilon"]), []).append(
                    float(row["ignition_time"])
                )
            epsilon_values = np.asarray(sorted(by_epsilon, reverse=True))
            means = np.asarray(
                [np.mean(by_epsilon[value]) for value in epsilon_values]
            )
            axes[0, 1].plot(
                np.log(1.0 / epsilon_values),
                means,
                linestyle=linestyle,
                marker=marker,
                markersize=3.5,
                linewidth=1.4,
                color=palette[gamma],
                label=rf"$\gamma={gamma:g}$, {initialization}",
            )
    axes[0, 1].set(
        xlabel="$\\log(1/\\epsilon)$",
        ylabel="half-plateau time",
        title="Logarithmic ignition clocks",
    )
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)

    selected_overlap = [
        row
        for row in overlap_rows
        if row["example"] == "mixed_distinct"
        and np.isclose(float(row["generator_eigenvalue"]), 1.0)
        and bool(row["subspace_overlap_identifiable"])
    ]
    for initialization, color in (
        ("gaussian", "#C44E52"),
        ("row_fixed", "#4C956C"),
        ("row_stiefel", "#1B6CA8"),
    ):
        group = [row for row in selected_overlap if row["initialization"] == initialization]
        times = sorted({float(row["time"]) for row in group})
        means = [
            np.mean(
                [
                    float(row["normalized_subspace_overlap"])
                    for row in group
                    if np.isclose(float(row["time"]), time_value)
                ]
            )
            for time_value in times
        ]
        axes[0, 2].plot(times, means, label=initialization, color=color, linewidth=1.8)
    axes[0, 2].set(
        xlabel="gradient-flow time",
        ylabel="overlap with top eigenspace of $Y$",
        ylim=(-0.02, 1.02),
        title="$M$ eigenvectors align during growth",
    )
    axes[0, 2].legend(frameon=False, fontsize=8)

    largest_dimension = max(config.bbp_dimensions)
    selected_bbp = [
        row for row in bbp_summary if row["dimension"] == largest_dimension
    ]
    relative_time = np.asarray([float(row["time_over_critical"]) for row in selected_bbp])
    mean_top = np.asarray([float(row["mean_top_eigenvalue"]) for row in selected_bbp])
    top_error = np.asarray(
        [float(row["standard_error_top_eigenvalue"]) for row in selected_bbp]
    )
    predicted_top = np.asarray(
        [float(row["predicted_top_branch"]) for row in selected_bbp]
    )
    axes[1, 0].errorbar(
        relative_time,
        mean_top,
        yerr=2.0 * top_error,
        fmt="o",
        markersize=4,
        color="#C44E52",
        label="Gaussian $M$",
    )
    axes[1, 0].plot(relative_time, predicted_top, color="black", linewidth=1.5, label="BBP limit")
    axes[1, 0].axvline(1.0, color="0.45", linestyle="--", linewidth=1.0)
    axes[1, 0].set(
        xlabel="$t/t_{\\rm BBP}$",
        ylabel="largest normalized eigenvalue",
        title=f"Dynamic BBP spectrum ($p={largest_dimension}$)",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    mean_overlap = np.asarray(
        [float(row["mean_squared_spike_overlap"]) for row in selected_bbp]
    )
    overlap_error = np.asarray(
        [float(row["standard_error_squared_overlap"]) for row in selected_bbp]
    )
    predicted_overlap = np.asarray(
        [float(row["predicted_squared_overlap"]) for row in selected_bbp]
    )
    axes[1, 1].errorbar(
        relative_time,
        mean_overlap,
        yerr=2.0 * overlap_error,
        fmt="o",
        markersize=4,
        color="#1B6CA8",
        label="Gaussian $M$",
    )
    axes[1, 1].plot(relative_time, predicted_overlap, color="black", linewidth=1.5, label="BBP limit")
    axes[1, 1].axvline(1.0, color="0.45", linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        xlabel="$t/t_{\\rm BBP}$",
        ylabel="squared spike overlap",
        ylim=(-0.03, 1.03),
        title="Eigenvector transition",
    )
    axes[1, 1].legend(frameon=False, fontsize=8)

    for case, label, color in (
        ("power_law", "power-law rates", "#1B6CA8"),
        ("exponential_rate_control", "exponential-rate control", "#C44E52"),
        ("flat_rate_control", "flat-rate control", "#4C956C"),
    ):
        selected = [row for row in tail_rows if row["case"] == case]
        tau = np.asarray([float(row["tau"]) for row in selected])
        loss = np.asarray([float(row["excess_loss"]) for row in selected])
        axes[1, 2].loglog(tau, loss / loss[0], color=color, linewidth=1.8, label=label)
    axes[1, 2].set(
        xlabel="rescaled time $\\tau$",
        ylabel="excess loss / initial value",
        title="Tail assumption and controls",
    )
    axes[1, 2].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
    figure.savefig(output_directory / "nqf_training_spectrum_atlas.pdf", bbox_inches="tight")
    figure.savefig(
        output_directory / "nqf_training_spectrum_atlas.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def summarize(
    config: Config,
    runtime_seconds: float,
    catalog_rows: Sequence[dict[str, object]],
    pattern_checks: dict[str, float],
    ignition_regressions: Sequence[dict[str, object]],
    gap_rows: Sequence[dict[str, object]],
    attention_summary: Sequence[dict[str, object]],
    tail_summary: Sequence[dict[str, object]],
    bbp_summary: Sequence[dict[str, object]],
) -> dict[str, object]:
    maximum_ignition_error = max(
        float(row["relative_slope_error"]) for row in ignition_regressions
    )
    maximum_gap_slope = max(abs(float(row["mean_gap_slope"])) for row in gap_rows)
    maximum_attention_pairing_error = max(
        float(row["positive_negative_pairing_error"])
        for row in attention_summary
    )
    power_law = next(row for row in tail_summary if row["case"] == "power_law")
    tail_relative_error = abs(
        float(power_law["fitted_log_log_slope"])
        - float(power_law["predicted_log_log_slope"])
    ) / abs(float(power_law["predicted_log_log_slope"]))
    largest_dimension = max(config.bbp_dimensions)
    largest_rows = [
        row for row in bbp_summary if row["dimension"] == largest_dimension
    ]
    null_row = next(row for row in largest_rows if np.isclose(row["time_over_critical"], 0.0))
    subcritical_row = next(row for row in largest_rows if np.isclose(row["time_over_critical"], 0.7))
    supercritical_row = next(row for row in largest_rows if np.isclose(row["time_over_critical"], 2.0))
    null_top_relative_error = abs(
        float(null_row["mean_top_eigenvalue"])
        - float(null_row["predicted_top_branch"])
    ) / float(null_row["predicted_top_branch"])
    supercritical_top_relative_error = abs(
        float(supercritical_row["mean_top_eigenvalue"])
        - float(supercritical_row["predicted_top_branch"])
    ) / float(supercritical_row["predicted_top_branch"])
    supercritical_overlap_error = abs(
        float(supercritical_row["mean_squared_spike_overlap"])
        - float(supercritical_row["predicted_squared_overlap"])
    )

    checks = {
        "categorical_class_count_is_280_through_p6": (
            config.max_pattern_dimension != 6 or len(catalog_rows) == 280
        ),
        "isotropic_frame_is_exact": pattern_checks["maximum_isotropy_identity_error"] < 2e-14,
        "generator_reconstruction_is_exact": pattern_checks[
            "maximum_generator_reconstruction_error"
        ]
        < 2e-14,
        "signed_initial_growth_rates_match": pattern_checks[
            "maximum_initial_growth_error"
        ]
        < 2e-6,
        "ignition_slopes_match": maximum_ignition_error < 0.025,
        "degenerate_gap_has_no_log_slope": maximum_gap_slope < 0.02,
        "attention_spectra_have_bipartite_pairing": maximum_attention_pairing_error
        < 2e-12,
        "power_law_tail_slope_matches": tail_relative_error < 0.035,
        "bbp_null_has_no_outlier": null_top_relative_error < 0.08,
        "bbp_subcritical_overlap_is_small": float(
            subcritical_row["mean_squared_spike_overlap"]
        )
        < 0.08,
        "bbp_supercritical_eigenvalue_matches": supercritical_top_relative_error < 0.08,
        "bbp_supercritical_overlap_matches": supercritical_overlap_error < 0.10,
    }
    return {
        "config": asdict(config),
        "runtime_seconds": runtime_seconds,
        "scope": {
            "categorical_completeness": (
                "All inertia and within-sign multiplicity partitions through the "
                "configured dimension; continuous eigenvalue magnitudes are not enumerated."
            ),
            "nqf_status": (
                "Exact for the synthetic isotropic and orthogonal-feature NQFs; "
                "local when transferred to a nonlinear neural module."
            ),
            "bbp_status": (
                "External spiked-Wishart control in the linearized NQF regime; "
                "BBP is not asserted by the NQF paper."
            ),
        },
        "counts": {
            "categorical_patterns": len(catalog_rows),
            "ignition_regression_rows": len(ignition_regressions),
            "attention_generator_spectra": len(attention_summary),
            "bbp_summary_rows": len(bbp_summary),
        },
        "errors": {
            **pattern_checks,
            "maximum_ignition_relative_slope_error": maximum_ignition_error,
            "maximum_degenerate_gap_slope": maximum_gap_slope,
            "maximum_attention_pairing_error": maximum_attention_pairing_error,
            "power_law_tail_relative_slope_error": tail_relative_error,
            "bbp_null_top_relative_error": null_top_relative_error,
            "bbp_supercritical_top_relative_error": supercritical_top_relative_error,
            "bbp_supercritical_overlap_absolute_error": supercritical_overlap_error,
        },
        "ignition_regressions": list(ignition_regressions),
        "degenerate_clock_gaps": list(gap_rows),
        "attention_spectrum_summary": list(attention_summary),
        "tail_scaling_summary": list(tail_summary),
        "bbp_selected_rows": {
            "null": null_row,
            "subcritical": subcritical_row,
            "supercritical": supercritical_row,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def run(config: Config, output_directory: Path) -> dict[str, object]:
    start = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)
    catalog, pattern_rows, pattern_checks = run_pattern_atlas(config, output_directory)
    ignition_rows, ignition_regressions, gap_rows = run_ignition_regression(
        config, output_directory
    )
    _, overlap_rows = run_spectrum_trajectories(config, output_directory)
    _, attention_summary = run_attention_spectra(config, output_directory)
    tail_rows, tail_summary = run_tail_tests(config, output_directory)
    _, bbp_summary = run_bbp(config, output_directory)
    make_figure(
        output_directory,
        pattern_rows,
        ignition_rows,
        overlap_rows,
        tail_rows,
        bbp_summary,
        config,
    )
    runtime = time.perf_counter() - start
    summary = summarize(
        config,
        runtime,
        catalog,
        pattern_checks,
        ignition_regressions,
        gap_rows,
        attention_summary,
        tail_summary,
        bbp_summary,
    )
    with (output_directory / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    metadata = {
        "command_seed": config.seed,
        "initialization_scale_convention": (
            "epsilon is the scale of M=WW^T. Gaussian entries have standard "
            "deviation sqrt(epsilon/width)."
        ),
        "paired_initializations": (
            "For each replicate one Gaussian direction is rescaled across every "
            "epsilon and transformed into the row-fixed and row-Stiefel controls."
        ),
        "bbp_aspect_convention": "p/dimension-of-width = p/d, not p/training-samples",
        "overlap_convention": (
            "Individual eigenvector overlaps are flagged as non-identifiable at "
            "degenerate eigenvalues. Generator eigenspace comparisons use the "
            "normalized projector overlap and carry a separate identifiability flag."
        ),
        "outputs": sorted(path.name for path in output_directory.iterdir()),
    }
    with (output_directory / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_integer_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spectral_self_attention/nqf_training_spectrum_atlas"),
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--max-pattern-dimension", type=int, default=Config.max_pattern_dimension)
    parser.add_argument("--width", type=int, default=Config.width)
    parser.add_argument("--replicates", type=int, default=Config.replicates)
    parser.add_argument("--spectrum-replicates", type=int, default=Config.spectrum_replicates)
    parser.add_argument("--tail-mode-count", type=int, default=Config.tail_mode_count)
    parser.add_argument("--bbp-dimensions", type=parse_integer_tuple, default=Config.bbp_dimensions)
    parser.add_argument("--bbp-replicates", type=parse_integer_tuple, default=Config.bbp_replicates)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        max_pattern_dimension=args.max_pattern_dimension,
        width=args.width,
        replicates=args.replicates,
        spectrum_replicates=args.spectrum_replicates,
        tail_mode_count=args.tail_mode_count,
        bbp_dimensions=args.bbp_dimensions,
        bbp_replicates=args.bbp_replicates,
    )
    summary = run(config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.no_strict and not summary["all_checks_pass"]:
        raise SystemExit("one or more spectral-atlas checks failed")


if __name__ == "__main__":
    main()
