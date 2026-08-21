"""Exact characterizations and numerical discovery of all equilibrium classes.

The spectral-Gram equations implemented here are necessary and sufficient.  The
planar multi-start solver is only a discovery/audit aid for fixed numerical data;
it is not used as a proof of global completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .simulator import (
    attention_weights,
    eigenspace_groups,
    normalize,
    state_tangent_jacobian,
    vector_field,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EquilibriumCertificate:
    """Residuals of equivalent equilibrium formulations."""

    normalized_field: float
    multiplier_equation: float
    spectral_gram_equation: float
    sphere_constraint: float
    multipliers: FloatArray


@dataclass(frozen=True)
class SpectralGramSystemCertificate:
    """Feasibility diagnostics for the exhaustive PSD formulation."""

    equation_residual: float
    diagonal_residual: float
    symmetry_residual: float
    minimum_eigenvalue: float
    ranks: tuple[int, ...]
    rank_violations: tuple[int, ...]


def unnormalized_attention_matrix(
    x: FloatArray, eigenvalues: ArrayLike, beta: float
) -> FloatArray:
    """Symmetric entrywise-exponential score matrix R."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    scores = np.einsum("id,jd,d->ij", x, x, eigenvalues, optimize=True)
    logits = beta * scores
    shift = np.max(logits)
    return np.exp(logits - shift)


def spectral_gram_matrices(
    x: FloatArray, eigenvalues: ArrayLike, tol: float = 1e-10
) -> tuple[list[float], list[FloatArray]]:
    """Return one token Gram matrix for each distinct eigenspace."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    values = []
    grams = []
    for group in eigenspace_groups(eigenvalues, tol=tol):
        coordinates = x[:, group]
        values.append(float(eigenvalues[group[0]]))
        grams.append(coordinates @ coordinates.T)
    return values, grams


def evaluate_spectral_gram_system(
    distinct_eigenvalues: ArrayLike,
    grams: list[FloatArray],
    beta: float,
    *,
    eigenspace_dimensions: ArrayLike | None = None,
    tol: float = 1e-9,
) -> SpectralGramSystemCertificate:
    """Evaluate every constraint in the necessary-and-sufficient PSD system."""
    values = np.asarray(distinct_eigenvalues, dtype=float)
    if len(grams) != values.size or not grams:
        raise ValueError("one Gram matrix is required per distinct eigenvalue")
    matrices = [np.asarray(gram, dtype=float) for gram in grams]
    n_tokens = matrices[0].shape[0]
    if any(matrix.shape != (n_tokens, n_tokens) for matrix in matrices):
        raise ValueError("all Gram matrices must be square with the same shape")

    symmetry_residual = max(
        float(np.linalg.norm(matrix - matrix.T, ord="fro")) for matrix in matrices
    )
    symmetric = [(matrix + matrix.T) / 2.0 for matrix in matrices]
    spectra = [np.linalg.eigvalsh(matrix) for matrix in symmetric]
    minimum_eigenvalue = min(float(np.min(spectrum)) for spectrum in spectra)
    ranks = tuple(int(np.sum(spectrum > tol)) for spectrum in spectra)
    if eigenspace_dimensions is None:
        rank_violations = tuple(0 for _ in ranks)
    else:
        dimensions = np.asarray(eigenspace_dimensions, dtype=int)
        if dimensions.size != len(ranks):
            raise ValueError("one eigenspace dimension is required per Gram matrix")
        rank_violations = tuple(
            max(rank - int(dimension), 0)
            for rank, dimension in zip(ranks, dimensions, strict=True)
        )

    diagonal = np.sum([np.diag(matrix) for matrix in symmetric], axis=0)
    diagonal_residual = float(np.max(np.abs(diagonal - 1.0)))
    h_matrix = sum(
        value * matrix for value, matrix in zip(values, symmetric, strict=True)
    )
    logits = beta * h_matrix
    logits -= np.max(logits)
    r_matrix = np.exp(logits)
    multiplier_matrix = np.diag(np.diag(r_matrix @ h_matrix))
    equation_residual = max(
        float(
            np.linalg.norm(
                (value * r_matrix - multiplier_matrix) @ matrix,
                ord="fro",
            )
        )
        for value, matrix in zip(values, symmetric, strict=True)
    )
    return SpectralGramSystemCertificate(
        equation_residual=equation_residual,
        diagonal_residual=diagonal_residual,
        symmetry_residual=symmetry_residual,
        minimum_eigenvalue=minimum_eigenvalue,
        ranks=ranks,
        rank_violations=rank_violations,
    )


def factor_spectral_grams(
    grams: list[FloatArray],
    eigenspace_dimensions: ArrayLike,
    *,
    tol: float = 1e-9,
) -> FloatArray:
    """Reconstruct token coordinates from feasible spectral Gram matrices."""
    dimensions = np.asarray(eigenspace_dimensions, dtype=int)
    if len(grams) != dimensions.size or np.any(dimensions < 1):
        raise ValueError("invalid eigenspace dimensions")
    factors = []
    n_tokens = np.asarray(grams[0]).shape[0]
    for gram, dimension in zip(grams, dimensions, strict=True):
        matrix = np.asarray(gram, dtype=float)
        if matrix.shape != (n_tokens, n_tokens):
            raise ValueError("all Gram matrices must have the same square shape")
        spectrum, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        positive = spectrum > tol
        if np.sum(positive) > dimension or np.min(spectrum) < -tol:
            raise ValueError("Gram matrix violates PSD or rank constraints")
        factor = vectors[:, positive] * np.sqrt(spectrum[positive])[None, :]
        factor = np.pad(factor, ((0, 0), (0, dimension - factor.shape[1])))
        factors.append(factor)
    return np.concatenate(factors, axis=1)


def equilibrium_certificate(
    x: FloatArray,
    eigenvalues: ArrayLike,
    beta: float,
    *,
    eigenspace_tol: float = 1e-10,
) -> EquilibriumCertificate:
    """Check the vector, multiplier, and exact spectral-Gram equations."""
    x = np.asarray(x, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if x.ndim != 2 or x.shape[1] != eigenvalues.size:
        raise ValueError("x must have shape (n_tokens, len(eigenvalues))")

    r_matrix = unnormalized_attention_matrix(x, eigenvalues, beta)
    h_matrix = np.einsum("id,jd,d->ij", x, x, eigenvalues, optimize=True)
    multipliers = np.diag(r_matrix @ h_matrix)
    multiplier_matrix = np.diag(multipliers)
    raw_output = r_matrix @ (x * eigenvalues)
    multiplier_residual = raw_output - multiplier_matrix @ x

    values, grams = spectral_gram_matrices(
        x, eigenvalues, tol=eigenspace_tol
    )
    gram_residual = 0.0
    for eigenvalue, gram in zip(values, grams, strict=True):
        residual = (eigenvalue * r_matrix - multiplier_matrix) @ gram
        gram_residual = max(gram_residual, float(np.linalg.norm(residual, ord="fro")))

    return EquilibriumCertificate(
        normalized_field=float(
            np.max(np.linalg.norm(vector_field(x, eigenvalues, beta), axis=1))
        ),
        multiplier_equation=float(np.linalg.norm(multiplier_residual, ord="fro")),
        spectral_gram_equation=gram_residual,
        sphere_constraint=float(np.max(np.abs(np.sum(x * x, axis=1) - 1.0))),
        multipliers=multipliers,
    )


def cluster_equilibrium_residual(
    centers: FloatArray,
    multiplicities: ArrayLike,
    eigenvalues: ArrayLike,
    beta: float,
) -> FloatArray:
    """Exact tangent equations after grouping identical tokens into clusters."""
    centers = normalize(np.asarray(centers, dtype=float))
    multiplicities = np.asarray(multiplicities, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if centers.ndim != 2 or centers.shape[0] != multiplicities.size:
        raise ValueError("one positive multiplicity is required per center")
    if np.any(multiplicities <= 0) or centers.shape[1] != eigenvalues.size:
        raise ValueError("invalid multiplicities or dimension")
    scores = np.einsum(
        "ad,bd,d->ab", centers, centers, eigenvalues, optimize=True
    )
    logits = beta * scores
    logits -= np.max(logits, axis=1, keepdims=True)
    coefficients = np.exp(logits) * multiplicities[None, :]
    raw_output = coefficients @ (centers * eigenvalues)
    radial = np.sum(centers * raw_output, axis=1, keepdims=True)
    return raw_output - radial * centers


def classify_beta_zero_equilibrium(
    x: FloatArray, eigenvalues: ArrayLike, tol: float = 1e-9
) -> str:
    """Complete beta=0 classification for a supplied configuration."""
    x = normalize(np.asarray(x, dtype=float))
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    mean_output = np.sum(x, axis=0) * eigenvalues
    if np.linalg.norm(mean_output) <= tol:
        return "zero_mean_output"
    residual = mean_output[None, :] - np.sum(
        x * mean_output[None, :], axis=1, keepdims=True
    ) * x
    if np.max(np.linalg.norm(residual, axis=1)) > tol:
        return "not_equilibrium"
    direction = mean_output / np.linalg.norm(mean_output)
    if np.max(1.0 - np.abs(x @ direction)) > tol:
        return "not_equilibrium"
    rayleigh = float(np.sum(eigenvalues * direction * direction))
    eigen_residual = eigenvalues * direction - rayleigh * direction
    if np.linalg.norm(eigen_residual) > tol:
        return "not_equilibrium"
    return "unbalanced_eigenline"


def regular_simplex(n_vertices: int) -> FloatArray:
    """Rows are a centered regular simplex in dimension n_vertices - 1."""
    if n_vertices < 2:
        raise ValueError("a simplex needs at least two vertices")
    centered = np.eye(n_vertices) - np.ones((n_vertices, n_vertices)) / n_vertices
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vh[: n_vertices - 1].T
    return normalize(coordinates)


def regular_polygon(n_vertices: int, phase: float = 0.0) -> FloatArray:
    """Equally spaced unit vectors in the plane."""
    if n_vertices < 2:
        raise ValueError("a polygon needs at least two vertices")
    angles = phase + 2.0 * np.pi * np.arange(n_vertices) / n_vertices
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def planar_equilibrium_residual(
    angles: ArrayLike, eigenvalues: tuple[float, float], beta: float
) -> FloatArray:
    """Scalar tangent residual for each token on S^1."""
    angles = np.asarray(angles, dtype=float)
    x = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    tangent = np.stack((-np.sin(angles), np.cos(angles)), axis=1)
    weights = attention_weights(x, np.asarray(eigenvalues), beta)
    output = weights @ (x * np.asarray(eigenvalues))
    return np.sum(tangent * output, axis=1)


def canonical_planar_key(angles: ArrayLike, decimals: int = 6) -> tuple[float, ...]:
    """Quotient token permutations and global coordinate sign symmetries."""
    angles = np.mod(np.asarray(angles, dtype=float), 2.0 * np.pi)
    transforms = (
        angles,
        -angles,
        np.pi - angles,
        angles + np.pi,
    )
    keys = []
    for transformed in transforms:
        values = np.sort(np.mod(transformed, 2.0 * np.pi))
        rounded = np.round(values, decimals=decimals)
        rounded[np.isclose(rounded, 2.0 * np.pi, atol=10.0 ** (-decimals))] = 0.0
        keys.append(tuple(np.sort(rounded).tolist()))
    return min(keys)


def _cluster_count_from_angles(angles: FloatArray, tol: float = 2e-5) -> int:
    points = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    clusters: list[FloatArray] = []
    for point in points:
        if not any(np.linalg.norm(point - center) < tol for center in clusters):
            clusters.append(point)
    return len(clusters)


@dataclass(frozen=True)
class PlanarEquilibrium:
    angles: FloatArray
    residual: float
    cluster_count: int
    max_linear_rate: float
    stable: bool


def find_planar_equilibria(
    eigenvalues: tuple[float, float],
    beta: float,
    n_tokens: int,
    *,
    random_starts: int = 5000,
    seed: int = 260426085,
    residual_tol: float = 2e-9,
) -> list[PlanarEquilibrium]:
    """Discover distinct equilibria on S^1 by deterministic multi-start roots."""
    if n_tokens < 1 or beta < 0:
        raise ValueError("n_tokens must be positive and beta nonnegative")
    rng = np.random.default_rng(seed)
    starts = [rng.uniform(0.0, 2.0 * np.pi, size=n_tokens) for _ in range(random_starts)]
    eigen_angles = (0.0, np.pi, np.pi / 2.0, 3.0 * np.pi / 2.0)
    starts.extend(np.asarray(values) for values in product(eigen_angles, repeat=n_tokens))
    if n_tokens >= 2:
        starts.extend(
            regular_polygon(n_tokens, phase)[:, 1].copy() * 0.0
            + phase
            + 2.0 * np.pi * np.arange(n_tokens) / n_tokens
            for phase in np.linspace(0.0, np.pi, 17, endpoint=False)
        )

    equilibria: dict[tuple[float, ...], PlanarEquilibrium] = {}
    for start in starts:
        result = least_squares(
            planar_equilibrium_residual,
            start,
            args=(eigenvalues, beta),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=2500,
        )
        residual = float(np.max(np.abs(result.fun)))
        if residual > residual_tol:
            continue
        angles = np.mod(result.x, 2.0 * np.pi)
        key = canonical_planar_key(angles)
        if key in equilibria:
            continue
        state = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        rates = np.linalg.eigvals(
            state_tangent_jacobian(state, np.asarray(eigenvalues), beta)
        ).real
        max_rate = float(np.max(rates))
        equilibria[key] = PlanarEquilibrium(
            angles=np.asarray(key),
            residual=residual,
            cluster_count=_cluster_count_from_angles(angles),
            max_linear_rate=max_rate,
            stable=max_rate < -1e-7,
        )
    return sorted(equilibria.values(), key=lambda item: tuple(item.angles))
