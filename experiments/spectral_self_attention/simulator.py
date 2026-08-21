"""Numerics and diagnostics for symmetric spherical self-attention.

The state has shape ``(..., n_tokens, dimension)``.  The interaction matrix is
diagonal in the supplied eigenbasis, so only its eigenvalues are needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def normalize(x: FloatArray) -> FloatArray:
    """Normalize the last axis, rejecting zero vectors."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector")
    return x / norms


def random_sphere(
    rng: np.random.Generator, batch: int, n_tokens: int, dimension: int
) -> FloatArray:
    """Independent uniform samples on the unit sphere."""
    return normalize(rng.normal(size=(batch, n_tokens, dimension)))


def attention_weights(x: FloatArray, eigenvalues: ArrayLike, beta: float) -> FloatArray:
    """Row-stochastic attention matrix for each configuration."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    vx = x * eigenvalues
    scores = np.einsum("...id,...jd->...ij", x, vx, optimize=True)
    logits = beta * scores
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def vector_field(x: FloatArray, eigenvalues: ArrayLike, beta: float) -> FloatArray:
    """Evaluate equation (2.2) of arXiv:2604.26085 in an eigenbasis."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    weights = attention_weights(x, eigenvalues, beta)
    mean_vx = np.einsum(
        "...ij,...jd->...id", weights, x * eigenvalues, optimize=True
    )
    radial = np.sum(x * mean_vx, axis=-1, keepdims=True)
    return mean_vx - radial * x


def energy(x: FloatArray, eigenvalues: ArrayLike, beta: float) -> FloatArray:
    """Interaction energy, with the divergent constant removed at beta=0."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    vx = x * eigenvalues
    scores = np.einsum("...id,...jd->...ij", x, vx, optimize=True)
    if beta == 0:
        return 0.5 * np.sum(scores, axis=(-2, -1))
    return np.sum(np.exp(beta * scores), axis=(-2, -1)) / (2.0 * beta)


@dataclass(frozen=True)
class Trajectory:
    times: FloatArray
    states: FloatArray
    energies: FloatArray


def integrate(
    x0: FloatArray,
    eigenvalues: ArrayLike,
    beta: float,
    *,
    t_final: float = 40.0,
    dt: float = 0.02,
    save_every: int | None = None,
) -> Trajectory:
    """Retraction RK4 integrator, vectorized over an optional batch axis.

    Every RK stage is retracted to the sphere.  This is deliberately simple and
    deterministic; convergence and time-step checks are part of the sweep.
    """
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    if dt <= 0 or t_final <= 0:
        raise ValueError("dt and t_final must be positive")

    x = normalize(np.asarray(x0, dtype=float).copy())
    n_steps = int(np.ceil(t_final / dt))
    actual_dt = t_final / n_steps
    if save_every is None:
        save_every = n_steps
    if save_every <= 0:
        raise ValueError("save_every must be positive")

    saved_states = [x.copy()]
    saved_times = [0.0]
    saved_energies = [energy(x, eigenvalues, beta)]

    for step in range(1, n_steps + 1):
        k1 = vector_field(x, eigenvalues, beta)
        x2 = normalize(x + 0.5 * actual_dt * k1)
        k2 = vector_field(x2, eigenvalues, beta)
        x3 = normalize(x + 0.5 * actual_dt * k2)
        k3 = vector_field(x3, eigenvalues, beta)
        x4 = normalize(x + actual_dt * k3)
        k4 = vector_field(x4, eigenvalues, beta)
        x = normalize(x + actual_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)

        if step % save_every == 0 or step == n_steps:
            saved_states.append(x.copy())
            saved_times.append(step * actual_dt)
            saved_energies.append(energy(x, eigenvalues, beta))

    return Trajectory(
        times=np.asarray(saved_times),
        states=np.asarray(saved_states),
        energies=np.asarray(saved_energies),
    )


def eigenspace_groups(eigenvalues: ArrayLike, tol: float = 1e-10) -> list[np.ndarray]:
    """Indices of equal-eigenvalue groups, in descending eigenvalue order."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    order = np.argsort(-eigenvalues, kind="stable")
    groups: list[list[int]] = []
    for index in order:
        if not groups or not np.isclose(
            eigenvalues[index], eigenvalues[groups[-1][0]], atol=tol, rtol=tol
        ):
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return [np.asarray(group, dtype=int) for group in groups]


@dataclass(frozen=True)
class Diagnostics:
    speed: FloatArray
    min_correlation: FloatArray
    max_correlation: FloatArray
    mean_abs_correlation: FloatArray
    mean_vector_norm: FloatArray
    modal_masses: FloatArray
    eigenspace_masses: FloatArray
    extreme_eigenspace_mass: FloatArray
    selected_group: NDArray[np.int64]
    selected_group_mass: FloatArray
    geometry: NDArray[np.str_]


def diagnostics(
    x: FloatArray,
    eigenvalues: ArrayLike,
    beta: float,
    *,
    correlation_tol: float = 2e-3,
    subspace_tol: float = 2e-3,
) -> Diagnostics:
    """Classify final geometry without assuming simple eigenvalues."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[None, ...]
    batch, n_tokens, _ = x.shape
    gram = np.einsum("bid,bjd->bij", x, x, optimize=True)
    if n_tokens > 1:
        offdiag = ~np.eye(n_tokens, dtype=bool)
        pairwise = gram[:, offdiag]
        min_correlation = np.min(pairwise, axis=1)
        max_correlation = np.max(pairwise, axis=1)
        mean_abs_correlation = np.mean(np.abs(pairwise), axis=1)
    else:
        min_correlation = np.ones(batch)
        max_correlation = np.ones(batch)
        mean_abs_correlation = np.ones(batch)

    modal_masses = np.mean(x * x, axis=1)
    groups = eigenspace_groups(eigenvalues)
    group_masses = np.stack(
        [np.sum(modal_masses[:, group], axis=1) for group in groups], axis=1
    )
    selected_group = np.argmax(group_masses, axis=1)
    selected_group_mass = np.max(group_masses, axis=1)
    if len(groups) == 1:
        extreme_eigenspace_mass = group_masses[:, 0]
    else:
        extreme_eigenspace_mass = group_masses[:, 0] + group_masses[:, -1]
    speed = np.linalg.norm(vector_field(x, eigenvalues, beta), axis=-1).max(axis=1)
    mean_vector_norm = np.linalg.norm(np.mean(x, axis=1), axis=1)

    labels = np.full(batch, "nonconverged", dtype="U24")
    consensus = min_correlation > 1.0 - correlation_tol
    bipolar = (
        (mean_abs_correlation > 1.0 - correlation_tol)
        & (min_correlation < -1.0 + correlation_tol)
    )
    subspace_selected = selected_group_mass > 1.0 - subspace_tol
    stationary = speed < 2e-5
    labels[stationary] = "stationary_other"
    mixed_extreme = (
        stationary
        & (extreme_eigenspace_mass > 1.0 - subspace_tol)
        & ~subspace_selected
    )
    labels[mixed_extreme] = "mixed_extreme_stationary"
    labels[subspace_selected & stationary] = "subspace_stationary"
    labels[bipolar & subspace_selected] = "bipolar"
    labels[consensus & subspace_selected] = "consensus"

    return Diagnostics(
        speed=speed,
        min_correlation=min_correlation,
        max_correlation=max_correlation,
        mean_abs_correlation=mean_abs_correlation,
        mean_vector_norm=mean_vector_norm,
        modal_masses=modal_masses,
        eigenspace_masses=group_masses,
        extreme_eigenspace_mass=extreme_eigenspace_mass,
        selected_group=selected_group.astype(np.int64),
        selected_group_mass=selected_group_mass,
        geometry=labels,
    )


def pure_mode_linear_rates(
    eigenvalues: ArrayLike,
    beta: float,
    mode: int,
    n_plus: int,
    n_minus: int,
) -> FloatArray:
    """All transverse rates at a pure-mode sign pattern.

    This directly evaluates the block linearization from Theorem 5.2, avoiding
    algebraic simplifications near beta=0 or balanced degeneracies.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    lam_p = float(eigenvalues[mode])
    if n_plus < 0 or n_minus < 0 or n_plus + n_minus == 0:
        raise ValueError("invalid group sizes")
    if n_plus == 0 or n_minus == 0:
        transverse = np.delete(eigenvalues - lam_p, mode)
        fluctuation = np.full((eigenvalues.size - 1) * (n_plus + n_minus - 1), -lam_p)
        return np.concatenate([transverse, fluctuation])

    ep = np.exp(beta * lam_p)
    em = np.exp(-beta * lam_p)
    den_plus = n_plus * ep + n_minus * em
    den_minus = n_minus * ep + n_plus * em
    a_plus, b_plus = ep / den_plus, em / den_plus
    a_minus, b_minus = ep / den_minus, em / den_minus
    gamma_plus = lam_p * (n_plus * a_plus - n_minus * b_plus)
    gamma_minus = lam_p * (n_minus * a_minus - n_plus * b_minus)

    rates: list[float] = []
    for k, lam_k in enumerate(eigenvalues):
        if k == mode:
            continue
        rates.extend([-gamma_plus] * max(n_plus - 1, 0))
        rates.extend([-gamma_minus] * max(n_minus - 1, 0))
        block = np.array(
            [
                [lam_k * n_plus * a_plus - gamma_plus, lam_k * n_minus * b_plus],
                [lam_k * n_plus * b_minus, lam_k * n_minus * a_minus - gamma_minus],
            ]
        )
        rates.extend(np.linalg.eigvals(block).real.tolist())
    return np.asarray(rates)


def is_linearly_stable_pure_mode(
    eigenvalues: ArrayLike,
    beta: float,
    mode: int,
    n_plus: int,
    n_minus: int,
    *,
    tol: float = 1e-10,
) -> bool:
    """Whether every transverse pure-mode linear rate is strictly negative."""
    rates = pure_mode_linear_rates(eigenvalues, beta, mode, n_plus, n_minus)
    return bool(rates.size and np.max(rates) < -tol)


def pure_mode_tangent_jacobian(
    eigenvalues: ArrayLike,
    beta: float,
    mode: int,
    n_plus: int,
    n_minus: int,
    *,
    epsilon: float = 1e-6,
) -> FloatArray:
    """Finite-difference Jacobian in orthonormal tangent coordinates.

    This is intentionally independent of ``pure_mode_linear_rates`` and serves
    as a numerical check of the paper's block linearization.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    n_tokens = n_plus + n_minus
    if n_tokens == 0 or n_plus < 0 or n_minus < 0:
        raise ValueError("invalid group sizes")
    signs = np.concatenate((np.ones(n_plus), -np.ones(n_minus)))
    base = np.zeros((n_tokens, eigenvalues.size))
    base[:, mode] = signs
    coordinates = [
        (token, transverse_mode)
        for transverse_mode in range(eigenvalues.size)
        if transverse_mode != mode
        for token in range(n_tokens)
    ]
    jacobian = np.empty((len(coordinates), len(coordinates)))
    for column, (token, transverse_mode) in enumerate(coordinates):
        direction = np.zeros_like(base)
        direction[token, transverse_mode] = 1.0
        plus = vector_field(normalize(base + epsilon * direction), eigenvalues, beta)
        minus = vector_field(normalize(base - epsilon * direction), eigenvalues, beta)
        derivative = (plus - minus) / (2.0 * epsilon)
        jacobian[:, column] = [
            derivative[row_token, row_mode]
            for row_token, row_mode in coordinates
        ]
    return jacobian


def state_tangent_jacobian(
    x: FloatArray,
    eigenvalues: ArrayLike,
    beta: float,
    *,
    epsilon: float = 1e-6,
) -> FloatArray:
    """Finite-difference linearization at an arbitrary spherical state."""
    x = np.asarray(x, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if x.ndim != 2 or x.shape[1] != eigenvalues.size:
        raise ValueError("x must have shape (n_tokens, len(eigenvalues))")
    tangent_bases = []
    for token in x:
        _, _, vh = np.linalg.svd(token[None, :], full_matrices=True)
        tangent_bases.append(vh[1:].T)
    coordinates = [
        (token, local_mode)
        for token in range(x.shape[0])
        for local_mode in range(x.shape[1] - 1)
    ]
    jacobian = np.empty((len(coordinates), len(coordinates)))
    for column, (token, local_mode) in enumerate(coordinates):
        direction = np.zeros_like(x)
        direction[token] = tangent_bases[token][:, local_mode]
        plus = vector_field(normalize(x + epsilon * direction), eigenvalues, beta)
        minus = vector_field(normalize(x - epsilon * direction), eigenvalues, beta)
        derivative = (plus - minus) / (2.0 * epsilon)
        jacobian[:, column] = [
            np.dot(derivative[row_token], tangent_bases[row_token][:, row_mode])
            for row_token, row_mode in coordinates
        ]
    return jacobian
