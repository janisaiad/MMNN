#!/usr/bin/env python3
"""Deterministic leading-width ReLU recursions for V, D, F, A, and B.

This implements the indexed recursions in Guillen--Misof--Gerken (2026)
without Monte Carlo sampling.  ReLU transport insertions are evaluated in
closed form.  The only numerical operations are Gaussian positive-orthant
moments of degree at most four and dimension at most four.  Rank-one and
rank-two covariances are evaluated by exact radial/angular reduction; the
remaining moments use deterministic adaptive cubature.

The tensor convention follows the public ``ntk-unlimited`` implementation:
``B[a,b,c,d]`` denotes the crossed B component and ``F[a,b,c,d]`` the crossed
F component used by its recursion expressions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


PLOT_COMPONENTS = (
    (0, 0, 0, 0),
    (0, 1, 0, 1),
    (0, 0, 2, 2),
    (0, 1, 0, 3),
    (0, 0, 2, 3),
    (0, 1, 2, 3),
)


@dataclass(frozen=True)
class CubatureConfig:
    rtol: float = 2e-7
    atol: float = 2e-10
    max_subdivisions: int = 20_000


def _half_abs_normal_moment(degree: int) -> float:
    """E[|Z|^degree 1{Z>0}] for Z standard normal."""
    return (
        2.0 ** (degree / 2.0 - 1.0)
        * math.gamma((degree + 1.0) / 2.0)
        / math.sqrt(math.pi)
    )


def _radial_normal_moment_2d(degree: int) -> float:
    """E[R^degree] for a two-dimensional standard normal radius."""
    return 2.0 ** (degree / 2.0) * math.gamma(1.0 + degree / 2.0)


class OrthantMomentTable:
    """Batched moments E[prod X_i^p_i 1{X_i>0 for i in S}]."""

    def __init__(
        self,
        covariance: np.ndarray,
        queries: dict[tuple[int, ...], set[tuple[int, ...]]],
        config: CubatureConfig,
    ) -> None:
        self.covariance = np.asarray(covariance, dtype=float)
        self.config = config
        self.values: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        self.errors: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        self.statuses: list[str] = []
        self._moment_cache: dict[tuple, tuple[float, float]] = {}
        self._orthant_cache: dict[tuple, tuple[float, float]] = {}
        for conditioned, exponents in sorted(queries.items()):
            self._compute_group(conditioned, sorted(exponents))

    @staticmethod
    def build_queries(data_size: int) -> dict[tuple[int, ...], set[tuple[int, ...]]]:
        queries: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
        for a, b, c, d in product(range(data_size), repeat=4):
            # E[R_a R_b R_c R_d]
            OrthantMomentTable._add_query(queries, (a, b, c, d), (a, b, c, d))
            # E[R_a R_b I_c I_d]
            OrthantMomentTable._add_query(queries, (a, b, c, d), (a, b))
            # E[I_a I_b I_c I_d]
            OrthantMomentTable._add_query(queries, (a, b, c, d), ())
        return queries

    @staticmethod
    def _add_query(
        queries: dict[tuple[int, ...], set[tuple[int, ...]]],
        conditioned_indices: tuple[int, ...],
        factor_indices: tuple[int, ...],
    ) -> None:
        conditioned = tuple(sorted(set(conditioned_indices)))
        counts = Counter(factor_indices)
        exponents = tuple(counts[i] for i in conditioned)
        queries[conditioned].add(exponents)

    def get(
        self,
        conditioned_indices: tuple[int, ...],
        factor_indices: tuple[int, ...],
    ) -> float:
        conditioned = tuple(sorted(set(conditioned_indices)))
        counts = Counter(factor_indices)
        exponents = tuple(counts[i] for i in conditioned)
        return self.values[(conditioned, exponents)]

    @property
    def max_error(self) -> float:
        return max(self.errors.values(), default=0.0)

    def _compute_group(
        self, conditioned: tuple[int, ...], exponents: list[tuple[int, ...]]
    ) -> None:
        covariance = self.covariance[np.ix_(conditioned, conditioned)]
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(float(eigenvalues[-1]), 1.0)
        positive = eigenvalues > 5e-13 * scale
        rank = int(np.count_nonzero(positive))
        if rank == 0:
            raise ValueError("zero covariance encountered")
        factor = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])

        if rank == 1:
            estimates = self._rank_one(factor[:, 0], exponents)
            errors = np.zeros_like(estimates)
            status = "closed_form_rank_1"
        elif rank == 2:
            estimates, errors = self._rank_two(factor, exponents)
            status = "angular_rank_2"
        else:
            estimates, errors, status = self._full_rank_moments(covariance, exponents)

        self.statuses.append(status)
        for exponent, estimate, error in zip(exponents, estimates, errors):
            key = (conditioned, exponent)
            self.values[key] = float(estimate)
            self.errors[key] = float(error)

    @staticmethod
    def _rank_one(coefficients: np.ndarray, exponents: list[tuple[int, ...]]):
        signs = np.sign(coefficients)
        if np.any(signs == 0.0) or np.any(signs != signs[0]):
            return np.zeros(len(exponents), dtype=float)
        halfline_sign = int(signs[0])
        estimates = []
        for exponent in exponents:
            degree = sum(exponent)
            coefficient = float(np.prod(coefficients ** np.asarray(exponent)))
            estimates.append(
                coefficient
                * halfline_sign**degree
                * _half_abs_normal_moment(degree)
            )
        return np.asarray(estimates)

    def _rank_two(
        self, factor: np.ndarray, exponents: list[tuple[int, ...]]
    ) -> tuple[np.ndarray, np.ndarray]:
        boundaries = [0.0, 2.0 * math.pi]
        for row in factor:
            angle = math.atan2(row[1], row[0])
            for boundary in (angle - math.pi / 2.0, angle + math.pi / 2.0):
                boundaries.append(boundary % (2.0 * math.pi))
        boundaries = np.asarray(sorted(set(round(x, 15) for x in boundaries)))
        degrees = np.asarray([sum(exponent) for exponent in exponents], dtype=int)
        radial = np.asarray([_radial_normal_moment_2d(int(d)) for d in degrees])
        total = np.zeros(len(exponents), dtype=float)
        total_error = 0.0

        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            midpoint = 0.5 * (lower + upper)
            direction = np.asarray([math.cos(midpoint), math.sin(midpoint)])
            if np.any(factor @ direction <= 0.0):
                continue

            def angular_integrand(theta):
                theta = np.atleast_1d(theta)
                directions = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
                projections = directions @ factor.T
                values = np.empty((theta.size, len(exponents)), dtype=float)
                for j, exponent in enumerate(exponents):
                    values[:, j] = np.prod(
                        projections ** np.asarray(exponent)[None, :], axis=1
                    )
                return values[0] if values.shape[0] == 1 else values

            estimate, error = integrate.quad_vec(
                angular_integrand,
                lower,
                upper,
                epsrel=self.config.rtol,
                epsabs=self.config.atol,
            )
            total += np.asarray(estimate)
            total_error += float(error)
        estimates = radial * total / (2.0 * math.pi)
        errors = radial * total_error / (2.0 * math.pi)
        return estimates, errors

    @staticmethod
    def _covariance_key(covariance: np.ndarray) -> tuple:
        covariance = np.asarray(covariance, dtype=float)
        return (covariance.shape, *np.round(covariance.ravel(), 14))

    def _orthant_probability(self, covariance: np.ndarray) -> tuple[float, float]:
        """Zero-mean Gaussian positive-orthant probability in dimension <= 4.

        Dimensions one through three use the classical arcsine formula.  In
        dimension four, Plackett's identity is integrated along the positive
        definite path R(t)=I+t(R-I), reducing the problem to one dimension.
        """
        covariance = np.asarray(covariance, dtype=float)
        key = self._covariance_key(covariance)
        if key in self._orthant_cache:
            return self._orthant_cache[key]
        dimension = covariance.shape[0]
        standard_deviations = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(standard_deviations, standard_deviations)
        correlation = np.clip(0.5 * (correlation + correlation.T), -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)

        if dimension == 0:
            result = (1.0, 0.0)
        elif dimension == 1:
            result = (0.5, 0.0)
        elif dimension == 2:
            result = (
                0.25 + math.asin(float(correlation[0, 1])) / (2.0 * math.pi),
                0.0,
            )
        elif dimension == 3:
            arcsines = sum(
                math.asin(float(correlation[i, j]))
                for i in range(3)
                for j in range(i + 1, 3)
            )
            result = (0.125 + arcsines / (4.0 * math.pi), 0.0)
        elif dimension == 4:
            identity = np.eye(4)
            off_diagonal = correlation - identity

            def path_derivative(t: float) -> float:
                current = identity + t * off_diagonal
                derivative = 0.0
                for i in range(4):
                    for j in range(i + 1, 4):
                        target_correlation = correlation[i, j]
                        if target_correlation == 0.0:
                            continue
                        pair = [i, j]
                        remaining = [k for k in range(4) if k not in pair]
                        pair_covariance = current[np.ix_(pair, pair)]
                        cross = current[np.ix_(remaining, pair)]
                        conditional = (
                            current[np.ix_(remaining, remaining)]
                            - cross @ np.linalg.solve(pair_covariance, cross.T)
                        )
                        conditional_rho = conditional[0, 1] / math.sqrt(
                            conditional[0, 0] * conditional[1, 1]
                        )
                        conditional_rho = float(np.clip(conditional_rho, -1.0, 1.0))
                        conditional_probability = 0.25 + math.asin(
                            conditional_rho
                        ) / (2.0 * math.pi)
                        rho = current[i, j]
                        marginal_density = 1.0 / (
                            2.0 * math.pi * math.sqrt(max(1.0 - rho * rho, 1e-30))
                        )
                        derivative += (
                            target_correlation
                            * marginal_density
                            * conditional_probability
                        )
                return derivative

            correction, error = integrate.quad(
                path_derivative,
                0.0,
                1.0,
                epsrel=min(self.config.rtol, 1e-10),
                epsabs=min(self.config.atol, 1e-12),
                limit=250,
            )
            result = (1.0 / 16.0 + correction, float(error))
        else:
            raise ValueError("only dimensions up to four are supported")
        self._orthant_cache[key] = result
        return result

    def _truncated_moment(
        self, covariance: np.ndarray, exponent: tuple[int, ...]
    ) -> tuple[float, float]:
        """Tallis integration-by-parts recursion for orthant moments."""
        covariance = np.asarray(covariance, dtype=float)
        key = (self._covariance_key(covariance), exponent)
        if key in self._moment_cache:
            return self._moment_cache[key]
        if sum(exponent) == 0:
            result = self._orthant_probability(covariance)
            self._moment_cache[key] = result
            return result

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(float(eigenvalues[-1]), 1.0)
        positive = eigenvalues > 5e-13 * scale
        rank = int(np.count_nonzero(positive))
        if rank < covariance.shape[0]:
            factor = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
            if rank == 1:
                value = float(self._rank_one(factor[:, 0], [exponent])[0])
                result = (value, 0.0)
            elif rank == 2:
                values, errors = self._rank_two(factor, [exponent])
                result = (float(values[0]), float(errors[0]))
            else:
                raise FloatingPointError("unsupported singular rank-three covariance")
            self._moment_cache[key] = result
            return result

        i = next(index for index, power in enumerate(exponent) if power > 0)
        reduced_i = list(exponent)
        reduced_i[i] -= 1
        value = 0.0
        error = 0.0
        dimension = covariance.shape[0]
        for j in range(dimension):
            bracket_value = 0.0
            bracket_error = 0.0
            if reduced_i[j] == 0:
                remaining = [k for k in range(dimension) if k != j]
                conditional_covariance = (
                    covariance[np.ix_(remaining, remaining)]
                    - np.outer(covariance[remaining, j], covariance[j, remaining])
                    / covariance[j, j]
                )
                conditional_exponent = tuple(reduced_i[k] for k in remaining)
                boundary_value, boundary_error = self._truncated_moment(
                    conditional_covariance, conditional_exponent
                )
                marginal_density_zero = 1.0 / math.sqrt(
                    2.0 * math.pi * covariance[j, j]
                )
                bracket_value += marginal_density_zero * boundary_value
                bracket_error += marginal_density_zero * boundary_error
            if reduced_i[j] > 0:
                reduced_ij = reduced_i.copy()
                reduced_ij[j] -= 1
                lower_value, lower_error = self._truncated_moment(
                    covariance, tuple(reduced_ij)
                )
                bracket_value += reduced_i[j] * lower_value
                bracket_error += reduced_i[j] * lower_error
            value += covariance[i, j] * bracket_value
            error += abs(covariance[i, j]) * bracket_error
        result = (value, error)
        self._moment_cache[key] = result
        return result

    def _full_rank_moments(
        self, covariance: np.ndarray, exponents: list[tuple[int, ...]]
    ) -> tuple[np.ndarray, np.ndarray, str]:
        results = [self._truncated_moment(covariance, exponent) for exponent in exponents]
        return (
            np.asarray([result[0] for result in results]),
            np.asarray([result[1] for result in results]),
            "tallis_plackett",
        )


def relu_pair_moments(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return E[R_a R_b] and E[I_a I_b]."""
    covariance = np.asarray(covariance, dtype=float)
    data_size = covariance.shape[0]
    rr = np.empty_like(covariance)
    ii = np.empty_like(covariance)
    variances = np.diag(covariance)
    for a, b in product(range(data_size), repeat=2):
        rho = covariance[a, b] / math.sqrt(variances[a] * variances[b])
        rho = float(np.clip(rho, -1.0, 1.0))
        theta = math.acos(rho)
        rr[a, b] = math.sqrt(variances[a] * variances[b]) * (
            math.sin(theta) + (math.pi - theta) * math.cos(theta)
        ) / (2.0 * math.pi)
        ii[a, b] = (math.pi - theta) / (2.0 * math.pi)
    return rr, ii


def relu_transport_operators(
    covariance: np.ndarray, ntk: np.ndarray, weight_variance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return E[Hess(RR)], E[Hess(Omega)], and E[grad(R I)]."""
    covariance = np.asarray(covariance, dtype=float)
    data_size = covariance.shape[0]
    rr_hessian = np.zeros((data_size,) * 4, dtype=float)
    ii_hessian = np.zeros_like(rr_hessian)
    ri_gradient = np.zeros((data_size,) * 3, dtype=float)
    _, indicator_pair = relu_pair_moments(covariance)
    variances = np.diag(covariance)

    for a, b in product(range(data_size), repeat=2):
        if a == b:
            rr_hessian[a, b, a, a] = 1.0
            ri_gradient[a, b, a] = 0.5
            continue
        determinant = max(
            variances[a] * variances[b] - covariance[a, b] ** 2, 0.0
        )
        root_determinant = math.sqrt(determinant)
        if root_determinant == 0.0:
            raise FloatingPointError("distinct samples became exactly collinear")

        rr_hessian[a, b, a, a] = root_determinant / (
            2.0 * math.pi * variances[a]
        )
        rr_hessian[a, b, b, b] = root_determinant / (
            2.0 * math.pi * variances[b]
        )
        rr_hessian[a, b, a, b] = indicator_pair[a, b]
        rr_hessian[a, b, b, a] = indicator_pair[a, b]

        joint_density_zero = 1.0 / (2.0 * math.pi * root_determinant)
        ii_hessian[a, b, a, b] = joint_density_zero
        ii_hessian[a, b, b, a] = joint_density_zero
        ii_hessian[a, b, a, a] = (
            -covariance[a, b] * joint_density_zero / variances[a]
        )
        ii_hessian[a, b, b, b] = (
            -covariance[a, b] * joint_density_zero / variances[b]
        )

        ri_gradient[a, b, a] = indicator_pair[a, b]
        ri_gradient[a, b, b] = root_determinant / (
            2.0 * math.pi * variances[b]
        )

    omega_hessian = rr_hessian + (
        weight_variance
        * ntk[:, :, None, None]
        * ii_hessian
    )
    return rr_hessian, omega_hessian, ri_gradient


def _tensor_sources(
    covariance: np.ndarray,
    ntk: np.ndarray,
    weight_variance: float,
    moment_table: OrthantMomentTable,
) -> dict[str, np.ndarray]:
    data_size = covariance.shape[0]
    rr, ii = relu_pair_moments(covariance)
    cov_rr_rr = np.empty((data_size,) * 4, dtype=float)
    cov_rr_omega = np.empty_like(cov_rr_rr)
    cov_omega_omega = np.empty_like(cov_rr_rr)
    rr_ii = np.empty_like(cov_rr_rr)
    iiii = np.empty_like(cov_rr_rr)

    for a, b, c, d in product(range(data_size), repeat=4):
        indices = (a, b, c, d)
        rrrr = moment_table.get(indices, indices)
        rr_cd_ii_ab = moment_table.get(indices, (c, d))
        rr_ab_ii_cd = moment_table.get(indices, (a, b))
        four_indicators = moment_table.get(indices, ())
        rr_ii[a, b, c, d] = rr_ab_ii_cd
        iiii[a, b, c, d] = four_indicators

        c_ab = weight_variance * ntk[a, b]
        c_cd = weight_variance * ntk[c, d]
        cov_rr = rrrr - rr[a, b] * rr[c, d]
        cov_ab_ii_cd = rr_ab_ii_cd - rr[a, b] * ii[c, d]
        cov_cd_ii_ab = rr_cd_ii_ab - rr[c, d] * ii[a, b]
        cov_ii = four_indicators - ii[a, b] * ii[c, d]
        cov_rr_rr[a, b, c, d] = cov_rr
        cov_rr_omega[a, b, c, d] = cov_rr + c_cd * cov_ab_ii_cd
        cov_omega_omega[a, b, c, d] = (
            cov_rr
            + c_ab * cov_cd_ii_ab
            + c_cd * cov_ab_ii_cd
            + c_ab * c_cd * cov_ii
        )
    return {
        "rr": rr,
        "ii": ii,
        "cov_rr_rr": cov_rr_rr,
        "cov_rr_omega": cov_rr_omega,
        "cov_omega_omega": cov_omega_omega,
        "rr_ii": rr_ii,
        "iiii": iiii,
    }


def tensor_step(
    covariance: np.ndarray,
    ntk: np.ndarray,
    tensors: dict[str, np.ndarray],
    weight_variance: float,
    moment_table: OrthantMomentTable,
) -> dict[str, np.ndarray]:
    """Advance all five leading four-index tensors by one layer."""
    sources = _tensor_sources(covariance, ntk, weight_variance, moment_table)
    rr_hessian, omega_hessian, ri_gradient = relu_transport_operators(
        covariance, ntk, weight_variance
    )
    indicator_pair = sources["ii"]
    V, D, F, A, B = (tensors[name] for name in ("V", "D", "F", "A", "B"))

    V_next = weight_variance**2 * sources["cov_rr_rr"]
    V_next += (weight_variance**2 / 4.0) * np.einsum(
        "ghij,abgh,cdij->abcd", V, rr_hessian, rr_hessian, optimize=True
    )

    D_next = weight_variance * sources["cov_rr_omega"]
    D_next += (weight_variance / 4.0) * np.einsum(
        "ghij,abgh,cdij->abcd", V, rr_hessian, omega_hessian, optimize=True
    )
    D_next += (weight_variance**2 / 2.0) * np.einsum(
        "cd,abgh,ghcd->abcd", indicator_pair, rr_hessian, D, optimize=True
    )

    F_next = weight_variance**2 * np.einsum(
        "acbd,bd->abcd", sources["rr_ii"], ntk, optimize=True
    )
    F_next += weight_variance**2 * np.einsum(
        "abg,cdh,gbhd->abcd", ri_gradient, ri_gradient, F, optimize=True
    )

    A_next = sources["cov_omega_omega"].copy()
    A_next += 0.25 * np.einsum(
        "ghij,abgh,cdij->abcd", V, omega_hessian, omega_hessian, optimize=True
    )
    A_next += (weight_variance / 2.0) * np.einsum(
        "cd,abgh,ghcd->abcd", indicator_pair, omega_hessian, D, optimize=True
    )
    A_next += (weight_variance / 2.0) * np.einsum(
        "ab,cdgh,ghab->abcd", indicator_pair, omega_hessian, D, optimize=True
    )
    A_next += weight_variance**2 * np.einsum(
        "ab,cd,abcd->abcd", indicator_pair, indicator_pair, A, optimize=True
    )

    B_next = weight_variance**2 * np.einsum(
        "abcd,ac,bd->abcd", sources["iiii"], ntk, ntk, optimize=True
    )
    B_next += weight_variance**2 * np.einsum(
        "ab,cd,abcd->abcd", indicator_pair, indicator_pair, B, optimize=True
    )
    return {"V": V_next, "D": D_next, "F": F_next, "A": A_next, "B": B_next}


def kernel_step(
    covariance: np.ndarray, ntk: np.ndarray, weight_variance: float
) -> tuple[np.ndarray, np.ndarray]:
    rr, ii = relu_pair_moments(covariance)
    return weight_variance * rr, rr + weight_variance * ii * ntk


def run_recursions(
    inputs: np.ndarray,
    depth: int,
    weight_variance: float,
    config: CubatureConfig,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    data_size, input_dimension = inputs.shape
    input_gram = inputs @ inputs.T / input_dimension
    covariance = weight_variance * input_gram
    ntk = input_gram.copy()
    zeros = np.zeros((data_size,) * 4, dtype=float)
    tensors = {name: zeros.copy() for name in ("V", "D", "F", "A", "B")}
    histories = {name: [value.copy()] for name, value in tensors.items()}
    kernel_history = [
        {
            "layer": 1,
            "covariance": covariance.copy(),
            "ntk": ntk.copy(),
            "cubature_max_error": 0.0,
        }
    ]
    queries = OrthantMomentTable.build_queries(data_size)

    for layer in range(2, depth + 1):
        moment_table = OrthantMomentTable(covariance, queries, config)
        tensors = tensor_step(
            covariance, ntk, tensors, weight_variance, moment_table
        )
        covariance, ntk = kernel_step(covariance, ntk, weight_variance)
        for name in histories:
            histories[name].append(tensors[name].copy())
        kernel_history.append(
            {
                "layer": layer,
                "covariance": covariance.copy(),
                "ntk": ntk.copy(),
                "cubature_max_error": moment_table.max_error,
                "cubature_statuses": sorted(set(moment_table.statuses)),
            }
        )
        print(
            f"layer {layer:3d}/{depth}: max raw cubature error "
            f"{moment_table.max_error:.3e}"
        )
    return {name: np.stack(values) for name, values in histories.items()}, kernel_history


def effective_power(values: np.ndarray, layers: np.ndarray, start: int) -> float:
    mask = (layers >= start) & np.isfinite(values) & (np.abs(values) > 0.0)
    slope, _ = np.polyfit(np.log(layers[mask]), np.log(np.abs(values[mask])), 1)
    return float(slope)


def save_outputs(
    output_dir: Path,
    tensors: dict[str, np.ndarray],
    kernel_history: list[dict],
    fit_start: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = np.arange(1, next(iter(tensors.values())).shape[0] + 1)
    data_size = next(iter(tensors.values())).shape[1]
    plot_components = tuple(
        component for component in PLOT_COMPONENTS if max(component) < data_size
    )
    if not plot_components:
        plot_components = ((0, 0, 0, 0),)
    np.savez_compressed(
        output_dir / "exact_relu_tensor_recursions.npz",
        layers=layers,
        **tensors,
        covariance=np.stack([row["covariance"] for row in kernel_history]),
        ntk=np.stack([row["ntk"] for row in kernel_history]),
    )

    rows = []
    for name, history in tensors.items():
        for component in plot_components:
            values = history[(slice(None),) + component]
            rows.append(
                {
                    "tensor": name,
                    "component": "".join(map(str, component)),
                    "fit_start": fit_start,
                    "fit_end": int(layers[-1]),
                    "effective_exponent": effective_power(values, layers, fit_start),
                    "asymptotic_exponent": {"V": 1, "D": 2, "F": 2, "A": 3, "B": 3}[name],
                }
            )
    with (output_dir / "effective_exponents.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    diagnostics = {
        "max_raw_cubature_error": max(
            row["cubature_max_error"] for row in kernel_history
        ),
        "layers": [
            {
                "layer": row["layer"],
                "cubature_max_error": row["cubature_max_error"],
                "covariance_condition_number": float(
                    np.linalg.cond(row["covariance"])
                ),
            }
            for row in kernel_history
        ],
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(plot_components)))
    fig, axes = plt.subplots(1, 5, figsize=(18.0, 3.7), constrained_layout=True)
    for axis, (name, history) in zip(axes, tensors.items()):
        for color, component in zip(colors, plot_components):
            values = np.abs(history[(slice(None),) + component])
            axis.loglog(
                layers[1:],
                values[1:],
                color=color,
                linewidth=1.8,
                label="".join(map(str, component)),
            )
        target = {"V": 1, "D": 2, "F": 2, "A": 3, "B": 3}[name]
        reference = np.nanmedian(
            np.abs(history[-1][tuple(zip(*plot_components))])
        )
        reference_layers = np.asarray([max(2, fit_start), layers[-1]])
        axis.loglog(
            reference_layers,
            reference * (reference_layers / layers[-1]) ** target,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=rf"$L^{target}$",
        )
        axis.set_title(name)
        axis.set_xlabel("depth $L$")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("absolute tensor component")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=7, frameon=False)
    fig.savefig(output_dir / "exact_relu_tensor_depth_laws.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "exact_relu_tensor_depth_laws.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--weight-variance", type=float, default=2.0)
    parser.add_argument("--fit-start", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=2e-7)
    parser.add_argument("--atol", type=float, default=2e-10)
    parser.add_argument("--max-subdivisions", type=int, default=20_000)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/feynman/paper_depth_inputs.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/exact_relu_tensor_recursions"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = np.asarray(json.loads(args.input_file.read_text())["input"], dtype=float)
    config = CubatureConfig(args.rtol, args.atol, args.max_subdivisions)
    tensors, kernel_history = run_recursions(
        inputs, args.depth, args.weight_variance, config
    )
    save_outputs(args.output_dir, tensors, kernel_history, args.fit_start)
    print(f"wrote deterministic recursion results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
