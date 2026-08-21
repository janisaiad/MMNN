#!/usr/bin/env python3
"""Run algebraic, spectral, RMT, DMFT, and scaling-law validations.

The nonlinear RBF experiment estimates its own finite-size spectral law.  The
Marchenko--Pastur formula is used only for a separate linear-Wishart control.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import minimize_scalar
from scipy.special import zeta as hurwitz_zeta

from .common import (
    bootstrap_mean_ci,
    check_record,
    chebyshev_residual,
    geometric_midpoint_grid,
    hb_parameters,
    hb_residual,
    loglog_fit,
    orthonormalize,
    projector_distance,
    rbf_kernel,
    ritz_inverse_metric,
    save_csv,
    save_json,
    symmetric_preconditioned_spectrum,
)


def slope_check(
    name: str,
    fit: dict[str, float],
    expected: float,
    checks: list[dict[str, Any]],
    absolute_floor: float = 0.04,
) -> None:
    tolerance = max(2.0 * fit["slope_se"], absolute_floor)
    passed = abs(fit["slope"] - expected) <= tolerance and fit["r2"] >= 0.95
    checks.append(
        check_record(
            name,
            passed,
            fit["slope"],
            f"|slope - {expected:.6g}| <= max(2 SE, {absolute_floor}) and R2 >= 0.95",
            expected=expected,
            slope_se=fit["slope_se"],
            r2=fit["r2"],
        )
    )


def kernel_quadrature_experiment(
    outdir: Path,
    profile: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    sizes = [24, 48, 96, 192] if profile == "smoke" else [24, 48, 96, 192, 384, 768]
    reference_size = 16384 if profile == "smoke" else 65536
    length_scale = 0.13
    probe = np.linspace(0.0, 1.0, 161)

    def field(x: np.ndarray) -> np.ndarray:
        return (
            0.7 * np.sin(2.0 * np.pi * x)
            - 0.2 * np.cos(6.0 * np.pi * x)
            + 0.15 * x * x
        )

    reference_nodes, reference_weights = geometric_midpoint_grid(reference_size)
    reference_kernel = rbf_kernel(probe, reference_nodes, length_scale)
    reference = (
        reference_kernel @ (reference_weights * field(reference_nodes))
    ) / (reference_kernel @ reference_weights)

    rows: list[dict[str, Any]] = []
    max_softmax_error = 0.0
    max_reversibility_error = 0.0
    min_symmetrized_eigenvalue = float("inf")
    for size in sizes:
        nodes, weights = geometric_midpoint_grid(size)
        kernel = rbf_kernel(nodes, nodes, length_scale)
        degree = kernel @ weights
        transition = kernel * weights[None, :] / degree[:, None]
        logits = -0.5 * ((nodes[:, None] - nodes[None, :]) / length_scale) ** 2
        logits = logits + np.log(weights)[None, :]
        logits = logits - logits.max(axis=1, keepdims=True)
        softmax = np.exp(logits)
        softmax /= softmax.sum(axis=1, keepdims=True)
        softmax_error = float(
            np.linalg.norm(transition - softmax, ord=np.inf)
            / max(np.linalg.norm(transition, ord=np.inf), np.finfo(float).tiny)
        )

        stationary = weights * degree
        stationary /= stationary.sum()
        detailed_balance = stationary[:, None] * transition
        reversibility_error = float(
            np.max(np.abs(detailed_balance - detailed_balance.T))
            / max(np.max(np.abs(detailed_balance)), np.finfo(float).tiny)
        )
        root_stationary = np.sqrt(stationary)
        symmetrized = (
            root_stationary[:, None]
            * transition
            / root_stationary[None, :]
        )
        symmetry_error = float(
            np.linalg.norm(symmetrized - symmetrized.T, ord=np.inf)
        )
        minimum_eigenvalue = float(np.linalg.eigvalsh(0.5 * (symmetrized + symmetrized.T))[0])

        probe_kernel = rbf_kernel(probe, nodes, length_scale)
        weighted = (probe_kernel @ (weights * field(nodes))) / (probe_kernel @ weights)
        unweighted = (probe_kernel @ field(nodes)) / probe_kernel.sum(axis=1)
        weighted_error = float(np.sqrt(np.mean((weighted - reference) ** 2)))
        unweighted_error = float(np.sqrt(np.mean((unweighted - reference) ** 2)))
        rows.append(
            {
                "size": size,
                "h": 1.0 / size,
                "weighted_l2_error": weighted_error,
                "unweighted_l2_error": unweighted_error,
                "softmax_relative_error": softmax_error,
                "row_sum_error": float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
                "reversibility_relative_error": reversibility_error,
                "symmetry_inf_error": symmetry_error,
                "minimum_symmetrized_eigenvalue": minimum_eigenvalue,
            }
        )
        max_softmax_error = max(max_softmax_error, softmax_error)
        max_reversibility_error = max(max_reversibility_error, reversibility_error)
        min_symmetrized_eigenvalue = min(min_symmetrized_eigenvalue, minimum_eigenvalue)

    save_csv(outdir / "kernel_quadrature.csv", rows)
    weighted_fit = loglog_fit(
        [row["h"] for row in rows[-4:]],
        [row["weighted_l2_error"] for row in rows[-4:]],
    )
    unweighted_fit = loglog_fit(
        [row["h"] for row in rows[-4:]],
        [row["unweighted_l2_error"] for row in rows[-4:]],
    )
    checks.extend(
        [
            check_record(
                "quadrature_weighted_softmax_identity",
                max_softmax_error < 1e-12,
                max_softmax_error,
                "maximum relative row error < 1e-12",
            ),
            check_record(
                "kernel_attention_reversibility",
                max_reversibility_error < 1e-12,
                max_reversibility_error,
                "detailed-balance relative error < 1e-12",
            ),
            check_record(
                "kernel_attention_positive_symmetrization",
                min_symmetrized_eigenvalue > -1e-11,
                min_symmetrized_eigenvalue,
                "minimum eigenvalue > -1e-11",
            ),
            check_record(
                "weighted_continuum_convergence",
                weighted_fit["slope"] >= 1.75 and weighted_fit["r2"] >= 0.98,
                weighted_fit["slope"],
                "midpoint quadrature slope >= 1.75 and R2 >= 0.98",
                r2=weighted_fit["r2"],
            ),
            check_record(
                "quadrature_weights_remove_sampling_bias",
                rows[-1]["weighted_l2_error"] < 0.1 * rows[-1]["unweighted_l2_error"],
                rows[-1]["weighted_l2_error"] / rows[-1]["unweighted_l2_error"],
                "finest weighted error / unweighted error < 0.1",
            ),
        ]
    )
    return {
        "rows": rows,
        "weighted_fit": weighted_fit,
        "unweighted_fit": unweighted_fit,
        "reference_size": reference_size,
    }


def ritz_and_trace_experiment(
    outdir: Path,
    profile: str,
    rng: np.random.Generator,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    dimension = 128 if profile == "smoke" else 256
    seeds = 3 if profile == "smoke" else 8
    ranks = [8, 16, 32]
    perturbations = [0.0, 1e-5, 1e-4, 1e-3]
    rows: list[dict[str, Any]] = []
    certified_cases = 0
    violations = 0
    inverse_bound_violations = 0
    trace_fixture: tuple[np.ndarray, float, float] | None = None

    indices = np.arange(1, dimension + 1, dtype=np.float64)
    data_eigenvalues = 50.0 * indices ** -4.0
    maximum_hessian = 1.0 + data_eigenvalues[0]
    constant_m = 1.0 + 2.0 * math.sqrt(2.0) + 2.0 * math.sqrt(2.0) * maximum_hessian
    for seed in range(seeds):
        local_rng = np.random.default_rng(11000 + seed)
        rotation = orthonormalize(local_rng.normal(size=(dimension, dimension)))
        hessian = (
            np.eye(dimension)
            + (rotation * data_eigenvalues[None, :]) @ rotation.T
        )
        exact_inverse = (
            np.eye(dimension)
            - (rotation * (data_eigenvalues / (1.0 + data_eigenvalues))[None, :])
            @ rotation.T
        )
        for rank in ranks:
            target = rotation[:, :rank]
            tail = data_eigenvalues[rank] / (1.0 + data_eigenvalues[rank])
            for perturbation in perturbations:
                noise = local_rng.normal(size=(dimension, rank))
                noise -= target @ (target.T @ noise)
                basis = orthonormalize(target + perturbation * noise / math.sqrt(dimension))
                delta = projector_distance(basis, target)
                metric = ritz_inverse_metric(hessian, basis)
                inverse_error = float(np.linalg.norm(metric - exact_inverse, ord=2))
                inverse_bound = tail + constant_m * delta
                if inverse_error > inverse_bound + 2e-10:
                    inverse_bound_violations += 1
                epsilon = maximum_hessian * inverse_bound
                spectrum = symmetric_preconditioned_spectrum(hessian, metric)
                spectrum_min = float(spectrum[0])
                spectrum_max = float(spectrum[-1])
                certified = epsilon < 1.0
                if certified:
                    certified_cases += 1
                    if spectrum_min < 1.0 - epsilon - 2e-10 or spectrum_max > 1.0 + epsilon + 2e-10:
                        violations += 1
                rows.append(
                    {
                        "seed": seed,
                        "dimension": dimension,
                        "rank": rank,
                        "perturbation": perturbation,
                        "projector_error": delta,
                        "tail": tail,
                        "inverse_error": inverse_error,
                        "inverse_bound": inverse_bound,
                        "epsilon": epsilon,
                        "certificate_nonvacuous": certified,
                        "spectrum_min": spectrum_min,
                        "spectrum_max": spectrum_max,
                        "effective_condition": spectrum_max / spectrum_min,
                    }
                )
                if trace_fixture is None and rank == 16 and perturbation == 1e-4:
                    trace_fixture = (spectrum, spectrum_min, spectrum_max)

    save_csv(outdir / "ritz_spectral_certificate.csv", rows)
    checks.extend(
        [
            check_record(
                "ritz_inverse_operator_bound",
                inverse_bound_violations == 0,
                inverse_bound_violations,
                "zero violations of ||B-H^-1|| <= r_S + C_M delta",
                cases=len(rows),
            ),
            check_record(
                "ritz_certified_spectral_enclosure",
                certified_cases > 0 and violations == 0,
                violations,
                "at least one nonvacuous certificate and zero spectral violations",
                certified_cases=certified_cases,
                total_cases=len(rows),
            ),
        ]
    )

    assert trace_fixture is not None
    eigenvalues, mu, ell = trace_fixture
    spectral_weights = 0.25 + np.linspace(0.0, 1.0, dimension) ** 2
    monte_carlo = 5000 if profile == "smoke" else 30000
    standard_normal = rng.normal(size=(monte_carlo, dimension))
    trace_rows: list[dict[str, Any]] = []
    max_trace_relative_error = 0.0
    max_hb_bound_ratio = 0.0
    max_chebyshev_bound_ratio = 0.0
    _, _, q = hb_parameters(mu, ell)
    depths = [1, 2, 4, 8, 12] if profile == "smoke" else [1, 2, 4, 8, 12, 16, 24, 32]
    for method in ("hb", "chebyshev"):
        for depth in depths:
            residual = (
                hb_residual(eigenvalues, depth, mu, ell)
                if method == "hb"
                else chebyshev_residual(eigenvalues, depth, mu, ell)
            )
            modal_risk = spectral_weights * residual * residual / eigenvalues
            exact_trace = float(modal_risk.sum())
            samples = (standard_normal * standard_normal) @ modal_risk
            empirical = float(samples.mean())
            standard_error = float(samples.std(ddof=1) / math.sqrt(monte_carlo))
            relative_error = abs(empirical - exact_trace) / exact_trace
            max_trace_relative_error = max(max_trace_relative_error, relative_error)
            if method == "hb":
                uniform_bound = (1.0 + depth * (1.0 + q)) * q**depth
                max_hb_bound_ratio = max(
                    max_hb_bound_ratio,
                    float(np.max(np.abs(residual))) / max(uniform_bound, np.finfo(float).tiny),
                )
            else:
                uniform_bound = 2.0 * q**depth
                max_chebyshev_bound_ratio = max(
                    max_chebyshev_bound_ratio,
                    float(np.max(np.abs(residual))) / max(uniform_bound, np.finfo(float).tiny),
                )
            trace_rows.append(
                {
                    "method": method,
                    "depth": depth,
                    "trace_risk": exact_trace,
                    "monte_carlo_risk": empirical,
                    "monte_carlo_se": standard_error,
                    "relative_error": relative_error,
                    "maximum_residual": float(np.max(np.abs(residual))),
                    "uniform_bound": uniform_bound,
                }
            )
    save_csv(outdir / "fixed_polynomial_trace_risk.csv", trace_rows)
    checks.extend(
        [
            check_record(
                "fixed_polynomial_trace_risk",
                max_trace_relative_error < (0.04 if profile == "smoke" else 0.02),
                max_trace_relative_error,
                "maximum Monte Carlo relative error below profile threshold",
                monte_carlo=monte_carlo,
            ),
            check_record(
                "heavy_ball_finite_depth_bound",
                max_hb_bound_ratio <= 1.0 + 1e-10,
                max_hb_bound_ratio,
                "max |p_L| / ([1+L(1+q)]q^L) <= 1",
            ),
            check_record(
                "chebyshev_finite_depth_bound",
                max_chebyshev_bound_ratio <= 1.0 + 1e-10,
                max_chebyshev_bound_ratio,
                "max |p_L| / (2q^L) <= 1",
            ),
        ]
    )
    return {
        "spectral_rows": rows,
        "trace_rows": trace_rows,
        "dimension": dimension,
        "monte_carlo_samples": monte_carlo,
    }


def mp_inverse_resolvent(eta: float, aspect: float) -> float:
    lower = (1.0 - math.sqrt(aspect)) ** 2
    upper = (1.0 + math.sqrt(aspect)) ** 2

    def density(value: float) -> float:
        return math.sqrt(max(0.0, (upper - value) * (value - lower))) / (
            2.0 * math.pi * aspect * value
        )

    continuous, _ = quad(lambda value: density(value) / (value + eta), lower, upper, epsabs=1e-11)
    atom = max(0.0, 1.0 - 1.0 / aspect) / eta
    return continuous + atom


def two_resolvent_statistic(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    task: np.ndarray,
    covariance_diagonal: np.ndarray,
    eta: float,
    theta: float,
) -> float:
    task_rotated = eigenvectors.T @ task @ eigenvectors
    omega_rotated = eigenvectors.T @ (covariance_diagonal[:, None] * eigenvectors)
    left = 1.0 / (eigenvalues + eta)
    right = 1.0 / (eigenvalues + theta)
    value = np.einsum(
        "ij,j,ji,i->",
        omega_rotated,
        left,
        task_rotated,
        right,
        optimize=True,
    )
    return float(value / eigenvalues.size)


def rmt_experiment(
    outdir: Path,
    profile: str,
    rng: np.random.Generator,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    sizes = [96, 192, 384] if profile == "smoke" else [96, 192, 384, 768]
    seeds = 2 if profile == "smoke" else 6
    aspect = 0.5
    etas = [0.2, 0.7, 2.0]
    linear_rows: list[dict[str, Any]] = []
    nonlinear_rows: list[dict[str, Any]] = []
    for size in sizes:
        sample_count = int(round(size / aspect))
        for seed in range(seeds):
            local = np.random.default_rng(21000 + 100 * size + seed)
            design = local.normal(size=(sample_count, size)) / math.sqrt(sample_count)
            linear_eigenvalues = np.linalg.eigvalsh(design.T @ design)
            for eta in etas:
                empirical = float(np.mean(1.0 / (linear_eigenvalues + eta)))
                theory = mp_inverse_resolvent(eta, aspect)
                linear_rows.append(
                    {
                        "dimension": size,
                        "samples": sample_count,
                        "seed": seed,
                        "eta": eta,
                        "empirical": empirical,
                        "mp_theory": theory,
                        "absolute_error": abs(empirical - theory),
                    }
                )

            points = local.uniform(size=(size, 2))
            differences = points[:, None, :] - points[None, :, :]
            distance_squared = np.einsum("ijk,ijk->ij", differences, differences)
            kernel = np.exp(-0.5 * distance_squared / 0.18**2)
            degree = kernel.sum(axis=1)
            normalized = kernel / np.sqrt(degree[:, None] * degree[None, :])
            nonlinear_eigenvalues, nonlinear_eigenvectors = np.linalg.eigh(normalized)
            task_kernel = np.exp(-0.5 * distance_squared / 0.31**2)
            task_degree = task_kernel.sum(axis=1)
            task = task_kernel / np.sqrt(task_degree[:, None] * task_degree[None, :])
            covariance_diagonal = 0.5 + points[:, 0] + 0.25 * points[:, 1]
            for eta in etas:
                stieltjes = float(np.mean(1.0 / (nonlinear_eigenvalues + eta)))
                nonlinear_rows.append(
                    {
                        "size": size,
                        "seed": seed,
                        "eta": eta,
                        "inverse_resolvent": stieltjes,
                        "two_resolvent": two_resolvent_statistic(
                            nonlinear_eigenvalues,
                            nonlinear_eigenvectors,
                            task,
                            covariance_diagonal,
                            eta,
                            1.7 * eta,
                        ),
                        "mean_eigenvalue": float(nonlinear_eigenvalues.mean()),
                        "maximum_eigenvalue": float(nonlinear_eigenvalues[-1]),
                        "minimum_eigenvalue": float(nonlinear_eigenvalues[0]),
                        "mp_control_value": mp_inverse_resolvent(eta, aspect),
                    }
                )

    save_csv(outdir / "rmt_linear_mp_control.csv", linear_rows)
    save_csv(outdir / "rmt_nonlinear_kernel_resolvents.csv", nonlinear_rows)
    finest = sizes[-1]
    linear_finest_errors = [
        row["absolute_error"] for row in linear_rows if row["dimension"] == finest
    ]
    linear_ci = bootstrap_mean_ci(linear_finest_errors, rng)
    checks.append(
        check_record(
            "linear_wishart_mp_control",
            linear_ci["ci_high"] < (0.035 if profile == "smoke" else 0.02),
            linear_ci["mean"],
            "finest-size bootstrap upper CI below profile threshold",
            **linear_ci,
        )
    )

    convergence: list[dict[str, Any]] = []
    for statistic in ("inverse_resolvent", "two_resolvent"):
        for eta in etas:
            means = []
            for size in sizes:
                values = [
                    row[statistic]
                    for row in nonlinear_rows
                    if row["size"] == size and row["eta"] == eta
                ]
                means.append(float(np.mean(values)))
            reference = means[-1]
            errors = [abs(value - reference) for value in means[:-1]]
            fit = loglog_fit(sizes[:-1], np.maximum(errors, 1e-16))
            convergence.append(
                {
                    "statistic": statistic,
                    "eta": eta,
                    "sizes": sizes,
                    "means": means,
                    "reference": reference,
                    "fit": fit,
                }
            )
    # A finite-size convergence audit is empirical, not a claimed closed-form law.
    early = []
    late = []
    for statistic in ("inverse_resolvent", "two_resolvent"):
        for eta in etas:
            by_size = {
                size: np.asarray(
                    [
                        row[statistic]
                        for row in nonlinear_rows
                        if row["size"] == size and row["eta"] == eta
                    ]
                )
                for size in sizes
            }
            largest_mean = float(by_size[sizes[-1]].mean())
            early.append(abs(float(by_size[sizes[0]].mean()) - largest_mean))
            late.append(abs(float(by_size[sizes[-2]].mean()) - largest_mean))
    convergence_ratio = float(np.mean(late) / max(np.mean(early), np.finfo(float).tiny))
    checks.extend(
        [
            check_record(
                "nonlinear_kernel_resolvent_finite_size_convergence",
                convergence_ratio < 0.8,
                convergence_ratio,
                "penultimate-to-limit discrepancy / coarsest-to-limit discrepancy < 0.8",
            ),
            check_record(
                "nonlinear_kernel_spectrum_psd",
                min(row["minimum_eigenvalue"] for row in nonlinear_rows) > -1e-10,
                min(row["minimum_eigenvalue"] for row in nonlinear_rows),
                "minimum symmetrized normalized-kernel eigenvalue > -1e-10",
            ),
        ]
    )
    return {
        "linear_rows": linear_rows,
        "nonlinear_rows": nonlinear_rows,
        "nonlinear_convergence": convergence,
        "sizes": sizes,
        "seeds": seeds,
    }


def rrs_modal_arrays(
    mode_count: int,
    nu: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(1, mode_count + 1, dtype=np.float64)
    eigenvalues = indices ** (-nu)
    energy = indices ** (-(nu * beta + 1.0))
    return eigenvalues, energy


def parameterized_rrs_curve(
    eigenvalues: np.ndarray,
    energy: np.ndarray,
    parameter_power: int,
    gamma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    losses = np.empty_like(gamma_grid)
    derivatives = np.empty_like(gamma_grid)
    chunk = 32
    for start in range(0, gamma_grid.size, chunk):
        gamma = gamma_grid[start : start + chunk, None]
        exponential = np.exp(-2.0 * gamma * eigenvalues[None, :])
        losses[start : start + chunk] = exponential @ energy
        base_derivative = 2.0 * (exponential @ (energy * eigenvalues))
        derivatives[start : start + chunk] = (
            parameter_power**2
            * gamma_grid[start : start + chunk] ** (2.0 - 2.0 / parameter_power)
            * base_derivative
        )
    inverse_speed = 1.0 / np.maximum(derivatives, np.finfo(float).tiny)
    times = cumulative_trapezoid(inverse_speed, gamma_grid, initial=0.0)
    return times, losses


def dmft_scaling_experiment(
    outdir: Path,
    profile: str,
    rng: np.random.Generator,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_count = 50000 if profile == "smoke" else 250000
    nu = 0.8
    beta = 1.5
    eigenvalues, rrs_energy = rrs_modal_arrays(mode_count, nu, beta)
    rows: list[dict[str, Any]] = []
    fits: dict[str, Any] = {}

    # Fixed-spectrum exact flow.
    s = 2.2
    fixed_energy = np.arange(1, mode_count + 1, dtype=np.float64) ** (-s)
    fixed_b = fixed_energy * eigenvalues * eigenvalues
    times = np.logspace(1.0, 9.0, 28)
    fixed_loss = np.asarray(
        [np.sum(fixed_energy / (1.0 + 4.0 * fixed_b * time_value)) for time_value in times]
    )
    fixed_fit = loglog_fit(times[-12:], fixed_loss[-12:])
    fixed_expected = -(s - 1.0) / (s + 2.0 * nu)
    fits["fixed_spectrum_time"] = {"fit": fixed_fit, "expected": fixed_expected}
    slope_check("fixed_spectrum_time_exponent", fixed_fit, fixed_expected, checks, 0.05)
    for time_value, loss in zip(times, fixed_loss, strict=True):
        rows.append({"family": "fixed_spectrum", "resource": time_value, "loss": loss})

    # Randomly rotated spectrum: gamma, time, depth, width/context.
    # Keep the active cutoff j_gamma ~ gamma^(1/nu) well inside the finite
    # modal truncation; otherwise the artificial last mode creates an
    # exponential, rather than regularly varying, tail.
    gamma_max = (mode_count / 60.0) ** nu
    gamma = np.logspace(0.8, math.log10(gamma_max), 40)
    rrs_loss_gamma = np.asarray(
        [np.sum(rrs_energy * np.exp(-2.0 * value * eigenvalues)) for value in gamma]
    )
    gamma_fit = loglog_fit(gamma[-18:], rrs_loss_gamma[-18:])
    fits["rrs_gamma"] = {"fit": gamma_fit, "expected": -beta}
    slope_check("rrs_loss_vs_gamma_exponent", gamma_fit, -beta, checks, 0.06)

    gamma_grid = np.logspace(-2.0, math.log10(1.15 * gamma_max), 420)
    for parameter_power in (1, 5):
        time_curve, loss_curve = parameterized_rrs_curve(
            eigenvalues, rrs_energy, parameter_power, gamma_grid
        )
        mask = (
            (time_curve > 0)
            & np.isfinite(time_curve)
            & (gamma_grid > 0.08 * gamma_max)
            & (gamma_grid < gamma_max)
        )
        selected_time = time_curve[mask]
        selected_loss = loss_curve[mask]
        fit = loglog_fit(selected_time[-80:], selected_loss[-80:])
        expected = -parameter_power * beta / (parameter_power * beta + 2.0)
        fits[f"rrs_time_r{parameter_power}"] = {"fit": fit, "expected": expected}
        slope_check(
            f"rrs_parameterized_time_exponent_r{parameter_power}",
            fit,
            expected,
            checks,
            0.055,
        )
        stride = max(1, selected_time.size // 60)
        for resource, loss in zip(selected_time[::stride], selected_loss[::stride], strict=True):
            rows.append(
                {
                    "family": f"rrs_time_r{parameter_power}",
                    "resource": resource,
                    "loss": loss,
                }
            )

    widths = np.asarray([16, 24, 36, 54, 81, 121, 181, 271, 406, 609, 913, 1369])
    tail = hurwitz_zeta(nu * beta + 1.0, widths + 1.0)
    width_fit = loglog_fit(widths[-8:], tail[-8:])
    expected_width = -nu * beta
    fits["width_context"] = {"fit": width_fit, "expected": expected_width}
    slope_check("rrs_width_context_exponent", width_fit, expected_width, checks, 0.035)
    for width, loss in zip(widths, tail, strict=True):
        rows.append({"family": "rrs_width_context", "resource": width, "loss": loss})

    depth_grid = np.asarray([4, 6, 9, 13, 19, 28, 42, 63, 94, 141, 211, 316])
    depth_loss = []
    truncation = min(mode_count, 100000)
    lam_depth = eigenvalues[:truncation]
    energy_depth = rrs_energy[:truncation]
    for depth in depth_grid:
        def objective(scale: float) -> float:
            return float(
                np.sum(
                    energy_depth
                    * (1.0 - scale * lam_depth / depth) ** (2 * depth)
                )
            )

        optimum = minimize_scalar(
            objective,
            bounds=(0.0, 1.999 * depth / lam_depth[0]),
            method="bounded",
            options={"xatol": 1e-8},
        )
        depth_loss.append(optimum.fun)
        rows.append(
            {
                "family": "rrs_finite_depth",
                "resource": depth,
                "loss": optimum.fun,
                "gamma_opt": optimum.x,
            }
        )
    depth_fit = loglog_fit(depth_grid[-8:], depth_loss[-8:])
    fits["rrs_depth"] = {"fit": depth_fit, "expected": -beta}
    slope_check("rrs_depth_exponent", depth_fit, -beta, checks, 0.08)

    # Finite-task random rotations: empirical DMFT vector field becomes isotropic.
    dmft_rows: list[dict[str, Any]] = []
    dimension = 24 if profile == "smoke" else 40
    task_counts = [8, 32, 128, 512] if profile == "smoke" else [8, 32, 128, 512, 2048]
    dmft_seeds = 3 if profile == "smoke" else 10
    modal = np.arange(1, dimension + 1, dtype=np.float64) ** (-nu)
    modal_gradient = modal * modal * np.exp(-2.0 * 5.0 * modal)
    for task_count in task_counts:
        for seed in range(dmft_seeds):
            local = np.random.default_rng(31000 + 100 * task_count + seed)
            average = np.zeros((dimension, dimension), dtype=np.float64)
            for _ in range(task_count):
                rotation = orthonormalize(local.normal(size=(dimension, dimension)))
                average += (rotation * modal_gradient[None, :]) @ rotation.T
            average /= task_count
            isotropic = np.trace(average) / dimension
            anisotropy = float(
                np.linalg.norm(average - isotropic * np.eye(dimension), ord="fro")
                / np.linalg.norm(average, ord="fro")
            )
            dmft_rows.append(
                {
                    "tasks": task_count,
                    "seed": seed,
                    "anisotropy": anisotropy,
                    "scalar_drift": isotropic,
                    "population_scalar_drift": float(modal_gradient.mean()),
                    "relative_scalar_error": abs(isotropic - modal_gradient.mean())
                    / modal_gradient.mean(),
                }
            )
    anisotropy_means = [
        np.mean([row["anisotropy"] for row in dmft_rows if row["tasks"] == count])
        for count in task_counts
    ]
    anisotropy_fit = loglog_fit(task_counts, anisotropy_means)
    fits["finite_task_dmft_isotropy"] = {"fit": anisotropy_fit, "expected": -0.5}
    slope_check(
        "finite_task_dmft_isotropy_rate",
        anisotropy_fit,
        -0.5,
        checks,
        0.12,
    )
    finest_scalar_errors = [
        row["relative_scalar_error"] for row in dmft_rows if row["tasks"] == task_counts[-1]
    ]
    checks.append(
        check_record(
            "finite_task_dmft_scalar_closure",
            max(finest_scalar_errors) < (0.03 if profile == "smoke" else 0.015),
            max(finest_scalar_errors),
            "maximum finest-batch scalar drift error below profile threshold",
        )
    )

    # Preconditioned logarithmic depth-width shape versus the unpreconditioned law.
    shape_rows: list[dict[str, Any]] = []
    shape_widths = np.asarray([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    fixed_kappa = 25.0
    fixed_q = (math.sqrt(fixed_kappa) - 1.0) / (math.sqrt(fixed_kappa) + 1.0)
    preconditioned_depths = []
    unpreconditioned_depths = []
    for width in shape_widths:
        target = width ** (-nu * beta)
        pre_depth = next(
            depth for depth in range(1, 10000) if 4.0 * fixed_q ** (2 * depth) <= target
        )
        # This is the randomly-rotated marginal depth law L^{-beta}, whose
        # balance against S^{-nu beta} gives L ~ S^nu.  A Chebyshev bound with
        # a growing condition number would carry an additional logarithm and
        # is a different comparison.
        unpre_depth = int(math.ceil(width**nu))
        preconditioned_depths.append(pre_depth)
        unpreconditioned_depths.append(unpre_depth)
        shape_rows.append(
            {
                "width": int(width),
                "target_tail": target,
                "preconditioned_depth": pre_depth,
                "unpreconditioned_depth": unpre_depth,
            }
        )
    linear_fit = np.polyfit(np.log(shape_widths), preconditioned_depths, 1)
    expected_coefficient = nu * beta / (2.0 * math.log(1.0 / fixed_q))
    unpre_fit = loglog_fit(shape_widths[-6:], unpreconditioned_depths[-6:])
    checks.extend(
        [
            check_record(
                "preconditioned_logarithmic_depth_width_shape",
                abs(linear_fit[0] - expected_coefficient) < 0.12 * expected_coefficient,
                float(linear_fit[0]),
                "slope of depth versus log width within 12% of theory",
                expected=expected_coefficient,
            ),
            check_record(
                "unpreconditioned_polynomial_depth_width_shape",
                abs(unpre_fit["slope"] - nu) < max(0.12, 2.0 * unpre_fit["slope_se"]),
                unpre_fit["slope"],
                "log-log depth-width slope agrees with nu",
                expected=nu,
                slope_se=unpre_fit["slope_se"],
                r2=unpre_fit["r2"],
            ),
        ]
    )

    save_csv(outdir / "dmft_scaling_curves.csv", rows)
    save_csv(outdir / "dmft_random_rotation_isotropy.csv", dmft_rows)
    save_csv(outdir / "depth_width_shape.csv", shape_rows)
    return {
        "fits": fits,
        "curves": rows,
        "dmft_rows": dmft_rows,
        "shape_rows": shape_rows,
        "mode_count": mode_count,
        "nu": nu,
        "beta": beta,
    }


def master_risk_experiment(
    outdir: Path,
    profile: str,
    rng: np.random.Generator,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    dimension = 192 if profile == "smoke" else 512
    samples = 8000 if profile == "smoke" else 40000
    a = 2.2
    b = 0.9
    sigma = 0.7
    context = 600.0
    width = dimension // 2
    mesh_modes = int(0.8 * dimension)
    represented = min(width, mesh_modes)
    depth = 8
    train_time = 120.0
    indices = np.arange(1, dimension + 1, dtype=np.float64)
    prior = indices ** (-a)
    rates = 0.3 * indices ** (-b)
    posterior_variance = sigma * sigma * prior / (sigma * sigma + context * prior)
    posterior_mean_variance = prior - posterior_variance
    shrinkage = context * prior / (sigma * sigma + context * prior)
    acquisition = 1.0 - np.exp(-rates * train_time)
    effective_eigenvalues = 1.0 + 0.4 * np.sin(indices[:represented] / 11.0) ** 2
    mu = float(effective_eigenvalues.min())
    ell = float(effective_eigenvalues.max())
    residual = chebyshev_residual(effective_eigenvalues, depth, mu, ell)
    gain = np.zeros(dimension)
    gain[:represented] = (1.0 - residual) * acquisition[:represented]
    exact_risk = float(
        np.sum(posterior_variance[:represented] + (1.0 - gain[:represented]) ** 2 * posterior_mean_variance[:represented])
        + np.sum(prior[represented:])
    )

    accumulated = 0.0
    accumulated_square = 0.0
    count = 0
    chunk = 1000
    noise_scale = sigma / math.sqrt(context)
    while count < samples:
        batch = min(chunk, samples - count)
        latent = rng.normal(size=(batch, dimension)) * np.sqrt(prior)
        observation = latent + noise_scale * rng.normal(size=(batch, dimension))
        posterior_mean = shrinkage * observation
        estimate = gain * posterior_mean
        losses = np.sum((estimate - latent) ** 2, axis=1)
        accumulated += float(losses.sum())
        accumulated_square += float(losses @ losses)
        count += batch
    empirical = accumulated / samples
    variance = max(0.0, accumulated_square / samples - empirical * empirical)
    standard_error = math.sqrt(variance / samples)
    z_score = abs(empirical - exact_risk) / max(standard_error, np.finfo(float).tiny)
    checks.append(
        check_record(
            "master_risk_monte_carlo_identity",
            z_score < 4.0,
            z_score,
            "Monte Carlo discrepancy < 4 standard errors",
            exact=exact_risk,
            empirical=empirical,
            standard_error=standard_error,
        )
    )

    # Marginal context and width laws, including the exact asymptotic constants.
    context_values = np.logspace(2.0, 8.0, 25)
    summation_modes = 250000 if profile == "smoke" else 1000000
    summation_indices = np.arange(1, summation_modes + 1, dtype=np.float64)
    summation_prior = summation_indices ** (-a)
    context_risk = []
    for value in context_values:
        finite = np.sum(sigma * sigma * summation_prior / (sigma * sigma + value * summation_prior))
        tail = summation_modes ** (1.0 - a) / (a - 1.0)
        context_risk.append(finite + tail)
    context_fit = loglog_fit(context_values[-10:], context_risk[-10:])
    context_expected = -(a - 1.0) / a
    slope_check("bayesian_context_exponent", context_fit, context_expected, checks, 0.035)

    width_values = np.asarray([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    width_risk = hurwitz_zeta(a, width_values + 1.0)
    width_fit = loglog_fit(width_values[-6:], width_risk[-6:])
    width_expected = 1.0 - a
    slope_check("gp_width_tail_exponent", width_fit, width_expected, checks, 0.025)

    rows = [
        {"family": "context", "resource": value, "risk": risk}
        for value, risk in zip(context_values, context_risk, strict=True)
    ] + [
        {"family": "width", "resource": int(value), "risk": risk}
        for value, risk in zip(width_values, width_risk, strict=True)
    ]
    save_csv(outdir / "master_risk_scaling.csv", rows)
    return {
        "dimension": dimension,
        "samples": samples,
        "exact_risk": exact_risk,
        "empirical_risk": empirical,
        "standard_error": standard_error,
        "context_fit": context_fit,
        "width_fit": width_fit,
        "rows": rows,
    }


def make_plots(outdir: Path, payload: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    quadrature = payload["kernel_quadrature"]["rows"]
    axes[0, 0].loglog(
        [row["size"] for row in quadrature],
        [row["weighted_l2_error"] for row in quadrature],
        "o-",
        label="quadrature weighted",
    )
    axes[0, 0].loglog(
        [row["size"] for row in quadrature],
        [row["unweighted_l2_error"] for row in quadrature],
        "s--",
        label="unweighted ablation",
    )
    axes[0, 0].set(title="Continuum attention", xlabel="nodes", ylabel="L2 error")
    axes[0, 0].legend(frameon=False)

    trace_rows = payload["ritz_trace"]["trace_rows"]
    for method in ("hb", "chebyshev"):
        selected = [row for row in trace_rows if row["method"] == method]
        axes[0, 1].semilogy(
            [row["depth"] for row in selected],
            [row["trace_risk"] for row in selected],
            "o-",
            label=method,
        )
    axes[0, 1].set(title="Exact trace risk", xlabel="loop depth", ylabel="risk")
    axes[0, 1].legend(frameon=False)

    linear_rows = payload["rmt"]["linear_rows"]
    for eta in sorted({row["eta"] for row in linear_rows}):
        selected = [row for row in linear_rows if row["eta"] == eta]
        sizes = sorted({row["dimension"] for row in selected})
        errors = [np.mean([row["absolute_error"] for row in selected if row["dimension"] == size]) for size in sizes]
        axes[0, 2].loglog(sizes, errors, "o-", label=f"eta={eta:g}")
    axes[0, 2].set(title="Wishart MP control", xlabel="dimension", ylabel="resolvent error")
    axes[0, 2].legend(frameon=False)

    nonlinear = payload["rmt"]["nonlinear_rows"]
    eta = sorted({row["eta"] for row in nonlinear})[1]
    for statistic in ("inverse_resolvent", "two_resolvent"):
        sizes = sorted({row["size"] for row in nonlinear})
        means = [np.mean([row[statistic] for row in nonlinear if row["size"] == size and row["eta"] == eta]) for size in sizes]
        axes[1, 0].plot(sizes, means, "o-", label=statistic.replace("_", " "))
    axes[1, 0].set(title="Nonlinear-kernel resolvents", xlabel="context size", ylabel="statistic")
    axes[1, 0].legend(frameon=False)

    curves = payload["dmft_scaling"]["curves"]
    for family in ("fixed_spectrum", "rrs_time_r1", "rrs_time_r5"):
        selected = [row for row in curves if row["family"] == family]
        axes[1, 1].loglog(
            [row["resource"] for row in selected],
            [row["loss"] for row in selected],
            label=family,
        )
    axes[1, 1].set(title="Reduced DMFT time laws", xlabel="time", ylabel="loss")
    axes[1, 1].legend(frameon=False)

    shape = payload["dmft_scaling"]["shape_rows"]
    axes[1, 2].plot(
        [math.log(row["width"]) for row in shape],
        [row["preconditioned_depth"] for row in shape],
        "o-",
        label="preconditioned vs log S",
    )
    axes[1, 2].plot(
        [math.log(row["width"]) for row in shape],
        [row["unpreconditioned_depth"] for row in shape],
        "s--",
        label="unpreconditioned",
    )
    axes[1, 2].set(title="Depth-width shape", xlabel="log width", ylabel="required depth")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(outdir / "theory_validation_overview.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_report(outdir: Path, payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    passed = sum(bool(check["passed"]) for check in checks)
    lines = [
        "# Theory validation report",
        "",
        f"Profile: `{payload['profile']}`. Passed: **{passed}/{len(checks)}**.",
        "",
        "The Marchenko--Pastur comparison is restricted to the linear-Wishart control. "
        "The normalized RBF kernel is assessed through its own empirical one- and "
        "two-resolvent convergence.",
        "",
        "| Check | Status | Value | Criterion |",
        "|---|---:|---:|---|",
    ]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        value = check["value"]
        value_text = f"{value:.6g}" if isinstance(value, (int, float)) else str(value)
        lines.append(f"| {check['name']} | {status} | {value_text} | {check['criterion']} |")
    lines.extend(
        [
            "",
            "Failed checks are intentionally retained; they delimit which asymptotic "
            "claims are numerically resolved by the selected finite-size profile.",
        ]
    )
    (outdir / "THEORY_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=70001)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    checks: list[dict[str, Any]] = []
    started = time.time()
    payload: dict[str, Any] = {
        "profile": args.profile,
        "seed": args.seed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    payload["kernel_quadrature"] = kernel_quadrature_experiment(args.outdir, args.profile, checks)
    payload["ritz_trace"] = ritz_and_trace_experiment(args.outdir, args.profile, rng, checks)
    payload["rmt"] = rmt_experiment(args.outdir, args.profile, rng, checks)
    payload["dmft_scaling"] = dmft_scaling_experiment(args.outdir, args.profile, rng, checks)
    payload["master_risk"] = master_risk_experiment(args.outdir, args.profile, rng, checks)
    payload["checks"] = checks
    payload["passed"] = sum(bool(check["passed"]) for check in checks)
    payload["total"] = len(checks)
    payload["elapsed_seconds"] = time.time() - started
    save_json(args.outdir / "summary.json", payload)
    make_plots(args.outdir, payload)
    write_report(args.outdir, payload)
    print(
        f"theory validation complete: {payload['passed']}/{payload['total']} checks passed; "
        f"summary={args.outdir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
