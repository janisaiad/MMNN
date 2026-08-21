#!/usr/bin/env python3
"""Validate quadrature-aware feature and Ritz transfer across nonuniform meshes."""

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

from .common import (
    check_record,
    geometric_midpoint_grid,
    loglog_fit,
    orthonormalize,
    projector_distance,
    ritz_inverse_metric,
    save_csv,
    save_json,
    symmetric_preconditioned_spectrum,
)


def periodic_rbf(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    sine = np.sin(np.pi * (left[:, None] - right[None, :]))
    return np.exp(-2.0 * sine * sine / (length_scale * length_scale))


def fourier_fields(nodes: np.ndarray, modes: int) -> np.ndarray:
    fields = [np.ones_like(nodes)]
    frequency = 1
    while len(fields) < modes:
        fields.append(math.sqrt(2.0) * np.cos(2.0 * np.pi * frequency * nodes))
        if len(fields) < modes:
            fields.append(math.sqrt(2.0) * np.sin(2.0 * np.pi * frequency * nodes))
        frequency += 1
    return np.column_stack(fields[:modes])


def polar_isometry(matrix: np.ndarray) -> np.ndarray:
    gram = matrix.T @ matrix
    values, vectors = np.linalg.eigh(gram)
    inverse_root = (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T
    return matrix @ inverse_root


def mesh_metric(
    size: int,
    modes: int,
    feature_rank: int,
    length_scale: float,
    weighted: bool,
) -> dict[str, Any]:
    nodes, weights = geometric_midpoint_grid(size, warp=0.17)
    modal_values = fourier_fields(nodes, modes)
    modal_isometry = polar_isometry(np.sqrt(weights)[:, None] * modal_values)
    data_eigenvalues = 40.0 * np.arange(1, modes + 1, dtype=np.float64) ** -3.5
    hessian = (
        np.eye(size)
        + (modal_isometry * data_eigenvalues[None, :]) @ modal_isometry.T
    )
    kernel = periodic_rbf(nodes, nodes, length_scale)
    attention_weights = weights if weighted else np.full(size, 1.0 / size)
    transition = kernel * attention_weights[None, :]
    transition /= transition.sum(axis=1, keepdims=True)
    seed_fields = modal_values[:, :feature_rank]
    contextual_fields = transition @ seed_fields
    basis = orthonormalize(np.sqrt(weights)[:, None] * contextual_fields, feature_rank)
    metric = ritz_inverse_metric(hessian, basis)
    lifted_metric = modal_isometry.T @ metric @ modal_isometry
    lifted_projector = modal_isometry.T @ (basis @ basis.T) @ modal_isometry
    target_projector = np.zeros((modes, modes), dtype=np.float64)
    target_projector[:feature_rank, :feature_rank] = np.eye(feature_rank)
    target_metric = np.eye(modes, dtype=np.float64)
    target_metric[:feature_rank, :feature_rank] = np.diag(
        1.0 / (1.0 + data_eigenvalues[:feature_rank])
    )
    spectrum = symmetric_preconditioned_spectrum(hessian, metric)
    return {
        "size": size,
        "h": 1.0 / size,
        "weighted": weighted,
        "projector_error": float(np.linalg.norm(lifted_projector - target_projector, ord=2)),
        "metric_error": float(np.linalg.norm(lifted_metric - target_metric, ord=2)),
        "direct_projector_error": projector_distance(basis, modal_isometry[:, :feature_rank]),
        "effective_min": float(spectrum[0]),
        "effective_max": float(spectrum[-1]),
        "effective_condition": float(spectrum[-1] / spectrum[0]),
        "lifted_metric": lifted_metric,
        "lifted_projector": lifted_projector,
    }


def run(profile: str, outdir: Path) -> dict[str, Any]:
    sizes = [32, 48, 72, 108, 162] if profile == "smoke" else [32, 48, 72, 108, 162, 243, 364, 546]
    modes = 16
    feature_rank = 8
    length_scale = 0.18
    checks: list[dict[str, Any]] = []
    records = [
        mesh_metric(size, modes, feature_rank, length_scale, weighted)
        for weighted in (True, False)
        for size in sizes
    ]
    csv_rows = [
        {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}
        for record in records
    ]

    commutator_rows: list[dict[str, Any]] = []
    for weighted in (True, False):
        selected = sorted(
            [record for record in records if record["weighted"] == weighted],
            key=lambda record: record["size"],
        )
        reference = selected[-1]["lifted_metric"]
        for record in selected[:-1]:
            # In common Fourier coordinates this is the exact-lift version of
            # ||I_hh' B_h - B_h' I_hh'||.
            commutator = float(np.linalg.norm(record["lifted_metric"] - reference, ord=2))
            commutator_rows.append(
                {
                    "weighted": weighted,
                    "size": record["size"],
                    "reference_size": selected[-1]["size"],
                    "h": record["h"],
                    "commutator": commutator,
                }
            )

    save_csv(outdir / "mesh_transfer.csv", csv_rows)
    save_csv(outdir / "mesh_commutator.csv", commutator_rows)

    weighted_records = [record for record in records if record["weighted"]]
    unweighted_records = [record for record in records if not record["weighted"]]
    weighted_metric_fit = loglog_fit(
        [record["h"] for record in weighted_records[-5:]],
        [record["metric_error"] for record in weighted_records[-5:]],
    )
    weighted_projector_fit = loglog_fit(
        [record["h"] for record in weighted_records[-5:]],
        [record["projector_error"] for record in weighted_records[-5:]],
    )
    weighted_commutator = [row for row in commutator_rows if row["weighted"]]
    commutator_fit = loglog_fit(
        [row["h"] for row in weighted_commutator[-4:]],
        [row["commutator"] for row in weighted_commutator[-4:]],
    )
    condition_numbers = [record["effective_condition"] for record in weighted_records]
    finest_bias_ratio = (
        unweighted_records[-1]["metric_error"]
        / max(weighted_records[-1]["metric_error"], np.finfo(float).tiny)
    )
    checks.extend(
        [
            check_record(
                "weighted_feature_projector_convergence",
                weighted_projector_fit["slope"] >= 1.5 and weighted_projector_fit["r2"] >= 0.95,
                weighted_projector_fit["slope"],
                "projector error slope >= 1.5 and R2 >= 0.95",
                r2=weighted_projector_fit["r2"],
            ),
            check_record(
                "weighted_ritz_metric_convergence",
                weighted_metric_fit["slope"] >= 1.5 and weighted_metric_fit["r2"] >= 0.95,
                weighted_metric_fit["slope"],
                "metric error slope >= 1.5 and R2 >= 0.95",
                r2=weighted_metric_fit["r2"],
            ),
            check_record(
                "ritz_transfer_commutator_convergence",
                commutator_fit["slope"] >= 1.3 and commutator_fit["r2"] >= 0.9,
                commutator_fit["slope"],
                "common-lift commutator slope >= 1.3 and R2 >= 0.9",
                r2=commutator_fit["r2"],
            ),
            check_record(
                "mesh_uniform_effective_condition",
                max(condition_numbers) / min(condition_numbers) < 1.2,
                max(condition_numbers) / min(condition_numbers),
                "max/min effective condition over meshes < 1.2",
                finest_condition=condition_numbers[-1],
            ),
            check_record(
                "unweighted_mesh_bias_ablation",
                finest_bias_ratio > 10.0,
                finest_bias_ratio,
                "finest unweighted / weighted metric error > 10",
            ),
        ]
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for weighted, label in ((True, "quadrature weighted"), (False, "unweighted")):
        selected = [record for record in records if record["weighted"] == weighted]
        axes[0].loglog(
            [record["size"] for record in selected],
            [record["projector_error"] for record in selected],
            "o-",
            label=label,
        )
        axes[1].loglog(
            [record["size"] for record in selected],
            [record["metric_error"] for record in selected],
            "o-",
            label=label,
        )
        axes[2].plot(
            [record["size"] for record in selected],
            [record["effective_condition"] for record in selected],
            "o-",
            label=label,
        )
    axes[0].set(title="Feature covariance", xlabel="mesh nodes", ylabel="projector error")
    axes[1].set(title="Ritz transfer", xlabel="mesh nodes", ylabel="metric error")
    axes[2].set(title="Mesh-independent spectrum", xlabel="mesh nodes", ylabel="condition number")
    for axis in axes:
        axis.grid(which="both", alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(outdir / "discretization_validation.png", dpi=190, bbox_inches="tight")
    plt.close(figure)

    payload = {
        "profile": profile,
        "sizes": sizes,
        "modes": modes,
        "feature_rank": feature_rank,
        "length_scale": length_scale,
        "weighted_metric_fit": weighted_metric_fit,
        "weighted_projector_fit": weighted_projector_fit,
        "commutator_fit": commutator_fit,
        "rows": csv_rows,
        "commutator_rows": commutator_rows,
        "checks": checks,
        "passed": sum(bool(check["passed"]) for check in checks),
        "total": len(checks),
    }
    save_json(outdir / "summary.json", payload)
    lines = [
        "# Discretization validation report",
        "",
        f"Passed: **{payload['passed']}/{payload['total']}**.",
        "",
        "| Check | Status | Value | Criterion |",
        "|---|---:|---:|---|",
    ]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"| {check['name']} | {status} | {float(check['value']):.6g} | {check['criterion']} |"
        )
    (outdir / "DISCRETIZATION_VALIDATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = run(args.profile, args.outdir)
    payload["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    payload["elapsed_seconds"] = time.time() - started
    save_json(args.outdir / "summary.json", payload)
    print(
        f"discretization validation complete: {payload['passed']}/{payload['total']} checks passed; "
        f"summary={args.outdir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
