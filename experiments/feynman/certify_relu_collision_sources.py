#!/usr/bin/env python3
"""Audit the five exact critical-ReLU A-source collision constants.

The theorem is analytic; this script provides an independent finite-depth
certificate from a saved deterministic recursion.  It removes the transported
A term, separates the four remaining terms, normalizes by
``s_a s_b s_c s_d L^2``, and compares them with their rational limits.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path

import numpy as np

from run_exact_relu_tensor_recursions import (
    CubatureConfig,
    OrthantMomentTable,
    _tensor_sources,
    relu_transport_operators,
)


REPRESENTATIVES = {
    "both_diagonal": (0, 0, 1, 1),
    "one_off_diagonal": (0, 0, 1, 2),
    "same_off_diagonal_pair": (0, 1, 0, 1),
    "one_shared_label": (0, 1, 0, 2),
    "four_distinct_labels": (0, 1, 2, 3),
}

EXACT_TERMS = {
    "both_diagonal": (Fraction(1, 4), 0, Fraction(3, 4), Fraction(3, 4)),
    "one_off_diagonal": (
        Fraction(1, 16),
        0,
        Fraction(3, 10),
        Fraction(13, 16),
    ),
    "same_off_diagonal_pair": (
        Fraction(1, 64),
        0,
        Fraction(111, 320),
        Fraction(111, 320),
    ),
    "one_shared_label": (
        Fraction(1, 64),
        0,
        Fraction(437, 1280),
        Fraction(437, 1280),
    ),
    "four_distinct_labels": (
        Fraction(1, 64),
        0,
        Fraction(217, 640),
        Fraction(217, 640),
    ),
}

TERM_NAMES = ("cov_omega_omega", "one_quarter_OVO", "OD_left", "OD_right")


def exact_source_total(collision_class: str) -> Fraction:
    return sum(EXACT_TERMS[collision_class], Fraction(0))


def finite_depth_terms(archive_path: Path) -> tuple[int, dict[str, np.ndarray]]:
    archive = np.load(archive_path)
    if archive["A"].shape[1] < 4:
        raise ValueError("the certificate requires at least four input labels")
    state_index = len(archive["layers"]) - 2
    depth = int(archive["layers"][state_index])
    covariance = archive["covariance"][state_index]
    ntk = archive["ntk"][state_index]
    tensors = {name: archive[name][state_index] for name in ("V", "D")}

    queries = OrthantMomentTable.build_queries(covariance.shape[0])
    moments = OrthantMomentTable(
        covariance,
        queries,
        CubatureConfig(rtol=2e-7, atol=2e-10, max_subdivisions=20_000),
    )
    sources = _tensor_sources(covariance, ntk, 2.0, moments)
    _, omega_hessian, _ = relu_transport_operators(covariance, ntk, 2.0)
    indicator_pair = sources["ii"]
    terms = {
        "cov_omega_omega": sources["cov_omega_omega"],
        "one_quarter_OVO": 0.25
        * np.einsum(
            "ghij,abgh,cdij->abcd",
            tensors["V"],
            omega_hessian,
            omega_hessian,
            optimize=True,
        ),
        "OD_left": np.einsum(
            "cd,abgh,ghcd->abcd",
            indicator_pair,
            omega_hessian,
            tensors["D"],
            optimize=True,
        ),
        "OD_right": np.einsum(
            "ab,cdgh,ghab->abcd",
            indicator_pair,
            omega_hessian,
            tensors["D"],
            optimize=True,
        ),
        "covariance": covariance,
    }
    return depth, terms


def write_certificate(archive_path: Path, output_file: Path) -> None:
    depth, arrays = finite_depth_terms(archive_path)
    standard_deviations = np.sqrt(np.diag(arrays.pop("covariance")))
    rows = []
    for collision_class, component in REPRESENTATIVES.items():
        scale = float(np.prod(standard_deviations[list(component)])) * depth**2
        finite_values = [float(arrays[name][component] / scale) for name in TERM_NAMES]
        exact_values = EXACT_TERMS[collision_class]
        for term, finite_value, exact_value in zip(
            TERM_NAMES, finite_values, exact_values
        ):
            rows.append(
                {
                    "depth": depth,
                    "collision_class": collision_class,
                    "component": "".join(map(str, component)),
                    "term": term,
                    "exact_limit": str(exact_value),
                    "finite_depth_value": finite_value,
                    "absolute_error": abs(finite_value - float(exact_value)),
                }
            )
        finite_total = sum(finite_values)
        exact_total = exact_source_total(collision_class)
        rows.append(
            {
                "depth": depth,
                "collision_class": collision_class,
                "component": "".join(map(str, component)),
                "term": "total",
                "exact_limit": str(exact_total),
                "finite_depth_value": finite_total,
                "absolute_error": abs(finite_total - float(exact_total)),
            }
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "data/feynman/exact_relu_equi_m4_depth6000/"
            "exact_relu_tensor_recursions.npz"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(
            "data/feynman/relu_tensor_asymptotics/"
            "a_collision_source_certificate.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_certificate(args.archive, args.output_file)
    print(f"wrote collision-source certificate to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
