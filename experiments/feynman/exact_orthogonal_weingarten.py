#!/usr/bin/env python3
"""Exact orthogonal-Weingarten calculus for Haar-Stiefel projectors.

For U in St(n,r), P=UU^T, and G=(n/r)P, this module evaluates arbitrary
fixed-order entry moments and connected cumulants.  The implementation is a
literal finite-dimensional realization of

  E prod_s O[I_s,A_s]
    = sum_{pi,sigma in P_2(2k)} delta_I^pi delta_A^sigma Wg^O_n(pi,sigma).

The column indices of each projector insertion occur in identical pairs.
Summing them over 1,...,r gives r**loops(sigma join tau), where tau is the
canonical pairing (0,1)(2,3)... .  No large-n or large-r approximation is
used.  Exact arithmetic is provided by SymPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp


Pairing = tuple[tuple[int, int], ...]


@lru_cache(maxsize=None)
def pair_partitions(num_points: int) -> tuple[Pairing, ...]:
    """All perfect matchings of range(num_points), in canonical order."""
    if num_points < 0 or num_points % 2:
        raise ValueError("the number of points must be a nonnegative even integer")
    if num_points == 0:
        return ((),)
    first = 0
    matchings = []
    for partner in range(1, num_points):
        remaining = [index for index in range(num_points) if index not in (first, partner)]
        for submatching in pair_partitions(num_points - 2):
            relabelled = tuple(
                tuple(sorted((remaining[a], remaining[b]))) for a, b in submatching
            )
            matching = tuple(sorted(((first, partner), *relabelled)))
            matchings.append(matching)
    return tuple(sorted(set(matchings)))


def canonical_pairing(order: int) -> Pairing:
    return tuple((2 * index, 2 * index + 1) for index in range(order))


def join_component_count(left: Pairing, right: Pairing) -> int:
    """Number of connected components in the union of two pairings."""
    points = {point for pair in left + right for point in pair}
    adjacency = {point: [] for point in points}
    for a, b in left + right:
        adjacency[a].append(b)
        adjacency[b].append(a)
    components = 0
    unseen = set(points)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return components


def coset_type(pairing: Pairing, reference: Pairing | None = None) -> tuple[int, ...]:
    """Partition of k given by half-cycle lengths of pairing union reference."""
    if reference is None:
        reference = canonical_pairing(len(pairing))
    points = {point for pair in pairing + reference for point in pair}
    adjacency = {point: [] for point in points}
    for a, b in pairing + reference:
        adjacency[a].append(b)
        adjacency[b].append(a)
    sizes = []
    unseen = set(points)
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size // 2)
    return tuple(sorted(sizes, reverse=True))


@lru_cache(maxsize=32)
def orthogonal_gram_matrix(order: int, dimension) -> sp.Matrix:
    pairings = pair_partitions(2 * order)
    n = sp.sympify(dimension)
    return sp.Matrix(
        [
            [n ** join_component_count(left, right) for right in pairings]
            for left in pairings
        ]
    )


@lru_cache(maxsize=32)
def orthogonal_weingarten_matrix(order: int, dimension) -> sp.Matrix:
    """Exact inverse Gram matrix Wg^O_n for generic/admissible n."""
    return orthogonal_gram_matrix(order, dimension).inv()


@lru_cache(maxsize=128)
def projector_moment_coefficients(
    order: int, dimension, rank, normalized: bool = True
) -> tuple[tuple[Pairing, sp.Expr], ...]:
    """Invariant-pairing coefficients of E prod_t G[i_t,j_t]."""
    if order < 1:
        return (((), sp.Integer(1)),)
    n = sp.sympify(dimension)
    r = sp.sympify(rank)
    pairings = pair_partitions(2 * order)
    reference = canonical_pairing(order)
    weingarten = orthogonal_weingarten_matrix(order, n)
    column_sums = sp.Matrix(
        [r ** join_component_count(pairing, reference) for pairing in pairings]
    )
    coefficients = weingarten * column_sums
    if normalized:
        coefficients *= (n / r) ** order
    return tuple(
        (pairing, sp.factor(sp.cancel(coefficient)))
        for pairing, coefficient in zip(pairings, coefficients)
    )


def pairing_delta(pairing: Pairing, external_indices: tuple[int, ...]) -> int:
    return int(all(external_indices[a] == external_indices[b] for a, b in pairing))


def projector_entry_moment(
    entries: tuple[tuple[int, int], ...],
    dimension,
    rank,
    normalized: bool = True,
) -> sp.Expr:
    """Exact moment of projector entries in the supplied concrete indices."""
    if not entries:
        return sp.Integer(1)
    external_indices = tuple(index for entry in entries for index in entry)
    value = sp.Integer(0)
    for pairing, coefficient in projector_moment_coefficients(
        len(entries), dimension, rank, normalized
    ):
        value += coefficient * pairing_delta(pairing, external_indices)
    return sp.factor(sp.cancel(value))


@lru_cache(maxsize=None)
def set_partitions(items: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Set partitions in canonical block order."""
    if not items:
        return ((),)
    first, rest = items[0], items[1:]
    partitions = set()
    for partition in set_partitions(rest):
        partitions.add(tuple(sorted(((first,), *partition))))
        for block_index in range(len(partition)):
            blocks = list(partition)
            blocks[block_index] = tuple(sorted((first, *blocks[block_index])))
            partitions.add(tuple(sorted(blocks)))
    return tuple(sorted(partitions))


def projector_entry_cumulant(
    entries: tuple[tuple[int, int], ...], dimension, rank
) -> sp.Expr:
    """Exact joint cumulant of normalized projector entries G_ij."""
    order = len(entries)
    if order == 0:
        return sp.Integer(0)
    value = sp.Integer(0)
    for partition in set_partitions(tuple(range(order))):
        num_blocks = len(partition)
        coefficient = (-1) ** (num_blocks - 1) * math.factorial(num_blocks - 1)
        term = sp.Integer(coefficient)
        for block in partition:
            block_entries = tuple(entries[index] for index in block)
            term *= projector_entry_moment(block_entries, dimension, rank)
        value += term
    return sp.factor(sp.cancel(value))


def invariant_coefficients_by_coset_type(
    order: int, dimension, rank
) -> dict[tuple[int, ...], sp.Expr]:
    """Compress moment coefficients into the double-coset classes of P_2(2k)."""
    grouped: dict[tuple[int, ...], set[sp.Expr]] = {}
    for pairing, coefficient in projector_moment_coefficients(order, dimension, rank):
        grouped.setdefault(coset_type(pairing), set()).add(sp.factor(coefficient))
    result = {}
    for kind, values in grouped.items():
        if len(values) != 1:
            raise AssertionError(f"coefficient not constant on coset type {kind}")
        result[kind] = values.pop()
    return result


def gamma_defect(dimension, rank) -> sp.Expr:
    n, r = sp.sympify(dimension), sp.sympify(rank)
    return sp.factor(n * (n - r) / (r * (n - 1) * (n + 2)))


def write_symbolic_tables(output_dir: Path, max_order: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n, r = sp.symbols("n r", integer=True, positive=True)
    rows = []
    for order in range(1, max_order + 1):
        for kind, coefficient in invariant_coefficients_by_coset_type(
            order, n, r
        ).items():
            rows.append(
                {
                    "order": order,
                    "coset_type": "+".join(map(str, kind)),
                    "coefficient": str(sp.factor(coefficient)),
                    "full_rank_value": str(sp.simplify(coefficient.subs(r, n))),
                }
            )
    with (output_dir / "projector_moment_coefficients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    examples = {
        "mean_G_11": str(projector_entry_moment(((0, 0),), n, r)),
        "cov_G_11_G_11": str(projector_entry_cumulant(((0, 0), (0, 0)), n, r)),
        "cov_G_12_G_12": str(projector_entry_cumulant(((0, 1), (0, 1)), n, r)),
        "cum3_G_11": str(
            projector_entry_cumulant(((0, 0), (0, 0), (0, 0)), n, r)
        ),
        "gamma": str(gamma_defect(n, r)),
    }
    (output_dir / "projector_cumulant_examples.json").write_text(
        json.dumps(examples, indent=2)
    )


def numerical_self_check(dimension: int, rank: int, max_order: int) -> None:
    for order in range(1, max_order + 1):
        diagonal_entries = tuple((0, 0) for _ in range(order))
        actual = projector_entry_moment(diagonal_entries, dimension, rank)
        expected = (
            sp.Rational(dimension, rank) ** order
            * sp.rf(sp.Rational(rank, 2), order)
            / sp.rf(sp.Rational(dimension, 2), order)
        )
        if sp.simplify(actual - expected) != 0:
            raise AssertionError((order, actual, expected))
    if rank == dimension:
        entries = ((0, 0), (0, 1), (1, 1))
        for order in range(2, min(max_order, len(entries)) + 1):
            assert projector_entry_cumulant(entries[:order], dimension, rank) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=9)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/exact_orthogonal_weingarten"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numerical_self_check(args.dimension, args.rank, args.max_order)
    write_symbolic_tables(args.output_dir, min(args.max_order, 3))
    print(f"wrote exact Weingarten tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
