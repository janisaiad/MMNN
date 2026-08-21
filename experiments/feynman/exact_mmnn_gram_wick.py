#!/usr/bin/env python3
"""Exact Wick calculus for the frozen-left MMNN Gram vertex.

Write a concatenated pair of MMNN blocks as

    ... -> A_l -> W_{l+1} -> ...,

where ``W`` is frozen and ``A`` is trainable.  If
``W = X / sqrt(r)`` with iid standard-normal entries, differentiation with
respect to ``A`` inserts the rank-r matrix

    G = W W^T = X X^T / r.

For k entries of G let tau=(0,1)(2,3)... be the pairing internal to the
quadratic insertions.  Wick's theorem gives the exact finite-r identity

    E prod_t G[i_t,j_t]
      = sum_{pi in P_2(2k)} delta_I^pi r^(#(pi join tau)-k).

Taking a connected cumulant simply keeps those Wick pairings for which
``pi join tau`` is connected.  This is the radial-plus-angular counterpart
of the orthogonal-Weingarten projector formula; no asymptotic limit is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import sympy as sp

from exact_orthogonal_weingarten import (
    Pairing,
    canonical_pairing,
    coset_type,
    join_component_count,
    pairing_delta,
    pair_partitions,
)


@lru_cache(maxsize=128)
def gram_moment_coefficients(
    order: int, rank
) -> tuple[tuple[Pairing, sp.Expr], ...]:
    """Pairing coefficients for moments of G=XX^T/r."""
    if order < 1:
        return (((), sp.Integer(1)),)
    r = sp.sympify(rank)
    reference = canonical_pairing(order)
    return tuple(
        (
            pairing,
            sp.factor(r ** (join_component_count(pairing, reference) - order)),
        )
        for pairing in pair_partitions(2 * order)
    )


@lru_cache(maxsize=128)
def gram_cumulant_coefficients(
    order: int, rank
) -> tuple[tuple[Pairing, sp.Expr], ...]:
    """Connected pairing coefficients for cumulants of G entries."""
    if order < 1:
        return ()
    r = sp.sympify(rank)
    reference = canonical_pairing(order)
    coefficient = r ** (1 - order)
    return tuple(
        (pairing, coefficient)
        for pairing in pair_partitions(2 * order)
        if join_component_count(pairing, reference) == 1
    )


def gram_entry_moment(
    entries: tuple[tuple[int, int], ...], rank
) -> sp.Expr:
    """Exact joint moment of concrete entries of G=XX^T/r."""
    if not entries:
        return sp.Integer(1)
    external_indices = tuple(index for entry in entries for index in entry)
    value = sum(
        coefficient * pairing_delta(pairing, external_indices)
        for pairing, coefficient in gram_moment_coefficients(len(entries), rank)
    )
    return sp.factor(value)


def gram_entry_cumulant(
    entries: tuple[tuple[int, int], ...], rank
) -> sp.Expr:
    """Exact connected cumulant of concrete entries of G=XX^T/r."""
    if not entries:
        return sp.Integer(0)
    external_indices = tuple(index for entry in entries for index in entry)
    value = sum(
        coefficient * pairing_delta(pairing, external_indices)
        for pairing, coefficient in gram_cumulant_coefficients(len(entries), rank)
    )
    return sp.factor(value)


def coefficients_by_coset_type(
    order: int, rank, *, connected: bool
) -> dict[tuple[int, ...], sp.Expr]:
    """Compress coefficients by the pairing coset type relative to tau."""
    coefficients = (
        gram_cumulant_coefficients(order, rank)
        if connected
        else gram_moment_coefficients(order, rank)
    )
    grouped: dict[tuple[int, ...], set[sp.Expr]] = {}
    for pairing, coefficient in coefficients:
        grouped.setdefault(coset_type(pairing), set()).add(sp.factor(coefficient))
    result = {}
    for kind, values in grouped.items():
        if len(values) != 1:
            raise AssertionError(f"coefficient not constant on coset type {kind}")
        result[kind] = values.pop()
    return result


def write_symbolic_tables(output_dir: Path, max_order: int = 5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    r = sp.symbols("r", integer=True, positive=True)
    rows: list[dict[str, object]] = []
    for order in range(1, max_order + 1):
        for connected in (False, True):
            for kind, coefficient in coefficients_by_coset_type(
                order, r, connected=connected
            ).items():
                rows.append(
                    {
                        "order": order,
                        "kind": "cumulant" if connected else "moment",
                        "coset_type": "+".join(map(str, kind)),
                        "coefficient_per_pairing": str(coefficient),
                    }
                )
    with (output_dir / "gaussian_gram_pairing_coefficients.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    examples = {
        "mean_G_ij": "delta_ij",
        "cov_G_ij_G_kl": "(delta_ik delta_jl + delta_il delta_jk)/r",
        "cum3_G_11": str(gram_entry_cumulant(((0, 0),) * 3, r)),
        "cum4_G_11": str(gram_entry_cumulant(((0, 0),) * 4, r)),
        "general_connected_rule": (
            "r^(1-k) times the sum of delta_I^pi over pairings pi for which "
            "pi join tau is connected"
        ),
    }
    (output_dir / "gaussian_gram_cumulant_examples.json").write_text(
        json.dumps(examples, indent=2)
    )


def numerical_self_check(rank: int, max_order: int) -> None:
    for order in range(1, max_order + 1):
        diagonal = ((0, 0),) * order
        expected_moment = sp.prod(1 + sp.Rational(2 * j, rank) for j in range(order))
        expected_cumulant = (
            sp.Integer(1)
            if order == 1
            else sp.Integer(2) ** (order - 1)
            * sp.factorial(order - 1)
            / sp.Integer(rank) ** (order - 1)
        )
        assert sp.simplify(gram_entry_moment(diagonal, rank) - expected_moment) == 0
        assert sp.simplify(gram_entry_cumulant(diagonal, rank) - expected_cumulant) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-order", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/exact_mmnn_gram_wick"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numerical_self_check(args.rank, args.max_order)
    write_symbolic_tables(args.output_dir, args.max_order)
    print(f"wrote exact MMNN Gram-Wick tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
