"""Quantify persistent synchronized token clusters in continuous attractors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.large_scale_cycle_census import wrap


def component_sizes(angles: np.ndarray, tolerance: float) -> tuple[int, ...]:
    n_tokens = len(angles)
    parent = list(range(n_tokens))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    distance = np.abs(wrap(angles[:, None] - angles[None, :]))
    for first in range(n_tokens):
        for second in range(first + 1, n_tokens):
            if distance[first, second] < tolerance:
                union(first, second)
    counts = Counter(find(index) for index in range(n_tokens))
    return tuple(sorted(counts.values(), reverse=True))


def summarize_histories(histories: list[np.ndarray]) -> dict[str, object]:
    n_tokens = histories[0].shape[-1]
    pair_indices = np.triu_indices(n_tokens, 1)
    tolerances = (1e-6, 1e-4, 1e-2)
    pair_sync = {}
    partitions = {}
    for tolerance in tolerances:
        all_pair_fractions = []
        partition_counter: Counter[tuple[int, ...]] = Counter()
        for history in histories:
            pair_distance = np.abs(
                wrap(history[:, :, None] - history[:, None, :])
            )
            all_pair_fractions.append(
                np.mean(pair_distance[:, pair_indices[0], pair_indices[1]] < tolerance, axis=0)
            )
            partition_counter.update(
                component_sizes(angles, tolerance) for angles in history
            )
        pair_sync[str(tolerance)] = np.mean(all_pair_fractions, axis=0).tolist()
        total = sum(partition_counter.values())
        partitions[str(tolerance)] = [
            {"sizes": list(sizes), "count": count, "fraction": count / total}
            for sizes, count in partition_counter.most_common(12)
        ]
    persistent_matrix = np.eye(n_tokens)
    fractions = np.zeros((n_tokens, n_tokens))
    for history in histories:
        distance = np.abs(wrap(history[:, :, None] - history[:, None, :]))
        fractions += np.mean(distance < 1e-4, axis=0)
    fractions /= len(histories)
    persistent_matrix[fractions > 0.99] = 1.0
    persistent_matrix[fractions <= 0.99] = 0.0
    return {
        "histories": len(histories),
        "n_tokens": n_tokens,
        "pair_indices": np.stack(pair_indices, axis=-1).tolist(),
        "pair_synchronization_fraction": pair_sync,
        "partition_distribution": partitions,
        "persistent_pair_matrix_tolerance_1e-4": persistent_matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.traces]
    histories = [np.asarray(payload["angles"], dtype=float) for payload in payloads]
    result = summarize_histories(histories)
    result["identities"] = [payload["identity"] for payload in payloads]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
