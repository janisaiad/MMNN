"""Frozen-agent basin maps and high-dimensional multi-head diagnostics.

This module studies a probe token on a sphere while every other token is held
fixed.  It is intentionally a controlled slice through the full many-token
dynamics: the colour of an initial probe records the stable destination reached
under one head or under the sum of two heads.

The high-dimensional experiment uses independent dense symmetric Gaussian
(Wigner) matrices, scaled so their spectra remain order one as dimension grows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def normalize(x: FloatArray) -> FloatArray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero vector")
    return x / norms


def rotation(axis: FloatArray, angle: float) -> FloatArray:
    """Three-dimensional right-handed rotation matrix."""
    axis = normalize(np.asarray(axis, dtype=float))
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def probe_field(
    probes: FloatArray,
    anchors: FloatArray,
    matrices: list[FloatArray],
    beta: float,
) -> FloatArray:
    """Tangent velocity of probes; anchors are keys/values but do not move."""
    batch = probes.shape[0]
    all_tokens = np.concatenate(
        [probes[:, None, :], np.broadcast_to(anchors, (batch, *anchors.shape))],
        axis=1,
    )
    force = np.zeros_like(probes)
    for matrix in matrices:
        transformed = np.einsum("bjd,ed->bje", all_tokens, matrix, optimize=True)
        scores = np.einsum("bd,bjd->bj", probes, transformed, optimize=True)
        weights = softmax(beta * scores)
        force += np.einsum("bj,bjd->bd", weights, transformed, optimize=True)
    return force - np.sum(force * probes, axis=1, keepdims=True) * probes


def integrate_probes(
    initial: FloatArray,
    anchors: FloatArray,
    matrices: list[FloatArray],
    beta: float,
    *,
    t_final: float,
    dt: float,
) -> FloatArray:
    """Retraction RK4 for a vectorized collection of independent probes."""
    probes = normalize(np.asarray(initial, dtype=float).copy())
    n_steps = int(np.ceil(t_final / dt))
    step = t_final / n_steps
    for _ in range(n_steps):
        k1 = probe_field(probes, anchors, matrices, beta)
        p2 = normalize(probes + 0.5 * step * k1)
        k2 = probe_field(p2, anchors, matrices, beta)
        p3 = normalize(probes + 0.5 * step * k2)
        k3 = probe_field(p3, anchors, matrices, beta)
        p4 = normalize(probes + step * k3)
        k4 = probe_field(p4, anchors, matrices, beta)
        probes = normalize(probes + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0)
    return probes


def cluster_endpoints(
    endpoints: FloatArray, *, angular_tolerance: float = 0.025
) -> tuple[NDArray[np.int64], FloatArray, NDArray[np.int64]]:
    """Cluster converged endpoints, then sort centres deterministically."""
    raw_labels = np.full(len(endpoints), -1, dtype=np.int64)
    centres: list[FloatArray] = []
    cosine_cutoff = np.cos(angular_tolerance)
    for index, endpoint in enumerate(endpoints):
        if centres:
            similarities = np.asarray(centres) @ endpoint
            nearest = int(np.argmax(similarities))
            if similarities[nearest] >= cosine_cutoff:
                raw_labels[index] = nearest
                continue
        raw_labels[index] = len(centres)
        centres.append(endpoint.copy())

    centre_array = np.asarray(
        [normalize(np.mean(endpoints[raw_labels == label], axis=0)) for label in range(len(centres))]
    )
    # One reassignment removes order dependence after centres are averaged.
    raw_labels = np.argmax(endpoints @ centre_array.T, axis=1).astype(np.int64)
    centre_array = np.asarray(
        [normalize(np.mean(endpoints[raw_labels == label], axis=0)) for label in range(len(centre_array))]
    )

    if endpoints.shape[1] == 2:
        key = np.mod(np.arctan2(centre_array[:, 1], centre_array[:, 0]), 2 * np.pi)
        order = np.argsort(key)
    else:
        # North-to-south, then azimuth: stable labels for the sphere display.
        azimuth = np.mod(np.arctan2(centre_array[:, 1], centre_array[:, 0]), 2 * np.pi)
        order = np.lexsort((azimuth, -centre_array[:, 2]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    labels = inverse[raw_labels]
    centres_sorted = centre_array[order]
    counts = np.bincount(labels, minlength=len(order)).astype(np.int64)
    return labels, centres_sorted, counts


def fibonacci_sphere(size: int) -> FloatArray:
    index = np.arange(size, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / size
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * index
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])


def basin_payload() -> dict[str, object]:
    beta_circle = 5.0
    anchor_angles = np.array([0.0, 0.55, 2.1, 3.4, 5.0])
    circle_anchors = np.column_stack([np.cos(anchor_angles), np.sin(anchor_angles)])
    eigenvalues_2d = np.diag([2.0, -3.0])
    angle_b = np.pi / 3.0
    rb = np.array(
        [[np.cos(angle_b), -np.sin(angle_b)], [np.sin(angle_b), np.cos(angle_b)]]
    )
    head_a_2d = eigenvalues_2d
    head_b_2d = rb @ eigenvalues_2d @ rb.T
    scenarios_2d = {
        "A": [head_a_2d],
        "B": [head_b_2d],
        "A+B": [head_a_2d, head_b_2d],
    }
    initial_angles = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    initial_circle = np.column_stack([np.cos(initial_angles), np.sin(initial_angles)])
    circle_results: dict[str, object] = {}
    for name, matrices in scenarios_2d.items():
        endpoints = integrate_probes(
            initial_circle,
            circle_anchors,
            matrices,
            beta_circle,
            t_final=35.0,
            dt=0.04,
        )
        labels, centres, counts = cluster_endpoints(endpoints)
        circle_results[name] = {
            "labels": labels.tolist(),
            "final_angles": np.mod(np.arctan2(endpoints[:, 1], endpoints[:, 0]), 2 * np.pi).round(7).tolist(),
            "centres": centres.round(8).tolist(),
            "counts": counts.tolist(),
        }

    beta_sphere = 4.0
    sphere_anchors = normalize(
        np.array(
            [
                [1.0, 0.1, 0.2],
                [-0.4, 0.9, 0.15],
                [-0.6, -0.5, 0.7],
                [0.2, -0.7, -0.8],
                [0.7, 0.4, -0.55],
            ]
        )
    )
    spectrum_3d = np.diag([2.0, 0.5, -3.0])
    r3 = rotation(np.array([0.0, 1.0, 1.0]), 1.0)
    head_a_3d = spectrum_3d
    head_b_3d = r3 @ spectrum_3d @ r3.T
    scenarios_3d = {
        "A": [head_a_3d],
        "B": [head_b_3d],
        "A+B": [head_a_3d, head_b_3d],
    }
    initial_sphere = fibonacci_sphere(3600)
    sphere_results: dict[str, object] = {}
    for name, matrices in scenarios_3d.items():
        endpoints = integrate_probes(
            initial_sphere,
            sphere_anchors,
            matrices,
            beta_sphere,
            t_final=30.0,
            dt=0.04,
        )
        labels, centres, counts = cluster_endpoints(endpoints, angular_tolerance=0.04)
        sphere_results[name] = {
            "labels": labels.tolist(),
            "centres": centres.round(8).tolist(),
            "counts": counts.tolist(),
        }

    return {
        "circle": {
            "beta": beta_circle,
            "anchors": circle_anchors.round(8).tolist(),
            "initial_angles": initial_angles.round(8).tolist(),
            "scenarios": circle_results,
        },
        "sphere": {
            "beta": beta_sphere,
            "anchors": sphere_anchors.round(8).tolist(),
            "initial": initial_sphere.round(8).tolist(),
            "scenarios": sphere_results,
        },
    }


def high_dimension_rows(seed: int = 260426085) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    dimensions = [2, 4, 8, 16, 32, 64, 128, 256]
    betas = [1.0, 4.0, 8.0]
    n_tokens = 24
    n_heads = 6
    trials = 96
    random_beta = 3.0
    rows: list[dict[str, float | int]] = []

    for dimension in dimensions:
        samples: dict[str, list[float]] = {
            "token_abs_cosine": [],
            "head_abs_cosine": [],
            "head_sum_ratio": [],
            "random_head_neff": [],
            **{f"self_mass_beta_{int(beta)}": [] for beta in betas},
        }
        off_diagonal = ~np.eye(n_tokens, dtype=bool)
        for _ in range(trials):
            tokens = normalize(rng.normal(size=(n_tokens, dimension)))
            gram = tokens @ tokens.T
            samples["token_abs_cosine"].append(float(np.mean(np.abs(gram[off_diagonal]))))

            for beta in betas:
                weights = softmax(beta * gram)
                samples[f"self_mass_beta_{int(beta)}"].append(float(np.mean(np.diag(weights))))

            gaussian = rng.normal(size=(n_heads, dimension, dimension))
            matrices = (gaussian + np.swapaxes(gaussian, 1, 2)) / np.sqrt(
                2.0 * dimension
            )
            transformed = np.einsum(
                "id,hde->hie", tokens, matrices, optimize=True
            )
            scores = np.einsum("id,hjd->hij", tokens, transformed, optimize=True)
            weights = softmax(random_beta * scores)
            entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=-1)
            samples["random_head_neff"].append(float(np.mean(np.exp(entropy))))
            forces = np.einsum("hij,hjd->hid", weights, transformed, optimize=True)
            radial = np.einsum("hid,id->hi", forces, tokens, optimize=True)
            tangent = forces - radial[:, :, None] * tokens[None, :, :]
            norms = np.linalg.norm(tangent, axis=-1)

            pair_cosines: list[FloatArray] = []
            for left in range(n_heads):
                for right in range(left + 1, n_heads):
                    denominator = np.maximum(norms[left] * norms[right], 1e-14)
                    cosine = np.sum(tangent[left] * tangent[right], axis=-1) / denominator
                    pair_cosines.append(np.abs(cosine))
            samples["head_abs_cosine"].append(float(np.mean(pair_cosines)))

            denominator = np.sqrt(np.sum(norms * norms, axis=0))
            ratio = np.linalg.norm(np.sum(tangent, axis=0), axis=-1) / np.maximum(denominator, 1e-14)
            samples["head_sum_ratio"].append(float(np.mean(ratio)))

        row: dict[str, float | int] = {
            "dimension": dimension,
            "n_tokens": n_tokens,
            "n_heads": n_heads,
            "trials": trials,
            "random_matrix_beta": random_beta,
            "token_abs_cosine_theory": float(np.sqrt(2.0 / (np.pi * dimension))),
            "aligned_head_abs_cosine": 1.0,
            "aligned_head_sum_ratio": float(np.sqrt(n_heads)),
        }
        for metric, values in samples.items():
            values_array = np.asarray(values)
            row[metric] = float(np.mean(values_array))
            row[f"{metric}_se"] = float(np.std(values_array, ddof=1) / np.sqrt(trials))
        for beta in betas:
            row[f"self_mass_limit_beta_{int(beta)}"] = float(
                np.exp(beta) / (np.exp(beta) + n_tokens - 1)
            )
        rows.append(row)
    return rows


def write_outputs(output: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    basins = basin_payload()
    (output / "frozen_probe_basins.json").write_text(
        json.dumps(basins, separators=(",", ":")), encoding="utf-8"
    )
    rows = high_dimension_rows(seed)
    with (output / "high_dimension_decoupling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seed": seed,
        "basin_protocol": "one probe moves on S1 or S2; five other tokens are frozen; the probe is also its own key/value",
        "head_protocol": "symmetric aligned score/value matrices; total tangent force is the sum of head forces",
        "high_dimension_protocol": "96 trials, 24 random unit tokens, six independent dense symmetric Gaussian (Wigner) heads scaled by 1/sqrt(2d)",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spectral_self_attention/multihead"),
    )
    parser.add_argument("--seed", type=int, default=260426085)
    args = parser.parse_args()
    write_outputs(args.output, args.seed)


if __name__ == "__main__":
    main()
