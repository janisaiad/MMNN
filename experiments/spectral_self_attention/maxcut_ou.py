"""Controlled MaxCut experiment for symmetric attention with OU weight noise.

The moving variables are angles theta_i on S^1.  The Hamiltonian is

    H(theta, D) = - sum_{i<j} w_ij exp(u_i^T D u_j)
                   - kappa sum_i cos(theta_i)^2.

On the binary states theta_i in {0, pi}, the second term is constant and the
first term with D=-gamma I is an affine transform of the weighted MaxCut
objective.  The D variables are symmetric matrix-valued Ornstein--Uhlenbeck
heads around -gamma I.  Their noise is annealed to zero so that the final
Hamiltonian is the exact target problem rather than a noisy surrogate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    gamma: float = 1.35
    well_start: float = 0.02
    well_end: float = 0.42
    dt: float = 0.025
    steps: int = 1400
    ou_rate: float = 1.2
    ou_sigma: float = 0.9
    ou_fraction: float = 0.72
    heads: int = 4
    stationary_ou: bool = False


def random_weighted_graph(
    rng: np.random.Generator,
    n: int,
    edge_probability: float,
) -> np.ndarray:
    upper_mask = rng.random((n, n)) < edge_probability
    upper_mask = np.triu(upper_mask, 1)
    upper_weights = rng.uniform(0.25, 1.75, size=(n, n)) * upper_mask
    weights = upper_weights + upper_weights.T
    # Avoid isolated vertices, which are irrelevant to MaxCut but awkward for
    # comparing continuous trajectories.
    for i in np.flatnonzero(weights.sum(axis=1) == 0):
        choices = np.delete(np.arange(n), i)
        j = int(rng.choice(choices))
        weights[i, j] = weights[j, i] = float(rng.uniform(0.25, 1.75))
    return weights


def cut_values(signs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted cut values for signs with shape (..., n)."""
    dots = signs[..., :, None] * signs[..., None, :]
    return 0.25 * np.sum(weights * (1.0 - dots), axis=(-2, -1))


def exact_maxcut(weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Brute-force optimum with the first sign fixed to remove global flip."""
    n = weights.shape[0]
    count = 1 << (n - 1)
    codes = np.arange(count, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(n - 1, dtype=np.uint64)) & 1).astype(float)
    signs = np.concatenate([np.ones((count, 1)), 2.0 * bits - 1.0], axis=1)
    values = cut_values(signs, weights)
    index = int(np.argmax(values))
    return float(values[index]), signs[index]


def binary_hamiltonian(signs: np.ndarray, weights: np.ndarray, gamma: float) -> np.ndarray:
    dots = signs[..., :, None] * signs[..., None, :]
    pair = -0.5 * np.sum(weights * np.exp(-gamma * dots), axis=(-2, -1))
    return pair


def _schedule(step: int, config: SolverConfig, use_ou: bool) -> tuple[float, float]:
    progress = step / max(config.steps - 1, 1)
    # Keep the binary wells weak while the landscape is being explored, then
    # strengthen them smoothly as the OU perturbation disappears.
    well_progress = np.clip((progress - 0.30) / 0.70, 0.0, 1.0)
    well = config.well_start + (config.well_end - config.well_start) * well_progress**2
    if not use_ou:
        return well, 0.0
    if config.stationary_ou:
        return well, config.ou_sigma
    noise_progress = np.clip(progress / config.ou_fraction, 0.0, 1.0)
    sigma = config.ou_sigma * 0.5 * (1.0 + np.cos(np.pi * noise_progress))
    return well, sigma


def _symmetric_standard_normal(
    rng: np.random.Generator,
    shape_prefix: tuple[int, ...],
) -> np.ndarray:
    """Symmetric 2x2 Gaussian matrices with unit-variance independent entries."""
    diagonal = rng.normal(size=shape_prefix + (2,))
    off_diagonal = rng.normal(size=shape_prefix)
    noise = np.empty(shape_prefix + (2, 2))
    noise[..., 0, 0] = diagonal[..., 0]
    noise[..., 1, 1] = diagonal[..., 1]
    noise[..., 0, 1] = off_diagonal
    noise[..., 1, 0] = off_diagonal
    return noise


def solve_attention_maxcut(
    weights: np.ndarray,
    rng: np.random.Generator,
    restarts: int,
    config: SolverConfig,
    use_ou: bool,
    record_trace: bool = False,
    initial_theta: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run projected angle dynamics for many random initializations at once."""
    n = weights.shape[0]
    scale = max(float(weights.sum(axis=1).max()), 1.0)
    normalized_weights = weights / scale
    if initial_theta is None:
        theta = rng.uniform(-np.pi, np.pi, size=(restarts, n))
    else:
        theta = np.asarray(initial_theta, dtype=float).copy()
        if theta.shape != (restarts, n):
            raise ValueError(f"initial_theta must have shape {(restarts, n)}, got {theta.shape}")
    target_matrix = -config.gamma * np.eye(2)
    active_heads = config.heads if use_ou else 1
    matrices = np.broadcast_to(
        target_matrix, (restarts, active_heads, 2, 2)
    ).copy()
    if use_ou and config.stationary_ou:
        matrices += config.ou_sigma * _symmetric_standard_normal(
            rng, (restarts, active_heads)
        )
    trace_steps: list[int] = []
    trace_best: list[float] = []
    trace_mean: list[float] = []
    best_values = np.full(restarts, -np.inf)
    best_signs = np.ones((restarts, n))
    best_steps = np.zeros(restarts, dtype=int)
    previous_sampled_signs = np.where(np.cos(theta) >= 0.0, 1.0, -1.0)
    token_flips = np.zeros(restarts, dtype=int)

    for step in range(config.steps):
        well, sigma = _schedule(step, config, use_ou)
        if use_ou:
            if config.stationary_ou:
                decay = np.exp(-config.ou_rate * config.dt)
                innovation = sigma * np.sqrt(max(0.0, 1.0 - decay**2))
                matrices = (
                    target_matrix
                    + decay * (matrices - target_matrix)
                    + innovation
                    * _symmetric_standard_normal(rng, (restarts, active_heads))
                )
            else:
                noise = rng.normal(size=matrices.shape)
                noise = (noise + np.swapaxes(noise, -1, -2)) / np.sqrt(2.0)
                matrices += (
                    config.ou_rate * (target_matrix - matrices) * config.dt
                    + np.sqrt(2.0 * config.ou_rate * config.dt) * sigma * noise
                )
            np.clip(matrices, -3.5, 3.5, out=matrices)

        unit = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        tangent = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
        scores = np.einsum("rid,rhde,rje->rhij", unit, matrices, unit)
        tangent_values = np.einsum(
            "rid,rhde,rje->rhij", tangent, matrices, unit
        )
        kernel = np.exp(np.clip(scores, -8.0, 8.0))
        angular_force = np.mean(
            np.sum(normalized_weights[None, None, :, :] * kernel * tangent_values, axis=3),
            axis=1,
        )
        angular_force -= well * np.sin(2.0 * theta)

        # A conservative clipping of the Euler displacement prevents rare OU
        # spikes from becoming numerical jumps.
        displacement = np.clip(config.dt * angular_force, -0.18, 0.18)
        theta = (theta + displacement + np.pi) % (2.0 * np.pi) - np.pi

        if step % 10 == 0 or step == config.steps - 1:
            signs = np.where(np.cos(theta) >= 0.0, 1.0, -1.0)
            values = cut_values(signs, weights)
            token_flips += np.sum(signs != previous_sampled_signs, axis=1)
            previous_sampled_signs = signs.copy()
            improved = values > best_values
            best_values[improved] = values[improved]
            best_signs[improved] = signs[improved]
            best_steps[improved] = step
            if record_trace and (step % 20 == 0 or step == config.steps - 1):
                trace_steps.append(step)
                trace_best.append(float(best_values.max()))
                trace_mean.append(float(best_values.mean()))

    final_signs = np.where(np.cos(theta) >= 0.0, 1.0, -1.0)
    final_values = cut_values(final_signs, weights)
    signs = best_signs
    values = best_values
    order = np.argsort(best_values)[::-1]
    return {
        "theta": theta,
        "signs": signs,
        "values": values,
        "order": order,
        "final_signs": final_signs,
        "final_values": final_values,
        "best_steps": best_steps,
        "token_flips": token_flips,
        "trace_steps": np.asarray(trace_steps),
        "trace_best": np.asarray(trace_best),
        "trace_mean": np.asarray(trace_mean),
    }


def run_benchmark(
    seed: int,
    instances: int,
    n: int,
    edge_probability: float,
    restarts: int,
    config: SolverConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    representative: dict[str, object] | None = None

    for instance in range(instances):
        weights = random_weighted_graph(rng, n, edge_probability)
        optimum, optimum_signs = exact_maxcut(weights)
        initial_theta = rng.uniform(-np.pi, np.pi, size=(restarts, n))
        initial_signs = np.where(np.cos(initial_theta) >= 0.0, 1.0, -1.0)
        initial_values = cut_values(initial_signs, weights)
        fixed = solve_attention_maxcut(
            weights,
            rng,
            restarts,
            config,
            use_ou=False,
            initial_theta=initial_theta,
        )
        ou = solve_attention_maxcut(
            weights,
            rng,
            restarts,
            config,
            use_ou=True,
            record_trace=instance == 0,
            initial_theta=initial_theta,
        )
        fixed_best = float(fixed["values"].max())
        ou_best = float(ou["values"].max())
        random_best = float(initial_values.max())
        fixed_integrality = float(np.mean(np.abs(np.cos(fixed["theta"]))))
        ou_integrality = float(np.mean(np.abs(np.cos(ou["theta"]))))
        tolerance = 1e-9 * max(1.0, abs(optimum))
        record = {
            "instance": instance,
            "edges": int(np.count_nonzero(np.triu(weights, 1))),
            "optimum": optimum,
            "random_best": random_best,
            "fixed_best": fixed_best,
            "ou_best": ou_best,
            "random_ratio": random_best / optimum,
            "fixed_ratio": fixed_best / optimum,
            "ou_ratio": ou_best / optimum,
            "fixed_exact": bool(abs(fixed_best - optimum) <= tolerance),
            "ou_exact": bool(abs(ou_best - optimum) <= tolerance),
            "fixed_restart_success": float(np.mean(np.abs(fixed["values"] - optimum) <= tolerance)),
            "ou_restart_success": float(np.mean(np.abs(ou["values"] - optimum) <= tolerance)),
            "fixed_integrality": fixed_integrality,
            "ou_integrality": ou_integrality,
        }
        records.append(record)
        if instance == 0:
            edge_i, edge_j = np.nonzero(np.triu(weights, 1))
            representative = {
                "weights": weights.tolist(),
                "edges": [
                    [int(i), int(j), float(weights[i, j])]
                    for i, j in zip(edge_i, edge_j, strict=True)
                ],
                "optimum_signs": optimum_signs.tolist(),
                "fixed_signs": fixed["signs"][fixed["order"][0]].tolist(),
                "ou_signs": ou["signs"][ou["order"][0]].tolist(),
                "trace_steps": ou["trace_steps"].tolist(),
                "trace_best": ou["trace_best"].tolist(),
                "trace_mean": ou["trace_mean"].tolist(),
            }

    fixed_exact = np.asarray([r["fixed_exact"] for r in records], dtype=float)
    ou_exact = np.asarray([r["ou_exact"] for r in records], dtype=float)
    random_ratio = np.asarray([r["random_ratio"] for r in records], dtype=float)
    fixed_ratio = np.asarray([r["fixed_ratio"] for r in records], dtype=float)
    ou_ratio = np.asarray([r["ou_ratio"] for r in records], dtype=float)
    ratio_difference = ou_ratio - fixed_ratio
    summary = {
        "instances": instances,
        "n": n,
        "edge_probability": edge_probability,
        "restarts": restarts,
        "random_mean_ratio": float(random_ratio.mean()),
        "random_worst_ratio": float(random_ratio.min()),
        "random_exact_instances": int(np.sum(np.isclose(random_ratio, 1.0))),
        "fixed_exact_instances": int(fixed_exact.sum()),
        "ou_exact_instances": int(ou_exact.sum()),
        "fixed_exact_rate": float(fixed_exact.mean()),
        "ou_exact_rate": float(ou_exact.mean()),
        "fixed_mean_ratio": float(fixed_ratio.mean()),
        "ou_mean_ratio": float(ou_ratio.mean()),
        "fixed_worst_ratio": float(fixed_ratio.min()),
        "ou_worst_ratio": float(ou_ratio.min()),
        "ou_minus_fixed_mean_ratio": float(ratio_difference.mean()),
        "ou_wins": int(np.sum(ratio_difference > 1e-12)),
        "ties": int(np.sum(np.abs(ratio_difference) <= 1e-12)),
        "ou_losses": int(np.sum(ratio_difference < -1e-12)),
        "mean_fixed_restart_success": float(
            np.mean([r["fixed_restart_success"] for r in records])
        ),
        "mean_ou_restart_success": float(
            np.mean([r["ou_restart_success"] for r in records])
        ),
        "mean_fixed_integrality": float(
            np.mean([r["fixed_integrality"] for r in records])
        ),
        "mean_ou_integrality": float(
            np.mean([r["ou_integrality"] for r in records])
        ),
    }
    return {
        "seed": seed,
        "config": asdict(config),
        "summary": summary,
        "records": records,
        "representative": representative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=260518870)
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--edge-probability", type=float, default=0.45)
    parser.add_argument("--restarts", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spectral_self_attention/maxcut_ou_benchmark.json"),
    )
    args = parser.parse_args()
    result = run_benchmark(
        seed=args.seed,
        instances=args.instances,
        n=args.n,
        edge_probability=args.edge_probability,
        restarts=args.restarts,
        config=SolverConfig(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
