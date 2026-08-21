"""Sweep the OU correlation time in the attention-based MaxCut dynamics.

Unlike the annealed-noise benchmark, this experiment starts every OU head in
its stationary Gaussian distribution and keeps its variance constant.  Only
the OU reversion rate changes, so the one-time distribution of the weights is
the same and their temporal persistence is the controlled variable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.maxcut_ou import (
    SolverConfig,
    exact_maxcut,
    random_weighted_graph,
    solve_attention_maxcut,
)


def run_slow_ou_sweep(
    seed: int,
    n: int,
    instances: int,
    restarts: int,
    edge_probability: float,
    rates: tuple[float, ...],
    config: SolverConfig,
) -> dict[str, object]:
    graph_rng = np.random.default_rng(seed)
    problems: list[tuple[np.ndarray, np.ndarray, float]] = []
    for _ in range(instances):
        weights = random_weighted_graph(graph_rng, n, edge_probability)
        initial_theta = graph_rng.uniform(-np.pi, np.pi, size=(restarts, n))
        optimum, _ = exact_maxcut(weights)
        problems.append((weights, initial_theta, optimum))

    fixed_records: list[dict[str, float | bool]] = []
    fixed_config = replace(config, stationary_ou=False)
    for instance, (weights, initial_theta, optimum) in enumerate(problems):
        result = solve_attention_maxcut(
            weights,
            np.random.default_rng(np.random.SeedSequence([seed, 1, instance])),
            restarts,
            fixed_config,
            use_ou=False,
            initial_theta=initial_theta,
        )
        tolerance = 1e-9 * max(1.0, abs(optimum))
        best = float(result["values"].max())
        final = float(result["final_values"].max())
        fixed_records.append(
            {
                "best_ratio": best / optimum,
                "final_ratio": final / optimum,
                "best_exact": bool(abs(best - optimum) <= tolerance),
                "final_exact": bool(abs(final - optimum) <= tolerance),
            }
        )

    rate_results: list[dict[str, object]] = []
    for rate_index, rate in enumerate(rates):
        records: list[dict[str, float | bool | int]] = []
        rate_config = replace(
            config,
            stationary_ou=True,
            ou_rate=rate,
        )
        for instance, (weights, initial_theta, optimum) in enumerate(problems):
            result = solve_attention_maxcut(
                weights,
                np.random.default_rng(
                    np.random.SeedSequence([seed, 2, rate_index, instance])
                ),
                restarts,
                rate_config,
                use_ou=True,
                initial_theta=initial_theta,
            )
            tolerance = 1e-9 * max(1.0, abs(optimum))
            best = float(result["values"].max())
            final = float(result["final_values"].max())
            best_mask = np.isclose(result["values"], best, rtol=0.0, atol=tolerance)
            discovery_step = int(np.min(result["best_steps"][best_mask]))
            records.append(
                {
                    "instance": instance,
                    "best_ratio": best / optimum,
                    "final_ratio": final / optimum,
                    "best_exact": bool(abs(best - optimum) <= tolerance),
                    "final_exact": bool(abs(final - optimum) <= tolerance),
                    "mean_token_flips": float(np.mean(result["token_flips"]) / n),
                    "discovery_time": discovery_step * config.dt,
                }
            )

        best_ratios = np.asarray([r["best_ratio"] for r in records], dtype=float)
        final_ratios = np.asarray([r["final_ratio"] for r in records], dtype=float)
        summary = {
            "rate": rate,
            "correlation_time": None if rate == 0.0 else 1.0 / rate,
            "best_exact_instances": int(sum(bool(r["best_exact"]) for r in records)),
            "final_exact_instances": int(sum(bool(r["final_exact"]) for r in records)),
            "best_mean_ratio": float(best_ratios.mean()),
            "final_mean_ratio": float(final_ratios.mean()),
            "best_worst_ratio": float(best_ratios.min()),
            "final_worst_ratio": float(final_ratios.min()),
            "mean_token_flips": float(np.mean([r["mean_token_flips"] for r in records])),
            "mean_discovery_time": float(np.mean([r["discovery_time"] for r in records])),
        }
        rate_results.append({"summary": summary, "records": records})

    fixed_best = np.asarray([r["best_ratio"] for r in fixed_records], dtype=float)
    fixed_final = np.asarray([r["final_ratio"] for r in fixed_records], dtype=float)
    fixed_summary = {
        "best_exact_instances": int(sum(bool(r["best_exact"]) for r in fixed_records)),
        "final_exact_instances": int(sum(bool(r["final_exact"]) for r in fixed_records)),
        "best_mean_ratio": float(fixed_best.mean()),
        "final_mean_ratio": float(fixed_final.mean()),
        "best_worst_ratio": float(fixed_best.min()),
        "final_worst_ratio": float(fixed_final.min()),
    }
    return {
        "seed": seed,
        "n": n,
        "instances": instances,
        "restarts": restarts,
        "edge_probability": edge_probability,
        "config": asdict(config),
        "fixed": {"summary": fixed_summary, "records": fixed_records},
        "rates": rate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=260518871)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--edge-probability", type=float, default=0.45)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spectral_self_attention/maxcut_slow_ou.json"),
    )
    args = parser.parse_args()
    config = SolverConfig(ou_sigma=0.7, heads=4)
    result = run_slow_ou_sweep(
        seed=args.seed,
        n=args.n,
        instances=args.instances,
        restarts=args.restarts,
        edge_probability=args.edge_probability,
        rates=tuple(args.rates),
        config=config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "fixed": result["fixed"]["summary"],
        "rates": [entry["summary"] for entry in result["rates"]],
    }, indent=2))


if __name__ == "__main__":
    main()
