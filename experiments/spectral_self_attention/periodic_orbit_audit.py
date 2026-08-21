"""Refine and certify periodic-orbit examples emitted by the large census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from experiments.spectral_self_attention.mlp_equilibrium_taxonomy import (
    QuadraticMLP,
    SerialBlock,
    wrap,
)


def block_from_record(record: dict[str, object], family: int) -> SerialBlock:
    model = record["model"]
    assert isinstance(model, dict)
    mlp = QuadraticMLP(
        bias=np.asarray(model["mlp_bias"], dtype=float),
        linear=np.asarray(model["linear"], dtype=float),
        hidden=np.asarray(model["hidden"], dtype=float),
        hidden_bias=np.asarray(model["hidden_bias"], dtype=float),
        output=np.asarray(model["output"], dtype=float),
        kind="potential" if family in (1, 3) else "general",
    )
    return SerialBlock(
        score=np.asarray(model["score"], dtype=float),
        value=np.asarray(model["value"], dtype=float),
        beta=float(model["beta"]),
        step_size=float(model["step_size"]),
        mlp=mlp,
    )


def iterate(block: SerialBlock, angles: np.ndarray, steps: int) -> np.ndarray:
    output = np.asarray(angles, dtype=float)
    for _ in range(steps):
        output = block.map_angles(output)
    return output


def periodic_residual(
    block: SerialBlock, angles: np.ndarray, period: int
) -> np.ndarray:
    return wrap(iterate(block, angles, period) - angles)


def map_power_jacobian(
    block: SerialBlock,
    angles: np.ndarray,
    period: int,
    epsilon: float = 1e-6,
) -> np.ndarray:
    dimension = angles.size
    jacobian = np.empty((dimension, dimension))
    for column in range(dimension):
        direction = np.zeros(dimension)
        direction[column] = epsilon
        plus = iterate(block, angles + direction, period)
        minus = iterate(block, angles - direction, period)
        jacobian[:, column] = wrap(plus - minus) / (2.0 * epsilon)
    return jacobian


def refine_example(
    block: SerialBlock,
    start: np.ndarray,
    period: int,
) -> dict[str, object]:
    result = least_squares(
        lambda angles: periodic_residual(block, angles, period),
        np.asarray(start, dtype=float),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=4000,
    )
    point = wrap(result.x)
    primitive_residuals = {
        f"p{candidate}": float(
            np.max(np.abs(periodic_residual(block, point, candidate)))
        )
        for candidate in range(1, period + 1)
    }
    orbit = []
    current = point
    for _ in range(period):
        orbit.append(current.tolist())
        current = block.map_angles(current)
    eigenvalues = np.linalg.eigvals(map_power_jacobian(block, point, period))
    radius = float(np.max(np.abs(eigenvalues)))
    is_primitive = primitive_residuals[f"p{period}"] < 1e-9 and all(
        primitive_residuals[f"p{candidate}"] > 1e-5
        for candidate in range(1, period)
    )
    return {
        "period": period,
        "point": point.tolist(),
        "orbit": orbit,
        "primitive_residuals": primitive_residuals,
        "primitive": is_primitive,
        "spectral_radius": radius,
        "stable": radius < 1.0 - 1e-6,
        "eigenvalues": [
            [float(value.real), float(value.imag)] for value in eigenvalues
        ],
        "optimizer_evaluations": int(result.nfev),
    }


def audit_files(inputs: list[Path]) -> dict[str, object]:
    results: dict[str, object] = {}
    for path in inputs:
        census = json.loads(path.read_text())
        family = int(census["family"])
        family_results: dict[str, object] = {}
        for name, record in census["examples"].items():
            if not name.startswith("p"):
                continue
            period = int(name[1:])
            block = block_from_record(record, family)
            tail = np.asarray(record["cycle_tail"], dtype=float)
            family_results[name] = {
                "n_tokens": int(record["n_tokens"]),
                "model": record["model"],
                "certificate": refine_example(block, tail[0], period),
            }
        results[str(family)] = {
            "family_name": census["family_name"],
            "periodic_examples": family_results,
        }
    return {"families": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_files(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        family: {
            name: {
                "primitive": record["certificate"]["primitive"],
                "stable": record["certificate"]["stable"],
                "radius": record["certificate"]["spectral_radius"],
            }
            for name, record in data["periodic_examples"].items()
        }
        for family, data in result["families"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
