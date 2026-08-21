"""Quantify why Gram eigendecomposition is inadmissible for polar Muon."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmnn.full_training_frequency import FullyTrainedPeriodicMLP
from mmnn.spectral_power import spectral_power_direction


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "full_training_results" / "spectral_backend_audit.json"


def gram_spectral_power_direction(
    gradient: torch.Tensor,
    power: float,
    *,
    relative_floor: float = 1.0e-7,
) -> torch.Tensor:
    """Reproduce the rejected Gram-eigendecomposition shortcut."""
    rows, columns = gradient.shape
    if rows >= columns:
        eigenvalues, right = torch.linalg.eigh(gradient.T @ gradient)
        singular_values = torch.sqrt(eigenvalues.clamp_min(0.0))
        threshold = relative_floor * singular_values.max()
        active = singular_values > threshold
        inverse_power = torch.where(
            active,
            singular_values.clamp_min(threshold).pow(power - 1.0),
            torch.zeros_like(singular_values),
        )
        transformed = ((gradient @ right) * inverse_power.unsqueeze(0)) @ right.T
    else:
        eigenvalues, left = torch.linalg.eigh(gradient @ gradient.T)
        singular_values = torch.sqrt(eigenvalues.clamp_min(0.0))
        threshold = relative_floor * singular_values.max()
        active = singular_values > threshold
        inverse_power = torch.where(
            active,
            singular_values.clamp_min(threshold).pow(power - 1.0),
            torch.zeros_like(singular_values),
        )
        transformed = (left * inverse_power.unsqueeze(0)) @ (left.T @ gradient)
    norm = torch.linalg.vector_norm(transformed)
    return transformed * (math.sqrt(gradient.numel()) / norm)


def representative_gradients(seed: int) -> list[tuple[str, torch.Tensor]]:
    x = 2.0 * math.pi * torch.arange(128) / 128
    target = (
        torch.cos(x)
        + 0.65 * torch.cos(4.0 * x)
        + 0.45 * torch.cos(8.0 * x)
        + 0.30 * torch.cos(16.0 * x)
    )
    model = FullyTrainedPeriodicMLP(
        x,
        width=128,
        affine_depth=5,
        seed=seed,
    )
    loss = 0.5 * torch.mean((model() - target).square())
    named_parameters = tuple(model.named_parameters())
    gradients = torch.autograd.grad(
        loss,
        tuple(parameter for _, parameter in named_parameters),
    )
    return [
        (name, gradient.detach())
        for (name, parameter), gradient in zip(named_parameters, gradients, strict=True)
        if parameter.ndim == 2 and parameter.shape[0] == parameter.shape[1]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for seed in range(3):
        for name, gradient in representative_gradients(seed):
            singular_values = torch.linalg.svdvals(gradient)
            numerical_rank = int(
                torch.sum(singular_values > 1.0e-7 * singular_values.max())
            )
            for power in (0.0, 1.0 / 3.0, 2.0 / 3.0):
                direct = spectral_power_direction(gradient, power)
                gram = gram_spectral_power_direction(gradient, power)
                cosine = torch.nn.functional.cosine_similarity(
                    direct.flatten(), gram.flatten(), dim=0
                )
                relative_error = torch.linalg.vector_norm(
                    gram - direct
                ) / torch.linalg.vector_norm(direct)
                records.append(
                    {
                        "seed": seed,
                        "block": name,
                        "power": power,
                        "direct_numerical_rank": numerical_rank,
                        "cosine": float(cosine),
                        "relative_error": float(relative_error),
                    }
                )

    aggregate: dict[str, dict[str, float]] = {}
    for power in (0.0, 1.0 / 3.0, 2.0 / 3.0):
        selected = [row for row in records if row["power"] == power]
        aggregate[f"p={power:g}"] = {
            "comparisons": len(selected),
            "median_cosine": float(np.median([row["cosine"] for row in selected])),
            "minimum_cosine": float(np.min([row["cosine"] for row in selected])),
            "median_relative_error": float(
                np.median([row["relative_error"] for row in selected])
            ),
            "maximum_relative_error": float(
                np.max([row["relative_error"] for row in selected])
            ),
        }
    report = {
        "dtype": "float32",
        "relative_floor": 1.0e-7,
        "conclusion": (
            "The Gram shortcut is excluded because it squares the condition "
            "number and materially changes small singular sectors."
        ),
        "aggregate": aggregate,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
