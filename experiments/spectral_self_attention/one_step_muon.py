"""Compare one GD step with one Muon step near a polygonal attractor.

This is a finite-particle analogue of the one-step adjoint calculation in
Isobe--Inoue--Imaizumi (arXiv:2605.07772). A depth-dependent, matrix-shaped
control is initialized at zero. We backpropagate a terminal rotation loss and
compare the raw gradient, its exact polar factor, and the practical five-step
Newton--Schulz approximation at equal continuous L2 control norm.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.spectral_self_attention.mean_field_extensions import (
    MIXED_EXP_ROOT,
    SHARPNESS,
)


def newton_schulz_scalar_magnitudes(
    gradient_norms,
    *,
    epsilon: float = 1e-7,
    steps: int = 5,
):
    """Muon's Newton--Schulz map for a batch of rank-one row matrices."""
    import torch

    values = gradient_norms / torch.clamp(gradient_norms, min=epsilon)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        values = a * values + b * values**3 + c * values**5
    return values


def run_experiment() -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    import torch

    torch.set_default_dtype(torch.float64)
    matrix = torch.diag(torch.tensor([2.0, -3.0]))
    base = torch.tensor(
        [0.0, np.arccos(MIXED_EXP_ROOT), -np.arccos(MIXED_EXP_ROOT)]
    )
    target = base + 0.9
    bins = 40
    substeps = 4
    horizon = 8.0
    dt = horizon / (bins * substeps)
    bin_width = horizon / bins

    def attention(angles):
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        weights = torch.exp(SHARPNESS * (points @ matrix @ points.T)) / 3.0
        force = (weights @ (points @ matrix)) / weights.sum(dim=1, keepdim=True)
        tangents = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=1)
        return (force * tangents).sum(dim=1)

    def features(angles):
        return torch.stack(
            [
                torch.ones_like(angles),
                torch.cos(angles),
                torch.sin(angles),
                torch.cos(2.0 * angles),
                torch.sin(2.0 * angles),
            ],
            dim=1,
        )

    def rollout(control):
        angles = base
        states = [angles]
        for bin_index in range(bins):
            for _ in range(substeps):
                angles = angles + dt * (
                    attention(angles) + features(angles) @ control[bin_index]
                )
            states.append(angles)
        return angles, torch.stack(states)

    zero_control = torch.zeros((bins, 5), requires_grad=True)
    terminal, _ = rollout(zero_control)
    initial_loss = (1.0 - torch.cos(terminal - target)).mean()
    initial_loss.backward()
    gradient = zero_control.grad.detach()
    gradient_norms = torch.linalg.vector_norm(gradient, dim=1)

    exact_polar = torch.where(
        gradient_norms[:, None] > 0.0,
        gradient / gradient_norms[:, None],
        torch.zeros_like(gradient),
    )
    ns_magnitudes = newton_schulz_scalar_magnitudes(gradient_norms)
    practical_polar = exact_polar * ns_magnitudes[:, None]

    rows: list[dict[str, float | int | str]] = []
    outcome_summary: dict[str, dict[str, float | int | None]] = {}
    for budget in [0.1, 0.3, 1.0]:
        raw_updates = {
            "gradient_descent": gradient,
            "exact_muon": exact_polar,
            "newton_schulz_5": practical_polar,
        }
        for optimizer, raw_update in raw_updates.items():
            scale = budget / torch.sqrt(bin_width * torch.sum(raw_update**2))
            control = -scale * raw_update
            end, states = rollout(control)
            terminal_loss = (1.0 - torch.cos(end - target)).mean()
            polygon_gap = (1.0 - torch.cos(states - base)).mean(dim=1)
            crossings = torch.nonzero(polygon_gap > 0.01)
            first_crossing = int(crossings[0, 0]) if len(crossings) else None
            outcome_summary[f"{budget}:{optimizer}"] = {
                "terminal_loss": float(terminal_loss),
                "terminal_polygon_gap": float(polygon_gap[-1]),
                "midpoint_polygon_gap": float(polygon_gap[bins // 2]),
                "first_gap_0.01_bin": first_crossing,
            }
            control_norms = torch.linalg.vector_norm(control, dim=1)
            for bin_index in range(bins):
                rows.append(
                    {
                        "budget": budget,
                        "optimizer": optimizer,
                        "depth_bin": bin_index,
                        "gradient_norm": float(gradient_norms[bin_index]),
                        "control_norm": float(control_norms[bin_index]),
                        "polygon_gap": float(polygon_gap[bin_index + 1]),
                    }
                )

    summary: dict[str, object] = {
        "protocol": {
            "depth_bins": bins,
            "features": 5,
            "horizon": horizon,
            "target_rotation_radians": 0.9,
            "comparison_norm": "sqrt(bin_width * sum Frobenius(control_bin)^2)",
        },
        "untrained_terminal_loss": float(initial_loss),
        "first_gradient_norm": float(gradient_norms[0]),
        "last_gradient_norm": float(gradient_norms[-1]),
        "last_to_first_gradient_ratio": float(gradient_norms[-1] / gradient_norms[0]),
        "newton_schulz_magnitudes": {
            str(index): float(ns_magnitudes[index])
            for index in [0, 5, 10, 15, 20, 25, 30, 35, 39]
        },
        "outcomes": outcome_summary,
    }
    return rows, summary


def write_results(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = run_experiment()
    with (output / "depth_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spectral_self_attention/one_step_muon"),
    )
    args = parser.parse_args()
    write_results(args.output)


if __name__ == "__main__":
    main()
