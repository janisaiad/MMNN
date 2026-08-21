#!/usr/bin/env python3
"""Draw the three acquisition geometries discussed in the LSM references."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


BLUE = "#4c78a8"
ORANGE = "#f58518"
GREEN = "#54a24b"
RED = "#e45756"
GREY = "#888888"


def obstacle(axis: Axes) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    radius = 0.52 + 0.12 * np.cos(3.0 * theta) - 0.05 * np.sin(2.0 * theta)
    points = np.column_stack([radius * np.cos(theta), 0.80 * radius * np.sin(theta)])
    axis.add_patch(Polygon(points, closed=True, facecolor="#dddddd", edgecolor="#222222", lw=1.5))
    axis.text(0.0, 0.0, "$D$", ha="center", va="center", fontsize=12)


def sampling_domain(axis: Axes) -> None:
    axis.add_patch(
        Rectangle(
            (-1.05, -1.05),
            2.10,
            2.10,
            facecolor="#f7f7f7",
            edgecolor="#bbbbbb",
            linestyle="--",
            linewidth=1.0,
            zorder=-2,
        )
    )
    axis.text(-1.00, 0.92, r"$\Omega\subset\mathbb{R}^2$", color=GREY, fontsize=9)


def ring_sensors(axis: Axes, count: int = 14, radius: float = 1.45) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    axis.scatter(
        points[:, 0],
        points[:, 1],
        marker="^",
        s=25,
        color=BLUE,
        edgecolor="white",
        linewidth=0.4,
        zorder=4,
    )
    return points


def arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            color=color,
            linewidth=0.9,
            alpha=0.8,
        )
    )


def finish(axis: Axes, title: str, subtitle: str) -> None:
    axis.set_xlim(-2.75, 2.75)
    axis.set_ylim(-2.20, 2.35)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=11, weight="bold", pad=6)
    axis.text(0.0, -2.05, subtitle, ha="center", va="top", fontsize=8.5)


def active_multistatic(axis: Axes) -> None:
    sampling_domain(axis)
    obstacle(axis)
    ring_sensors(axis)
    incident_angles = np.asarray([0.25, 1.35, 2.55, 3.75, 5.05])
    for angle in incident_angles:
        start = 2.45 * np.asarray([np.cos(angle), np.sin(angle)])
        end = 0.65 * np.asarray([np.cos(angle), np.sin(angle)])
        arrow(axis, tuple(start), tuple(end), ORANGE)
    axis.text(1.30, 1.34, r"receivers $\widehat x_i$", color=BLUE, fontsize=8)
    axis.text(-2.48, 1.42, r"directions $\widehat d_j$", color=ORANGE, fontsize=8)
    axis.text(0.0, -1.68, r"complex far-field matrix $F\in\mathbb{C}^{m\times n}$", ha="center", fontsize=9)
    finish(
        axis,
        "Active multistatic LSM",
        "several deterministic plane waves + several receivers",
    )


def passive_random_sources(axis: Axes) -> None:
    sampling_domain(axis)
    obstacle(axis)
    ring_sensors(axis, count=12, radius=1.35)
    source_angles = np.asarray([0.12, 0.72, 1.62, 2.30, 3.18, 4.02, 4.87, 5.72])
    sources = 2.30 * np.column_stack([np.cos(source_angles), np.sin(source_angles)])
    axis.scatter(sources[:, 0], sources[:, 1], marker="*", s=55, color=ORANGE, zorder=4)
    for source in sources[[0, 3, 5, 7]]:
        arrow(axis, tuple(source), (0.0, 0.0), ORANGE)
    axis.add_patch(Circle((0.0, 0.0), 1.35, fill=False, edgecolor=BLUE, alpha=0.25))
    axis.text(-2.48, 1.42, r"sources $z_\ell$", color=ORANGE, fontsize=8)
    axis.text(1.20, 1.34, "receivers $x_j$", color=BLUE, fontsize=8)
    axis.text(0.0, -1.68, r"correlation matrix $C\in\mathbb{C}^{J\times J}$", ha="center", fontsize=9)
    finish(
        axis,
        "Random-source LSM",
        "$L$ uncontrolled sources (or random-field realizations) + $J$ receivers",
    )


def random_small_scatterer(axis: Axes) -> None:
    sampling_domain(axis)
    obstacle(axis)
    ring_sensors(axis, count=12, radius=1.35)
    primary = (-2.45, 1.35)
    axis.scatter(*primary, marker="*", s=80, color=GREEN, zorder=5)
    axis.text(-2.58, 1.66, r"fixed source $z_\varepsilon$", color=GREEN, fontsize=8)
    scatterer_angles = np.asarray([0.35, 1.15, 2.15, 3.12, 4.05, 5.05, 5.72])
    positions = 2.20 * np.column_stack([np.cos(scatterer_angles), np.sin(scatterer_angles)])
    for position in positions:
        axis.add_patch(
            Circle(tuple(position), 0.075, facecolor=RED, edgecolor="white", linewidth=0.4, alpha=0.55)
        )
    highlighted = positions[1]
    axis.add_patch(Circle(tuple(highlighted), 0.10, facecolor=RED, edgecolor="#8c2d2d", linewidth=0.8))
    arrow(axis, primary, tuple(highlighted), GREEN)
    arrow(axis, tuple(highlighted), (0.0, 0.0), RED)
    axis.text(0.0, 1.98, r"$L$ positions of $D_\varepsilon(y_\ell)$", color=RED, fontsize=8, ha="center")
    axis.text(0.0, -1.70, r"assemble modified $\widetilde C_{jm}$ after mean removal", ha="center", fontsize=9)
    finish(
        axis,
        "Small-random-scatterer LSM",
        "one primary source; one moving scatterer, optionally $R>1$ per acquisition",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    active_multistatic(axes[0])
    passive_random_sources(axes[1])
    random_small_scatterer(axes[2])
    figure.tight_layout(w_pad=1.0)
    figure.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
