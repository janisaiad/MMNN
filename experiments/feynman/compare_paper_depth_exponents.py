#!/usr/bin/env python3
"""Compare paper regressions with deterministic recursion and true powers.

The numbers printed in the Feynman-diagram paper are finite-width Monte Carlo
log-log slopes.  The second column computed here applies the same regression
window to the deterministic leading-width tensor recursion.  The final column
is the asymptotic regular-variation exponent, not another fit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("0000", "0101", "0022", "0103", "0023", "0123")
PAPER_MONTE_CARLO = {
    "V": (1.19, 1.31, 1.49, 1.42, 1.47, 1.55),
    "D": (2.06, 2.34, 2.67, 2.35, 2.55, 2.49),
    "F": (2.24, 2.35, 2.69, 2.20, 2.67, 2.24),
    "A": (3.16, 3.39, 3.74, 3.39, 3.67, 3.54),
    "B": (3.28, 3.34, 3.78, 3.16, 3.76, 3.43),
}
FIT_START = {
    "V": (7, 10, 10, 10, 10, 10),
    "D": (7, 10, 10, 10, 10, 10),
    "F": (7, 10, 10, 10, 10, 10),
    "A": (7, 10, 10, 10, 10, 10),
    "B": (5, 12, 10, 10, 10, 10),
}
ASYMPTOTIC_POWER = {"V": 1, "D": 2, "F": 2, "A": 3, "B": 3}


def fitted_slope(layers: np.ndarray, values: np.ndarray, start: int) -> float:
    mask = (layers >= start) & np.isfinite(values) & (np.abs(values) > 0)
    slope, _ = np.polyfit(np.log(layers[mask]), np.log(np.abs(values[mask])), 1)
    return float(slope)


def component_tuple(label: str) -> tuple[int, int, int, int]:
    if len(label) != 4 or not label.isdigit():
        raise ValueError(label)
    return tuple(map(int, label))  # type: ignore[return-value]


def build_rows(npz_path: Path) -> list[dict[str, object]]:
    archive = np.load(npz_path)
    layers = archive["layers"]
    rows = []
    for tensor in ("V", "D", "F", "A", "B"):
        for index, label in enumerate(COMPONENTS):
            component = component_tuple(label)
            deterministic = fitted_slope(
                layers,
                archive[tensor][(slice(None),) + component],
                FIT_START[tensor][index],
            )
            monte_carlo = PAPER_MONTE_CARLO[tensor][index]
            rows.append(
                {
                    "tensor": tensor,
                    "component": label,
                    "fit_start": FIT_START[tensor][index],
                    "fit_end": int(layers[-1]),
                    "paper_monte_carlo_slope": monte_carlo,
                    "deterministic_recursion_slope": deterministic,
                    "difference_mc_minus_recursion": monte_carlo - deterministic,
                    "asymptotic_power": ASYMPTOTIC_POWER[tensor],
                }
            )
    return rows


def save_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    tensors = ("V", "D", "F", "A", "B")
    colors = {"V": "#2A6FBB", "D": "#E07A1F", "F": "#2F9E73", "A": "#9B51B6", "B": "#C64545"}
    fig, axes = plt.subplots(1, 5, figsize=(15.8, 3.45), sharex=True)
    x = np.arange(len(COMPONENTS))
    for axis, tensor in zip(axes, tensors):
        selected = [row for row in rows if row["tensor"] == tensor]
        mc = np.asarray([row["paper_monte_carlo_slope"] for row in selected])
        deterministic = np.asarray(
            [row["deterministic_recursion_slope"] for row in selected]
        )
        axis.axhline(
            ASYMPTOTIC_POWER[tensor], color="black", linestyle="--", linewidth=1.4
        )
        axis.scatter(
            x - 0.09,
            mc,
            marker="o",
            s=37,
            color=colors[tensor],
            label="paper: Monte Carlo fit",
            zorder=3,
        )
        axis.scatter(
            x + 0.09,
            deterministic,
            marker="D",
            s=31,
            facecolor="white",
            edgecolor=colors[tensor],
            linewidth=1.4,
            label="exact recursion: same window",
            zorder=3,
        )
        axis.set_title(tensor, fontsize=12, fontweight="bold")
        axis.set_xticks(x, COMPONENTS, rotation=55, fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        if tensor in ("V",):
            axis.set_ylim(0.9, 1.75)
        elif tensor in ("D", "F"):
            axis.set_ylim(1.75, 2.9)
        else:
            axis.set_ylim(2.55, 4.0)
    axes[0].set_ylabel("log–log slope over the plotted depths")
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="black", linestyle="--"))
    labels.append("proved asymptotic power")
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Finite-window exponents are effective slopes, not new critical powers",
        fontsize=13,
    )
    fig.subplots_adjust(bottom=0.31, top=0.82, left=0.055, right=0.99, wspace=0.25)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recursion-file",
        type=Path,
        default=Path(
            "data/feynman/exact_relu_tensor_recursions/exact_relu_tensor_recursions.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/paper_exponent_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.recursion_file)
    csv_path = args.output_dir / "paper_vs_exact_recursion_exponents.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_plot(rows, args.output_dir / "paper_vs_exact_recursion_exponents")
    print(f"wrote comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
