#!/usr/bin/env python3
"""Exact leading depth coefficients for critical ReLU Feynman tensors.

Let ``s_a=sqrt(K_aa)`` for the fixed critical preactivation covariance and
``d_ab=1{a != b}``.  The exact deterministic tensor recursions imply, for
every fixed collection of non-collinear inputs,

    V_abcd(L) ~ v_abcd L,
    D_abcd(L) ~ d_abcd L^2,
    F_abcd(L) ~ f_abcd L^2,
    A_abcd(L) ~ a_abcd L^3,
    B_abcd(L) ~ b_abcd L^3.

The coefficients below result from the coalescing ReLU collision sectors,
including the transverse Ward terms amplified by the indicator Hessian.
They are not regressions.  In particular all are nonzero for positive input
norms, proving that the powers 1,2,2,3,3 are attained componentwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_exact_relu_tensor_recursions import PLOT_COMPONENTS


POWERS = {"V": 1, "D": 2, "F": 2, "A": 3, "B": 3}


def unequal(a: int, b: int) -> int:
    return int(a != b)


def normalized_leading_coefficient(
    tensor: str, component: tuple[int, int, int, int]
) -> Fraction:
    """Coefficient after dividing by s_a s_b s_c s_d."""
    a, b, c, d = component
    if tensor == "V":
        return Fraction(5, 1)
    if tensor == "D":
        return Fraction(3, 2 + 3 * unequal(c, d))
    if tensor == "F":
        return Fraction(1, 2 * (1 + 3 * unequal(b, d)))
    if tensor == "A":
        number_of_off_diagonal_ntk_pairs = unequal(a, b) + unequal(c, d)
        if number_of_off_diagonal_ntk_pairs == 0:
            return Fraction(7, 12)
        if number_of_off_diagonal_ntk_pairs == 1:
            return Fraction(47, 240)
        first_pair, second_pair = frozenset((a, b)), frozenset((c, d))
        if first_pair == second_pair:
            return Fraction(227, 2880)
        if first_pair & second_pair:
            return Fraction(149, 1920)
        return Fraction(37, 480)
    if tensor == "B":
        denominator = (
            6
            * (1 + 3 * unequal(a, c))
            * (1 + 3 * unequal(b, d))
            * (1 + unequal(a, b) + unequal(c, d))
        )
        return Fraction(1, denominator)
    raise ValueError(tensor)


def leading_coefficient(
    tensor: str,
    component: tuple[int, int, int, int],
    standard_deviations: np.ndarray,
) -> float:
    scale = float(np.prod(standard_deviations[list(component)]))
    return scale * float(normalized_leading_coefficient(tensor, component))


def ntk_leading_coefficient(a: int, b: int, standard_deviations: np.ndarray):
    return (
        standard_deviations[a]
        * standard_deviations[b]
        / (2.0 * (1.0 + 3.0 * unequal(a, b)))
    )


def relu_angle_map(theta: float) -> float:
    correlation = (
        math.sin(theta) + (math.pi - theta) * math.cos(theta)
    ) / math.pi
    return math.acos(float(np.clip(correlation, -1.0, 1.0)))


def save_convergence_analysis(
    recursion_file: Path, input_file: Path, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(recursion_file)
    layers = archive["layers"].astype(float)
    inputs = np.asarray(json.loads(input_file.read_text())["input"], dtype=float)
    covariance_diagonal = 2.0 * np.sum(inputs * inputs, axis=1) / inputs.shape[1]
    standard_deviations = np.sqrt(covariance_diagonal)
    rows = []
    fig, axes = plt.subplots(1, 5, figsize=(17.2, 3.55), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.07, 0.93, len(PLOT_COMPONENTS)))
    for axis, tensor in zip(axes, ("V", "D", "F", "A", "B")):
        power = POWERS[tensor]
        for color, component in zip(colors, PLOT_COMPONENTS):
            coefficient = leading_coefficient(
                tensor, component, standard_deviations
            )
            values = archive[tensor][(slice(None),) + component]
            ratio = values / (coefficient * layers**power)
            label = "".join(map(str, component))
            mask = layers >= 10
            axis.semilogx(
                layers[mask], ratio[mask], color=color, linewidth=1.45, label=label
            )
            rows.append(
                {
                    "tensor": tensor,
                    "component": label,
                    "power": power,
                    "normalized_coefficient_exact": str(
                        normalized_leading_coefficient(tensor, component)
                    ),
                    "coefficient_for_inputs": coefficient,
                    "ratio_at_max_depth": float(ratio[-1]),
                    "relative_error_at_max_depth": float(ratio[-1] - 1.0),
                }
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
        axis.set_title(tensor, fontsize=12, fontweight="bold")
        axis.set_xlabel("depth $L$")
        axis.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("tensor / exact leading term")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("Convergence to the componentwise critical ReLU coefficients")
    fig.subplots_adjust(bottom=0.25, top=0.84, left=0.055, right=0.995, wspace=0.22)
    fig.savefig(output_dir / "relu_tensor_leading_coefficients.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "relu_tensor_leading_coefficients.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)
    with (output_dir / "relu_tensor_leading_coefficients.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log_corrections = {
        "angle_map": (
            "theta_next = theta - theta^2/(3*pi) - theta^3/(18*pi^2) "
            "+ O(theta^4)"
        ),
        "distinct_input_angle": (
            "theta_ab(L) = 3*pi/[L + (3/2) log L + C_ab + o(1)]"
        ),
        "ntk_diagonal": "H_aa(L) = s_a^2 L/2 + O(1)",
        "ntk_off_diagonal": (
            "H_ab(L) = (s_a s_b/8)[L + (3/2) log L] + O(1), a != b"
        ),
        "interpretation": (
            "the plotted noninteger slopes are finite-window effective slopes; "
            "the first logarithmic drift already enters through angle and NTK transport"
        ),
    }
    (output_dir / "critical_log_corrections.json").write_text(
        json.dumps(log_corrections, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recursion-file",
        type=Path,
        default=Path(
            "data/feynman/exact_relu_tensor_depth2000/exact_relu_tensor_recursions.npz"
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/feynman/paper_depth_inputs.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/relu_tensor_asymptotics"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_convergence_analysis(args.recursion_file, args.input_file, args.output_dir)
    print(f"wrote exact asymptotic coefficients to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
