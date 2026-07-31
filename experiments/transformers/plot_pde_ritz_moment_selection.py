"""Plot the paired PDE controller selection against scalar-HVP work."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_methods(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["method"]: {
            key: float(value)
            for key, value in row.items()
            if key not in {"design", "method"} and value
        }
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hb",
        type=Path,
        default=Path("experiments/transformers/pde_matrix_free_learning_tight_certificate/aggregate.csv"),
    )
    parser.add_argument(
        "--measure",
        type=Path,
        default=Path("experiments/transformers/pde_moment_chebyshev_tight_shared_head/aggregate.csv"),
    )
    parser.add_argument(
        "--ritz-q1",
        type=Path,
        default=Path("experiments/transformers/pde_ritz_moment_shared_head/q1/aggregate.csv"),
    )
    parser.add_argument(
        "--ritz-q2",
        type=Path,
        default=Path("experiments/transformers/pde_ritz_moment_shared_head/q2/aggregate.csv"),
    )
    parser.add_argument(
        "--ritz-q3",
        type=Path,
        default=Path("experiments/transformers/pde_ritz_moment_shared_head/q3/aggregate.csv"),
    )
    parser.add_argument(
        "--ritz-q4",
        type=Path,
        default=Path("experiments/transformers/pde_ritz_moment_shared_head/q4/aggregate.csv"),
    )
    parser.add_argument(
        "--ritz-trained",
        type=Path,
        default=Path("experiments/transformers/pde_ritz_moment_training/aggregate.csv"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    hb = read_methods(args.hb)
    measure = read_methods(args.measure)
    q1 = read_methods(args.ritz_q1)
    q2 = read_methods(args.ritz_q2)
    q3 = read_methods(args.ritz_q3)
    q4 = read_methods(args.ritz_q4)
    trained = read_methods(args.ritz_trained)
    entries = [
        ("pure PCG-8", measure["identity_pcg"], 8),
        ("head + HB-8", hb["trained_head_hb"], 20),
        ("head + learned measure", measure["learned_moment_chebyshev"], 20),
        ("head + Ritz q=1", q1["ritz_moment_chebyshev"], 24),
        ("head + Ritz q=2", q2["ritz_moment_chebyshev"], 28),
        ("head + Ritz q=3", q3["ritz_moment_chebyshev"], 32),
        ("head + Ritz q=4", q4["ritz_moment_chebyshev"], 36),
        (
            "fine-tuned Ritz q=2",
            trained["trained_head_ritz_moment_chebyshev"],
            28,
        ),
        ("pure PCG-28", q2["identity_pcg_ritz_equal_work"], 28),
        ("head + PCG-8", measure["same_preconditioner_pcg"], 20),
    ]

    labels = [entry[0] for entry in entries]
    risks = [entry[1]["h_relative_mean"] for entry in entries]
    errors = [entry[1]["h_relative_std"] for entry in entries]
    work = [entry[2] for entry in entries]
    colors = [
        "#7f7f7f",
        "#2ca02c",
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#c5a5cf",
        "#6f4c9b",
        "#8c564b",
        "#4c78a8",
        "#1f77b4",
    ]

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    axes[0].bar(
        range(len(entries)),
        risks,
        yerr=errors,
        color=colors,
        capsize=3,
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(
        range(len(entries)),
        [label.replace(" + ", "\n+\n") for label in labels],
        rotation=28,
        ha="right",
        fontsize=8,
    )
    axes[0].set_ylabel("mean relative H-risk")
    axes[0].set_title("Same PDE law, three seeds")
    axes[0].grid(axis="y", alpha=0.25)

    annotation_offsets = {
        "head + Ritz q=2": (6, 12),
        "head + Ritz q=3": (6, 10),
        "head + Ritz q=4": (6, -14),
        "fine-tuned Ritz q=2": (6, -16),
        "pure PCG-28": (6, 5),
    }
    for label, risk, cost, color in zip(labels, risks, work, colors, strict=True):
        axes[1].scatter(cost, risk, s=75, color=color, zorder=3)
        axes[1].annotate(
            label,
            (cost, risk),
            xytext=annotation_offsets.get(label, (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("sequential scalar-HVP equivalent")
    axes[1].set_ylabel("mean relative H-risk")
    axes[1].set_title("Risk–work frontier")
    axes[1].grid(alpha=0.25)
    axes[1].set_xlim(6, 40)

    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
