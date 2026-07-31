"""Aggregate paired multi-seed matrix-free solver audits.

The PCG-4 and PCG-10 summaries must use the same evaluation seeds.  The
result separates the shallow comparison (HB/Chebyshev-10 with a rare PCG-4
fallback) from the equal-HVP comparison against PCG-10.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RISK = "solver_z_h_relative_squared"


def load_summary(path: str) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def risk_row(
    seed: int,
    label: str,
    controller: str,
    summary: dict,
    hvp_depth: int,
    fallback_rate: float = 0.0,
    fallback_depth: int = 0,
) -> dict:
    risk = summary["controllers"][controller][RISK]
    return {
        "seed": seed,
        "label": label,
        "controller": controller,
        "hvp_depth": hvp_depth,
        "fallback_rate": fallback_rate,
        "expected_hvp_per_prompt": hvp_depth + fallback_rate * fallback_depth,
        **{f"h_risk_{key}": value for key, value in risk.items()},
    }


def paired_rows(pcg4_summaries: list[dict], pcg10_summaries: list[dict]) -> list[dict]:
    if len(pcg4_summaries) != len(pcg10_summaries):
        raise ValueError("PCG-4 and PCG-10 summary counts must agree")
    rows = []
    for seed, (shallow, equal_hvp) in enumerate(
        zip(pcg4_summaries, pcg10_summaries, strict=True)
    ):
        for key in ("examples", "eval_seed", "prompt_length", "z_scale"):
            if shallow.get(key) != equal_hvp.get(key):
                raise ValueError(
                    f"seed {seed}: paired summaries disagree on {key}"
                )
        hb_fallback = shallow["hybrid_fallback_rate"]
        cheb_fallback = shallow["chebyshev_fallback_rate"]
        rows.extend(
            [
                risk_row(seed, "HB learned", "learned_hb", shallow, 10),
                risk_row(
                    seed,
                    "Chebyshev learned",
                    "learned_chebyshev",
                    shallow,
                    10,
                ),
                risk_row(
                    seed,
                    "HB + PCG-4 guard",
                    "certified_hb_pcg",
                    shallow,
                    10,
                    hb_fallback,
                    4,
                ),
                risk_row(
                    seed,
                    "Chebyshev + PCG-4 guard",
                    "residual_guarded_chebyshev_pcg",
                    shallow,
                    10,
                    cheb_fallback,
                    4,
                ),
                risk_row(seed, "PCG-4", "pcg", shallow, 4),
                risk_row(seed, "PCG-10", "pcg", equal_hvp, 10),
                risk_row(seed, "HB oracle", "oracle_hb", shallow, 10),
                risk_row(
                    seed,
                    "Chebyshev oracle",
                    "oracle_chebyshev",
                    shallow,
                    10,
                ),
            ]
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    labels = list(dict.fromkeys(row["label"] for row in rows))
    output = []
    for label in labels:
        selected = [row for row in rows if row["label"] == label]
        means = np.asarray([row["h_risk_mean"] for row in selected])
        output.append(
            {
                "label": label,
                "seeds": len(selected),
                "mean_h_risk_across_seeds": float(means.mean()),
                "geometric_mean_h_risk_across_seeds": float(
                    np.exp(np.log(means).mean())
                ),
                "max_seed_h_risk": float(means.max()),
                "mean_expected_hvp_per_prompt": float(
                    np.mean(
                        [row["expected_hvp_per_prompt"] for row in selected]
                    )
                ),
                "mean_fallback_rate": float(
                    np.mean([row["fallback_rate"] for row in selected])
                ),
            }
        )
    return output


def plot(path: Path, rows: list[dict]) -> None:
    labels = list(dict.fromkeys(row["label"] for row in rows))
    means = []
    per_seed = []
    for label in labels:
        values = np.asarray(
            [row["h_risk_mean"] for row in rows if row["label"] == label]
        )
        means.append(values.mean())
        per_seed.append(values)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    positions = np.arange(len(labels))
    axes[0].bar(positions, means, color="#4C78A8", alpha=0.72)
    offsets = np.linspace(-0.18, 0.18, len(per_seed[0]))
    for seed, offset in enumerate(offsets):
        axes[0].scatter(
            positions + offset,
            [values[seed] for values in per_seed],
            s=34,
            label=f"seed {seed}",
            zorder=3,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"mean relative $H$-risk")
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].set_title("Matrix-free controller accuracy")
    axes[0].grid(axis="y", which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    guarded = [
        row
        for row in rows
        if row["label"] in {"HB + PCG-4 guard", "Chebyshev + PCG-4 guard"}
    ]
    guard_labels = ["HB guard", "Chebyshev guard"]
    for index, full_label in enumerate(
        ["HB + PCG-4 guard", "Chebyshev + PCG-4 guard"]
    ):
        values = [
            100.0 * row["fallback_rate"]
            for row in guarded
            if row["label"] == full_label
        ]
        axes[1].bar(
            index,
            np.mean(values),
            color="#F58518" if index == 0 else "#54A24B",
            alpha=0.75,
        )
        axes[1].scatter(
            np.full(len(values), index) + offsets,
            values,
            color="black",
            s=28,
            zorder=3,
        )
    axes[1].set_xticks([0, 1], guard_labels)
    axes[1].set_ylabel("PCG fallback rate (%)")
    axes[1].set_title("Rare interval failures")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcg4-summaries", nargs="+", required=True)
    parser.add_argument("--pcg10-summaries", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    shallow = [load_summary(path) for path in args.pcg4_summaries]
    equal_hvp = [load_summary(path) for path in args.pcg10_summaries]
    rows = paired_rows(shallow, equal_hvp)
    outdir = Path(args.outdir)
    write_rows(outdir / "per_seed.csv", rows)
    write_rows(outdir / "aggregate.csv", aggregate(rows))
    plot(outdir / "matrix_free_multiseed_comparison.png", rows)


if __name__ == "__main__":
    main()
