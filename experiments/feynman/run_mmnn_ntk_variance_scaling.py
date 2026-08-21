#!/usr/bin/env python3
"""Faithful finite-MMNN validation of width, rank, and depth scaling.

Networks use the exact frozen-feature/trainable-readout MMNN recursion from
``exact_mmnn_ntk_recursion.py``.  Frozen factors are sampled either as iid
Gaussian matrices (the architecture used by MMNN) or as scaled Haar-Stiefel
matrices (the whitened orientation surrogate).  The experiment measures the
ensemble covariance of the empirical scalar-output NTK and tests the double
expansion in 1/n and 1/r rather than silently replacing the real MMNN by a
projector-only model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exact_mmnn_ntk_recursion import MMNNLayer, exact_pathwise_ntk


def frozen_matrix(
    rng: np.random.Generator, width: int, input_rank: int, ensemble: str
) -> np.ndarray:
    if ensemble == "gaussian":
        return rng.normal(size=(width, input_rank))
    if ensemble == "stiefel":
        raw = rng.normal(size=(width, input_rank))
        q, r = np.linalg.qr(raw, mode="reduced")
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        q *= signs[None, :]
        return np.sqrt(width) * q
    raise ValueError(ensemble)


def sample_layers(
    rng: np.random.Generator,
    *,
    input_rank: int,
    hidden_rank: int,
    width: int,
    depth: int,
    ensemble: str,
) -> list[MMNNLayer]:
    ranks = [input_rank] + [hidden_rank] * (depth - 1) + [1]
    layers = []
    for rank_in, rank_out in zip(ranks, ranks[1:]):
        layers.append(
            MMNNLayer(
                W=frozen_matrix(rng, width, rank_in, ensemble),
                b=np.zeros(width),
                A=np.sqrt(2.0) * rng.normal(size=(rank_out, width)),
                c=np.zeros(rank_out),
            )
        )
    return layers


def estimate_ntk_statistics(
    inputs: np.ndarray,
    *,
    width: int,
    rank: int,
    depth: int,
    samples: int,
    ensemble: str,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kernels = []
    for _ in range(samples):
        layers = sample_layers(
            rng,
            input_rank=inputs.shape[1],
            hidden_rank=rank,
            width=width,
            depth=depth,
            ensemble=ensemble,
        )
        kernels.append(exact_pathwise_ntk(inputs, layers)[:, :, 0, 0])
    kernels = np.asarray(kernels)
    mean = kernels.mean(axis=0)
    centered = kernels - mean
    upper = np.triu_indices(len(inputs))
    variances = centered[:, upper[0], upper[1]].var(axis=0, ddof=1)
    off_diagonal = np.triu_indices(len(inputs), k=1)
    return {
        "entry_variance_mean": float(variances.mean()),
        "offdiagonal_variance_mean": float(
            centered[:, off_diagonal[0], off_diagonal[1]].var(axis=0, ddof=1).mean()
        ),
        "selected_variance_01": float(centered[:, 0, 1].var(ddof=1)),
        "mean_ntk_frobenius": float(np.linalg.norm(mean)),
    }


def write_scaling_fits(rows: list[dict], output_dir: Path) -> list[dict]:
    """Write the perturbative 1/r and 1/n linear-fit diagnostics."""
    fit_rows = []
    for ensemble in ("gaussian", "stiefel"):
        for sweep, coordinate in (("rank", "inverse_rank"), ("width", "inverse_width")):
            selected = [
                row
                for row in rows
                if row["ensemble"] == ensemble
                and row["sweep"] == sweep
                and float(row["perturbative_parameter"]) < 1.0
            ]
            x = np.asarray([float(row[coordinate]) for row in selected])
            y = np.asarray(
                [float(row["offdiagonal_variance_mean"]) for row in selected]
            )
            slope, intercept = np.polyfit(x, y, 1)
            prediction = intercept + slope * x
            residual = float(np.sum((y - prediction) ** 2))
            total = float(np.sum((y - y.mean()) ** 2))
            fit_rows.append(
                {
                    "ensemble": ensemble,
                    "sweep": sweep,
                    "coordinate": coordinate,
                    "number_of_points": len(selected),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": 1.0 - residual / total,
                    "criterion": "L^3(1/n+1/r)<1",
                }
            )
    with (output_dir / "mmnn_ntk_scaling_fits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fit_rows[0].keys())
        writer.writeheader()
        writer.writerows(fit_rows)
    return fit_rows


def run_experiment(output_dir: Path, samples: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    angles = np.asarray([0.05, 0.47, 1.06, 1.74, 2.51, 3.42])
    inputs = np.stack((np.cos(angles), np.sin(angles)), axis=1) * np.sqrt(2.0)
    configurations = []
    for ensemble in ("gaussian", "stiefel"):
        for depth in (2, 4, 6, 8, 12, 16):
            configurations.append((ensemble, 96, 16, depth, "depth"))
        # Keep the rank/width sweeps shallow so that a substantial part of
        # each sweep lies inside L^3(1/n+1/r)<1.  The separate depth sweep
        # deliberately crosses that perturbative boundary.
        for rank in (4, 8, 12, 16, 24, 32, 48, 64, 96):
            configurations.append((ensemble, 96, rank, 2, "rank"))
        for width in (48, 64, 96, 128, 192):
            configurations.append((ensemble, width, 16, 2, "width"))

    rows = []
    for index, (ensemble, width, rank, depth, sweep) in enumerate(configurations):
        statistics = estimate_ntk_statistics(
            inputs,
            width=width,
            rank=rank,
            depth=depth,
            samples=samples,
            ensemble=ensemble,
            seed=seed + 1009 * index,
        )
        rows.append(
            {
                "ensemble": ensemble,
                "sweep": sweep,
                "width": width,
                "rank": rank,
                "depth": depth,
                "samples": samples,
                "inverse_width": 1.0 / width,
                "inverse_rank": 1.0 / rank,
                "gaussian_double_expansion": 1.0 / width + 1.0 / rank,
                "perturbative_parameter": depth**3 * (1.0 / width + 1.0 / rank),
                **statistics,
            }
        )
        print(
            f"{ensemble:8s} {sweep:5s} n={width:3d} r={rank:3d} "
            f"L={depth:2d} variance={statistics['offdiagonal_variance_mean']:.4e}"
        )
    with (output_dir / "mmnn_ntk_variance_scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_scaling_fits(rows, output_dir)

    colors = {"gaussian": "#276FBF", "stiefel": "#D1495B"}
    fig, axes = plt.subplots(1, 3, figsize=(14.7, 3.75), constrained_layout=True)
    for ensemble in ("gaussian", "stiefel"):
        selected = [
            row
            for row in rows
            if row["ensemble"] == ensemble and row["sweep"] == "depth"
        ]
        depth = np.asarray([row["depth"] for row in selected])
        variance = np.asarray([row["offdiagonal_variance_mean"] for row in selected])
        slope, intercept = np.polyfit(np.log(depth), np.log(variance), 1)
        axes[0].loglog(
            depth,
            variance,
            "o-",
            color=colors[ensemble],
            label=rf"{ensemble}, slope {slope:.2f}",
        )
        axes[0].loglog(
            depth,
            np.exp(intercept) * depth**slope,
            linestyle="--",
            color=colors[ensemble],
            alpha=0.65,
        )
        selected = [
            row
            for row in rows
            if row["ensemble"] == ensemble and row["sweep"] == "rank"
        ]
        inverse_rank = np.asarray([row["inverse_rank"] for row in selected])
        variance = np.asarray([row["offdiagonal_variance_mean"] for row in selected])
        order = np.argsort(inverse_rank)
        axes[1].plot(
            inverse_rank[order],
            variance[order],
            "o-",
            color=colors[ensemble],
            label=ensemble,
        )
        perturbative = np.asarray(
            [row["perturbative_parameter"] < 1.0 for row in selected]
        )
        if np.count_nonzero(perturbative) >= 2:
            slope, intercept = np.polyfit(
                inverse_rank[perturbative], variance[perturbative], 1
            )
            fit_x = np.linspace(
                inverse_rank[perturbative].min(),
                inverse_rank[perturbative].max(),
                100,
            )
            axes[1].plot(
                fit_x,
                intercept + slope * fit_x,
                linestyle="--",
                color=colors[ensemble],
                alpha=0.65,
            )

        selected = [
            row
            for row in rows
            if row["ensemble"] == ensemble and row["sweep"] == "width"
        ]
        inverse_width = np.asarray([row["inverse_width"] for row in selected])
        variance = np.asarray([row["offdiagonal_variance_mean"] for row in selected])
        order = np.argsort(inverse_width)
        axes[2].plot(
            inverse_width[order],
            variance[order],
            "o-",
            color=colors[ensemble],
            label=ensemble,
        )
        perturbative = np.asarray(
            [row["perturbative_parameter"] < 1.0 for row in selected]
        )
        if np.count_nonzero(perturbative) >= 2:
            slope, intercept = np.polyfit(
                inverse_width[perturbative], variance[perturbative], 1
            )
            fit_x = np.linspace(
                inverse_width[perturbative].min(),
                inverse_width[perturbative].max(),
                100,
            )
            axes[2].plot(
                fit_x,
                intercept + slope * fit_x,
                linestyle="--",
                color=colors[ensemble],
                alpha=0.65,
            )
    threshold = (1.0 / (1.0 / 96.0 + 1.0 / 16.0)) ** (1.0 / 3.0)
    axes[0].axvline(
        threshold,
        color="0.35",
        linestyle=":",
        linewidth=1.1,
        label=r"$L^3(1/n+1/r)=1$",
    )
    axes[0].axvspan(threshold, 16.0, color="0.5", alpha=0.07)
    axes[0].set_title("depth amplification and breakdown")
    axes[0].set_xlabel("MMNN depth $L$")
    axes[0].set_ylabel("mean off-diagonal NTK variance")
    axes[1].set_title("rank expansion at $L=2$, $n=96$")
    axes[1].set_xlabel("$1/r$")
    axes[1].set_ylabel("mean off-diagonal NTK variance")
    axes[2].set_title("width expansion at $L=2$, $r=16$")
    axes[2].set_xlabel("$1/n$")
    axes[2].set_ylabel("mean off-diagonal NTK variance")
    for axis in axes:
        axis.grid(alpha=0.22, which="both")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "mmnn_ntk_variance_scaling.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "mmnn_ntk_variance_scaling.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/feynman/mmnn_ntk_variance_scaling"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.output_dir, args.samples, args.seed)
    print(f"wrote faithful MMNN variance results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
