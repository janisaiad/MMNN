#!/usr/bin/env python3
"""Generate the diagnostic figures used by the English bottleneck report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from poc import (
    FixedGeometryGP,
    exact_krr,
    load_icl_checkpoint,
    query_weights,
    rbf_kernel,
    set_seed,
    weighted_episode_mse,
)


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def lookup(
    rows: list[dict[str, str]],
    *,
    variant: str | None = None,
    n_context: int | None = None,
    method: str | None = None,
    budget: int | None = None,
    design: str | None = None,
    predictor: str | None = None,
) -> float:
    conditions = {
        "variant": variant,
        "n_context": None if n_context is None else str(n_context),
        "method": method,
        "budget": None if budget is None else str(budget),
        "design": design,
        "predictor": predictor,
    }
    matches = [
        row
        for row in rows
        if all(expected is None or row.get(key) == expected for key, expected in conditions.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(matches)}")
    value_key = "mean_mse" if "mean_mse" in matches[0] else "mean_weighted_mse"
    return float(matches[0][value_key])


def greedy_posterior_risk(
    kernel: Tensor,
    weights: Tensor,
    budget: int,
    noise_variance: float,
) -> tuple[Tensor, Tensor]:
    """Sequentially minimize integrated posterior variance."""
    covariance = kernel.double().clone()
    weights = weights.double()
    selected: list[int] = []
    risks: list[float] = []
    for _ in range(budget):
        gain = (weights[:, None] * covariance.square()).sum(dim=0)
        gain = gain / (covariance.diag() + noise_variance)
        if selected:
            gain[torch.tensor(selected)] = -torch.inf
        index = int(gain.argmax())
        selected.append(index)
        column = covariance[:, index].clone()
        covariance -= torch.outer(column, column) / (
            covariance[index, index] + noise_variance
        )
        covariance = 0.5 * (covariance + covariance.T)
        risks.append(float(weights @ covariance.diag()))
    return torch.tensor(selected), torch.tensor(risks)


def posterior_standard_deviation(
    task: FixedGeometryGP,
    indices: Tensor,
) -> Tensor:
    kernel = task.kernel
    cross = kernel[:, indices]
    context = kernel[indices][:, indices]
    eye = torch.eye(indices.numel(), dtype=kernel.dtype, device=kernel.device)
    solved = torch.linalg.solve(context + task.cfg.noise_std**2 * eye, cross.T)
    variance = kernel.diag() - (cross * solved.T).sum(dim=-1)
    return variance.clamp_min(0.0).sqrt()


def prediction(
    model: torch.nn.Module,
    task: FixedGeometryGP,
    observed: Tensor,
    indices: Tensor,
    depth: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batched_indices = indices.unsqueeze(0)
    x_context, y_context = task.gather_context(observed, batched_indices)
    looped, _ = model(x_context, y_context, task.grid, depth=depth)
    exact = exact_krr(
        x_context,
        y_context,
        task.grid,
        task.cfg.true_lengthscale,
        task.cfg.ridge,
    )
    return looped[0], exact[0], x_context[0, :, 0], y_context[0]


def make_decomposition_figure(
    summary_dir: Path,
    task: FixedGeometryGP,
    icl_rows: list[dict[str, str]],
    design_rows: list[dict[str, str]],
) -> dict[str, float | list[float]]:
    random_exact = lookup(
        icl_rows, variant="matched", n_context=12, method="exact_krr"
    )
    random_loop = lookup(icl_rows, variant="matched", n_context=12, method="model")
    design_exact = lookup(
        design_rows,
        budget=8,
        design="learned",
        predictor="exact_krr",
    )
    design_loop = lookup(
        design_rows,
        budget=8,
        design="learned",
        predictor="looped_icl",
    )

    weights = query_weights(task.grid).double()
    report_kernel = rbf_kernel(
        task.grid.double(), task.grid.double(), task.cfg.true_lengthscale
    )
    _, noisy_risk = greedy_posterior_risk(
        report_kernel, weights, budget=24, noise_variance=task.cfg.noise_std**2
    )
    _, noiseless_risk = greedy_posterior_risk(
        report_kernel, weights, budget=24, noise_variance=1.0e-10
    )
    weighted_kernel = (
        weights.sqrt()[:, None]
        * report_kernel
        * weights.sqrt()[None, :]
    )
    eigenvalues = torch.linalg.eigvalsh(weighted_kernel).flip(0)
    spectral_tail = torch.tensor(
        [float(eigenvalues[budget:].sum()) for budget in range(1, 25)]
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    bayes = np.asarray([random_exact, design_exact])
    solver = np.asarray([random_loop - random_exact, design_loop - design_exact])
    positions = np.arange(2)
    axes[0].bar(positions, bayes, color="#4c78a8", label="Bayes risk: data + noise")
    axes[0].bar(
        positions,
        solver,
        bottom=bayes,
        color="#f58518",
        label="finite-loop excess",
    )
    for position, total, floor in zip(positions, bayes + solver, bayes, strict=True):
        axes[0].text(
            position,
            total + 0.004,
            f"{100.0 * floor / total:.1f}% data/noise",
            ha="center",
            fontsize=9,
        )
    axes[0].set_xticks(positions, ["12 random sensors\nuniform loss", "8 learned sensors\nweighted loss"])
    axes[0].set_ylabel("held-out latent-function MSE")
    axes[0].set_ylim(0.0, 0.14)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_title("Observed error decomposition")

    budgets = np.arange(1, 25)
    axes[1].plot(
        budgets,
        noisy_risk.numpy(),
        "o-",
        color="#4c78a8",
        markersize=3,
        label=r"point sensors, $\sigma=0.1$",
    )
    axes[1].plot(
        budgets,
        noiseless_risk.clamp_min(1.0e-12).numpy(),
        "s-",
        color="#54a24b",
        markersize=3,
        label="point sensors, noiseless",
    )
    axes[1].plot(
        budgets,
        spectral_tail.clamp_min(1.0e-12).numpy(),
        "--",
        color="#999999",
        label="ideal noiseless rank-$B$ bound",
    )
    axes[1].axhline(0.01, color="#e45756", linestyle=":", label="MSE = 0.01")
    axes[1].axvline(8, color="#222222", linestyle=":", alpha=0.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("measurement budget $B$")
    axes[1].set_ylabel("weighted posterior risk")
    axes[1].set_title("Information floor versus sensor budget")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5)
    figure.tight_layout()
    figure.savefig(summary_dir / "bottleneck_decomposition.png", dpi=240)
    plt.close(figure)

    return {
        "random_12_exact_mse": random_exact,
        "random_12_loop_mse": random_loop,
        "random_12_solver_excess": random_loop - random_exact,
        "learned_8_exact_weighted_mse": design_exact,
        "learned_8_loop_weighted_mse": design_loop,
        "learned_8_solver_excess": design_loop - design_exact,
        "weighted_greedy_noisy_risk": noisy_risk.tolist(),
        "weighted_greedy_noiseless_risk": noiseless_risk.tolist(),
        "weighted_ideal_rank_bound": spectral_tail.tolist(),
    }


@torch.no_grad()
def make_reconstruction_figure(
    results_dir: Path,
    summary_dir: Path,
    task: FixedGeometryGP,
    model: torch.nn.Module,
) -> dict[str, float | int | list[int]]:
    selection_path = results_dir / "design_separated" / "seed_0" / "selected_indices.json"
    selections = json.loads(selection_path.read_text())
    learned_indices = torch.tensor(selections["learned"], dtype=torch.long)
    uniform_weights = torch.full((task.cfg.grid_size,), 1.0 / task.cfg.grid_size)
    global_indices, _ = greedy_posterior_risk(
        task.kernel, uniform_weights, budget=16, noise_variance=task.cfg.noise_std**2
    )

    set_seed(20_260_804)
    latent_batch, observed_batch = task.sample(512)
    batch_indices = learned_indices.unsqueeze(0).expand(latent_batch.shape[0], -1)
    x_context, y_context = task.gather_context(observed_batch, batch_indices)
    exact_batch = exact_krr(
        x_context,
        y_context,
        task.grid,
        task.cfg.true_lengthscale,
        task.cfg.ridge,
    )
    weights = query_weights(task.grid)
    episode_losses = weighted_episode_mse(exact_batch, latent_batch, weights)
    order = episode_losses.argsort()
    representative = int(order[order.numel() // 2])
    latent = latent_batch[representative]
    observed = observed_batch[representative : representative + 1]

    cases = [
        ("Current: learned $B=8$, $T=12$", learned_indices, 12),
        ("More compute only: learned $B=8$, $T=32$", learned_indices, 32),
        ("More information: global $B=16$, $T=32$", global_indices, 32),
    ]
    x_grid = task.grid[:, 0]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharex=True, sharey=True)
    metrics: dict[str, float | int | list[int]] = {
        "representative_episode": representative,
        "learned_indices": learned_indices.tolist(),
        "global_greedy_16_indices": global_indices.tolist(),
    }
    for case_number, (axis, (title, indices, depth)) in enumerate(
        zip(axes, cases, strict=True), start=1
    ):
        looped, exact, x_context, y_context = prediction(
            model, task, observed, indices, depth
        )
        std = posterior_standard_deviation(task, indices)
        loop_global = float((looped - latent).square().mean())
        exact_global = float((exact - latent).square().mean())
        loop_weighted = float(((looped - latent).square() * weights).sum())
        metrics[f"case_{case_number}_loop_global_mse"] = loop_global
        metrics[f"case_{case_number}_exact_global_mse"] = exact_global
        metrics[f"case_{case_number}_loop_weighted_mse"] = loop_weighted

        axis.fill_between(
            x_grid.numpy(),
            (exact - 1.96 * std).numpy(),
            (exact + 1.96 * std).numpy(),
            color="#4c78a8",
            alpha=0.16,
            label="95% posterior band",
        )
        axis.plot(x_grid, latent, color="#222222", linewidth=2.0, label="latent function")
        axis.plot(x_grid, exact, color="#4c78a8", linewidth=1.6, label="exact GP mean")
        axis.plot(
            x_grid,
            looped,
            color="#f58518",
            linewidth=1.4,
            linestyle="--",
            label="looped predictor",
        )
        axis.scatter(
            x_context,
            y_context,
            color="#e45756",
            s=22,
            zorder=4,
            label="measurements",
        )
        axis.set_title(
            title + f"\nglobal MSE: {loop_global:.3f} (GP: {exact_global:.3f})",
            fontsize=10,
        )
        axis.set_xlabel("location")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("function value")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(summary_dir / "representative_reconstruction_en.png", dpi=240)
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    summary_dir = results_dir / "summary"
    icl_rows = read_summary(summary_dir / "icl_summary.csv")
    design_rows = read_summary(summary_dir / "design_summary.csv")

    checkpoint = results_dir / "icl" / "matched" / "seed_0" / "final.pt"
    model, geometry, _ = load_icl_checkpoint(checkpoint, torch.device("cpu"))
    task = FixedGeometryGP(geometry, torch.device("cpu"))
    decomposition = make_decomposition_figure(summary_dir, task, icl_rows, design_rows)
    reconstruction = make_reconstruction_figure(
        results_dir, summary_dir, task, model
    )
    weight_ratio = float(query_weights(task.grid).max() / query_weights(task.grid).min())
    payload = {
        **decomposition,
        **reconstruction,
        "objective_center_to_boundary_weight_ratio": weight_ratio,
        "noise_standard_deviation": task.cfg.noise_std,
        "kernel_lengthscale": task.cfg.true_lengthscale,
    }
    (summary_dir / "bottleneck_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
