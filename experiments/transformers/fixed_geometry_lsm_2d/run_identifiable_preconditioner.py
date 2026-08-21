#!/usr/bin/env python3
"""Audit and train an identifiable A-equivariant LSM preconditioner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import nnls
from torch import Tensor, nn

from .foundation_uq import EquivariantChebyshevPCGLSMLoop, FoundationPCGLSMLoop
from .lsm_core import (
    build_bayesian_system,
    exact_bayesian_lsm,
    ranking_loss,
    set_seed,
)
from .run_experiments import task_metrics, write_rows
from .run_foundation_uq import (
    SCENARIOS,
    acquire,
    aggregate,
    draw_training_batch,
    make_model,
    physics_cache,
    sample_multi_obstacle_masks,
)

DEPTHS = (4, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-results-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--eval-tasks", type=int, default=48)
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_graph_model(
    kind: str,
    seed: int,
    graph_results_dir: Path,
    physics,
) -> nn.Module:
    checkpoint = torch.load(
        graph_results_dir / f"{kind}_seed_{seed}.pt",
        map_location=physics.device,
    )
    model = make_model(kind, physics, depth=int(checkpoint["depth"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def train_equivariant(
    seed: int,
    cache,
    *,
    steps: int,
    batch_size: int,
    depth: int,
    log_every: int,
) -> tuple[EquivariantChebyshevPCGLSMLoop, list[dict[str, object]]]:
    set_seed(seed + 700_000)
    physics = cache[8.0]
    model = EquivariantChebyshevPCGLSMLoop(
        physics.kernel,
        physics.feature_kernel,
        physics.cfg.ridge_rel,
        depth=depth,
        polynomial_degree=12,
        moment_degree=8,
        controller_width=96,
        coefficient_mode="learned",
    ).to(physics.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-6)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for step in range(steps + 1):
        set_seed(seed * 1_000_000 + step + 91_000)
        far_field, probe, kernel, feature, mask, metadata = draw_training_batch(
            cache, batch_size
        )
        score, info = model(
            far_field,
            probe,
            kernel=kernel,
            feature_kernel=feature,
            depth=depth,
            identify_witnesses=8,
        )
        identification = info["identification_loss"].mean()
        rank = ranking_loss(score, mask, n_pairs=48 if steps > 200 else 12)
        residual = torch.log1p(info["relative_residual"]).mean()
        loss = identification + 0.05 * rank + 0.05 * residual
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % log_every == 0 or step == steps:
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "loss": float(loss.detach()),
                    "identification_loss": float(identification.detach()),
                    "ranking_loss": float(rank.detach()),
                    "relative_residual": float(info["relative_residual"].mean().detach()),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - start,
                    **metadata,
                }
            )
            print(
                f"equivariant seed={seed} step={step:04d} loss={float(loss):.4f} "
                f"id={float(identification):.4f} "
                f"residual={float(info['relative_residual'].mean()):.4f}"
            )
    model.eval()
    return model, rows


def graph_commutator(operator: Tensor, feature_kernel: Tensor) -> Tensor:
    batch_size, n_angles, _ = operator.shape
    if feature_kernel.ndim == 2:
        feature = feature_kernel.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        feature = feature_kernel
    identity = torch.eye(n_angles, device=operator.device, dtype=operator.dtype)
    laplacian = identity.unsqueeze(0) - feature
    commutator = operator @ laplacian - laplacian @ operator
    numerator = torch.linalg.matrix_norm(commutator, ord="fro")
    denominator = (
        torch.linalg.matrix_norm(operator, ord="fro")
        * torch.linalg.matrix_norm(laplacian, ord="fro")
    ).clamp_min(1.0e-12)
    return numerator / denominator


def oracle_graph_preconditioner(
    operator: Tensor,
    feature_kernel: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size, n_angles, _ = operator.shape
    if feature_kernel.ndim == 2:
        feature = feature_kernel.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        feature = feature_kernel
    identity = torch.eye(n_angles, device=operator.device, dtype=operator.dtype)
    laplacian = identity.unsqueeze(0) - feature
    laplacian_squared = laplacian @ laplacian
    coefficients = []
    preconditioners = []
    for task in range(batch_size):
        target = identity - operator[task]
        basis = torch.stack(
            [
                laplacian[task] @ operator[task],
                laplacian_squared[task] @ operator[task],
            ],
            dim=-1,
        )
        design = torch.cat(
            [basis.real.reshape(-1, 2), basis.imag.reshape(-1, 2)], dim=0
        ).double().cpu().numpy()
        response = torch.cat([target.real.flatten(), target.imag.flatten()]).double()
        solution, _ = nnls(design, response.cpu().numpy())
        coefficient = torch.tensor(
            solution,
            device=operator.device,
            dtype=operator.real.dtype,
        )
        preconditioner = (
            identity
            + coefficient[0] * laplacian[task]
            + coefficient[1] * laplacian_squared[task]
        )
        diagonal_mean = preconditioner.diagonal().real.mean().clamp_min(1.0e-8)
        preconditioners.append(preconditioner / diagonal_mean)
        coefficients.append(coefficient)
    return torch.stack(preconditioners), torch.stack(coefficients)


def preconditioned_condition(operator: Tensor, factor: Tensor) -> Tensor:
    whitened = factor.mH @ operator @ factor
    eigenvalues = torch.linalg.eigvalsh(0.5 * (whitened + whitened.mH)).clamp_min(1.0e-10)
    return eigenvalues.amax(dim=-1) / eigenvalues.amin(dim=-1)


def preconditioner_square_root(preconditioner: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(preconditioner)
    return (eigenvectors * eigenvalues.clamp_min(1.0e-10).sqrt()[:, None, :]) @ eigenvectors.mH


@torch.no_grad()
def evaluate_seed(
    seed: int,
    cache,
    graph_small: nn.Module,
    graph_foundation: nn.Module,
    equivariant: EquivariantChebyshevPCGLSMLoop,
    analytic: EquivariantChebyshevPCGLSMLoop,
    *,
    eval_tasks: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    task_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        set_seed(seed + 5_000_000 + scenario_index * 10_000)
        physics = cache[scenario.wavenumber]
        completed = 0
        while completed < eval_tasks:
            batch_size = min(12, eval_tasks - completed)
            mask, _, _ = sample_multi_obstacle_masks(
                physics,
                batch_size,
                scenario.count,
                mode=scenario.mode,
            )
            far_field, probe, kernel, feature = acquire(
                physics,
                mask,
                noise=scenario.noise,
                aperture_degrees=scenario.aperture_degrees,
                heteroscedastic=scenario.heteroscedastic,
            )
            system = build_bayesian_system(
                far_field, kernel, probe, physics.cfg.ridge_rel
            )
            identity_score, identity_info = graph_small(
                far_field,
                probe,
                kernel=kernel,
                feature_kernel=feature,
                depth=8,
                fixed_preconditioner=(0.0, 0.0),
            )
            evaluations: dict[str, tuple[Tensor, dict[str, Tensor]]] = {
                "identity": (identity_score, identity_info),
                "graph-small": graph_small(
                    far_field,
                    probe,
                    kernel=kernel,
                    feature_kernel=feature,
                    depth=8,
                ),
                "graph-foundation": graph_foundation(
                    far_field,
                    probe,
                    kernel=kernel,
                    feature_kernel=feature,
                    depth=8,
                ),
                "A-equivariant": equivariant(
                    far_field,
                    probe,
                    kernel=kernel,
                    feature_kernel=feature,
                    depth=8,
                    certify=True,
                ),
                "analytic-Chebyshev": analytic(
                    far_field,
                    probe,
                    kernel=kernel,
                    feature_kernel=feature,
                    depth=8,
                    certify=True,
                ),
                "exact": exact_bayesian_lsm(
                    far_field,
                    probe,
                    kernel,
                    physics.cfg.ridge_rel,
                ),
            }
            oracle_preconditioner, oracle_coefficients = oracle_graph_preconditioner(
                system["operator"], feature
            )
            _, oracle_residual, _ = FoundationPCGLSMLoop._pcg(
                system["operator"],
                system["rhs"],
                oracle_preconditioner,
                8,
                return_history=False,
            )
            commutator = graph_commutator(system["operator"], feature)
            oracle_sqrt = preconditioner_square_root(oracle_preconditioner)
            oracle_condition = preconditioned_condition(
                system["operator"], oracle_sqrt
            )

            equivariant_info = evaluations["A-equivariant"][1]
            analytic_info = evaluations["analytic-Chebyshev"][1]
            conditions = {
                "A-equivariant": preconditioned_condition(
                    system["operator"], equivariant_info["factor"]
                ),
                "analytic-Chebyshev": preconditioned_condition(
                    system["operator"], analytic_info["factor"]
                ),
            }
            for model_name, (score, info) in evaluations.items():
                metrics = task_metrics(score, mask, physics.grid)
                for task_index, metric in enumerate(metrics):
                    task_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "scenario": scenario.name,
                            "category": scenario.category,
                            "task": completed + task_index,
                            "depth": 8 if model_name != "exact" else 0,
                            **metric,
                            "relative_residual": float(
                                info["relative_residual"][task_index]
                            ),
                        }
                    )
            for task_index in range(batch_size):
                diagnostic_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario.name,
                        "category": scenario.category,
                        "task": completed + task_index,
                        "commutator": float(commutator[task_index]),
                        "oracle_graph_residual": float(oracle_residual[task_index]),
                        "oracle_graph_condition": float(oracle_condition[task_index]),
                        "oracle_graph_c1": float(oracle_coefficients[task_index, 0]),
                        "oracle_graph_c2": float(oracle_coefficients[task_index, 1]),
                        "equivariant_epsilon": float(
                            equivariant_info["certificate_epsilon"][task_index]
                        ),
                        "equivariant_condition_bound": float(
                            equivariant_info["condition_bound"][task_index]
                        ),
                        "equivariant_actual_condition": float(
                            conditions["A-equivariant"][task_index]
                        ),
                        "equivariant_certified": float(
                            equivariant_info["certificate_epsilon"][task_index] < 0.95
                        ),
                        "analytic_epsilon": float(
                            analytic_info["certificate_epsilon"][task_index]
                        ),
                        "analytic_actual_condition": float(
                            conditions["analytic-Chebyshev"][task_index]
                        ),
                        "analytic_certified": float(
                            analytic_info["certificate_epsilon"][task_index] < 0.95
                        ),
                    }
                )
            completed += batch_size
    return task_rows, diagnostic_rows


def plot_results(
    path: Path,
    task_summary: list[dict[str, object]],
    diagnostic_summary: list[dict[str, object]],
) -> None:
    model_order = (
        "identity",
        "graph-small",
        "graph-foundation",
        "A-equivariant",
        "analytic-Chebyshev",
        "exact",
    )
    macro = {}
    for model in model_order:
        selected = [row for row in task_summary if row["model"] == model]
        macro[model] = {
            "ap": np.mean([row["average_precision_mean"] for row in selected]),
            "residual": np.mean([row["relative_residual_mean"] for row in selected]),
        }
    commutator = np.mean([row["commutator_mean"] for row in diagnostic_summary])
    oracle_residual = np.mean(
        [row["oracle_graph_residual_mean"] for row in diagnostic_summary]
    )
    epsilon = np.mean(
        [row["equivariant_epsilon_mean"] for row in diagnostic_summary]
    )
    certified = np.mean(
        [row["equivariant_certified_mean"] for row in diagnostic_summary]
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    positions = np.arange(len(model_order))
    axes[0].bar(positions, [macro[name]["ap"] for name in model_order])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("macro average precision")
    axes[1].bar(positions, [macro[name]["residual"] for name in model_order])
    axes[1].set_yscale("log")
    axes[1].set_title("macro sampling residual")
    for axis in axes[:2]:
        axis.set_xticks(positions, model_order, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    labels = ("commutator", "oracle p(L) residual", "equivariant epsilon", "certified rate")
    axes[2].bar(labels, (commutator, oracle_residual, epsilon, certified))
    axes[2].tick_params(axis="x", rotation=28)
    axes[2].set_title("identifiability diagnostics")
    axes[2].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_results_dir = args.graph_results_dir.resolve()
    device = torch.device(args.device)
    cache = physics_cache(device)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if args.quick:
        seeds = seeds[:1]
    steps = min(args.steps, 100) if args.quick else args.steps
    eval_tasks = min(args.eval_tasks, 12) if args.quick else args.eval_tasks
    batch_size = 4 if args.quick else 12
    depth = 6 if args.quick else 8
    log_every = 20 if args.quick else 100
    start = time.perf_counter()
    training_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    parameter_counts: dict[str, int] = {}

    for seed in seeds:
        graph_small = load_graph_model("small", seed, graph_results_dir, cache[8.0])
        graph_foundation = load_graph_model(
            "foundation", seed, graph_results_dir, cache[8.0]
        )
        equivariant, rows = train_equivariant(
            seed,
            cache,
            steps=steps,
            batch_size=batch_size,
            depth=depth,
            log_every=log_every,
        )
        training_rows.extend(rows)
        analytic = EquivariantChebyshevPCGLSMLoop(
            cache[8.0].kernel,
            cache[8.0].feature_kernel,
            cache[8.0].cfg.ridge_rel,
            depth=depth,
            polynomial_degree=12,
            moment_degree=8,
            controller_width=96,
            coefficient_mode="analytic",
        ).to(device)
        analytic.eval()
        parameter_counts = {
            "graph-small": count_parameters(graph_small),
            "graph-foundation": count_parameters(graph_foundation),
            "A-equivariant": count_parameters(equivariant),
            "analytic-Chebyshev": 0,
        }
        torch.save(
            {
                "model": equivariant.state_dict(),
                "seed": seed,
                "steps": steps,
                "depth": depth,
            },
            output_dir / f"equivariant_seed_{seed}.pt",
        )
        seed_tasks, seed_diagnostics = evaluate_seed(
            seed,
            cache,
            graph_small,
            graph_foundation,
            equivariant,
            analytic,
            eval_tasks=eval_tasks,
        )
        task_rows.extend(seed_tasks)
        diagnostic_rows.extend(seed_diagnostics)

    task_summary = aggregate(
        task_rows,
        ("average_precision", "auc", "relative_residual"),
    )
    diagnostic_summary = aggregate(
        diagnostic_rows,
        (
            "commutator",
            "oracle_graph_residual",
            "oracle_graph_condition",
            "equivariant_epsilon",
            "equivariant_actual_condition",
            "equivariant_certified",
            "analytic_epsilon",
            "analytic_actual_condition",
            "analytic_certified",
        ),
        keys=("scenario", "category"),
    )
    write_rows(output_dir / "training.csv", training_rows)
    write_rows(output_dir / "tasks.csv", task_rows)
    write_rows(output_dir / "diagnostics.csv", diagnostic_rows)
    write_rows(output_dir / "summary_tasks.csv", task_summary)
    write_rows(output_dir / "summary_diagnostics.csv", diagnostic_summary)
    protocol = {
        "description": "identifiable A-equivariant Chebyshev factor versus p(L_Gamma)",
        "seeds": list(seeds),
        "steps": steps,
        "outer_pcg_depth": depth,
        "chebyshev_degree": 12,
        "identification_witnesses": 8,
        "practical_certificate_threshold": 0.95,
        "eval_tasks_per_seed_and_scenario": eval_tasks,
        "parameter_counts": parameter_counts,
        "elapsed_seconds": time.perf_counter() - start,
    }
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    plot_results(
        output_dir / "preconditioner_identifiability.png",
        task_summary,
        diagnostic_summary,
    )
    print(f"completed in {protocol['elapsed_seconds']:.1f}s: {output_dir}")


if __name__ == "__main__":
    main()
