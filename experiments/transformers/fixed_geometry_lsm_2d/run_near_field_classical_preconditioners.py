#!/usr/bin/env python3
"""Matched-depth classical-PCG controls for the near-field scaling study."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor

from .foundation_uq import posterior_covariance, posterior_score_moments
from .near_field_lsm import (
    PosteriorMomentLSMLoop,
    build_near_field_system,
    near_field_score,
)
from .run_near_field_depth_scaling import synchronize
from .run_near_field_scaling import (
    EvaluationBatch,
    append_rows,
    build_physics_cache,
    comma_ints,
    existing_keys,
    make_evaluation_cache,
    numerical_metrics,
)


METHODS = (
    "optimized-CG",
    "Jacobi-PCG",
    "block-Jacobi-PCG",
    "angular-Jacobi-PCG",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--contexts", default="8,12,16,24,32,48")
    parser.add_argument("--depths", default="32,48,64,96,128")
    parser.add_argument("--eval-tasks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--runtime-repeats", type=int, default=5)
    parser.add_argument("--refresh-runtime", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def inverse_sqrt(matrix: Tensor) -> Tensor:
    """Hermitian inverse square root for a batch of small SPD matrices."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return (
        eigenvectors * eigenvalues.clamp_min(1.0e-8).rsqrt().unsqueeze(-2)
    ) @ eigenvectors.mH


def classical_factor(
    hessian: Tensor,
    receiver_feature: Tensor,
    method: str,
    *,
    block_size: int,
) -> Tensor:
    """Construct a task-wise SPD factor without learned parameters."""
    batch_size, context_size, _ = hessian.shape
    if method == "optimized-CG":
        identity = torch.eye(
            context_size,
            device=hessian.device,
            dtype=hessian.dtype,
        )
        return identity.unsqueeze(0).expand(batch_size, -1, -1)
    if method == "Jacobi-PCG":
        diagonal = hessian.diagonal(dim1=-2, dim2=-1).real.clamp_min(1.0e-8)
        return torch.diag_embed(diagonal.rsqrt()).to(hessian.dtype)
    if method == "block-Jacobi-PCG":
        factor = torch.zeros_like(hessian)
        for start in range(0, context_size, block_size):
            stop = min(context_size, start + block_size)
            factor[:, start:stop, start:stop] = inverse_sqrt(
                hessian[:, start:stop, start:stop]
            )
        return factor
    if method == "angular-Jacobi-PCG":
        _, eigenvectors_real = torch.linalg.eigh(receiver_feature.real)
        eigenvectors = eigenvectors_real.to(hessian.dtype)
        rotated = eigenvectors.mH.unsqueeze(0) @ hessian @ eigenvectors.unsqueeze(0)
        diagonal = rotated.diagonal(dim1=-2, dim2=-1).real.clamp_min(1.0e-8)
        modal = torch.diag_embed(diagonal.rsqrt()).to(hessian.dtype)
        return eigenvectors.unsqueeze(0) @ modal @ eigenvectors.mH.unsqueeze(0)
    raise ValueError(f"unknown classical preconditioner: {method}")


@torch.no_grad()
def solve_classical_pcg(
    batch: EvaluationBatch,
    method: str,
    *,
    depth: int,
    block_size: int,
    condition_diagnostics: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    system = build_near_field_system(batch.near_field, batch.kernel, batch.probe)
    factor = classical_factor(
        system["hessian"],
        batch.feature,
        method,
        block_size=block_size,
    )
    transformed = factor.mH @ system["hessian"] @ factor
    row_bound = transformed.abs().sum(dim=-1).amax(dim=-1).clamp_min(1.0)
    operator = transformed / row_bound[:, None, None]
    mean_rhs = factor.mH @ batch.probe / row_bound[:, None, None]
    covariance_rhs_unscaled = batch.near_field @ system["kernel"]
    covariance_rhs = (
        factor.mH @ covariance_rhs_unscaled / row_bound[:, None, None]
    )
    n_mean_rhs = mean_rhs.shape[-1]
    joint_rhs = torch.cat([mean_rhs, covariance_rhs], dim=-1)
    ones = torch.ones(batch.near_field.shape[0], device=operator.device)
    joint_iterate, relative, _, alpha, beta = PosteriorMomentLSMLoop._iterate(
        operator,
        joint_rhs,
        ones,
        ones,
        depth,
        method="pcg",
        return_history=False,
    )
    mean_iterate = joint_iterate[..., :n_mean_rhs]
    covariance_iterate = joint_iterate[..., n_mean_rhs:]
    q_mean = factor @ mean_iterate
    q_covariance = factor @ covariance_iterate
    _, mean_coefficients = near_field_score(
        batch.near_field,
        batch.kernel,
        q_mean,
    )
    covariance = posterior_covariance(
        batch.near_field,
        batch.kernel,
        q_covariance,
    )
    score_mean, score_std = posterior_score_moments(
        mean_coefficients,
        covariance,
        batch.kernel,
    )
    mean_residual = mean_rhs - operator @ mean_iterate
    covariance_residual = covariance_rhs - operator @ covariance_iterate
    mean_relative = torch.sqrt(
        mean_residual.abs().square().sum(dim=(1, 2))
        / mean_rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
    )
    covariance_relative = torch.sqrt(
        covariance_residual.abs().square().sum(dim=(1, 2))
        / covariance_rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
    )
    original_mean_residual = batch.probe - system["hessian"] @ q_mean
    original_covariance_residual = (
        covariance_rhs_unscaled - system["hessian"] @ q_covariance
    )
    original_mean_relative = torch.sqrt(
        original_mean_residual.abs().square().sum(dim=(1, 2))
        / batch.probe.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
    )
    original_covariance_relative = torch.sqrt(
        original_covariance_residual.abs().square().sum(dim=(1, 2))
        / covariance_rhs_unscaled.abs()
        .square()
        .sum(dim=(1, 2))
        .clamp_min(1.0e-12)
    )
    info = {
        "score_mean": score_mean,
        "score_std": score_std,
        "posterior_covariance": covariance,
        "mean_relative_residual": mean_relative,
        "covariance_relative_residual": covariance_relative,
        "transformed_mean_relative_residual": mean_relative,
        "transformed_covariance_relative_residual": covariance_relative,
        "original_mean_relative_residual": original_mean_relative,
        "original_covariance_relative_residual": original_covariance_relative,
        "relative_residual": relative,
        "alpha": alpha,
        "beta": beta,
    }
    if condition_diagnostics:
        raw_eigenvalues = torch.linalg.eigvalsh(system["hessian"])
        transformed_eigenvalues = torch.linalg.eigvalsh(transformed)
        raw_condition = raw_eigenvalues.amax(dim=-1) / raw_eigenvalues.amin(
            dim=-1
        ).clamp_min(1.0e-8)
        transformed_condition = transformed_eigenvalues.amax(
            dim=-1
        ) / transformed_eigenvalues.amin(dim=-1).clamp_min(1.0e-8)
        info.update(
            {
                "raw_condition": raw_condition,
                "transformed_condition": transformed_condition,
                "condition_reduction": raw_condition / transformed_condition,
            }
        )
    return score_mean, info


@torch.no_grad()
def timed_solve(
    batch: EvaluationBatch,
    method: str,
    *,
    depth: int,
    block_size: int,
    repeats: int,
) -> float:
    for _ in range(2):
        solve_classical_pcg(
            batch,
            method,
            depth=depth,
            block_size=block_size,
            condition_diagnostics=False,
        )
    synchronize(batch.near_field.device)
    started = time.perf_counter()
    for _ in range(repeats):
        solve_classical_pcg(
            batch,
            method,
            depth=depth,
            block_size=block_size,
            condition_diagnostics=False,
        )
    synchronize(batch.near_field.device)
    return 1_000.0 * (time.perf_counter() - started) / repeats


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.eval_tasks < 1 or args.block_size < 1 or args.runtime_repeats < 1:
        raise ValueError("task, block and timing counts must be positive")
    device = torch.device(args.device)
    seeds = comma_ints(args.seeds)
    contexts = comma_ints(args.contexts)
    depths = comma_ints(args.depths)
    result_path = args.results_dir / "classical_preconditioners.csv"
    runtime_path = args.results_dir / "classical_preconditioner_runtime.csv"
    if not args.resume:
        result_path.unlink(missing_ok=True)
        runtime_path.unlink(missing_ok=True)
    elif args.refresh_runtime:
        runtime_path.unlink(missing_ok=True)
    result_keys = existing_keys(
        result_path,
        ("seed", "method", "depth", "context_size", "scenario", "task"),
    )
    runtime_keys = existing_keys(
        runtime_path,
        ("seed", "method", "depth", "context_size"),
    )
    physics = build_physics_cache(contexts, device)
    for seed in seeds:
        evaluation = make_evaluation_cache(seed, contexts, physics, args.eval_tasks)
        runtime_batches = {
            context: evaluation[(context, "ID four obstacles")]
            for context in contexts
        }
        for depth in depths:
            for method in METHODS:
                rows: list[dict[str, object]] = []
                for (context, scenario_name), batch in evaluation.items():
                    score, info = solve_classical_pcg(
                        batch,
                        method,
                        depth=depth,
                        block_size=args.block_size,
                    )
                    metrics = numerical_metrics(score, info, batch)
                    for task, values in enumerate(metrics):
                        row = {
                            "seed": seed,
                            "method": method,
                            "depth": depth,
                            "context_size": context,
                            "context_measurements": context * context,
                            "scenario": scenario_name,
                            "regime": batch.scenario.regime,
                            "task": task,
                            "block_size": args.block_size,
                            "raw_condition": float(info["raw_condition"][task]),
                            "transformed_condition": float(
                                info["transformed_condition"][task]
                            ),
                            "condition_reduction": float(
                                info["condition_reduction"][task]
                            ),
                            **values,
                        }
                        key = tuple(
                            str(row[column])
                            for column in (
                                "seed",
                                "method",
                                "depth",
                                "context_size",
                                "scenario",
                                "task",
                            )
                        )
                        if key not in result_keys:
                            rows.append(row)
                            result_keys.add(key)
                append_rows(result_path, rows)

                runtime_rows = []
                for context, batch in runtime_batches.items():
                    key = tuple(str(value) for value in (seed, method, depth, context))
                    if key in runtime_keys:
                        continue
                    runtime_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "depth": depth,
                            "context_size": context,
                            "context_measurements": context * context,
                            "batch_size": args.eval_tasks,
                            "block_size": args.block_size,
                            "inference_ms": timed_solve(
                                batch,
                                method,
                                depth=depth,
                                block_size=args.block_size,
                                repeats=args.runtime_repeats,
                            ),
                        }
                    )
                    runtime_keys.add(key)
                append_rows(runtime_path, runtime_rows)
                print(
                    f"seed={seed} depth={depth:3d} {method}: {len(rows):4d} rows",
                    flush=True,
                )


if __name__ == "__main__":
    main()
