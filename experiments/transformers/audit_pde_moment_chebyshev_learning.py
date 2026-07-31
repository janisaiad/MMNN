#!/usr/bin/env python3
"""Audit prompt-conditioned spectral-measure Chebyshev on PDE inverse tasks.

The neural controller sees only seven prompt invariants and predicts cluster
nodes, masses, and a positive expansion of a Ritz scale.  A small weighted
Gram solve constructs the solution-polynomial coefficients exactly, and a
fixed Clenshaw loop applies that polynomial.  Full spectra are used only as a
pretraining loss and as an evaluation oracle; they are absent at inference.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from .constructive_weakform_richardson_transformer import (
        TaskConfig,
        sample_weak_batch,
    )
    from .exact_loop_transformer_decoder import ExactLoopTransformerDecoder
    from .first_principles_decoder_cells import (
        apply_fixed_preconditioner,
        materialize_preconditioner,
        risk_optimal_solution_chebyshev_coefficients,
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
        shifted_chebyshev_basis,
    )
    from .predict_pde_law_hyperparameters import (
        chebyshev_task_risk,
        task_risk,
    )
except ImportError:
    from constructive_weakform_richardson_transformer import (
        TaskConfig,
        sample_weak_batch,
    )
    from exact_loop_transformer_decoder import ExactLoopTransformerDecoder
    from first_principles_decoder_cells import (
        apply_fixed_preconditioner,
        materialize_preconditioner,
        risk_optimal_solution_chebyshev_coefficients,
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
        shifted_chebyshev_basis,
    )
    from predict_pde_law_hyperparameters import (
        chebyshev_task_risk,
        task_risk,
    )

Tensor = torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a nonempty comma-separated list")
    return values


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in parse_values(value)]


def make_config(args: argparse.Namespace, design: str) -> TaskConfig:
    return TaskConfig(
        K=args.dimension,
        prompt_len=args.prompt_length,
        prior_var=args.prior_variance,
        noise_var=args.noise_variance,
        design=design,
        dtype=args.dtype,
        pde_state_dim=args.state_dimension,
    )


def make_decoder(
    args: argparse.Namespace,
    device: torch.device,
) -> ExactLoopTransformerDecoder:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    model = ExactLoopTransformerDecoder(
        dimension=args.dimension,
        depth=args.depth,
        head_dimension=args.head_dimension,
        slots=args.slots,
        controller="moment_chebyshev",
        spectral_lmax_bound=args.spectral_lmax_bound,
        spectral_measure_clusters=args.spectral_clusters,
        spectral_measure_hidden_dimension=args.spectral_hidden_dimension,
        moment_gram_regularization=args.gram_regularization,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=args.refinements,
    ).to(device=device, dtype=dtype)
    for parameter in model.preconditioner_head.parameters():
        parameter.requires_grad_(False)
    return model


def scaled_prompt(batch, cfg: TaskConfig) -> tuple[Tensor, Tensor, float]:
    scale = math.sqrt(1.0 / cfg.noise_var)
    return batch.G * scale, batch.b * scale, 1.0 / cfg.prior_var


def spectral_training_data(
    model: ExactLoopTransformerDecoder,
    equations: Tensor,
    normal: Tensor,
    target: Tensor,
    ridge: float,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Return prompt invariants, exact effective spectrum, and energy masses."""

    with torch.no_grad():
        preconditioner, info = model.preconditioner_head(equations, ridge)
        dense = materialize_preconditioner(preconditioner)
        factor = torch.linalg.cholesky(dense)
        effective = factor.transpose(-1, -2) @ normal @ factor
        effective = 0.5 * (effective + effective.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(effective)
        transformed_target = torch.linalg.solve_triangular(
            factor,
            target.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        spectral_target = torch.einsum(
            "bji,bj->bi",
            eigenvectors,
            transformed_target,
        )
        energy = eigenvalues * spectral_target.square()
        weights = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return info, eigenvalues, weights


def predicted_spectral_risk(
    model: ExactLoopTransformerDecoder,
    info: dict[str, Tensor],
    eigenvalues: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert model.measure_head is not None
    nodes, masses, upper = model.measure_head(
        info["interval_features"],
        info.get(
            "projected_effective_target",
            info["projected_eigenvalues"][:, 0],
        ),
        info["certified_effective_lmax"],
    )
    coefficients = risk_optimal_solution_chebyshev_coefficients(
        nodes,
        masses,
        model.depth,
        upper,
        model.moment_gram_regularization,
    )
    basis = shifted_chebyshev_basis(eigenvalues, model.depth, upper)
    residual = 1.0 - eigenvalues * torch.einsum(
        "bkd,bd->bk",
        basis,
        coefficients,
    )
    risk = (weights * residual.square()).sum(dim=-1)
    return risk, nodes, masses, upper, coefficients


def train_measure_head(
    args: argparse.Namespace,
    cfg: TaskConfig,
    design: str,
    seed: int,
    device: torch.device,
) -> tuple[ExactLoopTransformerDecoder, dict[str, Tensor], list[dict]]:
    set_seed(seed)
    model = make_decoder(args, device)
    if args.preconditioner_checkpoint_dir is not None:
        checkpoint_path = (
            args.preconditioner_checkpoint_dir
            / f"model_r{args.refinements}_seed{seed}.pt"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        head_state = {
            name.removeprefix("preconditioner_head."): value
            for name, value in checkpoint["model"].items()
            if name.startswith("preconditioner_head.")
        }
        model.preconditioner_head.load_state_dict(head_state)
    initial = copy.deepcopy(model.state_dict())
    assert model.measure_head is not None
    optimizer = torch.optim.AdamW(
        model.measure_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    set_seed(seed + 50_000)
    validation_batch = sample_weak_batch(args.validation_tasks, cfg, device)
    validation_equations, _, validation_ridge = scaled_prompt(
        validation_batch,
        cfg,
    )
    validation_info, validation_eigenvalues, validation_weights = (
        spectral_training_data(
            model,
            validation_equations,
            validation_batch.H,
            validation_batch.beta_post,
            validation_ridge,
        )
    )
    validation_info = {
        key: validation_info[key]
        for key in [
            "interval_features",
            "projected_eigenvalues",
            "projected_effective_target",
            "certified_effective_lmax",
        ]
    }
    del validation_batch, validation_equations
    set_seed(seed)
    best_validation = float("inf")
    best_step = 0
    best_measure_state = copy.deepcopy(model.measure_head.state_dict())
    tail_count = max(1, math.ceil(args.cvar_fraction * args.batch_size))
    history = []
    for step in range(1, args.training_steps + 1):
        batch = sample_weak_batch(args.batch_size, cfg, device)
        equations, _, ridge = scaled_prompt(batch, cfg)
        info, eigenvalues, weights = spectral_training_data(
            model,
            equations,
            batch.H,
            batch.beta_post,
            ridge,
        )
        risk, nodes, masses, upper, _ = predicted_spectral_risk(
            model,
            info,
            eigenvalues,
            weights,
        )
        cluster_indices = torch.linspace(
            0,
            eigenvalues.shape[-1] - 1,
            args.spectral_clusters,
            device=device,
        ).round().long()
        target_nodes = eigenvalues[:, cluster_indices]
        target_masses = weights[:, cluster_indices]
        target_masses = target_masses / target_masses.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-30)
        node_loss = (
            target_masses
            * (torch.log(nodes) - torch.log(target_nodes)).square()
        ).sum(dim=-1).mean()
        mass_loss = -(
            target_masses * torch.log(masses.clamp_min(1e-30))
        ).sum(dim=-1).mean()
        desired_upper = args.coverage_margin * eigenvalues[:, -1]
        log_gap = torch.log(desired_upper) - torch.log(upper)
        under_coverage = torch.relu(log_gap)
        over_coverage = torch.relu(-log_gap)
        robust_risk = torch.log1p(risk)
        cvar = robust_risk.topk(tail_count).values.mean()
        loss = (
            robust_risk.mean()
            + args.cvar_weight * cvar
            + args.under_coverage_weight * under_coverage.square().mean()
            + args.over_coverage_weight * over_coverage.square().mean()
            + args.node_supervision_weight * node_loss
            + args.mass_supervision_weight * mass_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.measure_head.parameters(),
            args.gradient_clip,
        )
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.training_steps:
            with torch.no_grad():
                validation_risk, _, _, validation_upper, _ = (
                    predicted_spectral_risk(
                        model,
                        validation_info,
                        validation_eigenvalues,
                        validation_weights,
                    )
                )
                validation_under = torch.relu(
                    torch.log(validation_eigenvalues[:, -1])
                    - torch.log(validation_upper)
                )
                validation_score = (
                    validation_risk.mean()
                    + 100.0 * validation_under.square().mean()
                ).item()
            if validation_score < best_validation:
                best_validation = validation_score
                best_step = step
                best_measure_state = copy.deepcopy(
                    model.measure_head.state_dict()
                )
            row = {
                "design": design,
                "seed": seed,
                "step": step,
                "loss": loss.item(),
                "spectral_risk_mean": risk.mean().item(),
                "spectral_risk_q99": torch.quantile(risk, 0.99).item(),
                "coverage_rate": (upper >= eigenvalues[:, -1]).float().mean().item(),
                "upper_ratio_mean": (
                    upper / info["projected_eigenvalues"][:, 0]
                ).mean().item(),
                "node_loss": node_loss.item(),
                "mass_loss": mass_loss.item(),
                "validation_spectral_risk": validation_risk.mean().item(),
                "validation_coverage_rate": (
                    validation_upper >= validation_eigenvalues[:, -1]
                ).float().mean().item(),
                "best_step": best_step,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    model.measure_head.load_state_dict(best_measure_state)
    return model, initial, history


def energy_error(x: Tensor, target: Tensor, normal: Tensor) -> Tensor:
    error = x - target
    numerator = torch.einsum("bi,bij,bj->b", error, normal, error)
    denominator = torch.einsum(
        "bi,bij,bj->b",
        target,
        normal,
        target,
    ).clamp_min(1e-30)
    return numerator / denominator


def make_hvp(equations: Tensor, ridge: float):
    def hvp(vector: Tensor) -> Tensor:
        scores = torch.einsum("bmk,bk->bm", equations, vector)
        return torch.einsum("bmk,bm->bk", equations, scores) + ridge * vector

    return hvp


def method_cost(
    name: str,
    depth: int,
    setup_rounds: int,
    slots: int,
) -> tuple[int, int, int]:
    if name.startswith("identity_"):
        solver = depth + setup_rounds * slots if name.endswith("equal_work") else depth
        return solver, 0, solver
    return depth, setup_rounds, depth + setup_rounds * slots


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    cfg: TaskConfig,
    design: str,
    seed: int,
    model: ExactLoopTransformerDecoder,
    initial_state: dict[str, Tensor],
    device: torch.device,
) -> tuple[list[dict], dict]:
    initial = make_decoder(args, device)
    initial.load_state_dict(initial_state)
    model.eval()
    initial.eval()
    totals = defaultdict(lambda: defaultdict(float))
    setup_rounds = args.refinements + 1
    remaining = args.evaluation_tasks
    theory_sum = 0.0
    theory_count = 0
    covered = 0
    fallbacks = 0
    pcg_dominance_violations = 0
    set_seed(seed + 100_000)
    while remaining:
        size = min(args.evaluation_batch_size, remaining)
        batch = sample_weak_batch(size, cfg, device)
        equations, observations, ridge = scaled_prompt(batch, cfg)
        geometry = model.build_prompt_geometry(equations, ridge)
        initial_geometry = initial.build_prompt_geometry(equations, ridge)
        learned, learned_info = model.solve_with_geometry(geometry, observations)
        initialized, _ = initial.solve_with_geometry(initial_geometry, observations)
        rhs = learned_info["rhs"]
        preconditioner = geometry.preconditioner
        hvp = make_hvp(equations, ridge)
        dense = materialize_preconditioner(preconditioner)
        factor = torch.linalg.cholesky(dense)
        effective = factor.transpose(-1, -2) @ batch.H @ factor
        eigenvalues, eigenvectors = torch.linalg.eigh(
            0.5 * (effective + effective.transpose(-1, -2))
        )
        lower, upper = eigenvalues[:, 0], eigenvalues[:, -1]
        transformed_target = torch.linalg.solve_triangular(
            factor,
            batch.beta_post.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        spectral_target = torch.einsum(
            "bji,bj->bi",
            eigenvectors,
            transformed_target,
        )
        energy = eigenvalues * spectral_target.square()
        weights = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-30)

        sqrt_lower, sqrt_upper = torch.sqrt(lower), torch.sqrt(upper)
        hb_step = 4.0 / (sqrt_upper + sqrt_lower).square()
        hb_momentum = (
            (sqrt_upper - sqrt_lower) / (sqrt_upper + sqrt_lower)
        ).square()
        richardson_step = 2.0 / (lower + upper)
        same_pcg = run_pcg_state_machine(
            hvp,
            rhs,
            preconditioner,
            args.depth,
        )[0]
        identity = torch.eye(
            args.dimension,
            device=device,
            dtype=rhs.dtype,
        ).expand(size, -1, -1)
        equal_depth = args.depth + setup_rounds * args.slots
        methods = {
            "learned_moment_chebyshev": learned,
            "initial_moment_chebyshev": initialized,
            "oracle_interval_richardson": run_heavy_ball_state_machine(
                hvp,
                rhs,
                preconditioner,
                args.depth,
                richardson_step,
                torch.zeros_like(richardson_step),
            )[0],
            "oracle_interval_hb": run_heavy_ball_state_machine(
                hvp,
                rhs,
                preconditioner,
                args.depth,
                hb_step,
                hb_momentum,
            )[0],
            "oracle_interval_chebyshev": run_chebyshev_state_machine(
                hvp,
                rhs,
                preconditioner,
                args.depth,
                lower,
                upper,
            )[0],
            "same_preconditioner_pcg": same_pcg,
            "identity_pcg": run_pcg_state_machine(
                hvp,
                rhs,
                identity,
                args.depth,
            )[0],
            "identity_pcg_equal_work": run_pcg_state_machine(
                hvp,
                rhs,
                identity,
                equal_depth,
            )[0],
        }
        final_residual = rhs - hvp(learned)
        preconditioned_residual = apply_fixed_preconditioner(
            preconditioner,
            final_residual,
        )
        preconditioned_rhs = apply_fixed_preconditioner(preconditioner, rhs)
        residual_ratio = torch.einsum(
            "bi,bi->b",
            final_residual,
            preconditioned_residual,
        ) / torch.einsum(
            "bi,bi->b",
            rhs,
            preconditioned_rhs,
        ).clamp_min(1e-30)
        fallback = residual_ratio > args.guard_residual_ratio
        methods["learned_moment_guarded_pcg"] = torch.where(
            fallback[:, None],
            same_pcg,
            learned,
        )
        fallbacks += fallback.sum().item()

        basis = shifted_chebyshev_basis(
            eigenvalues,
            args.depth,
            learned_info["spectral_upper"],
        )
        polynomial_residual = 1.0 - eigenvalues * torch.einsum(
            "bkd,bd->bk",
            basis,
            learned_info["moment_solution_coefficients"],
        )
        theory = (weights * polynomial_residual.square()).sum(dim=-1)
        theory_sum += theory.sum().item()
        theory_count += size
        covered += (
            learned_info["spectral_upper"] >= upper
        ).sum().item()

        clean_query = torch.einsum("bi,bi->b", batch.gq, batch.beta_true)
        posterior_query = torch.einsum("bi,bi->b", batch.gq, batch.beta_post)
        per_task_energy = {}
        for name, prediction in methods.items():
            relative = energy_error(prediction, batch.beta_post, batch.H)
            per_task_energy[name] = relative
            prediction_query = torch.einsum("bi,bi->b", batch.gq, prediction)
            values = totals[name]
            values["count"] += size
            values["energy"] += relative.sum().item()
            values["query_num"] += (
                prediction_query - posterior_query
            ).square().sum().item()
            values["query_den"] += posterior_query.square().sum().item()
            values["statistical_query"] += (
                prediction_query - clean_query
            ).square().sum().item()
            values["coefficient"] += (
                prediction - batch.beta_post
            ).square().mean(dim=-1).sum().item()
        pcg_dominance_violations += (
            per_task_energy["same_preconditioner_pcg"]
            > per_task_energy["learned_moment_chebyshev"] + 1e-5
        ).sum().item()

        richardson_theory = task_risk(
            eigenvalues.double(),
            weights.double(),
            args.depth,
            richardson_step.double()[:, None],
            torch.zeros_like(richardson_step).double()[:, None],
        )
        hb_theory = task_risk(
            eigenvalues.double(),
            weights.double(),
            args.depth,
            hb_step.double()[:, None],
            hb_momentum.double()[:, None],
        )
        chebyshev_theory = chebyshev_task_risk(
            eigenvalues.double(),
            weights.double(),
            args.depth,
            lower.double(),
            upper.double(),
        )
        totals["oracle_interval_richardson"]["predicted"] += (
            richardson_theory.sum().item()
        )
        totals["oracle_interval_hb"]["predicted"] += hb_theory.sum().item()
        totals["oracle_interval_chebyshev"]["predicted"] += (
            chebyshev_theory.sum().item()
        )
        totals["learned_moment_chebyshev"]["predicted"] += theory.sum().item()
        remaining -= size

    rows = []
    for name, values in totals.items():
        count = values["count"]
        solver_hvps, block_rounds, scalar_equivalent = method_cost(
            name,
            args.depth,
            setup_rounds,
            args.slots,
        )
        rows.append(
            {
                "design": design,
                "seed": seed,
                "method": name,
                "solver_hvps": solver_hvps,
                "setup_block_hvp_rounds": block_rounds,
                "total_scalar_hvp_equivalent": scalar_equivalent,
                "h_relative": values["energy"] / count,
                "predicted_h_relative": (
                    values["predicted"] / count
                    if values["predicted"]
                    else float("nan")
                ),
                "query_relative": values["query_num"]
                / max(values["query_den"], 1e-30),
                "coefficient_mse": values["coefficient"] / count,
                "statistical_query_mse": values["statistical_query"] / count,
            }
        )
    diagnostics = {
        "design": design,
        "seed": seed,
        "learned_theory_h_relative": theory_sum / theory_count,
        "spectral_upper_coverage_rate": covered / theory_count,
        "guard_fallback_rate": fallbacks / theory_count,
        "pcg_dominance_violation_rate": pcg_dominance_violations / theory_count,
        "setup_block_hvp_rounds": setup_rounds,
        "setup_scalar_hvp_equivalent": setup_rounds * args.slots,
    }
    return rows, diagnostics


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["design"], row["method"])].append(row)
    output = []
    metrics = [
        "h_relative",
        "predicted_h_relative",
        "query_relative",
        "coefficient_mse",
        "statistical_query_mse",
    ]
    for (design, method), group in sorted(groups.items()):
        result = {
            "design": design,
            "method": method,
            "seeds": len(group),
            "solver_hvps": group[0]["solver_hvps"],
            "setup_block_hvp_rounds": group[0]["setup_block_hvp_rounds"],
            "total_scalar_hvp_equivalent": group[0][
                "total_scalar_hvp_equivalent"
            ],
        }
        for metric in metrics:
            values = torch.tensor([row[metric] for row in group])
            finite = values[torch.isfinite(values)]
            result[f"{metric}_mean"] = (
                finite.mean().item() if finite.numel() else float("nan")
            )
            result[f"{metric}_std"] = (
                finite.std(unbiased=False).item() if finite.numel() else float("nan")
            )
        output.append(result)
    return output


def plot(rows: list[dict], outdir: Path) -> None:
    if plt is None:
        return
    order = [
        "learned_moment_chebyshev",
        "learned_moment_guarded_pcg",
        "oracle_interval_richardson",
        "oracle_interval_hb",
        "oracle_interval_chebyshev",
        "same_preconditioner_pcg",
        "identity_pcg",
        "identity_pcg_equal_work",
    ]
    designs = sorted({row["design"] for row in rows})
    figure, axes = plt.subplots(
        len(designs),
        2,
        figsize=(13, 4.5 * len(designs)),
        squeeze=False,
    )
    for index, design in enumerate(designs):
        lookup = {
            row["method"]: row for row in rows if row["design"] == design
        }
        names = [name for name in order if name in lookup]
        for column, metric in enumerate(["h_relative", "query_relative"]):
            axis = axes[index, column]
            axis.bar(
                range(len(names)),
                [lookup[name][f"{metric}_mean"] for name in names],
                yerr=[lookup[name][f"{metric}_std"] for name in names],
                capsize=3,
            )
            axis.set_yscale("log")
            axis.set_xticks(
                range(len(names)),
                [name.replace("_", "\n") for name in names],
                fontsize=7,
            )
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(f"{design}: {metric.replace('_', ' ')}")
    figure.tight_layout()
    figure.savefig(outdir / "moment_chebyshev_pde_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--designs",
        default="pde_elliptic_correlated,pde_elliptic",
    )
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--state-dimension", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=0.02)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--head-dimension", type=int, default=8)
    parser.add_argument("--spectral-lmax-bound", type=float, default=4.0)
    parser.add_argument("--spectral-clusters", type=int, default=32)
    parser.add_argument("--spectral-hidden-dimension", type=int, default=32)
    parser.add_argument("--gram-regularization", type=float, default=1e-5)
    parser.add_argument("--training-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-tasks", type=int, default=4096)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--cvar-fraction", type=float, default=0.05)
    parser.add_argument("--cvar-weight", type=float, default=0.2)
    parser.add_argument("--validation-tasks", type=int, default=4096)
    parser.add_argument("--coverage-margin", type=float, default=12.0)
    parser.add_argument("--under-coverage-weight", type=float, default=100.0)
    parser.add_argument("--over-coverage-weight", type=float, default=0.001)
    parser.add_argument("--node-supervision-weight", type=float, default=1.0)
    parser.add_argument("--mass-supervision-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--guard-residual-ratio", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--preconditioner-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "optional directory containing model_r{refinements}_seed{seed}.pt "
            "from the paired matrix-free HB training audit"
        ),
    )
    args = parser.parse_args()

    designs = parse_values(args.designs)
    allowed = {"pde_elliptic", "pde_elliptic_correlated"}
    if any(design not in allowed for design in designs):
        raise ValueError(f"designs must lie in {sorted(allowed)}")
    if args.slots >= args.dimension:
        raise ValueError("slots must be smaller than dimension")
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "config.json").open("w") as handle:
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    rows, diagnostics, history = [], [], []
    for design in designs:
        cfg = make_config(args, design)
        for seed in parse_ints(args.seeds):
            model, initial, run_history = train_measure_head(
                args,
                cfg,
                design,
                seed,
                device,
            )
            run_rows, run_diagnostics = evaluate(
                args,
                cfg,
                design,
                seed,
                model,
                initial,
                device,
            )
            rows.extend(run_rows)
            diagnostics.append(run_diagnostics)
            history.extend(run_history)
            torch.save(
                {
                    "model": model.state_dict(),
                    "initial": initial,
                    "args": vars(args),
                    "design": design,
                    "seed": seed,
                },
                outdir / f"model_{design}_seed{seed}.pt",
            )

    aggregate_rows = aggregate(rows)
    write_csv(outdir / "per_seed.csv", rows)
    write_csv(outdir / "aggregate.csv", aggregate_rows)
    write_csv(outdir / "diagnostics.csv", diagnostics)
    write_csv(outdir / "training.csv", history)
    plot(aggregate_rows, outdir)
    summary = {
        "architecture": (
            "The MLP predicts only prompt spectral nodes, masses, and an upper "
            "scale. Gram coefficient construction and Clenshaw are exact."
        ),
        "fairness": (
            "A rank-S block setup with r refinements uses r+1 block-HVP rounds "
            "or (r+1)S sequential scalar-HVP equivalents."
        ),
        "claim_boundary": (
            "PCG with the same preconditioner remains the instancewise Krylov "
            "baseline; learned gains can only come from amortized geometry, "
            "fixed-depth task risk, or hardware-parallel block setup."
        ),
        "aggregate": aggregate_rows,
        "diagnostics": diagnostics,
    }
    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
