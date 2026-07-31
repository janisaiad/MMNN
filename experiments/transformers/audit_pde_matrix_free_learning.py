#!/usr/bin/env python3
"""Separate learned routing from hard-coded block-power on elliptic PDE tasks.

One softmax head and two shared Heavy--Ball scalars are trained end to end.
Evaluation crosses the resulting fixed prompt geometry with exact HB,
Chebyshev, and PCG cells and compares it with the initialized head, Jacobi,
an unpreconditioned equal-work PCG, and an exact top-Ritz oracle.
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
        materialize_preconditioner,
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
except ImportError:
    from constructive_weakform_richardson_transformer import (
        TaskConfig,
        sample_weak_batch,
    )
    from exact_loop_transformer_decoder import ExactLoopTransformerDecoder
    from first_principles_decoder_cells import (
        materialize_preconditioner,
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )

Tensor = torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("expected a nonempty integer list")
    return values


def task_config(args: argparse.Namespace) -> TaskConfig:
    return TaskConfig(
        K=args.dimension,
        prompt_len=args.prompt_length,
        prior_var=args.prior_variance,
        noise_var=args.noise_variance,
        design=args.design,
        dtype=args.dtype,
        pde_state_dim=args.state_dimension,
    )


def decoder(
    args: argparse.Namespace,
    refinements: int,
    device: torch.device,
) -> ExactLoopTransformerDecoder:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    return ExactLoopTransformerDecoder(
        dimension=args.dimension,
        depth=args.depth,
        head_dimension=args.head_dimension,
        slots=args.slots,
        controller=args.training_controller,
        spectral_lmax_bound=args.spectral_lmax_bound,
        step_init=args.step_init,
        momentum_init=args.momentum_init,
        spectral_krylov_steps=args.spectral_krylov_steps,
        complement_measure_hidden_dimension=args.complement_measure_hidden_dimension,
        moment_gram_regularization=args.gram_regularization,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=refinements,
    ).to(device=device, dtype=dtype)


def scaled_prompt(batch, cfg: TaskConfig) -> tuple[Tensor, Tensor, float]:
    scale = math.sqrt(1.0 / cfg.noise_var)
    return batch.G * scale, batch.b * scale, 1.0 / cfg.prior_var


def energy_error(x: Tensor, target: Tensor, normal: Tensor) -> Tensor:
    error = x - target
    numerator = torch.einsum("bi,bij,bj->b", error, normal, error)
    denominator = torch.einsum(
        "bi,bij,bj->b", target, normal, target
    ).clamp_min(1e-30)
    return numerator / denominator


def query_error(x: Tensor, target: Tensor, query: Tensor) -> Tensor:
    error = torch.einsum("bi,bi->b", query, x - target)
    reference = torch.einsum("bi,bi->b", query, target)
    return error.square().mean() / reference.square().mean().clamp_min(1e-30)


def train(
    args: argparse.Namespace,
    cfg: TaskConfig,
    refinements: int,
    seed: int,
    device: torch.device,
):
    set_seed(seed)
    model = decoder(args, refinements, device)
    if args.initial_head_checkpoint_dir is not None:
        checkpoint = torch.load(
            args.initial_head_checkpoint_dir
            / f"model_r{refinements}_seed{seed}.pt",
            map_location=device,
            weights_only=True,
        )
        head_state = {
            name.removeprefix("preconditioner_head."): value
            for name, value in checkpoint["model"].items()
            if name.startswith("preconditioner_head.")
        }
        model.preconditioner_head.load_state_dict(head_state)
    if args.freeze_preconditioner_head:
        for parameter in model.preconditioner_head.parameters():
            parameter.requires_grad_(False)
    initial = copy.deepcopy(model.state_dict())
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    tail = max(1, math.ceil(args.cvar_fraction * args.batch_size))
    history = []
    for step in range(1, args.training_steps + 1):
        batch = sample_weak_batch(args.batch_size, cfg, device)
        equations, observations, ridge = scaled_prompt(batch, cfg)
        prediction, _ = model(equations, observations, ridge)
        per_task = energy_error(prediction, batch.beta_post, batch.H)
        mean_energy = per_task.mean()
        cvar = per_task.topk(tail).values.mean()
        query = query_error(prediction, batch.beta_post, batch.gq)
        loss = mean_energy + args.cvar_weight * cvar + args.query_weight * query
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.training_steps:
            row = {
                "seed": seed,
                "refinements": refinements,
                "step": step,
                "loss": loss.item(),
                "energy": mean_energy.item(),
                "cvar": cvar.item(),
                "query": query.item(),
            }
            if args.training_controller == "heavy_ball":
                alpha, beta = model.heavy_ball_coefficients()
                row.update(
                    {
                        "hb_step": alpha.item(),
                        "hb_momentum": beta.item(),
                    }
                )
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return model, initial, history


def hvp_from_prompt(equations: Tensor, ridge: float):
    def hvp(vector: Tensor) -> Tensor:
        scores = torch.einsum("bmk,bk->bm", equations, vector)
        return torch.einsum("bmk,bm->bk", equations, scores) + ridge * vector

    return hvp


def oracle_top(normal: Tensor, slots: int) -> tuple[Tensor, Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(normal)
    directions = eigenvectors[:, :, -slots:]
    selected = eigenvalues[:, -slots:]
    multipliers = selected[:, :1] / selected
    batch, dimension, _ = normal.shape
    identity = torch.eye(
        dimension, device=normal.device, dtype=normal.dtype
    ).expand(batch, -1, -1)
    matrix = (
        identity
        - directions @ directions.transpose(-1, -2)
        + torch.einsum(
            "bki,bi,bli->bkl", directions, multipliers, directions
        )
    )
    return matrix, directions


def oracle_polynomials(
    hvp,
    rhs: Tensor,
    preconditioner,
    normal: Tensor,
    depth: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dense = materialize_preconditioner(preconditioner)
    dimension = dense.shape[-1]
    identity = torch.eye(
        dimension, device=dense.device, dtype=dense.dtype
    ).expand_as(dense)
    factor = torch.linalg.cholesky(dense + 1e-10 * identity)
    effective = factor.transpose(-1, -2) @ normal @ factor
    effective = 0.5 * (effective + effective.transpose(-1, -2))
    spectrum, eigenvectors = torch.linalg.eigh(effective)
    spectrum = spectrum.clamp_min(1e-12)
    lower, upper = spectrum[:, 0], spectrum[:, -1]
    roots = torch.sqrt(upper), torch.sqrt(lower)
    step = 4.0 / (roots[0] + roots[1]).square()
    momentum = ((roots[0] - roots[1]) / (roots[0] + roots[1])).square()
    hb = run_heavy_ball_state_machine(
        hvp, rhs, preconditioner, depth, step, momentum
    )[0]
    chebyshev = run_chebyshev_state_machine(
        hvp, rhs, preconditioner, depth, lower, upper
    )[0]
    normalized = (spectrum / upper[:, None]).double()
    powers = torch.arange(
        1,
        depth + 1,
        device=spectrum.device,
        dtype=torch.float64,
    )
    vandermonde = normalized.unsqueeze(-1).pow(powers)
    coefficients = torch.linalg.lstsq(
        vandermonde,
        -torch.ones(
            *normalized.shape,
            1,
            device=spectrum.device,
            dtype=torch.float64,
        ),
    ).solution.squeeze(-1)
    residual = 1.0 + torch.einsum(
        "bkl,bl->bk",
        vandermonde,
        coefficients,
    )
    transformed_rhs = torch.einsum(
        "bji,bj->bi",
        factor,
        rhs,
    )
    spectral_rhs = torch.einsum(
        "bji,bj->bi",
        eigenvectors,
        transformed_rhs,
    )
    spectral_exact = spectral_rhs / spectrum
    spectral_approximation = (
        (1.0 - residual).to(spectral_exact.dtype) * spectral_exact
    )
    moment_polynomial = torch.einsum(
        "bij,bjk,bk->bi",
        factor,
        eigenvectors,
        spectral_approximation,
    )
    return hb, chebyshev, moment_polynomial, spectrum


def affected(directions: Tensor, normal: Tensor) -> Tensor:
    return torch.linalg.qr(
        torch.cat([directions, normal @ directions], dim=-1),
        mode="reduced",
    ).Q


def projected_fraction(subspace: Tensor, vector: Tensor) -> Tensor:
    coordinates = torch.einsum("bks,bk->bs", subspace, vector)
    return coordinates.square().sum(-1) / vector.square().sum(-1).clamp_min(1e-30)


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    cfg: TaskConfig,
    trained: ExactLoopTransformerDecoder,
    initial_state: dict[str, Tensor],
    refinements: int,
    seed: int,
    device: torch.device,
):
    initial = decoder(args, refinements, device)
    initial.load_state_dict(initial_state)
    trained.eval()
    initial.eval()
    totals = defaultdict(lambda: defaultdict(float))
    overlaps = defaultdict(list)
    gains = []
    spectral_conditions = []
    spectral_effective_ranks = []
    near_ridge_counts = []
    set_seed(seed + 100_000)
    remaining = args.evaluation_tasks
    while remaining:
        size = min(args.evaluation_batch_size, remaining)
        batch = sample_weak_batch(size, cfg, device)
        equations, observations, ridge = scaled_prompt(batch, cfg)
        rhs = torch.einsum("bmk,bm->bk", equations, observations)
        hvp = hvp_from_prompt(equations, ridge)
        trained_pre, trained_info = trained.preconditioner_head(equations, ridge)
        initial_pre, initial_info = initial.preconditioner_head(equations, ridge)
        identity = torch.eye(
            args.dimension, device=device, dtype=rhs.dtype
        ).expand(size, -1, -1)
        jacobi = torch.diag_embed(
            torch.diagonal(batch.H, dim1=-2, dim2=-1).reciprocal()
        )
        oracle, oracle_directions = oracle_top(batch.H, args.slots)
        # The final projected Ritz matrix needs one additional block HVP
        # beyond the refinement steps.  Charge every block column when
        # comparing with sequential scalar-HVP work.
        extra_measure_rounds = (
            args.spectral_krylov_steps
            if args.training_controller
            in {
                "ritz_moment_chebyshev",
                "corrected_ritz_moment_chebyshev",
            }
            else 0
        )
        equal_depth = (
            args.depth
            + (refinements + 1 + extra_measure_rounds) * args.slots
        )
        methods = {
            "trained_head_pcg": run_pcg_state_machine(
                hvp, rhs, trained_pre, args.depth
            )[0],
            "initial_head_pcg": run_pcg_state_machine(
                hvp, rhs, initial_pre, args.depth
            )[0],
            "identity_pcg": run_pcg_state_machine(
                hvp, rhs, identity, args.depth
            )[0],
            "identity_pcg_equal_work": run_pcg_state_machine(
                hvp, rhs, identity, equal_depth
            )[0],
            "jacobi_pcg_equal_work": run_pcg_state_machine(
                hvp, rhs, jacobi, equal_depth
            )[0],
            "oracle_top_pcg": run_pcg_state_machine(
                hvp, rhs, oracle, args.depth
            )[0],
        }
        if args.training_controller == "heavy_ball":
            alpha, beta = trained.heavy_ball_coefficients()
            alpha0, beta0 = initial.heavy_ball_coefficients()
            methods.update(
                {
                    "trained_head_hb": run_heavy_ball_state_machine(
                        hvp, rhs, trained_pre, args.depth, alpha, beta
                    )[0],
                    "initial_head_hb": run_heavy_ball_state_machine(
                        hvp, rhs, initial_pre, args.depth, alpha0, beta0
                    )[0],
                }
            )
        else:
            trained_name = f"trained_head_{args.training_controller}"
            initial_name = f"initial_head_{args.training_controller}"
            trained_solution, trained_controller_info = trained(
                equations,
                observations,
                ridge,
            )
            initial_solution, initial_controller_info = initial(
                equations,
                observations,
                ridge,
            )
            methods.update(
                {
                    trained_name: trained_solution,
                    initial_name: initial_solution,
                }
            )
            if args.training_controller == "corrected_ritz_moment_chebyshev":
                for prefix, controller_info in [
                    ("trained", trained_controller_info),
                    ("initial", initial_controller_info),
                ]:
                    overlaps[f"{prefix}_complement_gate"].append(
                        controller_info["complement_energy_gate"]
                    )
                    overlaps[f"{prefix}_complement_balance"].append(
                        controller_info["complement_balance"]
                    )
                    overlaps[f"{prefix}_complement_spread_fraction"].append(
                        controller_info["complement_spread"]
                        / controller_info["spectral_upper"]
                    )
        (
            oracle_hb,
            oracle_chebyshev,
            oracle_moment_polynomial,
            effective_spectrum,
        ) = oracle_polynomials(
            hvp,
            rhs,
            trained_pre,
            batch.H,
            args.depth,
        )
        methods["trained_head_oracle_hb"] = oracle_hb
        methods["trained_head_oracle_chebyshev"] = oracle_chebyshev
        methods["oracle_prompt_esd_polynomial"] = oracle_moment_polynomial
        spectral_conditions.append(
            (effective_spectrum[:, -1] / effective_spectrum[:, 0]).cpu()
        )
        spectral_effective_ranks.append(
            (
                effective_spectrum.sum(-1).square()
                / effective_spectrum.square().sum(-1)
            ).cpu()
        )
        near_ridge_counts.append(
            (
                effective_spectrum
                <= 1.001 * effective_spectrum[:, :1]
            ).sum(-1).cpu()
        )
        clean_query = torch.einsum("bi,bi->b", batch.gq, batch.beta_true)
        target_query = torch.einsum("bi,bi->b", batch.gq, batch.beta_post)
        per_task = {}
        for name, prediction in methods.items():
            energy = energy_error(prediction, batch.beta_post, batch.H)
            per_task[name] = energy
            prediction_query = torch.einsum("bi,bi->b", batch.gq, prediction)
            values = totals[name]
            values["count"] += size
            values["energy"] += energy.sum().item()
            values["query_num"] += (
                prediction_query - target_query
            ).square().sum().item()
            values["query_den"] += target_query.square().sum().item()
            values["stat_query"] += (
                prediction_query - clean_query
            ).square().sum().item()
            values["coefficient"] += (
                prediction - batch.beta_post
            ).square().mean(-1).sum().item()
        trained_span = affected(trained_info["directions"], batch.H)
        initial_span = affected(initial_info["directions"], batch.H)
        for prefix, span in [("trained", trained_span), ("initial", initial_span)]:
            overlaps[f"{prefix}_query"].append(
                projected_fraction(span, batch.gq).cpu()
            )
            overlaps[f"{prefix}_posterior"].append(
                projected_fraction(span, batch.beta_post).cpu()
            )
        for prefix, directions in [
            ("trained", trained_info["directions"]),
            ("initial", initial_info["directions"]),
        ]:
            top = torch.linalg.matrix_norm(
                directions.transpose(-1, -2) @ oracle_directions,
                ord="fro",
                dim=(-2, -1),
            ).square() / args.slots
            overlaps[f"{prefix}_top"].append(top.cpu())
        gains.append(
            (
                torch.log10(
                    per_task["identity_pcg_equal_work"].clamp_min(1e-30)
                )
                - torch.log10(per_task["trained_head_pcg"].clamp_min(1e-30))
            ).cpu()
        )
        remaining -= size
    rows = []
    for name, values in totals.items():
        count = values["count"]
        rows.append(
            {
                "seed": seed,
                "refinements": refinements,
                "method": name,
                "depth": (
                    args.depth + (refinements + 1) * args.slots
                    if name.endswith("equal_work")
                    else args.depth
                ),
                "h_relative": values["energy"] / count,
                "query_relative": values["query_num"] / max(values["query_den"], 1e-30),
                "coefficient_mse": values["coefficient"] / count,
                "statistical_query_mse": values["stat_query"] / count,
            }
        )
    gain = torch.cat(gains)
    trained_query = torch.cat(overlaps["trained_query"])
    correlation = torch.corrcoef(torch.stack([gain, trained_query]))[0, 1]
    summary = {
        "seed": seed,
        "refinements": refinements,
        "mean_log10_equal_work_gain": gain.mean().item(),
        "gain_query_overlap_correlation": correlation.item(),
        "effective_condition_mean": torch.cat(
            spectral_conditions
        ).mean().item(),
        "spectral_effective_rank_mean": torch.cat(
            spectral_effective_ranks
        ).mean().item(),
        "near_ridge_eigenvalue_count_mean": torch.cat(
            near_ridge_counts
        ).double().mean().item(),
    }
    for name, chunks in overlaps.items():
        values = torch.cat(chunks)
        summary[f"{name}_mean"] = values.mean().item()
        summary[f"{name}_q10"] = torch.quantile(values, 0.1).item()
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["refinements"], row["method"])].append(row)
    output = []
    metrics = ["h_relative", "query_relative", "coefficient_mse", "statistical_query_mse"]
    for (refinements, method), group in sorted(groups.items()):
        row = {"refinements": refinements, "method": method, "seeds": len(group)}
        for metric in metrics:
            values = torch.tensor([item[metric] for item in group])
            row[f"{metric}_mean"] = values.mean().item()
            row[f"{metric}_std"] = values.std(unbiased=False).item()
        output.append(row)
    return output


def plot(rows: list[dict], outdir: Path) -> None:
    if plt is None:
        return
    methods = [
        "trained_head_hb",
        "trained_head_ritz_moment_chebyshev",
        "initial_head_ritz_moment_chebyshev",
        "trained_head_corrected_ritz_moment_chebyshev",
        "initial_head_corrected_ritz_moment_chebyshev",
        "identity_pcg",
        "trained_head_pcg",
        "oracle_prompt_esd_polynomial",
        "identity_pcg_equal_work",
        "jacobi_pcg_equal_work",
    ]
    labels = {
        "trained_head_hb": "learned geometry\n+ HB-8",
        "trained_head_ritz_moment_chebyshev": "Ritz--Cheb-8\ntrained probes",
        "initial_head_ritz_moment_chebyshev": "Ritz--Cheb-8\nfrozen probes",
        "trained_head_corrected_ritz_moment_chebyshev": (
            "learned 3-stat\nRitz--Cheb-8"
        ),
        "initial_head_corrected_ritz_moment_chebyshev": (
            "exact closure\nRitz--Cheb-8"
        ),
        "identity_pcg": "pure PCG-8",
        "trained_head_pcg": "learned geometry\n+ PCG-8",
        "oracle_prompt_esd_polynomial": "oracle spectral\npolynomial-8",
        "identity_pcg_equal_work": "pure PCG\nequal work",
        "jacobi_pcg_equal_work": "Jacobi PCG\nequal work",
    }
    refinements = sorted({row["refinements"] for row in rows})
    figure, axes = plt.subplots(len(refinements), 2, figsize=(12, 4 * len(refinements)), squeeze=False)
    for index, refinement in enumerate(refinements):
        lookup = {
            row["method"]: row for row in rows if row["refinements"] == refinement
        }
        names = [
            name
            for name in methods
            if name in lookup and lookup[name]["h_relative_mean"] < 1e3
        ]
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
                [labels[name] for name in names],
                fontsize=7,
            )
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(f"r={refinement}: {metric.replace('_', ' ')}")
    figure.tight_layout()
    figure.savefig(outdir / "pde_matrix_free_solver_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--design",
        choices=["pde_elliptic", "pde_elliptic_correlated"],
        default="pde_elliptic_correlated",
    )
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--state-dimension", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=0.02)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--head-dimension", type=int, default=8)
    parser.add_argument("--refinement-grid", default="0,2")
    parser.add_argument("--spectral-lmax-bound", type=float, default=4.0)
    parser.add_argument("--step-init", type=float, default=0.4)
    parser.add_argument("--momentum-init", type=float, default=0.1)
    parser.add_argument(
        "--training-controller",
        choices=[
            "heavy_ball",
            "ritz_moment_chebyshev",
            "corrected_ritz_moment_chebyshev",
        ],
        default="heavy_ball",
    )
    parser.add_argument("--spectral-krylov-steps", type=int, default=2)
    parser.add_argument("--complement-measure-hidden-dimension", type=int, default=12)
    parser.add_argument("--gram-regularization", type=float, default=1e-5)
    parser.add_argument("--training-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-tasks", type=int, default=4096)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--query-weight", type=float, default=1.0)
    parser.add_argument("--cvar-weight", type=float, default=0.2)
    parser.add_argument("--cvar-fraction", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument(
        "--initial-head-checkpoint-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--freeze-preconditioner-head", action="store_true")
    args = parser.parse_args()
    if args.slots >= args.dimension:
        raise ValueError("slots must be smaller than dimension")
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    cfg = task_config(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.evaluation_only:
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
    rows, overlap_rows, history = [], [], []
    for refinements in parse_ints(args.refinement_grid):
        for seed in parse_ints(args.seeds):
            checkpoint_path = outdir / f"model_r{refinements}_seed{seed}.pt"
            if args.evaluation_only:
                with torch.serialization.safe_globals([type(Path())]):
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=device,
                        weights_only=True,
                    )
                trained = decoder(args, refinements, device)
                trained.load_state_dict(checkpoint["model"])
                initial = checkpoint["initial"]
                run_history = []
            else:
                trained, initial, run_history = train(
                    args, cfg, refinements, seed, device
                )
            run_rows, run_overlap = evaluate(
                args, cfg, trained, initial, refinements, seed, device
            )
            rows.extend(run_rows)
            overlap_rows.append(run_overlap)
            history.extend(run_history)
            if not args.evaluation_only:
                torch.save(
                    {
                        "model": trained.state_dict(),
                        "initial": initial,
                        "args": vars(args),
                        "seed": seed,
                        "refinements": refinements,
                    },
                    checkpoint_path,
                )
    aggregate_rows = aggregate(rows)
    write_csv(outdir / "per_seed.csv", rows)
    write_csv(outdir / "aggregate.csv", aggregate_rows)
    write_csv(outdir / "overlap_summary.csv", overlap_rows)
    if history:
        write_csv(outdir / "training.csv", history)
    plot(aggregate_rows, outdir)
    summary = {
        "claim_boundary": (
            "PCG remains optimal at fixed preconditioner; the learned head is "
            "compared with favorable head-free equal-work PCG."
        ),
        "aggregate": aggregate_rows,
        "overlaps": overlap_rows,
    }
    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
