#!/usr/bin/env python3
"""Paired causal audit of which exact solver objective learns which subspace.

The benchmark is deliberately shallow.  Every task has the same two-cluster
population spectrum, but its low-rank outlier eigenspace is independently Haar rotated.
Consequently a covariance stored in model weights is useless: the single
softmax head has to recover the current slow space from the prompt.

Richardson, Heavy--Ball, and PCG start from the same head parameters, see the
same minibatches in the same order, and receive the same optimization budget.
Only their exact hard-coded solver cell differs.  At test time every trained
head is crossed with every solver cell.  This separates

    solver used to learn geometry  x  solver used to consume geometry.

No eigenspectrum is supplied to the learned head.  Exact eigenspectra are used
only after inference for diagnostics and for explicitly labelled oracle-
interval controls.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from .exact_loop_transformer_decoder import ExactLoopTransformerDecoder
    from .first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
except ImportError:
    from exact_loop_transformer_decoder import ExactLoopTransformerDecoder
    from first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )

Tensor = torch.Tensor
TRAINING_CONTROLLERS = ("richardson", "heavy_ball", "pcg")
HEAD_NAMES = ("initial", *TRAINING_CONTROLLERS)


@dataclass(frozen=True)
class RotatedLowRankBatch:
    equations: Tensor
    observations: Tensor
    query: Tensor
    coefficient_true: Tensor
    normal: Tensor
    rhs: Tensor
    posterior_mean: Tensor
    posterior_covariance: Tensor
    population_outlier_space: Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def haar_orthogonal(
    batch: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    gaussian = torch.randn(
        batch,
        dimension,
        dimension,
        device=device,
        dtype=dtype,
    )
    orthogonal, triangular = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diagonal(triangular, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return orthogonal * signs.unsqueeze(-2)


def sample_rotated_low_rank_batch(
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> RotatedLowRankBatch:
    """Draw a randomly rotated slow-rank inverse problem in scaled form."""

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dimension = args.dimension
    rotation = haar_orthogonal(
        batch_size,
        dimension,
        device=device,
        dtype=dtype,
    )
    population_eigenvalues = torch.ones(
        dimension,
        device=device,
        dtype=dtype,
    )
    population_eigenvalues[: args.outlier_rank] = args.condition
    population_eigenvalues /= population_eigenvalues.mean()
    population_sqrt = torch.diag_embed(
        population_eigenvalues.sqrt().expand(batch_size, -1)
    )

    standard_rows = torch.randn(
        batch_size,
        args.prompt_length,
        dimension,
        device=device,
        dtype=dtype,
    )
    raw_equations = (
        standard_rows @ population_sqrt @ rotation.transpose(-1, -2)
    ) / math.sqrt(dimension)
    standard_query = torch.randn(
        batch_size,
        dimension,
        device=device,
        dtype=dtype,
    )
    raw_query = torch.einsum(
        "bi,bij,bjk->bk",
        standard_query,
        population_sqrt,
        rotation.transpose(-1, -2),
    ) / math.sqrt(dimension)
    coefficient_true = torch.randn(
        batch_size,
        dimension,
        device=device,
        dtype=dtype,
    ) * math.sqrt(args.prior_variance)
    clean_observations = torch.einsum(
        "bmk,bk->bm", raw_equations, coefficient_true
    )
    noisy_observations = clean_observations + torch.randn_like(
        clean_observations
    ) * math.sqrt(args.noise_variance)

    noise_scale = math.sqrt(1.0 / args.noise_variance)
    equations = raw_equations * noise_scale
    observations = noisy_observations * noise_scale
    ridge = 1.0 / args.prior_variance
    identity = torch.eye(
        dimension,
        device=device,
        dtype=dtype,
    ).expand(batch_size, -1, -1)
    normal = equations.transpose(-1, -2) @ equations + ridge * identity
    rhs = torch.einsum("bmk,bm->bk", equations, observations)
    posterior_mean = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
    posterior_covariance = torch.linalg.inv(normal)
    # The first columns correspond to the deliberately strong low-rank modes.
    population_outlier_space = rotation[:, :, : args.outlier_rank]
    return RotatedLowRankBatch(
        equations=equations,
        observations=observations,
        query=raw_query,
        coefficient_true=coefficient_true,
        normal=normal,
        rhs=rhs,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        population_outlier_space=population_outlier_space,
    )


def build_decoder(
    args: argparse.Namespace,
    controller: str,
    device: torch.device,
) -> ExactLoopTransformerDecoder:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    return ExactLoopTransformerDecoder(
        dimension=args.dimension,
        depth=args.depth,
        head_dimension=args.head_dimension,
        slots=args.slots,
        controller=controller,
        spectral_lmax_bound=args.spectral_lmax_bound,
        step_init=args.step_init,
        momentum_init=args.momentum_init,
        preconditioner_head_type="equivariant_matrix_free_nystrom",
        prompt_subspace_refinement_steps=args.refinement_steps,
    ).to(device=device, dtype=dtype)


def relative_energy(
    prediction: Tensor,
    target: Tensor,
    normal: Tensor,
) -> Tensor:
    error = prediction - target
    numerator = torch.einsum("bi,bij,bj->b", error, normal, error)
    denominator = torch.einsum(
        "bi,bij,bj->b", target, normal, target
    ).clamp_min(1e-30)
    return numerator / denominator


def hvp_from_normal(normal: Tensor):
    def hvp(vector: Tensor) -> Tensor:
        return torch.einsum("bij,bj->bi", normal, vector)

    return hvp


def effective_spectrum(
    preconditioner: Tensor,
    normal: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    dimension = normal.shape[-1]
    identity = torch.eye(
        dimension,
        device=normal.device,
        dtype=normal.dtype,
    )
    factor = torch.linalg.cholesky(preconditioner + 1e-10 * identity)
    operator = factor.transpose(-1, -2) @ normal @ factor
    operator = 0.5 * (operator + operator.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    return eigenvalues.clamp_min(1e-12), eigenvectors, factor


def oracle_hb_coefficients(spectrum: Tensor) -> tuple[Tensor, Tensor]:
    lower = spectrum[:, 0]
    upper = spectrum[:, -1]
    sqrt_lower = torch.sqrt(lower)
    sqrt_upper = torch.sqrt(upper)
    step = 4.0 / (sqrt_upper + sqrt_lower).square()
    momentum = ((sqrt_upper - sqrt_lower) / (sqrt_upper + sqrt_lower)).square()
    return step, momentum


def polynomial_residual(
    spectrum: Tensor,
    depth: int,
    step: float | Tensor,
    momentum: float | Tensor,
) -> Tensor:
    """Exact HB/Richardson residual polynomial for x_-1=x_0=0."""

    alpha = torch.as_tensor(
        step,
        device=spectrum.device,
        dtype=spectrum.dtype,
    )
    beta = torch.as_tensor(
        momentum,
        device=spectrum.device,
        dtype=spectrum.dtype,
    )
    if alpha.ndim == 0:
        alpha = alpha.expand(spectrum.shape[0])
    if beta.ndim == 0:
        beta = beta.expand(spectrum.shape[0])
    previous = torch.ones_like(spectrum)
    current = torch.ones_like(spectrum)
    for _ in range(depth):
        following = (
            (1.0 + beta[:, None] - alpha[:, None] * spectrum) * current
            - beta[:, None] * previous
        )
        previous, current = current, following
    return current


def spectral_prediction(
    spectrum: Tensor,
    eigenvectors: Tensor,
    factor: Tensor,
    rhs: Tensor,
    residual: Tensor,
) -> Tensor:
    transformed_rhs = torch.einsum("bji,bj->bi", factor, rhs)
    spectral_rhs = torch.einsum("bji,bj->bi", eigenvectors, transformed_rhs)
    spectral_solution = spectral_rhs / spectrum
    weights = spectrum * spectral_solution.square()
    return (weights * residual.square()).sum(dim=-1) / weights.sum(
        dim=-1
    ).clamp_min(1e-30)


def principal_overlap(left: Tensor, right: Tensor) -> Tensor:
    rank = min(left.shape[-1], right.shape[-1])
    return torch.linalg.matrix_norm(
        left.transpose(-1, -2) @ right,
        ord="fro",
        dim=(-2, -1),
    ).square() / rank


def projected_fraction(space: Tensor, vector: Tensor) -> Tensor:
    coordinates = torch.einsum("bks,bk->bs", space, vector)
    return coordinates.square().sum(dim=-1) / vector.square().sum(
        dim=-1
    ).clamp_min(1e-30)


def train_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[
    dict[str, ExactLoopTransformerDecoder],
    dict[str, Tensor],
    list[dict],
]:
    set_seed(seed)
    models = {
        controller: build_decoder(args, controller, device)
        for controller in TRAINING_CONTROLLERS
    }
    common_head_state = copy.deepcopy(
        models["richardson"].preconditioner_head.state_dict()
    )
    for model in models.values():
        model.preconditioner_head.load_state_dict(common_head_state)
    initial_head_state = copy.deepcopy(common_head_state)
    optimizers = {
        controller: torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for controller, model in models.items()
    }
    tail = max(1, math.ceil(args.cvar_fraction * args.batch_size))
    history: list[dict] = []
    # Constructors consumed different amounts of randomness.  Reset so the
    # training stream depends only on the public seed, not controller details.
    set_seed(seed + 10_000)
    for step in range(1, args.training_steps + 1):
        batch = sample_rotated_low_rank_batch(args.batch_size, args, device)
        for controller in TRAINING_CONTROLLERS:
            model = models[controller]
            prediction, _ = model(
                batch.equations,
                batch.observations,
                1.0 / args.prior_variance,
            )
            task_risk = relative_energy(
                prediction,
                batch.posterior_mean,
                batch.normal,
            )
            mean_risk = task_risk.mean()
            cvar = task_risk.topk(tail).values.mean()
            loss = mean_risk + args.cvar_weight * cvar
            optimizer = optimizers[controller]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()
            if (
                step == 1
                or step % args.log_every == 0
                or step == args.training_steps
            ):
                if controller == "pcg":
                    solver_step = math.nan
                    solver_momentum = math.nan
                else:
                    solver_step_tensor, solver_momentum_tensor = (
                        model.heavy_ball_coefficients()
                    )
                    solver_step = solver_step_tensor.item()
                    solver_momentum = solver_momentum_tensor.item()
                row = {
                    "seed": seed,
                    "training_step": step,
                    "training_controller": controller,
                    "loss": loss.item(),
                    "mean_h_relative": mean_risk.item(),
                    "cvar_h_relative": cvar.item(),
                    "solver_step": solver_step,
                    "solver_momentum": solver_momentum,
                }
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    return models, initial_head_state, history


def head_models(
    args: argparse.Namespace,
    trained: dict[str, ExactLoopTransformerDecoder],
    initial_head_state: dict[str, Tensor],
    device: torch.device,
) -> dict[str, ExactLoopTransformerDecoder]:
    initial = build_decoder(args, "pcg", device)
    initial.preconditioner_head.load_state_dict(initial_head_state)
    return {"initial": initial, **trained}


@torch.no_grad()
def evaluate_seed(
    args: argparse.Namespace,
    seed: int,
    trained: dict[str, ExactLoopTransformerDecoder],
    initial_head_state: dict[str, Tensor],
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    models = head_models(args, trained, initial_head_state, device)
    for model in models.values():
        model.eval()
    richardson_step, _ = trained["richardson"].heavy_ball_coefficients()
    hb_step, hb_momentum = trained["heavy_ball"].heavy_ball_coefficients()
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    geometry_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    theory_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    setup_block_rounds = args.refinement_steps + 1
    setup_scalar_hvps = setup_block_rounds * args.slots
    set_seed(seed + 100_000)
    remaining = args.evaluation_tasks
    while remaining:
        count = min(args.evaluation_batch_size, remaining)
        batch = sample_rotated_low_rank_batch(count, args, device)
        hvp = hvp_from_normal(batch.normal)
        _, normal_eigenvectors = torch.linalg.eigh(batch.normal)
        empirical_outlier_space = normal_eigenvectors[
            :, :, -args.outlier_rank :
        ]
        identity = torch.eye(
            args.dimension,
            device=device,
            dtype=batch.normal.dtype,
        ).expand(count, -1, -1)

        evaluated_heads: dict[str, tuple[Tensor, dict]] = {}
        for head_name, model in models.items():
            geometry = model.build_prompt_geometry(
                batch.equations,
                1.0 / args.prior_variance,
            )
            preconditioner = geometry.preconditioner.materialize()
            info = geometry.head_info
            evaluated_heads[head_name] = (preconditioner, info)
        evaluated_heads["identity"] = (
            identity,
            {
                "directions": identity[:, :, : args.slots],
                "attention": batch.normal.new_full(
                    (count, args.slots, args.prompt_length),
                    1.0 / args.prompt_length,
                ),
            },
        )

        for head_name, (preconditioner, info) in evaluated_heads.items():
            spectrum, eigenvectors, factor = effective_spectrum(
                preconditioner,
                batch.normal,
            )
            lower, upper = spectrum[:, 0], spectrum[:, -1]
            oracle_richardson_step = 2.0 / (lower + upper)
            oracle_hb_step, oracle_hb_momentum = oracle_hb_coefficients(spectrum)
            predictions = {
                "learned_global_richardson": run_heavy_ball_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth,
                    richardson_step,
                    0.0,
                )[0],
                "learned_global_heavy_ball": run_heavy_ball_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth,
                    hb_step,
                    hb_momentum,
                )[0],
                "oracle_interval_richardson": run_heavy_ball_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth,
                    oracle_richardson_step,
                    0.0,
                )[0],
                "oracle_interval_heavy_ball": run_heavy_ball_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth,
                    oracle_hb_step,
                    oracle_hb_momentum,
                )[0],
                "oracle_interval_chebyshev": run_chebyshev_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth,
                    lower,
                    upper,
                )[0],
                "pcg": run_pcg_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth,
                )[0],
            }
            if head_name == "identity":
                predictions["pcg_equal_block_work"] = run_pcg_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth + setup_block_rounds,
                )[0]
                predictions["pcg_equal_scalar_work"] = run_pcg_state_machine(
                    hvp,
                    batch.rhs,
                    preconditioner,
                    args.depth + setup_scalar_hvps,
                )[0]

            target_query = torch.einsum(
                "bi,bi->b", batch.query, batch.posterior_mean
            )
            for cell_name, prediction in predictions.items():
                task_risk = relative_energy(
                    prediction,
                    batch.posterior_mean,
                    batch.normal,
                )
                predicted_query = torch.einsum(
                    "bi,bi->b", batch.query, prediction
                )
                values = totals[(head_name, cell_name)]
                values["count"] += count
                values["h_relative"] += task_risk.sum().item()
                values["coefficient_mse"] += (
                    prediction - batch.posterior_mean
                ).square().mean(dim=-1).sum().item()
                values["query_numerator"] += (
                    predicted_query - target_query
                ).square().sum().item()
                values["query_denominator"] += target_query.square().sum().item()

            directions = torch.linalg.qr(
                info["directions"], mode="reduced"
            ).Q
            attention = info["attention"].clamp_min(1e-30)
            geometry = geometry_totals[head_name]
            geometry["count"] += count
            geometry["effective_condition"] += (
                upper / lower
            ).sum().item()
            geometry["empirical_outlier_overlap"] += principal_overlap(
                directions,
                empirical_outlier_space,
            ).sum().item()
            geometry["population_outlier_overlap"] += principal_overlap(
                directions,
                batch.population_outlier_space,
            ).sum().item()
            geometry["posterior_overlap"] += projected_fraction(
                directions,
                batch.posterior_mean,
            ).sum().item()
            geometry["query_overlap"] += projected_fraction(
                directions,
                batch.query,
            ).sum().item()
            geometry["attention_entropy"] += (
                -(attention * attention.log()).sum(dim=-1).mean(dim=-1)
            ).sum().item()

            spectral_cases = {
                "learned_global_richardson": (richardson_step, 0.0),
                "learned_global_heavy_ball": (hb_step, hb_momentum),
                "oracle_interval_richardson": (
                    oracle_richardson_step,
                    0.0,
                ),
                "oracle_interval_heavy_ball": (
                    oracle_hb_step,
                    oracle_hb_momentum,
                ),
            }
            for cell_name, (step, momentum) in spectral_cases.items():
                residual = polynomial_residual(
                    spectrum,
                    args.depth,
                    step,
                    momentum,
                )
                prediction = spectral_prediction(
                    spectrum,
                    eigenvectors,
                    factor,
                    batch.rhs,
                    residual,
                )
                realized = relative_energy(
                    predictions[cell_name],
                    batch.posterior_mean,
                    batch.normal,
                )
                values = theory_totals[(head_name, cell_name)]
                values["count"] += count
                values["predicted"] += prediction.sum().item()
                values["realized"] += realized.sum().item()
                values["absolute_gap"] += (
                    prediction - realized
                ).abs().sum().item()
        remaining -= count

    risk_rows = []
    for (head_name, cell_name), values in sorted(totals.items()):
        count = values["count"]
        risk_rows.append(
            {
                "seed": seed,
                "trained_head": head_name,
                "evaluation_cell": cell_name,
                "depth": (
                    args.depth + setup_block_rounds
                    if cell_name == "pcg_equal_block_work"
                    else args.depth + setup_scalar_hvps
                    if cell_name == "pcg_equal_scalar_work"
                    else args.depth
                ),
                "setup_block_rounds": (
                    0 if head_name == "identity" else setup_block_rounds
                ),
                "setup_scalar_hvp_equivalents": (
                    0 if head_name == "identity" else setup_scalar_hvps
                ),
                "h_relative": values["h_relative"] / count,
                "coefficient_mse": values["coefficient_mse"] / count,
                "query_relative": values["query_numerator"]
                / max(values["query_denominator"], 1e-30),
            }
        )
    geometry_rows = []
    for head_name, values in sorted(geometry_totals.items()):
        count = values["count"]
        geometry_rows.append(
            {
                "seed": seed,
                "trained_head": head_name,
                **{
                    key: value / count
                    for key, value in values.items()
                    if key != "count"
                },
            }
        )
    theory_rows = []
    for (head_name, cell_name), values in sorted(theory_totals.items()):
        count = values["count"]
        theory_rows.append(
            {
                "seed": seed,
                "trained_head": head_name,
                "evaluation_cell": cell_name,
                "predicted_h_relative": values["predicted"] / count,
                "realized_h_relative": values["realized"] / count,
                "mean_absolute_gap": values["absolute_gap"] / count,
            }
        )
    return risk_rows, geometry_rows, theory_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, members in sorted(groups.items()):
        result = dict(zip(group_keys, group))
        result["seeds"] = len(members)
        for metric in metrics:
            values = torch.tensor(
                [member[metric] for member in members], dtype=torch.float64
            )
            result[f"{metric}_mean"] = values.mean().item()
            result[f"{metric}_std"] = values.std(unbiased=False).item()
        output.append(result)
    return output


def make_plot(
    risk_aggregate: list[dict],
    geometry_aggregate: list[dict],
    output: Path,
) -> None:
    if plt is None:
        return
    heads = ["initial", "richardson", "heavy_ball", "pcg"]
    cells = [
        "learned_global_richardson",
        "learned_global_heavy_ball",
        "oracle_interval_chebyshev",
        "pcg",
    ]
    lookup = {
        (row["trained_head"], row["evaluation_cell"]): row
        for row in risk_aggregate
    }
    matrix = np.full((len(heads), len(cells)), np.nan)
    for row_index, head in enumerate(heads):
        for column_index, cell in enumerate(cells):
            row = lookup.get((head, cell))
            if row is not None:
                matrix[row_index, column_index] = math.log10(
                    max(row["h_relative_mean"], 1e-30)
                )

    geometry_lookup = {
        row["trained_head"]: row for row in geometry_aggregate
    }
    figure, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    image = axes[0].imshow(matrix, cmap="viridis_r", aspect="auto")
    axes[0].set_xticks(range(len(cells)), [
        "Richardson\nglobal",
        "HB\nglobal",
        "Chebyshev\noracle interval",
        "PCG",
    ])
    axes[0].set_yticks(range(len(heads)), heads)
    axes[0].set_title(r"$\log_{10}$ relative $H$-risk")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[0].text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < -2 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    x = np.arange(len(heads))
    axes[1].bar(
        x - 0.18,
        [
            geometry_lookup[name]["empirical_outlier_overlap_mean"]
            for name in heads
        ],
        width=0.36,
        label="empirical outlier space",
    )
    axes[1].bar(
        x + 0.18,
        [
            geometry_lookup[name]["population_outlier_overlap_mean"]
            for name in heads
        ],
        width=0.36,
        label="population outlier space",
    )
    axes[1].set_xticks(x, heads, rotation=20)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("principal-space overlap")
    axes[1].set_title("What geometry was learned?")
    axes[1].legend(fontsize=8)

    conditions = [
        geometry_lookup[name]["effective_condition_mean"] for name in heads
    ]
    axes[2].bar(x, conditions)
    axes[2].set_xticks(x, heads, rotation=20)
    axes[2].set_yscale("log")
    axes[2].set_ylabel(r"mean $\kappa(B^{1/2}HB^{1/2})$")
    axes[2].set_title("Conditioning after the head")

    work_methods = [
        ("heavy_ball", "learned_global_heavy_ball", "head + HB"),
        ("heavy_ball", "oracle_interval_chebyshev", "head + Cheb"),
        ("heavy_ball", "pcg", "head + PCG"),
        ("identity", "pcg_equal_block_work", "pure PCG\nequal rounds"),
        ("identity", "pcg_equal_scalar_work", "pure PCG\nequal scalar HVP"),
    ]
    work_values = [
        lookup[(head, cell)]["h_relative_mean"]
        for head, cell, _ in work_methods
    ]
    axes[3].bar(
        np.arange(len(work_methods)),
        work_values,
        color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
    )
    axes[3].set_xticks(
        np.arange(len(work_methods)),
        [label for _, _, label in work_methods],
        rotation=25,
    )
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"relative $H$-risk")
    axes[3].set_title("Does setup beat pure PCG?")
    figure.suptitle(
        "Paired shallow RRS audit: training objective x exact solver cell"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=12)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--outlier-rank", type=int, default=2)
    parser.add_argument("--condition", type=float, default=100.0)
    parser.add_argument("--prior-variance", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=0.02)
    parser.add_argument("--depth", type=int, default=4)
    # One extra slot anchors the outlier correction to the top of the bulk.
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--head-dimension", type=int, default=8)
    parser.add_argument("--refinement-steps", type=int, default=1)
    parser.add_argument("--spectral-lmax-bound", type=float, default=4.0)
    parser.add_argument("--step-init", type=float, default=0.25)
    parser.add_argument("--momentum-init", type=float, default=0.1)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--cvar-fraction", type=float, default=0.1)
    parser.add_argument("--cvar-weight", type=float, default=0.1)
    parser.add_argument("--evaluation-tasks", type=int, default=4096)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/transformers/rotated_low_rank_controller_learning"
        ),
    )
    args = parser.parse_args()
    if not 0 < args.outlier_rank < args.slots < args.dimension:
        raise ValueError("require 0 < outlier_rank < slots < dimension")
    if args.condition <= 1:
        raise ValueError("condition must exceed one")
    if args.training_steps <= 0 or args.depth <= 0:
        raise ValueError("training_steps and depth must be positive")
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    all_training: list[dict] = []
    all_risks: list[dict] = []
    all_geometry: list[dict] = []
    all_theory: list[dict] = []
    for seed in parse_seeds(args.seeds):
        trained, initial_head_state, history = train_seed(
            args, seed, device
        )
        risks, geometry, theory = evaluate_seed(
            args,
            seed,
            trained,
            initial_head_state,
            device,
        )
        all_training.extend(history)
        all_risks.extend(risks)
        all_geometry.extend(geometry)
        all_theory.extend(theory)

    risk_metrics = ("h_relative", "coefficient_mse", "query_relative")
    geometry_metrics = (
        "effective_condition",
        "empirical_outlier_overlap",
        "population_outlier_overlap",
        "posterior_overlap",
        "query_overlap",
        "attention_entropy",
    )
    theory_metrics = (
        "predicted_h_relative",
        "realized_h_relative",
        "mean_absolute_gap",
    )
    risk_aggregate = aggregate(
        all_risks,
        ("trained_head", "evaluation_cell"),
        risk_metrics,
    )
    geometry_aggregate = aggregate(
        all_geometry,
        ("trained_head",),
        geometry_metrics,
    )
    theory_aggregate = aggregate(
        all_theory,
        ("trained_head", "evaluation_cell"),
        theory_metrics,
    )
    write_csv(args.output / "training.csv", all_training)
    write_csv(args.output / "risk_per_seed.csv", all_risks)
    write_csv(args.output / "risk_aggregate.csv", risk_aggregate)
    write_csv(args.output / "geometry_per_seed.csv", all_geometry)
    write_csv(args.output / "geometry_aggregate.csv", geometry_aggregate)
    write_csv(args.output / "theory_per_seed.csv", all_theory)
    write_csv(args.output / "theory_aggregate.csv", theory_aggregate)
    make_plot(
        risk_aggregate,
        geometry_aggregate,
        args.output / "controller_subspace_cross.png",
    )
    summary = {
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": str(device),
        "setup_cost": {
            "block_hvp_rounds": args.refinement_steps + 1,
            "sequential_scalar_hvp_equivalents": (
                args.refinement_steps + 1
            )
            * args.slots,
        },
        "risk_aggregate": risk_aggregate,
        "geometry_aggregate": geometry_aggregate,
        "theory_aggregate": theory_aggregate,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
