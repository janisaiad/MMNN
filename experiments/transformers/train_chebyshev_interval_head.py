#!/usr/bin/env python3
"""Train only the spectral-interval MLP for the tied Chebyshev decoder.

The prompt-conditioned preconditioner is loaded and frozen.  Seven fixed
summary reductions of its symmetric effective operator are exposed to a small
MLP.  Eigenvalues are used only as supervised targets and evaluation oracles,
never as MLP inputs or inference-time Chebyshev arithmetic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from .constructive_weakform_richardson_transformer import (
        TaskConfig,
        sample_weak_batch,
    )
    from .first_principles_decoder_cells import (
        fixed_prompt_linear_attention_hvp,
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from .first_principles_inverse_decoder import (
        PromptSpectralIntervalMLP,
        spectral_interval_coverage_loss,
    )
    from .structured_one_head_heavyball import StructuredOneHeadHeavyBall
except ImportError:
    from constructive_weakform_richardson_transformer import (
        TaskConfig,
        sample_weak_batch,
    )
    from first_principles_decoder_cells import (
        fixed_prompt_linear_attention_hvp,
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from first_principles_inverse_decoder import (
        PromptSpectralIntervalMLP,
        spectral_interval_coverage_loss,
    )
    from structured_one_head_heavyball import StructuredOneHeadHeavyBall

Tensor = torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frozen_preconditioner(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = checkpoint["args"]
    dtype = torch.float64 if args["dtype"] == "float64" else torch.float32
    model = StructuredOneHeadHeavyBall(
        dimension=args["K"],
        depth=args["depth"],
        head_dimension=args["head_dimension"],
        slots=args["slots"],
        max_strength=args["max_strength"],
        strength_init=args["strength_init"],
        head_mode=args["head_mode"],
        spectral_lmax_bound=args["spectral_lmax_bound"],
        step_init=args["step_init"],
        momentum_init=args["momentum_init"],
        solver_cell=args.get("solver_cell", "pcg"),
        base_preconditioner=args.get("base_preconditioner", "jacobi"),
        base_blocks=args.get("base_blocks", 2),
        strength_scaling=args.get("strength_scaling", "fixed"),
        reference_prompt_length=args.get("reference_prompt_length", args["prompt_len"]),
        slot_orthogonalization=args.get("slot_orthogonalization", "independent"),
        correction_mode=args.get("correction_mode", "positive"),
        subspace_refinement_steps=args.get("subspace_refinement_steps", 0) or 0,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.preconditioner_head, args, dtype


def symmetric_effective_operator(preconditioner: Tensor, normal_matrix: Tensor) -> Tensor:
    dimension = normal_matrix.shape[-1]
    eye = torch.eye(
        dimension,
        device=normal_matrix.device,
        dtype=normal_matrix.dtype,
    )
    factor = torch.linalg.cholesky(preconditioner + 1e-10 * eye)
    operator = factor.transpose(-1, -2) @ normal_matrix @ factor
    return 0.5 * (operator + operator.transpose(-1, -2))


def effective_spectrum_features(operator: Tensor, prompt_length: int) -> Tensor:
    """Seven fixed reductions; no eigendecomposition is used here."""

    dimension = operator.shape[-1]
    diagonal = torch.diagonal(operator, dim1=-2, dim2=-1).clamp_min(1e-12)
    trace_mean = diagonal.mean(dim=-1)
    frobenius_mean = torch.linalg.matrix_norm(
        operator,
        ord="fro",
        dim=(-2, -1),
    ) / math.sqrt(dimension)
    absolute_row_sum = operator.abs().sum(dim=-1)
    off_diagonal = operator - torch.diag_embed(diagonal)
    off_fraction = torch.linalg.matrix_norm(
        off_diagonal,
        ord="fro",
        dim=(-2, -1),
    ) / torch.linalg.matrix_norm(operator, ord="fro", dim=(-2, -1)).clamp_min(1e-12)
    batch_size = operator.shape[0]
    prompt_feature = operator.new_full((batch_size,), math.log1p(prompt_length))
    return torch.stack(
        [
            torch.log(trace_mean),
            torch.log(frobenius_mean.clamp_min(1e-12)),
            torch.log(diagonal.amin(dim=-1)),
            torch.log(diagonal.amax(dim=-1)),
            torch.log(absolute_row_sum.amax(dim=-1).clamp_min(1e-12)),
            torch.log1p(off_fraction),
            prompt_feature,
        ],
        dim=-1,
    )


def task_config(saved: Dict, prompt_length: int) -> TaskConfig:
    return TaskConfig(
        K=saved["K"],
        prompt_len=prompt_length,
        prior_var=saved["prior_var"],
        noise_var=saved["noise_var"],
        design=saved["design"],
        cond=saved["cond"],
        dtype=saved["dtype"],
        pde_state_dim=saved.get("pde_state_dim", 0),
    )


def interval_batch(head, cfg: TaskConfig, batch_size: int, device: torch.device):
    batch = sample_weak_batch(batch_size, cfg, device)
    with torch.no_grad():
        preconditioner, _ = head(batch.G, batch.H)
        operator = symmetric_effective_operator(preconditioner, batch.H)
        features = effective_spectrum_features(operator, cfg.prompt_len)
        eigenvalues = torch.linalg.eigvalsh(operator).clamp_min(1e-12)
    return batch, preconditioner, features, eigenvalues[:, 0], eigenvalues[:, -1]


def relative_h_error(prediction: Tensor, target: Tensor, normal_matrix: Tensor) -> float:
    return relative_h_loss(prediction, target, normal_matrix).item()


def relative_h_loss(prediction: Tensor, target: Tensor, normal_matrix: Tensor) -> Tensor:
    error = prediction - target
    numerator = torch.einsum("bk,bkl,bl->b", error, normal_matrix, error)
    denominator = torch.einsum(
        "bk,bkl,bl->b",
        target,
        normal_matrix,
        target,
    ).clamp_min(1e-30)
    return (numerator / denominator).mean()


@torch.no_grad()
def evaluate(
    interval_head,
    preconditioner_head,
    saved: Dict,
    device: torch.device,
    batch_size: int,
    prompt_grid: list[int],
) -> list[Dict[str, float | int]]:
    rows = []
    for prompt_length in prompt_grid:
        cfg = task_config(saved, prompt_length)
        set_seed(1000 + prompt_length)
        batch, preconditioner, features, true_min, true_max = interval_batch(
            preconditioner_head,
            cfg,
            batch_size,
            device,
        )
        predicted_min, predicted_max = interval_head(features)
        covered = (predicted_min <= true_min) & (predicted_max >= true_max)
        noise_precision = 1.0 / cfg.noise_var
        prior_precision = 1.0 / cfg.prior_var

        def hvp(vector: Tensor) -> Tensor:
            return fixed_prompt_linear_attention_hvp(
                batch.G,
                vector,
                noise_precision,
                prior_precision,
            )

        depth = saved["depth"]
        learned_chebyshev = run_chebyshev_state_machine(
            hvp,
            batch.c,
            preconditioner,
            depth,
            predicted_min,
            predicted_max,
        )[0]
        oracle_chebyshev = run_chebyshev_state_machine(
            hvp,
            batch.c,
            preconditioner,
            depth,
            true_min,
            true_max,
        )[0]
        pcg = run_pcg_state_machine(
            hvp,
            batch.c,
            preconditioner,
            depth,
        )[0]
        condition_sqrt = torch.sqrt(true_max / true_min)
        momentum = ((condition_sqrt - 1.0) / (condition_sqrt + 1.0)).square()
        step_size = 4.0 / (torch.sqrt(true_max) + torch.sqrt(true_min)).square()
        heavy_ball = run_heavy_ball_state_machine(
            hvp,
            batch.c,
            preconditioner,
            depth,
            step_size,
            momentum,
        )[0]
        rows.append(
            {
                "prompt_length": prompt_length,
                "coverage": covered.double().mean().item(),
                "lower_coverage": (predicted_min <= true_min).double().mean().item(),
                "upper_coverage": (predicted_max >= true_max).double().mean().item(),
                "predicted_log_width": torch.log(
                    predicted_max / predicted_min
                ).mean().item(),
                "oracle_log_width": torch.log(true_max / true_min).mean().item(),
                "learned_chebyshev_hrel": relative_h_error(
                    learned_chebyshev, batch.beta_post, batch.H
                ),
                "oracle_chebyshev_hrel": relative_h_error(
                    oracle_chebyshev, batch.beta_post, batch.H
                ),
                "heavy_ball_hrel": relative_h_error(
                    heavy_ball, batch.beta_post, batch.H
                ),
                "pcg_hrel": relative_h_error(pcg, batch.beta_post, batch.H),
            }
        )
    return rows


def main(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    preconditioner_head, saved, dtype = load_frozen_preconditioner(
        args.checkpoint,
        device,
    )
    interval_head = PromptSpectralIntervalMLP(
        hidden_dimension=args.hidden_dimension,
        safety_margin=args.safety_margin,
    ).to(device=device, dtype=dtype)
    if args.interval_checkpoint:
        interval_checkpoint = torch.load(
            args.interval_checkpoint,
            map_location=device,
            weights_only=True,
        )
        interval_head.load_state_dict(interval_checkpoint["model"])
    optimizer = torch.optim.AdamW(
        interval_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    prompt_grid = [int(value) for value in args.prompt_grid.split(",")]
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training.csv"
    set_seed(args.seed)
    for step in range(1, args.steps + 1):
        prompt_length = random.choice(prompt_grid)
        cfg = task_config(saved, prompt_length)
        batch, preconditioner, features, true_min, true_max = interval_batch(
            preconditioner_head,
            cfg,
            args.batch_size,
            device,
        )
        predicted_min, predicted_max = interval_head(features)
        coverage_loss = spectral_interval_coverage_loss(
            predicted_min,
            predicted_max,
            true_min,
            true_max,
        )
        if args.solver_loss_weight > 0.0:
            noise_precision = 1.0 / cfg.noise_var
            prior_precision = 1.0 / cfg.prior_var

            def hvp(vector: Tensor) -> Tensor:
                return fixed_prompt_linear_attention_hvp(
                    batch.G,
                    vector,
                    noise_precision,
                    prior_precision,
                )

            prediction = run_chebyshev_state_machine(
                hvp,
                batch.c,
                preconditioner,
                saved["depth"],
                predicted_min,
                predicted_max,
            )[0]
            solver_loss = relative_h_loss(prediction, batch.beta_post, batch.H)
        else:
            solver_loss = torch.zeros_like(coverage_loss)
        loss = coverage_loss + args.solver_loss_weight * solver_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": loss.item(),
                "coverage_loss": coverage_loss.item(),
                "solver_loss": solver_loss.item(),
                "coverage": (
                    (predicted_min <= true_min) & (predicted_max >= true_max)
                ).double().mean().item(),
            }
            print(json.dumps(row, sort_keys=True))
            exists = metrics_path.exists()
            with metrics_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
    rows = evaluate(
        interval_head,
        preconditioner_head,
        saved,
        device,
        args.eval_batch_size,
        prompt_grid,
    )
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    with (output / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "model": interval_head.state_dict(),
            "args": vars(args),
            "source_checkpoint": args.checkpoint,
        },
        output / "interval_head.pt",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-grid", default="16,32,64,128")
    parser.add_argument("--hidden-dimension", type=int, default=16)
    parser.add_argument("--safety-margin", type=float, default=0.1)
    parser.add_argument("--interval-checkpoint", default="")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--solver-loss-weight", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
