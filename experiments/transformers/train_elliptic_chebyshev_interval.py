#!/usr/bin/env python3
"""Train only a Chebyshev interval MLP on the frozen elliptic ICL trunk.

The physical dictionary, weak encoder, one-head Ritz preconditioner, and exact
Chebyshev recurrence are frozen.  The MLP predicts only two spectral endpoints
from seven fixed prompt reductions.  Exact eigenvalues are training/calibration
targets and are never inference inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .evaluate_trained_loop_controllers import build_model
    from .exact_loop_transformer_decoder import (
        effective_spectrum_features,
        normal_equations,
        symmetric_effective_operator,
    )
    from .first_principles_decoder_cells import (
        chebyshev_coefficient_schedule,
        run_precomputed_chebyshev_state_machine,
    )
    from .first_principles_inverse_decoder import spectral_interval_coverage_loss
    from .pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )
except ImportError:
    from evaluate_trained_loop_controllers import build_model
    from exact_loop_transformer_decoder import (
        effective_spectrum_features,
        normal_equations,
        symmetric_effective_operator,
    )
    from first_principles_decoder_cells import (
        chebyshev_coefficient_schedule,
        run_precomputed_chebyshev_state_machine,
    )
    from first_principles_inverse_decoder import spectral_interval_coverage_loss
    from pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )

Tensor = torch.Tensor


def log_uniform(lower: float, upper: float) -> float:
    draw = random.random()
    return math.exp((1.0 - draw) * math.log(lower) + draw * math.log(upper))


def exact_chebyshev(
    normal_matrix: Tensor,
    rhs: Tensor,
    preconditioner: Tensor,
    depth: int,
    spectral_min: Tensor,
    spectral_max: Tensor,
) -> Tensor:
    """The same fixed recurrence as the decoder, without diagnostic syncs."""

    def hvp(vector: Tensor) -> Tensor:
        return torch.einsum("bij,bj->bi", normal_matrix, vector)

    step_schedule, momentum_schedule = chebyshev_coefficient_schedule(
        rhs, depth, spectral_min, spectral_max
    )
    # Avoid the diagnostic wrapper in the differentiable training path.
    x = torch.zeros_like(rhs)
    x_previous = torch.zeros_like(rhs)
    for layer in range(depth):
        residual = rhs - hvp(x)
        preconditioned_residual = torch.einsum(
            "bij,bj->bi", preconditioner, residual
        )
        x_next = (
            x
            + step_schedule[:, layer, None] * preconditioned_residual
            + momentum_schedule[:, layer, None] * (x - x_previous)
        )
        x_previous, x = x, x_next
    return x


def per_task_h_relative_squared(
    prediction: Tensor,
    target: Tensor,
    normal_matrix: Tensor,
) -> Tensor:
    error = prediction - target
    numerator = torch.einsum("bi,bij,bj->b", error, normal_matrix, error)
    denominator = torch.einsum(
        "bi,bij,bj->b", target, normal_matrix, target
    ).clamp_min(1e-30)
    return numerator / denominator


def frozen_batch(model, family, saved, args, device, z_scale: float):
    batch = sample_icl_batch(
        family,
        args.batch_size,
        saved["m"],
        z_scale,
        saved["f_std"],
        saved["noise_std"],
        device,
    )
    with torch.no_grad():
        equations, observations = model.weak_system(
            batch.f_prompt, batch.u_prompt
        )
        normal_matrix, rhs = normal_equations(
            equations,
            observations,
            model.lam_z,
            model.coefficient_ridge_metric(),
        )
        preconditioner, _ = model.loop_decoder.preconditioner_head(
            equations, normal_matrix
        )
        effective = symmetric_effective_operator(preconditioner, normal_matrix)
        features = effective_spectrum_features(effective, equations.shape[1])
        eigenvalues = torch.linalg.eigvalsh(effective).clamp_min(1e-12)
        target = torch.linalg.solve(normal_matrix, rhs.unsqueeze(-1)).squeeze(-1)
    return (
        normal_matrix,
        rhs,
        preconditioner,
        features,
        eigenvalues[:, 0],
        eigenvalues[:, -1],
        target,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dimension", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--z-scale-min", type=float, default=0.1)
    parser.add_argument("--z-scale-max", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=5.0)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--solver-weight", type=float, default=10.0)
    parser.add_argument("--cvar-fraction", type=float, default=0.05)
    parser.add_argument("--cvar-weight", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--calibration-batch-size", type=int, default=512)
    parser.add_argument("--calibration-quantile", type=float, default=0.99)
    args = parser.parse_args()
    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar_fraction must lie in (0,1]")

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    source = torch.load(args.checkpoint, map_location=device, weights_only=True)
    source_saved = copy.deepcopy(source["args"])
    if source_saved["solver"] != "primal_loop_heavy_ball":
        raise ValueError("source checkpoint must contain the frozen HB trunk")
    set_seed(args.seed)
    family = make_true_family(
        source_saved["d"],
        source_saved["K"],
        source_saved["basis_scale"],
        source_saved["A0_scale"],
        device,
        operator_family=source_saved.get("operator_family", "dense_spd"),
    )
    chebyshev_saved = copy.deepcopy(source_saved)
    chebyshev_saved["solver"] = "primal_loop_chebyshev"
    chebyshev_saved["adaptive_heavy_ball"] = 0
    chebyshev_saved["chebyshev_hidden_dimension"] = args.hidden_dimension
    chebyshev_saved["interval_lower_calibration"] = 1.0
    chebyshev_saved["interval_upper_calibration"] = 1.0
    model = build_model(chebyshev_saved, family, device)
    source_state = source["model"]
    with torch.no_grad():
        model.A0.copy_(source_state["A0"])
        model.Abasis.copy_(source_state["Abasis"])
        model.probes.copy_(source_state["probes"])
    head_prefix = "loop_decoder.preconditioner_head."
    head_state = {
        key.removeprefix(head_prefix): value
        for key, value in source_state.items()
        if key.startswith(head_prefix)
    }
    model.loop_decoder.preconditioner_head.load_state_dict(head_state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    interval_head = model.loop_decoder.interval_head
    assert interval_head is not None
    for parameter in interval_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        interval_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for step in range(1, args.steps + 1):
        z_scale = log_uniform(args.z_scale_min, args.z_scale_max)
        (
            normal_matrix,
            rhs,
            preconditioner,
            features,
            true_min,
            true_max,
            target,
        ) = frozen_batch(model, family, source_saved, args, device, z_scale)
        predicted_min, predicted_max = interval_head(features)
        prediction = exact_chebyshev(
            normal_matrix,
            rhs,
            preconditioner,
            args.depth,
            predicted_min,
            predicted_max,
        )
        h_relative = per_task_h_relative_squared(
            prediction, target, normal_matrix
        )
        tail_count = max(1, math.ceil(args.cvar_fraction * args.batch_size))
        cvar = h_relative.topk(tail_count).values.mean()
        coverage = spectral_interval_coverage_loss(
            predicted_min, predicted_max, true_min, true_max
        )
        endpoint = F.mse_loss(
            torch.log(predicted_min), torch.log(true_min)
        ) + F.mse_loss(torch.log(predicted_max), torch.log(true_max))
        solver = h_relative.mean()
        loss = (
            args.coverage_weight * coverage
            + args.endpoint_weight * endpoint
            + args.solver_weight * solver
            + args.cvar_weight * cvar
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(interval_head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            covered = (
                (predicted_min <= true_min) & (predicted_max >= true_max)
            ).float().mean()
            print(
                json.dumps(
                    {
                        "step": step,
                        "z_scale": z_scale,
                        "loss": loss.item(),
                        "coverage_loss": coverage.item(),
                        "endpoint_loss": endpoint.item(),
                        "solver_h_relative_squared": solver.item(),
                        "cvar_h_relative_squared": cvar.item(),
                        "coverage": covered.item(),
                    }
                ),
                flush=True,
            )

    lower_ratios, upper_ratios = [], []
    original_batch_size = args.batch_size
    args.batch_size = args.calibration_batch_size
    with torch.no_grad():
        for index in range(args.calibration_batches):
            set_seed(args.seed + 100000 + index)
            batch_data = frozen_batch(
                model,
                family,
                source_saved,
                args,
                device,
                args.z_scale_max,
            )
            features, true_min, true_max = batch_data[3], batch_data[4], batch_data[5]
            predicted_min, predicted_max = interval_head(features)
            lower_ratios.append(predicted_min / true_min)
            upper_ratios.append(true_max / predicted_max)
    args.batch_size = original_batch_size
    lower_factor = max(
        1.0,
        torch.quantile(
            torch.cat(lower_ratios), args.calibration_quantile
        ).item(),
    )
    upper_factor = max(
        1.0,
        torch.quantile(
            torch.cat(upper_ratios), args.calibration_quantile
        ).item(),
    )
    model.loop_decoder.interval_lower_calibration = lower_factor
    model.loop_decoder.interval_upper_calibration = upper_factor
    chebyshev_saved["interval_lower_calibration"] = lower_factor
    chebyshev_saved["interval_upper_calibration"] = upper_factor
    chebyshev_saved["chebyshev_interval_training"] = vars(args)
    chebyshev_saved["chebyshev_training_depth"] = args.depth
    print(
        json.dumps(
            {
                "calibration_quantile": args.calibration_quantile,
                "interval_lower_calibration": lower_factor,
                "interval_upper_calibration": upper_factor,
            }
        ),
        flush=True,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "args": chebyshev_saved},
        outdir / "model_final.pt",
    )


if __name__ == "__main__":
    main()
