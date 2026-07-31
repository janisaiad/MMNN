#!/usr/bin/env python3
"""Fit only the two-scalar spectral policy of a frozen loop-HB decoder.

The encoder, one-head Ritz preconditioner, and exact Heavy-Ball recurrence are
frozen.  A small MLP observes seven fixed prompt statistics and predicts a
spectral interval; the HB coefficients are then given by the analytic minimax
formula.  Eigenvalues supervise training only and are absent at inference.
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
        materialize_preconditioner,
        run_heavy_ball_state_machine,
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
        materialize_preconditioner,
        run_heavy_ball_state_machine,
    )
    from first_principles_inverse_decoder import spectral_interval_coverage_loss
    from pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )

Tensor = torch.Tensor


def relative_h_loss(prediction: Tensor, target: Tensor, normal_matrix: Tensor) -> Tensor:
    error = prediction - target
    numerator = torch.einsum("bk,bkl,bl->b", error, normal_matrix, error)
    denominator = torch.einsum(
        "bk,bkl,bl->b", target, normal_matrix, target
    ).clamp_min(1e-30)
    return (numerator / denominator).mean()


def log_uniform(lower: float, upper: float) -> float:
    draw = random.random()
    return math.exp((1.0 - draw) * math.log(lower) + draw * math.log(upper))


def make_interval_batch(model, true_family, saved, args, device):
    z_scale = log_uniform(args.z_scale_min, args.z_scale_max)
    batch = sample_icl_batch(
        true_family,
        args.batch_size,
        saved["m"],
        z_scale,
        saved["f_std"],
        saved["noise_std"],
        device,
    )
    with torch.no_grad():
        equations, observations = model.weak_system(batch.f_prompt, batch.u_prompt)
        normal_matrix, rhs = normal_equations(
            equations,
            observations,
            model.lam_z,
            model.coefficient_ridge_metric(),
        )
        if model.loop_decoder.matrix_free_preconditioner:
            preconditioner, preconditioner_info = (
                model.loop_decoder.preconditioner_head(
                    equations,
                    model.lam_z,
                    model.coefficient_ridge_metric(),
                )
            )
            features = preconditioner_info["interval_features"]
        else:
            preconditioner, _ = model.loop_decoder.preconditioner_head(
                equations,
                normal_matrix,
            )
            features = None
        dense_preconditioner = materialize_preconditioner(preconditioner)
        effective = symmetric_effective_operator(
            dense_preconditioner,
            normal_matrix,
        )
        if features is None:
            features = effective_spectrum_features(
                effective,
                equations.shape[1],
            )
        eigenvalues = torch.linalg.eigvalsh(effective).clamp_min(1e-12)
        target = torch.linalg.solve(normal_matrix, rhs.unsqueeze(-1)).squeeze(-1)
    return (
        equations,
        normal_matrix,
        rhs,
        preconditioner,
        features,
        eigenvalues[:, 0],
        eigenvalues[:, -1],
        target,
        z_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--z-scale-min", type=float, default=0.1)
    parser.add_argument("--z-scale-max", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=5.0)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--solver-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--calibration-batch-size", type=int, default=512)
    parser.add_argument("--calibration-quantile", type=float, default=0.99)
    parser.add_argument(
        "--certified-output",
        type=int,
        default=0,
        help="save the same weights with the residual-guarded HB-to-PCG controller",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    source = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved = copy.deepcopy(source["args"])
    if saved["solver"] != "primal_loop_heavy_ball":
        raise ValueError("source checkpoint must use primal_loop_heavy_ball")
    set_seed(saved["seed"])
    true_family = make_true_family(
        saved["d"],
        saved["K"],
        saved["basis_scale"],
        saved["A0_scale"],
        device,
        operator_family=saved.get("operator_family", "dense_spd"),
    )
    adaptive_saved = copy.deepcopy(saved)
    adaptive_saved["adaptive_heavy_ball"] = 1
    model = build_model(adaptive_saved, true_family, device)
    missing, unexpected = model.load_state_dict(source["model"], strict=False)
    expected_missing = set()
    if not bool(saved.get("adaptive_heavy_ball", 0)):
        expected_missing = {
            "loop_decoder.interval_head.network.0.weight",
            "loop_decoder.interval_head.network.0.bias",
            "loop_decoder.interval_head.network.2.weight",
            "loop_decoder.interval_head.network.2.bias",
        }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(f"unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    interval_head = model.loop_decoder.interval_head
    assert interval_head is not None
    for parameter in interval_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        interval_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for step in range(1, args.steps + 1):
        (
            equations,
            normal_matrix,
            rhs,
            preconditioner,
            features,
            true_min,
            true_max,
            target,
            z_scale,
        ) = make_interval_batch(model, true_family, saved, args, device)
        predicted_min, predicted_max = interval_head(features)
        sqrt_min, sqrt_max = torch.sqrt(predicted_min), torch.sqrt(predicted_max)
        hb_step = 4.0 / (sqrt_max + sqrt_min).square()
        hb_momentum = ((sqrt_max - sqrt_min) / (sqrt_max + sqrt_min)).square()

        def hvp(vector: Tensor) -> Tensor:
            return torch.einsum("bkl,bl->bk", normal_matrix, vector)

        prediction = run_heavy_ball_state_machine(
            hvp,
            rhs,
            preconditioner,
            model.z_depth,
            hb_step,
            hb_momentum,
        )[0]
        coverage = spectral_interval_coverage_loss(
            predicted_min, predicted_max, true_min, true_max
        )
        endpoint = F.mse_loss(torch.log(predicted_min), torch.log(true_min)) + F.mse_loss(
            torch.log(predicted_max), torch.log(true_max)
        )
        solver = relative_h_loss(prediction, target, normal_matrix)
        loss = (
            args.coverage_weight * coverage
            + args.endpoint_weight * endpoint
            + args.solver_weight * solver
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(interval_head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            covered = ((predicted_min <= true_min) & (predicted_max >= true_max)).float().mean()
            print(
                json.dumps(
                    {
                        "step": step,
                        "z_scale": z_scale,
                        "loss": loss.item(),
                        "coverage_loss": coverage.item(),
                        "endpoint_loss": endpoint.item(),
                        "solver_h_relative_squared": solver.item(),
                        "coverage": covered.item(),
                    }
                ),
                flush=True,
            )

    lower_ratios = []
    upper_ratios = []
    with torch.no_grad():
        original_batch_size = args.batch_size
        args.batch_size = args.calibration_batch_size
        for calibration_index in range(args.calibration_batches):
            set_seed(args.seed + 100000 + calibration_index)
            old_min, old_max = args.z_scale_min, args.z_scale_max
            args.z_scale_min = args.z_scale_max
            (
                _equations,
                _normal_matrix,
                _rhs,
                _preconditioner,
                features,
                true_min,
                true_max,
                _target,
                _z_scale,
            ) = make_interval_batch(model, true_family, saved, args, device)
            args.z_scale_min, args.z_scale_max = old_min, old_max
            predicted_min, predicted_max = interval_head(features)
            lower_ratios.append(predicted_min / true_min)
            upper_ratios.append(true_max / predicted_max)
        args.batch_size = original_batch_size
    lower_factor = max(
        1.0,
        torch.quantile(torch.cat(lower_ratios), args.calibration_quantile).item(),
    )
    upper_factor = max(
        1.0,
        torch.quantile(torch.cat(upper_ratios), args.calibration_quantile).item(),
    )
    model.loop_decoder.interval_lower_calibration = lower_factor
    model.loop_decoder.interval_upper_calibration = upper_factor
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
    adaptive_saved["adaptive_interval_training"] = vars(args)
    adaptive_saved["interval_lower_calibration"] = lower_factor
    adaptive_saved["interval_upper_calibration"] = upper_factor
    if args.certified_output:
        adaptive_saved["solver"] = "primal_loop_certified_hb_pcg"
    torch.save(
        {"model": model.state_dict(), "args": adaptive_saved},
        outdir / "model_final.pt",
    )


if __name__ == "__main__":
    main()
