#!/usr/bin/env python3
"""Predict global Heavy--Ball coefficients from the weighted spectral law.

This is the finite-task estimator of equation (joint-hyperparameter-predictor)
in ``three_controller_encoder_decoder_generalization.tex``.  It never changes
the encoder, dictionary, or one-head preconditioner.  The only optimization is
the exact two-scalar spectral risk implied by the HB residual polynomial.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch

try:
    from .evaluate_trained_loop_controllers import build_model
    from .exact_loop_transformer_decoder import normal_equations
    from .pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )
except ImportError:
    from evaluate_trained_loop_controllers import build_model
    from exact_loop_transformer_decoder import normal_equations
    from pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )

Tensor = torch.Tensor


def hb_residual_polynomial(
    eigenvalues: Tensor,
    depth: int,
    step: Tensor,
    momentum: Tensor,
) -> Tensor:
    """Evaluate the exact HB residual polynomial at every eigenvalue."""

    previous = torch.ones_like(eigenvalues)
    current = torch.ones_like(eigenvalues)
    for _ in range(depth):
        following = (
            (1.0 + momentum - step * eigenvalues) * current
            - momentum * previous
        )
        previous, current = current, following
    return current


def spectral_h_relative_squared(
    eigenvalues: Tensor,
    normalized_energy_weights: Tensor,
    depth: int,
    step: Tensor,
    momentum: Tensor,
) -> Tensor:
    """Per-task H-relative squared error from its positive spectral measure."""

    residual = hb_residual_polynomial(eigenvalues, depth, step, momentum)
    return (normalized_energy_weights * residual.square()).sum(dim=-1)


def coefficients(raw_step: Tensor, raw_momentum: Tensor, lmax: float) -> tuple[Tensor, Tensor]:
    momentum = 0.999 * torch.sigmoid(raw_momentum)
    cap = 2.0 * (1.0 + momentum) / lmax
    step = 0.999 * cap * torch.sigmoid(raw_step)
    return step, momentum


def inverse_parameterization(step: float, momentum: float, lmax: float) -> tuple[float, float]:
    momentum_fraction = min(max(momentum / 0.999, 1e-8), 1.0 - 1e-8)
    raw_momentum = math.log(momentum_fraction / (1.0 - momentum_fraction))
    cap = 2.0 * (1.0 + momentum) / lmax
    step_fraction = min(max(step / (0.999 * cap), 1e-8), 1.0 - 1e-8)
    raw_step = math.log(step_fraction / (1.0 - step_fraction))
    return raw_step, raw_momentum


@torch.no_grad()
def collect_spectral_measure(model, family, saved, args, device) -> tuple[Tensor, Tensor]:
    eigenvalue_chunks = []
    weight_chunks = []
    for _ in range(args.calibration_batches):
        draw = random.random()
        z_scale = math.exp(
            (1.0 - draw) * math.log(args.z_scale_min)
            + draw * math.log(args.z_scale_max)
        )
        batch = sample_icl_batch(
            family,
            args.batch_size,
            saved["m"],
            z_scale,
            saved["f_std"],
            saved["noise_std"],
            device,
        )
        equations, observations = model.weak_system(batch.f_prompt, batch.u_prompt)
        normal_matrix, rhs = normal_equations(
            equations,
            observations,
            model.lam_z,
            model.coefficient_ridge_metric(),
        )
        preconditioner, _ = model.loop_decoder.preconditioner_head(
            equations, normal_matrix
        )
        identity = torch.eye(
            normal_matrix.shape[-1],
            device=device,
            dtype=normal_matrix.dtype,
        )
        factor = torch.linalg.cholesky(preconditioner + 1e-10 * identity)
        effective = factor.transpose(-1, -2) @ normal_matrix @ factor
        effective = 0.5 * (effective + effective.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(effective)
        transformed_rhs = torch.einsum("bji,bj->bi", factor, rhs)
        spectral_rhs = torch.einsum(
            "bji,bj->bi", eigenvectors, transformed_rhs
        )
        spectral_solution = spectral_rhs / eigenvalues.clamp_min(1e-12)
        energy_weights = eigenvalues * spectral_solution.square()
        normalized = energy_weights / energy_weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        eigenvalue_chunks.append(eigenvalues.double().cpu())
        weight_chunks.append(normalized.double().cpu())
    return torch.cat(eigenvalue_chunks), torch.cat(weight_chunks)


def objective(
    eigenvalues: Tensor,
    weights: Tensor,
    depth: int,
    raw_step: Tensor,
    raw_momentum: Tensor,
    lmax: float,
    cvar_fraction: float,
    cvar_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    step, momentum = coefficients(raw_step, raw_momentum, lmax)
    per_task = spectral_h_relative_squared(
        eigenvalues, weights, depth, step, momentum
    )
    tail_count = max(1, math.ceil(cvar_fraction * per_task.numel()))
    cvar = per_task.topk(tail_count).values.mean()
    value = per_task.mean() + cvar_weight * cvar
    return value, per_task.mean(), cvar, step, momentum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--z-scale-min", type=float, default=0.1)
    parser.add_argument("--z-scale-max", type=float, default=1.0)
    parser.add_argument("--cvar-fraction", type=float, default=0.05)
    parser.add_argument("--cvar-weight", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=37000)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    source = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved = copy.deepcopy(source["args"])
    if saved["solver"] != "primal_loop_heavy_ball":
        raise ValueError("checkpoint must use a global primal_loop_heavy_ball")
    if bool(saved.get("adaptive_heavy_ball", 0)):
        raise ValueError("adaptive HB has no global two-scalar spectral predictor")
    set_seed(args.seed)
    family = make_true_family(
        saved["d"],
        saved["K"],
        saved["basis_scale"],
        saved["A0_scale"],
        device,
        operator_family=saved.get("operator_family", "dense_spd"),
    )
    model = build_model(saved, family, device)
    model.load_state_dict(source["model"])
    model.eval()
    eigenvalues, weights = collect_spectral_measure(
        model, family, saved, args, device
    )
    lmax = float(saved["loop_lmax_bound"])
    empirical_min = eigenvalues.min().item()
    empirical_max = eigenvalues.max().item()
    if empirical_max >= lmax:
        raise ValueError(
            f"spectral maximum {empirical_max:.6g} violates lmax bound {lmax:.6g}"
        )
    sqrt_min, sqrt_max = math.sqrt(empirical_min), math.sqrt(empirical_max)
    minimax_step = 4.0 / (sqrt_max + sqrt_min) ** 2
    minimax_momentum = ((sqrt_max - sqrt_min) / (sqrt_max + sqrt_min)) ** 2

    with torch.no_grad():
        current_step, current_momentum = model.loop_decoder.heavy_ball_coefficients()
    starts = [
        (current_step.item(), current_momentum.item()),
        (minimax_step, minimax_momentum),
        (0.5 * minimax_step, 0.25 * minimax_momentum),
        (0.8 * minimax_step, 0.5 * minimax_momentum),
        (1.1 * minimax_step, min(0.95, 1.25 * minimax_momentum)),
    ][: args.restarts]
    best = None
    for restart, (initial_step, initial_momentum) in enumerate(starts):
        raw_values = inverse_parameterization(
            initial_step, initial_momentum, lmax
        )
        raw_step = torch.tensor(raw_values[0], dtype=torch.float64, requires_grad=True)
        raw_momentum = torch.tensor(
            raw_values[1], dtype=torch.float64, requires_grad=True
        )
        optimizer = torch.optim.Adam([raw_step, raw_momentum], lr=args.lr)
        for _ in range(args.steps):
            value, _, _, _, _ = objective(
                eigenvalues,
                weights,
                args.depth,
                raw_step,
                raw_momentum,
                lmax,
                args.cvar_fraction,
                args.cvar_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
        with torch.no_grad():
            values = objective(
                eigenvalues,
                weights,
                args.depth,
                raw_step,
                raw_momentum,
                lmax,
                args.cvar_fraction,
                args.cvar_weight,
            )
            candidate = {
                "restart": restart,
                "objective": values[0].item(),
                "mean_h_relative_squared": values[1].item(),
                "cvar_h_relative_squared": values[2].item(),
                "step": values[3].item(),
                "momentum": values[4].item(),
                "raw_step": raw_step.item(),
                "raw_momentum": raw_momentum.item(),
            }
        if best is None or candidate["objective"] < best["objective"]:
            best = candidate
    assert best is not None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "calibration_tasks": eigenvalues.shape[0],
        "coefficient_dimension": eigenvalues.shape[1],
        "depth": args.depth,
        "spectral_min": empirical_min,
        "spectral_max": empirical_max,
        "spectral_lmax_bound": lmax,
        "current_step": current_step.item(),
        "current_momentum": current_momentum.item(),
        "minimax_step": minimax_step,
        "minimax_momentum": minimax_momentum,
        "predicted": best,
        "args": vars(args),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    predicted_state = copy.deepcopy(source["model"])
    predicted_state["loop_decoder.raw_step"] = torch.tensor(
        best["raw_step"],
        dtype=predicted_state["loop_decoder.raw_step"].dtype,
        device=predicted_state["loop_decoder.raw_step"].device,
    )
    predicted_state["loop_decoder.raw_momentum"] = torch.tensor(
        best["raw_momentum"],
        dtype=predicted_state["loop_decoder.raw_momentum"].dtype,
        device=predicted_state["loop_decoder.raw_momentum"].device,
    )
    saved["spectral_hyperparameter_prediction"] = summary
    torch.save(
        {"model": predicted_state, "args": saved},
        outdir / "model_predicted.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
