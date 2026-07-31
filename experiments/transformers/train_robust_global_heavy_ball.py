#!/usr/bin/env python3
"""Fit only two global Heavy--Ball scalars on a frozen ICL decoder.

The objective is deliberately tail-aware: it combines mean and CVaR
preconditioned energy error.  Stability is enforced structurally by the
decoder's uniform spectral upper bound and audited with the exact Jury margin
during training.  Eigenvalues are diagnostics only; inference uses two global
scalars and no interval MLP.
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
    from .exact_loop_transformer_decoder import normal_equations, symmetric_effective_operator
    from .first_principles_decoder_cells import run_heavy_ball_state_machine
    from .pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )
except ImportError:
    from evaluate_trained_loop_controllers import build_model
    from exact_loop_transformer_decoder import normal_equations, symmetric_effective_operator
    from first_principles_decoder_cells import run_heavy_ball_state_machine
    from pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )

Tensor = torch.Tensor


def log_uniform(lower: float, upper: float) -> float:
    draw = random.random()
    return math.exp((1.0 - draw) * math.log(lower) + draw * math.log(upper))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--z-scale-min", type=float, default=0.1)
    parser.add_argument("--z-scale-max", type=float, default=1.0)
    parser.add_argument("--spectral-lmax-bound", type=float, default=3.0)
    parser.add_argument("--cvar-fraction", type=float, default=0.05)
    parser.add_argument("--cvar-weight", type=float, default=10.0)
    parser.add_argument("--jury-margin-target", type=float, default=0.1)
    parser.add_argument("--jury-weight", type=float, default=100.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    if not 0.0 < args.cvar_fraction <= 1.0:
        raise ValueError("cvar_fraction must lie in (0,1]")
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    source = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved = copy.deepcopy(source["args"])
    if saved["solver"] != "primal_loop_heavy_ball":
        raise ValueError("source checkpoint must use primal_loop_heavy_ball")
    if bool(saved.get("adaptive_heavy_ball", 0)):
        raise ValueError("source checkpoint must use global HB scalars")
    saved["loop_lmax_bound"] = args.spectral_lmax_bound
    # ``step_init`` is only a constructor seed; the checkpoint raw parameter
    # is loaded immediately afterwards.  Keep the seed inside the new bound.
    constructor_cap = 2.0 * (1.0 + saved["hb_beta_init"]) / args.spectral_lmax_bound
    saved["loop_step_init"] = min(
        saved["loop_step_init"], 0.5 * 0.999 * constructor_cap
    )
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    raw_step = model.loop_decoder.raw_step
    raw_momentum = model.loop_decoder.raw_momentum
    assert raw_step is not None and raw_momentum is not None
    raw_step.requires_grad_(True)
    raw_momentum.requires_grad_(True)
    optimizer = torch.optim.Adam([raw_step, raw_momentum], lr=args.lr)

    for step_index in range(1, args.steps + 1):
        z_scale = log_uniform(args.z_scale_min, args.z_scale_max)
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
            target = torch.linalg.solve(
                normal_matrix, rhs.unsqueeze(-1)
            ).squeeze(-1)
            spectral_max = torch.linalg.eigvalsh(
                symmetric_effective_operator(preconditioner, normal_matrix)
            )[:, -1]

        def hvp(vector: Tensor) -> Tensor:
            return torch.einsum("bij,bj->bi", normal_matrix, vector)

        hb_step, momentum = model.loop_decoder.heavy_ball_coefficients()
        prediction = run_heavy_ball_state_machine(
            hvp,
            rhs,
            preconditioner,
            args.depth,
            hb_step,
            momentum,
        )[0]
        error = prediction - target
        numerator = torch.einsum(
            "bi,bij,bj->b", error, normal_matrix, error
        )
        denominator = torch.einsum(
            "bi,bij,bj->b", target, normal_matrix, target
        ).clamp_min(1e-30)
        relative_energy = numerator / denominator
        tail_count = max(1, math.ceil(args.cvar_fraction * args.batch_size))
        cvar = relative_energy.topk(tail_count).values.mean()
        jury_margin = 2.0 * (1.0 + momentum) - hb_step * spectral_max
        jury_penalty = torch.relu(
            args.jury_margin_target - jury_margin
        ).square().mean()
        loss = (
            relative_energy.mean()
            + args.cvar_weight * cvar
            + args.jury_weight * jury_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step_index == 1 or step_index % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step_index,
                        "z_scale": z_scale,
                        "loss": loss.item(),
                        "mean_h_relative_squared": relative_energy.mean().item(),
                        "cvar_h_relative_squared": cvar.item(),
                        "jury_margin_min": jury_margin.min().item(),
                        "hb_step": hb_step.item(),
                        "hb_momentum": momentum.item(),
                    }
                ),
                flush=True,
            )

    saved["robust_global_hb_training"] = vars(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": saved}, outdir / "model_final.pt")


if __name__ == "__main__":
    main()
