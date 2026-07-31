#!/usr/bin/env python3
"""Cross-evaluate exact loop controllers with one frozen learned ICL model.

This is the decisive non-inferiority check: dictionary, prompt, one-head Ritz
preconditioner, depth, and HVP budget are shared.  Only the fixed recurrent
controller is changed.  Exact eigenspectra are used solely for oracle controls
and never by the learned Heavy-Ball path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict

import torch

try:
    from .exact_loop_transformer_decoder import (
        effective_spectrum_features,
        normal_equations,
        symmetric_effective_operator,
    )
    from .first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from .pure_icl_parametric_operator_richardson_attention import (
        ParametricOperatorICL,
        make_true_family,
        ridge_forward_solve,
        sample_icl_batch,
        set_seed,
    )
except ImportError:
    from exact_loop_transformer_decoder import (
        effective_spectrum_features,
        normal_equations,
        symmetric_effective_operator,
    )
    from first_principles_decoder_cells import (
        run_chebyshev_state_machine,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from pure_icl_parametric_operator_richardson_attention import (
        ParametricOperatorICL,
        make_true_family,
        ridge_forward_solve,
        sample_icl_batch,
        set_seed,
    )

Tensor = torch.Tensor


def build_model(saved: Dict, true_family, device: torch.device) -> ParametricOperatorICL:
    return ParametricOperatorICL(
        d=saved["d"],
        K=saved["K"],
        R=saved["R"],
        lam_z=saved["lam_z"],
        gamma_u=saved["gamma_u"],
        solver=saved["solver"],
        z_depth=saved["z_depth"],
        learn_dictionary=bool(saved["learn_dictionary"]),
        learn_probes=bool(saved["learn_probes"]),
        true_family=true_family,
        init=saved["init"],
        init_noise=saved["init_noise"],
        heads=saved["heads"],
        d_head=saved["d_head"],
        qk_from=saved["qk_from"],
        use_safe_scale=bool(saved["use_safe_scale"]),
        hb_alpha_init=saved["hb_alpha_init"],
        hb_beta_init=saved["hb_beta_init"],
        subspace_slots=saved["subspace_slots"],
        loop_lmax_bound=saved["loop_lmax_bound"],
        loop_step_init=saved["loop_step_init"],
        chebyshev_hidden_dimension=saved["chebyshev_hidden_dimension"],
        adaptive_heavy_ball=bool(saved.get("adaptive_heavy_ball", 0)),
        interval_lower_calibration=saved.get("interval_lower_calibration", 1.0),
        interval_upper_calibration=saved.get("interval_upper_calibration", 1.0),
        hybrid_residual_threshold=saved.get("hybrid_residual_threshold", 1e-8),
        dictionary_projection=saved.get("dictionary_projection", "none"),
        freeze_A0=bool(saved.get("freeze_A0", 0)),
        covariant_ridge=bool(saved.get("covariant_ridge", 0)),
        loop_preconditioner_head=saved.get(
            "loop_preconditioner_head",
            "coordinate_ritz",
        ),
        prompt_subspace_refinement_steps=saved.get(
            "prompt_subspace_refinement_steps",
            2,
        ),
        chebyshev_interval_policy=saved.get(
            "chebyshev_interval_policy",
            "learned",
        ),
    ).to(device)


def confidence_summary(values: Tensor) -> Dict[str, float]:
    values = values.detach().double().cpu()
    count = values.numel()
    mean = values.mean().item()
    std = values.std(unbiased=True).item() if count > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_halfwidth": 1.96 * std / math.sqrt(max(count, 1)),
        "median": values.median().item(),
        "q95": torch.quantile(values, 0.95).item(),
        "q99": torch.quantile(values, 0.99).item(),
        "max": values.max().item(),
    }


def solve_all(
    model: ParametricOperatorICL,
    batch,
    hybrid_residual_threshold: float,
    hb_depth: int,
    pcg_depth: int,
) -> tuple[Dict[str, Tensor], Tensor, Tensor, Dict[str, Tensor]]:
    if model.loop_decoder is None:
        raise ValueError("checkpoint must contain a primal_loop controller")
    equations, observations = model.weak_system(batch.f_prompt, batch.u_prompt)
    ridge_metric = model.coefficient_ridge_metric()
    normal_matrix, rhs = normal_equations(
        equations, observations, model.lam_z, ridge_metric
    )
    preconditioner, preconditioner_info = model.loop_decoder.preconditioner_head(
        equations,
        normal_matrix,
    )

    def hvp(vector: Tensor) -> Tensor:
        scores = torch.einsum("bmk,bk->bm", equations, vector)
        moment = torch.einsum("bmk,bm->bk", equations, scores)
        ridge_action = (
            vector
            if ridge_metric is None
            else torch.einsum("kl,bl->bk", ridge_metric, vector)
        )
        return moment + model.lam_z * ridge_action

    effective = symmetric_effective_operator(preconditioner, normal_matrix)
    if model.loop_decoder.adaptive_heavy_ball:
        features = preconditioner_info.get("interval_features")
        if features is None:
            features = effective_spectrum_features(effective, equations.shape[1])
        learned_min, learned_max = model.loop_decoder.interval_head(features)
        learned_min = learned_min / model.loop_decoder.interval_lower_calibration
        learned_max = learned_max * model.loop_decoder.interval_upper_calibration
        learned_sqrt_min, learned_sqrt_max = torch.sqrt(learned_min), torch.sqrt(learned_max)
        learned_step = 4.0 / (learned_sqrt_max + learned_sqrt_min).square()
        learned_momentum = (
            (learned_sqrt_max - learned_sqrt_min)
            / (learned_sqrt_max + learned_sqrt_min)
        ).square()
    else:
        learned_step, learned_momentum = model.loop_decoder.heavy_ball_coefficients()
    eigenvalues = torch.linalg.eigvalsh(effective).clamp_min(1e-12)
    spectral_min, spectral_max = eigenvalues[:, 0], eigenvalues[:, -1]
    sqrt_min, sqrt_max = torch.sqrt(spectral_min), torch.sqrt(spectral_max)
    oracle_hb_step = 4.0 / (sqrt_max + sqrt_min).square()
    oracle_hb_momentum = ((sqrt_max - sqrt_min) / (sqrt_max + sqrt_min)).square()
    oracle_richardson_step = 2.0 / (spectral_min + spectral_max)

    zero = learned_momentum.new_zeros(())
    solutions = {
        "exact": torch.linalg.solve(normal_matrix, rhs.unsqueeze(-1)).squeeze(-1),
        "learned_hb": run_heavy_ball_state_machine(
            hvp, rhs, preconditioner, hb_depth, learned_step, learned_momentum
        )[0],
        "oracle_hb": run_heavy_ball_state_machine(
            hvp, rhs, preconditioner, hb_depth, oracle_hb_step, oracle_hb_momentum
        )[0],
        "richardson_same_step": run_heavy_ball_state_machine(
            hvp, rhs, preconditioner, hb_depth, learned_step, zero
        )[0],
        "oracle_richardson": run_heavy_ball_state_machine(
            hvp, rhs, preconditioner, hb_depth, oracle_richardson_step, zero
        )[0],
        "oracle_chebyshev": run_chebyshev_state_machine(
            hvp, rhs, preconditioner, hb_depth, spectral_min, spectral_max
        )[0],
        "pcg": run_pcg_state_machine(hvp, rhs, preconditioner, pcg_depth)[0],
    }
    if "effective_eigenvalues_predicted" in preconditioner_info:
        predicted_spectrum = preconditioner_info[
            "effective_eigenvalues_predicted"
        ]
        solutions["head_exact_chebyshev"] = run_chebyshev_state_machine(
            hvp,
            rhs,
            preconditioner,
            hb_depth,
            predicted_spectrum.amin(dim=-1),
            predicted_spectrum.amax(dim=-1),
        )[0]
    hb_final_residual = rhs - hvp(solutions["learned_hb"])
    preconditioned_final_residual = torch.einsum(
        "bij,bj->bi", preconditioner, hb_final_residual
    )
    preconditioned_rhs = torch.einsum("bij,bj->bi", preconditioner, rhs)
    residual_ratio = torch.einsum(
        "bi,bi->b", hb_final_residual, preconditioned_final_residual
    ) / torch.einsum("bi,bi->b", rhs, preconditioned_rhs).clamp_min(1e-30)
    fallback_mask = residual_ratio > hybrid_residual_threshold
    solutions["certified_hb_pcg"] = torch.where(
        fallback_mask[:, None], solutions["pcg"], solutions["learned_hb"]
    )
    return solutions, normal_matrix, eigenvalues, {
        "learned_step": torch.as_tensor(learned_step).expand_as(spectral_max),
        "learned_momentum": torch.as_tensor(learned_momentum).expand_as(spectral_max),
        "jury_margin": 2.0 * (1.0 + learned_momentum) - learned_step * spectral_max,
        "hb_final_preconditioned_residual_ratio": residual_ratio,
        "hybrid_fallback_mask": fallback_mask,
    }


@torch.no_grad()
def evaluate(args) -> Dict:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved = checkpoint["args"]
    if saved["solver"] != "primal_loop_heavy_ball":
        raise ValueError("the current cross-evaluator expects a trained primal_loop_heavy_ball checkpoint")
    set_seed(saved["seed"])
    true_family = make_true_family(
        saved["d"],
        saved["K"],
        saved["basis_scale"],
        saved["A0_scale"],
        device,
        operator_family=saved.get("operator_family", "dense_spd"),
    )
    model = build_model(saved, true_family, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    collected: Dict[str, Dict[str, list[Tensor]]] = {}
    condition_numbers = []
    jury_margins = []
    hb_residual_ratios = []
    hybrid_fallback_masks = []
    prompt_length = args.prompt_length or saved["m"]
    z_scale = args.z_scale if args.z_scale is not None else saved["z_scale"]
    noise_std = args.noise_std if args.noise_std is not None else saved["noise_std"]
    for repetition in range(args.repetitions):
        set_seed(args.eval_seed + repetition)
        batch = sample_icl_batch(
            true_family,
            args.batch_size,
            prompt_length,
            z_scale,
            saved["f_std"],
            noise_std,
            device,
        )
        hb_depth = args.hb_depth or model.z_depth
        pcg_depth = args.pcg_depth or model.z_depth
        solutions, normal_matrix, eigenvalues, controller_diagnostics = solve_all(
            model,
            batch,
            args.hybrid_residual_threshold,
            hb_depth,
            pcg_depth,
        )
        condition_numbers.append(eigenvalues[:, -1] / eigenvalues[:, 0])
        jury_margins.append(controller_diagnostics["jury_margin"])
        hb_residual_ratios.append(
            controller_diagnostics["hb_final_preconditioned_residual_ratio"]
        )
        hybrid_fallback_masks.append(controller_diagnostics["hybrid_fallback_mask"])
        exact_z = solutions["exact"]
        exact_A = model.A0.unsqueeze(0) + torch.einsum("bk,kij->bij", exact_z, model.Abasis)
        exact_u = ridge_forward_solve(exact_A, batch.f_star, model.gamma_u)
        for name, solution in solutions.items():
            operator = model.A0.unsqueeze(0) + torch.einsum("bk,kij->bij", solution, model.Abasis)
            prediction = ridge_forward_solve(operator, batch.f_star, model.gamma_u)
            z_error = solution - exact_z
            z_denominator = torch.einsum(
                "bk,bkl,bl->b", exact_z, normal_matrix, exact_z
            ).clamp_min(1e-30)
            metrics = {
                "u_mse": (prediction - batch.u_star).square().mean(dim=-1),
                "u_relative": (prediction - batch.u_star).norm(dim=-1)
                / batch.u_star.norm(dim=-1).clamp_min(1e-12),
                "solver_u_mse": (prediction - exact_u).square().mean(dim=-1),
                "solver_z_h_relative_squared": torch.einsum(
                    "bk,bkl,bl->b", z_error, normal_matrix, z_error
                )
                / z_denominator,
            }
            target = collected.setdefault(name, {metric: [] for metric in metrics})
            for metric, values in metrics.items():
                target[metric].append(values)

    all_jury_margins = torch.cat(jury_margins)
    all_hb_residual_ratios = torch.cat(hb_residual_ratios)
    all_hybrid_fallback_masks = torch.cat(hybrid_fallback_masks)
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": args.repetitions * args.batch_size,
        "depth": saved["z_depth"],
        "hb_depth": args.hb_depth or saved["z_depth"],
        "pcg_depth": args.pcg_depth or saved["z_depth"],
        "slots": saved["subspace_slots"],
        "prompt_length": prompt_length,
        "z_scale": z_scale,
        "noise_std": noise_std,
        "effective_condition": confidence_summary(torch.cat(condition_numbers)),
        "heavy_ball_jury_margin": confidence_summary(all_jury_margins),
        "heavy_ball_jury_margin_min": all_jury_margins.min().item(),
        "heavy_ball_jury_violation_rate": (all_jury_margins <= 0).double().mean().item(),
        "heavy_ball_final_preconditioned_residual_ratio": confidence_summary(
            all_hb_residual_ratios
        ),
        "hybrid_residual_threshold": args.hybrid_residual_threshold,
        "hybrid_fallback_rate": all_hybrid_fallback_masks.double().mean().item(),
        "controllers": {},
    }
    for name, metrics in collected.items():
        output["controllers"][name] = {
            metric: confidence_summary(torch.cat(chunks))
            for metric, chunks in metrics.items()
        }
    return output


def write_flat_csv(path: Path, result: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for controller, metrics in result["controllers"].items():
        row = {"controller": controller}
        for metric, summary in metrics.items():
            for statistic, value in summary.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--prompt-length", type=int, default=0)
    parser.add_argument("--z-scale", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--hybrid-residual-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--hb-depth",
        type=int,
        default=0,
        help="HB/polynomial depth; zero uses the checkpoint depth",
    )
    parser.add_argument(
        "--pcg-depth",
        type=int,
        default=0,
        help="PCG depth; zero uses the checkpoint depth",
    )
    args = parser.parse_args()
    result = evaluate(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_flat_csv(outdir / "controllers.csv", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
