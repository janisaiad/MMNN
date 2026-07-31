#!/usr/bin/env python3
"""Paired elliptic comparison of robust HB, learned Chebyshev, and PCG.

Both checkpoints must share the physical encoder and one-head Ritz
preconditioner exactly.  Every controller then receives the same prompts,
normal equations, preconditioner, and right-hand sides.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

try:
    from .evaluate_trained_loop_controllers import build_model, confidence_summary
    from .exact_loop_transformer_decoder import (
        effective_spectrum_features,
        normal_equations,
        symmetric_effective_operator,
    )
    from .first_principles_decoder_cells import (
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from .pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        ridge_forward_solve,
        sample_icl_batch,
        set_seed,
    )
    from .train_elliptic_chebyshev_interval import exact_chebyshev
except ImportError:
    from evaluate_trained_loop_controllers import build_model, confidence_summary
    from exact_loop_transformer_decoder import (
        effective_spectrum_features,
        normal_equations,
        symmetric_effective_operator,
    )
    from first_principles_decoder_cells import (
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
    from pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        ridge_forward_solve,
        sample_icl_batch,
        set_seed,
    )
    from train_elliptic_chebyshev_interval import exact_chebyshev

Tensor = torch.Tensor


def assert_shared_trunk(hb_model, chebyshev_model) -> None:
    named_pairs = [
        ("A0", hb_model.A0, chebyshev_model.A0),
        ("Abasis", hb_model.Abasis, chebyshev_model.Abasis),
        ("probes", hb_model.probes, chebyshev_model.probes),
    ]
    for name, left, right in named_pairs:
        if not torch.equal(left, right):
            raise ValueError(f"checkpoint trunks differ at {name}")
    hb_head = hb_model.loop_decoder.preconditioner_head.state_dict()
    chebyshev_head = chebyshev_model.loop_decoder.preconditioner_head.state_dict()
    if hb_head.keys() != chebyshev_head.keys():
        raise ValueError("preconditioner heads have different state keys")
    for key in hb_head:
        if not torch.equal(hb_head[key], chebyshev_head[key]):
            raise ValueError(f"preconditioner heads differ at {key}")


def controller_metrics(
    model,
    solution: Tensor,
    exact_solution: Tensor,
    normal_matrix: Tensor,
    batch,
) -> dict[str, Tensor]:
    operator = model.A0.unsqueeze(0) + torch.einsum(
        "bk,kij->bij", solution, model.Abasis
    )
    prediction = ridge_forward_solve(operator, batch.f_star, model.gamma_u)
    exact_operator = model.A0.unsqueeze(0) + torch.einsum(
        "bk,kij->bij", exact_solution, model.Abasis
    )
    exact_prediction = ridge_forward_solve(
        exact_operator, batch.f_star, model.gamma_u
    )
    coefficient_error = solution - exact_solution
    denominator = torch.einsum(
        "bi,bij,bj->b", exact_solution, normal_matrix, exact_solution
    ).clamp_min(1e-30)
    return {
        "u_mse": (prediction - batch.u_star).square().mean(dim=-1),
        "u_relative": (prediction - batch.u_star).norm(dim=-1)
        / batch.u_star.norm(dim=-1).clamp_min(1e-12),
        "solver_u_mse": (prediction - exact_prediction).square().mean(dim=-1),
        "solver_z_h_relative_squared": torch.einsum(
            "bi,bij,bj->b", coefficient_error, normal_matrix, coefficient_error
        )
        / denominator,
    }


@torch.no_grad()
def calibrate_constant_interval(
    model,
    family,
    saved,
    args,
    device: torch.device,
) -> tuple[float, float]:
    """Fit a prompt-independent interval on tasks disjoint from evaluation."""

    minima, maxima = [], []
    for index in range(args.constant_calibration_batches):
        set_seed(args.constant_calibration_seed + index)
        batch = sample_icl_batch(
            family,
            args.batch_size,
            saved["m"],
            args.z_scale,
            saved["f_std"],
            args.noise_std,
            device,
        )
        equations, observations = model.weak_system(
            batch.f_prompt, batch.u_prompt
        )
        normal_matrix, _ = normal_equations(
            equations,
            observations,
            model.lam_z,
            model.coefficient_ridge_metric(),
        )
        preconditioner, _ = model.loop_decoder.preconditioner_head(
            equations, normal_matrix
        )
        eigenvalues = torch.linalg.eigvalsh(
            symmetric_effective_operator(preconditioner, normal_matrix)
        ).clamp_min(1e-12)
        minima.append(eigenvalues[:, 0])
        maxima.append(eigenvalues[:, -1])
    constant_min = torch.quantile(
        torch.cat(minima), args.constant_lower_quantile
    ).item()
    constant_max = torch.quantile(
        torch.cat(maxima), args.constant_upper_quantile
    ).item()
    return constant_min, constant_max


@torch.no_grad()
def evaluate(args) -> dict:
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    hb_checkpoint = torch.load(
        args.hb_checkpoint, map_location=device, weights_only=True
    )
    chebyshev_checkpoint = torch.load(
        args.chebyshev_checkpoint, map_location=device, weights_only=True
    )
    hb_saved = hb_checkpoint["args"]
    chebyshev_saved = chebyshev_checkpoint["args"]
    if hb_saved["solver"] != "primal_loop_heavy_ball":
        raise ValueError("HB checkpoint has the wrong controller")
    if chebyshev_saved["solver"] != "primal_loop_chebyshev":
        raise ValueError("Chebyshev checkpoint has the wrong controller")
    family = make_true_family(
        hb_saved["d"],
        hb_saved["K"],
        hb_saved["basis_scale"],
        hb_saved["A0_scale"],
        device,
        operator_family=hb_saved.get("operator_family", "dense_spd"),
    )
    hb_model = build_model(hb_saved, family, device)
    hb_model.load_state_dict(hb_checkpoint["model"])
    chebyshev_model = build_model(chebyshev_saved, family, device)
    chebyshev_model.load_state_dict(chebyshev_checkpoint["model"])
    hb_model.eval()
    chebyshev_model.eval()
    assert_shared_trunk(hb_model, chebyshev_model)
    chebyshev_depths = sorted(
        {int(value) for value in args.chebyshev_depths.split(",")}
    )
    constant_min, constant_max = calibrate_constant_interval(
        hb_model, family, hb_saved, args, device
    )

    collected: dict[str, dict[str, list[Tensor]]] = {}
    condition_numbers, raw_coverage, calibrated_coverage = [], [], []
    shuffled_coverage, constant_coverage = [], []
    lower_ratios, upper_ratios = [], []
    for repetition in range(args.repetitions):
        set_seed(args.eval_seed + repetition)
        batch = sample_icl_batch(
            family,
            args.batch_size,
            hb_saved["m"],
            args.z_scale,
            hb_saved["f_std"],
            args.noise_std,
            device,
        )
        equations, observations = hb_model.weak_system(
            batch.f_prompt, batch.u_prompt
        )
        ridge_metric = hb_model.coefficient_ridge_metric()
        normal_matrix, rhs = normal_equations(
            equations, observations, hb_model.lam_z, ridge_metric
        )
        preconditioner, _ = hb_model.loop_decoder.preconditioner_head(
            equations, normal_matrix
        )
        effective = symmetric_effective_operator(preconditioner, normal_matrix)
        eigenvalues = torch.linalg.eigvalsh(effective).clamp_min(1e-12)
        true_min, true_max = eigenvalues[:, 0], eigenvalues[:, -1]
        condition_numbers.append(true_max / true_min)
        features = effective_spectrum_features(effective, equations.shape[1])
        raw_min, raw_max = chebyshev_model.loop_decoder.interval_head(features)
        predicted_min = (
            raw_min / chebyshev_model.loop_decoder.interval_lower_calibration
        )
        predicted_max = (
            raw_max * chebyshev_model.loop_decoder.interval_upper_calibration
        )
        shuffled_features = features.roll(shifts=1, dims=0)
        shuffled_min, shuffled_max = chebyshev_model.loop_decoder.interval_head(
            shuffled_features
        )
        shuffled_min = (
            shuffled_min
            / chebyshev_model.loop_decoder.interval_lower_calibration
        )
        shuffled_max = (
            shuffled_max
            * chebyshev_model.loop_decoder.interval_upper_calibration
        )
        constant_minimum = true_min.new_full(true_min.shape, constant_min)
        constant_maximum = true_max.new_full(true_max.shape, constant_max)
        raw_coverage.append((raw_min <= true_min) & (raw_max >= true_max))
        calibrated_coverage.append(
            (predicted_min <= true_min) & (predicted_max >= true_max)
        )
        shuffled_coverage.append(
            (shuffled_min <= true_min) & (shuffled_max >= true_max)
        )
        constant_coverage.append(
            (constant_minimum <= true_min) & (constant_maximum >= true_max)
        )
        lower_ratios.append(predicted_min / true_min)
        upper_ratios.append(true_max / predicted_max)
        exact_solution = torch.linalg.solve(
            normal_matrix, rhs.unsqueeze(-1)
        ).squeeze(-1)

        def hvp(vector: Tensor) -> Tensor:
            return torch.einsum("bij,bj->bi", normal_matrix, vector)

        hb_step, hb_momentum = hb_model.loop_decoder.heavy_ball_coefficients()
        solutions = {
            "exact": exact_solution,
            f"robust_hb_{args.hb_depth}": run_heavy_ball_state_machine(
                hvp,
                rhs,
                preconditioner,
                args.hb_depth,
                hb_step,
                hb_momentum,
            )[0],
            f"pcg_{args.pcg_depth}": run_pcg_state_machine(
                hvp, rhs, preconditioner, args.pcg_depth
            )[0],
        }
        for depth in chebyshev_depths:
            solutions[f"learned_chebyshev_{depth}"] = exact_chebyshev(
                normal_matrix,
                rhs,
                preconditioner,
                depth,
                predicted_min,
                predicted_max,
            )
            solutions[f"shuffled_chebyshev_{depth}"] = exact_chebyshev(
                normal_matrix,
                rhs,
                preconditioner,
                depth,
                shuffled_min,
                shuffled_max,
            )
            solutions[f"constant_chebyshev_{depth}"] = exact_chebyshev(
                normal_matrix,
                rhs,
                preconditioner,
                depth,
                constant_minimum,
                constant_maximum,
            )
            solutions[f"oracle_chebyshev_{depth}"] = exact_chebyshev(
                normal_matrix,
                rhs,
                preconditioner,
                depth,
                true_min,
                true_max,
            )
        oracle_richardson_step = 2.0 / (true_min + true_max)
        solutions[f"oracle_richardson_{args.hb_depth}"] = (
            run_heavy_ball_state_machine(
                hvp,
                rhs,
                preconditioner,
                args.hb_depth,
                oracle_richardson_step,
                rhs.new_zeros(()),
            )[0]
        )
        for name, solution in solutions.items():
            metrics = controller_metrics(
                hb_model,
                solution,
                exact_solution,
                normal_matrix,
                batch,
            )
            target = collected.setdefault(
                name, {metric: [] for metric in metrics}
            )
            for metric, values in metrics.items():
                target[metric].append(values)

    all_raw_coverage = torch.cat(raw_coverage)
    all_calibrated_coverage = torch.cat(calibrated_coverage)
    all_shuffled_coverage = torch.cat(shuffled_coverage)
    all_constant_coverage = torch.cat(constant_coverage)
    return {
        "hb_checkpoint": str(Path(args.hb_checkpoint).resolve()),
        "chebyshev_checkpoint": str(Path(args.chebyshev_checkpoint).resolve()),
        "shared_trunk_verified": True,
        "examples": args.batch_size * args.repetitions,
        "z_scale": args.z_scale,
        "noise_std": args.noise_std,
        "hb_depth": args.hb_depth,
        "pcg_depth": args.pcg_depth,
        "chebyshev_depths": chebyshev_depths,
        "effective_condition": confidence_summary(torch.cat(condition_numbers)),
        "interval": {
            "constant_min": constant_min,
            "constant_max": constant_max,
            "raw_coverage": all_raw_coverage.double().mean().item(),
            "calibrated_coverage": all_calibrated_coverage.double().mean().item(),
            "shuffled_coverage": all_shuffled_coverage.double().mean().item(),
            "constant_coverage": all_constant_coverage.double().mean().item(),
            "lower_ratio": confidence_summary(torch.cat(lower_ratios)),
            "upper_ratio": confidence_summary(torch.cat(upper_ratios)),
        },
        "controllers": {
            name: {
                metric: confidence_summary(torch.cat(chunks))
                for metric, chunks in metrics.items()
            }
            for name, metrics in collected.items()
        },
    }


def write_csv(path: Path, result: dict) -> None:
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
    parser.add_argument("--hb-checkpoint", required=True)
    parser.add_argument("--chebyshev-checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=300000)
    parser.add_argument("--z-scale", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--hb-depth", type=int, default=32)
    parser.add_argument("--pcg-depth", type=int, default=16)
    parser.add_argument("--chebyshev-depths", default="16,24,32")
    parser.add_argument("--constant-calibration-batches", type=int, default=8)
    parser.add_argument("--constant-calibration-seed", type=int, default=600000)
    parser.add_argument("--constant-lower-quantile", type=float, default=0.01)
    parser.add_argument("--constant-upper-quantile", type=float, default=0.99)
    args = parser.parse_args()
    result = evaluate(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(outdir / "controllers.csv", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
