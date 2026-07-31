#!/usr/bin/env python3
"""Compare global inverse-covariance and prompt-conditioned preconditioners.

The benchmark implements the randomly rotated structured (RRS) distinction in
the latent PDE coefficient space.  Each prompt normal system is conjugated by
an independent Haar rotation.  A context-independent population covariance
then contains no directional information, whereas the equivariant one-head
preconditioner can still route its finite correction budget to the prompt's
slow eigenspace.

Every preconditioner receives its own depth-specific, stable Heavy--Ball
coefficients fitted to independent draws from the known generative law.  The
evaluation also reports exact-interval Chebyshev and PCG, so an improvement
cannot be attributed to an unfair choice of outer-loop hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import torch

try:
    from .evaluate_trained_loop_controllers import build_model
    from .exact_loop_transformer_decoder import normal_equations
    from .first_principles_decoder_cells import run_pcg_state_machine
    from .predict_heavy_ball_hyperparameters import (
        inverse_parameterization,
        objective,
    )
    from .predict_pde_law_hyperparameters import (
        chebyshev_task_risk,
        summarize_task_risk,
        task_risk,
    )
    from .pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )
except ImportError:
    from evaluate_trained_loop_controllers import build_model
    from exact_loop_transformer_decoder import normal_equations
    from first_principles_decoder_cells import run_pcg_state_machine
    from predict_heavy_ball_hyperparameters import (
        inverse_parameterization,
        objective,
    )
    from predict_pde_law_hyperparameters import (
        chebyshev_task_risk,
        summarize_task_risk,
        task_risk,
    )
    from pure_icl_parametric_operator_richardson_attention import (
        make_true_family,
        sample_icl_batch,
        set_seed,
    )

Tensor = torch.Tensor
PreconditionerFactory = Callable[[Tensor], Tensor]


def parse_depths(value: str) -> list[int]:
    depths = sorted({int(item) for item in value.split(",") if item.strip()})
    if not depths or depths[0] <= 0:
        raise ValueError("depths must be positive comma-separated integers")
    return depths


def haar_orthogonal(
    batch: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    """Draw Haar orthogonal matrices by sign-corrected Gaussian QR."""

    gaussian = torch.randn(
        batch,
        dimension,
        dimension,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    orthogonal, triangular = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diagonal(triangular, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return orthogonal * signs.unsqueeze(-2)


def rotate_system(normal: Tensor, rhs: Tensor, rotation: Tensor) -> tuple[Tensor, Tensor]:
    """Apply ``H -> Q^T H Q`` and ``c -> Q^T c`` to a batch of systems."""

    rotated_normal = (
        rotation.transpose(-1, -2) @ normal @ rotation
    )
    rotated_rhs = torch.einsum("bji,bj->bi", rotation, rhs)
    return rotated_normal, rotated_rhs


@torch.no_grad()
def draw_normal_systems(
    model,
    family,
    saved: dict,
    tasks: int,
    batch_size: int,
    z_scale: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    normals = []
    right_sides = []
    remaining = tasks
    while remaining:
        count = min(batch_size, remaining)
        batch = sample_icl_batch(
            family,
            count,
            saved["m"],
            z_scale,
            saved["f_std"],
            saved["noise_std"],
            device,
        )
        equations, observations = model.weak_system(
            batch.f_prompt,
            batch.u_prompt,
        )
        normal, rhs = normal_equations(
            equations,
            observations,
            model.lam_z,
            model.coefficient_ridge_metric(),
        )
        normals.append(normal)
        right_sides.append(rhs)
        remaining -= count
    return torch.cat(normals), torch.cat(right_sides)


def spectral_measure(
    normal: Tensor,
    rhs: Tensor,
    preconditioner: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return effective eigenvalues and normalized solution-energy weights."""

    dimension = normal.shape[-1]
    identity = torch.eye(
        dimension,
        device=normal.device,
        dtype=normal.dtype,
    )
    factor = torch.linalg.cholesky(preconditioner + 1e-12 * identity)
    effective = factor.transpose(-1, -2) @ normal @ factor
    effective = 0.5 * (effective + effective.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(effective)
    transformed_rhs = torch.einsum("bji,bj->bi", factor, rhs)
    spectral_rhs = torch.einsum("bji,bj->bi", eigenvectors, transformed_rhs)
    spectral_solution = spectral_rhs / eigenvalues.clamp_min(1e-14)
    energy = eigenvalues * spectral_solution.square()
    weights = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return eigenvalues, weights


def fit_heavy_ball(
    eigenvalues: Tensor,
    weights: Tensor,
    *,
    depth: int,
    cvar_fraction: float,
    cvar_weight: float,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    """Fit only the two exact HB scalars to a law-level spectral measure."""

    spectral_min = eigenvalues.min().item()
    spectral_max = eigenvalues.max().item()
    lmax = 1.01 * spectral_max
    sqrt_min = math.sqrt(spectral_min)
    sqrt_max = math.sqrt(spectral_max)
    minimax_step = 4.0 / (sqrt_max + sqrt_min) ** 2
    minimax_momentum = ((sqrt_max - sqrt_min) / (sqrt_max + sqrt_min)) ** 2
    richardson_step = 2.0 / (spectral_min + spectral_max)
    starts = [
        (minimax_step, minimax_momentum),
        (richardson_step, 0.0),
        (0.8 * minimax_step, 0.5 * minimax_momentum),
        (1.1 * minimax_step, min(0.95, 1.25 * minimax_momentum)),
    ]
    best: dict[str, float] | None = None
    for restart, (initial_step, initial_momentum) in enumerate(starts):
        raw_values = inverse_parameterization(
            initial_step,
            initial_momentum,
            lmax,
        )
        raw_step = torch.tensor(
            raw_values[0],
            dtype=eigenvalues.dtype,
            device=eigenvalues.device,
            requires_grad=True,
        )
        raw_momentum = torch.tensor(
            raw_values[1],
            dtype=eigenvalues.dtype,
            device=eigenvalues.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam(
            [raw_step, raw_momentum],
            lr=learning_rate,
        )
        for _ in range(steps):
            value, _, _, _, _ = objective(
                eigenvalues,
                weights,
                depth,
                raw_step,
                raw_momentum,
                lmax,
                cvar_fraction,
                cvar_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
        with torch.no_grad():
            value, mean, cvar, step, momentum = objective(
                eigenvalues,
                weights,
                depth,
                raw_step,
                raw_momentum,
                lmax,
                cvar_fraction,
                cvar_weight,
            )
            candidate = {
                "restart": float(restart),
                "objective": value.item(),
                "mean_risk": mean.item(),
                "cvar_risk": cvar.item(),
                "step": step.item(),
                "momentum": momentum.item(),
                "spectral_min": spectral_min,
                "spectral_max": spectral_max,
                "stability_lmax": lmax,
            }
        if best is None or candidate["objective"] < best["objective"]:
            best = candidate
    assert best is not None
    return best


def confidence_summary(values: Tensor) -> dict[str, float]:
    values = values.detach().double().cpu()
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "q95": torch.quantile(values, 0.95).item(),
        "q99": torch.quantile(values, 0.99).item(),
        "max": values.max().item(),
    }


def pcg_relative_energy_error(
    normal: Tensor,
    rhs: Tensor,
    preconditioner: Tensor,
    depth: int,
) -> Tensor:
    def hvp(vector: Tensor) -> Tensor:
        return torch.einsum("bij,bj->bi", normal, vector)

    estimate = run_pcg_state_machine(
        hvp,
        rhs,
        preconditioner,
        depth,
    )[0]
    exact = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
    error = estimate - exact
    numerator = torch.einsum("bi,bij,bj->b", error, normal, error)
    denominator = torch.einsum("bi,bi->b", exact, rhs).clamp_min(1e-30)
    return numerator / denominator


@torch.no_grad()
def prompt_head(model, normal: Tensor) -> Tensor:
    dummy = normal.new_empty(normal.shape[0], 0, normal.shape[-1])
    return model.loop_decoder.preconditioner_head(dummy, normal)[0]


def make_preconditioners(
    model,
    calibration_normal: Tensor,
) -> dict[str, PreconditionerFactory]:
    dimension = calibration_normal.shape[-1]
    identity = torch.eye(
        dimension,
        device=calibration_normal.device,
        dtype=calibration_normal.dtype,
    )
    # Haar conjugation gives E[Q^T H Q] = E[tr(H)/K] I exactly.  Thus this is
    # the population inverse-covariance preconditioner, not a weak empirical
    # baseline chosen for convenience.
    population_scale = (
        torch.diagonal(calibration_normal, dim1=-2, dim2=-1)
        .sum(dim=-1)
        .mean()
        / dimension
    )

    def global_covariance(normal: Tensor) -> Tensor:
        return (identity / population_scale).expand(normal.shape[0], -1, -1)

    def jacobi(normal: Tensor) -> Tensor:
        diagonal = torch.diagonal(normal, dim1=-2, dim2=-1).clamp_min(1e-12)
        return torch.diag_embed(diagonal.reciprocal())

    def learned_prompt(normal: Tensor) -> Tensor:
        return prompt_head(model, normal.to(dtype=next(model.parameters()).dtype)).to(
            dtype=normal.dtype
        )

    def direct_inverse(normal: Tensor) -> Tensor:
        return torch.linalg.solve(
            normal,
            identity.expand(normal.shape[0], -1, -1),
        )

    return {
        "global_covariance": global_covariance,
        "jacobi": jacobi,
        "learned_prompt_head": learned_prompt,
        "direct_inverse": direct_inverse,
    }


def empirical_isotropy(normal: Tensor) -> float:
    mean = normal.mean(dim=0)
    dimension = mean.shape[-1]
    isotropic = torch.trace(mean) / dimension * torch.eye(
        dimension,
        device=mean.device,
        dtype=mean.dtype,
    )
    return (
        torch.linalg.matrix_norm(mean - isotropic)
        / torch.linalg.matrix_norm(isotropic).clamp_min(1e-30)
    ).item()


def evaluate_preconditioner(
    name: str,
    factory: PreconditionerFactory,
    calibration_normal: Tensor,
    calibration_rhs: Tensor,
    evaluation_normal: Tensor,
    evaluation_rhs: Tensor,
    args,
) -> dict:
    calibration_preconditioner = factory(calibration_normal)
    evaluation_preconditioner = factory(evaluation_normal)
    calibration_eigenvalues, calibration_weights = spectral_measure(
        calibration_normal,
        calibration_rhs,
        calibration_preconditioner,
    )
    evaluation_eigenvalues, evaluation_weights = spectral_measure(
        evaluation_normal,
        evaluation_rhs,
        evaluation_preconditioner,
    )
    fitted = fit_heavy_ball(
        calibration_eigenvalues,
        calibration_weights,
        depth=args.hb_depth,
        cvar_fraction=args.cvar_fraction,
        cvar_weight=args.cvar_weight,
        steps=args.optimizer_steps,
        learning_rate=args.learning_rate,
    )
    step = evaluation_eigenvalues.new_tensor(fitted["step"])
    momentum = evaluation_eigenvalues.new_tensor(fitted["momentum"])
    hb_risk = task_risk(
        evaluation_eigenvalues,
        evaluation_weights,
        args.hb_depth,
        step,
        momentum,
    )
    chebyshev_risk = chebyshev_task_risk(
        evaluation_eigenvalues,
        evaluation_weights,
        args.hb_depth,
        evaluation_eigenvalues[:, 0],
        evaluation_eigenvalues[:, -1],
    )
    pcg_risks = {
        depth: pcg_relative_energy_error(
            evaluation_normal,
            evaluation_rhs,
            evaluation_preconditioner,
            depth,
        )
        for depth in parse_depths(args.pcg_depths)
    }
    condition = evaluation_eigenvalues[:, -1] / evaluation_eigenvalues[:, 0]
    jury_margin = (
        2.0 * (1.0 + momentum)
        - step * evaluation_eigenvalues[:, -1]
    )
    return {
        "name": name,
        "hb_fit": fitted,
        "condition_number": confidence_summary(condition),
        "hb_h_relative_squared": summarize_task_risk(
            hb_risk,
            args.cvar_fraction,
        ),
        "chebyshev_h_relative_squared": summarize_task_risk(
            chebyshev_risk,
            args.cvar_fraction,
        ),
        "pcg_h_relative_squared": {
            str(depth): summarize_task_risk(risk, args.cvar_fraction)
            for depth, risk in pcg_risks.items()
        },
        "evaluation_spectral_max": evaluation_eigenvalues.max().item(),
        "jury_margin_min": jury_margin.min().item(),
        "jury_violation_rate": (jury_margin <= 0).double().mean().item(),
    }


def run(args) -> dict:
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved = checkpoint["args"]
    if saved.get("loop_preconditioner_head") != "equivariant_ritz_softmax":
        raise ValueError("checkpoint must use the equivariant Ritz softmax head")
    set_seed(saved["seed"])
    family = make_true_family(
        saved["d"],
        saved["K"],
        saved["basis_scale"],
        saved["A0_scale"],
        device,
        operator_family=saved.get("operator_family", "dense_spd"),
    )
    model = build_model(saved, family, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    set_seed(args.seed)
    calibration_normal, calibration_rhs = draw_normal_systems(
        model,
        family,
        saved,
        args.calibration_tasks,
        args.batch_size,
        args.z_scale,
        device,
    )
    evaluation_normal, evaluation_rhs = draw_normal_systems(
        model,
        family,
        saved,
        args.evaluation_tasks,
        args.batch_size,
        args.z_scale,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(args.rotation_seed)
    calibration_rotation = haar_orthogonal(
        args.calibration_tasks,
        saved["K"],
        device=device,
        dtype=calibration_normal.dtype,
        generator=generator,
    )
    evaluation_rotation = haar_orthogonal(
        args.evaluation_tasks,
        saved["K"],
        device=device,
        dtype=evaluation_normal.dtype,
        generator=generator,
    )
    unrotated_evaluation_normal = evaluation_normal
    rotated_calibration = rotate_system(
        calibration_normal,
        calibration_rhs,
        calibration_rotation,
    )
    rotated_evaluation = rotate_system(
        evaluation_normal,
        evaluation_rhs,
        evaluation_rotation,
    )
    calibration_normal, calibration_rhs = (
        item.double() for item in rotated_calibration
    )
    evaluation_normal, evaluation_rhs = (
        item.double() for item in rotated_evaluation
    )
    factories = make_preconditioners(model, calibration_normal)
    controllers = {
        name: evaluate_preconditioner(
            name,
            factory,
            calibration_normal,
            calibration_rhs,
            evaluation_normal,
            evaluation_rhs,
            args,
        )
        for name, factory in factories.items()
    }

    with torch.no_grad():
        original = prompt_head(model, unrotated_evaluation_normal)
        rotated = prompt_head(model, rotated_evaluation[0])
        expected = (
            evaluation_rotation.transpose(-1, -2)
            @ original
            @ evaluation_rotation
        )
        gauge_error = (
            torch.linalg.matrix_norm(rotated - expected, dim=(-2, -1))
            / torch.linalg.matrix_norm(expected, dim=(-2, -1)).clamp_min(1e-30)
        )

    global_result = controllers["global_covariance"]
    prompt_result = controllers["learned_prompt_head"]
    ratios = {}
    for solver in ("hb", "chebyshev"):
        key = f"{solver}_h_relative_squared"
        ratios[f"prompt_over_global_{solver}_mean_risk"] = (
            prompt_result[key]["mean_risk"]
            / global_result[key]["mean_risk"]
        )
        ratios[f"prompt_over_global_{solver}_cvar_risk"] = (
            prompt_result[key]["cvar_risk"]
            / global_result[key]["cvar_risk"]
        )
    for depth in parse_depths(args.pcg_depths):
        prompt_risk = prompt_result["pcg_h_relative_squared"][str(depth)]
        global_risk = global_result["pcg_h_relative_squared"][str(depth)]
        ratios[f"prompt_over_global_pcg{depth}_mean_risk"] = (
            prompt_risk["mean_risk"] / global_risk["mean_risk"]
        )
        ratios[f"prompt_over_global_pcg{depth}_cvar_risk"] = (
            prompt_risk["cvar_risk"] / global_risk["cvar_risk"]
        )

    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_seed": saved["seed"],
        "device": str(device),
        "coefficient_dimension": saved["K"],
        "calibration_tasks": args.calibration_tasks,
        "evaluation_tasks": args.evaluation_tasks,
        "hb_depth": args.hb_depth,
        "pcg_depths": parse_depths(args.pcg_depths),
        "rrs_population_identity": "E[Q^T H Q] = E[tr(H)/K] I",
        "rotated_calibration_mean_relative_anisotropy": empirical_isotropy(
            calibration_normal
        ),
        "prompt_head_gauge_relative_error": confidence_summary(gauge_error),
        "controllers": controllers,
        "ratios": ratios,
        "scope": (
            "Latent PDE RRS benchmark. Direct inverse is an accuracy ceiling; "
            "the current exact-eigenspectrum head is not claimed to beat its "
            "construction cost."
        ),
        "args": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-tasks", type=int, default=2048)
    parser.add_argument("--evaluation-tasks", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--z-scale", type=float, default=0.5)
    parser.add_argument("--hb-depth", type=int, default=10)
    parser.add_argument("--pcg-depths", default="4,8")
    parser.add_argument("--cvar-fraction", type=float, default=0.05)
    parser.add_argument("--cvar-weight", type=float, default=1.0)
    parser.add_argument("--optimizer-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--seed", type=int, default=82000)
    parser.add_argument("--rotation-seed", type=int, default=83000)
    args = parser.parse_args()
    result = run(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for name, controller in result["controllers"].items():
        rows.append(
            {
                "checkpoint_seed": result["checkpoint_seed"],
                "preconditioner": name,
                "condition_mean": controller["condition_number"]["mean"],
                "hb_mean_risk": controller["hb_h_relative_squared"]["mean_risk"],
                "chebyshev_mean_risk": controller[
                    "chebyshev_h_relative_squared"
                ]["mean_risk"],
                "hb_step": controller["hb_fit"]["step"],
                "hb_momentum": controller["hb_fit"]["momentum"],
                "jury_violation_rate": controller["jury_violation_rate"],
                **{
                    f"pcg{depth}_mean_risk": controller[
                        "pcg_h_relative_squared"
                    ][str(depth)]["mean_risk"]
                    for depth in result["pcg_depths"]
                },
            }
        )
    with (outdir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
