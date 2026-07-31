#!/usr/bin/env python3
"""One attention head learns a preconditioner; exact algebra runs HB or PCG.

There is deliberately no MLP.  The head sees normalized weak-equation tokens,
returns a small prompt-dependent subspace, and a fixed SPD formula turns that
subspace into an inverse preconditioner.  Residuals, ridge terms, the
preconditioner application, and HeavyBall memory are hard-coded linear
relations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from .constructive_weakform_richardson_transformer import (
        TaskConfig,
        _heavy_ball_solve,
        _pcg_solve,
        build_preconditioner,
        run_constructive_loop,
        sample_weak_batch,
        weak_normal_hvp,
    )
    from .first_principles_decoder_cells import (
        fixed_prompt_linear_attention_hvp,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )
except ImportError:
    from constructive_weakform_richardson_transformer import (
        TaskConfig,
        _heavy_ball_solve,
        _pcg_solve,
        build_preconditioner,
        run_constructive_loop,
        sample_weak_batch,
        weak_normal_hvp,
    )
    from first_principles_decoder_cells import (
        fixed_prompt_linear_attention_hvp,
        run_heavy_ball_state_machine,
        run_pcg_state_machine,
    )

Tensor = torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, row: Dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def scalar_logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


class OneHeadSpectralPreconditioner(nn.Module):
    """One softmax head with several queries, followed by a fixed SPD map."""

    def __init__(
        self,
        dimension: int,
        head_dimension: int,
        slots: int,
        max_strength: float,
        strength_init: float,
        base_preconditioner: str = "jacobi",
        base_blocks: int = 2,
        strength_scaling: str = "fixed",
        reference_prompt_length: int = 32,
        slot_orthogonalization: str = "independent",
        correction_mode: str = "positive",
        subspace_refinement_steps: int = 0,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.head_dimension = head_dimension
        self.slots = slots
        self.max_strength = min(max_strength, 0.999)
        self.base_preconditioner = base_preconditioner
        self.base_blocks = base_blocks
        self.strength_scaling = strength_scaling
        self.reference_prompt_length = reference_prompt_length
        self.slot_orthogonalization = slot_orthogonalization
        self.correction_mode = correction_mode
        self.subspace_refinement_steps = subspace_refinement_steps
        self.key = nn.Linear(2 * dimension, head_dimension, bias=False)
        self.value = nn.Linear(2 * dimension, dimension, bias=False)
        self.slot_queries = nn.Parameter(
            torch.randn(slots, head_dimension) / math.sqrt(head_dimension)
        )
        if correction_mode == "positive":
            initial_fraction = strength_init / self.max_strength
            raw_strength = scalar_logit(initial_fraction)
        elif correction_mode == "signed":
            initial_fraction = min(max(strength_init / self.max_strength, -0.999), 0.999)
            raw_strength = math.atanh(initial_fraction)
        elif correction_mode == "ritz":
            raw_strength = None
        else:
            raise ValueError(f"unknown correction mode {correction_mode}")
        if raw_strength is None:
            self.register_parameter("raw_strength", None)
        else:
            self.raw_strength = nn.Parameter(torch.full((slots,), raw_strength))
        self._initialize_identity_geometry()

    def _initialize_identity_geometry(self) -> None:
        with torch.no_grad():
            self.key.weight.zero_()
            diagonal = min(2 * self.dimension, self.head_dimension)
            self.key.weight[:diagonal, :diagonal] = torch.eye(diagonal)
            self.value.weight.zero_()
            self.value.weight[:, self.dimension :] = torch.eye(self.dimension)

    def _base_inverse(self, normal_matrix: Tensor) -> Tensor:
        if self.base_preconditioner == "jacobi":
            diagonal = torch.diagonal(normal_matrix, dim1=-2, dim2=-1).clamp_min(1e-10)
            return torch.diag_embed(diagonal.reciprocal())
        if self.base_preconditioner == "scalar_mean":
            diagonal_mean = torch.diagonal(
                normal_matrix,
                dim1=-2,
                dim2=-1,
            ).mean(dim=-1).clamp_min(1e-10)
            eye = torch.eye(
                self.dimension,
                device=normal_matrix.device,
                dtype=normal_matrix.dtype,
            ).expand(normal_matrix.shape[0], -1, -1)
            return diagonal_mean.reciprocal()[:, None, None] * eye
        if self.base_preconditioner != "block_jacobi":
            raise ValueError(f"unknown base preconditioner {self.base_preconditioner}")
        base_inverse = torch.zeros_like(normal_matrix)
        coordinate_blocks = torch.tensor_split(
            torch.arange(self.dimension, device=normal_matrix.device),
            max(1, self.base_blocks),
        )
        for indices in coordinate_blocks:
            block = normal_matrix[:, indices[:, None], indices[None, :]]
            block_inverse = torch.linalg.inv(block)
            base_inverse[:, indices[:, None], indices[None, :]] = block_inverse
        return base_inverse

    def forward(self, equations: Tensor, normal_matrix: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        base_inverse = self._base_inverse(normal_matrix)
        base_factor = torch.linalg.cholesky(base_inverse)
        # H is an exact known moment of the weak tokens.  Compute it outside
        # the learned head, then let the head learn only how to combine its
        # normalized coordinate rows into useful spectral directions.
        coordinate_tokens = torch.einsum(
            "bki,bkl,blj->bij",
            base_factor,
            normal_matrix,
            base_factor,
        )
        coordinate_identity = torch.eye(
            self.dimension,
            device=normal_matrix.device,
            dtype=normal_matrix.dtype,
        ).expand(normal_matrix.shape[0], -1, -1)
        head_tokens = torch.cat([coordinate_tokens, coordinate_identity], dim=-1)
        keys = self.key(head_tokens)
        scores = torch.einsum("sd,bkd->bsk", self.slot_queries, keys)
        scores = scores / math.sqrt(self.head_dimension)
        attention = torch.softmax(scores, dim=-1)
        values = self.value(head_tokens)
        raw_directions = torch.einsum("bsk,bkj->bsj", attention, values).transpose(1, 2)
        if self.slot_orthogonalization == "independent":
            directions = raw_directions / raw_directions.norm(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
        elif self.slot_orthogonalization == "qr":
            directions = torch.linalg.qr(raw_directions, mode="reduced").Q
        else:
            raise ValueError(
                f"unknown slot orthogonalization {self.slot_orthogonalization}"
            )
        for _ in range(self.subspace_refinement_steps):
            directions = torch.linalg.qr(
                torch.einsum("bkl,bls->bks", coordinate_tokens, directions),
                mode="reduced",
            ).Q

        eye = torch.eye(
            self.dimension,
            device=normal_matrix.device,
            dtype=normal_matrix.dtype,
        ).expand(normal_matrix.shape[0], -1, -1)
        if self.correction_mode == "ritz":
            projected_operator = torch.einsum(
                "bks,bkl,blt->bst",
                directions,
                coordinate_tokens,
                directions,
            )
            slot_eye = torch.eye(
                self.slots,
                device=normal_matrix.device,
                dtype=normal_matrix.dtype,
            ).expand(normal_matrix.shape[0], -1, -1)
            projected_inverse = torch.linalg.inv(projected_operator)
            slot_correction = projected_inverse - slot_eye
            low_rank = torch.einsum(
                "bks,bst,blt->bkl",
                directions,
                slot_correction,
                directions,
            )
            normalized_inverse = eye + low_rank
            batch_strengths = torch.linalg.eigvalsh(projected_inverse) - 1.0
        else:
            if self.correction_mode == "positive":
                base_strengths = self.max_strength * torch.sigmoid(self.raw_strength)
            else:
                base_strengths = self.max_strength * torch.tanh(self.raw_strength)
            if self.strength_scaling == "fixed":
                strength_scale = 1.0
            elif self.strength_scaling == "inverse_prompt":
                strength_scale = self.reference_prompt_length / equations.shape[1]
            else:
                raise ValueError(f"unknown strength scaling {self.strength_scaling}")
            strengths = (base_strengths * strength_scale).clamp(
                min=-self.max_strength,
                max=self.max_strength,
            )
            batch_strengths = strengths[None, :].expand(normal_matrix.shape[0], -1)
            low_rank = torch.einsum(
                "bks,bs,bls->bkl",
                directions,
                batch_strengths,
                directions,
            )
            # The scalar normalization does not change exact-arithmetic PCG;
            # it keeps the represented inverse uniformly scaled.
            normalized_inverse = (eye + low_rank) / (
                1.0 + batch_strengths.abs().sum(dim=-1)[:, None, None]
            )
        preconditioner = torch.einsum(
            "bik,bkl,bjl->bij",
            base_factor,
            normalized_inverse,
            base_factor,
        )
        info = {
            "attention": attention,
            "directions": directions,
            "strengths": batch_strengths,
            "normalized_inverse": normalized_inverse,
            "base_inverse": base_inverse,
        }
        return preconditioner, info


class OneHeadSymmetricKernelPreconditioner(nn.Module):
    """Tied Q=K self-attention, symmetrically normalized into an SPD map."""

    def __init__(
        self,
        dimension: int,
        head_dimension: int,
        max_strength: float,
        strength_init: float,
        diagnostic_slots: int,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.head_dimension = head_dimension
        self.max_strength = min(max_strength, 0.999)
        self.diagnostic_slots = diagnostic_slots
        self.embedding = nn.Linear(dimension, head_dimension, bias=False)
        initial_fraction = strength_init / self.max_strength
        self.raw_strength = nn.Parameter(torch.tensor(scalar_logit(initial_fraction)))
        with torch.no_grad():
            self.embedding.weight.zero_()
            diagonal = min(dimension, head_dimension)
            self.embedding.weight[:diagonal, :diagonal] = torch.eye(diagonal)

    def forward(self, _equations: Tensor, normal_matrix: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        diagonal = torch.diagonal(normal_matrix, dim1=-2, dim2=-1).clamp_min(1e-10)
        diagonal_inv_sqrt = diagonal.rsqrt()
        coordinate_tokens = (
            diagonal_inv_sqrt[:, :, None]
            * normal_matrix
            * diagonal_inv_sqrt[:, None, :]
        )
        embeddings = self.embedding(coordinate_tokens)
        scores = torch.einsum("bkd,bjd->bkj", embeddings, embeddings)
        scores = scores / math.sqrt(self.head_dimension)
        # A batchwise scalar shift preserves symmetry and cancels in the
        # subsequent normalization while preventing exponential overflow.
        scores = scores - scores.amax(dim=(-2, -1), keepdim=True)
        kernel = torch.exp(scores)
        kernel_degree = kernel.sum(dim=-1).clamp_min(1e-10)
        symmetric_attention = (
            kernel_degree.rsqrt()[:, :, None]
            * kernel
            * kernel_degree.rsqrt()[:, None, :]
        )
        strength = self.max_strength * torch.sigmoid(self.raw_strength)
        eye = torch.eye(
            self.dimension,
            device=normal_matrix.device,
            dtype=normal_matrix.dtype,
        ).expand(normal_matrix.shape[0], -1, -1)
        normalized_inverse = (1.0 - strength) * eye + strength * symmetric_attention
        preconditioner = (
            diagonal_inv_sqrt[:, :, None]
            * normalized_inverse
            * diagonal_inv_sqrt[:, None, :]
        )
        with torch.no_grad():
            _, eigenvectors = torch.linalg.eigh(normalized_inverse)
            directions = eigenvectors[:, :, -self.diagnostic_slots :]
        info = {
            "attention": symmetric_attention,
            "directions": directions,
            "strengths": strength.reshape(1),
            "normalized_inverse": normalized_inverse,
            "base_inverse": torch.diag_embed(diagonal.reciprocal()),
        }
        return preconditioner, info


class StructuredOneHeadHeavyBall(nn.Module):
    """Attention learns P(C); the selected solver cell is exact and explicit."""

    def __init__(
        self,
        dimension: int,
        depth: int,
        head_dimension: int,
        slots: int,
        max_strength: float,
        strength_init: float,
        head_mode: str,
        spectral_lmax_bound: float,
        step_init: float,
        momentum_init: float,
        solver_cell: str = "heavy_ball",
        base_preconditioner: str = "jacobi",
        base_blocks: int = 2,
        strength_scaling: str = "fixed",
        reference_prompt_length: int = 32,
        slot_orthogonalization: str = "independent",
        correction_mode: str = "positive",
        subspace_refinement_steps: int = 0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.spectral_lmax_bound = spectral_lmax_bound
        self.solver_cell = solver_cell
        if head_mode == "symmetric_kernel":
            if base_preconditioner != "jacobi":
                raise ValueError("symmetric_kernel currently supports only a Jacobi base")
            self.preconditioner_head = OneHeadSymmetricKernelPreconditioner(
                dimension,
                head_dimension,
                max_strength,
                strength_init,
                diagnostic_slots=slots,
            )
        elif head_mode == "slots":
            self.preconditioner_head = OneHeadSpectralPreconditioner(
                dimension,
                head_dimension,
                slots,
                max_strength,
                strength_init,
                base_preconditioner,
                base_blocks,
                strength_scaling,
                reference_prompt_length,
                slot_orthogonalization,
                correction_mode,
                subspace_refinement_steps,
            )
        else:
            raise ValueError(f"unknown head mode {head_mode}")
        if solver_cell == "heavy_ball":
            beta_fraction = momentum_init / 0.999
            self.raw_momentum = nn.Parameter(torch.tensor(scalar_logit(beta_fraction)))
            step_cap = 2.0 * (1.0 + momentum_init) / spectral_lmax_bound
            step_fraction = step_init / (0.999 * step_cap)
            self.raw_step = nn.Parameter(torch.tensor(scalar_logit(step_fraction)))
        elif solver_cell != "pcg":
            raise ValueError(f"unknown solver cell {solver_cell}")

    def coefficients(self) -> Tuple[Tensor, Tensor]:
        if self.solver_cell != "heavy_ball":
            raise RuntimeError("PCG computes its scalar coefficients exactly at each iteration")
        momentum = 0.999 * torch.sigmoid(self.raw_momentum)
        stable_step_cap = 2.0 * (1.0 + momentum) / self.spectral_lmax_bound
        step = 0.999 * stable_step_cap * torch.sigmoid(self.raw_step)
        return step, momentum

    def forward(self, batch, cfg: TaskConfig) -> Tuple[Tensor, Dict[str, Tensor]]:
        preconditioner, info = self.preconditioner_head(batch.G, batch.H)
        noise_precision = 1.0 / cfg.noise_var
        prior_precision = 1.0 / cfg.prior_var

        def prompt_hvp(vector: Tensor) -> Tensor:
            return fixed_prompt_linear_attention_hvp(
                batch.G,
                vector,
                noise_precision,
                prior_precision,
            )

        if self.solver_cell == "heavy_ball":
            step, momentum = self.coefficients()
            prediction, _, _ = run_heavy_ball_state_machine(
                prompt_hvp,
                batch.c,
                preconditioner,
                self.depth,
                step,
                momentum,
            )
            info.update({"step": step, "momentum": momentum})
        else:
            prediction, _, _ = run_pcg_state_machine(
                prompt_hvp,
                batch.c,
                preconditioner,
                self.depth,
            )
        info["preconditioner"] = preconditioner
        return prediction, info


@torch.no_grad()
def spectral_diagnostics(
    info: Dict[str, Tensor],
    normal_matrix: Tensor,
    target: Tensor,
) -> Dict[str, float]:
    preconditioner = info["preconditioner"]
    dimension = normal_matrix.shape[-1]
    eye = torch.eye(dimension, device=normal_matrix.device, dtype=normal_matrix.dtype)
    chol = torch.linalg.cholesky(preconditioner + 1e-10 * eye)
    symmetric = torch.einsum("bki,bkl,blj->bij", chol, normal_matrix, chol)
    symmetric = 0.5 * (symmetric + symmetric.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(symmetric).clamp_min(1e-12)

    base_factor = torch.linalg.cholesky(info["base_inverse"] + 1e-10 * eye)
    normalized_matrix = torch.einsum(
        "bki,bkl,blj->bij",
        base_factor,
        normal_matrix,
        base_factor,
    )
    _, eigenvectors = torch.linalg.eigh(normalized_matrix)
    slots = info["directions"].shape[-1]
    slow_space = eigenvectors[:, :, :slots]
    fast_space = eigenvectors[:, :, -slots:]
    learned_space = torch.linalg.qr(info["directions"], mode="reduced").Q
    overlap = torch.linalg.matrix_norm(
        torch.einsum("bki,bkj->bij", slow_space, learned_space),
        ord="fro",
        dim=(-2, -1),
    ).pow(2) / slots
    fast_overlap = torch.linalg.matrix_norm(
        torch.einsum("bki,bkj->bij", fast_space, learned_space),
        ord="fro",
        dim=(-2, -1),
    ).pow(2) / slots
    normalized_target = torch.linalg.solve_triangular(
        base_factor,
        target.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    target_coordinates = torch.einsum("bks,bk->bs", learned_space, normalized_target)
    target_overlap = target_coordinates.pow(2).sum(dim=-1) / normalized_target.pow(2).sum(
        dim=-1
    ).clamp_min(1e-12)
    return {
        "effective_kappa": (eigenvalues[:, -1] / eigenvalues[:, 0]).mean().item(),
        "effective_lmax": eigenvalues[:, -1].mean().item(),
        "effective_lmax_max": eigenvalues[:, -1].max().item(),
        "slow_space_overlap": overlap.mean().item(),
        "fast_space_overlap": fast_overlap.mean().item(),
        "target_space_overlap": target_overlap.mean().item(),
    }


def spectral_subspace_loss(
    info: Dict[str, Tensor],
    normal_matrix: Tensor,
    target: str,
) -> Tensor:
    """Principal-angle supervision used only during preconditioner training."""
    dimension = normal_matrix.shape[-1]
    eye = torch.eye(dimension, device=normal_matrix.device, dtype=normal_matrix.dtype)
    base_factor = torch.linalg.cholesky(info["base_inverse"] + 1e-10 * eye)
    normalized_matrix = torch.einsum(
        "bki,bkl,blj->bij",
        base_factor,
        normal_matrix,
        base_factor,
    )
    with torch.no_grad():
        _, eigenvectors = torch.linalg.eigh(normalized_matrix)
        slots = info["directions"].shape[-1]
        if target == "fast":
            target_space = eigenvectors[:, :, -slots:]
        elif target == "slow":
            target_space = eigenvectors[:, :, :slots]
        else:
            raise ValueError(f"unknown spectral target {target}")
    learned_space = torch.linalg.qr(info["directions"], mode="reduced").Q
    overlap = torch.linalg.matrix_norm(
        torch.einsum("bki,bkj->bij", target_space, learned_space),
        ord="fro",
        dim=(-2, -1),
    ).square() / slots
    return 1.0 - overlap.mean()


def effective_log_condition_loss(
    preconditioner: Tensor,
    normal_matrix: Tensor,
) -> Tensor:
    """Differentiable HB-oriented loss on the symmetric effective spectrum."""

    dimension = normal_matrix.shape[-1]
    eye = torch.eye(
        dimension,
        device=normal_matrix.device,
        dtype=normal_matrix.dtype,
    )
    factor = torch.linalg.cholesky(preconditioner + 1e-10 * eye)
    symmetric = torch.einsum(
        "bki,bkl,blj->bij",
        factor,
        normal_matrix,
        factor,
    )
    symmetric = 0.5 * (symmetric + symmetric.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(symmetric).clamp_min(1e-12)
    return (torch.log(eigenvalues[:, -1]) - torch.log(eigenvalues[:, 0])).mean()


def build_config(args) -> TaskConfig:
    return TaskConfig(
        K=args.K,
        prompt_len=args.prompt_len,
        prior_var=args.prior_var,
        noise_var=args.noise_var,
        design=args.design,
        cond=args.cond,
        dtype=args.dtype,
        pde_state_dim=args.pde_state_dim,
    )


def train(args) -> None:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = build_config(args)
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    model = StructuredOneHeadHeavyBall(
        dimension=args.K,
        depth=args.depth,
        head_dimension=args.head_dimension,
        slots=args.slots,
        max_strength=args.max_strength,
        strength_init=args.strength_init,
        head_mode=args.head_mode,
        spectral_lmax_bound=args.spectral_lmax_bound,
        step_init=args.step_init,
        momentum_init=args.momentum_init,
        solver_cell=args.solver_cell,
        base_preconditioner=args.base_preconditioner,
        base_blocks=args.base_blocks,
        strength_scaling=args.strength_scaling,
        reference_prompt_length=args.reference_prompt_length,
        slot_orthogonalization=args.slot_orthogonalization,
        correction_mode=args.correction_mode,
        subspace_refinement_steps=args.subspace_refinement_steps,
    ).to(device=device, dtype=model_dtype)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    if args.head_checkpoint:
        checkpoint = torch.load(args.head_checkpoint, map_location=device, weights_only=True)
        head_state = {
            name.removeprefix("preconditioner_head."): value
            for name, value in checkpoint["model"].items()
            if name.startswith("preconditioner_head.")
        }
        model.preconditioner_head.load_state_dict(head_state)
    initial_head = copy.deepcopy(model.preconditioner_head).eval()
    for parameter in initial_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = output / "training.csv"
    set_seed(args.eval_seed)
    evaluation = sample_weak_batch(args.eval_batch_size, cfg, device)
    set_seed(args.seed + 1)
    training_prompt_grid = (
        parse_int_grid(args.train_prompt_grid) if args.train_prompt_grid else [cfg.prompt_len]
    )
    training_condition_grid = (
        parse_float_grid(args.train_cond_grid) if args.train_cond_grid else [cfg.cond]
    )
    best_evaluation_mse = math.inf

    for training_step in range(1, args.steps + 1):
        training_cfg = replace(
            cfg,
            prompt_len=random.choice(training_prompt_grid),
            cond=random.choice(training_condition_grid),
        )
        batch = sample_weak_batch(args.batch_size, training_cfg, device)
        prediction, info = model(batch, training_cfg)
        error = prediction - batch.beta_post
        numerator = torch.einsum("bk,bkl,bl->b", error, batch.H, error)
        denominator = torch.einsum(
            "bk,bkl,bl->b", batch.beta_post, batch.H, batch.beta_post
        ).clamp_min(1e-10)
        solver_loss = (numerator / denominator).mean()
        if args.spectral_loss_weight > 0.0:
            subspace_loss = spectral_subspace_loss(
                info,
                batch.H,
                args.spectral_target,
            )
        else:
            subspace_loss = torch.zeros_like(solver_loss)
        if args.condition_loss_weight > 0.0:
            condition_loss = effective_log_condition_loss(
                info["preconditioner"],
                batch.H,
            )
        else:
            condition_loss = torch.zeros_like(solver_loss)
        loss = (
            solver_loss
            + args.spectral_loss_weight * subspace_loss
            + args.condition_loss_weight * condition_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if training_step == 1 or training_step % args.log_every == 0:
            with torch.no_grad():
                learned, eval_info = model(evaluation, cfg)
                diagnostics = spectral_diagnostics(
                    eval_info,
                    evaluation.H,
                    evaluation.beta_post,
                )
                initial_preconditioner, _ = initial_head(evaluation.G, evaluation.H)
                jacobi, _ = build_preconditioner(
                    evaluation.H,
                    evaluation.eigvals,
                    "jacobi",
                    heads=1,
                    d_head=1,
                    eta_mode="opt",
                    eta_multiplier=1.0,
                    spectral_order="small",
                )
                noise_precision = 1.0 / cfg.noise_var
                prior_precision = 1.0 / cfg.prior_var

                def evaluation_hvp(vector: Tensor) -> Tensor:
                    return weak_normal_hvp(
                        evaluation.G,
                        vector,
                        noise_precision,
                        prior_precision,
                    )

                if args.solver_cell == "heavy_ball":
                    learned_coeff_jacobi, _, _ = _heavy_ball_solve(
                        evaluation_hvp,
                        evaluation.c,
                        jacobi,
                        args.depth,
                        eval_info["step"],
                        eval_info["momentum"],
                    )
                    initial_head_final_coeff, _, _ = _heavy_ball_solve(
                        evaluation_hvp,
                        evaluation.c,
                        initial_preconditioner,
                        args.depth,
                        eval_info["step"],
                        eval_info["momentum"],
                    )
                    step_size = eval_info["step"].item()
                    momentum = eval_info["momentum"].item()
                else:
                    learned_coeff_jacobi, _, _ = _pcg_solve(
                        evaluation_hvp,
                        evaluation.c,
                        jacobi,
                        args.depth,
                    )
                    initial_head_final_coeff, _, _ = _pcg_solve(
                        evaluation_hvp,
                        evaluation.c,
                        initial_preconditioner,
                        args.depth,
                    )
                    step_size = math.nan
                    momentum = math.nan
                learned_preconditioner_pcg, _, _ = _pcg_solve(
                    evaluation_hvp,
                    evaluation.c,
                    eval_info["preconditioner"],
                    args.depth,
                )
                baselines = {
                    solver: run_constructive_loop(
                        evaluation,
                        cfg,
                        depth=args.depth,
                        precond="jacobi",
                        solver=solver,
                    )
                    for solver in ["richardson", "heavy_ball", "chebyshev", "pcg"]
                }
                evaluation_mse = torch.mean((learned - evaluation.beta_post).pow(2)).item()
                row: Dict[str, float | int | str] = {
                    "training_step": training_step,
                    "loss": loss.item(),
                    "solver_loss": solver_loss.item(),
                    "spectral_loss": subspace_loss.item(),
                    "condition_loss": condition_loss.item(),
                    "learned_mse": evaluation_mse,
                    "learned_coeff_jacobi_mse": torch.mean(
                        (learned_coeff_jacobi - evaluation.beta_post).pow(2)
                    ).item(),
                    "initial_head_final_coeff_mse": torch.mean(
                        (initial_head_final_coeff - evaluation.beta_post).pow(2)
                    ).item(),
                    "learned_preconditioner_pcg_mse": torch.mean(
                        (learned_preconditioner_pcg - evaluation.beta_post).pow(2)
                    ).item(),
                    "step_size": step_size,
                    "momentum": momentum,
                    "strength_mean": eval_info["strengths"].mean().item(),
                    **diagnostics,
                }
                for solver, result in baselines.items():
                    row[f"{solver}_mse"] = torch.mean(
                        (result.beta_L - evaluation.beta_post).pow(2)
                    ).item()
                append_csv(metrics_path, row)
                print(json.dumps(row, sort_keys=True))
                if evaluation_mse < best_evaluation_mse:
                    best_evaluation_mse = evaluation_mse
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "args": vars(args),
                            "task": asdict(cfg),
                            "training_step": training_step,
                            "evaluation_mse": evaluation_mse,
                        },
                        output / "model_best.pt",
                    )

    torch.save(
        {"model": model.state_dict(), "args": vars(args), "task": asdict(cfg)},
        output / "model_final.pt",
    )


def parse_float_grid(raw: str) -> list[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def parse_int_grid(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def benchmark_runtime_ms(function, repeats: int, device: torch.device) -> float:
    for _ in range(min(3, repeats)):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(1000.0 * (time.perf_counter() - start))
    return float(np.median(timings))


def build_oracle_ritz_preconditioner(
    normal_matrix: Tensor,
    eigenvalues: Tensor,
    base_preconditioner: str,
    base_blocks: int,
    slots: int,
    target: str,
) -> Tensor:
    if base_preconditioner == "scalar_mean":
        diagonal_mean = torch.diagonal(
            normal_matrix,
            dim1=-2,
            dim2=-1,
        ).mean(dim=-1).clamp_min(1e-10)
        dimension = normal_matrix.shape[-1]
        eye = torch.eye(
            dimension,
            device=normal_matrix.device,
            dtype=normal_matrix.dtype,
        ).expand(normal_matrix.shape[0], -1, -1)
        base_inverse = diagonal_mean.reciprocal()[:, None, None] * eye
    else:
        base_name = "diagonal_exact" if base_preconditioner == "jacobi" else "block_jacobi"
        base_inverse, _ = build_preconditioner(
            normal_matrix,
            eigenvalues,
            base_name,
            heads=base_blocks,
            d_head=1,
        )
    base_factor = torch.linalg.cholesky(base_inverse)
    normalized_operator = torch.einsum(
        "bki,bkl,blj->bij",
        base_factor,
        normal_matrix,
        base_factor,
    )
    spectral_values, spectral_vectors = torch.linalg.eigh(normalized_operator)
    if target == "fast":
        selected_values = spectral_values[:, -slots:]
        selected_vectors = spectral_vectors[:, :, -slots:]
    elif target == "slow":
        selected_values = spectral_values[:, :slots]
        selected_vectors = spectral_vectors[:, :, :slots]
    else:
        raise ValueError(f"unknown Ritz target {target}")
    corrections = selected_values.reciprocal() - 1.0
    dimension = normal_matrix.shape[-1]
    eye = torch.eye(
        dimension,
        device=normal_matrix.device,
        dtype=normal_matrix.dtype,
    ).expand(normal_matrix.shape[0], -1, -1)
    normalized_inverse = eye + torch.einsum(
        "bks,bs,bls->bkl",
        selected_vectors,
        corrections,
        selected_vectors,
    )
    return torch.einsum(
        "bik,bkl,bjl->bij",
        base_factor,
        normalized_inverse,
        base_factor,
    )


@torch.no_grad()
def evaluate_checkpoint(args) -> None:
    """Evaluate one frozen head across prompt lengths and condition numbers."""
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in eval mode")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = checkpoint["args"]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_dtype = torch.float64 if saved["dtype"] == "float64" else torch.float32
    model = StructuredOneHeadHeavyBall(
        dimension=saved["K"],
        depth=saved["depth"],
        head_dimension=saved["head_dimension"],
        slots=saved["slots"],
        max_strength=saved["max_strength"],
        strength_init=saved["strength_init"],
        head_mode=saved["head_mode"],
        spectral_lmax_bound=saved["spectral_lmax_bound"],
        step_init=saved["step_init"],
        momentum_init=saved["momentum_init"],
        solver_cell=saved.get("solver_cell", "heavy_ball"),
        base_preconditioner=saved.get("base_preconditioner", "jacobi"),
        base_blocks=saved.get("base_blocks", 2),
        strength_scaling=saved.get("strength_scaling", "fixed"),
        reference_prompt_length=saved.get("reference_prompt_length", saved["prompt_len"]),
        slot_orthogonalization=saved.get("slot_orthogonalization", "independent"),
        correction_mode=saved.get("correction_mode", "positive"),
        subspace_refinement_steps=saved.get("subspace_refinement_steps", 0),
    ).to(device=device, dtype=model_dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "ood_evaluation.csv"
    prompt_grid = parse_int_grid(args.eval_prompt_grid)
    condition_grid = parse_float_grid(args.eval_cond_grid)
    for prompt_length in prompt_grid:
        for condition_number in condition_grid:
            cfg = TaskConfig(
                K=saved["K"],
                prompt_len=prompt_length,
                prior_var=saved["prior_var"],
                noise_var=saved["noise_var"],
                design=saved["design"],
                cond=condition_number,
                dtype=saved["dtype"],
                pde_state_dim=saved.get("pde_state_dim", 0),
            )
            set_seed(args.eval_seed + prompt_length + round(condition_number))
            batch = sample_weak_batch(args.eval_batch_size, cfg, device)
            learned, info = model(batch, cfg)
            diagnostics = spectral_diagnostics(info, batch.H, batch.beta_post)
            jacobi, _ = build_preconditioner(
                batch.H,
                batch.eigvals,
                "jacobi",
                heads=1,
                d_head=1,
                eta_mode="opt",
                eta_multiplier=1.0,
                spectral_order="small",
            )
            noise_precision = 1.0 / cfg.noise_var
            prior_precision = 1.0 / cfg.prior_var

            def evaluation_hvp(vector: Tensor) -> Tensor:
                return weak_normal_hvp(
                    batch.G,
                    vector,
                    noise_precision,
                    prior_precision,
                )

            if model.solver_cell == "heavy_ball":
                jacobi_same_coefficients, _, _ = _heavy_ball_solve(
                    evaluation_hvp,
                    batch.c,
                    jacobi,
                    saved["depth"],
                    info["step"],
                    info["momentum"],
                )
                step_size = info["step"].item()
                momentum = info["momentum"].item()
            else:
                jacobi_same_coefficients, _, _ = _pcg_solve(
                    evaluation_hvp,
                    batch.c,
                    jacobi,
                    saved["depth"],
                )
                step_size = math.nan
                momentum = math.nan
            learned_preconditioner_pcg, _, _ = _pcg_solve(
                evaluation_hvp,
                batch.c,
                info["preconditioner"],
                saved["depth"],
            )
            baselines = {
                solver: run_constructive_loop(
                    batch,
                    cfg,
                    depth=saved["depth"],
                    precond="jacobi",
                    solver=solver,
                )
                for solver in ["richardson", "heavy_ball", "chebyshev", "pcg"]
            }
            stronger_pcg = {}
            stronger_preconditioners = {}
            for label, preconditioner_name, spectral_order in [
                ("pcg_lowrank_small", "lowrank_spectral", "small"),
                ("pcg_lowrank_large", "lowrank_spectral", "large"),
                ("pcg_lowrank_mixed", "lowrank_spectral", "mixed"),
                ("pcg_block_jacobi", "block_jacobi", "small"),
            ]:
                comparison_preconditioner, _ = build_preconditioner(
                    batch.H,
                    batch.eigvals,
                    preconditioner_name,
                    heads=saved["slots"],
                    d_head=1,
                    eta_mode="opt",
                    eta_multiplier=1.0,
                    spectral_order=spectral_order,
                )
                stronger_pcg[label], _, _ = _pcg_solve(
                    evaluation_hvp,
                    batch.c,
                    comparison_preconditioner,
                    saved["depth"],
                )
                stronger_preconditioners[label] = comparison_preconditioner
            for ritz_target in ["slow", "fast"]:
                oracle_ritz = build_oracle_ritz_preconditioner(
                    batch.H,
                    batch.eigvals,
                    saved.get("base_preconditioner", "jacobi"),
                    saved.get("base_blocks", 2),
                    saved["slots"],
                    ritz_target,
                )
                label = f"pcg_oracle_ritz_{ritz_target}"
                stronger_pcg[label], _, _ = _pcg_solve(
                    evaluation_hvp,
                    batch.c,
                    oracle_ritz,
                    saved["depth"],
                )
                stronger_preconditioners[label] = oracle_ritz

            def mse(prediction: Tensor) -> float:
                return torch.mean((prediction - batch.beta_post).pow(2)).item()

            row: Dict[str, float | int | str] = {
                "prompt_length": prompt_length,
                "condition_number": condition_number,
                "learned_mse": mse(learned),
                "learned_coeff_jacobi_mse": mse(jacobi_same_coefficients),
                "learned_preconditioner_pcg_mse": mse(learned_preconditioner_pcg),
                "step_size": step_size,
                "momentum": momentum,
                "strength_mean": info["strengths"].mean().item(),
                **diagnostics,
            }
            for solver, result in baselines.items():
                row[f"{solver}_mse"] = mse(result.beta_L)
            for label, prediction in stronger_pcg.items():
                row[f"{label}_mse"] = mse(prediction)
            if args.baseline_depth_grid:
                original_depth = model.depth
                for comparison_depth in parse_int_grid(args.baseline_depth_grid):
                    block_prediction, _, _ = _pcg_solve(
                        evaluation_hvp,
                        batch.c,
                        stronger_preconditioners["pcg_block_jacobi"],
                        comparison_depth,
                    )
                    row[f"pcg_block_jacobi_depth_{comparison_depth}_mse"] = mse(
                        block_prediction
                    )
                    model.depth = comparison_depth
                    learned_depth_prediction, _ = model(batch, cfg)
                    row[f"learned_depth_{comparison_depth}_mse"] = mse(
                        learned_depth_prediction
                    )
                model.depth = original_depth
            direct_solution = torch.linalg.solve(batch.H, batch.c.unsqueeze(-1)).squeeze(-1)
            row["direct_solve_mse"] = mse(direct_solution)
            if args.timing_repeats > 0:
                row["learned_runtime_ms"] = benchmark_runtime_ms(
                    lambda: model(batch, cfg),
                    args.timing_repeats,
                    device,
                )

                def fixed_pcg_runtime(preconditioner_name: str, heads: int) -> None:
                    runtime_preconditioner, _ = build_preconditioner(
                        batch.H,
                        batch.eigvals,
                        preconditioner_name,
                        heads=heads,
                        d_head=1,
                        eta_mode="opt",
                        eta_multiplier=1.0,
                        spectral_order="small",
                    )
                    _pcg_solve(
                        evaluation_hvp,
                        batch.c,
                        runtime_preconditioner,
                        saved["depth"],
                    )

                row["block_jacobi_runtime_ms"] = benchmark_runtime_ms(
                    lambda: fixed_pcg_runtime("block_jacobi", saved["base_blocks"]),
                    args.timing_repeats,
                    device,
                )
                row["jacobi_runtime_ms"] = benchmark_runtime_ms(
                    lambda: fixed_pcg_runtime("diagonal_exact", 1),
                    args.timing_repeats,
                    device,
                )
                row["direct_solve_runtime_ms"] = benchmark_runtime_ms(
                    lambda: torch.linalg.solve(batch.H, batch.c.unsqueeze(-1)),
                    args.timing_repeats,
                    device,
                )
            append_csv(metrics_path, row)
            print(json.dumps(row, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--outdir", default="runs_structured_one_head_heavyball")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--prior-var", type=float, default=1.0)
    parser.add_argument("--noise-var", type=float, default=0.02)
    parser.add_argument(
        "--design",
        choices=[
            "isotropic",
            "correlated",
            "spiked",
            "pde_elliptic",
            "pde_elliptic_correlated",
        ],
        default="correlated",
    )
    parser.add_argument("--cond", type=float, default=1000.0)
    parser.add_argument("--pde-state-dim", type=int, default=0)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--head-dimension", type=int, default=16)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--head-mode", choices=["symmetric_kernel", "slots"], default="symmetric_kernel")
    parser.add_argument("--solver-cell", choices=["heavy_ball", "pcg"], default="heavy_ball")
    parser.add_argument(
        "--base-preconditioner",
        choices=["jacobi", "block_jacobi", "scalar_mean"],
        default="jacobi",
    )
    parser.add_argument("--base-blocks", type=int, default=2)
    parser.add_argument(
        "--strength-scaling",
        choices=["fixed", "inverse_prompt"],
        default="fixed",
    )
    parser.add_argument("--reference-prompt-length", type=int, default=32)
    parser.add_argument(
        "--slot-orthogonalization",
        choices=["independent", "qr"],
        default="independent",
    )
    parser.add_argument(
        "--correction-mode",
        choices=["positive", "signed", "ritz"],
        default="positive",
    )
    parser.add_argument("--subspace-refinement-steps", type=int, default=0)
    parser.add_argument("--max-strength", type=float, default=0.95)
    parser.add_argument("--strength-init", type=float, default=0.05)
    parser.add_argument("--spectral-lmax-bound", type=float, default=1.1)
    parser.add_argument("--step-init", type=float, default=1.0)
    parser.add_argument("--momentum-init", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.0)
    parser.add_argument("--condition-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-target", choices=["slow", "fast"], default="slow")
    parser.add_argument("--head-checkpoint", default="")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-prompt-grid", default="")
    parser.add_argument("--train-cond-grid", default="")
    parser.add_argument("--eval-prompt-grid", default="16,32,64,128")
    parser.add_argument("--eval-cond-grid", default="10,100,1000")
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--baseline-depth-grid", default="")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    if parsed_args.mode == "train":
        train(parsed_args)
    else:
        evaluate_checkpoint(parsed_args)
