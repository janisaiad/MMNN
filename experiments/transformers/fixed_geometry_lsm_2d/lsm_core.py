"""Core physics and looped-attention model for a 2D LSM proof of concept.

The forward data use a Born/Foldy discretisation of an active multistatic
inverse-scattering experiment.  Every task is a new obstacle and produces a
complex far-field matrix.  The recurrent model does not regress an image from
that matrix.  Instead it solves the Bayesian linear-sampling systems for all
probing points, using a prescribed nonlinear angular kernel as its prior
feature geometry.

No exact solve is used by the training loss.  The direct solve implemented at
the bottom of this file is an evaluation-only numerical reference.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ShapeFamily = Literal["ellipse", "disk", "kite", "two_disks"]
StationaryMethod = Literal["richardson", "heavy_ball", "chebyshev"]


@dataclass(frozen=True)
class PhysicsConfig:
    """Fixed acquisition geometry and data-generation parameters."""

    n_angles: int = 32
    grid_size: int = 32
    domain_half_width: float = 0.8
    wavenumber: float = 8.0
    noise_rel: float = 0.03
    ridge_rel: float = 0.01
    kernel_gamma: float = 1.0
    kernel_mix: float = 0.20


@dataclass(frozen=True)
class LoopConfig:
    """Architecture and training hyperparameters."""

    depth: int = 12
    controller_width: int = 32
    steps: int = 1800
    batch_size: int = 12
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    log_every: int = 50
    eval_tasks: int = 192
    eval_batch_size: int = 24
    rank_pairs: int = 48


def config_dict(physics: PhysicsConfig, loop: LoopConfig) -> dict[str, object]:
    return {"physics": asdict(physics), "loop": asdict(loop)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complex_normal_like(value: Tensor) -> Tensor:
    """Unit-variance circular complex Gaussian noise."""
    return (torch.randn_like(value.real) + 1j * torch.randn_like(value.real)) / math.sqrt(2.0)


def angular_kernel(
    angles: Tensor,
    gamma: float,
    mix: float,
) -> tuple[Tensor, Tensor]:
    """Return a fixed softmax attention and its positive symmetric kernel.

    The nonlinear features are prescribed, not learned.  Let

        W_ij = exp(gamma cos(theta_i-theta_j)).

    ``attention`` is row-softmax(W's logits), while the covariance uses the
    symmetric normalisation D^{-1/2} W D^{-1/2}.  Convex mixing with the
    identity keeps the prior well conditioned without changing its geometry.
    """
    logits = gamma * torch.cos(angles[:, None] - angles[None, :])
    weights = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    attention = weights / weights.sum(dim=-1, keepdim=True)
    degree = weights.sum(dim=-1).clamp_min(1.0e-12)
    symmetric = weights / torch.sqrt(degree[:, None] * degree[None, :])
    identity = torch.eye(angles.numel(), device=angles.device, dtype=angles.dtype)
    covariance = (1.0 - mix) * identity + mix * symmetric
    return attention, covariance


def angular_feature_kernel(angles: Tensor, gamma: float) -> Tensor:
    """Symmetric PSD normalisation of the prescribed angular softmax features."""
    logits = gamma * torch.cos(angles[:, None] - angles[None, :])
    weights = torch.exp(logits - logits.amax())
    degree = weights.sum(dim=-1).clamp_min(1.0e-12)
    return weights / torch.sqrt(degree[:, None] * degree[None, :])


class FixedGeometryBornLSM:
    """On-the-fly 2D multistatic LSM tasks on a fixed circular array."""

    def __init__(self, cfg: PhysicsConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64

        self.angles = (
            torch.arange(cfg.n_angles, device=device, dtype=self.real_dtype)
            * (2.0 * math.pi / cfg.n_angles)
        )
        self.directions = torch.stack([torch.cos(self.angles), torch.sin(self.angles)], dim=-1)
        axis = torch.linspace(
            -cfg.domain_half_width,
            cfg.domain_half_width,
            cfg.grid_size,
            device=device,
            dtype=self.real_dtype,
        )
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        self.grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        self.grid_x = grid_x
        self.grid_y = grid_y

        phase = cfg.wavenumber * (self.directions @ self.grid.T)
        self.phase_receiver = torch.exp(-1j * phase).to(self.complex_dtype)
        self.phase_source = torch.exp(1j * phase.T).to(self.complex_dtype)
        self.probe_rhs = self.phase_receiver
        attention, covariance = angular_kernel(
            self.angles,
            cfg.kernel_gamma,
            cfg.kernel_mix,
        )
        self.attention = attention
        self.feature_kernel = angular_feature_kernel(self.angles, cfg.kernel_gamma).to(
            self.complex_dtype
        )
        self.kernel = covariance.to(self.complex_dtype)

    @property
    def n_probes(self) -> int:
        return self.grid.shape[0]

    def _base_coordinates(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        center = (torch.rand(batch_size, 2, device=self.device) - 0.5) * 0.62
        angle = torch.rand(batch_size, device=self.device) * math.pi
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        delta = self.grid[None, :, :] - center[:, None, :]
        local_x = cosine[:, None] * delta[..., 0] + sine[:, None] * delta[..., 1]
        local_y = -sine[:, None] * delta[..., 0] + cosine[:, None] * delta[..., 1]
        return center, local_x, local_y, angle

    def sample_masks(self, batch_size: int, family: ShapeFamily = "ellipse") -> Tensor:
        """Sample obstacle masks; ellipses are in-distribution training tasks."""
        center, local_x, local_y, angle = self._base_coordinates(batch_size)
        del angle
        radius = 0.17 + 0.13 * torch.rand(batch_size, 1, device=self.device)

        if family == "disk":
            radius_x = radius
            radius_y = radius
            return ((local_x / radius_x).square() + (local_y / radius_y).square() <= 1.0)

        if family == "ellipse":
            aspect = 0.68 + 0.64 * torch.rand(batch_size, 1, device=self.device)
            radius_x = radius * aspect
            radius_y = radius / aspect
            return ((local_x / radius_x).square() + (local_y / radius_y).square() <= 1.0)

        if family == "kite":
            polar_angle = torch.atan2(local_y, local_x)
            polar_radius = torch.sqrt(local_x.square() + local_y.square())
            boundary = radius * (
                1.0
                + 0.24 * torch.cos(3.0 * polar_angle)
                + 0.10 * torch.sin(2.0 * polar_angle)
            )
            return polar_radius <= boundary.clamp_min(0.08)

        if family == "two_disks":
            separation = 0.27 + 0.13 * torch.rand(batch_size, 1, device=self.device)
            orientation = torch.rand(batch_size, 1, device=self.device) * math.pi
            offset = torch.cat([torch.cos(orientation), torch.sin(orientation)], dim=-1)
            offset = 0.5 * separation * offset
            radius_pair = 0.12 + 0.055 * torch.rand(batch_size, 2, device=self.device)
            delta_left = self.grid[None, :, :] - (center - offset)[:, None, :]
            delta_right = self.grid[None, :, :] - (center + offset)[:, None, :]
            left = delta_left.square().sum(dim=-1) <= radius_pair[:, :1].square()
            right = delta_right.square().sum(dim=-1) <= radius_pair[:, 1:].square()
            return left | right

        raise ValueError(f"unknown shape family: {family}")

    def far_field_from_mask(
        self,
        mask: Tensor,
        *,
        noise_rel: float | None = None,
        receiver_noise_profile: Tensor | None = None,
    ) -> Tensor:
        """Assemble the Born/Foldy far-field matrix and add relative noise."""
        weights = mask.to(self.real_dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        far_field = torch.einsum(
            "rz,bz,zs->brs",
            self.phase_receiver,
            weights.to(self.complex_dtype),
            self.phase_source,
        )

        relative_noise = self.cfg.noise_rel if noise_rel is None else float(noise_rel)
        if relative_noise > 0.0:
            scale = torch.linalg.matrix_norm(far_field, ord="fro") / self.cfg.n_angles
            noise = complex_normal_like(far_field)
            if receiver_noise_profile is not None:
                profile = receiver_noise_profile.to(self.device, self.real_dtype)
                profile = profile / profile.square().mean().sqrt().clamp_min(1.0e-8)
                noise = noise * profile[None, :, None]
            far_field = far_field + relative_noise * scale[:, None, None] * noise
        return far_field

    def acquisition_at_angles(
        self,
        mask: Tensor,
        angles: Tensor,
        *,
        noise_rel: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Differentiable acquisition for trainable experiment-design angles."""
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        phase = self.cfg.wavenumber * (directions @ self.grid.T)
        phase_receiver = torch.exp(-1j * phase).to(self.complex_dtype)
        phase_source = torch.exp(1j * phase.T).to(self.complex_dtype)
        weights = mask.to(self.real_dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        far_field = torch.einsum(
            "rz,bz,zs->brs",
            phase_receiver,
            weights.to(self.complex_dtype),
            phase_source,
        )
        relative_noise = self.cfg.noise_rel if noise_rel is None else float(noise_rel)
        if relative_noise > 0.0:
            scale = torch.linalg.matrix_norm(far_field, ord="fro") / angles.numel()
            far_field = far_field + (
                relative_noise
                * scale[:, None, None]
                * complex_normal_like(far_field)
            )
        attention, kernel = angular_kernel(
            angles,
            self.cfg.kernel_gamma,
            self.cfg.kernel_mix,
        )
        feature_kernel = angular_feature_kernel(angles, self.cfg.kernel_gamma)
        return (
            far_field,
            phase_receiver,
            attention,
            kernel.to(self.complex_dtype),
            feature_kernel.to(self.complex_dtype),
        )

    def sample_batch(
        self,
        batch_size: int,
        family: ShapeFamily = "ellipse",
        *,
        noise_rel: float | None = None,
        receiver_noise_profile: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        mask = self.sample_masks(batch_size, family)
        far_field = self.far_field_from_mask(
            mask,
            noise_rel=noise_rel,
            receiver_noise_profile=receiver_noise_profile,
        )
        return far_field, mask

    def subset(
        self,
        far_field: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Restrict a multistatic task to one shared source/receiver subset."""
        selected = far_field.index_select(1, indices).index_select(2, indices)
        probe = self.probe_rhs.index_select(0, indices)
        angles = self.angles.index_select(0, indices)
        attention, kernel = angular_kernel(
            angles,
            self.cfg.kernel_gamma,
            self.cfg.kernel_mix,
        )
        return selected, probe, attention, kernel.to(self.complex_dtype)


def build_bayesian_system(
    far_field: Tensor,
    kernel: Tensor,
    probe_rhs: Tensor,
    ridge_rel: float,
) -> dict[str, Tensor]:
    """Build and safely scale the dual Bayesian LSM system.

    Jacobi scaling followed by a Gershgorin row bound puts the spectrum in a
    stable range without an eigendecomposition.  This is part of the numerical
    architecture, not a direct solve.
    """
    batch, n_receivers, _ = far_field.shape
    if kernel.ndim == 2:
        kernel_batch = kernel.unsqueeze(0).expand(batch, -1, -1)
    else:
        kernel_batch = kernel
    h_data = far_field @ kernel_batch @ far_field.mH
    mean_diagonal = h_data.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1.0e-7)
    ridge = float(ridge_rel) * mean_diagonal
    identity = torch.eye(n_receivers, device=far_field.device, dtype=far_field.dtype)
    hessian = h_data + ridge[:, None, None] * identity

    diagonal = hessian.diagonal(dim1=-2, dim2=-1).real.clamp_min(1.0e-8)
    inverse_sqrt = diagonal.rsqrt()
    jacobi = inverse_sqrt[:, :, None] * hessian * inverse_sqrt[:, None, :]
    row_bound = jacobi.abs().sum(dim=-1).amax(dim=-1).clamp_min(1.0)
    operator = jacobi / row_bound[:, None, None]

    if probe_rhs.ndim == 2:
        rhs = probe_rhs.unsqueeze(0).expand(batch, -1, -1)
    else:
        rhs = probe_rhs
    scaled_rhs = inverse_sqrt[:, :, None] * rhs / row_bound[:, None, None]
    return {
        "operator": operator,
        "rhs": scaled_rhs,
        "inverse_sqrt": inverse_sqrt,
        "hessian": hessian,
        "ridge": ridge,
        "row_bound": row_bound,
        "kernel": kernel_batch,
    }


def system_summary_features(system: dict[str, Tensor]) -> Tensor:
    """Six invariant features shared by every learned solver encoder."""
    operator = system["operator"]
    diagonal = operator.diagonal(dim1=-2, dim2=-1).real
    off_diagonal = operator - torch.diag_embed(operator.diagonal(dim1=-2, dim2=-1))
    row_energy = operator.abs().sum(dim=-1)
    ridge_fraction = system["ridge"] / (
        system["hessian"].diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1.0e-8)
    )
    return torch.stack(
        [
            diagonal.mean(dim=-1),
            diagonal.std(dim=-1, unbiased=False),
            off_diagonal.abs().square().mean(dim=(-2, -1)).sqrt(),
            row_energy.mean(dim=-1),
            row_energy.std(dim=-1, unbiased=False),
            torch.log(ridge_fraction.clamp_min(1.0e-8)),
        ],
        dim=-1,
    )


def prior_score(far_field: Tensor, kernel: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Return -log posterior-mean prior norm and the LSM coefficients."""
    if kernel.ndim == 2:
        kernel_batch = kernel.unsqueeze(0).expand(far_field.shape[0], -1, -1)
    else:
        kernel_batch = kernel
    coefficients = kernel_batch @ far_field.mH @ q
    whitened = torch.linalg.solve(kernel_batch, coefficients)
    norm_squared = (coefficients.conj() * whitened).sum(dim=1).real.clamp_min(1.0e-12)
    return -0.5 * torch.log(norm_squared), coefficients


class TiedBayesianLSMLoop(nn.Module):
    """A tied residual-attention loop for Bayesian linear sampling.

    The only learned object is a tiny invariant controller for the shared
    Richardson/heavy-ball cell.  The angular softmax kernel, physical matrix,
    probing right-hand sides, and ridge model remain explicit.
    """

    def __init__(
        self,
        kernel: Tensor,
        ridge_rel: float,
        depth: int,
        controller_width: int = 32,
    ) -> None:
        super().__init__()
        self.register_buffer("kernel", kernel.detach().clone())
        self.register_buffer("ridge_rel", torch.tensor(float(ridge_rel)))
        self.depth = int(depth)
        self.controller = nn.Sequential(
            nn.Linear(6, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, 2),
        )
        nn.init.zeros_(self.controller[-1].weight)
        with torch.no_grad():
            self.controller[-1].bias.copy_(torch.tensor([0.0, -3.0]))

    @property
    def geometry_is_frozen(self) -> bool:
        return not self.kernel.requires_grad

    def _controller_step(
        self,
        residual: Tensor,
        previous_residual_norm: Tensor,
        rhs_norm: Tensor,
        update_norm: Tensor,
        system: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual_norm = residual.abs().square().mean(dim=(1, 2)).sqrt().clamp_min(1.0e-10)
        relative = residual_norm / rhs_norm
        ratio = residual_norm / previous_residual_norm.clamp_min(1.0e-10)
        diagonal = system["operator"].diagonal(dim1=-2, dim2=-1).real
        ridge_fraction = system["ridge"] / (
            system["hessian"].diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1.0e-8)
        )
        features = torch.stack(
            [
                torch.log(relative.clamp_min(1.0e-8)),
                torch.log(ratio.clamp_min(1.0e-8)),
                torch.log1p(update_norm / rhs_norm),
                diagonal.mean(dim=-1),
                diagonal.std(dim=-1, unbiased=False),
                torch.log(ridge_fraction.clamp_min(1.0e-8)),
            ],
            dim=-1,
        )
        raw_eta, raw_beta = self.controller(features).unbind(dim=-1)
        eta = 1.90 * torch.sigmoid(raw_eta)
        beta = 0.95 * torch.sigmoid(raw_beta)
        return eta, beta, residual_norm

    def forward(
        self,
        far_field: Tensor,
        probe_rhs: Tensor,
        *,
        kernel: Tensor | None = None,
        depth: int | None = None,
        return_history: bool = False,
        fixed_coefficients: tuple[float, float] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        active_kernel = self.kernel if kernel is None else kernel
        system = build_bayesian_system(
            far_field,
            active_kernel,
            probe_rhs,
            float(self.ridge_rel.item()),
        )
        operator = system["operator"]
        rhs = system["rhs"]
        iterate = torch.zeros_like(rhs)
        previous_iterate = torch.zeros_like(rhs)
        rhs_norm = rhs.abs().square().mean(dim=(1, 2)).sqrt().clamp_min(1.0e-10)
        previous_residual_norm = rhs_norm
        update_norm = torch.zeros_like(rhs_norm)
        residual_history: list[Tensor] = []
        eta_history: list[Tensor] = []
        beta_history: list[Tensor] = []

        n_steps = self.depth if depth is None else int(depth)
        for _ in range(n_steps):
            residual = rhs - operator @ iterate
            if fixed_coefficients is None:
                eta, beta, residual_norm = self._controller_step(
                    residual,
                    previous_residual_norm,
                    rhs_norm,
                    update_norm,
                    system,
                )
            else:
                eta = torch.full_like(rhs_norm, float(fixed_coefficients[0]))
                beta = torch.full_like(rhs_norm, float(fixed_coefficients[1]))
                residual_norm = residual.abs().square().mean(dim=(1, 2)).sqrt()
            iterate_next = (
                iterate
                + eta[:, None, None] * residual
                + beta[:, None, None] * (iterate - previous_iterate)
            )
            update_norm = (iterate_next - iterate).abs().square().mean(dim=(1, 2)).sqrt()
            previous_iterate, iterate = iterate, iterate_next
            previous_residual_norm = residual_norm
            if return_history:
                residual_history.append(residual_norm / rhs_norm)
                eta_history.append(eta)
                beta_history.append(beta)

        final_residual = rhs - operator @ iterate
        relative_residual = (
            final_residual.abs().square().mean(dim=(1, 2)).sqrt() / rhs_norm
        )
        q = system["inverse_sqrt"][:, :, None] * iterate
        score, coefficients = prior_score(far_field, active_kernel, q)
        info: dict[str, Tensor] = {
            "q": q,
            "coefficients": coefficients,
            "relative_residual": relative_residual,
            "ridge": system["ridge"],
            "row_bound": system["row_bound"],
        }
        if return_history:
            info["residual_history"] = torch.stack(residual_history, dim=1)
            info["eta_history"] = torch.stack(eta_history, dim=1)
            info["beta_history"] = torch.stack(beta_history, dim=1)
        return score, info


class TiedKrylovLSMLoop(nn.Module):
    """Learned, tied complex-CG loop with fixed nonlinear feature geometry.

    The Krylov coefficients are computed in context from each sampling system.
    A shared controller supplies bounded damping corrections.  Consequently the
    architecture can recover ordinary CG, but it can also learn useful early
    stopping dynamics from obstacle supervision without ever seeing a direct
    solve as a target.
    """

    def __init__(
        self,
        kernel: Tensor,
        ridge_rel: float,
        depth: int,
        controller_width: int = 32,
    ) -> None:
        super().__init__()
        self.register_buffer("kernel", kernel.detach().clone())
        self.register_buffer("ridge_rel", torch.tensor(float(ridge_rel)))
        self.depth = int(depth)
        self.controller = nn.Sequential(
            nn.Linear(6, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, 2),
        )
        nn.init.zeros_(self.controller[-1].weight)
        with torch.no_grad():
            # Initial damping is approximately 0.70; training can recover 1.
            self.controller[-1].bias.fill_(-1.3862944)

    @property
    def geometry_is_frozen(self) -> bool:
        return not self.kernel.requires_grad

    def _damping(
        self,
        residual: Tensor,
        direction: Tensor,
        alpha_cg: Tensor,
        beta_previous: Tensor,
        rhs_energy: Tensor,
        ridge_fraction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual_energy = residual.abs().square().sum(dim=1).clamp_min(1.0e-12)
        direction_energy = direction.abs().square().sum(dim=1).clamp_min(1.0e-12)
        alignment = (residual.conj() * direction).sum(dim=1).real
        alignment = alignment / torch.sqrt(residual_energy * direction_energy)
        ridge_column = ridge_fraction[:, None].expand_as(alpha_cg)
        features = torch.stack(
            [
                0.5 * torch.log(residual_energy / rhs_energy.clamp_min(1.0e-12)),
                torch.log(alpha_cg.abs().clamp_min(1.0e-8)),
                torch.log1p(beta_previous.clamp_min(0.0)),
                alignment,
                torch.log(ridge_column.clamp_min(1.0e-8)),
                torch.log1p(direction_energy / rhs_energy.clamp_min(1.0e-12)),
            ],
            dim=-1,
        )
        raw_alpha, raw_beta = self.controller(features).unbind(dim=-1)
        # Both factors lie in [0.5, 1.5], with exact CG at one.
        alpha_scale = 0.5 + torch.sigmoid(raw_alpha)
        beta_scale = 0.5 + torch.sigmoid(raw_beta)
        return alpha_scale, beta_scale

    def forward(
        self,
        far_field: Tensor,
        probe_rhs: Tensor,
        *,
        kernel: Tensor | None = None,
        depth: int | None = None,
        return_history: bool = False,
        fixed_damping: tuple[float, float] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        active_kernel = self.kernel if kernel is None else kernel
        system = build_bayesian_system(
            far_field,
            active_kernel,
            probe_rhs,
            float(self.ridge_rel.item()),
        )
        operator = system["operator"]
        rhs = system["rhs"]
        iterate = torch.zeros_like(rhs)
        residual = rhs.clone()
        direction = residual.clone()
        residual_energy = residual.abs().square().sum(dim=1).clamp_min(1.0e-12)
        rhs_energy = residual_energy.clone()
        beta_previous = torch.zeros_like(residual_energy)
        ridge_fraction = system["ridge"] / (
            system["hessian"].diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1.0e-8)
        )
        residual_history: list[Tensor] = []
        alpha_scale_history: list[Tensor] = []
        beta_scale_history: list[Tensor] = []

        n_steps = self.depth if depth is None else int(depth)
        for _ in range(n_steps):
            operator_direction = operator @ direction
            denominator = (direction.conj() * operator_direction).sum(dim=1).real
            alpha_cg = residual_energy / denominator.clamp_min(1.0e-12)
            if fixed_damping is None:
                alpha_scale, beta_scale = self._damping(
                    residual,
                    direction,
                    alpha_cg,
                    beta_previous,
                    rhs_energy,
                    ridge_fraction,
                )
            else:
                alpha_scale = torch.full_like(alpha_cg, float(fixed_damping[0]))
                beta_scale = torch.full_like(alpha_cg, float(fixed_damping[1]))
            alpha = alpha_scale * alpha_cg
            iterate = iterate + alpha[:, None, :] * direction
            residual_next = residual - alpha[:, None, :] * operator_direction
            residual_energy_next = residual_next.abs().square().sum(dim=1).clamp_min(1.0e-12)
            beta_cg = residual_energy_next / residual_energy.clamp_min(1.0e-12)
            beta = beta_scale * beta_cg
            direction = residual_next + beta[:, None, :] * direction
            residual = residual_next
            residual_energy = residual_energy_next
            beta_previous = beta_cg
            if return_history:
                relative = torch.sqrt(residual_energy.sum(dim=-1) / rhs_energy.sum(dim=-1))
                residual_history.append(relative)
                alpha_scale_history.append(alpha_scale.mean(dim=-1))
                beta_scale_history.append(beta_scale.mean(dim=-1))

        relative_residual = torch.sqrt(residual_energy.sum(dim=-1) / rhs_energy.sum(dim=-1))
        q = system["inverse_sqrt"][:, :, None] * iterate
        score, coefficients = prior_score(far_field, active_kernel, q)
        info: dict[str, Tensor] = {
            "q": q,
            "coefficients": coefficients,
            "relative_residual": relative_residual,
            "ridge": system["ridge"],
            "row_bound": system["row_bound"],
        }
        if return_history:
            info["residual_history"] = torch.stack(residual_history, dim=1)
            info["alpha_scale_history"] = torch.stack(alpha_scale_history, dim=1)
            info["beta_scale_history"] = torch.stack(beta_scale_history, dim=1)
        return score, info


class TiedAttentionPCGLSMLoop(nn.Module):
    """Tied PCG with an SPD preconditioner built from fixed softmax features.

    Let S be the symmetric normalisation of the prescribed angular softmax
    kernel and L=I-S its graph Laplacian.  A task-invariant controller selects
    nonnegative coefficients in

        P = I + a L + b L^2.

    P is therefore positive definite and stays fixed throughout one solve.
    The complex PCG coefficients remain the exact in-context coefficients, so
    training cannot destroy Krylov conjugacy as learned damping can.
    """

    def __init__(
        self,
        kernel: Tensor,
        feature_kernel: Tensor,
        ridge_rel: float,
        depth: int,
        controller_width: int = 24,
    ) -> None:
        super().__init__()
        self.register_buffer("kernel", kernel.detach().clone())
        self.register_buffer("feature_kernel", feature_kernel.detach().clone())
        self.register_buffer("ridge_rel", torch.tensor(float(ridge_rel)))
        self.depth = int(depth)
        self.controller = nn.Sequential(
            nn.Linear(6, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, 2),
        )
        nn.init.zeros_(self.controller[-1].weight)
        # Starts extremely close to identity-preconditioned standard CG.
        nn.init.constant_(self.controller[-1].bias, -8.0)

    @property
    def geometry_is_frozen(self) -> bool:
        return not self.kernel.requires_grad and not self.feature_kernel.requires_grad

    def _preconditioner(
        self,
        system: dict[str, Tensor],
        feature_kernel: Tensor,
        *,
        fixed_coefficients: tuple[float, float] | None,
    ) -> tuple[Tensor, Tensor]:
        operator = system["operator"]
        features = system_summary_features(system)
        if fixed_coefficients is None:
            coefficients = 8.0 * torch.sigmoid(self.controller(features))
        else:
            coefficients = torch.tensor(
                fixed_coefficients,
                device=operator.device,
                dtype=operator.real.dtype,
            ).expand(operator.shape[0], -1)

        batch, n_angles, _ = operator.shape
        if feature_kernel.ndim == 2:
            feature = feature_kernel.unsqueeze(0).expand(batch, -1, -1)
        else:
            feature = feature_kernel
        identity = torch.eye(n_angles, device=operator.device, dtype=operator.dtype)
        laplacian = identity.unsqueeze(0) - feature
        laplacian_squared = laplacian @ laplacian
        preconditioner = (
            identity.unsqueeze(0)
            + coefficients[:, :1, None] * laplacian
            + coefficients[:, 1:, None] * laplacian_squared
        )
        diagonal_mean = preconditioner.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
        preconditioner = preconditioner / diagonal_mean[:, None, None].clamp_min(1.0e-8)
        return preconditioner, coefficients

    def forward(
        self,
        far_field: Tensor,
        probe_rhs: Tensor,
        *,
        kernel: Tensor | None = None,
        feature_kernel: Tensor | None = None,
        depth: int | None = None,
        return_history: bool = False,
        fixed_preconditioner: tuple[float, float] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        active_kernel = self.kernel if kernel is None else kernel
        active_feature_kernel = self.feature_kernel if feature_kernel is None else feature_kernel
        system = build_bayesian_system(
            far_field,
            active_kernel,
            probe_rhs,
            float(self.ridge_rel.item()),
        )
        operator = system["operator"]
        rhs = system["rhs"]
        preconditioner, coefficients = self._preconditioner(
            system,
            active_feature_kernel,
            fixed_coefficients=fixed_preconditioner,
        )
        iterate = torch.zeros_like(rhs)
        residual = rhs.clone()
        preconditioned_residual = preconditioner @ residual
        direction = preconditioned_residual.clone()
        rz = (residual.conj() * preconditioned_residual).sum(dim=1).real.clamp_min(1.0e-12)
        rhs_energy = residual.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        residual_history: list[Tensor] = []

        n_steps = self.depth if depth is None else int(depth)
        for _ in range(n_steps):
            operator_direction = operator @ direction
            denominator = (direction.conj() * operator_direction).sum(dim=1).real
            alpha = rz / denominator.clamp_min(1.0e-12)
            iterate = iterate + alpha[:, None, :] * direction
            residual = residual - alpha[:, None, :] * operator_direction
            preconditioned_residual = preconditioner @ residual
            rz_next = (residual.conj() * preconditioned_residual).sum(dim=1).real.clamp_min(1.0e-12)
            beta = rz_next / rz.clamp_min(1.0e-12)
            direction = preconditioned_residual + beta[:, None, :] * direction
            rz = rz_next
            if return_history:
                residual_energy = residual.abs().square().sum(dim=(1, 2))
                residual_history.append(torch.sqrt(residual_energy / rhs_energy))

        residual_energy = residual.abs().square().sum(dim=(1, 2))
        relative_residual = torch.sqrt(residual_energy / rhs_energy)
        q = system["inverse_sqrt"][:, :, None] * iterate
        score, lsm_coefficients = prior_score(far_field, active_kernel, q)
        info: dict[str, Tensor] = {
            "q": q,
            "coefficients": lsm_coefficients,
            "relative_residual": relative_residual,
            "preconditioner_coefficients": coefficients,
            "ridge": system["ridge"],
            "row_bound": system["row_bound"],
        }
        if return_history:
            info["residual_history"] = torch.stack(residual_history, dim=1)
        return score, info


class TiedStationaryLSMLoop(nn.Module):
    """Richardson, heavy-ball, or Chebyshev with the same LSM encoder/decoder.

    Every variant receives the same six invariant system summaries through the
    same 6-width-width-2 MLP, uses the same fixed GP kernel, and decodes the
    final iterate with :func:`prior_score`.  Only the recurrent update differs.
    """

    def __init__(
        self,
        kernel: Tensor,
        ridge_rel: float,
        depth: int,
        method: StationaryMethod,
        controller_width: int = 32,
    ) -> None:
        super().__init__()
        if method not in ("richardson", "heavy_ball", "chebyshev"):
            raise ValueError(f"unknown stationary method: {method}")
        self.register_buffer("kernel", kernel.detach().clone())
        self.register_buffer("ridge_rel", torch.tensor(float(ridge_rel)))
        self.depth = int(depth)
        self.method: StationaryMethod = method
        self.controller = nn.Sequential(
            nn.Linear(6, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, 2),
        )
        nn.init.zeros_(self.controller[-1].weight)
        with torch.no_grad():
            if method == "richardson":
                self.controller[-1].bias.copy_(torch.tensor([0.0, -4.0]))
            elif method == "heavy_ball":
                self.controller[-1].bias.copy_(torch.tensor([0.0, -3.0]))
            else:
                self.controller[-1].bias.copy_(torch.tensor([-4.0, -4.0]))

    @property
    def geometry_is_frozen(self) -> bool:
        return not self.kernel.requires_grad

    def _richardson_or_heavy_ball(
        self,
        system: dict[str, Tensor],
        raw: Tensor,
        n_steps: int,
        return_history: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        operator = system["operator"]
        rhs = system["rhs"]
        raw_first, raw_second = raw.unbind(dim=-1)
        if self.method == "richardson":
            upper_bound = 1.0 + F.softplus(raw_second)
            eta = 1.95 * torch.sigmoid(raw_first) / upper_bound
            beta = torch.zeros_like(eta)
        else:
            beta = 0.95 * torch.sigmoid(raw_second)
            eta = 1.95 * (1.0 + beta) * torch.sigmoid(raw_first)

        iterate = torch.zeros_like(rhs)
        previous_iterate = torch.zeros_like(rhs)
        rhs_energy = rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        residual_history: list[Tensor] = []
        for _ in range(n_steps):
            residual = rhs - operator @ iterate
            iterate_next = (
                iterate
                + eta[:, None, None] * residual
                + beta[:, None, None] * (iterate - previous_iterate)
            )
            previous_iterate, iterate = iterate, iterate_next
            if return_history:
                next_residual = rhs - operator @ iterate
                residual_history.append(
                    torch.sqrt(next_residual.abs().square().sum(dim=(1, 2)) / rhs_energy)
                )
        info = {
            "eta": eta,
            "beta": beta,
        }
        if return_history:
            info["residual_history"] = torch.stack(residual_history, dim=1)
        return iterate, info

    def _chebyshev(
        self,
        system: dict[str, Tensor],
        raw: Tensor,
        n_steps: int,
        return_history: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        operator = system["operator"]
        rhs = system["rhs"]
        raw_lower, raw_upper = raw.unbind(dim=-1)
        diagonal = system["hessian"].diagonal(dim1=-2, dim2=-1).real
        safe_lower = system["ridge"] / diagonal.amax(dim=-1).clamp_min(1.0e-8)
        safe_lower = safe_lower / system["row_bound"]
        lower_multiplier = 1.0 + 15.0 * torch.sigmoid(raw_lower)
        upper = 1.0 + F.softplus(raw_upper)
        lower = (safe_lower * lower_multiplier).clamp_min(1.0e-6)
        lower = torch.minimum(lower, 0.80 * upper)
        center = 0.5 * (upper + lower)
        radius = 0.5 * (upper - lower)

        iterate = torch.zeros_like(rhs)
        previous_iterate = torch.zeros_like(rhs)
        rhs_energy = rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        alpha_previous = center.reciprocal()
        residual_history: list[Tensor] = []
        beta_last = torch.zeros_like(center)
        for step in range(n_steps):
            residual = rhs - operator @ iterate
            if step == 0:
                alpha = alpha_previous
                beta = torch.zeros_like(alpha)
            else:
                beta = (0.5 * radius * alpha_previous).square()
                alpha = (center - beta / alpha_previous).reciprocal()
            iterate_next = (
                iterate
                + alpha[:, None, None] * residual
                + beta[:, None, None] * (iterate - previous_iterate)
            )
            previous_iterate, iterate = iterate, iterate_next
            alpha_previous = alpha
            beta_last = beta
            if return_history:
                next_residual = rhs - operator @ iterate
                residual_history.append(
                    torch.sqrt(next_residual.abs().square().sum(dim=(1, 2)) / rhs_energy)
                )
        info = {
            "lower_bound": lower,
            "upper_bound": upper,
            "eta": alpha_previous,
            "beta": beta_last,
        }
        if return_history:
            info["residual_history"] = torch.stack(residual_history, dim=1)
        return iterate, info

    def forward(
        self,
        far_field: Tensor,
        probe_rhs: Tensor,
        *,
        kernel: Tensor | None = None,
        depth: int | None = None,
        return_history: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        active_kernel = self.kernel if kernel is None else kernel
        system = build_bayesian_system(
            far_field,
            active_kernel,
            probe_rhs,
            float(self.ridge_rel.item()),
        )
        raw = self.controller(system_summary_features(system))
        n_steps = self.depth if depth is None else int(depth)
        if self.method == "chebyshev":
            iterate, algorithm_info = self._chebyshev(
                system,
                raw,
                n_steps,
                return_history,
            )
        else:
            iterate, algorithm_info = self._richardson_or_heavy_ball(
                system,
                raw,
                n_steps,
                return_history,
            )
        final_residual = system["rhs"] - system["operator"] @ iterate
        rhs_energy = system["rhs"].abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        relative_residual = torch.sqrt(
            final_residual.abs().square().sum(dim=(1, 2)) / rhs_energy
        )
        q = system["inverse_sqrt"][:, :, None] * iterate
        score, coefficients = prior_score(far_field, active_kernel, q)
        info: dict[str, Tensor] = {
            "q": q,
            "coefficients": coefficients,
            "relative_residual": relative_residual,
            "ridge": system["ridge"],
            **algorithm_info,
        }
        return score, info


def ranking_loss(score: Tensor, mask: Tensor, n_pairs: int = 48, margin: float = 0.25) -> Tensor:
    """Balanced differentiable AUC surrogate with no image-regression target."""
    positive_weights = mask.to(score.dtype)
    negative_weights = (~mask).to(score.dtype)
    positive_indices = torch.multinomial(positive_weights, n_pairs, replacement=True)
    negative_indices = torch.multinomial(negative_weights, n_pairs, replacement=True)
    positive_score = score.gather(1, positive_indices)
    negative_score = score.gather(1, negative_indices)
    return F.softplus(margin - positive_score[:, :, None] + negative_score[:, None, :]).mean()


def training_objective(score: Tensor, mask: Tensor, info: dict[str, Tensor], n_pairs: int) -> Tensor:
    rank = ranking_loss(score, mask, n_pairs=n_pairs)
    residual = torch.log(info["relative_residual"].clamp_min(1.0e-7)).mean()
    return rank + 0.08 * residual


@torch.no_grad()
def exact_bayesian_lsm(
    far_field: Tensor,
    probe_rhs: Tensor,
    kernel: Tensor,
    ridge_rel: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Direct-solve reference used only for held-out evaluation."""
    system = build_bayesian_system(far_field, kernel, probe_rhs, ridge_rel)
    iterate = torch.linalg.solve(system["operator"], system["rhs"])
    q = system["inverse_sqrt"][:, :, None] * iterate
    score, coefficients = prior_score(far_field, kernel, q)
    residual = system["rhs"] - system["operator"] @ iterate
    rhs_norm = system["rhs"].abs().square().mean(dim=(1, 2)).sqrt().clamp_min(1.0e-10)
    relative_residual = residual.abs().square().mean(dim=(1, 2)).sqrt() / rhs_norm
    return score, {
        "q": q,
        "coefficients": coefficients,
        "relative_residual": relative_residual,
        "ridge": system["ridge"],
    }
