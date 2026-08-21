"""Original near-field LSM with sound-soft MFS data and moment-controlled HB.

This module is deliberately separate from the Born far-field proof of concept.
Point sources illuminate sound-soft obstacles, receivers record the scattered
near field, and a method-of-fundamental-solutions (MFS) forward solve enforces
the Dirichlet boundary condition.  The inverse architecture keeps the GP prior,
noise whitening, population preconditioning, in-context Krylov moments, and
the heavy-ball solver as distinct mathematical objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .foundation_uq import posterior_covariance, posterior_score_moments
from .lsm_core import angular_feature_kernel, angular_kernel, complex_normal_like

NearFieldShape = Literal["mixed", "disk", "ellipse", "kite", "star"]
NearFieldSolver = Literal["richardson", "heavy_ball", "chebyshev", "pcg"]


@dataclass(frozen=True)
class NearFieldConfig:
    n_sensors: int = 24
    grid_size: int = 28
    domain_half_width: float = 0.8
    receiver_radius: float = 2.2
    source_radius: float = 2.4
    wavenumber: float = 8.0
    boundary_points_per_component: int = 20
    kernel_gamma: float = 1.0
    kernel_mix: float = 0.20
    receiver_noise_correlation: float = 0.18


def helmholtz_green(targets: Tensor, sources: Tensor, wavenumber: float) -> Tensor:
    """Two-dimensional outgoing Helmholtz fundamental solution."""
    difference = targets[..., :, None, :] - sources[..., None, :, :]
    distance = torch.linalg.vector_norm(difference, dim=-1).clamp_min(1.0e-5)
    argument = float(wavenumber) * distance
    hankel = torch.special.bessel_j0(argument) + 1j * torch.special.bessel_y0(argument)
    return (0.25j * hankel).to(torch.complex64)


class NearFieldSoundSoftLSM:
    """Online physical near-field tasks for one to six sound-soft obstacles."""

    def __init__(self, cfg: NearFieldConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
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
        self.base_angles = torch.arange(
            cfg.n_sensors, device=device, dtype=self.real_dtype
        ) * (2.0 * math.pi / cfg.n_sensors)

    @property
    def n_probes(self) -> int:
        return self.grid.shape[0]

    def acquisition_geometry(
        self,
        *,
        aperture_degrees: float = 360.0,
        jitter_fraction: float = 0.0,
        rotation: float = 0.0,
    ) -> dict[str, Tensor]:
        if aperture_degrees >= 359.9:
            receiver_angles = self.base_angles.clone()
        else:
            half = 0.5 * math.radians(aperture_degrees)
            receiver_angles = torch.linspace(
                -half,
                half,
                self.cfg.n_sensors,
                device=self.device,
                dtype=self.real_dtype,
            )
        if jitter_fraction > 0.0:
            gap = 2.0 * math.pi / self.cfg.n_sensors
            receiver_angles = (
                receiver_angles
                + (2.0 * torch.rand_like(receiver_angles) - 1.0) * jitter_fraction * gap
            )
            receiver_angles, _ = torch.sort(receiver_angles)
        receiver_angles = receiver_angles + float(rotation)
        source_angles = receiver_angles + math.pi / self.cfg.n_sensors
        receivers = self.cfg.receiver_radius * torch.stack(
            [torch.cos(receiver_angles), torch.sin(receiver_angles)], dim=-1
        )
        sources = self.cfg.source_radius * torch.stack(
            [torch.cos(source_angles), torch.sin(source_angles)], dim=-1
        )
        _, source_kernel = angular_kernel(
            source_angles,
            self.cfg.kernel_gamma,
            self.cfg.kernel_mix,
        )
        receiver_feature = angular_feature_kernel(
            receiver_angles, self.cfg.kernel_gamma
        )
        _, receiver_covariance = angular_kernel(
            receiver_angles,
            self.cfg.kernel_gamma,
            self.cfg.receiver_noise_correlation,
        )
        probe = helmholtz_green(receivers, self.grid, self.cfg.wavenumber)
        return {
            "receiver_angles": receiver_angles,
            "source_angles": source_angles,
            "receivers": receivers,
            "sources": sources,
            "source_kernel": source_kernel.to(self.complex_dtype),
            "receiver_feature": receiver_feature.to(self.complex_dtype),
            "noise_correlation": receiver_covariance.to(self.complex_dtype),
            "probe": probe,
        }

    def sample_obstacles(
        self,
        batch_size: int,
        count: int,
        *,
        mode: NearFieldShape = "mixed",
    ) -> tuple[Tensor, Tensor, Tensor]:
        if count < 1 or count > 6:
            raise ValueError("count must lie between one and six")
        points_per_component = self.cfg.boundary_points_per_component
        parameter = torch.arange(
            points_per_component, device=self.device, dtype=self.real_dtype
        ) * (2.0 * math.pi / points_per_component)
        task_center = (torch.rand(batch_size, 2, device=self.device) - 0.5) * 0.24
        base_angle = 2.0 * math.pi * torch.rand(batch_size, 1, device=self.device)
        mask = torch.zeros(
            batch_size, self.n_probes, device=self.device, dtype=torch.bool
        )
        boundaries = []
        fictitious_sources = []
        for component in range(count):
            if count == 1:
                center = task_center
            else:
                theta_center = (
                    base_angle[:, 0]
                    + 2.0 * math.pi * component / count
                    + 0.12 * torch.randn(batch_size, device=self.device)
                )
                ring = 0.15 + 0.042 * count
                center = task_center + ring * torch.stack(
                    [torch.cos(theta_center), torch.sin(theta_center)], dim=-1
                )
            radius = 0.075 + 0.040 * torch.rand(batch_size, 1, device=self.device)
            rotation = math.pi * torch.rand(batch_size, 1, device=self.device)
            if mode == "mixed":
                shape_code = torch.randint(0, 3, (batch_size, 1), device=self.device)
            else:
                code = {"disk": 0, "ellipse": 1, "kite": 2, "star": 3}[mode]
                shape_code = torch.full(
                    (batch_size, 1), code, device=self.device, dtype=torch.long
                )

            angle = parameter[None, :]
            radial_disk = radius.expand(-1, points_per_component)
            aspect = 0.65 + 0.70 * torch.rand(batch_size, 1, device=self.device)
            ellipse_x = radius * aspect * torch.cos(angle)
            ellipse_y = radius / aspect * torch.sin(angle)
            radial_kite = radius * (
                1.0 + 0.22 * torch.cos(3.0 * angle) + 0.08 * torch.sin(2.0 * angle)
            )
            radial_star = radius * (1.0 + 0.30 * torch.cos(5.0 * angle))
            radial = torch.where(
                shape_code == 2,
                radial_kite,
                torch.where(shape_code == 3, radial_star, radial_disk),
            )
            local_x = radial * torch.cos(angle)
            local_y = radial * torch.sin(angle)
            local_x = torch.where(shape_code == 1, ellipse_x, local_x)
            local_y = torch.where(shape_code == 1, ellipse_y, local_y)
            cosine = torch.cos(rotation)
            sine = torch.sin(rotation)
            boundary = (
                torch.stack(
                    [
                        cosine * local_x - sine * local_y,
                        sine * local_x + cosine * local_y,
                    ],
                    dim=-1,
                )
                + center[:, None, :]
            )
            fictitious = center[:, None, :] + 0.52 * (boundary - center[:, None, :])
            boundaries.append(boundary)
            fictitious_sources.append(fictitious)

            delta = self.grid[None, :, :] - center[:, None, :]
            local_grid_x = cosine * delta[..., 0] + sine * delta[..., 1]
            local_grid_y = -sine * delta[..., 0] + cosine * delta[..., 1]
            polar_angle = torch.atan2(local_grid_y, local_grid_x)
            polar_radius = torch.sqrt(local_grid_x.square() + local_grid_y.square())
            disk_mask = polar_radius <= radius
            ellipse_mask = (local_grid_x / (radius * aspect)).square() + (
                local_grid_y / (radius / aspect)
            ).square() <= 1.0
            kite_boundary = radius * (
                1.0
                + 0.22 * torch.cos(3.0 * polar_angle)
                + 0.08 * torch.sin(2.0 * polar_angle)
            )
            star_boundary = radius * (1.0 + 0.30 * torch.cos(5.0 * polar_angle))
            component_mask = torch.where(
                shape_code == 1,
                ellipse_mask,
                torch.where(
                    shape_code == 2,
                    polar_radius <= kite_boundary,
                    torch.where(
                        shape_code == 3, polar_radius <= star_boundary, disk_mask
                    ),
                ),
            )
            empty = ~component_mask.any(dim=-1)
            if empty.any():
                nearest = delta.square().sum(dim=-1).argmin(dim=-1)
                component_mask[empty, nearest[empty]] = True
            mask |= component_mask
        return mask, torch.cat(boundaries, dim=1), torch.cat(fictitious_sources, dim=1)

    @torch.no_grad()
    def simulate(
        self,
        batch_size: int,
        count: int,
        *,
        mode: NearFieldShape = "mixed",
        noise_rel: float = 0.10,
        aperture_degrees: float = 360.0,
        jitter_fraction: float = 0.0,
        rotation: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        geometry = self.acquisition_geometry(
            aperture_degrees=aperture_degrees,
            jitter_fraction=jitter_fraction,
            rotation=rotation,
        )
        mask, boundary, fictitious = self.sample_obstacles(batch_size, count, mode=mode)
        boundary_matrix = helmholtz_green(boundary, fictitious, self.cfg.wavenumber)
        incident_trace = helmholtz_green(
            boundary,
            geometry["sources"],
            self.cfg.wavenumber,
        )
        coefficients = torch.linalg.solve(boundary_matrix, -incident_trace)
        receiver_matrix = helmholtz_green(
            geometry["receivers"], fictitious, self.cfg.wavenumber
        )
        noiseless = receiver_matrix @ coefficients
        boundary_residual = torch.linalg.matrix_norm(
            boundary_matrix @ coefficients + incident_trace, ord="fro"
        ) / torch.linalg.matrix_norm(incident_trace, ord="fro").clamp_min(1.0e-12)

        correlation = geometry["noise_correlation"]
        eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
        correlation_sqrt = (
            eigenvectors * eigenvalues.clamp_min(1.0e-6).sqrt()[None, :]
        ) @ eigenvectors.mH
        correlation_inverse_sqrt = (
            eigenvectors * eigenvalues.clamp_min(1.0e-6).rsqrt()[None, :]
        ) @ eigenvectors.mH
        signal_rms = torch.linalg.matrix_norm(noiseless, ord="fro") / math.sqrt(
            self.cfg.n_sensors**2
        )
        noise_scale = (float(noise_rel) * signal_rms).clamp_min(1.0e-5)
        noise = correlation_sqrt @ complex_normal_like(noiseless)
        measured = noiseless + noise_scale[:, None, None] * noise
        whitened_near_field = (
            correlation_inverse_sqrt.unsqueeze(0) @ measured
        ) / noise_scale[:, None, None]
        whitened_probe = (correlation_inverse_sqrt @ geometry["probe"]).unsqueeze(
            0
        ) / noise_scale[:, None, None]
        whitened_probe = whitened_probe.expand(batch_size, -1, -1)
        diagnostics = {
            "boundary_residual": boundary_residual,
            "noise_scale": noise_scale,
            "noiseless_near_field": noiseless,
            "measured_near_field": measured,
            **geometry,
        }
        return (
            whitened_near_field,
            whitened_probe,
            geometry["source_kernel"],
            geometry["receiver_feature"],
            mask,
            diagnostics,
        )


def build_near_field_system(
    near_field: Tensor,
    source_kernel: Tensor,
    probe_rhs: Tensor,
) -> dict[str, Tensor]:
    batch_size, n_receivers, _ = near_field.shape
    if source_kernel.ndim == 2:
        kernel_batch = source_kernel.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        kernel_batch = source_kernel
    identity = torch.eye(n_receivers, device=near_field.device, dtype=near_field.dtype)
    hessian = near_field @ kernel_batch @ near_field.mH + identity
    row_bound = hessian.abs().sum(dim=-1).amax(dim=-1).clamp_min(1.0)
    operator = hessian / row_bound[:, None, None]
    rhs = probe_rhs / row_bound[:, None, None]
    return {
        "operator": operator,
        "rhs": rhs,
        "hessian": hessian,
        "row_bound": row_bound,
        "safe_lower": row_bound.reciprocal(),
        "safe_upper": torch.ones_like(row_bound),
        "kernel": kernel_batch,
    }


def near_field_score(
    near_field: Tensor,
    source_kernel: Tensor,
    q: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size = near_field.shape[0]
    if source_kernel.ndim == 2:
        kernel_batch = source_kernel.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        kernel_batch = source_kernel
    coefficients = kernel_batch @ near_field.mH @ q
    inverse_coefficients = torch.linalg.solve(kernel_batch, coefficients)
    energy = (
        (coefficients.conj() * inverse_coefficients).sum(dim=1).real.clamp_min(1.0e-12)
    )
    return -0.5 * torch.log(energy), coefficients


class GeometryPopulationFactor(nn.Module):
    """Geometry-conditioned shared covariance whitening in the kernel basis."""

    def __init__(self, width: int = 48) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("population-factor width must be at least four")
        self.gain = nn.Sequential(
            nn.Linear(3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.constant_(self.gain[-1].bias, math.log(math.e - 1.0))

    def forward(
        self, receiver_feature: Tensor, batch_size: int
    ) -> tuple[Tensor, Tensor]:
        if receiver_feature.ndim != 2:
            raise ValueError("one acquisition geometry must be shared by a batch")
        eigenvalues, eigenvectors = torch.linalg.eigh(receiver_feature.real)
        gaps = torch.diff(eigenvalues, prepend=eigenvalues[:1])
        features = torch.stack(
            [eigenvalues, torch.log(eigenvalues.clamp_min(1.0e-6)), gaps], dim=-1
        )
        gains = F.softplus(self.gain(features).squeeze(-1)).clamp(0.05, 20.0)
        factor = (
            eigenvectors.to(receiver_feature.dtype) * gains.sqrt()[None, :]
        ) @ eigenvectors.to(receiver_feature.dtype).mH
        return factor.unsqueeze(0).expand(batch_size, -1, -1), gains


class ContextConditionedPopulationFactor(nn.Module):
    """A prompt-dependent SPD factor fixed throughout one PCG solve.

    The shared angular-kernel eigenbasis supplies the population geometry.  A
    permutation-equivariant token encoder reads task-specific Hessian
    statistics in that basis and changes only the positive modal gains.  The
    resulting factor is therefore SPD, depends on the current near-field
    prompt, and remains constant across recurrent depth, as required by
    standard PCG rather than flexible CG.
    """

    def __init__(self, width: int = 48, *, analytic_base: bool = False) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("context-factor width must be at least four")
        self.analytic_base = bool(analytic_base)
        self.population = (
            None if self.analytic_base else GeometryPopulationFactor(width)
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(6, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.correction = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(
        self, receiver_feature: Tensor, hessian: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if receiver_feature.ndim != 2:
            raise ValueError("one acquisition geometry must be shared by a batch")
        if hessian.ndim != 3 or hessian.shape[-1] != receiver_feature.shape[-1]:
            raise ValueError("hessian and receiver geometry are incompatible")
        batch_size, context_size, _ = hessian.shape
        eigenvalues, eigenvectors_real = torch.linalg.eigh(receiver_feature.real)
        eigenvectors = eigenvectors_real.to(hessian.dtype)
        rotated = eigenvectors.mH.unsqueeze(0) @ hessian @ eigenvectors.unsqueeze(0)
        diagonal = rotated.diagonal(dim1=-2, dim2=-1).real.clamp_min(1.0e-8)
        row_l1 = rotated.abs().sum(dim=-1).clamp_min(1.0e-8)
        row_l2 = rotated.abs().square().sum(dim=-1).sqrt().clamp_min(1.0e-8)
        if self.analytic_base:
            scale = torch.exp(torch.log(diagonal).mean(dim=-1, keepdim=True))
            population_gains = (scale / diagonal).clamp(0.02, 50.0)
        else:
            assert self.population is not None
            _, shared_gains = self.population(receiver_feature, batch_size)
            population_gains = shared_gains.unsqueeze(0).expand(batch_size, -1)

        def relative_log(values: Tensor) -> Tensor:
            scale = values.mean(dim=-1, keepdim=True).clamp_min(1.0e-8)
            return torch.log(values / scale)

        gaps = torch.diff(eigenvalues, prepend=eigenvalues[:1])
        geometry_features = torch.stack(
            [
                eigenvalues,
                torch.log(eigenvalues.clamp_min(1.0e-6)),
                gaps,
            ],
            dim=-1,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        task_features = torch.stack(
            [relative_log(diagonal), relative_log(row_l1), relative_log(row_l2)],
            dim=-1,
        )
        token_features = self.token_encoder(
            torch.cat([geometry_features, task_features], dim=-1)
        )
        pooled = token_features.mean(dim=1, keepdim=True).expand(
            -1, context_size, -1
        )
        # A residual model should refine, not erase, the analytic GP-basis
        # preconditioner.  The trust region limits one prompt-wise correction
        # to a multiplicative factor in [exp(-1/2), exp(1/2)].
        correction_radius = 0.5 if self.analytic_base else 2.0
        log_correction = correction_radius * torch.tanh(
            self.correction(torch.cat([token_features, pooled], dim=-1)).squeeze(-1)
        )
        gains = (population_gains * torch.exp(log_correction)).clamp(0.02, 50.0)
        factor = (
            eigenvectors.unsqueeze(0) * gains.sqrt().unsqueeze(1)
        ) @ eigenvectors.mH.unsqueeze(0)
        return factor, gains, population_gains


class PosteriorMomentLSMLoop(nn.Module):
    """Parallel posterior-mean/covariance loops for original near-field LSM.

    The two right-hand-side blocks implement the posterior predictive moments:

        H Q_mu = Phi,       H Q_Sigma = F K,
        m = K F* Q_mu,      Sigma = K - K F* Q_Sigma.

    Both blocks use exactly the same population factor and tied heavy-ball
    coefficients.  A separate RHS-weighted Krylov sketch is only an in-context
    spectral statistic used by the endpoint controller; it is not called a
    posterior moment and never replaces either Bayesian solve.
    """

    def __init__(
        self,
        source_kernel: Tensor,
        receiver_feature: Tensor,
        n_probes: int,
        depth: int = 20,
        *,
        moment_degree: int = 6,
        sketch_size: int = 6,
        controller_width: int = 128,
        population_width: int = 48,
        use_population_factor: bool = True,
        task_conditioned_factor: bool = False,
        analytic_context_factor: bool = False,
        context_adaptive_sketch: bool = False,
        method: NearFieldSolver = "heavy_ball",
    ) -> None:
        super().__init__()
        if method not in ("richardson", "heavy_ball", "chebyshev", "pcg"):
            raise ValueError(f"unknown near-field solver: {method}")
        self.register_buffer("source_kernel", source_kernel.detach().clone())
        self.register_buffer("receiver_feature", receiver_feature.detach().clone())
        generator = torch.Generator(device="cpu").manual_seed(20260804)
        sketch = torch.randn(n_probes, sketch_size, generator=generator)
        sketch, _ = torch.linalg.qr(sketch, mode="reduced")
        self.register_buffer("probe_sketch", sketch)
        covariance_sketch = torch.randn(
            source_kernel.shape[-1], sketch_size, generator=generator
        )
        covariance_sketch, _ = torch.linalg.qr(covariance_sketch, mode="reduced")
        self.register_buffer("covariance_sketch", covariance_sketch)
        self.depth = int(depth)
        self.moment_degree = int(moment_degree)
        self.sketch_size = int(sketch_size)
        self.use_population_factor = bool(use_population_factor)
        self.task_conditioned_factor = bool(task_conditioned_factor)
        self.analytic_context_factor = bool(analytic_context_factor)
        if self.analytic_context_factor and not self.task_conditioned_factor:
            raise ValueError("analytic context base requires task conditioning")
        self.context_adaptive_sketch = bool(context_adaptive_sketch)
        self.method: NearFieldSolver = method
        self.population: nn.Module
        if self.task_conditioned_factor:
            self.population = ContextConditionedPopulationFactor(
                width=population_width,
                analytic_base=self.analytic_context_factor,
            )
        else:
            self.population = GeometryPopulationFactor(width=population_width)
        joint_sketch_size = 2 * sketch_size
        feature_dimension = (
            moment_degree + 1
        ) * 2 * joint_sketch_size * joint_sketch_size + 4
        self.controller = nn.Sequential(
            nn.Linear(feature_dimension, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, 2),
        )
        nn.init.zeros_(self.controller[-1].weight)
        with torch.no_grad():
            self.controller[-1].bias.copy_(torch.tensor([-8.0, 8.0]))

    def _covariance_context_sketch(
        self,
        context_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return a deterministic orthonormal sketch for any context length.

        The legacy Gaussian buffer is retained for checkpoint compatibility.
        Scaling-law models opt into the harmonic sketch, whose column count is
        fixed while its row count follows the number of source/receiver tokens.
        This lets one tied encoder operate on several context lengths without
        padding the physical near-field operator.
        """
        if not self.context_adaptive_sketch:
            if context_size != self.covariance_sketch.shape[0]:
                raise ValueError(
                    "context length differs from the legacy covariance sketch; "
                    "construct the model with context_adaptive_sketch=True"
                )
            return self.covariance_sketch.to(device=device, dtype=dtype)
        if context_size < self.sketch_size:
            raise ValueError("context length must be at least the sketch size")
        positions = torch.arange(context_size, device=device, dtype=torch.float32)
        modes = torch.arange(self.sketch_size, device=device, dtype=torch.float32)
        sketch = torch.cos(
            math.pi * (positions[:, None] + 0.5) * modes[None, :] / float(context_size)
        )
        sketch, _ = torch.linalg.qr(sketch, mode="reduced")
        return sketch.to(dtype=dtype)

    def _population_transform(
        self,
        system: dict[str, Tensor],
        receiver_feature: Tensor,
        covariance_rhs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        hessian = system["hessian"]
        mean_rhs_unscaled = system["rhs"] * system["row_bound"][:, None, None]
        batch_size, n_receivers, _ = hessian.shape
        if self.use_population_factor:
            if self.task_conditioned_factor:
                factor, gains, base_gains = self.population(receiver_feature, hessian)
            else:
                factor, gains = self.population(receiver_feature, batch_size)
                base_gains = gains
        else:
            identity = torch.eye(
                n_receivers, device=hessian.device, dtype=hessian.dtype
            )
            factor = identity.unsqueeze(0).expand(batch_size, -1, -1)
            gains = torch.ones(n_receivers, device=hessian.device)
            base_gains = gains
        transformed_hessian = factor.mH @ hessian @ factor
        row_bound = transformed_hessian.abs().sum(dim=-1).amax(dim=-1).clamp_min(1.0)
        operator = transformed_hessian / row_bound[:, None, None]
        mean_rhs = factor.mH @ mean_rhs_unscaled / row_bound[:, None, None]
        covariance_rhs = factor.mH @ covariance_rhs / row_bound[:, None, None]
        safe_lower = (
            torch.linalg.eigvalsh(factor.mH @ factor).amin(dim=-1) / row_bound
        ).clamp_min(1.0e-7)
        return operator, mean_rhs, covariance_rhs, factor, gains, base_gains, safe_lower

    def _moment_features(
        self,
        operator: Tensor,
        mean_rhs: Tensor,
        covariance_rhs: Tensor,
        safe_lower: Tensor,
        gains: Tensor,
    ) -> Tensor:
        mean_sketch = self.probe_sketch.to(mean_rhs.device, mean_rhs.real.dtype)
        covariance_sketch = self._covariance_context_sketch(
            covariance_rhs.shape[-1],
            device=covariance_rhs.device,
            dtype=covariance_rhs.real.dtype,
        )
        z0 = torch.cat(
            [
                mean_rhs @ mean_sketch.to(mean_rhs.dtype),
                covariance_rhs @ covariance_sketch.to(covariance_rhs.dtype),
            ],
            dim=-1,
        )
        z = z0
        normalization = z0.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        moments = []
        for _ in range(self.moment_degree + 1):
            moment = z0.mH @ z / normalization[:, None, None]
            moments.extend(
                [moment.real.flatten(start_dim=1), moment.imag.flatten(start_dim=1)]
            )
            z = operator @ z
        if gains.ndim == 1:
            gain_statistics = (
                torch.stack(
                    [
                        gains.mean(),
                        gains.std(unbiased=False),
                        gains.amin(),
                        gains.amax(),
                    ]
                )
                .unsqueeze(0)
                .expand(operator.shape[0], -1)
            )
        else:
            gain_statistics = torch.stack(
                [
                    gains.mean(dim=-1),
                    gains.std(dim=-1, unbiased=False),
                    gains.amin(dim=-1),
                    gains.amax(dim=-1),
                ],
                dim=-1,
            )
        return torch.cat(
            [*moments, torch.log(safe_lower)[:, None], gain_statistics[:, :3]], dim=-1
        )

    @staticmethod
    def hb_coefficients(lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        square_root_lower = torch.sqrt(lower.clamp_min(1.0e-9))
        square_root_upper = torch.sqrt(upper.clamp_min(1.0e-9))
        denominator = (square_root_upper + square_root_lower).square()
        alpha = 4.0 / denominator.clamp_min(1.0e-9)
        beta = (
            (square_root_upper - square_root_lower)
            / (square_root_upper + square_root_lower).clamp_min(1.0e-9)
        ).square()
        return alpha, beta

    @staticmethod
    def _iterate(
        operator: Tensor,
        rhs: Tensor,
        lower: Tensor,
        upper: Tensor,
        depth: int,
        *,
        method: NearFieldSolver,
        return_history: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor]:
        iterate = torch.zeros_like(rhs)
        previous = torch.zeros_like(rhs)
        rhs_energy = rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        history = []
        if method == "pcg":
            residual = rhs.clone()
            direction = residual.clone()
            residual_energy = residual.abs().square().sum(dim=1).clamp_min(1.0e-20)
            alpha_columns = torch.zeros_like(residual_energy)
            beta_columns = torch.zeros_like(residual_energy)
            for _ in range(depth):
                operator_direction = operator @ direction
                denominator = (
                    (direction.conj() * operator_direction).sum(dim=1).real
                ).clamp_min(1.0e-20)
                alpha_columns = residual_energy / denominator
                iterate = iterate + alpha_columns[:, None, :] * direction
                residual = residual - alpha_columns[:, None, :] * operator_direction
                next_energy = residual.abs().square().sum(dim=1).clamp_min(1.0e-20)
                beta_columns = next_energy / residual_energy
                direction = residual + beta_columns[:, None, :] * direction
                residual_energy = next_energy
                if return_history:
                    history.append(
                        torch.sqrt(residual.abs().square().sum(dim=(1, 2)) / rhs_energy)
                    )
            relative = torch.sqrt(residual.abs().square().sum(dim=(1, 2)) / rhs_energy)
            stacked = torch.stack(history, dim=1) if return_history else None
            return (
                iterate,
                relative,
                stacked,
                alpha_columns.mean(dim=-1),
                beta_columns.mean(dim=-1),
            )
        if method == "richardson":
            alpha = 2.0 / (lower + upper).clamp_min(1.0e-9)
            beta = torch.zeros_like(alpha)
            alpha_previous = alpha
        elif method == "heavy_ball":
            alpha, beta = PosteriorMomentLSMLoop.hb_coefficients(lower, upper)
            alpha_previous = alpha
        else:
            center = 0.5 * (upper + lower)
            radius = 0.5 * (upper - lower)
            alpha_previous = center.reciprocal()
            alpha = alpha_previous
            beta = torch.zeros_like(alpha)
        for step in range(depth):
            residual = rhs - operator @ iterate
            if method == "chebyshev":
                if step == 0:
                    alpha = alpha_previous
                    beta = torch.zeros_like(alpha)
                else:
                    beta = (0.5 * radius * alpha_previous).square()
                    alpha = (center - beta / alpha_previous).reciprocal()
            following = (
                iterate
                + alpha[:, None, None] * residual
                + beta[:, None, None] * (iterate - previous)
            )
            previous, iterate = iterate, following
            alpha_previous = alpha
            if return_history:
                next_residual = rhs - operator @ iterate
                history.append(
                    torch.sqrt(
                        next_residual.abs().square().sum(dim=(1, 2)) / rhs_energy
                    )
                )
        residual = rhs - operator @ iterate
        relative = torch.sqrt(residual.abs().square().sum(dim=(1, 2)) / rhs_energy)
        stacked = torch.stack(history, dim=1) if return_history else None
        return iterate, relative, stacked, alpha, beta

    def forward(
        self,
        near_field: Tensor,
        probe_rhs: Tensor,
        *,
        source_kernel: Tensor | None = None,
        receiver_feature: Tensor | None = None,
        depth: int | None = None,
        return_history: bool = False,
        force_safe: bool = False,
        force_oracle: bool = False,
        certify: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        active_kernel = self.source_kernel if source_kernel is None else source_kernel
        active_feature = (
            self.receiver_feature if receiver_feature is None else receiver_feature
        )
        system = build_near_field_system(near_field, active_kernel, probe_rhs)
        kernel_batch = system["kernel"]
        covariance_rhs_unscaled = near_field @ kernel_batch
        (
            operator,
            mean_rhs,
            covariance_rhs,
            population_factor,
            population_gains,
            base_population_gains,
            safe_lower,
        ) = self._population_transform(system, active_feature, covariance_rhs_unscaled)
        needs_endpoint_controller = (
            self.method != "pcg" or self.training or certify or force_oracle
        )
        if needs_endpoint_controller:
            features = self._moment_features(
                operator,
                mean_rhs,
                covariance_rhs,
                safe_lower,
                population_gains,
            )
            raw_lower, raw_upper = self.controller(features).unbind(dim=-1)
            predicted_lower = safe_lower + (1.0 - safe_lower) * torch.sigmoid(
                raw_lower
            )
            predicted_upper = predicted_lower + (
                1.0 - predicted_lower
            ) * torch.sigmoid(raw_upper)
        else:
            features = operator.real.new_empty((operator.shape[0], 0))
            predicted_lower = safe_lower
            predicted_upper = torch.ones_like(safe_lower)
        eigenvalues = None
        if certify or force_oracle:
            eigenvalues = torch.linalg.eigvalsh(operator)
            true_lower = eigenvalues.amin(dim=-1)
            true_upper = eigenvalues.amax(dim=-1)
        if force_oracle:
            used_lower = true_lower
            used_upper = true_upper
        elif force_safe:
            used_lower = safe_lower
            used_upper = torch.ones_like(safe_lower)
        else:
            used_lower = predicted_lower
            used_upper = predicted_upper
        n_steps = self.depth if depth is None else int(depth)
        n_mean_rhs = mean_rhs.shape[-1]
        joint_rhs = torch.cat([mean_rhs, covariance_rhs], dim=-1)
        joint_iterate, relative_residual, history, alpha, beta = self._iterate(
            operator,
            joint_rhs,
            used_lower,
            used_upper,
            n_steps,
            method=self.method,
            return_history=return_history,
        )
        mean_iterate = joint_iterate[..., :n_mean_rhs]
        covariance_iterate = joint_iterate[..., n_mean_rhs:]
        q_mean = population_factor @ mean_iterate
        q_covariance = population_factor @ covariance_iterate
        plug_in_score, mean_coefficients = near_field_score(
            near_field, active_kernel, q_mean
        )
        covariance = posterior_covariance(near_field, active_kernel, q_covariance)
        score_mean, score_std = posterior_score_moments(
            mean_coefficients, covariance, active_kernel
        )

        mean_residual = mean_rhs - operator @ mean_iterate
        covariance_residual = covariance_rhs - operator @ covariance_iterate
        mean_relative_residual = torch.sqrt(
            mean_residual.abs().square().sum(dim=(1, 2))
            / mean_rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        )
        covariance_relative_residual = torch.sqrt(
            covariance_residual.abs().square().sum(dim=(1, 2))
            / covariance_rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        )
        original_mean_residual = probe_rhs - system["hessian"] @ q_mean
        original_covariance_residual = (
            covariance_rhs_unscaled - system["hessian"] @ q_covariance
        )
        original_mean_relative_residual = torch.sqrt(
            original_mean_residual.abs().square().sum(dim=(1, 2))
            / probe_rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        )
        original_covariance_relative_residual = torch.sqrt(
            original_covariance_residual.abs().square().sum(dim=(1, 2))
            / covariance_rhs_unscaled.abs()
            .square()
            .sum(dim=(1, 2))
            .clamp_min(1.0e-12)
        )
        info: dict[str, Tensor] = {
            "q": q_mean,
            "q_mean": q_mean,
            "q_covariance": q_covariance,
            "coefficients": mean_coefficients,
            "mean_coefficients": mean_coefficients,
            "posterior_covariance": covariance,
            "plug_in_score": plug_in_score,
            "score_mean": score_mean,
            "score_std": score_std,
            "relative_residual": relative_residual,
            "mean_relative_residual": mean_relative_residual,
            "covariance_relative_residual": covariance_relative_residual,
            "transformed_mean_relative_residual": mean_relative_residual,
            "transformed_covariance_relative_residual": (
                covariance_relative_residual
            ),
            "original_mean_relative_residual": original_mean_relative_residual,
            "original_covariance_relative_residual": (
                original_covariance_relative_residual
            ),
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
            "safe_lower": safe_lower,
            "used_lower": used_lower,
            "used_upper": used_upper,
            "alpha": alpha,
            "beta": beta,
            "operator": operator,
            "population_factor": population_factor,
            "population_gains": population_gains,
            "base_population_gains": base_population_gains,
            "moment_features": features,
        }
        if certify or force_oracle:
            certified = (predicted_lower <= true_lower) & (
                predicted_upper >= true_upper
            )
            info.update(
                {
                    "true_lower": true_lower,
                    "true_upper": true_upper,
                    "certified": certified,
                    "lower_violation": (predicted_lower - true_lower).clamp_min(0.0),
                    "upper_violation": (true_upper - predicted_upper).clamp_min(0.0),
                }
            )
        if history is not None:
            info["residual_history"] = history
        return score_mean, info


# Convenient and backward-compatible names.  The architecture now computes
# posterior predictive moments; "Krylov moments" only names its endpoint-
# controller statistic.
PosteriorMomentHBLSMLoop = PosteriorMomentLSMLoop
KrylovMomentHBLSMLoop = PosteriorMomentLSMLoop


@torch.no_grad()
def exact_near_field_lsm(
    near_field: Tensor,
    probe_rhs: Tensor,
    source_kernel: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    system = build_near_field_system(near_field, source_kernel, probe_rhs)
    covariance_rhs = near_field @ system["kernel"]
    n_mean_rhs = probe_rhs.shape[-1]
    joint_rhs = torch.cat([probe_rhs, covariance_rhs], dim=-1)
    joint_q = torch.linalg.solve(system["hessian"], joint_rhs)
    q_mean = joint_q[..., :n_mean_rhs]
    q_covariance = joint_q[..., n_mean_rhs:]
    plug_in_score, mean_coefficients = near_field_score(
        near_field, source_kernel, q_mean
    )
    covariance = posterior_covariance(near_field, source_kernel, q_covariance)
    score_mean, score_std = posterior_score_moments(
        mean_coefficients, covariance, source_kernel
    )
    mean_residual = system["hessian"] @ q_mean - probe_rhs
    covariance_residual = system["hessian"] @ q_covariance - covariance_rhs
    mean_relative = torch.sqrt(
        mean_residual.abs().square().sum(dim=(1, 2))
        / probe_rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
    )
    covariance_relative = torch.sqrt(
        covariance_residual.abs().square().sum(dim=(1, 2))
        / covariance_rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
    )
    relative = torch.sqrt(
        (
            mean_residual.abs().square().sum(dim=(1, 2))
            + covariance_residual.abs().square().sum(dim=(1, 2))
        )
        / (
            probe_rhs.abs().square().sum(dim=(1, 2))
            + covariance_rhs.abs().square().sum(dim=(1, 2))
        ).clamp_min(1.0e-12)
    )
    return score_mean, {
        "q": q_mean,
        "q_mean": q_mean,
        "q_covariance": q_covariance,
        "coefficients": mean_coefficients,
        "mean_coefficients": mean_coefficients,
        "posterior_covariance": covariance,
        "plug_in_score": plug_in_score,
        "score_mean": score_mean,
        "score_std": score_std,
        "relative_residual": relative,
        "mean_relative_residual": mean_relative,
        "covariance_relative_residual": covariance_relative,
        "transformed_mean_relative_residual": mean_relative,
        "transformed_covariance_relative_residual": covariance_relative,
        "original_mean_relative_residual": mean_relative,
        "original_covariance_relative_residual": covariance_relative,
    }
