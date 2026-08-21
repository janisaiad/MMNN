"""Larger fixed-kernel LSM controller and analytic Bayesian uncertainty tools.

The model in this module remains a solver, not an image regressor.  A
kernel-message-passing encoder reads the scaled Bayesian LSM operator once and
produces an SPD polynomial preconditioner.  Complex PCG and the GP/LSM decoder
are otherwise exact and tied across depth.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .lsm_core import (
    FixedGeometryBornLSM,
    build_bayesian_system,
    prior_score,
    system_summary_features,
)

MultiShapeMode = Literal[
    "mixed",
    "disk",
    "ellipse",
    "kite",
    "star",
    "crescent",
    "mirrored_kites",
]
CoefficientMode = Literal["learned", "analytic"]


class KernelMessageBlock(nn.Module):
    """Residual message passing through a prescribed angular kernel."""

    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.update = nn.Sequential(
            nn.Linear(2 * width, expansion),
            nn.GELU(),
            nn.Linear(expansion, width),
        )

    def forward(self, tokens: Tensor, attention: Tensor) -> Tensor:
        messages = attention @ tokens
        update = self.update(torch.cat([self.norm(tokens), messages], dim=-1))
        return tokens + update


class FoundationPCGLSMLoop(nn.Module):
    """Kernel-conditioned PCG foundation controller for Bayesian LSM.

    The fixed softmax feature kernel supplies every message-passing edge.  The
    encoder produces nonnegative coefficients for

        P_D = I + sum_j c_j (L_Gamma / 2)^j,

    so the learned preconditioner is SPD.  The same preconditioner and exact
    complex-PCG cell are reused at every loop depth.
    """

    def __init__(
        self,
        kernel: Tensor,
        feature_kernel: Tensor,
        ridge_rel: float,
        depth: int = 16,
        *,
        width: int = 192,
        n_blocks: int = 6,
        expansion: int = 384,
        polynomial_degree: int = 8,
    ) -> None:
        super().__init__()
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("kernel must be a square matrix")
        self.register_buffer("kernel", kernel.detach().clone())
        self.register_buffer("feature_kernel", feature_kernel.detach().clone())
        self.register_buffer("ridge_rel", torch.tensor(float(ridge_rel)))
        self.depth = int(depth)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.polynomial_degree = int(polynomial_degree)
        n_angles = kernel.shape[0]
        token_features = 3 * n_angles + 5
        self.input_projection = nn.Linear(token_features, width)
        self.blocks = nn.ModuleList(
            KernelMessageBlock(width, expansion) for _ in range(n_blocks)
        )
        self.output_norm = nn.LayerNorm(width)
        self.controller = nn.Sequential(
            nn.Linear(3 * width + 6, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, polynomial_degree),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.constant_(self.controller[-1].bias, -5.0)

    @property
    def geometry_is_frozen(self) -> bool:
        return not self.kernel.requires_grad and not self.feature_kernel.requires_grad

    @staticmethod
    def _as_batch(matrix: Tensor, batch_size: int) -> Tensor:
        if matrix.ndim == 2:
            return matrix.unsqueeze(0).expand(batch_size, -1, -1)
        return matrix

    def _encode_preconditioner(
        self,
        system: dict[str, Tensor],
        feature_kernel: Tensor,
    ) -> tuple[Tensor, Tensor]:
        operator = system["operator"]
        batch_size, n_angles, _ = operator.shape
        feature = self._as_batch(feature_kernel, batch_size).real
        attention = feature / feature.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        diagonal = operator.diagonal(dim1=-2, dim2=-1).real
        row_abs = operator.abs()
        row_sum = row_abs.sum(dim=-1)
        row_energy = row_abs.square().sum(dim=-1).sqrt()
        row_phase = torch.angle(operator).mean(dim=-1)
        ridge_fraction = system["ridge"] / (
            system["hessian"]
            .diagonal(dim1=-2, dim2=-1)
            .real.mean(dim=-1)
            .clamp_min(1.0e-8)
        )
        ridge_tokens = torch.log(ridge_fraction.clamp_min(1.0e-8))[:, None].expand(
            -1, n_angles
        )
        token_features = torch.cat(
            [
                operator.real,
                operator.imag,
                row_abs,
                diagonal[..., None],
                row_sum[..., None],
                row_energy[..., None],
                row_phase[..., None],
                ridge_tokens[..., None],
            ],
            dim=-1,
        )
        tokens = self.input_projection(token_features)
        for block in self.blocks:
            tokens = block(tokens, attention)
        tokens = self.output_norm(tokens)
        pooled = torch.cat(
            [
                tokens.mean(dim=1),
                tokens.std(dim=1, unbiased=False),
                tokens.amax(dim=1),
                system_summary_features(system),
            ],
            dim=-1,
        )
        coefficients = 4.0 * torch.sigmoid(self.controller(pooled))

        identity = torch.eye(n_angles, device=operator.device, dtype=operator.dtype)
        laplacian = identity.unsqueeze(0) - feature.to(operator.dtype)
        scaled_laplacian = 0.5 * laplacian
        power = scaled_laplacian
        preconditioner = identity.unsqueeze(0).expand(batch_size, -1, -1).clone()
        for degree in range(self.polynomial_degree):
            preconditioner = (
                preconditioner + coefficients[:, degree, None, None] * power
            )
            power = power @ scaled_laplacian
        diagonal_mean = preconditioner.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
        preconditioner = preconditioner / diagonal_mean[:, None, None].clamp_min(1.0e-8)
        return preconditioner, coefficients

    @staticmethod
    def _pcg(
        operator: Tensor,
        rhs: Tensor,
        preconditioner: Tensor,
        n_steps: int,
        *,
        return_history: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        iterate = torch.zeros_like(rhs)
        residual = rhs.clone()
        preconditioned = preconditioner @ residual
        direction = preconditioned.clone()
        rz = (residual.conj() * preconditioned).sum(dim=1).real.clamp_min(1.0e-12)
        rhs_energy = rhs.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        history: list[Tensor] = []
        for _ in range(n_steps):
            operator_direction = operator @ direction
            denominator = (direction.conj() * operator_direction).sum(dim=1).real
            alpha = rz / denominator.clamp_min(1.0e-12)
            iterate = iterate + alpha[:, None, :] * direction
            residual = residual - alpha[:, None, :] * operator_direction
            preconditioned = preconditioner @ residual
            rz_next = (
                (residual.conj() * preconditioned).sum(dim=1).real.clamp_min(1.0e-12)
            )
            beta = rz_next / rz.clamp_min(1.0e-12)
            direction = preconditioned + beta[:, None, :] * direction
            rz = rz_next
            if return_history:
                residual_energy = residual.abs().square().sum(dim=(1, 2))
                history.append(torch.sqrt(residual_energy / rhs_energy))
        residual_energy = residual.abs().square().sum(dim=(1, 2))
        relative_residual = torch.sqrt(residual_energy / rhs_energy)
        stacked = torch.stack(history, dim=1) if return_history else None
        return iterate, relative_residual, stacked

    def forward(
        self,
        far_field: Tensor,
        probe_rhs: Tensor,
        *,
        kernel: Tensor | None = None,
        feature_kernel: Tensor | None = None,
        depth: int | None = None,
        return_history: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        active_kernel = self.kernel if kernel is None else kernel
        active_feature = self.feature_kernel if feature_kernel is None else feature_kernel
        system = build_bayesian_system(
            far_field,
            active_kernel,
            probe_rhs,
            float(self.ridge_rel.item()),
        )
        preconditioner, coefficients = self._encode_preconditioner(
            system, active_feature
        )
        n_steps = self.depth if depth is None else int(depth)
        iterate, relative_residual, history = self._pcg(
            system["operator"],
            system["rhs"],
            preconditioner,
            n_steps,
            return_history=return_history,
        )
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
        if history is not None:
            info["residual_history"] = history
        return score, info


class EquivariantChebyshevPCGLSMLoop(nn.Module):
    """Identifiable A-equivariant inverse-square-root preconditioner.

    The current context supplies every eigendirection through powers of its LSM
    operator A.  The controller can only select the scalar Chebyshev spectral
    law.  With C_theta(A) built by a tied recurrence, P=C C* is PSD and the
    witness objective identifies A^{-1/2} up to a unitary factor.
    """

    def __init__(
        self,
        kernel: Tensor,
        feature_kernel: Tensor,
        ridge_rel: float,
        depth: int = 12,
        *,
        polynomial_degree: int = 8,
        moment_degree: int = 8,
        controller_width: int = 96,
        coefficient_mode: CoefficientMode = "learned",
    ) -> None:
        super().__init__()
        if coefficient_mode not in ("learned", "analytic"):
            raise ValueError(f"unknown coefficient mode: {coefficient_mode}")
        self.register_buffer("kernel", kernel.detach().clone())
        self.register_buffer("feature_kernel", feature_kernel.detach().clone())
        self.register_buffer("ridge_rel", torch.tensor(float(ridge_rel)))
        self.depth = int(depth)
        self.polynomial_degree = int(polynomial_degree)
        self.moment_degree = int(moment_degree)
        self.coefficient_mode: CoefficientMode = coefficient_mode
        self.controller = nn.Sequential(
            nn.Linear(2 * moment_degree + 2, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, polynomial_degree + 1),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    @property
    def geometry_is_frozen(self) -> bool:
        return not self.kernel.requires_grad and not self.feature_kernel.requires_grad

    @staticmethod
    def _safe_interval(system: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        diagonal = system["hessian"].diagonal(dim1=-2, dim2=-1).real
        lower = system["ridge"] / diagonal.amax(dim=-1).clamp_min(1.0e-8)
        lower = (lower / system["row_bound"]).clamp_min(1.0e-6)
        upper = torch.ones_like(lower)
        lower = torch.minimum(lower, 0.95 * upper)
        return lower, upper

    def _spectral_features(
        self,
        operator: Tensor,
        lower: Tensor,
        upper: Tensor,
    ) -> Tensor:
        n_angles = operator.shape[-1]
        power = operator
        moments = []
        for _ in range(self.moment_degree):
            moment = power.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / n_angles
            moments.append(moment.clamp_min(1.0e-10))
            power = power @ operator
        moment_tensor = torch.stack(moments, dim=-1)
        return torch.cat(
            [
                moment_tensor,
                torch.log(moment_tensor),
                torch.log(lower)[:, None],
                upper[:, None],
            ],
            dim=-1,
        )

    def _analytic_coefficients(self, lower: Tensor, upper: Tensor) -> Tensor:
        quadrature = max(64, 8 * (self.polynomial_degree + 1))
        index = torch.arange(quadrature, device=lower.device, dtype=lower.dtype)
        theta = math.pi * (index + 0.5) / quadrature
        nodes = torch.cos(theta)
        center = 0.5 * (upper + lower)
        radius = 0.5 * (upper - lower)
        spectrum = center[:, None] + radius[:, None] * nodes[None, :]
        values = spectrum.clamp_min(1.0e-8).rsqrt()
        degrees = torch.arange(
            self.polynomial_degree + 1,
            device=lower.device,
            dtype=lower.dtype,
        )
        basis = torch.cos(degrees[:, None] * theta[None, :])
        coefficients = (2.0 / quadrature) * (values @ basis.T)
        coefficients[:, 0] *= 0.5
        return coefficients

    def _learned_coefficients(
        self,
        operator: Tensor,
        lower: Tensor,
        upper: Tensor,
    ) -> Tensor:
        features = self._spectral_features(operator, lower, upper)
        raw = self.controller(features)
        scale = 0.35 * lower.rsqrt().clamp(max=40.0)
        coefficients = scale[:, None] * torch.tanh(raw)
        coefficients[:, 0] = coefficients[:, 0] + 1.0
        return coefficients

    def _factor(
        self,
        system: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        operator = system["operator"]
        batch_size, n_angles, _ = operator.shape
        lower, upper = self._safe_interval(system)
        if self.coefficient_mode == "analytic":
            coefficients = self._analytic_coefficients(lower, upper)
        else:
            coefficients = self._learned_coefficients(operator, lower, upper)
        identity = torch.eye(n_angles, device=operator.device, dtype=operator.dtype)
        center = 0.5 * (upper + lower)
        radius = 0.5 * (upper - lower).clamp_min(1.0e-6)
        scaled = (
            operator - center[:, None, None] * identity
        ) / radius[:, None, None]
        chebyshev_previous = identity.unsqueeze(0).expand(batch_size, -1, -1)
        factor = coefficients[:, :1, None] * chebyshev_previous
        if self.polynomial_degree >= 1:
            chebyshev_current = scaled
            factor = factor + coefficients[:, 1:2, None] * chebyshev_current
            for degree in range(2, self.polynomial_degree + 1):
                chebyshev_next = (
                    2.0 * scaled @ chebyshev_current - chebyshev_previous
                )
                factor = factor + coefficients[:, degree : degree + 1, None] * chebyshev_next
                chebyshev_previous, chebyshev_current = (
                    chebyshev_current,
                    chebyshev_next,
                )
        factor = 0.5 * (factor + factor.mH)
        return factor, coefficients, lower, upper

    @staticmethod
    def identification_loss(
        operator: Tensor,
        factor: Tensor,
        *,
        n_witnesses: int = 8,
    ) -> Tensor:
        batch_size, n_angles, _ = operator.shape
        real = torch.randn(
            batch_size,
            n_angles,
            n_witnesses,
            device=operator.device,
            dtype=operator.real.dtype,
        )
        imaginary = torch.randn_like(real)
        witnesses = (real + 1j * imaginary) / math.sqrt(2.0)
        transformed = factor.mH @ operator @ factor @ witnesses
        numerator = (witnesses - transformed).abs().square().sum(dim=(1, 2))
        denominator = witnesses.abs().square().sum(dim=(1, 2)).clamp_min(1.0e-12)
        return numerator / denominator

    @staticmethod
    def certificate(operator: Tensor, factor: Tensor) -> tuple[Tensor, Tensor]:
        whitened = factor.mH @ operator @ factor
        eigenvalues = torch.linalg.eigvalsh(0.5 * (whitened + whitened.mH))
        minimum = eigenvalues.amin(dim=-1).clamp_min(1.0e-12)
        maximum = eigenvalues.amax(dim=-1).clamp_min(1.0e-12)
        # A scalar multiple of a PCG preconditioner leaves all iterates
        # unchanged.  Certify after the optimal scalar centring, which can be
        # estimated by a few extremal Ritz steps at inference.
        center = 0.5 * (minimum + maximum)
        epsilon = ((maximum - minimum) / (2.0 * center)).clamp(0.0, 1.0)
        condition_bound = (1.0 + epsilon) / (1.0 - epsilon).clamp_min(1.0e-12)
        return epsilon, condition_bound

    def forward(
        self,
        far_field: Tensor,
        probe_rhs: Tensor,
        *,
        kernel: Tensor | None = None,
        feature_kernel: Tensor | None = None,
        depth: int | None = None,
        return_history: bool = False,
        identify_witnesses: int = 0,
        certify: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del feature_kernel
        active_kernel = self.kernel if kernel is None else kernel
        system = build_bayesian_system(
            far_field,
            active_kernel,
            probe_rhs,
            float(self.ridge_rel.item()),
        )
        factor, coefficients, lower, upper = self._factor(system)
        preconditioner = factor @ factor.mH
        n_steps = self.depth if depth is None else int(depth)
        iterate, relative_residual, history = FoundationPCGLSMLoop._pcg(
            system["operator"],
            system["rhs"],
            preconditioner,
            n_steps,
            return_history=return_history,
        )
        q = system["inverse_sqrt"][:, :, None] * iterate
        score, lsm_coefficients = prior_score(far_field, active_kernel, q)
        info: dict[str, Tensor] = {
            "q": q,
            "coefficients": lsm_coefficients,
            "relative_residual": relative_residual,
            "chebyshev_coefficients": coefficients,
            "factor": factor,
            "operator": system["operator"],
            "spectral_lower_bound": lower,
            "spectral_upper_bound": upper,
            "ridge": system["ridge"],
            "row_bound": system["row_bound"],
        }
        if identify_witnesses > 0:
            info["identification_loss"] = self.identification_loss(
                system["operator"], factor, n_witnesses=identify_witnesses
            )
        if certify:
            epsilon, condition_bound = self.certificate(system["operator"], factor)
            info["certificate_epsilon"] = epsilon
            info["condition_bound"] = condition_bound
        if history is not None:
            info["residual_history"] = history
        return score, info


def sample_multi_obstacle_masks(
    physics: FixedGeometryBornLSM,
    batch_size: int,
    counts: int | Tensor,
    *,
    mode: MultiShapeMode = "mixed",
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate unions of one to six separated random components.

    Returns the union mask, the component masks, and component centres.  The
    latter two are retained for component-level evaluation and are not exposed
    to the model.
    """
    device = physics.device
    if isinstance(counts, int):
        count_tensor = torch.full((batch_size,), counts, device=device, dtype=torch.long)
    else:
        count_tensor = counts.to(device=device, dtype=torch.long)
    if count_tensor.shape != (batch_size,):
        raise ValueError("counts must be scalar or have shape (batch_size,)")
    max_count = int(count_tensor.max().item())
    if max_count < 1 or max_count > 6:
        raise ValueError("component count must lie between one and six")

    grid = physics.grid[None, :, :]
    task_center = (torch.rand(batch_size, 2, device=device) - 0.5) * 0.26
    base_angle = 2.0 * math.pi * torch.rand(batch_size, 1, device=device)
    centres = torch.zeros(batch_size, max_count, 2, device=device)
    component_masks = torch.zeros(
        batch_size,
        max_count,
        physics.n_probes,
        device=device,
        dtype=torch.bool,
    )

    if mode == "mirrored_kites":
        if not torch.all(count_tensor == 2):
            raise ValueError("mirrored_kites requires exactly two components")
        orientation = base_angle[:, 0]
        axis = torch.stack([torch.cos(orientation), torch.sin(orientation)], dim=-1)
        centres[:, 0] = task_center - 0.24 * axis
        centres[:, 1] = task_center + 0.24 * axis

    for component in range(max_count):
        active = component < count_tensor
        if mode != "mirrored_kites":
            denominator = count_tensor.clamp_min(2).to(physics.real_dtype)
            theta = (
                base_angle[:, 0]
                + 2.0 * math.pi * component / denominator
                + 0.16 * torch.randn(batch_size, device=device)
            )
            ring_radius = torch.where(
                count_tensor == 1,
                torch.zeros_like(theta),
                0.14 + 0.045 * count_tensor.to(physics.real_dtype),
            )
            centres[:, component, 0] = task_center[:, 0] + ring_radius * torch.cos(theta)
            centres[:, component, 1] = task_center[:, 1] + ring_radius * torch.sin(theta)

        delta = grid - centres[:, component, None, :]
        rotation = math.pi * torch.rand(batch_size, 1, device=device)
        cosine = torch.cos(rotation)
        sine = torch.sin(rotation)
        local_x = cosine * delta[..., 0] + sine * delta[..., 1]
        local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
        radius = 0.075 + 0.045 * torch.rand(batch_size, 1, device=device)

        if mode == "mixed":
            shape_code = torch.randint(0, 3, (batch_size, 1), device=device)
        elif mode == "disk":
            shape_code = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        elif mode == "ellipse":
            shape_code = torch.ones(batch_size, 1, device=device, dtype=torch.long)
        elif mode in ("kite", "mirrored_kites"):
            shape_code = torch.full(
                (batch_size, 1), 2, device=device, dtype=torch.long
            )
        elif mode == "star":
            shape_code = torch.full(
                (batch_size, 1), 3, device=device, dtype=torch.long
            )
        elif mode == "crescent":
            shape_code = torch.full(
                (batch_size, 1), 4, device=device, dtype=torch.long
            )
        else:
            raise ValueError(f"unknown multi-obstacle mode: {mode}")

        disk = local_x.square() + local_y.square() <= radius.square()
        aspect = 0.62 + 0.76 * torch.rand(batch_size, 1, device=device)
        ellipse = (
            (local_x / (radius * aspect)).square()
            + (local_y / (radius / aspect)).square()
            <= 1.0
        )
        polar_angle = torch.atan2(local_y, local_x)
        polar_radius = torch.sqrt(local_x.square() + local_y.square())
        kite_boundary = radius * (
            1.0
            + 0.24 * torch.cos(3.0 * polar_angle)
            + 0.10 * torch.sin(2.0 * polar_angle)
        )
        kite = polar_radius <= kite_boundary.clamp_min(0.035)
        star_boundary = radius * (1.0 + 0.32 * torch.cos(5.0 * polar_angle))
        star = polar_radius <= star_boundary.clamp_min(0.035)
        outer = local_x.square() + local_y.square() <= (1.18 * radius).square()
        inner = (local_x - 0.58 * radius).square() + local_y.square() <= radius.square()
        crescent = outer & ~inner
        candidates = torch.stack([disk, ellipse, kite, star, crescent], dim=1)
        selected = candidates.gather(
            1,
            shape_code[:, :, None].expand(-1, -1, physics.n_probes),
        ).squeeze(1)
        empty = active & ~selected.any(dim=-1)
        if empty.any():
            nearest = delta.square().sum(dim=-1).argmin(dim=-1)
            selected[empty, nearest[empty]] = True
        component_masks[:, component] = selected & active[:, None]

    union = component_masks.any(dim=1)
    return union, component_masks, centres


def posterior_covariance(
    far_field: Tensor,
    kernel: Tensor,
    solved_fk: Tensor,
) -> Tensor:
    """Posterior covariance K-KF*H^{-1}FK from recurrently solved columns."""
    batch_size = far_field.shape[0]
    if kernel.ndim == 2:
        kernel_batch = kernel.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        kernel_batch = kernel
    covariance = kernel_batch - kernel_batch @ far_field.mH @ solved_fk
    covariance = 0.5 * (covariance + covariance.mH)
    # A diagonal spectral shift is phase-invariant and differentiable for
    # proper-complex covariances.  Reconstructing from complex eigenvectors
    # makes the backward pass ill-defined because each eigenvector has an
    # arbitrary phase, even though the covariance itself is well defined.
    minimum_eigenvalue = torch.linalg.eigvalsh(covariance).amin(dim=-1)
    shift = (1.0e-8 - minimum_eigenvalue).clamp_min(0.0)
    identity = torch.eye(
        covariance.shape[-1], device=covariance.device, dtype=covariance.dtype
    )
    return covariance + shift[:, None, None] * identity


def posterior_score_moments(
    mean_coefficients: Tensor,
    covariance: Tensor,
    kernel: Tensor,
) -> tuple[Tensor, Tensor]:
    """Moment-matched mean/std of the Bayesian LSM log-energy score.

    The coefficient posterior is proper complex Gaussian.  Its quadratic GP
    energy has analytic first two moments; a log-normal moment match then gives
    a stable approximation for s=-log(Q)/2.
    """
    batch_size = mean_coefficients.shape[0]
    if kernel.ndim == 2:
        kernel_batch = kernel.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        kernel_batch = kernel
    kernel_inverse = torch.linalg.inv(kernel_batch)
    whitened_covariance = kernel_inverse @ covariance
    trace = whitened_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real
    trace_square = (
        (whitened_covariance @ whitened_covariance)
        .diagonal(dim1=-2, dim2=-1)
        .sum(dim=-1)
        .real
    )
    inverse_mean = kernel_inverse @ mean_coefficients
    mean_energy = (
        (mean_coefficients.conj() * inverse_mean).sum(dim=1).real
        + trace[:, None]
    ).clamp_min(1.0e-10)
    covariance_inverse_mean = covariance @ inverse_mean
    noncentral = (
        (inverse_mean.conj() * covariance_inverse_mean).sum(dim=1).real
    ).clamp_min(0.0)
    variance_energy = (trace_square[:, None] + 2.0 * noncentral).clamp_min(1.0e-12)
    log_variance = torch.log1p(variance_energy / mean_energy.square())
    log_mean = torch.log(mean_energy) - 0.5 * log_variance
    score_mean = -0.5 * log_mean
    score_std = 0.5 * torch.sqrt(log_variance.clamp_min(1.0e-12))
    return score_mean, score_std


def occupancy_probability(score_mean: Tensor, score_std: Tensor, threshold: float) -> Tensor:
    """Posterior probability that the LSM score exceeds a fixed threshold."""
    standardized = (score_mean - float(threshold)) / score_std.clamp_min(1.0e-6)
    return 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))


def balanced_brier(probability: Tensor, target: Tensor) -> Tensor:
    """Equal-weight positive/negative Brier score for sparse obstacle masks."""
    target_float = target.to(probability.dtype)
    squared = (probability - target_float).square()
    positive = (squared * target_float).sum(dim=-1) / target_float.sum(dim=-1).clamp_min(1.0)
    negative_weight = 1.0 - target_float
    negative = (squared * negative_weight).sum(dim=-1) / negative_weight.sum(dim=-1).clamp_min(
        1.0
    )
    return 0.5 * (positive + negative)


def balanced_nll(probability: Tensor, target: Tensor) -> Tensor:
    """Equal-weight Bernoulli NLL for sparse obstacle masks."""
    probability = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
    target_float = target.to(probability.dtype)
    positive = -(target_float * torch.log(probability)).sum(dim=-1) / target_float.sum(
        dim=-1
    ).clamp_min(1.0)
    negative_weight = 1.0 - target_float
    negative = -(
        negative_weight * torch.log1p(-probability)
    ).sum(dim=-1) / negative_weight.sum(dim=-1).clamp_min(1.0)
    return 0.5 * (positive + negative)
