#!/usr/bin/env python3
"""Training-only PoCs for fixed-geometry kernel ICL and experiment design.

The modelling choice is deliberately explicit: an RBF softmax nonlinearity
defines the feature geometry and its lengthscale is a non-trainable buffer.
Training is restricted to (i) the tied iterative controller, (ii) a standard
Transformer control, and (iii) a sequential experimental-design policy.

No exact linear solve is used by either training loss. Exact KRR and greedy
posterior-variance design appear only as held-out evaluation references.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class GeometryConfig:
    grid_size: int = 64
    true_lengthscale: float = 0.18
    noise_std: float = 0.10
    ridge: float = 0.01
    min_context: int = 6
    max_context: int = 16
    loop_depth: int = 12


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 128
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-4
    log_every: int = 100
    eval_batches: int = 12
    eval_batch_size: int = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rbf_kernel(x: Tensor, z: Tensor, lengthscale: Tensor | float) -> Tensor:
    """RBF kernel; softmax logits are exactly its negative squared distances."""
    ell = torch.as_tensor(lengthscale, device=x.device, dtype=x.dtype)
    dist2 = (x[..., :, None, :] - z[..., None, :, :]).square().sum(dim=-1)
    return torch.exp(-0.5 * dist2 / ell.square())


def query_weights(grid: Tensor) -> Tensor:
    """Nonuniform prediction objective used by the experiment-design PoC."""
    x = grid.squeeze(-1)
    weights = 0.20 + torch.exp(-0.5 * (x / 0.32).square())
    return weights / weights.sum()


class FixedGeometryGP:
    """Finite-grid draws from one fixed Gaussian-process geometry."""

    def __init__(self, cfg: GeometryConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.grid = torch.linspace(-1.0, 1.0, cfg.grid_size, device=device).unsqueeze(-1)
        self.kernel = rbf_kernel(self.grid, self.grid, cfg.true_lengthscale)
        eye = torch.eye(cfg.grid_size, device=device)
        self.cholesky = torch.linalg.cholesky(self.kernel + 1.0e-5 * eye)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        z = torch.randn(batch_size, self.cfg.grid_size, device=self.device)
        latent = z @ self.cholesky.T
        observed = latent + self.cfg.noise_std * torch.randn_like(latent)
        return latent, observed

    def random_context(self, batch_size: int, n_context: int) -> Tensor:
        scores = torch.rand(batch_size, self.cfg.grid_size, device=self.device)
        return scores.topk(n_context, dim=-1, largest=False).indices

    def gather_context(self, observed: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
        batch = observed.shape[0]
        x = self.grid[indices]
        y = observed.gather(1, indices)
        assert x.shape[:2] == (batch, indices.shape[1])
        return x, y


class TiedKernelLoop(nn.Module):
    """Geometry-conditioned, tied Richardson/heavy-ball attention cell.

    For context kernel K, D=diag(K1), P=D^{-1}K and b=D^{-1}y,
    the shared update is

        alpha_{t+1} = alpha_t + eta [b - P alpha_t - ridge D^{-1} alpha_t]
                        + beta [alpha_t - alpha_{t-1}].

    The RBF lengthscale is fixed by construction. A small invariant controller
    learns eta and beta from prompt-geometry summaries and is reused at every
    loop iteration.
    """

    def __init__(
        self,
        lengthscale: float,
        ridge: float,
        depth: int,
        grid_size: int,
        controller_width: int = 32,
    ) -> None:
        super().__init__()
        self.register_buffer("lengthscale", torch.tensor(float(lengthscale)))
        self.register_buffer("ridge", torch.tensor(float(ridge)))
        self.depth = int(depth)
        self.grid_size = int(grid_size)
        self.controller = nn.Sequential(
            nn.Linear(6, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, controller_width),
            nn.GELU(),
            nn.Linear(controller_width, 2),
        )
        nn.init.zeros_(self.controller[-1].weight)
        with torch.no_grad():
            self.controller[-1].bias.copy_(torch.tensor([0.0, -2.0]))

    @property
    def kernel_is_frozen(self) -> bool:
        return not self.lengthscale.requires_grad

    def coefficients(self, degree: Tensor, n_context: int) -> tuple[Tensor, Tensor]:
        n = float(n_context)
        degree_scaled = degree / n
        ridge_column = torch.full_like(degree_scaled[:, 0], float(self.ridge.item()))
        features = torch.stack(
            [
                torch.full_like(ridge_column, math.log(n / self.grid_size + 1.0e-6)),
                degree_scaled.mean(dim=-1),
                degree_scaled.std(dim=-1, unbiased=False),
                degree_scaled.amin(dim=-1),
                degree_scaled.amax(dim=-1),
                ridge_column,
            ],
            dim=-1,
        )
        raw_eta, raw_beta = self.controller(features).unbind(dim=-1)
        beta = 0.95 * torch.sigmoid(raw_beta)
        eta_cap = 1.95 * (1.0 + beta) / (1.0 + self.ridge)
        eta = eta_cap * torch.sigmoid(raw_eta)
        return eta, beta

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_query: Tensor,
        *,
        depth: int | None = None,
        return_layers: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, n_context, _ = x_context.shape
        if x_query.ndim == 2:
            x_query = x_query.unsqueeze(0).expand(batch, -1, -1)
        kernel_cc = rbf_kernel(x_context, x_context, self.lengthscale)
        degree = kernel_cc.sum(dim=-1).clamp_min(1.0e-8)
        attention = kernel_cc / degree.unsqueeze(-1)  # fixed RBF softmax
        rhs = y_context / degree
        eta, beta = self.coefficients(degree, n_context)

        alpha = torch.zeros_like(y_context)
        alpha_previous = torch.zeros_like(alpha)
        predictions: list[Tensor] = []
        kernel_qc = rbf_kernel(x_query, x_context, self.lengthscale)
        n_steps = self.depth if depth is None else int(depth)
        for _ in range(n_steps):
            operator_alpha = torch.einsum("bij,bj->bi", attention, alpha)
            operator_alpha = operator_alpha + self.ridge * alpha / degree
            residual = rhs - operator_alpha
            alpha_next = alpha + eta[:, None] * residual + beta[:, None] * (alpha - alpha_previous)
            alpha_previous, alpha = alpha, alpha_next
            if return_layers:
                predictions.append(torch.einsum("bqn,bn->bq", kernel_qc, alpha))

        prediction = torch.einsum("bqn,bn->bq", kernel_qc, alpha)
        info: dict[str, Tensor] = {
            "eta": eta,
            "beta": beta,
            "alpha": alpha,
            "degree": degree,
            "attention": attention,
        }
        if return_layers:
            info["predictions"] = torch.stack(predictions, dim=1)
        return prediction, info


class VanillaICLTransformer(nn.Module):
    """Learned-dot-product control with no prescribed kernel geometry."""

    def __init__(self, d_model: int = 64, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=2 * d_model,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_query: Tensor,
        *,
        depth: int | None = None,
        return_layers: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del depth, return_layers
        batch, n_context, _ = x_context.shape
        if x_query.ndim == 2:
            x_query = x_query.unsqueeze(0).expand(batch, -1, -1)
        context_flag = torch.ones(batch, n_context, 1, device=x_context.device)
        query_flag = torch.zeros(batch, x_query.shape[1], 1, device=x_context.device)
        context_tokens = torch.cat([x_context, y_context.unsqueeze(-1), context_flag], dim=-1)
        query_tokens = torch.cat([x_query, torch.zeros_like(x_query), query_flag], dim=-1)
        hidden = self.input(torch.cat([context_tokens, query_tokens], dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        query_hidden = self.norm(hidden[:, n_context:])
        return self.output(query_hidden).squeeze(-1), {}


class DesignPolicy(nn.Module):
    """Shared sequential scorer trained only through downstream prediction loss."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(9, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def logits(
        self,
        grid: Tensor,
        weights: Tensor,
        coverage: Tensor,
        step: int,
        budget: int,
        batch_size: int,
    ) -> Tensor:
        x = grid.squeeze(-1)
        fixed = torch.stack(
            [
                x,
                x.square(),
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.sin(2.0 * math.pi * x),
                torch.cos(2.0 * math.pi * x),
                weights / weights.mean(),
            ],
            dim=-1,
        )
        fixed = fixed.unsqueeze(0).expand(batch_size, -1, -1)
        step_feature = torch.full_like(coverage.unsqueeze(-1), step / max(budget - 1, 1))
        features = torch.cat([fixed, coverage.unsqueeze(-1), step_feature], dim=-1)
        return self.scorer(features).squeeze(-1)

    def exclusion_mask(self, used: Tensor, coverage: Tensor) -> Tensor:
        del coverage
        return used

    def exclusion_coverage(self, cumulative: Tensor, maximum: Tensor) -> Tensor:
        del cumulative
        return maximum

    def select(
        self,
        observed: Tensor,
        grid: Tensor,
        kernel: Tensor,
        weights: Tensor,
        budget: int,
        *,
        tau: float = 0.5,
        stochastic: bool = True,
        return_prefixes: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        batch, grid_size = observed.shape
        used = torch.zeros(batch, grid_size, dtype=torch.bool, device=observed.device)
        selected_mass = torch.zeros(batch, grid_size, device=observed.device)
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        hard_indices: list[Tensor] = []
        prefixes: list[tuple[Tensor, Tensor]] = []
        for step in range(budget):
            coverage = torch.einsum("ij,bj->bi", kernel, selected_mass).clamp(max=1.0)
            logits = self.logits(grid, weights, coverage, step, budget, batch)
            max_similarity = (
                kernel.unsqueeze(0) * selected_mass.unsqueeze(1)
            ).amax(dim=-1)
            exclusion_coverage = self.exclusion_coverage(coverage, max_similarity)
            excluded = self.exclusion_mask(used, exclusion_coverage)
            # The fallback is defensive for future geometries with very small grids.
            all_excluded = excluded.all(dim=-1, keepdim=True)
            excluded = torch.where(all_excluded, used, excluded)
            logits = logits.masked_fill(excluded, -1.0e9)
            if stochastic:
                choice = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            else:
                index = logits.argmax(dim=-1)
                choice = F.one_hot(index, grid_size).to(logits.dtype)
            index = choice.detach().argmax(dim=-1)
            used = used.scatter(1, index[:, None], True)
            selected_mass = (selected_mass + choice).clamp(max=1.0)
            xs.append(choice @ grid)
            ys.append(torch.einsum("bg,bg->b", choice, observed))
            hard_indices.append(index)
            if return_prefixes:
                prefixes.append((torch.stack(xs, dim=1), torch.stack(ys, dim=1)))
        return (
            torch.stack(xs, dim=1),
            torch.stack(ys, dim=1),
            torch.stack(hard_indices, dim=1),
            prefixes,
        )


class DiverseDesignPolicy(DesignPolicy):
    """Design scorer with a positive, learned fixed-kernel repulsion."""

    def __init__(self, width: int = 64) -> None:
        super().__init__(width)
        self.raw_repulsion = nn.Parameter(torch.tensor(2.0))

    def logits(
        self,
        grid: Tensor,
        weights: Tensor,
        coverage: Tensor,
        step: int,
        budget: int,
        batch_size: int,
    ) -> Tensor:
        utility = super().logits(grid, weights, coverage, step, budget, batch_size)
        repulsion = 0.5 + F.softplus(self.raw_repulsion)
        return utility - repulsion * coverage


class SeparatedDesignPolicy(DiverseDesignPolicy):
    """Diverse policy with a fixed cumulative kernel-coverage gate."""

    def exclusion_mask(self, used: Tensor, coverage: Tensor) -> Tensor:
        return used | (coverage > 0.85)

    def exclusion_coverage(self, cumulative: Tensor, maximum: Tensor) -> Tensor:
        del maximum
        return cumulative


class BoundedGeometryDesignPolicy(DesignPolicy):
    """Bounded learned utility plus non-trainable kernel diversity pressure."""

    def logits(
        self,
        grid: Tensor,
        weights: Tensor,
        coverage: Tensor,
        step: int,
        budget: int,
        batch_size: int,
    ) -> Tensor:
        raw_utility = super().logits(grid, weights, coverage, step, budget, batch_size)
        bounded_utility = 2.0 * torch.tanh(0.5 * raw_utility)
        return bounded_utility - 4.0 * coverage

    def exclusion_mask(self, used: Tensor, coverage: Tensor) -> Tensor:
        return used | (coverage > 0.85)


def build_design_policy(policy_type: str) -> DesignPolicy:
    if policy_type == "plain":
        return DesignPolicy()
    if policy_type == "diverse":
        return DiverseDesignPolicy()
    if policy_type == "separated":
        return SeparatedDesignPolicy()
    if policy_type == "bounded_geometry":
        return BoundedGeometryDesignPolicy()
    raise ValueError(f"unknown design policy: {policy_type}")


def exact_krr(
    x_context: Tensor,
    y_context: Tensor,
    x_query: Tensor,
    lengthscale: float,
    ridge: float,
) -> Tensor:
    """Held-out reference only; never called from a training loss."""
    batch, n_context, _ = x_context.shape
    if x_query.ndim == 2:
        x_query = x_query.unsqueeze(0).expand(batch, -1, -1)
    kernel_cc = rbf_kernel(x_context, x_context, lengthscale)
    eye = torch.eye(n_context, device=x_context.device).expand(batch, -1, -1)
    alpha = torch.linalg.solve(kernel_cc + ridge * eye, y_context.unsqueeze(-1)).squeeze(-1)
    return torch.einsum("bqn,bn->bq", rbf_kernel(x_query, x_context, lengthscale), alpha)


def nadaraya_watson(x_context: Tensor, y_context: Tensor, x_query: Tensor, lengthscale: float) -> Tensor:
    batch = x_context.shape[0]
    if x_query.ndim == 2:
        x_query = x_query.unsqueeze(0).expand(batch, -1, -1)
    kernel_qc = rbf_kernel(x_query, x_context, lengthscale)
    return torch.einsum("bqn,bn->bq", kernel_qc / kernel_qc.sum(-1, keepdim=True).clamp_min(1e-8), y_context)


def weighted_episode_mse(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    return ((prediction - target).square() * weights.unsqueeze(0)).sum(dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_model(variant: str, geometry: GeometryConfig) -> nn.Module:
    if variant == "matched":
        return TiedKernelLoop(
            geometry.true_lengthscale,
            geometry.ridge,
            geometry.loop_depth,
            geometry.grid_size,
        )
    if variant == "mismatched":
        return TiedKernelLoop(
            0.50 * geometry.true_lengthscale,
            geometry.ridge,
            geometry.loop_depth,
            geometry.grid_size,
        )
    if variant == "transformer":
        return VanillaICLTransformer()
    raise ValueError(f"unknown variant: {variant}")


def maybe_resume(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not checkpoint_path.exists():
        return 0
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def train_icl(
    outdir: Path,
    variant: str,
    seed: int,
    geometry: GeometryConfig,
    train: TrainConfig,
    device: torch.device,
) -> Path:
    run_dir = outdir / "icl" / variant / f"seed_{seed}"
    final_path = run_dir / "final.pt"
    if (run_dir / "complete.json").exists() and final_path.exists():
        print(f"SKIP completed {run_dir}", flush=True)
        return final_path
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    task = FixedGeometryGP(geometry, device)
    weights = torch.full((geometry.grid_size,), 1.0 / geometry.grid_size, device=device)
    model = build_model(variant, geometry).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.lr, weight_decay=train.weight_decay)
    last_path = run_dir / "last.pt"
    start = maybe_resume(last_path, model, optimizer, device)
    log_path = run_dir / "train.csv"

    for step in range(start + 1, train.steps + 1):
        n_context = random.randint(geometry.min_context, geometry.max_context)
        latent, observed = task.sample(train.batch_size)
        indices = task.random_context(train.batch_size, n_context)
        x_context, y_context = task.gather_context(observed, indices)
        prediction, _ = model(x_context, y_context, task.grid)
        loss = weighted_episode_mse(prediction, latent, weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % train.log_every == 0 or step == train.steps:
            row: dict[str, object] = {
                "step": step,
                "variant": variant,
                "seed": seed,
                "n_context": n_context,
                "train_mse": float(loss.detach()),
            }
            if isinstance(model, TiedKernelLoop):
                with torch.no_grad():
                    _, info = model(x_context[:16], y_context[:16], task.grid)
                row.update(
                    eta=float(info["eta"].mean()),
                    beta=float(info["beta"].mean()),
                    fixed_lengthscale=float(model.lengthscale),
                )
            append_csv(log_path, row)
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                last_path,
            )
            print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "model": model.state_dict(),
        "variant": variant,
        "seed": seed,
        "geometry": asdict(geometry),
        "train": asdict(train),
        "parameters": count_parameters(model),
    }
    torch.save(payload, final_path)
    evaluate_icl_checkpoint(final_path, run_dir, train.eval_batches, train.eval_batch_size, device)
    save_json(
        run_dir / "complete.json",
        {"variant": variant, "seed": seed, "steps": train.steps, "parameters": count_parameters(model)},
    )
    return final_path


@torch.no_grad()
def load_icl_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, GeometryConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    geometry = GeometryConfig(**checkpoint["geometry"])
    model = build_model(checkpoint["variant"], geometry).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, geometry, checkpoint


@torch.no_grad()
def evaluate_icl_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    eval_batches: int,
    batch_size: int,
    device: torch.device,
) -> None:
    model, geometry, checkpoint = load_icl_checkpoint(checkpoint_path, device)
    task = FixedGeometryGP(geometry, device)
    weights = torch.full((geometry.grid_size,), 1.0 / geometry.grid_size, device=device)
    output_path = run_dir / "evaluation.csv"
    if output_path.exists():
        output_path.unlink()
    for n_context in [4, 8, 12, 16, 24]:
        values: dict[str, list[Tensor]] = {"model": [], "exact_krr": [], "nw": []}
        eta_values: list[Tensor] = []
        beta_values: list[Tensor] = []
        for _ in range(eval_batches):
            latent, observed = task.sample(batch_size)
            indices = task.random_context(batch_size, n_context)
            x_context, y_context = task.gather_context(observed, indices)
            prediction, info = model(x_context, y_context, task.grid)
            exact = exact_krr(
                x_context,
                y_context,
                task.grid,
                geometry.true_lengthscale,
                geometry.ridge,
            )
            nw = nadaraya_watson(
                x_context,
                y_context,
                task.grid,
                geometry.true_lengthscale,
            )
            values["model"].append(weighted_episode_mse(prediction, latent, weights))
            values["exact_krr"].append(weighted_episode_mse(exact, latent, weights))
            values["nw"].append(weighted_episode_mse(nw, latent, weights))
            if "eta" in info:
                eta_values.append(info["eta"])
                beta_values.append(info["beta"])
        for method, chunks in values.items():
            samples = torch.cat(chunks).cpu().numpy()
            row: dict[str, object] = {
                "variant": checkpoint["variant"],
                "seed": checkpoint["seed"],
                "n_context": n_context,
                "method": method,
                "mse": float(samples.mean()),
                "se": float(samples.std(ddof=1) / math.sqrt(samples.size)),
                "episodes": int(samples.size),
                "parameters": int(checkpoint["parameters"]),
            }
            if eta_values:
                row["eta"] = float(torch.cat(eta_values).mean())
                row["beta"] = float(torch.cat(beta_values).mean())
            append_csv(output_path, row)

    if isinstance(model, TiedKernelLoop):
        depth_path = run_dir / "depth_curve.csv"
        if depth_path.exists():
            depth_path.unlink()
        for depth in [1, 2, 4, 8, 12, 16, 24, 32]:
            chunks = []
            for _ in range(eval_batches):
                latent, observed = task.sample(batch_size)
                indices = task.random_context(batch_size, 12)
                x_context, y_context = task.gather_context(observed, indices)
                prediction, _ = model(x_context, y_context, task.grid, depth=depth)
                chunks.append(weighted_episode_mse(prediction, latent, weights))
            samples = torch.cat(chunks).cpu().numpy()
            append_csv(
                depth_path,
                {
                    "variant": checkpoint["variant"],
                    "seed": checkpoint["seed"],
                    "depth": depth,
                    "mse": float(samples.mean()),
                    "se": float(samples.std(ddof=1) / math.sqrt(samples.size)),
                },
            )


def train_design(
    outdir: Path,
    checkpoint_path: Path,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    budget: int,
    device: torch.device,
    design_name: str = "design",
    policy_type: str = "plain",
) -> Path:
    run_dir = outdir / design_name / f"seed_{seed}"
    final_path = run_dir / "final.pt"
    if (run_dir / "complete.json").exists() and final_path.exists():
        print(f"SKIP completed {run_dir}", flush=True)
        return final_path
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(10_000 + seed)
    predictor, geometry, _ = load_icl_checkpoint(checkpoint_path, device)
    predictor.requires_grad_(False)
    task = FixedGeometryGP(geometry, device)
    weights = query_weights(task.grid)
    policy = build_design_policy(policy_type).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1.0e-4)
    last_path = run_dir / "last.pt"
    start = maybe_resume(last_path, policy, optimizer, device)
    log_path = run_dir / "train.csv"
    prefix_budgets = list(dict.fromkeys([2, 4, 6, budget]))

    for step in range(start + 1, steps + 1):
        latent, observed = task.sample(batch_size)
        tau = max(0.20, 1.25 * (0.20 ** (step / steps)))
        _, _, _, prefixes = policy.select(
            observed,
            task.grid,
            task.kernel,
            weights,
            budget,
            tau=tau,
            stochastic=True,
            return_prefixes=True,
        )
        losses = []
        for prefix in prefix_budgets:
            x_context, y_context = prefixes[prefix - 1]
            prediction, _ = predictor(x_context, y_context, task.grid)
            losses.append(weighted_episode_mse(prediction, latent, weights).mean())
        if policy_type in {"separated", "bounded_geometry"}:
            if len(losses) != 4:
                raise ValueError("separated policy expects a design budget above six")
            prefix_weights = torch.tensor([0.10, 0.15, 0.25, 0.50], device=device)
            loss = (torch.stack(losses) * prefix_weights).sum()
        else:
            loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                _, _, selected, _ = policy.select(
                    observed[:1],
                    task.grid,
                    task.kernel,
                    weights,
                    budget,
                    stochastic=False,
                )
            row = {
                "step": step,
                "seed": seed,
                "loss": float(loss.detach()),
                "tau": tau,
                "selected": " ".join(map(str, selected[0].tolist())),
            }
            append_csv(log_path, row)
            torch.save(
                {"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                last_path,
            )
            print(json.dumps(row, sort_keys=True), flush=True)

    torch.save(
        {
            "model": policy.state_dict(),
            "seed": seed,
            "geometry": asdict(geometry),
            "predictor_checkpoint": str(checkpoint_path.resolve()),
            "budget": budget,
            "parameters": count_parameters(policy),
            "policy_type": policy_type,
            "design_name": design_name,
        },
        final_path,
    )
    evaluate_design(final_path, run_dir, eval_batches=20, batch_size=256, device=device)
    save_json(run_dir / "complete.json", {"seed": seed, "steps": steps, "budget": budget})
    return final_path


def uniform_nested_indices(grid_size: int, budget: int, device: torch.device) -> Tensor:
    """Deterministic nested maximin space-filling control."""
    selected = [grid_size // 2]
    candidates = torch.arange(grid_size, device=device)
    while len(selected) < budget:
        distances = torch.stack([(candidates - index).abs() for index in selected]).amin(dim=0)
        distances[selected] = -1
        selected.append(int(distances.argmax()))
    return torch.tensor(selected, device=device)


@torch.no_grad()
def greedy_variance_indices(
    grid: Tensor,
    kernel: Tensor,
    weights: Tensor,
    ridge: float,
    budget: int,
) -> Tensor:
    """Evaluation-only weighted posterior-variance greedy reference."""
    selected: list[int] = []
    grid_size = grid.shape[0]
    for _ in range(budget):
        best_index = -1
        best_value = float("inf")
        for index in range(grid_size):
            if index in selected:
                continue
            trial = selected + [index]
            k_cs = kernel[:, trial]
            k_ss = kernel[trial][:, trial]
            eye = torch.eye(len(trial), device=grid.device)
            solved = torch.linalg.solve(k_ss + ridge * eye, k_cs.T)
            variance = kernel.diag() - (k_cs * solved.T).sum(dim=-1)
            objective = float((weights * variance.clamp_min(0.0)).sum())
            if objective < best_value:
                best_value = objective
                best_index = index
        selected.append(best_index)
    return torch.tensor(selected, device=grid.device)


@torch.no_grad()
def evaluate_design(
    design_checkpoint: Path,
    run_dir: Path,
    eval_batches: int,
    batch_size: int,
    device: torch.device,
) -> None:
    checkpoint = torch.load(design_checkpoint, map_location=device, weights_only=False)
    geometry = GeometryConfig(**checkpoint["geometry"])
    predictor, _, _ = load_icl_checkpoint(Path(checkpoint["predictor_checkpoint"]), device)
    policy = build_design_policy(checkpoint.get("policy_type", "plain")).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    task = FixedGeometryGP(geometry, device)
    weights = query_weights(task.grid)
    max_budget = int(checkpoint["budget"])
    dummy = torch.zeros(1, geometry.grid_size, device=device)
    _, _, learned_order, _ = policy.select(
        dummy, task.grid, task.kernel, weights, max_budget, stochastic=False
    )
    learned_order = learned_order[0]
    uniform_order = uniform_nested_indices(geometry.grid_size, max_budget, device)
    greedy_order = greedy_variance_indices(
        task.grid, task.kernel, weights, geometry.ridge, max_budget
    )
    save_json(
        run_dir / "selected_indices.json",
        {
            "learned": learned_order.tolist(),
            "uniform": uniform_order.tolist(),
            "greedy_variance": greedy_order.tolist(),
            "grid": task.grid.squeeze(-1).tolist(),
        },
    )

    output_path = run_dir / "evaluation.csv"
    if output_path.exists():
        output_path.unlink()
    for budget in [2, 4, 6, max_budget]:
        samples: dict[tuple[str, str], list[Tensor]] = {}
        for _ in range(eval_batches):
            latent, observed = task.sample(batch_size)
            orders = {
                "learned": learned_order[:budget].unsqueeze(0).expand(batch_size, -1),
                "uniform": uniform_order[:budget].unsqueeze(0).expand(batch_size, -1),
                "greedy_variance": greedy_order[:budget].unsqueeze(0).expand(batch_size, -1),
                "random": task.random_context(batch_size, budget),
            }
            for design_name, indices in orders.items():
                x_context, y_context = task.gather_context(observed, indices)
                model_prediction, _ = predictor(x_context, y_context, task.grid)
                exact_prediction = exact_krr(
                    x_context,
                    y_context,
                    task.grid,
                    geometry.true_lengthscale,
                    geometry.ridge,
                )
                samples.setdefault((design_name, "looped_icl"), []).append(
                    weighted_episode_mse(model_prediction, latent, weights)
                )
                samples.setdefault((design_name, "exact_krr"), []).append(
                    weighted_episode_mse(exact_prediction, latent, weights)
                )
        for (design_name, predictor_name), chunks in samples.items():
            values = torch.cat(chunks).cpu().numpy()
            append_csv(
                output_path,
                {
                    "seed": checkpoint["seed"],
                    "budget": budget,
                    "design": design_name,
                    "predictor": predictor_name,
                    "weighted_mse": float(values.mean()),
                    "se": float(values.std(ddof=1) / math.sqrt(values.size)),
                    "episodes": int(values.size),
                },
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_and_seed_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def fit_loglog(points: list[tuple[float, float]]) -> dict[str, float]:
    """Descriptive power-law fit, intentionally not a universality claim."""
    x = np.log(np.asarray([point[0] for point in points], dtype=float))
    y = np.log(np.asarray([point[1] for point in points], dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    residual = float(np.square(y - fitted).sum())
    total = float(np.square(y - y.mean()).sum())
    return {
        "exponent": float(-slope),
        "prefactor": float(np.exp(intercept)),
        "loglog_r2": 1.0 - residual / total if total > 0.0 else 1.0,
    }


@torch.no_grad()
def make_qualitative_figure(outdir: Path, summary_dir: Path, design_name: str) -> None:
    """One fixed held-out draw, shown under each design control."""
    design_paths = sorted((outdir / design_name).glob("seed_*/final.pt"))
    if not design_paths:
        return
    device = torch.device("cpu")
    design_checkpoint = torch.load(design_paths[0], map_location=device, weights_only=False)
    geometry = GeometryConfig(**design_checkpoint["geometry"])
    predictor, _, _ = load_icl_checkpoint(Path(design_checkpoint["predictor_checkpoint"]), device)
    policy = build_design_policy(design_checkpoint.get("policy_type", "plain")).to(device)
    policy.load_state_dict(design_checkpoint["model"])
    policy.eval()
    task = FixedGeometryGP(geometry, device)
    weights = query_weights(task.grid)
    budget = int(design_checkpoint["budget"])
    set_seed(20_260_803)
    latent, observed = task.sample(1)
    _, _, learned, _ = policy.select(
        observed, task.grid, task.kernel, weights, budget, stochastic=False
    )
    orders = {
        "Learned": learned[0],
        "Uniform/maximin": uniform_nested_indices(geometry.grid_size, budget, device),
        "Variance-greedy": greedy_variance_indices(
            task.grid, task.kernel, weights, geometry.ridge, budget
        ),
        "Random": task.random_context(1, budget)[0],
    }
    x_grid = task.grid.squeeze(-1).numpy()
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True, sharey=True)
    for axis, (name, indices) in zip(axes.flat, orders.items(), strict=True):
        x_context, y_context = task.gather_context(observed, indices.unsqueeze(0))
        prediction, _ = predictor(x_context, y_context, task.grid)
        axis.plot(x_grid, latent[0].numpy(), color="#222222", linewidth=2, label="latent function")
        axis.plot(x_grid, prediction[0].numpy(), color="#1b9e77", linewidth=1.8, label="looped ICL")
        axis.scatter(
            x_context[0, :, 0].numpy(),
            y_context[0].numpy(),
            color="#d95f02",
            s=28,
            zorder=3,
            label="measurements",
        )
        axis.set_title(name)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("location")
    axes[1, 1].set_xlabel("location")
    axes[0, 0].set_ylabel("function value")
    axes[1, 0].set_ylabel("function value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(summary_dir / "qualitative_reconstruction.png", dpi=220)
    plt.close(figure)


def write_architecture_audit(outdir: Path, summary_dir: Path) -> None:
    """Machine-readable evidence that the kernel hyperparameter stayed frozen."""
    records = []
    for path in sorted((outdir / "icl").glob("*/seed_*/final.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        geometry = GeometryConfig(**checkpoint["geometry"])
        model = build_model(checkpoint["variant"], geometry)
        model.load_state_dict(checkpoint["model"])
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        record = {
            "variant": checkpoint["variant"],
            "seed": checkpoint["seed"],
            "trainable_parameters": count_parameters(model),
            "trainable_tensors": trainable_names,
        }
        if isinstance(model, TiedKernelLoop):
            record.update(
                fixed_kernel_lengthscale=float(model.lengthscale),
                kernel_lengthscale_trainable=bool(model.lengthscale.requires_grad),
                tied_iterations=model.depth,
            )
        records.append(record)
    save_json(summary_dir / "architecture_audit.json", records)


def aggregate(outdir: Path, design_name: str = "design") -> None:
    """Create presentation-ready CSVs, figures and a concise Markdown brief."""
    summary_dir = outdir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    evaluation_paths = sorted((outdir / "icl").glob("*/seed_*/evaluation.csv"))
    design_paths = sorted((outdir / design_name).glob("seed_*/evaluation.csv"))
    if not evaluation_paths:
        raise FileNotFoundError("No ICL evaluations found")

    icl_rows = [row for path in evaluation_paths for row in read_csv(path)]
    grouped_icl: dict[tuple[str, int, str], list[float]] = {}
    for row in icl_rows:
        key = (row["variant"], int(row["n_context"]), row["method"])
        grouped_icl.setdefault(key, []).append(float(row["mse"]))
    icl_summary_path = summary_dir / "icl_summary.csv"
    if icl_summary_path.exists():
        icl_summary_path.unlink()
    for (variant, n_context, method), values in sorted(grouped_icl.items()):
        mean, std = mean_and_seed_std(values)
        append_csv(
            icl_summary_path,
            {
                "variant": variant,
                "n_context": n_context,
                "method": method,
                "mean_mse": mean,
                "seed_std": std,
                "seeds": len(values),
            },
        )

    plt.figure(figsize=(6.4, 4.2))
    styles = {
        ("matched", "model"): ("#1b9e77", "o", "Matched fixed-kernel loop"),
        ("mismatched", "model"): ("#d95f02", "s", "Mismatched-kernel loop"),
        ("transformer", "model"): ("#7570b3", "^", "Vanilla Transformer"),
        ("matched", "exact_krr"): ("#222222", "D", "Exact KRR (evaluation only)"),
        ("matched", "nw"): ("#999999", "x", "Nadaraya--Watson"),
    }
    for key, (color, marker, label) in styles.items():
        points = [
            (n, mean_and_seed_std(v)[0], mean_and_seed_std(v)[1])
            for (variant, n, method), v in grouped_icl.items()
            if (variant, method) == key
        ]
        if not points:
            continue
        points.sort()
        x, y, error = map(np.asarray, zip(*points))
        plt.errorbar(x, y, yerr=error, color=color, marker=marker, label=label, capsize=3)
    plt.yscale("log")
    plt.xlabel("Number of context observations")
    plt.ylabel("Held-out latent-function MSE")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(summary_dir / "icl_scaling.png", dpi=220)
    plt.close()

    depth_paths = sorted((outdir / "icl").glob("*/seed_*/depth_curve.csv"))
    if depth_paths:
        depth_rows = [row for path in depth_paths for row in read_csv(path)]
        grouped_depth: dict[tuple[str, int], list[float]] = {}
        for row in depth_rows:
            grouped_depth.setdefault((row["variant"], int(row["depth"])), []).append(float(row["mse"]))
        plt.figure(figsize=(6.4, 4.2))
        for variant, color in [("matched", "#1b9e77"), ("mismatched", "#d95f02")]:
            points = [
                (depth, mean_and_seed_std(values)[0], mean_and_seed_std(values)[1])
                for (name, depth), values in grouped_depth.items()
                if name == variant
            ]
            points.sort()
            if points:
                x, y, error = map(np.asarray, zip(*points))
                plt.errorbar(x, y, yerr=error, marker="o", color=color, label=variant, capsize=3)
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Number of tied loop iterations")
        plt.ylabel("Held-out latent-function MSE")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_dir / "depth_scaling.png", dpi=220)
        plt.close()

    design_summary_path = summary_dir / "design_summary.csv"
    grouped_design: dict[tuple[int, str, str], list[float]] = {}
    if design_paths:
        design_rows = [row for path in design_paths for row in read_csv(path)]
        for row in design_rows:
            key = (int(row["budget"]), row["design"], row["predictor"])
            grouped_design.setdefault(key, []).append(float(row["weighted_mse"]))
        if design_summary_path.exists():
            design_summary_path.unlink()
        for (budget, design, predictor), values in sorted(grouped_design.items()):
            mean, std = mean_and_seed_std(values)
            append_csv(
                design_summary_path,
                {
                    "budget": budget,
                    "design": design,
                    "predictor": predictor,
                    "mean_weighted_mse": mean,
                    "seed_std": std,
                    "seeds": len(values),
                },
            )
        plt.figure(figsize=(6.4, 4.2))
        for design, color, marker in [
            ("learned", "#1b9e77", "o"),
            ("uniform", "#7570b3", "s"),
            ("random", "#999999", "x"),
            ("greedy_variance", "#222222", "D"),
        ]:
            points = [
                (budget, mean_and_seed_std(values)[0], mean_and_seed_std(values)[1])
                for (budget, name, predictor), values in grouped_design.items()
                if name == design and predictor == "looped_icl"
            ]
            points.sort()
            if points:
                x, y, error = map(np.asarray, zip(*points))
                label = "variance-greedy" if design == "greedy_variance" else design
                plt.errorbar(x, y, yerr=error, marker=marker, color=color, label=label, capsize=3)
        plt.yscale("log")
        plt.xlabel("Measurement budget")
        plt.ylabel("Weighted prediction MSE")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_dir / "design_scaling.png", dpi=220)
        plt.close()

        selection_paths = sorted((outdir / design_name).glob("seed_*/selected_indices.json"))
        if selection_paths:
            first = json.loads(selection_paths[0].read_text())
            grid = np.asarray(first["grid"])
            weight = 0.20 + np.exp(-0.5 * (grid / 0.32) ** 2)
            figure, axis = plt.subplots(figsize=(7.4, 3.3))
            axis.plot(grid, weight / weight.max(), color="#666666", label="prediction weight")
            names = ["learned", "uniform", "greedy_variance"]
            colors = ["#1b9e77", "#7570b3", "#222222"]
            for level, (name, color) in enumerate(zip(names, colors, strict=True), start=1):
                for path in selection_paths:
                    payload = json.loads(path.read_text())
                    locations = grid[np.asarray(payload[name], dtype=int)]
                    axis.scatter(
                        locations,
                        np.full_like(locations, 1.0 + 0.16 * level),
                        s=22,
                        color=color,
                        alpha=0.75,
                    )
                label = "variance-greedy" if name == "greedy_variance" else name
                axis.text(1.03, 1.0 + 0.16 * level, label, ha="left", va="center", fontsize=8)
            axis.set_ylim(-0.03, 1.68)
            axis.set_xlim(-1.05, 1.22)
            axis.set_xlabel("candidate location")
            axis.set_ylabel("normalized objective weight")
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right", frameon=False)
            figure.tight_layout()
            figure.savefig(summary_dir / "selected_locations.png", dpi=220)
            plt.close(figure)

    matched_12 = grouped_icl.get(("matched", 12, "model"), [])
    mismatch_12 = grouped_icl.get(("mismatched", 12, "model"), [])
    transformer_12 = grouped_icl.get(("transformer", 12, "model"), [])
    exact_12 = grouped_icl.get(("matched", 12, "exact_krr"), [])
    scaling_fits: dict[str, object] = {
        "warning": "Finite-range descriptive fits only; not universal asymptotic laws.",
        "context": {},
        "design_budget": {},
    }
    for variant, method in [
        ("matched", "model"),
        ("mismatched", "model"),
        ("transformer", "model"),
        ("matched", "exact_krr"),
    ]:
        points = sorted(
            (float(n_context), mean_and_seed_std(values)[0])
            for (name, n_context, method_name), values in grouped_icl.items()
            if name == variant and method_name == method
        )
        if len(points) >= 3:
            scaling_fits["context"][f"{variant}:{method}"] = fit_loglog(points)
    for design in ["learned", "uniform", "random", "greedy_variance"]:
        points = sorted(
            (float(budget), mean_and_seed_std(values)[0])
            for (budget, name, predictor), values in grouped_design.items()
            if name == design and predictor == "looped_icl"
        )
        if len(points) >= 3:
            scaling_fits["design_budget"][design] = fit_loglog(points)
    save_json(summary_dir / "empirical_scaling_fits.json", scaling_fits)
    brief_lines = [
        "# Fixed-geometry kernel ICL and experiment design: proof-of-concept brief",
        "",
        "## Protocol",
        "",
        "- New GP function in every episode on a fixed 1D geometry (64 locations).",
        "- RBF softmax lengthscale fixed at 0.18; it is never trainable.",
        "- The loop controller is trained end-to-end from prediction error only; exact KRR is evaluation-only.",
        "- Experiment design is trained through the frozen looped predictor; weighted variance-greedy is evaluation-only.",
        "- Reported uncertainty is variation across independent training seeds.",
        "",
        "## Headline results",
        "",
    ]
    if matched_12:
        m, s = mean_and_seed_std(matched_12)
        brief_lines.append(f"- Matched fixed-kernel loop, 12 observations: MSE {m:.4g} ± {s:.2g} across seeds.")
    if exact_12:
        m, s = mean_and_seed_std(exact_12)
        brief_lines.append(f"- Exact KRR evaluation reference, 12 observations: MSE {m:.4g} ± {s:.2g}.")
    if mismatch_12:
        m, s = mean_and_seed_std(mismatch_12)
        brief_lines.append(f"- Mismatched fixed-kernel loop, 12 observations: MSE {m:.4g} ± {s:.2g}.")
        if matched_12:
            matched_mean, _ = mean_and_seed_std(matched_12)
            brief_lines.append(f"- Kernel mismatch multiplies 12-observation error by {m / matched_mean:.2f}×.")
    if transformer_12:
        m, s = mean_and_seed_std(transformer_12)
        brief_lines.append(f"- Vanilla Transformer, 12 observations: MSE {m:.4g} ± {s:.2g}.")
    if grouped_design:
        learned = grouped_design.get((8, "learned", "looped_icl"), [])
        random_values = grouped_design.get((8, "random", "looped_icl"), [])
        uniform = grouped_design.get((8, "uniform", "looped_icl"), [])
        if learned:
            lm, ls = mean_and_seed_std(learned)
            brief_lines.append(f"- Learned design, budget 8: weighted MSE {lm:.4g} ± {ls:.2g}.")
        if random_values:
            rm, rs = mean_and_seed_std(random_values)
            brief_lines.append(f"- Random design, budget 8: weighted MSE {rm:.4g} ± {rs:.2g}.")
            if learned:
                lm, _ = mean_and_seed_std(learned)
                brief_lines.append(f"- Learned design changes error versus random by {100.0 * (lm / rm - 1.0):+.1f}%.")
        if uniform:
            um, us = mean_and_seed_std(uniform)
            brief_lines.append(f"- Uniform/maximin design, budget 8: weighted MSE {um:.4g} ± {us:.2g}.")
            if learned:
                lm, _ = mean_and_seed_std(learned)
                brief_lines.append(f"- Learned design changes error versus uniform/maximin by {100.0 * (lm / um - 1.0):+.1f}%.")
    brief_lines.extend(
        [
            "",
            "## Claims this PoC can and cannot support",
            "",
            "It tests whether a prescribed nonlinear feature geometry plus a trained tied loop can perform ICL across fresh GP draws, and whether a downstream-trained design policy improves measurement allocation. It is not a universal scaling law, a PDE benchmark, or evidence about learned kernel identification.",
            "",
            "## Figures",
            "",
            "- `icl_scaling.png`: context-length scaling and controls.",
            "- `depth_scaling.png`: computation-time scaling from additional tied iterations.",
            "- `design_scaling.png`: measurement-budget scaling for learned and reference designs.",
            "- `selected_locations.png`: learned versus reference measurement locations.",
            "- `qualitative_reconstruction.png`: one held-out function reconstructed from each design.",
        ]
    )
    (summary_dir / "PRESENTATION_BRIEF.md").write_text("\n".join(brief_lines) + "\n")

    fr_lines = [
        "# Brief de présentation — ICL à géométrie noyau fixée",
        "",
        "## Message en une phrase",
        "",
        "On peut fixer la non-linéarité softmax/RBF comme choix de modélisation, puis entraîner seulement la dynamique récurrente et la politique de mesure : la première réalise une régression bayésienne in-context proche de KRR et la seconde apprend où mesurer.",
        "",
        "## Protocole minimal",
        "",
        "- Une nouvelle fonction GP est tirée à chaque épisode sur 64 positions fixes.",
        "- La longueur d’échelle RBF vaut 0,18 et n’est jamais entraînable.",
        "- La cellule de résolution est partagée sur 12 itérations et entraînée seulement par l’erreur de prédiction.",
        "- KRR exact et variance-greedy ne sont appelés qu’après entraînement, comme références d’évaluation.",
        "- Cinq graines indépendantes ; 5 000 mises à jour par modèle et par politique.",
        "",
        "## Résultats à annoncer",
        "",
    ]
    if matched_12:
        value, spread = mean_and_seed_std(matched_12)
        fr_lines.append(f"- Boucle noyau apparié, 12 observations : MSE {value:.4g} ± {spread:.2g}.")
    if exact_12:
        value, spread = mean_and_seed_std(exact_12)
        fr_lines.append(f"- Référence KRR exacte : MSE {value:.4g} ± {spread:.2g}.")
    if mismatch_12:
        value, spread = mean_and_seed_std(mismatch_12)
        fr_lines.append(f"- Même boucle avec mauvaise géométrie : MSE {value:.4g} ± {spread:.2g}.")
    if transformer_12:
        value, spread = mean_and_seed_std(transformer_12)
        fr_lines.append(f"- Transformer standard : MSE {value:.4g} ± {spread:.2g}.")
    if grouped_design:
        learned = grouped_design.get((8, "learned", "looped_icl"), [])
        random_values = grouped_design.get((8, "random", "looped_icl"), [])
        uniform = grouped_design.get((8, "uniform", "looped_icl"), [])
        if learned:
            value, spread = mean_and_seed_std(learned)
            fr_lines.append(f"- Design appris, budget 8 : MSE pondérée {value:.4g} ± {spread:.2g}.")
        if random_values:
            value, spread = mean_and_seed_std(random_values)
            fr_lines.append(f"- Design aléatoire, budget 8 : {value:.4g} ± {spread:.2g}.")
        if uniform:
            value, spread = mean_and_seed_std(uniform)
            fr_lines.append(f"- Design uniforme/maximin, budget 8 : {value:.4g} ± {spread:.2g}.")
    fr_lines.extend(
        [
            "",
            "## Déroulé conseillé en six transparents",
            "",
            "1. Hypothèse : la non-linéarité définit l’espace de features ; elle n’a pas besoin d’être apprise.",
            "2. Architecture : softmax RBF fixé, état dual, cellule récurrente partagée.",
            "3. Protocole : nouvelles fonctions à chaque épisode, contrôles appariés, aucune cible KRR pendant l’entraînement.",
            "4. Résultat ICL : scaling en nombre d’observations et en nombre de boucles.",
            "5. Design expérimental : politique aval versus aléatoire, uniforme et variance-greedy.",
            "6. Limites : domaine 1D et géométrie fixe ; prochaine étape = LSM/PDE à géométrie physique fixée.",
            "",
            "## Formulation prudente",
            "",
            "Ce PoC valide le mécanisme et les contrôles causaux essentiels. Il ne constitue ni une loi d’échelle universelle, ni encore un benchmark PDE, ni une preuve que le noyau peut être identifié automatiquement.",
        ]
    )
    (summary_dir / "BRIEF_PRESENTATION_FR.md").write_text("\n".join(fr_lines) + "\n")
    write_architecture_audit(outdir, summary_dir)
    make_qualitative_figure(outdir, summary_dir, design_name)


def smoke(device: torch.device) -> None:
    geometry = GeometryConfig(grid_size=24, min_context=4, max_context=6, loop_depth=4)
    task = FixedGeometryGP(geometry, device)
    latent, observed = task.sample(5)
    indices = task.random_context(5, 6)
    x_context, y_context = task.gather_context(observed, indices)
    model = TiedKernelLoop(
        geometry.true_lengthscale,
        geometry.ridge,
        geometry.loop_depth,
        geometry.grid_size,
    ).to(device)
    prediction, info = model(x_context, y_context, task.grid, return_layers=True)
    loss = F.mse_loss(prediction, latent)
    loss.backward()
    assert prediction.shape == latent.shape
    assert info["predictions"].shape == (5, geometry.loop_depth, geometry.grid_size)
    assert model.kernel_is_frozen
    assert model.lengthscale.grad is None
    policy = DesignPolicy(width=16).to(device)
    weights = query_weights(task.grid)
    x_selected, y_selected, selected, prefixes = policy.select(
        observed,
        task.grid,
        task.kernel,
        weights,
        budget=4,
        stochastic=True,
        return_prefixes=True,
    )
    designed_prediction, _ = model(x_selected, y_selected, task.grid)
    design_loss = weighted_episode_mse(designed_prediction, latent, weights).mean()
    design_loss.backward()
    assert selected.unique(dim=1).shape == selected.shape
    assert len(prefixes) == 4
    print(
        json.dumps(
            {
                "status": "ok",
                "device": str(device),
                "icl_loss": float(loss.detach()),
                "design_loss": float(design_loss.detach()),
                "kernel_frozen": model.kernel_is_frozen,
            },
            sort_keys=True,
        )
    )


def parse_seeds(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def run_suite(args: argparse.Namespace, device: torch.device) -> None:
    geometry = GeometryConfig(
        grid_size=args.grid_size,
        true_lengthscale=args.lengthscale,
        noise_std=args.noise_std,
        ridge=args.ridge,
        min_context=args.min_context,
        max_context=args.max_context,
        loop_depth=args.depth,
    )
    train = TrainConfig(
        steps=args.icl_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
    )
    outdir = Path(args.outdir)
    save_json(outdir / "protocol.json", {"geometry": asdict(geometry), "training": asdict(train), "seeds": parse_seeds(args.seeds)})
    for seed in parse_seeds(args.seeds):
        checkpoints = {}
        for variant in ["matched", "mismatched", "transformer"]:
            variant_train = train
            if variant == "transformer" and train.batch_size > 64:
                variant_train = TrainConfig(
                    **{**asdict(train), "batch_size": 64, "lr": min(train.lr, 5.0e-4)}
                )
            checkpoints[variant] = train_icl(outdir, variant, seed, geometry, variant_train, device)
        train_design(
            outdir,
            checkpoints["matched"],
            seed,
            args.design_steps,
            min(args.batch_size, 96),
            args.design_lr,
            args.log_every,
            args.design_budget,
            device,
            args.design_name,
            args.design_policy,
        )
    aggregate(outdir, args.design_name)


def run_design_only(args: argparse.Namespace, device: torch.device) -> None:
    """Train a new design policy against already-completed matched ICL runs."""
    outdir = Path(args.outdir)
    for seed in parse_seeds(args.seeds):
        checkpoint = outdir / "icl" / "matched" / f"seed_{seed}" / "final.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing matched ICL checkpoint: {checkpoint}")
        train_design(
            outdir,
            checkpoint,
            seed,
            args.design_steps,
            min(args.batch_size, 96),
            args.design_lr,
            args.log_every,
            args.design_budget,
            device,
            args.design_name,
            args.design_policy,
        )
    aggregate(outdir, args.design_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["smoke", "suite", "design", "aggregate"], default="smoke"
    )
    parser.add_argument("--outdir", default="experiments/transformers/fixed_geometry_icl_design/results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--lengthscale", type=float, default=0.18)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--min-context", type=int, default=6)
    parser.add_argument("--max-context", type=int, default=16)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--icl-steps", type=int, default=3000)
    parser.add_argument("--design-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--design-lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--design-budget", type=int, default=8)
    parser.add_argument("--design-name", default="design")
    parser.add_argument(
        "--design-policy",
        choices=["plain", "diverse", "separated", "bounded_geometry"],
        default="separated",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if args.mode == "smoke":
        smoke(device)
    elif args.mode == "suite":
        run_suite(args, device)
    elif args.mode == "design":
        run_design_only(args, device)
    else:
        aggregate(Path(args.outdir), args.design_name)


if __name__ == "__main__":
    main()
