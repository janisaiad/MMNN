#!/usr/bin/env python3
"""
Constructive weak-form inverse ICL experiments: recurrent Transformer as Richardson solver.

Goal
----
This file deliberately avoids Sinkhorn/unbalanced attention and avoids training by default.
It tests the constructive algorithmic claim:

    weak-form encoder -> (G_T, b_T)
    recurrent loop    -> beta_{l+1} = beta_l + B (c_T - H_T beta_l)

where

    H_T = noise_prec * G_T^T G_T + prior_prec I
    c_T = noise_prec * G_T^T b_T
    beta_* = H_T^{-1} c_T

It simulates Transformer heads as constructive preconditioner blocks B_h. The values are
analytic weak-form residual evidence

    v_i^l = g_i (b_i - g_i^T beta_l),

and the FFN/update applies a preconditioner to the aggregated residual gradient.

The script is meant to answer:
  - How many recurrent steps L are needed for a target approximation?
  - How do K, prompt length m, heads H, and per-head rank d_h affect the approximation?
  - When is the loop numerically stable?
  - How close is the iterative posterior mean/variance to exact Bayes/ridge?

Synthetic task
--------------
    beta ~ N(0, prior_var I_K)
    g_i  ~ N(0, Sigma_g / K)
    b_i  = g_i^T beta + eps_i, eps_i ~ N(0, noise_var)
    y_*  = g_*^T beta + eps_*

This is the abstract weak-form inverse regression task G beta = b.

Common commands
---------------
Smoke:
  python constructive_weakform_richardson_transformer.py --mode smoke --device cuda

Depth sweep:
  python constructive_weakform_richardson_transformer.py --mode sweep_depth \
    --K 16 --prompt-len 128 --precond scalar_opt --depth-grid 1,2,4,8,16,32,64 --device cuda

Capacity/head sweep:
  python constructive_weakform_richardson_transformer.py --mode sweep_capacity \
    --K-grid 8,16,32 --heads-grid 1,2,4,8 --d-head-grid 1,2,4,8 \
    --precond lowrank_spectral --depth 8 --device cuda

Prompt-size scaling:
  python constructive_weakform_richardson_transformer.py --mode sweep_prompt \
    --K 16 --precond scalar_opt --prompt-grid 16,32,64,128,256,512 --depth 32 --device cuda

Compare preconditioners:
  python constructive_weakform_richardson_transformer.py --mode sweep_precond \
    --K 16 --prompt-len 128 --depth 16 --device cuda

Outputs
-------
CSV files and simple plots are saved under --outdir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_csv(path: Path, row: Dict) -> None:
    exists = path.exists()
    # keep stable order by sorting keys only if file does not exist
    fieldnames = list(row.keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def eigvalsh_spd(A: Tensor) -> Tensor:
    return torch.linalg.eigvalsh(A)


def batch_eye(B: int, K: int, device: torch.device, dtype=torch.float32) -> Tensor:
    return torch.eye(K, device=device, dtype=dtype).expand(B, K, K)


def safe_inverse_spd(A: Tensor, jitter: float = 1e-7) -> Tensor:
    K = A.shape[-1]
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    return torch.linalg.inv(A + jitter * I)


def gaussian_nll(y: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = var.clamp_min(1e-10)
    return 0.5 * (torch.log(2.0 * torch.pi * var) + (y - mean).pow(2) / var)


def empirical_coverage(y: Tensor, mean: Tensor, var: Tensor, z: float = 1.959963984540054) -> Tuple[float, float]:
    std = var.clamp_min(1e-10).sqrt()
    lo = mean - z * std
    hi = mean + z * std
    cov = ((y >= lo) & (y <= hi)).float().mean().item()
    width = (hi - lo).mean().item()
    return cov, width


def parse_grid(s: str, typ=int) -> List:
    return [typ(x) for x in str(s).split(",") if str(x).strip() != ""]


# -----------------------------------------------------------------------------
# Data generation and exact posterior
# -----------------------------------------------------------------------------

@dataclass
class WeakBatch:
    G: Tensor                 # [B,m,K]
    b: Tensor                 # [B,m]
    gq: Tensor                # [B,K]
    yq: Tensor                # [B]
    beta_true: Tensor         # [B,K]
    H: Tensor                 # [B,K,K]
    c: Tensor                 # [B,K]
    beta_post: Tensor         # [B,K]
    cov_post: Tensor          # [B,K,K]
    mean_exact: Tensor        # [B]
    var_exact: Tensor         # [B]
    eigvals: Tensor           # [B,K]


@dataclass
class TaskConfig:
    K: int = 16
    prompt_len: int = 128
    prior_var: float = 1.0
    noise_var: float = 0.02
    design: str = "isotropic"       # isotropic | correlated | spiked
    cond: float = 10.0
    spike_strength: float = 4.0
    dtype: str = "float32"


def make_design_sqrt(K: int, cfg: TaskConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    if cfg.design == "isotropic":
        return torch.eye(K, device=device, dtype=dtype)
    if cfg.design == "correlated":
        # eigenvalues spanning [1, cond], normalized to mean 1
        vals = torch.logspace(0, math.log10(cfg.cond), K, device=device, dtype=dtype)
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    if cfg.design == "spiked":
        vals = torch.ones(K, device=device, dtype=dtype)
        vals[0] = cfg.spike_strength
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    raise ValueError(f"unknown design {cfg.design}")


def sample_weak_batch(batch_size: int, cfg: TaskConfig, device: torch.device) -> WeakBatch:
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    B, m, K = batch_size, cfg.prompt_len, cfg.K
    Csqrt = make_design_sqrt(K, cfg, device, dtype)
    # Scale rows by 1/sqrt(K) so signal magnitude remains O(1).
    G = torch.randn(B, m, K, device=device, dtype=dtype) @ Csqrt.T / math.sqrt(K)
    gq = torch.randn(B, K, device=device, dtype=dtype) @ Csqrt.T / math.sqrt(K)
    beta_true = torch.randn(B, K, device=device, dtype=dtype) * math.sqrt(cfg.prior_var)
    b_clean = torch.einsum("bmk,bk->bm", G, beta_true)
    b = b_clean + torch.randn(B, m, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)
    yq = torch.einsum("bk,bk->b", gq, beta_true) + torch.randn(B, device=device, dtype=dtype) * math.sqrt(cfg.noise_var)

    noise_prec = 1.0 / cfg.noise_var
    prior_prec = 1.0 / cfg.prior_var
    I = batch_eye(B, K, device, dtype)
    H = noise_prec * torch.einsum("bmk,bml->bkl", G, G) + prior_prec * I
    c = noise_prec * torch.einsum("bmk,bm->bk", G, b)
    cov_post = safe_inverse_spd(H, jitter=1e-8)
    beta_post = torch.einsum("bkl,bl->bk", cov_post, c)
    mean_exact = torch.einsum("bk,bk->b", gq, beta_post)
    var_exact = cfg.noise_var + torch.einsum("bk,bkl,bl->b", gq, cov_post, gq).clamp_min(0.0)
    eigvals = eigvalsh_spd(H)
    return WeakBatch(G, b, gq, yq, beta_true, H, c, beta_post, cov_post, mean_exact, var_exact, eigvals)


# -----------------------------------------------------------------------------
# Constructive preconditioners = heads
# -----------------------------------------------------------------------------

def scalar_eta(eigvals: Tensor, mode: str = "opt", multiplier: float = 1.0) -> Tensor:
    """Return batch scalar eta [B]."""
    lmin = eigvals[:, 0]
    lmax = eigvals[:, -1]
    if mode == "opt":
        eta = 2.0 / (lmax + lmin)
    elif mode == "lmax":
        eta = 1.0 / lmax
    elif mode == "half_lmax":
        eta = 0.5 / lmax
    else:
        raise ValueError(mode)
    return multiplier * eta


def build_preconditioner(H: Tensor, eigvals: Tensor, precond: str, heads: int, d_head: int,
                         eta_mode: str = "opt", eta_multiplier: float = 1.0,
                         spectral_order: str = "small", damping: float = 1e-7) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Build B matrix for beta update beta += B (c - H beta).

    precond:
      scalar_opt          B = eta I, eta=2/(lmax+lmin)
      scalar_lmax         B = eta I, eta=1/lmax
      jacobi              B = eta diag(H)^{-1}
      diagonal_exact      B = diag(H)^{-1}
      lowrank_spectral    B = exact inverse on r=heads*d_head eigendirs + scalar on rest
      spectral_full       B = H^{-1} exact, represented as spectral heads covering all dims
      block_jacobi        B = block diagonal inverse with `heads` contiguous coordinate blocks
    """
    Bsz, K, _ = H.shape
    device, dtype = H.device, H.dtype
    I = batch_eye(Bsz, K, device, dtype)
    stats: Dict[str, Tensor] = {}

    if precond in ["scalar_opt", "scalar_lmax", "scalar_half_lmax"]:
        mode = {"scalar_opt": "opt", "scalar_lmax": "lmax", "scalar_half_lmax": "half_lmax"}[precond]
        eta = scalar_eta(eigvals, mode=mode, multiplier=eta_multiplier)
        Bmat = eta[:, None, None] * I
        stats["eta_mean"] = eta.mean()
        stats["precond_rank"] = torch.tensor(0.0, device=device, dtype=dtype)
        return Bmat, stats

    if precond in ["jacobi", "diagonal_exact"]:
        diag = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(damping)
        Dinv = torch.diag_embed(1.0 / diag)
        if precond == "jacobi":
            # use conservative eta for preconditioned matrix D^-1 H
            M = torch.einsum("bkl,blm->bkm", Dinv, H)
            ev = torch.linalg.eigvals(M).abs().real
            eta = eta_multiplier * (1.0 / ev.max(dim=-1).values.clamp_min(1e-8))
            Bmat = eta[:, None, None] * Dinv
            stats["eta_mean"] = eta.mean()
        else:
            Bmat = Dinv
            stats["eta_mean"] = torch.tensor(1.0, device=device, dtype=dtype)
        stats["precond_rank"] = torch.tensor(float(K), device=device, dtype=dtype)
        return Bmat, stats

    if precond in ["spectral_full", "lowrank_spectral"]:
        evals, U = torch.linalg.eigh(H)  # ascending [B,K], U columns eigenvectors
        if precond == "spectral_full":
            inv_e = 1.0 / evals.clamp_min(damping)
            Bmat = torch.einsum("bkr,br,blr->bkl", U, inv_e, U)
            stats["precond_rank"] = torch.tensor(float(K), device=device, dtype=dtype)
            stats["eta_mean"] = torch.tensor(1.0, device=device, dtype=dtype)
            return Bmat, stats

        r = min(K, max(0, heads * d_head))
        if spectral_order == "small":
            idx = torch.arange(r, device=device)
        elif spectral_order == "large":
            idx = torch.arange(K - r, K, device=device)
        elif spectral_order == "mixed":
            # interleave small and large modes
            small = list(range((r + 1) // 2))
            large = list(range(K - (r // 2), K))
            order = []
            for a, b in zip(small, large):
                order += [a, b]
            if len(small) > len(large):
                order.append(small[-1])
            idx = torch.tensor(order[:r], device=device, dtype=torch.long)
        else:
            raise ValueError(spectral_order)
        Usel = U[:, :, idx] if r > 0 else U[:, :, :0]
        esel = evals[:, idx] if r > 0 else evals[:, :0]
        # exact inverse on selected subspace
        if r > 0:
            Bsel = torch.einsum("bkr,br,blr->bkl", Usel, 1.0 / esel.clamp_min(damping), Usel)
            Psel = torch.einsum("bkr,blr->bkl", Usel, Usel)
        else:
            Bsel = torch.zeros_like(H)
            Psel = torch.zeros_like(H)
        # scalar optimal step on remaining subspace
        eta = scalar_eta(evals, mode=eta_mode, multiplier=eta_multiplier)
        Bmat = Bsel + eta[:, None, None] * (I - Psel)
        stats["precond_rank"] = torch.tensor(float(r), device=device, dtype=dtype)
        stats["eta_mean"] = eta.mean()
        return Bmat, stats

    if precond == "block_jacobi":
        # Coordinate block inverse. Note H=1 gives full inverse; H=K gives diagonal-ish blocks.
        Bmat = torch.zeros_like(H)
        splits = np.array_split(np.arange(K), max(1, heads))
        for block in splits:
            idx = torch.tensor(block, device=device, dtype=torch.long)
            Hb = H[:, idx][:, :, idx]
            Hbinv = safe_inverse_spd(Hb, jitter=damping)
            # scatter into Bmat
            for local_i, global_i in enumerate(block):
                for local_j, global_j in enumerate(block):
                    Bmat[:, global_i, global_j] = Hbinv[:, local_i, local_j]
        stats["precond_rank"] = torch.tensor(float(K), device=device, dtype=H.dtype)
        stats["eta_mean"] = torch.tensor(1.0, device=device, dtype=H.dtype)
        return Bmat, stats

    raise ValueError(f"unknown precond {precond}")


# -----------------------------------------------------------------------------
# Constructive recurrent loop
# -----------------------------------------------------------------------------

@dataclass
class LoopResult:
    beta_L: Tensor
    alpha_L: Tensor
    mean_L: Tensor
    var_L: Tensor
    layer_beta_mse_post: List[float]
    layer_resid_norm: List[float]
    contraction_radius_mean: float
    contraction_radius_max: float
    precond_rank: float
    eta_mean: float


def run_constructive_loop(batch: WeakBatch, cfg: TaskConfig, depth: int, precond: str = "scalar_opt",
                          heads: int = 1, d_head: int = 1, eta_mode: str = "opt", eta_multiplier: float = 1.0,
                          spectral_order: str = "small", beta0: str = "zero", var_iter: bool = True) -> LoopResult:
    G, b, gq, H, c = batch.G, batch.b, batch.gq, batch.H, batch.c
    Bsz, m, K = G.shape
    device, dtype = G.device, G.dtype
    Bmat, pst = build_preconditioner(H, batch.eigvals, precond, heads, d_head, eta_mode, eta_multiplier, spectral_order)

    if beta0 == "zero":
        beta = torch.zeros(Bsz, K, device=device, dtype=dtype)
    elif beta0 == "prior_sample":
        beta = torch.randn(Bsz, K, device=device, dtype=dtype) * math.sqrt(cfg.prior_var)
    else:
        raise ValueError(beta0)

    # alpha solves H alpha = gq for predictive variance.
    alpha = torch.zeros(Bsz, K, device=device, dtype=dtype)
    layer_beta_mse_post: List[float] = []
    layer_resid_norm: List[float] = []

    for _ in range(depth):
        # Equivalent to analytic values v_i = g_i (b_i - g_i^T beta), aggregated into c-H beta.
        residual = c - torch.einsum("bkl,bl->bk", H, beta)
        beta = beta + torch.einsum("bkl,bl->bk", Bmat, residual)
        if var_iter:
            r_alpha = gq - torch.einsum("bkl,bl->bk", H, alpha)
            alpha = alpha + torch.einsum("bkl,bl->bk", Bmat, r_alpha)
        layer_beta_mse_post.append(((beta - batch.beta_post) ** 2).mean().item())
        layer_resid_norm.append(torch.norm(residual, dim=-1).mean().item())

    mean_L = torch.einsum("bk,bk->b", gq, beta)
    if var_iter:
        var_L = cfg.noise_var + torch.einsum("bk,bk->b", gq, alpha).clamp_min(0.0)
    else:
        var_L = batch.var_exact

    # contraction radius rho(I - B H)
    with torch.no_grad():
        I = batch_eye(Bsz, K, device, dtype)
        M = I - torch.einsum("bkl,blm->bkm", Bmat, H)
        eigM = torch.linalg.eigvals(M).abs()
        rho = eigM.max(dim=-1).values.real
        contraction_radius_mean = rho.mean().item()
        contraction_radius_max = rho.max().item()

    return LoopResult(
        beta_L=beta,
        alpha_L=alpha,
        mean_L=mean_L,
        var_L=var_L.clamp_min(1e-10),
        layer_beta_mse_post=layer_beta_mse_post,
        layer_resid_norm=layer_resid_norm,
        contraction_radius_mean=contraction_radius_mean,
        contraction_radius_max=contraction_radius_max,
        precond_rank=float(pst.get("precond_rank", torch.tensor(0.0)).item()),
        eta_mean=float(pst.get("eta_mean", torch.tensor(0.0)).item()),
    )


# -----------------------------------------------------------------------------
# Metrics and experiment runners
# -----------------------------------------------------------------------------

def summarize_batch(batch: WeakBatch, loop: LoopResult, cfg: TaskConfig, depth: int, precond: str,
                    heads: int, d_head: int, seed: int, run_id: int) -> Dict[str, float | int | str]:
    beta_mse_true = ((loop.beta_L - batch.beta_true) ** 2).mean().item()
    beta_mse_post = ((loop.beta_L - batch.beta_post) ** 2).mean().item()
    pred_mse_y = ((loop.mean_L - batch.yq) ** 2).mean().item()
    mean_mse_exact = ((loop.mean_L - batch.mean_exact) ** 2).mean().item()
    var_mse_exact = ((loop.var_L - batch.var_exact) ** 2).mean().item()
    exact_pred_mse_y = ((batch.mean_exact - batch.yq) ** 2).mean().item()
    nll_iter = gaussian_nll(batch.yq, loop.mean_L, loop.var_L).mean().item()
    nll_exact = gaussian_nll(batch.yq, batch.mean_exact, batch.var_exact).mean().item()
    cov_iter, width_iter = empirical_coverage(batch.yq, loop.mean_L, loop.var_L)
    cov_exact, width_exact = empirical_coverage(batch.yq, batch.mean_exact, batch.var_exact)
    eig = batch.eigvals
    kappa = (eig[:, -1] / eig[:, 0]).mean().item()
    row: Dict[str, float | int | str] = {
        "run_id": run_id,
        "seed": seed,
        "K": cfg.K,
        "prompt_len": cfg.prompt_len,
        "design": cfg.design,
        "cond": cfg.cond,
        "noise_var": cfg.noise_var,
        "prior_var": cfg.prior_var,
        "depth": depth,
        "precond": precond,
        "heads": heads,
        "d_head": d_head,
        "capacity_rank": heads * d_head,
        "precond_rank": loop.precond_rank,
        "eta_mean": loop.eta_mean,
        "kappa_mean": kappa,
        "eig_min_mean": eig[:, 0].mean().item(),
        "eig_max_mean": eig[:, -1].mean().item(),
        "contraction_radius_mean": loop.contraction_radius_mean,
        "contraction_radius_max": loop.contraction_radius_max,
        "beta_mse_true": beta_mse_true,
        "beta_mse_post": beta_mse_post,
        "pred_mse_y": pred_mse_y,
        "mean_mse_exact": mean_mse_exact,
        "var_mse_exact": var_mse_exact,
        "exact_pred_mse_y": exact_pred_mse_y,
        "nll_iter": nll_iter,
        "nll_exact": nll_exact,
        "coverage_iter": cov_iter,
        "width_iter": width_iter,
        "coverage_exact": cov_exact,
        "width_exact": width_exact,
        "layer_first_beta_mse_post": loop.layer_beta_mse_post[0] if loop.layer_beta_mse_post else float('nan'),
        "layer_last_beta_mse_post": loop.layer_beta_mse_post[-1] if loop.layer_beta_mse_post else float('nan'),
        "layer_first_resid_norm": loop.layer_resid_norm[0] if loop.layer_resid_norm else float('nan'),
        "layer_last_resid_norm": loop.layer_resid_norm[-1] if loop.layer_resid_norm else float('nan'),
    }
    return row


@dataclass
class ExpArgs:
    mode: str = "smoke"
    outdir: str = "runs_constructive_richardson"
    seed: int = 0
    device: str = "cuda"
    dtype: str = "float32"
    batch_size: int = 1024
    eval_batches: int = 8
    K: int = 16
    prompt_len: int = 128
    prior_var: float = 1.0
    noise_var: float = 0.02
    design: str = "isotropic"
    cond: float = 10.0
    spike_strength: float = 4.0
    depth: int = 16
    precond: str = "scalar_opt"
    heads: int = 4
    d_head: int = 1
    eta_mode: str = "opt"
    eta_multiplier: float = 1.0
    spectral_order: str = "small"
    beta0: str = "zero"
    var_iter: int = 1
    depth_grid: str = "1,2,4,8,16,32,64"
    prompt_grid: str = "16,32,64,128,256,512"
    K_grid: str = "4,8,16,32"
    heads_grid: str = "1,2,4,8"
    d_head_grid: str = "1,2,4,8"
    precond_grid: str = "scalar_opt,jacobi,lowrank_spectral,spectral_full"


def run_config(args: ExpArgs, cfg: TaskConfig, depth: int, precond: str, heads: int, d_head: int,
               out_csv: Path, run_id_base: int = 0) -> Dict[str, float | int | str]:
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    rows = []
    for r in range(args.eval_batches):
        batch = sample_weak_batch(args.batch_size, cfg, device)
        loop = run_constructive_loop(
            batch, cfg, depth=depth, precond=precond, heads=heads, d_head=d_head,
            eta_mode=args.eta_mode, eta_multiplier=args.eta_multiplier,
            spectral_order=args.spectral_order, beta0=args.beta0, var_iter=bool(args.var_iter)
        )
        row = summarize_batch(batch, loop, cfg, depth, precond, heads, d_head, args.seed, run_id_base + r)
        rows.append(row)
    df = pd.DataFrame(rows)
    avg = df.mean(numeric_only=True).to_dict()
    # keep categorical values from last row
    final = rows[-1].copy()
    for k, v in avg.items():
        final[k] = float(v)
    append_csv(out_csv, final)
    print(json.dumps(final, sort_keys=True))
    return final


def plot_summary(outdir: Path, csv_name: str = "results.csv") -> None:
    if plt is None:
        return
    path = outdir / csv_name
    if not path.exists():
        return
    df = pd.read_csv(path)
    try:
        if "depth" in df.columns and df["depth"].nunique() > 1:
            plt.figure()
            for key, sub in df.groupby([c for c in ["precond", "K", "prompt_len", "heads", "d_head"] if c in df.columns]):
                if len(sub) < 2:
                    continue
                sub = sub.sort_values("depth")
                plt.plot(sub.depth, sub.beta_mse_post, marker="o", label=str(key)[:80])
            plt.xscale("log"); plt.yscale("log")
            plt.xlabel("depth L"); plt.ylabel(r"MSE to posterior mean $\\beta_*$")
            if len(df) < 30:
                plt.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(outdir / "beta_mse_post_vs_depth.png", dpi=160); plt.close()
        if "prompt_len" in df.columns and df["prompt_len"].nunique() > 1:
            plt.figure()
            for key, sub in df.groupby([c for c in ["precond", "K", "depth"] if c in df.columns]):
                if len(sub) < 2:
                    continue
                sub = sub.sort_values("prompt_len")
                plt.plot(sub.prompt_len, sub.beta_mse_true, marker="o", label=str(key)[:80])
            plt.xscale("log"); plt.yscale("log")
            plt.xlabel("prompt length m"); plt.ylabel(r"MSE to true $\\beta$")
            if len(df) < 30:
                plt.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(outdir / "beta_mse_true_vs_prompt.png", dpi=160); plt.close()
        if set(["capacity_rank", "beta_mse_post"]).issubset(df.columns):
            plt.figure()
            plt.scatter(df.capacity_rank, df.beta_mse_post, c=df.K if "K" in df.columns else None)
            plt.yscale("log"); plt.xlabel("head capacity H*d_head"); plt.ylabel("MSE to posterior")
            plt.tight_layout(); plt.savefig(outdir / "mse_vs_head_capacity.png", dpi=160); plt.close()
        if set(["contraction_radius_mean", "beta_mse_post"]).issubset(df.columns):
            plt.figure()
            plt.scatter(df.contraction_radius_mean, df.beta_mse_post, c=df.depth if "depth" in df.columns else None)
            plt.yscale("log"); plt.xlabel("mean spectral radius rho(I-BH)"); plt.ylabel("MSE to posterior")
            plt.tight_layout(); plt.savefig(outdir / "mse_vs_contraction_radius.png", dpi=160); plt.close()
    except Exception as e:
        print(f"plotting failed: {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "single", "sweep_depth", "sweep_prompt", "sweep_capacity", "sweep_precond"], default="smoke")
    p.add_argument("--outdir", default="runs_constructive_richardson")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--prior-var", type=float, default=1.0)
    p.add_argument("--noise-var", type=float, default=0.02)
    p.add_argument("--design", choices=["isotropic", "correlated", "spiked"], default="isotropic")
    p.add_argument("--cond", type=float, default=10.0)
    p.add_argument("--spike-strength", type=float, default=4.0)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--precond", choices=["scalar_opt", "scalar_lmax", "scalar_half_lmax", "jacobi", "diagonal_exact", "lowrank_spectral", "spectral_full", "block_jacobi"], default="scalar_opt")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-head", type=int, default=1)
    p.add_argument("--eta-mode", choices=["opt", "lmax", "half_lmax"], default="opt")
    p.add_argument("--eta-multiplier", type=float, default=1.0)
    p.add_argument("--spectral-order", choices=["small", "large", "mixed"], default="small")
    p.add_argument("--beta0", choices=["zero", "prior_sample"], default="zero")
    p.add_argument("--var-iter", type=int, default=1)
    p.add_argument("--depth-grid", default="1,2,4,8,16,32,64")
    p.add_argument("--prompt-grid", default="16,32,64,128,256,512")
    p.add_argument("--K-grid", default="4,8,16,32")
    p.add_argument("--heads-grid", default="1,2,4,8")
    p.add_argument("--d-head-grid", default="1,2,4,8")
    p.add_argument("--precond-grid", default="scalar_opt,jacobi,lowrank_spectral,spectral_full")
    ns = p.parse_args()
    args = ExpArgs(**vars(ns))
    if args.mode == "smoke":
        args.batch_size = min(args.batch_size, 128)
        args.eval_batches = 2
        args.K = min(args.K, 8)
        args.prompt_len = min(args.prompt_len, 64)
        args.depth = min(args.depth, 8)
    set_seed(args.seed)
    outdir = ensure_dir(Path(args.outdir) / f"{args.mode}_{int(time.time())}")
    with (outdir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    out_csv = outdir / "results.csv"

    base_cfg = TaskConfig(K=args.K, prompt_len=args.prompt_len, prior_var=args.prior_var,
                          noise_var=args.noise_var, design=args.design, cond=args.cond,
                          spike_strength=args.spike_strength, dtype=args.dtype)

    rid = 0
    if args.mode in ["smoke", "single"]:
        run_config(args, base_cfg, args.depth, args.precond, args.heads, args.d_head, out_csv, rid)
    elif args.mode == "sweep_depth":
        for L in parse_grid(args.depth_grid, int):
            run_config(args, base_cfg, L, args.precond, args.heads, args.d_head, out_csv, rid); rid += 1000
    elif args.mode == "sweep_prompt":
        for m in parse_grid(args.prompt_grid, int):
            cfg = TaskConfig(**asdict(base_cfg)); cfg.prompt_len = m
            run_config(args, cfg, args.depth, args.precond, args.heads, args.d_head, out_csv, rid); rid += 1000
    elif args.mode == "sweep_capacity":
        for K in parse_grid(args.K_grid, int):
            for H in parse_grid(args.heads_grid, int):
                for dh in parse_grid(args.d_head_grid, int):
                    cfg = TaskConfig(**asdict(base_cfg)); cfg.K = K
                    run_config(args, cfg, args.depth, args.precond, H, dh, out_csv, rid); rid += 1000
    elif args.mode == "sweep_precond":
        for pre in parse_grid(args.precond_grid, str):
            run_config(args, base_cfg, args.depth, pre, args.heads, args.d_head, out_csv, rid); rid += 1000
    else:
        raise ValueError(args.mode)

    plot_summary(outdir)
    print(f"\nSaved results to: {outdir}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
