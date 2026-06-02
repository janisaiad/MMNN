#!/usr/bin/env python3
"""
Bayesian ICL / UQ lab for looping Transformers and KRR/Richardson baselines.

This script is designed to test the core mechanism discussed in the papers:
  - Transformers as in-context solvers for posterior predictive distributions (PPD)
  - KRR / GP posterior mean and variance via iterative Richardson solvers
  - recurrent/looping Transformers as amortized iterative Bayesian solvers
  - synthetic Bayesian tasks: Bayesian linear regression, RBF GP regression,
    and weak-form inverse ridge tasks G beta = b.

It is intentionally self-contained: PyTorch + numpy + pandas + matplotlib only.

Typical commands:
  python bayes_looping_transformer_uq_lab.py --mode smoke --device cuda
  python bayes_looping_transformer_uq_lab.py --mode train --task weak_inverse --device cuda
  python bayes_looping_transformer_uq_lab.py --mode richardson_sweep --task rbf --device cuda
  python bayes_looping_transformer_uq_lab.py --mode depth_sweep --task blr --device cuda

Author: generated for the PDE-ICL / Bayesian retrieval research program.
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
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_cholesky(K: Tensor, jitter: float = 1e-5, max_tries: int = 8) -> Tensor:
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    last_err = None
    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(K + jitter * (10 ** i) * eye)
        except RuntimeError as e:
            last_err = e
    raise last_err


def solve_spd(A: Tensor, b: Tensor, jitter: float = 1e-6) -> Tensor:
    # A: [B,n,n] or [n,n], b: [B,n,*] or [n,*]
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A + jitter * eye, b)


def gaussian_nll(y: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = torch.clamp(var, min=1e-8)
    return 0.5 * (torch.log(2 * torch.pi * var) + (y - mean) ** 2 / var)


def normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def normal_bin_probs(mean: Tensor, var: Tensor, bin_edges: Tensor) -> Tensor:
    # mean,var: [B], edges [C+1]
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    z = (bin_edges[None, :] - mean[:, None]) / std[:, None]
    cdf = normal_cdf(z)
    probs = cdf[:, 1:] - cdf[:, :-1]
    probs = torch.clamp(probs, min=1e-12)
    return probs / probs.sum(dim=-1, keepdim=True)


def bin_targets(y: Tensor, bin_edges: Tensor) -> Tensor:
    # returns class in [0,C-1]
    idx = torch.bucketize(y.detach(), bin_edges) - 1
    return torch.clamp(idx, 0, len(bin_edges) - 2).long()


def empirical_coverage(y: Tensor, mean: Tensor, var: Tensor, alpha: float = 0.05) -> Tuple[float, float]:
    # Normal credible interval with approx 1-alpha; use z for common 95 only or approximate.
    if abs(alpha - 0.05) < 1e-9:
        z = 1.959963984540054
    elif abs(alpha - 0.10) < 1e-9:
        z = 1.6448536269514722
    else:
        # fallback rough via torch distribution inverse unavailable; use 95
        z = 1.959963984540054
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    lo = mean - z * std
    hi = mean + z * std
    cov = ((y >= lo) & (y <= hi)).float().mean().item()
    width = (hi - lo).mean().item()
    return cov, width


def total_variation(p: Tensor, q: Tensor) -> Tensor:
    return 0.5 * torch.abs(p - q).sum(dim=-1)


# -----------------------------------------------------------------------------
# Synthetic Bayesian tasks
# -----------------------------------------------------------------------------

@dataclass
class Batch:
    x_ctx: Tensor      # [B,n,d_x]
    y_ctx: Tensor      # [B,n]
    x_q: Tensor        # [B,d_x]
    y_q: Tensor        # [B]
    exact_mean: Tensor # [B]
    exact_var: Tensor  # [B]
    extra: Dict[str, Tensor]


class BayesianTask:
    name: str
    d_x: int

    def sample(self, batch: int, n: int, device: torch.device) -> Batch:
        raise NotImplementedError

    def exact_ppd(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def richardson(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor, steps: int, step_size: Optional[float] = None,
                   normalized: bool = False) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        raise NotImplementedError


class BLRTask(BayesianTask):
    """Bayesian linear regression: w ~ N(0, prior_var I), y = x^T w + eps."""
    name = "blr"

    def __init__(self, d: int = 8, prior_var: float = 1.0, noise_var: float = 0.05, x_scale: float = 1.0):
        self.d_x = d
        self.d = d
        self.prior_var = prior_var
        self.noise_var = noise_var
        self.x_scale = x_scale

    def sample(self, batch: int, n: int, device: torch.device) -> Batch:
        X = self.x_scale * torch.randn(batch, n, self.d, device=device) / math.sqrt(self.d)
        xq = self.x_scale * torch.randn(batch, self.d, device=device) / math.sqrt(self.d)
        w = torch.randn(batch, self.d, device=device) * math.sqrt(self.prior_var)
        y = torch.einsum("bnd,bd->bn", X, w) + torch.randn(batch, n, device=device) * math.sqrt(self.noise_var)
        yq = torch.einsum("bd,bd->b", xq, w) + torch.randn(batch, device=device) * math.sqrt(self.noise_var)
        m, v = self.exact_ppd(X, y, xq)
        return Batch(X, y, xq, yq, m, v, {"w": w})

    def exact_ppd(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor) -> Tuple[Tensor, Tensor]:
        B, n, d = x_ctx.shape
        prior_prec = 1.0 / self.prior_var
        noise_prec = 1.0 / self.noise_var
        eye = torch.eye(d, device=x_ctx.device).expand(B, d, d)
        H = prior_prec * eye + noise_prec * torch.einsum("bnd,bne->bde", x_ctx, x_ctx)
        c = noise_prec * torch.einsum("bnd,bn->bd", x_ctx, y_ctx)
        Sigma_post = torch.linalg.inv(H)
        mu = torch.einsum("bde,be->bd", Sigma_post, c)
        mean = torch.einsum("bd,bd->b", x_q, mu)
        var_latent = torch.einsum("bd,bde,be->b", x_q, Sigma_post, x_q)
        var = var_latent + self.noise_var
        return mean, torch.clamp(var, min=1e-8)

    def richardson(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor, steps: int, step_size: Optional[float] = None,
                   normalized: bool = False) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        B, n, d = x_ctx.shape
        prior_prec = 1.0 / self.prior_var
        noise_prec = 1.0 / self.noise_var
        eye = torch.eye(d, device=x_ctx.device).expand(B, d, d)
        H = prior_prec * eye + noise_prec * torch.einsum("bnd,bne->bde", x_ctx, x_ctx)
        c = noise_prec * torch.einsum("bnd,bn->bd", x_ctx, y_ctx)
        eig = torch.linalg.eigvalsh(H)
        lmax = eig[:, -1]
        lmin = eig[:, 0]
        if step_size is None:
            # batch conservative step size
            eta = (2.0 / (lmax.max() + lmin.min())).item()
        else:
            eta = step_size
        beta = torch.zeros(B, d, device=x_ctx.device)
        residuals = []
        if normalized:
            # Jacobi/Richardson preconditioning
            diag = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-8)
            for _ in range(steps):
                r = c - torch.einsum("bde,be->bd", H, beta)
                beta = beta + eta * r / diag
                residuals.append(torch.norm(r, dim=-1).mean())
        else:
            for _ in range(steps):
                r = c - torch.einsum("bde,be->bd", H, beta)
                beta = beta + eta * r
                residuals.append(torch.norm(r, dim=-1).mean())
        # approximate posterior mean with exact posterior variance for diagnostics, unless also iterative variance wanted.
        Sigma_post = torch.linalg.inv(H)
        mean = torch.einsum("bd,bd->b", x_q, beta)
        var = torch.einsum("bd,bde,be->b", x_q, Sigma_post, x_q) + self.noise_var
        stats = {"eta": torch.tensor(eta), "kappa": (lmax / lmin).mean(), "resid_last": residuals[-1] if residuals else torch.tensor(0.)}
        return mean, var, stats


class WeakInverseTask(BLRTask):
    """Weak-form inverse ridge task G beta = b.

    This is mathematically BLR with x=g weak-form rows, y=b, query h, target h^T beta.
    It is separated as a named task because it is the canonical inverse PDE abstraction.
    """
    name = "weak_inverse"

    def __init__(self, K: int = 8, prior_var: float = 1.0, noise_var: float = 0.02,
                 row_scale: float = 1.0, correlated_design: bool = False, cond: float = 10.0):
        super().__init__(d=K, prior_var=prior_var, noise_var=noise_var, x_scale=row_scale)
        self.K = K
        self.correlated_design = correlated_design
        self.cond = cond
        if correlated_design:
            # fixed covariance eigenvalues for rows g_i
            vals = torch.logspace(0, math.log10(cond), K)
            self._cov_sqrt_cpu = torch.diag(torch.sqrt(vals / vals.mean()))
        else:
            self._cov_sqrt_cpu = None

    def sample(self, batch: int, n: int, device: torch.device) -> Batch:
        if self.correlated_design:
            C = self._cov_sqrt_cpu.to(device)
            G = torch.randn(batch, n, self.K, device=device) @ C.T / math.sqrt(self.K)
            h = torch.randn(batch, self.K, device=device) @ C.T / math.sqrt(self.K)
        else:
            G = torch.randn(batch, n, self.K, device=device) / math.sqrt(self.K)
            h = torch.randn(batch, self.K, device=device) / math.sqrt(self.K)
        beta = torch.randn(batch, self.K, device=device) * math.sqrt(self.prior_var)
        b = torch.einsum("bnk,bk->bn", G, beta) + torch.randn(batch, n, device=device) * math.sqrt(self.noise_var)
        yq = torch.einsum("bk,bk->b", h, beta) + torch.randn(batch, device=device) * math.sqrt(self.noise_var)
        m, v = self.exact_ppd(G, b, h)
        return Batch(G, b, h, yq, m, v, {"beta": beta})


class RBFGPTask(BayesianTask):
    """Gaussian process regression with RBF kernel."""
    name = "rbf"

    def __init__(self, d_x: int = 1, lengthscale: float = 0.25, signal_var: float = 1.0,
                 noise_var: float = 0.02, domain: float = 1.0):
        self.d_x = d_x
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.domain = domain

    def kernel(self, X: Tensor, Y: Tensor) -> Tensor:
        # X [...,n,d], Y [...,m,d] -> [...,n,m]
        X2 = (X ** 2).sum(dim=-1, keepdim=True)
        Y2 = (Y ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)
        dist2 = X2 + Y2 - 2 * X @ Y.transpose(-1, -2)
        return self.signal_var * torch.exp(-0.5 * dist2 / (self.lengthscale ** 2))

    def sample(self, batch: int, n: int, device: torch.device) -> Batch:
        X = (2 * torch.rand(batch, n, self.d_x, device=device) - 1) * self.domain
        xq = (2 * torch.rand(batch, 1, self.d_x, device=device) - 1) * self.domain
        Xall = torch.cat([X, xq], dim=1)
        K = self.kernel(Xall, Xall)
        L = safe_cholesky(K + 1e-6 * torch.eye(n + 1, device=device))
        f = torch.einsum("bij,bj->bi", L, torch.randn(batch, n + 1, device=device))
        y = f[:, :n] + torch.randn(batch, n, device=device) * math.sqrt(self.noise_var)
        yq = f[:, n] + torch.randn(batch, device=device) * math.sqrt(self.noise_var)
        m, v = self.exact_ppd(X, y, xq[:, 0, :])
        return Batch(X, y, xq[:, 0, :], yq, m, v, {"f_q": f[:, n]})

    def exact_ppd(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor) -> Tuple[Tensor, Tensor]:
        B, n, d = x_ctx.shape
        xq = x_q[:, None, :]
        K = self.kernel(x_ctx, x_ctx)
        Ky = K + self.noise_var * torch.eye(n, device=x_ctx.device)
        kq = self.kernel(x_ctx, xq).squeeze(-1)  # [B,n]
        kqq = self.kernel(xq, xq).squeeze(-1).squeeze(-1) + self.noise_var
        alpha = torch.linalg.solve(Ky + 1e-6 * torch.eye(n, device=x_ctx.device), y_ctx[..., None]).squeeze(-1)
        mean = torch.einsum("bn,bn->b", kq, alpha)
        vtmp = torch.linalg.solve(Ky + 1e-6 * torch.eye(n, device=x_ctx.device), kq[..., None]).squeeze(-1)
        var = kqq - torch.einsum("bn,bn->b", kq, vtmp)
        return mean, torch.clamp(var, min=1e-8)

    def richardson(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor, steps: int, step_size: Optional[float] = None,
                   normalized: bool = False) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        B, n, d = x_ctx.shape
        K = self.kernel(x_ctx, x_ctx)
        A = K + self.noise_var * torch.eye(n, device=x_ctx.device)
        kq = self.kernel(x_ctx, x_q[:, None, :]).squeeze(-1)
        eig = torch.linalg.eigvalsh(A)
        lmax = eig[:, -1]
        lmin = eig[:, 0]
        if step_size is None:
            eta = (2.0 / (lmax.max() + lmin.min())).item()
        else:
            eta = step_size
        alpha = torch.zeros(B, n, device=x_ctx.device)
        residuals = []
        if normalized:
            diag = torch.diagonal(A, dim1=-2, dim2=-1).clamp_min(1e-8)
            for _ in range(steps):
                r = y_ctx - torch.einsum("bij,bj->bi", A, alpha)
                alpha = alpha + eta * r / diag
                residuals.append(torch.norm(r, dim=-1).mean())
        else:
            for _ in range(steps):
                r = y_ctx - torch.einsum("bij,bj->bi", A, alpha)
                alpha = alpha + eta * r
                residuals.append(torch.norm(r, dim=-1).mean())
        mean = torch.einsum("bn,bn->b", kq, alpha)
        # Exact variance for diagnostic, plus option to compute Richardson variance later.
        vtmp = torch.linalg.solve(A + 1e-6 * torch.eye(n, device=x_ctx.device), kq[..., None]).squeeze(-1)
        kqq = self.kernel(x_q[:, None, :], x_q[:, None, :]).squeeze(-1).squeeze(-1) + self.noise_var
        var = kqq - torch.einsum("bn,bn->b", kq, vtmp)
        stats = {"eta": torch.tensor(eta), "kappa": (lmax / lmin).mean(), "resid_last": residuals[-1] if residuals else torch.tensor(0.)}
        return mean, torch.clamp(var, min=1e-8), stats


# -----------------------------------------------------------------------------
# Looping Transformer PFN model
# -----------------------------------------------------------------------------

class LoopBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 normalize_attention: bool = False, share_qk: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.normalize_attention = normalize_attention
        self.share_qk = share_qk
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = self.q if share_qk else nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, return_attn: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        # x [B,S,D]
        B, S, D = x.shape
        z = self.ln1(x)
        q = self.q(z).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(z).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(z).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.d_head)
        if mask is not None:
            # mask [S,S] boolean where True = allowed
            logits = logits.masked_fill(~mask[None, None, :, :], -1e9)
        attn = F.softmax(logits, dim=-1)
        if self.normalize_attention:
            # Jacobi-like row-sum normalization already row-stochastic; this normalizes by column load in a light way.
            # Kept simple: discourages exploding degree while staying close to standard attention.
            col = attn.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            attn = attn / col
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        y = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.dropout(self.o(y))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return (x, attn.detach() if return_attn else None)


class LoopingTransformerUQ(nn.Module):
    def __init__(self, x_dim: int, d_model: int = 128, n_heads: int = 4, depth: int = 8,
                 dropout: float = 0.0, shared: bool = True, normalize_attention: bool = False,
                 share_qk: bool = False, head_type: str = "gaussian", n_bins: int = 101,
                 y_min: float = -5.0, y_max: float = 5.0):
        super().__init__()
        self.x_dim = x_dim
        self.d_model = d_model
        self.depth = depth
        self.shared = shared
        self.head_type = head_type
        self.n_bins = n_bins
        self.register_buffer("bin_edges", torch.linspace(y_min, y_max, n_bins + 1))
        # token contains x, y, is_context, is_query
        self.inp = nn.Sequential(nn.Linear(x_dim + 3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if shared:
            self.block = LoopBlock(d_model, n_heads, dropout, normalize_attention, share_qk)
        else:
            self.blocks = nn.ModuleList([LoopBlock(d_model, n_heads, dropout, normalize_attention, share_qk) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(d_model)
        if head_type == "gaussian":
            self.out = nn.Linear(d_model, 2)  # mean, logvar
        elif head_type == "bins":
            self.out = nn.Linear(d_model, n_bins)
        else:
            raise ValueError(f"unknown head_type {head_type}")

    def make_tokens(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor) -> Tensor:
        B, n, d = x_ctx.shape
        yq_zero = torch.zeros(B, 1, device=x_ctx.device, dtype=x_ctx.dtype)
        x_all = torch.cat([x_ctx, x_q[:, None, :]], dim=1)
        y_all = torch.cat([y_ctx, yq_zero], dim=1)
        is_ctx = torch.cat([torch.ones(B, n, device=x_ctx.device), torch.zeros(B, 1, device=x_ctx.device)], dim=1)
        is_q = 1.0 - is_ctx
        tok = torch.cat([x_all, y_all[..., None], is_ctx[..., None], is_q[..., None]], dim=-1)
        return self.inp(tok)

    def causal_query_mask(self, n: int, device: torch.device) -> Tensor:
        S = n + 1
        # All context tokens can attend to context tokens; query can attend to all context and itself.
        mask = torch.ones(S, S, device=device, dtype=torch.bool)
        return mask

    def forward(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor, return_diagnostics: bool = False):
        B, n, _ = x_ctx.shape
        x = self.make_tokens(x_ctx, y_ctx, x_q)
        mask = self.causal_query_mask(n, x_ctx.device)
        attn_last = None
        spectra = []
        for l in range(self.depth):
            if self.shared:
                x, attn_last = self.block(x, mask, return_attn=return_diagnostics)
            else:
                x, attn_last = self.blocks[l](x, mask, return_attn=return_diagnostics)
            if return_diagnostics:
                with torch.no_grad():
                    # covariance spectrum of token states, averaged rough proxy
                    z = self.final_ln(x)
                    cov = torch.einsum("bsd,bse->bde", z, z) / z.shape[1]
                    ev = torch.linalg.eigvalsh(cov).mean(dim=0)
                    spectra.append(ev.detach().cpu())
        qtok = self.final_ln(x[:, -1, :])
        out = self.out(qtok)
        if self.head_type == "gaussian":
            mean = out[:, 0]
            var = F.softplus(out[:, 1]) + 1e-5
            diag = {"attn_last": attn_last, "spectra": spectra} if return_diagnostics else {}
            return mean, var, diag
        logits = out
        probs = F.softmax(logits, dim=-1)
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        mean = (probs * centers[None, :]).sum(dim=-1)
        var = (probs * (centers[None, :] - mean[:, None]) ** 2).sum(dim=-1) + 1e-5
        diag = {"attn_last": attn_last, "spectra": spectra, "probs": probs, "logits": logits} if return_diagnostics else {}
        return mean, var, diag


# -----------------------------------------------------------------------------
# Experiment logic
# -----------------------------------------------------------------------------

@dataclass
class ExpConfig:
    task: str = "weak_inverse"
    K: int = 8
    d_x: int = 8
    gp_dim: int = 1
    lengthscale: float = 0.25
    noise_var: float = 0.02
    prior_var: float = 1.0
    correlated_design: int = 0
    cond: float = 10.0
    batch_size: int = 128
    train_context: int = 128
    eval_context: int = 128
    steps: int = 2000
    lr: float = 3e-4
    depth: int = 8
    d_model: int = 128
    heads: int = 4
    shared: int = 1
    normalize_attention: int = 0
    share_qk: int = 0
    head_type: str = "gaussian"
    n_bins: int = 101
    y_min: float = -5.0
    y_max: float = 5.0
    eval_every: int = 200
    seed: int = 0
    device: str = "cuda"
    outdir: str = "runs_bayes_looping"
    num_eval_batches: int = 16


def make_task(cfg: ExpConfig) -> BayesianTask:
    if cfg.task == "blr":
        return BLRTask(d=cfg.d_x, prior_var=cfg.prior_var, noise_var=cfg.noise_var)
    if cfg.task == "weak_inverse":
        return WeakInverseTask(K=cfg.K, prior_var=cfg.prior_var, noise_var=cfg.noise_var,
                               correlated_design=bool(cfg.correlated_design), cond=cfg.cond)
    if cfg.task == "rbf":
        return RBFGPTask(d_x=cfg.gp_dim, lengthscale=cfg.lengthscale, noise_var=cfg.noise_var)
    raise ValueError(f"unknown task {cfg.task}")


def evaluate_model(model: LoopingTransformerUQ, task: BayesianTask, cfg: ExpConfig, n_ctx: int,
                   device: torch.device, tag: str = "eval") -> Dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for _ in range(cfg.num_eval_batches):
            batch = task.sample(cfg.batch_size, n_ctx, device)
            mean, var, diag = model(batch.x_ctx, batch.y_ctx, batch.x_q, return_diagnostics=False)
            mse = ((mean - batch.y_q) ** 2).mean()
            mean_mse = ((mean - batch.exact_mean) ** 2).mean()
            var_mse = ((var - batch.exact_var) ** 2).mean()
            nll = gaussian_nll(batch.y_q, mean, var).mean()
            exact_nll = gaussian_nll(batch.y_q, batch.exact_mean, batch.exact_var).mean()
            cov, width = empirical_coverage(batch.y_q, mean, var)
            exact_cov, exact_width = empirical_coverage(batch.y_q, batch.exact_mean, batch.exact_var)
            row = dict(mse=mse.item(), mean_mse=mean_mse.item(), var_mse=var_mse.item(),
                       nll=nll.item(), exact_nll=exact_nll.item(), coverage=cov, width=width,
                       exact_coverage=exact_cov, exact_width=exact_width)
            if model.head_type == "bins":
                _, _, diag2 = model(batch.x_ctx, batch.y_ctx, batch.x_q, return_diagnostics=True)
                p_model = diag2["probs"]
                p_exact = normal_bin_probs(batch.exact_mean, batch.exact_var, model.bin_edges)
                row["tv_ppd"] = total_variation(p_model, p_exact).mean().item()
                row["bin_ce"] = F.cross_entropy(diag2["logits"], bin_targets(batch.y_q, model.bin_edges)).item()
            rows.append(row)
    out = {f"{tag}_{k}": float(np.mean([r[k] for r in rows if k in r])) for k in rows[0].keys()}
    return out


def train(cfg: ExpConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    task = make_task(cfg)
    outdir = ensure_dir(Path(cfg.outdir) / f"{cfg.task}_L{cfg.depth}_D{cfg.d_model}_H{cfg.heads}_shared{cfg.shared}_norm{cfg.normalize_attention}_{int(time.time())}")
    with (outdir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)
    model = LoopingTransformerUQ(
        x_dim=task.d_x, d_model=cfg.d_model, n_heads=cfg.heads, depth=cfg.depth,
        shared=bool(cfg.shared), normalize_attention=bool(cfg.normalize_attention), share_qk=bool(cfg.share_qk),
        head_type=cfg.head_type, n_bins=cfg.n_bins, y_min=cfg.y_min, y_max=cfg.y_max,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    log_path = outdir / "log.csv"
    for step in range(1, cfg.steps + 1):
        model.train()
        batch = task.sample(cfg.batch_size, cfg.train_context, device)
        mean, var, diag = model(batch.x_ctx, batch.y_ctx, batch.x_q, return_diagnostics=False)
        if cfg.head_type == "gaussian":
            loss = gaussian_nll(batch.y_q, mean, var).mean()
        else:
            _, _, diag2 = model(batch.x_ctx, batch.y_ctx, batch.x_q, return_diagnostics=True)
            loss = F.cross_entropy(diag2["logits"], bin_targets(batch.y_q, model.bin_edges))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == 1 or step == cfg.steps:
            ev_train_ctx = evaluate_model(model, task, cfg, cfg.train_context, device, tag="trainctx")
            ev_eval_ctx = evaluate_model(model, task, cfg, cfg.eval_context, device, tag="evalctx")
            # Richardson exact-algorithm baseline on same eval context
            with torch.no_grad():
                b2 = task.sample(cfg.batch_size, cfg.eval_context, device)
                rmean, rvar, rstats = task.richardson(b2.x_ctx, b2.y_ctx, b2.x_q, steps=cfg.depth, normalized=bool(cfg.normalize_attention))
                rich_mse = ((rmean - b2.y_q) ** 2).mean().item()
                rich_mean_mse = ((rmean - b2.exact_mean) ** 2).mean().item()
                rich_var_mse = ((rvar - b2.exact_var) ** 2).mean().item()
            row = dict(step=step, loss=loss.item(), task=cfg.task, train_context=cfg.train_context,
                       eval_context=cfg.eval_context, depth=cfg.depth, d_model=cfg.d_model, heads=cfg.heads,
                       shared=cfg.shared, normalize_attention=cfg.normalize_attention, share_qk=cfg.share_qk,
                       rich_mse=rich_mse, rich_mean_mse=rich_mean_mse, rich_var_mse=rich_var_mse,
                       rich_kappa=float(rstats["kappa"].item()), rich_resid_last=float(rstats["resid_last"].item()),
                       **ev_train_ctx, **ev_eval_ctx)
            append_csv(log_path, row)
            print(json.dumps(row, sort_keys=True))
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, outdir / "model.pt")
    return outdir


def richardson_sweep(cfg: ExpConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    task = make_task(cfg)
    outdir = ensure_dir(Path(cfg.outdir) / f"richardson_{cfg.task}_{int(time.time())}")
    log_path = outdir / "richardson.csv"
    contexts = [int(x) for x in str(getattr(cfg, "sweep_contexts", "16,32,64,128,256,512")).split(",")]
    depths = [int(x) for x in str(getattr(cfg, "sweep_depths", "1,2,4,8,16,32,64")).split(",")]
    for n_ctx in contexts:
        for L in depths:
            vals = []
            for _ in range(cfg.num_eval_batches):
                b = task.sample(cfg.batch_size, n_ctx, device)
                exact_mse = ((b.exact_mean - b.y_q) ** 2).mean().item()
                for normalized in [0, 1]:
                    m, v, stats = task.richardson(b.x_ctx, b.y_ctx, b.x_q, steps=L, normalized=bool(normalized))
                    vals.append(dict(n_ctx=n_ctx, depth=L, normalized=normalized,
                                     pred_mse=((m - b.y_q) ** 2).mean().item(),
                                     mean_mse=((m - b.exact_mean) ** 2).mean().item(),
                                     var_mse=((v - b.exact_var) ** 2).mean().item(),
                                     exact_mse=exact_mse,
                                     kappa=float(stats["kappa"].item()),
                                     resid_last=float(stats["resid_last"].item())))
            df = pd.DataFrame(vals)
            for normalized in [0, 1]:
                sub = df[df.normalized == normalized]
                row = dict(task=cfg.task, n_ctx=n_ctx, depth=L, normalized=normalized,
                           pred_mse=sub.pred_mse.mean(), mean_mse=sub.mean_mse.mean(), var_mse=sub.var_mse.mean(),
                           exact_mse=sub.exact_mse.mean(), kappa=sub.kappa.mean(), resid_last=sub.resid_last.mean())
                append_csv(log_path, row)
                print(row)
    make_basic_plots(outdir)
    return outdir


def depth_sweep(cfg: ExpConfig) -> Path:
    depths = [int(x) for x in getattr(cfg, "sweep_depths", "2,4,8,16,32").split(",")]
    contexts = [int(x) for x in getattr(cfg, "sweep_contexts", str(cfg.train_context)).split(",")]
    parent = ensure_dir(Path(cfg.outdir) / f"depth_sweep_{cfg.task}_{int(time.time())}")
    rows = []
    for nctx in contexts:
        for L in depths:
            cfg2 = ExpConfig(**asdict(cfg))
            cfg2.depth = L
            cfg2.train_context = nctx
            cfg2.eval_context = cfg.eval_context
            cfg2.outdir = str(parent)
            out = train(cfg2)
            log = pd.read_csv(out / "log.csv")
            last = log.iloc[-1].to_dict()
            last["run_dir"] = str(out)
            rows.append(last)
            pd.DataFrame(rows).to_csv(parent / "summary.csv", index=False)
    make_basic_plots(parent)
    return parent


def make_basic_plots(outdir: Path) -> None:
    if plt is None:
        return
    for csv_name in ["richardson.csv", "summary.csv", "log.csv"]:
        path = outdir / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        try:
            if "depth" in df.columns and "mean_mse" in df.columns:
                plt.figure()
                for key, sub in df.groupby([c for c in ["normalized", "n_ctx"] if c in df.columns]):
                    label = str(key)
                    plt.plot(sub["depth"], sub["mean_mse"], marker="o", label=label)
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel("depth / Richardson steps")
                plt.ylabel("posterior mean MSE")
                plt.legend()
                plt.tight_layout()
                plt.savefig(outdir / f"{csv_name}_mean_mse_vs_depth.png", dpi=160)
                plt.close()
            if "evalctx_mean_mse" in df.columns:
                plt.figure()
                plt.plot(df["step"], df["evalctx_mean_mse"], label="model mean MSE")
                if "rich_mean_mse" in df.columns:
                    plt.plot(df["step"], df["rich_mean_mse"], label="Richardson baseline")
                plt.yscale("log")
                plt.xlabel("training step")
                plt.ylabel("mean MSE")
                plt.legend()
                plt.tight_layout()
                plt.savefig(outdir / f"{csv_name}_train_curve.png", dpi=160)
                plt.close()
        except Exception as e:
            print(f"plot failed for {path}: {e}")


def smoke(cfg: ExpConfig) -> Path:
    cfg.steps = min(cfg.steps, 10)
    cfg.eval_every = 5
    cfg.batch_size = min(cfg.batch_size, 16)
    cfg.train_context = min(cfg.train_context, 32)
    cfg.eval_context = min(cfg.eval_context, 32)
    return train(cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian ICL UQ lab with looping Transformers and KRR/Richardson baselines")
    p.add_argument("--mode", choices=["train", "smoke", "richardson_sweep", "depth_sweep"], default="smoke")
    p.add_argument("--task", choices=["blr", "rbf", "weak_inverse"], default="weak_inverse")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--d-x", type=int, default=8)
    p.add_argument("--gp-dim", type=int, default=1)
    p.add_argument("--lengthscale", type=float, default=0.25)
    p.add_argument("--noise-var", type=float, default=0.02)
    p.add_argument("--prior-var", type=float, default=1.0)
    p.add_argument("--correlated-design", type=int, default=0)
    p.add_argument("--cond", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-context", type=int, default=128)
    p.add_argument("--eval-context", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--shared", type=int, default=1)
    p.add_argument("--normalize-attention", type=int, default=0)
    p.add_argument("--share-qk", type=int, default=0)
    p.add_argument("--head-type", choices=["gaussian", "bins"], default="gaussian")
    p.add_argument("--n-bins", type=int, default=101)
    p.add_argument("--y-min", type=float, default=-5.0)
    p.add_argument("--y-max", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="runs_bayes_looping")
    p.add_argument("--num-eval-batches", type=int, default=16)
    p.add_argument("--sweep-depths", default="1,2,4,8,16,32,64")
    p.add_argument("--sweep-contexts", default="16,32,64,128,256,512")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_fields = {k: v for k, v in vars(args).items() if k in ExpConfig.__dataclass_fields__}
    cfg = ExpConfig(**cfg_fields)
    # attach sweep attrs dynamically
    cfg.sweep_depths = args.sweep_depths
    cfg.sweep_contexts = args.sweep_contexts
    if args.mode == "smoke":
        out = smoke(cfg)
    elif args.mode == "train":
        out = train(cfg)
    elif args.mode == "richardson_sweep":
        out = richardson_sweep(cfg)
    elif args.mode == "depth_sweep":
        out = depth_sweep(cfg)
    else:
        raise ValueError(args.mode)
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
