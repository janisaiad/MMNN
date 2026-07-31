#!/usr/bin/env python3
"""
richardson_transformer_weak_krr_lab.py

Self-contained CUDA/PyTorch lab for our PDE-ICL weak least-squares/Richardson
hypothesis and for the row-conditioned Richardson mechanism appearing in recent
Transformer/KRR papers.

The code has two main blocks.

A. Weak inverse least-squares:
    We generate synthetic prompts whose weak equations are

        G z ~= b,          G in R^{M x K}

    with z the low-dimensional coefficients of the task-specific operator
    A(z) = A0 + sum_k z_k A_k.  The ridge target is

        z_* = (G^T G + lambda I)^(-1) G^T b.

    We test:
      - exact Richardson;
      - Jacobi / spectral / low-rank preconditioning;
      - linear attention exact moment construction;
      - softmax signed-scalar routing;
      - softmax full-vector values and projected values;
      - trainable recurrent slot-attention models with shared weights;
      - diagnostics for spectra, query/slot alignment, effective update maps.

B. KRR row-conditioned Richardson:
    We generate Gaussian kernel regression tasks and compare:
      - exact KRR;
      - unpreconditioned Richardson;
      - row-conditioned Richardson using D^{-1} K, where D=diag(K 1);
      - a tiny Transformer trained on GP/KRR ICL tasks;
      - linear probes of each layer's query hidden state against Richardson
        iterates, to test whether training discovers row-conditioned iterations.

This file is intentionally broad. It is not meant to be one polished experiment;
it is a lab that lets us systematically falsify/refine our formulation.

Typical commands
----------------

Smoke test:
    python richardson_transformer_weak_krr_lab.py --mode smoke --device cuda

Constructive weak LS sweep:
    python richardson_transformer_weak_krr_lab.py --mode weak_sweep \
      --K-grid 8,16,32 --cond-grid 10,100,1000 \
      --capacity-grid below,at,above --precond-grid scalar_opt,jacobi,lowrank_spectral \
      --methods linear,softmax_scalar,softmax_vector_full,softmax_vector_projected \
      --depth 32 --device cuda --outdir runs/weak_sweep

Train weak recurrent slot Transformer:
    python richardson_transformer_weak_krr_lab.py --mode train_weak \
      --K 16 --M 128 --depth 8 --d-model 128 --heads 4 --d-head 16 \
      --value-mode affine_scalar --steps 20000 --batch-size 256 --device cuda \
      --outdir runs/train_weak_scalar

Train weak with full MLP values:
    python richardson_transformer_weak_krr_lab.py --mode train_weak \
      --K 16 --M 128 --depth 8 --d-model 128 --heads 4 --d-head 16 \
      --value-mode mlp_vector --steps 20000 --device cuda \
      --outdir runs/train_weak_mlp_vector

KRR Richardson sweep:
    python richardson_transformer_weak_krr_lab.py --mode krr_sweep \
      --n-context 64 --x-dim 1 --depth-grid 1,2,4,8,16,32,64 \
      --device cuda --outdir runs/krr_sweep

Train KRR Transformer and probe layers:
    python richardson_transformer_weak_krr_lab.py --mode train_krr_probe \
      --n-context 64 --x-dim 1 --d-model 128 --n-layers 8 --n-heads 4 \
      --steps 30000 --batch-size 128 --device cuda --outdir runs/krr_probe

Train a tied no-MLP kernel-attention HeavyBall decoder:
    python richardson_transformer_weak_krr_lab.py --mode train_krr_looped \
      --loop-solver heavy_ball --n-context 64 --x-dim 1 --depth 8 \
      --kernel-init-lengthscale 0.4 --learn-kernel 1 \
      --steps 10000 --batch-size 128 --device cuda \
      --outdir runs/krr_looped_heavy_ball
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# basic utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_outdir(outdir: str) -> Path:
    p = Path(outdir)
    if p.is_absolute():
        return p
    return project_root() / "data" / "transformers" / p.name


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_grid(s: str, typ=float) -> List:
    if s is None or s == "":
        return []
    return [typ(x) for x in str(s).split(",") if str(x).strip() != ""]


def batch_eye(B: int, K: int, device, dtype) -> Tensor:
    return torch.eye(K, device=device, dtype=dtype).expand(B, K, K)


def stable_solve(A: Tensor, b: Tensor, jitter: float = 1e-8) -> Tensor:
    K = A.shape[-1]
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A + jitter * I, b.unsqueeze(-1)).squeeze(-1)


def safe_cholesky(K: Tensor, jitter: float = 1e-6, max_tries: int = 6) -> Tensor:
    I = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    eps = jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(K + eps * I)
        except RuntimeError:
            eps *= 10.0
    return torch.linalg.cholesky(K + eps * I)


def mse(a: Tensor, b: Tensor) -> float:
    return (a - b).pow(2).mean().detach().item()


def relerr(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(eps)).mean().detach().item()


def explained_r2(y: Tensor, yhat: Tensor) -> float:
    ss_res = (y - yhat).pow(2).sum()
    ss_tot = (y - y.mean()).pow(2).sum().clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).detach().item()


# -----------------------------------------------------------------------------
# weak least-squares task generation
# -----------------------------------------------------------------------------

@dataclass
class WeakBatch:
    G: Tensor       # [B,M,K]
    b: Tensor       # [B,M]
    z_true: Tensor  # [B,K]
    H: Tensor       # [B,K,K] = G^T G + lambda I
    c: Tensor       # [B,K] = G^T b
    z_star: Tensor  # [B,K]
    eigvals: Tensor # [B,K]
    U: Tensor       # [B,K,K]


def design_sqrt(K: int, design: str, cond: float, device, dtype) -> Tensor:
    if design == "isotropic":
        return torch.eye(K, device=device, dtype=dtype)
    if design == "correlated":
        vals = torch.logspace(0.0, math.log10(cond), K, device=device, dtype=dtype)
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    if design == "spiked":
        vals = torch.ones(K, device=device, dtype=dtype)
        vals[0] = cond
        vals = vals / vals.mean()
        return torch.diag(vals.sqrt())
    raise ValueError(f"unknown design {design}")


def sample_weak_batch(
    B: int,
    M: int,
    K: int,
    lam: float,
    noise_std: float,
    design: str,
    cond: float,
    device,
    dtype=torch.float32,
) -> WeakBatch:
    Csqrt = design_sqrt(K, design, cond, device, dtype)
    G = torch.randn(B, M, K, device=device, dtype=dtype) @ Csqrt.T / math.sqrt(K)
    z_true = torch.randn(B, K, device=device, dtype=dtype)
    b = torch.einsum("bmk,bk->bm", G, z_true)
    if noise_std > 0:
        b = b + noise_std * torch.randn_like(b)
    I = batch_eye(B, K, device, dtype)
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * I
    c = torch.einsum("bmk,bm->bk", G, b)
    z_star = stable_solve(H, c)
    eigvals, U = torch.linalg.eigh(H)
    return WeakBatch(G, b, z_true, H, c, z_star, eigvals, U)


@dataclass
class ParametricABatch:
    G: Tensor
    b: Tensor
    z_true: Tensor
    z_star: Tensor
    H: Tensor
    c: Tensor
    A: Tensor
    A_basis: Tensor
    A0: Tensor
    f_star: Tensor
    u_star: Tensor


def sample_parametric_A_batch(
    B: int,
    d: int,
    K: int,
    m_prompt: int,
    lam: float,
    noise_std: float,
    basis_scale: float,
    device,
    dtype=torch.float32,
) -> ParametricABatch:
    """Finite-dimensional operator family A(z)=A0+sum z_k A_k.

    We sample prompt solutions u_i, produce f_i=A(z)u_i, and build weak rows
    using coordinate tests:
        g_{(i,r),k} = (A_k u_i)_r
        b_{(i,r)} = (f_i - A0 u_i)_r
    M = m_prompt * d.
    """
    # fixed-ish basis per batch item for now. A0 SPD, A_k small symmetric.
    I = torch.eye(d, device=device, dtype=dtype)
    A0 = (2.0 * I).expand(B, d, d).clone()
    R = torch.randn(B, K, d, d, device=device, dtype=dtype)
    A_basis = 0.5 * (R + R.transpose(-1, -2)) * (basis_scale / math.sqrt(d))
    z_true = torch.randn(B, K, device=device, dtype=dtype) * 0.5
    A = A0 + torch.einsum("bk,bkij->bij", z_true, A_basis)
    # make SPD safer by adding diagonal if needed
    mineig = torch.linalg.eigvalsh(A)[:, 0]
    shift = (0.25 - mineig).clamp_min(0.0)
    A = A + shift[:, None, None] * I
    A0_eff = A0 + shift[:, None, None] * I

    U_prompt = torch.randn(B, m_prompt, d, device=device, dtype=dtype)
    F_prompt = torch.einsum("bij,bmj->bmi", A, U_prompt)
    if noise_std > 0:
        F_prompt = F_prompt + noise_std * torch.randn_like(F_prompt)

    # rows G shape [B, m*d, K]
    Ak_u = torch.einsum("bkij,bmj->bmki", A_basis, U_prompt)  # [B,m,K,d]
    G = Ak_u.permute(0, 1, 3, 2).reshape(B, m_prompt * d, K)
    A0u = torch.einsum("bij,bmj->bmi", A0_eff, U_prompt)
    b = (F_prompt - A0u).reshape(B, m_prompt * d)

    H = torch.einsum("bmk,bml->bkl", G, G) + lam * torch.eye(K, device=device, dtype=dtype).expand(B, K, K)
    c = torch.einsum("bmk,bm->bk", G, b)
    z_star = stable_solve(H, c)

    f_star = torch.randn(B, d, device=device, dtype=dtype)
    u_star = stable_solve(A, f_star)
    return ParametricABatch(G, b, z_true, z_star, H, c, A, A_basis, A0_eff, f_star, u_star)


# -----------------------------------------------------------------------------
# preconditioners and constructive Richardson
# -----------------------------------------------------------------------------

def make_preconditioner(
    H: Tensor,
    precond: str,
    rank: int = 0,
    eta_mult: float = 1.0,
) -> Tuple[Tensor, Dict[str, float]]:
    B, K, _ = H.shape
    device, dtype = H.device, H.dtype
    I = batch_eye(B, K, device, dtype)
    eig, U = torch.linalg.eigh(H)
    lmin, lmax = eig[:, 0], eig[:, -1]
    eta_opt = eta_mult * 2.0 / (lmin + lmax).clamp_min(1e-12)

    if precond == "identity" or precond == "scalar_opt":
        P = eta_opt[:, None, None] * I
        rho = torch.linalg.eigvals(I - torch.einsum("bij,bjk->bik", P, H)).abs().real.max(-1).values
        return P, {"eta_mean": eta_opt.mean().item(), "rho_mean": rho.mean().item(), "precond_rank": 0}

    if precond == "scalar_lmax":
        eta = eta_mult / lmax.clamp_min(1e-12)
        P = eta[:, None, None] * I
        rho = torch.linalg.eigvals(I - torch.einsum("bij,bjk->bik", P, H)).abs().real.max(-1).values
        return P, {"eta_mean": eta.mean().item(), "rho_mean": rho.mean().item(), "precond_rank": 0}

    if precond == "jacobi":
        diag = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-12)
        Dinv = torch.diag_embed(1.0 / diag)
        # conservative scale based on Dinv H spectral radius
        M = torch.einsum("bij,bjk->bik", Dinv, H)
        rad = torch.linalg.eigvals(M).abs().real.max(-1).values.clamp_min(1e-12)
        eta = eta_mult / rad
        P = eta[:, None, None] * Dinv
        rho = torch.linalg.eigvals(I - torch.einsum("bij,bjk->bik", P, H)).abs().real.max(-1).values
        return P, {"eta_mean": eta.mean().item(), "rho_mean": rho.mean().item(), "precond_rank": K}

    if precond == "spectral_full":
        Pinv = torch.einsum("bkr,br,blr->bkl", U, 1.0 / eig.clamp_min(1e-12), U)
        rho = torch.linalg.eigvals(I - torch.einsum("bij,bjk->bik", Pinv, H)).abs().real.max(-1).values
        return Pinv, {"eta_mean": 1.0, "rho_mean": rho.mean().item(), "precond_rank": K}

    if precond == "lowrank_spectral":
        r = max(1, min(K, rank if rank > 0 else K // 2))
        # Use smallest eigenvalues by default, since slow modes are the small ones.
        Usel = U[:, :, :r]
        esel = eig[:, :r]
        Psel = torch.einsum("bkr,blr->bkl", Usel, Usel)
        Bsel = torch.einsum("bkr,br,blr->bkl", Usel, 1.0 / esel.clamp_min(1e-12), Usel)
        Prest = eta_opt[:, None, None] * (I - Psel)
        P = Bsel + Prest
        rho = torch.linalg.eigvals(I - torch.einsum("bij,bjk->bik", P, H)).abs().real.max(-1).values
        return P, {"eta_mean": eta_opt.mean().item(), "rho_mean": rho.mean().item(), "precond_rank": r}

    raise ValueError(precond)


@torch.no_grad()
def richardson_solve(
    G: Tensor,
    b: Tensor,
    lam: float,
    depth: int,
    precond: str = "scalar_opt",
    rank: int = 0,
    eta_mult: float = 1.0,
    return_layers: bool = False,
) -> Tuple[Tensor, Dict]:
    Bsz, M, K = G.shape
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(Bsz, K, G.device, G.dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    z_star = stable_solve(H, c)
    P, stats = make_preconditioner(H, precond, rank=rank, eta_mult=eta_mult)
    z = torch.zeros(Bsz, K, device=G.device, dtype=G.dtype)
    layers = []
    for _ in range(depth):
        r = b - torch.einsum("bmk,bk->bm", G, z)
        grad = torch.einsum("bmk,bm->bk", G, r) - lam * z
        z = z + torch.einsum("bkl,bl->bk", P, grad)
        if return_layers:
            layers.append(z.clone())
    out = {
        **stats,
        "z_mse_post": mse(z, z_star),
        "z_rel_post": relerr(z, z_star),
    }
    if return_layers:
        out["layers"] = torch.stack(layers, dim=1)
    return z, out


# -----------------------------------------------------------------------------
# constructive attention variants
# -----------------------------------------------------------------------------

def make_P_coordinate(K: int, capacity: str, device, dtype) -> Tensor:
    if capacity == "below":
        d = max(1, K // 2)
    elif capacity == "at":
        d = K
    elif capacity == "above":
        d = 2 * K
    else:
        d = int(capacity)
    P = torch.zeros(d, K, device=device, dtype=dtype)
    for a in range(min(d, K)):
        P[a, a] = 1.0
    # if above, extra rows are zero for coordinate mode, so do not double count
    return P


def constructive_attention_gradient(
    G: Tensor,
    r: Tensor,
    method: str,
    P: Tensor,
    tau: float,
    scale_context: bool = True,
    eps: float = 1e-8,
) -> Tuple[Tensor, Dict[str, float]]:
    """Return approximate G^T r using one projection head P.

    Methods:
      linear:
        P^T P G^T r.
      softmax_scalar:
        two signed softmax channels with scalar residual values; approximate.
      signed_relu_scalar:
        exact positive/negative weighting: weights proportional to relu(+-score).
      softmax_vector_full:
        softmax routing but value is full g_i r_i, so V carries full K directions.
      softmax_vector_projected:
        full value projected to P^T P subspace.
    """
    B, M, K = G.shape
    d = P.shape[0]
    Pg = torch.einsum("dk,bmk->bmd", P, G)  # [B,M,d]
    stats = {}

    if method == "linear":
        head_grad = torch.einsum("bmd,bm->bd", Pg, r)
        grad = torch.einsum("dk,bd->bk", P, head_grad)
        return grad, {"attn_entropy": 0.0, "grad_norm": grad.norm(dim=-1).mean().item()}

    if method == "softmax_scalar":
        logits_pos = Pg.transpose(1, 2) / tau        # [B,d,M]
        logits_neg = -Pg.transpose(1, 2) / tau
        Apos = torch.softmax(logits_pos, dim=-1)
        Aneg = torch.softmax(logits_neg, dim=-1)
        mpos = torch.einsum("bdm,bm->bd", Apos, r)
        mneg = torch.einsum("bdm,bm->bd", Aneg, r)
        # learned FFN would combine these; constructive approximation uses difference
        # and scale by context. This is not exact for arbitrary scores.
        msg = (mpos - mneg)
        if scale_context:
            msg = msg * M
        grad = torch.einsum("dk,bd->bk", P, msg)
        ent = (-(Apos.clamp_min(1e-12) * Apos.clamp_min(1e-12).log()).sum(-1).mean()
               -(Aneg.clamp_min(1e-12) * Aneg.clamp_min(1e-12).log()).sum(-1).mean()) / 2
        return grad, {"attn_entropy": ent.item(), "grad_norm": grad.norm(dim=-1).mean().item()}

    if method == "signed_relu_scalar":
        # Exact signed moment using two positive distributions and scalar values.
        pos = Pg.clamp_min(0.0).transpose(1, 2)  # [B,d,M]
        neg = (-Pg).clamp_min(0.0).transpose(1, 2)
        sum_pos = pos.sum(-1).clamp_min(eps)
        sum_neg = neg.sum(-1).clamp_min(eps)
        Apos = pos / sum_pos.unsqueeze(-1)
        Aneg = neg / sum_neg.unsqueeze(-1)
        mpos = torch.einsum("bdm,bm->bd", Apos, r) * sum_pos
        mneg = torch.einsum("bdm,bm->bd", Aneg, r) * sum_neg
        msg = mpos - mneg
        grad = torch.einsum("dk,bd->bk", P, msg)
        ent = (-(Apos.clamp_min(1e-12) * Apos.clamp_min(1e-12).log()).sum(-1).mean()
               -(Aneg.clamp_min(1e-12) * Aneg.clamp_min(1e-12).log()).sum(-1).mean()) / 2
        return grad, {"attn_entropy": ent.item(), "grad_norm": grad.norm(dim=-1).mean().item()}

    if method.startswith("softmax_vector"):
        # one distribution per projected coordinate, then average across coordinates.
        logits = Pg.transpose(1, 2) / tau  # [B,d,M]
        A = torch.softmax(logits, dim=-1)
        full_values = G * r.unsqueeze(-1)  # [B,M,K]
        if method == "softmax_vector_projected":
            PP = P.T @ P
            values = torch.einsum("kl,bml->bmk", PP, full_values)
        else:
            values = full_values
        # [B,d,K], then average over d and scale to sum
        msgs = torch.einsum("bdm,bmk->bdk", A, values)
        msg = msgs.mean(dim=1)
        if scale_context:
            msg = msg * M
        ent = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(-1).mean()
        return msg, {"attn_entropy": ent.item(), "grad_norm": msg.norm(dim=-1).mean().item()}

    raise ValueError(method)


@torch.no_grad()
def attention_richardson_solve(
    G: Tensor,
    b: Tensor,
    lam: float,
    depth: int,
    method: str,
    P: Tensor,
    tau: float,
    precond: str,
    eta_mult: float = 1.0,
    rank: int = 0,
) -> Dict:
    B, M, K = G.shape
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(B, K, G.device, G.dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    z_star = stable_solve(H, c)
    Pmat, pst = make_preconditioner(H, precond, rank=rank, eta_mult=eta_mult)
    z = torch.zeros(B, K, device=G.device, dtype=G.dtype)
    first_mse = None
    last_grad_mse = 0.0
    ent = 0.0
    for l in range(depth):
        r = b - torch.einsum("bmk,bk->bm", G, z)
        grad_attn, ast = constructive_attention_gradient(G, r, method, P, tau)
        grad_attn = grad_attn - lam * z
        grad_exact = c - torch.einsum("bkl,bl->bk", H, z)
        last_grad_mse = mse(grad_attn, grad_exact)
        ent = ast.get("attn_entropy", 0.0)
        z = z + torch.einsum("bkl,bl->bk", Pmat, grad_attn)
        if l == 0:
            first_mse = mse(z, z_star)
    return {
        "method": method,
        "precond": precond,
        "depth": depth,
        "z_mse_post": mse(z, z_star),
        "z_mse_true": mse(z, stable_solve(H, c)),  # same target here; kept for schema
        "z_rel_post": relerr(z, z_star),
        "first_mse_post": float(first_mse),
        "last_grad_mse": last_grad_mse,
        "attn_entropy": ent,
        **pst,
    }


# -----------------------------------------------------------------------------
# trainable weak recurrent slot-attention model
# -----------------------------------------------------------------------------

class WeakSlotTransformer(nn.Module):
    def __init__(
        self,
        K: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_head: int = 32,
        n_slots: int = 8,
        depth: int = 8,
        lam: float = 1e-3,
        value_mode: str = "affine_scalar",  # affine_scalar | mlp_vector | analytic_vector
        shared: bool = True,
        ffn_hidden: int = 256,
    ):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_slots = n_slots
        self.depth = depth
        self.lam = lam
        self.value_mode = value_mode
        self.shared = shared

        token_dim = K + 2  # g, b, residual
        self.key = nn.Linear(token_dim, n_heads * d_head)
        self.query_slots = nn.Parameter(torch.randn(n_heads, n_slots, d_head) / math.sqrt(d_head))

        if value_mode == "affine_scalar":
            self.value = nn.Linear(token_dim, n_heads * 1)
            update_in = n_heads * n_slots * 1 + K
        elif value_mode == "mlp_vector":
            self.value = nn.Sequential(
                nn.Linear(token_dim, ffn_hidden),
                nn.GELU(),
                nn.Linear(ffn_hidden, n_heads * K),
            )
            update_in = n_heads * n_slots * K + K
        elif value_mode == "analytic_vector":
            self.value = None
            update_in = n_heads * n_slots * K + K
        else:
            raise ValueError(value_mode)

        self.update = nn.Sequential(
            nn.Linear(update_in, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, K),
        )
        # A small learned damping helps stability.
        self.step_scale = nn.Parameter(torch.tensor(0.1))

    def layer_step(self, G: Tensor, b: Tensor, z: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, M, K = G.shape
        r = b - torch.einsum("bmk,bk->bm", G, z)
        token = torch.cat([G, b.unsqueeze(-1), r.unsqueeze(-1)], dim=-1)
        Kh = self.key(token).view(B, M, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,M,D]
        Qh = self.query_slots.unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,S,D]
        scores = torch.einsum("bhsd,bhmd->bhsm", Qh, Kh) / math.sqrt(self.d_head)
        A = torch.softmax(scores, dim=-1)

        if self.value_mode == "affine_scalar":
            Vh = self.value(token).view(B, M, self.n_heads, 1).transpose(1, 2)  # [B,H,M,1]
        elif self.value_mode == "mlp_vector":
            Vh = self.value(token).view(B, M, self.n_heads, K).transpose(1, 2)
        else:
            # full analytic value repeated per head. This is an oracle sanity check.
            full = G * r.unsqueeze(-1)
            Vh = full.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        O = torch.einsum("bhsm,bhmv->bhsv", A, Vh).reshape(B, -1)
        inp = torch.cat([z, O], dim=-1)
        delta = self.update(inp) * self.step_scale
        z_new = z + delta
        info = {
            "residual_norm": r.norm(dim=-1).mean().detach(),
            "attn_entropy": (-(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(-1).mean()).detach(),
            "delta_norm": delta.norm(dim=-1).mean().detach(),
        }
        return z_new, info

    def forward(self, G: Tensor, b: Tensor, return_layers: bool = False) -> Tuple[Tensor, Dict]:
        B, M, K = G.shape
        z = torch.zeros(B, K, device=G.device, dtype=G.dtype)
        layers = []
        infos = []
        for _ in range(self.depth):
            z, info = self.layer_step(G, b, z)
            if return_layers:
                layers.append(z)
                infos.append(info)
        out = {}
        if return_layers:
            out["layers"] = torch.stack(layers, dim=1)
            out["infos"] = infos
        return z, out


def train_weak_model(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    metrics_path = outdir / "train_metrics.csv"
    model = WeakSlotTransformer(
        K=args.K,
        d_model=args.d_model,
        n_heads=args.heads,
        d_head=args.d_head,
        n_slots=args.n_slots,
        depth=args.depth,
        lam=args.lam,
        value_mode=args.value_mode,
        ffn_hidden=args.ffn_hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for step in range(1, args.steps + 1):
        batch = sample_weak_batch(args.batch_size, args.M, args.K, args.lam, args.noise_std, args.design, args.cond, device)
        zhat, _ = model(batch.G, batch.b)
        loss = F.mse_loss(zhat, batch.z_star)
        if args.loss_true_weight > 0:
            loss = loss + args.loss_true_weight * F.mse_loss(zhat, batch.z_true)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                evalb = sample_weak_batch(args.eval_batch_size, args.M, args.K, args.lam, args.noise_std, args.design, args.cond, device)
                z_eval, info = model(evalb.G, evalb.b, return_layers=True)
                rich_z, rich_stats = richardson_solve(evalb.G, evalb.b, args.lam, args.depth, precond="jacobi")
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "eval_mse_post": mse(z_eval, evalb.z_star),
                    "eval_mse_true": mse(z_eval, evalb.z_true),
                    "eval_rel_post": relerr(z_eval, evalb.z_star),
                    "jacobi_same_depth_mse_post": rich_stats["z_mse_post"],
                    "attn_entropy_last": float(info["infos"][-1]["attn_entropy"]) if "infos" in info else 0.0,
                    "delta_norm_last": float(info["infos"][-1]["delta_norm"]) if "infos" in info else 0.0,
                }
                append_csv(metrics_path, row)
                print(json.dumps(row, indent=2))
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / f"model_step{step}.pt")

    torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / "model_final.pt")
    evaluate_weak_model(model, args, device, outdir / "eval_final.csv")


@torch.no_grad()
def evaluate_weak_model(model: WeakSlotTransformer, args, device, csv_path: Path) -> None:
    for cond in parse_grid(args.eval_cond_grid, float):
        for K_dummy in [args.K]:
            batch = sample_weak_batch(args.eval_batch_size, args.M, args.K, args.lam, args.noise_std, args.design, cond, device)
            zhat, info = model(batch.G, batch.b, return_layers=True)
            # Fit effective linear update delta ~= B_eff grad using one-step data from current model.
            z0 = torch.zeros_like(batch.z_true)
            r0 = batch.b - torch.einsum("bmk,bk->bm", batch.G, z0)
            grad0 = torch.einsum("bmk,bm->bk", batch.G, r0) - args.lam * z0
            z1, _ = model.layer_step(batch.G, batch.b, z0)
            delta0 = z1 - z0
            # least squares B_eff: delta = grad @ B_eff^T
            X = grad0
            Y = delta0
            B_eff_T = torch.linalg.lstsq(X, Y).solution  # [K,K] maps X to Y
            B_eff = B_eff_T.T
            Hmean = batch.H.mean(0)
            rho_eff = torch.linalg.eigvals(torch.eye(args.K, device=device) - B_eff @ Hmean).abs().real.max().item()
            row = {
                "cond": cond,
                "eval_mse_post": mse(zhat, batch.z_star),
                "eval_mse_true": mse(zhat, batch.z_true),
                "eval_rel_post": relerr(zhat, batch.z_star),
                "eff_rho_meanH": rho_eff,
                "layer0_mse_post": mse(info["layers"][:, 0], batch.z_star),
                "layerlast_mse_post": mse(info["layers"][:, -1], batch.z_star),
            }
            append_csv(csv_path, row)
            print("EVAL", json.dumps(row, indent=2))


# -----------------------------------------------------------------------------
# KRR / row-conditioned Richardson
# -----------------------------------------------------------------------------

def rbf_kernel(x: Tensor, y: Tensor, lengthscale: float, variance: float = 1.0) -> Tensor:
    # x [B,N,D], y [B,M,D]
    dist2 = torch.cdist(x / lengthscale, y / lengthscale).pow(2)
    return variance * torch.exp(-0.5 * dist2)


@dataclass
class KRRBatch:
    x_ctx: Tensor
    y_ctx: Tensor
    x_q: Tensor
    y_q: Tensor
    K: Tensor
    kq: Tensor
    alpha_exact: Tensor
    mean_exact: Tensor


def sample_gp_krr_batch(
    B: int,
    n: int,
    x_dim: int,
    lengthscale: float,
    noise_var: float,
    lam: float,
    device,
    dtype=torch.float32,
) -> KRRBatch:
    # Sample x uniformly in [-1,1]^d, sample f from GP at context+query.
    x_all = torch.rand(B, n + 1, x_dim, device=device, dtype=dtype) * 2 - 1
    x_ctx = x_all[:, :n]
    x_q = x_all[:, n:n+1]
    K_all = rbf_kernel(x_all, x_all, lengthscale)
    L = safe_cholesky(K_all, jitter=1e-5)
    f_all = torch.einsum("bij,bj->bi", L, torch.randn(B, n + 1, device=device, dtype=dtype))
    y_all = f_all + math.sqrt(noise_var) * torch.randn_like(f_all)
    y_ctx = y_all[:, :n]
    y_q = y_all[:, n]

    Kctx = rbf_kernel(x_ctx, x_ctx, lengthscale)
    kq = rbf_kernel(x_q, x_ctx, lengthscale).squeeze(1)
    A = Kctx + lam * torch.eye(n, device=device, dtype=dtype).expand(B, n, n)
    alpha = stable_solve(A, y_ctx)
    mean = torch.einsum("bn,bn->b", kq, alpha)
    return KRRBatch(x_ctx, y_ctx, x_q.squeeze(1), y_q, Kctx, kq, alpha, mean)


@torch.no_grad()
def krr_richardson_iterates(
    K: Tensor,
    y: Tensor,
    kq: Tensor,
    lam: float,
    depth: int,
    eta: float = 1.0,
    mode: str = "rowcond",
) -> Tuple[Tensor, Tensor]:
    B, n, _ = K.shape
    alpha = torch.zeros(B, n, device=K.device, dtype=K.dtype)
    preds = []
    alphas = []
    if mode == "rowcond":
        Dinv = 1.0 / K.sum(-1).clamp_min(1e-8)
    else:
        # scalar step safe for unprecond
        eig = torch.linalg.eigvalsh(K + lam * torch.eye(n, device=K.device, dtype=K.dtype).expand(B, n, n))
        eta_un = 2.0 / (eig[:, 0] + eig[:, -1]).clamp_min(1e-8)
    for _ in range(depth):
        if mode == "rowcond":
            Kalpha = torch.einsum("bij,bj->bi", K, alpha)
            residual = y - Kalpha - lam * alpha
            alpha = alpha + eta * Dinv * residual
        elif mode == "unprecond":
            Kalpha = torch.einsum("bij,bj->bi", K, alpha)
            residual = y - Kalpha - lam * alpha
            alpha = alpha + eta_un[:, None] * residual
        else:
            raise ValueError(mode)
        preds.append(torch.einsum("bn,bn->b", kq, alpha))
        alphas.append(alpha.clone())
    return torch.stack(preds, dim=1), torch.stack(alphas, dim=1)


def _scalar_logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class NoMLPKernelAttentionLoop(nn.Module):
    """Tied softmax-kernel attention with Richardson or HeavyBall state.

    The RBF logits are a standard one-head attention score after quadratic
    feature augmentation.  The context attention output is exactly
    ``D^{-1} K (y-u)``.  Mean and variance may occupy separate value channels;
    they do not require separate heads.  The normalizer exposes ``D`` for the
    ridge skip channel; no feed-forward network approximates a reciprocal or
    product.
    """

    def __init__(
        self,
        depth: int,
        lam: float,
        iteration: str,
        init_lengthscale: float,
        learn_kernel: bool,
        step_init: float = 0.8,
        momentum_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.lam = lam
        self.iteration = iteration
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(init_lengthscale)),
            requires_grad=learn_kernel,
        )
        if iteration == "heavy_ball":
            beta_fraction = min(max(momentum_init / 0.999, 1e-6), 1.0 - 1e-6)
            self.raw_momentum = nn.Parameter(torch.tensor(_scalar_logit(beta_fraction)))
        elif iteration == "richardson":
            self.register_buffer("raw_momentum", torch.tensor(float("-inf")))
        else:
            raise ValueError(f"unknown iteration {iteration}")
        momentum = momentum_init if iteration == "heavy_ball" else 0.0
        step_cap = 2.0 * (1.0 + momentum) / (1.0 + lam)
        step_fraction = step_init / (0.999 * step_cap)
        self.raw_step = nn.Parameter(torch.tensor(_scalar_logit(step_fraction)))

    def coefficients(self) -> Tuple[Tensor, Tensor]:
        if self.iteration == "heavy_ball":
            momentum = 0.999 * torch.sigmoid(self.raw_momentum)
        else:
            momentum = self.raw_step.new_zeros(())
        # Since K_ii=1, D_i >= 1 and lambda_max(D^{-1}(K+lambda I))
        # is bounded by 1+lambda for the exact row-normalized RBF operator.
        stable_step_cap = 2.0 * (1.0 + momentum) / (1.0 + self.lam)
        step = 0.999 * stable_step_cap * torch.sigmoid(self.raw_step)
        return step, momentum

    def kernel_attention(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        lengthscale = self.log_lengthscale.exp().clamp_min(1e-4)
        scaled = x / lengthscale
        dist2 = (scaled[:, :, None, :] - scaled[:, None, :, :]).pow(2).sum(dim=-1)
        logits = -0.5 * dist2
        attention = torch.softmax(logits, dim=-1)
        degree = torch.exp(logits).sum(dim=-1)
        return attention, degree, lengthscale

    def forward(
        self,
        x_ctx: Tensor,
        y_ctx: Tensor,
        x_q: Tensor,
        return_layers: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        attention, degree, lengthscale = self.kernel_attention(x_ctx)
        degree_inv = degree.reciprocal()
        ridge_diag = self.lam * degree_inv
        step, momentum = self.coefficients()
        query_scaled = x_q[:, None, :] / lengthscale
        context_scaled = x_ctx / lengthscale
        query_logits = -0.5 * (query_scaled - context_scaled).pow(2).sum(dim=-1)
        query_kernel = torch.exp(query_logits)
        values = torch.stack([y_ctx, query_kernel], dim=-1)
        state_prev = torch.zeros_like(values)
        state = torch.zeros_like(values)
        query_prev = torch.zeros_like(values[:, 0])
        query_state = torch.zeros_like(query_prev)
        state_layers = []
        pred_layers = []

        query_attention = torch.softmax(query_logits, dim=-1)
        query_degree = torch.exp(query_logits).sum(dim=-1)
        query_ridge = self.lam / query_degree

        for _ in range(self.depth):
            value_residual = values - state
            context_correction = torch.einsum("bij,bjc->bic", attention, value_residual)
            query_correction = torch.einsum("bi,bic->bc", query_attention, value_residual)
            state_next = (
                state
                + step * (context_correction - ridge_diag.unsqueeze(-1) * state)
                + momentum * (state - state_prev)
            )
            query_next = (
                query_state
                + step * (query_correction - query_ridge.unsqueeze(-1) * query_state)
                + momentum * (query_state - query_prev)
            )
            state_prev, state = state, state_next
            query_prev, query_state = query_state, query_next
            if return_layers:
                state_layers.append(state)
                pred_layers.append(query_state[:, 0])

        prediction = query_state[:, 0]
        info = {
            "attention": attention,
            "degree": degree,
            "lengthscale": lengthscale,
            "step": step,
            "momentum": momentum,
            "state_channels": state,
            "variance_reduction": query_state[:, 1],
        }
        if return_layers:
            info["state_layers"] = torch.stack(state_layers, dim=1)
            info["pred_layers"] = torch.stack(pred_layers, dim=1)
        return prediction, state[:, :, 0], info


def train_krr_looped(args, device) -> None:
    """Train the kernel metric and stable shared solver coefficients end to end."""
    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "train_krr_looped.csv"
    model = NoMLPKernelAttentionLoop(
        depth=args.depth,
        lam=args.krr_lam,
        iteration=args.loop_solver,
        init_lengthscale=args.kernel_init_lengthscale,
        learn_kernel=bool(args.learn_kernel),
        step_init=args.loop_step_init,
        momentum_init=args.loop_momentum_init,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for training_step in range(1, args.steps + 1):
        batch = sample_gp_krr_batch(
            args.batch_size,
            args.n_context,
            args.x_dim,
            args.lengthscale,
            args.noise_var,
            args.krr_lam,
            device,
        )
        prediction, state, train_info = model(batch.x_ctx, batch.y_ctx, batch.x_q)
        system = batch.K + args.krr_lam * torch.eye(
            args.n_context, device=device, dtype=batch.K.dtype
        )
        variance_alpha = stable_solve(system, batch.kq)
        target_state = torch.stack(
            [
                torch.einsum("bij,bj->bi", batch.K, batch.alpha_exact),
                torch.einsum("bij,bj->bi", batch.K, variance_alpha),
            ],
            dim=-1,
        )
        error = train_info["state_channels"] - target_state
        energy_error = torch.einsum("bic,bij,bjc->bc", error, system, error)
        energy_target = torch.einsum(
            "bic,bij,bjc->bc", target_state, system, target_state
        ).clamp_min(1e-10)
        solver_loss = (energy_error / energy_target).mean()
        prediction_loss = F.mse_loss(prediction, batch.mean_exact)
        variance_exact = torch.einsum("bi,bi->b", batch.kq, variance_alpha)
        variance_loss = F.mse_loss(train_info["variance_reduction"], variance_exact)
        loss = (
            solver_loss
            + args.loop_prediction_weight * prediction_loss
            + args.loop_variance_weight * variance_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if training_step == 1 or training_step % args.log_every == 0:
            with torch.no_grad():
                evaluation = sample_gp_krr_batch(
                    args.eval_batch_size,
                    args.n_context,
                    args.x_dim,
                    args.lengthscale,
                    args.noise_var,
                    args.krr_lam,
                    device,
                )
                eval_prediction, eval_state, info = model(
                    evaluation.x_ctx,
                    evaluation.y_ctx,
                    evaluation.x_q,
                    return_layers=True,
                )
                eval_system = evaluation.K + args.krr_lam * torch.eye(
                    args.n_context, device=device, dtype=evaluation.K.dtype
                )
                eval_variance_alpha = stable_solve(eval_system, evaluation.kq)
                eval_variance_exact = torch.einsum(
                    "bi,bi->b", evaluation.kq, eval_variance_alpha
                )
                richardson_predictions, _ = krr_richardson_iterates(
                    evaluation.K,
                    evaluation.y_ctx,
                    evaluation.kq,
                    args.krr_lam,
                    args.depth,
                    eta=args.row_eta,
                    mode="rowcond",
                )
                row = {
                    "step": training_step,
                    "solver": args.loop_solver,
                    "loss": loss.item(),
                    "solver_loss": solver_loss.item(),
                    "prediction_loss": prediction_loss.item(),
                    "variance_loss": variance_loss.item(),
                    "eval_state_mse": mse(
                        eval_state,
                        torch.einsum("bij,bj->bi", evaluation.K, evaluation.alpha_exact),
                    ),
                    "eval_mean_mse": mse(eval_prediction, evaluation.mean_exact),
                    "eval_variance_reduction_mse": mse(
                        info["variance_reduction"], eval_variance_exact
                    ),
                    "richardson_mean_mse": mse(
                        richardson_predictions[:, -1], evaluation.mean_exact
                    ),
                    "learned_lengthscale": info["lengthscale"].item(),
                    "true_lengthscale": args.lengthscale,
                    "step_size": info["step"].item(),
                    "momentum": info["momentum"].item(),
                }
                append_csv(csv_path, row)
                print(json.dumps(row, sort_keys=True))

    torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / "krr_looped_final.pt")


def run_krr_sweep(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "krr_sweep.csv"
    for depth in parse_grid(args.depth_grid, int):
        for mode in ["rowcond", "unprecond"]:
            rows = []
            for _ in range(args.eval_batches):
                batch = sample_gp_krr_batch(args.eval_batch_size, args.n_context, args.x_dim, args.lengthscale, args.noise_var, args.krr_lam, device)
                preds, _ = krr_richardson_iterates(batch.K, batch.y_ctx, batch.kq, args.krr_lam, depth, eta=args.row_eta, mode=mode)
                final = preds[:, -1]
                rows.append({
                    "mse_to_exact": mse(final, batch.mean_exact),
                    "mse_to_yq": mse(final, batch.y_q),
                    "rel_to_exact": ((final - batch.mean_exact).abs() / batch.mean_exact.abs().clamp_min(1e-8)).mean().item(),
                })
            row = {
                "depth": depth,
                "mode": mode,
                "n_context": args.n_context,
                "x_dim": args.x_dim,
                "lengthscale": args.lengthscale,
                "lam": args.krr_lam,
                "mse_to_exact": float(np.mean([r["mse_to_exact"] for r in rows])),
                "mse_to_yq": float(np.mean([r["mse_to_yq"] for r in rows])),
                "rel_to_exact": float(np.mean([r["rel_to_exact"] for r in rows])),
            }
            append_csv(csv_path, row)
            print(row)


class TinyTransformerKRR(nn.Module):
    def __init__(self, x_dim: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.x_dim = x_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.inp = nn.Linear(x_dim + 2, d_model)  # x, y, is_query
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x_ctx: Tensor, y_ctx: Tensor, x_q: Tensor, return_layers: bool = False):
        B, n, dx = x_ctx.shape
        q_y = torch.zeros(B, 1, device=x_ctx.device, dtype=x_ctx.dtype)
        is_ctx = torch.zeros(B, n, 1, device=x_ctx.device, dtype=x_ctx.dtype)
        is_q = torch.ones(B, 1, 1, device=x_ctx.device, dtype=x_ctx.dtype)
        ctx_tok = torch.cat([x_ctx, y_ctx.unsqueeze(-1), is_ctx], dim=-1)
        q_tok = torch.cat([x_q.unsqueeze(1), q_y.unsqueeze(-1), is_q], dim=-1)
        x = torch.cat([ctx_tok, q_tok], dim=1)
        h = self.inp(x)
        layers = []
        for layer in self.layers:
            h = layer(h)
            if return_layers:
                layers.append(self.norm(h[:, -1]).clone())
        hq = self.norm(h[:, -1])
        pred = self.out(hq).squeeze(-1)
        if return_layers:
            return pred, torch.stack(layers, dim=1)
        return pred, None


def train_krr_probe(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "train_krr.csv"
    model = TinyTransformerKRR(args.x_dim, args.d_model, args.n_layers, args.n_heads, args.d_ff, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for step in range(1, args.steps + 1):
        batch = sample_gp_krr_batch(args.batch_size, args.n_context, args.x_dim, args.lengthscale, args.noise_var, args.krr_lam, device)
        pred, _ = model(batch.x_ctx, batch.y_ctx, batch.x_q)
        # Train to noisy query y; Bayes optimal is posterior mean.
        loss = F.mse_loss(pred, batch.y_q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                ev = sample_gp_krr_batch(args.eval_batch_size, args.n_context, args.x_dim, args.lengthscale, args.noise_var, args.krr_lam, device)
                epred, _ = model(ev.x_ctx, ev.y_ctx, ev.x_q)
                row_preds, _ = krr_richardson_iterates(ev.K, ev.y_ctx, ev.kq, args.krr_lam, args.n_layers, eta=args.row_eta, mode="rowcond")
                un_preds, _ = krr_richardson_iterates(ev.K, ev.y_ctx, ev.kq, args.krr_lam, args.n_layers, mode="unprecond")
                row = {
                    "step": step,
                    "train_loss": loss.item(),
                    "eval_mse_y": mse(epred, ev.y_q),
                    "eval_mse_exact_mean": mse(epred, ev.mean_exact),
                    "rowcond_last_mse_exact": mse(row_preds[:, -1], ev.mean_exact),
                    "unprecond_last_mse_exact": mse(un_preds[:, -1], ev.mean_exact),
                }
                append_csv(csv_path, row)
                print(json.dumps(row, indent=2))

    torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / "krr_model_final.pt")
    probe_krr_model(model, args, device, outdir / "probe_results.csv")


@torch.no_grad()
def probe_krr_model(model: TinyTransformerKRR, args, device, csv_path: Path) -> None:
    # Collect a probe dataset.
    Hs = []
    targets_exact = []
    targets_row = []
    targets_un = []
    yqs = []
    for _ in range(args.probe_batches):
        batch = sample_gp_krr_batch(args.eval_batch_size, args.n_context, args.x_dim, args.lengthscale, args.noise_var, args.krr_lam, device)
        pred, layers = model(batch.x_ctx, batch.y_ctx, batch.x_q, return_layers=True)  # [B,L,D]
        row_preds, _ = krr_richardson_iterates(batch.K, batch.y_ctx, batch.kq, args.krr_lam, args.n_layers, eta=args.row_eta, mode="rowcond")
        un_preds, _ = krr_richardson_iterates(batch.K, batch.y_ctx, batch.kq, args.krr_lam, args.n_layers, mode="unprecond")
        Hs.append(layers.cpu())
        targets_exact.append(batch.mean_exact.cpu())
        targets_row.append(row_preds.cpu())
        targets_un.append(un_preds.cpu())
        yqs.append(batch.y_q.cpu())
    H = torch.cat(Hs, dim=0)  # [N,L,D]
    exact = torch.cat(targets_exact, dim=0)  # [N]
    rowt = torch.cat(targets_row, dim=0)  # [N,L]
    unt = torch.cat(targets_un, dim=0)
    N, L, D = H.shape
    # train/test split for probes
    idx = torch.randperm(N)
    ntr = int(0.7 * N)
    tr, te = idx[:ntr], idx[ntr:]

    for layer in range(L):
        Xtr = torch.cat([H[tr, layer], torch.ones(ntr, 1)], dim=-1)
        Xte = torch.cat([H[te, layer], torch.ones(N - ntr, 1)], dim=-1)
        for target_name, T in [("exact", exact), ("rowcond", rowt[:, layer]), ("unprecond", unt[:, layer])]:
            ytr = T[tr].unsqueeze(-1)
            yte = T[te]
            w = torch.linalg.lstsq(Xtr, ytr).solution
            pred = (Xte @ w).squeeze(-1)
            row = {
                "layer": layer + 1,
                "target": target_name,
                "probe_mse": F.mse_loss(pred, yte).item(),
                "probe_r2": explained_r2(yte, pred),
            }
            append_csv(csv_path, row)
            print("PROBE", row)


# -----------------------------------------------------------------------------
# sweeps
# -----------------------------------------------------------------------------

def run_weak_sweep(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "weak_sweep.csv"
    for K in parse_grid(args.K_grid, int):
        for cond in parse_grid(args.cond_grid, float):
            for cap in args.capacity_grid.split(","):
                P = make_P_coordinate(K, cap, device, torch.float32)
                rank = int(torch.linalg.matrix_rank(P).item())
                for precond in args.precond_grid.split(","):
                    for method in args.methods.split(","):
                        rows = []
                        for _ in range(args.eval_batches):
                            batch = sample_weak_batch(args.eval_batch_size, args.M, K, args.lam, args.noise_std, args.design, cond, device)
                            stats = attention_richardson_solve(
                                batch.G, batch.b, args.lam, args.depth, method, P, args.tau, precond,
                                eta_mult=args.eta_mult, rank=rank,
                            )
                            rows.append(stats)
                        row = {
                            "K": K, "M": args.M, "cond": cond, "capacity": cap, "proj_rank": rank,
                            "method": method, "precond": precond, "depth": args.depth, "tau": args.tau,
                        }
                        for key in rows[0].keys():
                            if key not in row and isinstance(rows[0][key], (int, float, np.floating)):
                                row[key] = float(np.mean([r[key] for r in rows]))
                        append_csv(csv_path, row)
                        print(json.dumps(row, indent=2))


def run_parametric_A_sweep(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "parametric_A_sweep.csv"
    for K in parse_grid(args.K_grid, int):
        for d in parse_grid(args.d_grid, int):
            for precond in args.precond_grid.split(","):
                rows = []
                for _ in range(args.eval_batches):
                    batch = sample_parametric_A_batch(
                        args.eval_batch_size, d, K, args.m_prompt, args.lam, args.noise_std,
                        args.basis_scale, device
                    )
                    zhat, st = richardson_solve(batch.G, batch.b, args.lam, args.depth, precond=precond, rank=min(K, args.rank))
                    Ahat = batch.A0 + torch.einsum("bk,bkij->bij", zhat, batch.A_basis)
                    uhat = stable_solve(Ahat, batch.f_star)
                    rows.append({
                        "z_mse_post": mse(zhat, batch.z_star),
                        "z_mse_true": mse(zhat, batch.z_true),
                        "u_mse": mse(uhat, batch.u_star),
                        "u_rel": relerr(uhat, batch.u_star),
                        **{k: v for k, v in st.items() if isinstance(v, (int, float))},
                    })
                row = {
                    "K": K, "d": d, "m_prompt": args.m_prompt, "M": args.m_prompt * d,
                    "precond": precond, "depth": args.depth,
                }
                for key in rows[0]:
                    row[key] = float(np.mean([r[key] for r in rows]))
                append_csv(csv_path, row)
                print(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        type=str,
        default="smoke",
        choices=[
            "smoke",
            "weak_sweep",
            "train_weak",
            "krr_sweep",
            "train_krr_probe",
            "train_krr_looped",
            "parametric_A_sweep",
        ],
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default=str(resolve_outdir("runs_richardson_transformer_lab")))
    p.add_argument("--seed", type=int, default=0)

    # weak LS
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--design", type=str, default="correlated", choices=["isotropic", "correlated", "spiked"])
    p.add_argument("--cond", type=float, default=10.0)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--eta-mult", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=8)

    p.add_argument("--K-grid", type=str, default="8,16,32")
    p.add_argument("--cond-grid", type=str, default="10,100,1000")
    p.add_argument("--capacity-grid", type=str, default="below,at,above")
    p.add_argument("--precond-grid", type=str, default="scalar_opt,jacobi,lowrank_spectral")
    p.add_argument("--methods", type=str, default="linear,softmax_scalar,signed_relu_scalar,softmax_vector_full,softmax_vector_projected")
    p.add_argument("--depth-grid", type=str, default="1,2,4,8,16,32,64")
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=512)

    # train weak model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--ffn-hidden", type=int, default=256)
    p.add_argument("--value-mode", type=str, default="affine_scalar",
                   choices=["affine_scalar", "mlp_vector", "analytic_vector"])
    p.add_argument("--shared", type=int, default=1)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--loss-true-weight", type=float, default=0.0)
    p.add_argument("--eval-cond-grid", type=str, default="10,100,1000")

    # parametric A
    p.add_argument("--d-grid", type=str, default="16,32,64")
    p.add_argument("--m-prompt", type=int, default=16)
    p.add_argument("--basis-scale", type=float, default=0.2)

    # KRR
    p.add_argument("--n-context", type=int, default=64)
    p.add_argument("--x-dim", type=int, default=1)
    p.add_argument("--lengthscale", type=float, default=0.25)
    p.add_argument("--noise-var", type=float, default=0.02)
    p.add_argument("--krr-lam", type=float, default=0.02)
    p.add_argument("--row-eta", type=float, default=0.8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--loop-solver", choices=["richardson", "heavy_ball"], default="heavy_ball")
    p.add_argument("--kernel-init-lengthscale", type=float, default=0.4)
    p.add_argument("--learn-kernel", type=int, default=1)
    p.add_argument("--loop-step-init", type=float, default=0.8)
    p.add_argument("--loop-momentum-init", type=float, default=0.05)
    p.add_argument("--loop-prediction-weight", type=float, default=1.0)
    p.add_argument("--loop-variance-weight", type=float, default=1.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.outdir = str(resolve_outdir(args.outdir))
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    if args.mode == "smoke":
        outdir = ensure_dir(args.outdir)
        batch = sample_weak_batch(64, 128, 16, args.lam, args.noise_std, "correlated", 10, device)
        for pre in ["scalar_opt", "jacobi", "spectral_full"]:
            z, st = richardson_solve(batch.G, batch.b, args.lam, 16, precond=pre)
            print("weak", pre, st)
        P = make_P_coordinate(16, "at", device, torch.float32)
        for meth in ["linear", "softmax_scalar", "signed_relu_scalar", "softmax_vector_full", "softmax_vector_projected"]:
            st = attention_richardson_solve(batch.G, batch.b, args.lam, 8, meth, P, args.tau, "scalar_opt")
            print("attn", meth, st)
        kb = sample_gp_krr_batch(32, 32, 1, args.lengthscale, args.noise_var, args.krr_lam, device)
        for mode in ["rowcond", "unprecond"]:
            preds, _ = krr_richardson_iterates(kb.K, kb.y_ctx, kb.kq, args.krr_lam, 8, mode=mode)
            print("krr", mode, mse(preds[:, -1], kb.mean_exact))
        print("smoke done")
        return

    if args.mode == "weak_sweep":
        run_weak_sweep(args, device)
    elif args.mode == "train_weak":
        train_weak_model(args, device)
    elif args.mode == "krr_sweep":
        run_krr_sweep(args, device)
    elif args.mode == "train_krr_probe":
        train_krr_probe(args, device)
    elif args.mode == "train_krr_looped":
        train_krr_looped(args, device)
    elif args.mode == "parametric_A_sweep":
        run_parametric_A_sweep(args, device)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
