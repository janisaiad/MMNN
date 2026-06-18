#!/usr/bin/env python3
"""
dual_primal_weak_transformer_lab.py

CUDA/PyTorch lab focused on the precise reconciliation:

    primal weak inverse:
        z_* = (G^T G + lambda I)^(-1) G^T b

    dual weak inverse:
        alpha_* = (G G^T + lambda I)^(-1) b
        z_* = G^T alpha_*

The goal is to test whether a Transformer-like attention mechanism trained on
weak least-squares tasks discovers the dual attention -> primal readout
formulation:

    Q/K over weak-equation tokens builds a dual token kernel,
    the recurrent token state is alpha_i,
    the final readout is z = G^T alpha.

This script separates:

1. exact algebraic identities;
2. hardcoded dual/primal Richardson baselines;
3. trainable dual-primal self-attention models;
4. diagnostics showing what the trained model actually learned.

Important distinction
---------------------

For the exact primal-dual identity, the dual kernel is linear:

    K_lin = G G^T.

But vanilla softmax naturally builds a positive row-normalized kernel such as

    softmax(g_i^T g_j / tau)

or an RBF-like kernel. This is not automatically the same as K_lin when entries
of K_lin are signed. Therefore the code tests:
    - exact linear dual solver;
    - dual Jacobi/diagonal preconditioning;
    - softmax row-conditioned surrogate;
    - signed-ReLU kernel surrogate;
    - trainable attention, with diagnostics against all of them.

Typical commands
----------------

Smoke:
    python dual_primal_weak_transformer_lab.py --mode smoke --device cuda

Hardcoded identity + solver sweep:
    python dual_primal_weak_transformer_lab.py --mode dual_primal_sweep \
      --K-grid 8,16,32 --M-grid 8,16,32,64,128,256 \
      --cond-grid 10,100,1000 --depth-grid 4,8,16,32 \
      --device cuda --outdir runs/dual_primal_sweep

Train dual-primal attention:
    python dual_primal_weak_transformer_lab.py --mode train_dual_primal \
      --K 16 --M 128 --cond 10 --depth 8 --d-model 128 --heads 4 --d-head 32 \
      --steps 20000 --batch-size 256 --device cuda --outdir runs/train_dual_primal

Train on log-uniform condition numbers:
    python dual_primal_weak_transformer_lab.py --mode train_dual_primal \
      --K 16 --M 128 --cond-min 10 --cond-max 1000 --sample-cond-loguniform 1 \
      --depth 8 --steps 30000 --device cuda --outdir runs/train_dual_primal_condmix

Probe a saved model:
    python dual_primal_weak_transformer_lab.py --mode probe_dual_primal \
      --checkpoint runs/train_dual_primal/model_final.pt --device cuda \
      --outdir runs/train_dual_primal_probe
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# utilities
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
    if len(p.parts) >= 2 and p.parts[0] == "data" and p.parts[1] == "transformers":
        return project_root() / p
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


def batch_eye(B: int, n: int, device, dtype) -> Tensor:
    return torch.eye(n, device=device, dtype=dtype).expand(B, n, n)


def stable_solve(A: Tensor, b: Tensor, jitter: float = 1e-8) -> Tensor:
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A + jitter * I, b.unsqueeze(-1)).squeeze(-1)


def mse(a: Tensor, b: Tensor) -> float:
    return (a - b).pow(2).mean().detach().item()


def relerr(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(eps)).mean().detach().item()


def r2_score(y: Tensor, yhat: Tensor) -> float:
    y = y.reshape(-1)
    yhat = yhat.reshape(-1)
    ss_res = (y - yhat).pow(2).sum()
    ss_tot = (y - y.mean()).pow(2).sum().clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).detach().item()


def corr_flat(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    c = (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(eps)
    return c.mean().detach().item()


# -----------------------------------------------------------------------------
# data generation
# -----------------------------------------------------------------------------

@dataclass
class WeakBatch:
    G: Tensor       # [B,M,K]
    b: Tensor       # [B,M]
    z_true: Tensor  # [B,K]
    H: Tensor       # [B,K,K] = G^T G + lambda I
    c: Tensor       # [B,K] = G^T b
    z_star: Tensor  # [B,K]
    Kdual: Tensor   # [B,M,M] = G G^T
    alpha_star: Tensor # [B,M]
    eig_H: Tensor


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
    raise ValueError(design)


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
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(B, K, device, dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    z_star = stable_solve(H, c)
    Kdual = torch.einsum("bik,bjk->bij", G, G)
    alpha_star = stable_solve(Kdual + lam * batch_eye(B, M, device, dtype), b)
    eig_H = torch.linalg.eigvalsh(H)
    return WeakBatch(G, b, z_true, H, c, z_star, Kdual, alpha_star, eig_H)


def sample_cond(args) -> float:
    if int(args.sample_cond_loguniform):
        u = random.random()
        return math.exp(math.log(args.cond_min) * (1 - u) + math.log(args.cond_max) * u)
    return args.cond


# -----------------------------------------------------------------------------
# primal / dual hardcoded solvers
# -----------------------------------------------------------------------------

@torch.no_grad()
def primal_richardson_layers(
    G: Tensor,
    b: Tensor,
    lam: float,
    depth: int,
    precond: str = "scalar_opt",
) -> Tensor:
    B, M, K = G.shape
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(B, K, G.device, G.dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    z = torch.zeros(B, K, device=G.device, dtype=G.dtype)
    layers = []
    eig = torch.linalg.eigvalsh(H)
    eta = 2.0 / (eig[:, 0] + eig[:, -1]).clamp_min(1e-12)
    if precond == "jacobi":
        Dinv = 1.0 / torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-12)
        DH = Dinv.unsqueeze(-1) * H
        rad = torch.linalg.eigvals(DH).abs().real.max(dim=-1).values.clamp_min(1e-12)
        eta_j = 1.0 / rad
    for _ in range(depth):
        grad = c - torch.einsum("bij,bj->bi", H, z)
        if precond == "scalar_opt":
            delta = eta[:, None] * grad
        elif precond == "jacobi":
            delta = eta_j[:, None] * Dinv * grad
        elif precond == "spectral_full":
            delta = stable_solve(H, grad)
        else:
            raise ValueError(precond)
        z = z + delta
        layers.append(z.clone())
    return torch.stack(layers, dim=1)


@torch.no_grad()
def dual_richardson_layers(
    G: Tensor,
    b: Tensor,
    lam: float,
    depth: int,
    mode: str = "unprecond",
    tau: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Return alpha layers [B,L,M] and z layers [B,L,K].

    modes:
      unprecond: exact linear dual Richardson on K=GG^T.
      diag: Jacobi/diagonal preconditioning on K+lambda I.
      rowsum_abs: D^{-1} with D_i=sum_j |K_ij| + lambda.
      softmax_exp: use A=softmax(K/tau) as row-conditioned surrogate.
      signed_relu: use positive/negative normalized pieces to implement signed row mixing.
    """
    B, M, Kdim = G.shape
    Kdual = torch.einsum("bik,bjk->bij", G, G)
    Adual = Kdual + lam * batch_eye(B, M, G.device, G.dtype)
    alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)
    alayers, zlayers = [], []

    eig = torch.linalg.eigvalsh(Adual)
    eta_un = 2.0 / (eig[:, 0] + eig[:, -1]).clamp_min(1e-12)

    if mode == "diag":
        Dinv = 1.0 / torch.diagonal(Adual, dim1=-2, dim2=-1).clamp_min(1e-12)
        DA = Dinv.unsqueeze(-1) * Adual
        rad = torch.linalg.eigvals(DA).abs().real.max(dim=-1).values.clamp_min(1e-12)
        eta_diag = 1.0 / rad

    if mode == "rowsum_abs":
        Dinv_abs = 1.0 / (Kdual.abs().sum(dim=-1) + lam).clamp_min(1e-12)

    if mode == "softmax_exp":
        Att = torch.softmax(Kdual / tau, dim=-1)

    if mode == "signed_relu":
        pos = Kdual.clamp_min(0)
        neg = (-Kdual).clamp_min(0)
        spos = pos.sum(-1).clamp_min(1e-12)
        sneg = neg.sum(-1).clamp_min(1e-12)
        Apos = pos / spos.unsqueeze(-1)
        Aneg = neg / sneg.unsqueeze(-1)

    for _ in range(depth):
        if mode == "unprecond":
            res = b - torch.einsum("bij,bj->bi", Adual, alpha)
            delta = eta_un[:, None] * res
        elif mode == "diag":
            res = b - torch.einsum("bij,bj->bi", Adual, alpha)
            delta = eta_diag[:, None] * Dinv * res
        elif mode == "rowsum_abs":
            res = b - torch.einsum("bij,bj->bi", Adual, alpha)
            delta = Dinv_abs * res
        elif mode == "softmax_exp":
            # surrogate row-conditioned update using softmax(K) not exact linear K
            Kalpha_row = torch.einsum("bij,bj->bi", Att, alpha)
            # b also row scaled heuristically by 1/(sum exp) hidden in softmax; this is not exact.
            delta = b - Kalpha_row - lam * alpha
        elif mode == "signed_relu":
            Kalpha = (torch.einsum("bij,bj->bi", Apos, alpha) * spos
                      - torch.einsum("bij,bj->bi", Aneg, alpha) * sneg)
            res = b - Kalpha - lam * alpha
            delta = res / (spos + sneg + lam).clamp_min(1e-12)
        else:
            raise ValueError(mode)
        alpha = alpha + delta
        alayers.append(alpha.clone())
        zlayers.append(torch.einsum("bmk,bm->bk", G, alpha))
    return torch.stack(alayers, dim=1), torch.stack(zlayers, dim=1)


def exact_identity_metrics(batch: WeakBatch) -> Dict:
    z_dual = torch.einsum("bmk,bm->bk", batch.G, batch.alpha_star)
    return {
        "identity_mse_z_primal_vs_dual": mse(batch.z_star, z_dual),
        "identity_rel_z_primal_vs_dual": relerr(z_dual, batch.z_star),
        "z_star_mse_true": mse(batch.z_star, batch.z_true),
        "cond_H_mean": (batch.eig_H[:, -1] / batch.eig_H[:, 0].clamp_min(1e-12)).mean().item(),
    }


# -----------------------------------------------------------------------------
# trainable dual-primal attention model
# -----------------------------------------------------------------------------

class DualPrimalAttentionModel(nn.Module):
    """Self-attention over weak-equation tokens, alpha state, primal readout.

    Each layer:
        token_i = [g_i, b_i, alpha_i]
        q_i = W_Q token_i, k_i = W_K token_i
        attn_i = softmax_j(q_i k_j)
        value_j = W_V token_j
        o_i = sum_j attn_ij value_j
        delta_alpha_i = local MLP([token_i, o_i])
        alpha_i += damp * delta_alpha_i

    Final:
        z = G^T alpha.

    This is the direct dual-attention / primal-readout architecture.
    """

    def __init__(
        self,
        K: int,
        d_model: int = 128,
        d_head: int = 32,
        n_heads: int = 4,
        depth: int = 8,
        value_dim: int = 32,
        ffn_hidden: int = 256,
        qk_from_g_only: bool = True,
        include_row_norm: bool = False,
    ):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.depth = depth
        self.value_dim = value_dim
        self.qk_from_g_only = qk_from_g_only
        self.include_row_norm = include_row_norm

        base_dim = K + 2  # g, b, alpha
        if include_row_norm:
            base_dim += 1
        self.base_dim = base_dim

        qk_in = K if qk_from_g_only else base_dim
        self.Wq = nn.Linear(qk_in, n_heads * d_head, bias=False)
        self.Wk = nn.Linear(qk_in, n_heads * d_head, bias=False)
        self.Wv = nn.Linear(base_dim, n_heads * value_dim, bias=True)

        self.local_update = nn.Sequential(
            nn.Linear(base_dim + n_heads * value_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, 1),
        )
        self.step_scale = nn.Parameter(torch.tensor(0.1))

    def build_token(self, G: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
        parts = [G, b.unsqueeze(-1), alpha.unsqueeze(-1)]
        if self.include_row_norm:
            parts.append(G.norm(dim=-1, keepdim=True))
        return torch.cat(parts, dim=-1)

    def layer_step(self, G: Tensor, b: Tensor, alpha: Tensor, return_attn: bool = False):
        B, M, K = G.shape
        tok = self.build_token(G, b, alpha)
        qk_src = G if self.qk_from_g_only else tok
        Q = self.Wq(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        Kmat = self.Wk(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(tok).view(B, M, self.n_heads, self.value_dim).transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", Q, Kmat) / math.sqrt(self.d_head)
        Att = torch.softmax(scores, dim=-1)
        O = torch.einsum("bhij,bhjv->bhiv", Att, V).transpose(1, 2).reshape(B, M, self.n_heads * self.value_dim)
        delta = self.local_update(torch.cat([tok, O], dim=-1)).squeeze(-1)
        alpha_new = alpha + self.step_scale * delta
        info = {
            "attn_entropy": (-(Att.clamp_min(1e-12) * Att.clamp_min(1e-12).log()).sum(-1).mean()).detach(),
            "delta_norm": delta.norm(dim=-1).mean().detach(),
        }
        if return_attn:
            info["Att"] = Att.detach()
            info["scores"] = scores.detach()
        return alpha_new, info

    def forward(self, G: Tensor, b: Tensor, return_layers: bool = False, return_attn: bool = False):
        B, M, K = G.shape
        alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)
        alayers, zlayers, infos = [], [], []
        for _ in range(self.depth):
            alpha, info = self.layer_step(G, b, alpha, return_attn=return_attn)
            if return_layers:
                alayers.append(alpha)
                zlayers.append(torch.einsum("bmk,bm->bk", G, alpha))
                infos.append(info)
        z = torch.einsum("bmk,bm->bk", G, alpha)
        out = {}
        if return_layers:
            out["alpha_layers"] = torch.stack(alayers, dim=1)
            out["z_layers"] = torch.stack(zlayers, dim=1)
            out["infos"] = infos
        return z, alpha, out

    def score_matrix(self, G: Tensor, b: Tensor, alpha: Optional[Tensor] = None) -> Tensor:
        if alpha is None:
            alpha = torch.zeros(G.shape[0], G.shape[1], device=G.device, dtype=G.dtype)
        tok = self.build_token(G, b, alpha)
        qk_src = G if self.qk_from_g_only else tok
        B, M, K = G.shape
        Q = self.Wq(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        Km = self.Wk(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", Q, Km) / math.sqrt(self.d_head)
        return scores


# -----------------------------------------------------------------------------
# diagnostics
# -----------------------------------------------------------------------------

@torch.no_grad()
def linear_probe_r2(X: Tensor, Y: Tensor) -> Tuple[float, float]:
    """Fit Y ~= X W with train/test split. X [N,D], Y [N,*]."""
    X = X.reshape(X.shape[0], -1).cpu()
    Y = Y.reshape(Y.shape[0], -1).cpu()
    N = X.shape[0]
    idx = torch.randperm(N)
    ntr = max(2, int(0.7 * N))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr = torch.cat([X[tr], torch.ones(ntr, 1)], dim=-1)
    Xte = torch.cat([X[te], torch.ones(N - ntr, 1)], dim=-1)
    W = torch.linalg.lstsq(Xtr, Y[tr]).solution
    pred = Xte @ W
    return F.mse_loss(pred, Y[te]).item(), r2_score(Y[te], pred)


@torch.no_grad()
def probe_dual_primal_model(
    model: DualPrimalAttentionModel,
    args,
    device,
    csv_path: Path,
    batch: Optional[WeakBatch] = None,
) -> None:
    if batch is None:
        batch = sample_weak_batch(args.probe_batch_size, args.M, args.K, args.lam, args.noise_std, args.design, args.cond, device)
    zhat, alpha, out = model(batch.G, batch.b, return_layers=True, return_attn=True)
    L = out["z_layers"].shape[1]

    # Reference trajectories.
    alpha_un, z_un = dual_richardson_layers(batch.G, batch.b, args.lam, L, mode="unprecond")
    alpha_diag, z_diag = dual_richardson_layers(batch.G, batch.b, args.lam, L, mode="diag")
    alpha_rowsum, z_rowsum = dual_richardson_layers(batch.G, batch.b, args.lam, L, mode="rowsum_abs")
    alpha_soft, z_soft = dual_richardson_layers(batch.G, batch.b, args.lam, L, mode="softmax_exp", tau=args.tau)
    alpha_signed, z_signed = dual_richardson_layers(batch.G, batch.b, args.lam, L, mode="signed_relu")
    z_primal_scalar = primal_richardson_layers(batch.G, batch.b, args.lam, L, precond="scalar_opt")
    z_primal_jac = primal_richardson_layers(batch.G, batch.b, args.lam, L, precond="jacobi")

    # Kernel diagnostics at alpha=0.
    scores = model.score_matrix(batch.G, batch.b).mean(dim=1)  # [B,M,M], avg heads
    Klin = batch.Kdual
    Att = torch.softmax(scores, dim=-1)
    Asoft_target = torch.softmax(Klin / args.tau, dim=-1)
    row = {
        "kind": "summary",
        "final_z_mse_post": mse(zhat, batch.z_star),
        "final_z_mse_true": mse(zhat, batch.z_true),
        "final_z_rel_post": relerr(zhat, batch.z_star),
        "score_corr_linear_kernel": corr_flat(scores, Klin),
        "attn_corr_softmax_linear": corr_flat(Att, Asoft_target),
        "attn_entropy": (-(Att.clamp_min(1e-12) * Att.clamp_min(1e-12).log()).sum(-1).mean()).item(),
    }
    append_csv(csv_path, row)

    refs_z = {
        "z_exact": batch.z_star,
        "z_dual_unprecond": z_un,
        "z_dual_diag": z_diag,
        "z_dual_rowsum_abs": z_rowsum,
        "z_dual_softmax_exp": z_soft,
        "z_dual_signed_relu": z_signed,
        "z_primal_scalar": z_primal_scalar,
        "z_primal_jacobi": z_primal_jac,
    }
    refs_alpha = {
        "alpha_exact": batch.alpha_star,
        "alpha_dual_unprecond": alpha_un,
        "alpha_dual_diag": alpha_diag,
        "alpha_dual_rowsum_abs": alpha_rowsum,
        "alpha_dual_softmax_exp": alpha_soft,
        "alpha_dual_signed_relu": alpha_signed,
    }

    for l in range(L):
        zl = out["z_layers"][:, l]
        al = out["alpha_layers"][:, l]
        base = {"kind": "layer", "layer": l + 1}
        for name, target in refs_z.items():
            if target.ndim == 3:
                t = target[:, l]
            else:
                t = target
            row = dict(base)
            row.update({"target": name, "direct_mse": mse(zl, t), "direct_r2": r2_score(t, zl)})
            append_csv(csv_path, row)
        for name, target in refs_alpha.items():
            if target.ndim == 3:
                t = target[:, l]
            else:
                t = target
            row = dict(base)
            row.update({"target": name, "direct_mse": mse(al, t), "direct_r2": r2_score(t, al)})
            append_csv(csv_path, row)

    # Hidden state here is alpha/z themselves. For a true hidden probe, this architecture
    # would need to store token hidden states. Still useful: which classical iterate is alpha/z closest to?
    print("PROBE SUMMARY", json.dumps(row, indent=2))


# -----------------------------------------------------------------------------
# training / sweeps
# -----------------------------------------------------------------------------

def run_dual_primal_sweep(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "dual_primal_sweep.csv"
    for K in parse_grid(args.K_grid, int):
        for M in parse_grid(args.M_grid, int):
            for cond in parse_grid(args.cond_grid, float):
                for depth in parse_grid(args.depth_grid, int):
                    rows = []
                    for _ in range(args.eval_batches):
                        batch = sample_weak_batch(args.eval_batch_size, M, K, args.lam, args.noise_std, args.design, cond, device)
                        ident = exact_identity_metrics(batch)
                        refs = {}
                        for mode in ["unprecond", "diag", "rowsum_abs", "softmax_exp", "signed_relu"]:
                            _, zlayers = dual_richardson_layers(batch.G, batch.b, args.lam, depth, mode=mode, tau=args.tau)
                            refs[f"dual_{mode}_mse"] = mse(zlayers[:, -1], batch.z_star)
                        for pre in ["scalar_opt", "jacobi"]:
                            zlayers = primal_richardson_layers(batch.G, batch.b, args.lam, depth, precond=pre)
                            refs[f"primal_{pre}_mse"] = mse(zlayers[:, -1], batch.z_star)
                        rows.append({**ident, **refs})
                    row = {"K": K, "M": M, "M_over_K": M / K, "cond": cond, "depth": depth}
                    for key in rows[0]:
                        row[key] = float(np.mean([r[key] for r in rows]))
                    append_csv(csv_path, row)
                    print(json.dumps(row, indent=2))


def train_dual_primal(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    csv_train = outdir / "train_metrics.csv"
    model = DualPrimalAttentionModel(
        K=args.K,
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.heads,
        depth=args.depth,
        value_dim=args.value_dim,
        ffn_hidden=args.ffn_hidden,
        qk_from_g_only=bool(args.qk_from_g_only),
        include_row_norm=bool(args.include_row_norm),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for step in range(1, args.steps + 1):
        cond = sample_cond(args)
        batch = sample_weak_batch(args.batch_size, args.M, args.K, args.lam, args.noise_std, args.design, cond, device)
        zhat, alpha, out = model(batch.G, batch.b, return_layers=True)
        loss = F.mse_loss(zhat, batch.z_star)
        if args.loss_alpha_weight > 0:
            loss = loss + args.loss_alpha_weight * F.mse_loss(alpha, batch.alpha_star)
        if args.loss_layers_weight > 0:
            # encourage every layer to be useful
            loss = loss + args.loss_layers_weight * F.mse_loss(out["z_layers"], batch.z_star[:, None, :].expand_as(out["z_layers"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                eval_cond = sample_cond(args)
                ev = sample_weak_batch(args.eval_batch_size, args.M, args.K, args.lam, args.noise_std, args.design, eval_cond, device)
                ez, ea, eout = model(ev.G, ev.b, return_layers=True)
                _, z_diag = dual_richardson_layers(ev.G, ev.b, args.lam, args.depth, mode="diag")
                z_pr_jac = primal_richardson_layers(ev.G, ev.b, args.lam, args.depth, precond="jacobi")
                scores = model.score_matrix(ev.G, ev.b).mean(dim=1)
                row = {
                    "step": step,
                    "train_cond": cond,
                    "eval_cond": eval_cond,
                    "loss": loss.item(),
                    "eval_mse_post": mse(ez, ev.z_star),
                    "eval_mse_true": mse(ez, ev.z_true),
                    "eval_rel_post": relerr(ez, ev.z_star),
                    "dual_diag_same_depth_mse": mse(z_diag[:, -1], ev.z_star),
                    "primal_jac_same_depth_mse": mse(z_pr_jac[:, -1], ev.z_star),
                    "layer0_mse": mse(eout["z_layers"][:, 0], ev.z_star),
                    "layerlast_mse": mse(eout["z_layers"][:, -1], ev.z_star),
                    "score_corr_linear_kernel": corr_flat(scores, ev.Kdual),
                    "attn_entropy_last": float(eout["infos"][-1]["attn_entropy"]),
                }
                append_csv(csv_train, row)
                print(json.dumps(row, indent=2))

        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / f"model_step{step}.pt")

    torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / "model_final.pt")
    probe_dual_primal_model(model, args, device, outdir / "probe_dual_primal.csv")


def load_model_from_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    saved_args = argparse.Namespace(**ckpt["args"])
    model = DualPrimalAttentionModel(
        K=saved_args.K,
        d_model=saved_args.d_model,
        d_head=saved_args.d_head,
        n_heads=saved_args.heads,
        depth=saved_args.depth,
        value_dim=saved_args.value_dim,
        ffn_hidden=saved_args.ffn_hidden,
        qk_from_g_only=bool(saved_args.qk_from_g_only),
        include_row_norm=bool(saved_args.include_row_norm),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, saved_args


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="smoke",
                   choices=["smoke", "dual_primal_sweep", "train_dual_primal", "probe_dual_primal"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default=str(resolve_outdir("runs_dual_primal_lab")))
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--seed", type=int, default=0)

    # task
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--design", type=str, default="correlated", choices=["isotropic", "correlated", "spiked"])
    p.add_argument("--cond", type=float, default=10.0)
    p.add_argument("--cond-min", type=float, default=10.0)
    p.add_argument("--cond-max", type=float, default=1000.0)
    p.add_argument("--sample-cond-loguniform", type=int, default=0)

    # model
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--value-dim", type=int, default=32)
    p.add_argument("--ffn-hidden", type=int, default=256)
    p.add_argument("--qk-from-g-only", type=int, default=1)
    p.add_argument("--include-row-norm", type=int, default=1)

    # train
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--loss-alpha-weight", type=float, default=0.0)
    p.add_argument("--loss-layers-weight", type=float, default=0.0)

    # sweep
    p.add_argument("--K-grid", type=str, default="8,16,32")
    p.add_argument("--M-grid", type=str, default="8,16,32,64,128,256")
    p.add_argument("--cond-grid", type=str, default="10,100,1000")
    p.add_argument("--depth-grid", type=str, default="4,8,16,32")
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--tau", type=float, default=5.0)
    return p


def main():
    args = build_parser().parse_args()
    args.outdir = str(resolve_outdir(args.outdir))
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    outdir = ensure_dir(args.outdir)
    print("device:", device)

    if args.mode == "smoke":
        batch = sample_weak_batch(64, args.M, args.K, args.lam, args.noise_std, args.design, args.cond, device)
        print("identity", exact_identity_metrics(batch))
        for mode in ["unprecond", "diag", "rowsum_abs", "softmax_exp", "signed_relu"]:
            _, zlayers = dual_richardson_layers(batch.G, batch.b, args.lam, args.depth, mode=mode, tau=args.tau)
            print("dual", mode, "mse", mse(zlayers[:, -1], batch.z_star))
        for pre in ["scalar_opt", "jacobi"]:
            zlayers = primal_richardson_layers(batch.G, batch.b, args.lam, args.depth, precond=pre)
            print("primal", pre, "mse", mse(zlayers[:, -1], batch.z_star))
        model = DualPrimalAttentionModel(args.K, args.d_model, args.d_head, args.heads, args.depth,
                                         args.value_dim, args.ffn_hidden,
                                         bool(args.qk_from_g_only), bool(args.include_row_norm)).to(device)
        z, alpha, out = model(batch.G, batch.b, return_layers=True, return_attn=True)
        print("untrained model z mse", mse(z, batch.z_star))
        print("smoke done")
        return

    if args.mode == "dual_primal_sweep":
        run_dual_primal_sweep(args, device)
    elif args.mode == "train_dual_primal":
        train_dual_primal(args, device)
    elif args.mode == "probe_dual_primal":
        if not args.checkpoint:
            raise ValueError("--checkpoint required")
        model, saved_args = load_model_from_checkpoint(Path(args.checkpoint), device)
        # Fill missing runtime args for probe from current args when needed.
        for key, val in vars(saved_args).items():
            if not hasattr(args, key):
                setattr(args, key, val)
        probe_dual_primal_model(model, args, device, ensure_dir(args.outdir) / "probe_dual_primal.csv")
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
