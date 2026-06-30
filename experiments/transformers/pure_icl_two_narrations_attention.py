#!/usr/bin/env python3
"""
pure_icl_parametric_operator_richardson_attention.py

Goal
----
Verify the exact pure ICL parametric-operator architecture:

    prompt:  (f_i, u_i)_{i=1..m}
    query:   f_*
    target:  u_* = A(z)^{-1} f_*

with

    A(z) = A0 + sum_{k=1}^K z_k A_k.

This file is intentionally structured in two consecutive stages:

1. HARDCODED CHECK
   Use the true operator dictionary A0,A_k to construct weak equations

       G z ~= b

   from the prompt. Then solve z and predict u_*.
   This verifies that the parametric weak-form architecture is mathematically correct.

2. TRAINING
   Train a structured encoder-decoder Transformer-like model end-to-end.

   Encoder:
       learns A0,A_k and builds weak equation tokens (g_a,b_a)
       from prompt pairs (f_i,u_i) and probes v_r.

   Decoder:
       either exact small primal ridge solve, or
       explicit dual Richardson attention with NO MLP update, NO learned eta,
       NO layer loss, NO forced loss toward GG^T.

   Forward head:
       reconstructs A(z_hat) and predicts u_* by a differentiable ridge solve.

The central decoder requested here is:

    alpha^0 = 0

    token a = (g_a, b_a, alpha_a, ||g_a||)

    Q_a = W_Q g_a
    K_b = W_K g_b
    V_b = alpha_b

    P_ab = softmax_b(Q_a^T K_b / sqrt(d_head))
    o_a = sum_b P_ab alpha_b

    eta_a = 1 / (||g_a||^2 + lambda)

    alpha_a^{l+1}
        = alpha_a^l + eta_a [ b_a - o_a - lambda alpha_a^l ]

    z_hat = G^T alpha^L

No MLP is used in the Richardson update. The only learned parts of the dual
attention solver are W_Q and W_K. The value is the alpha channel.

This lets us test whether pure attention with the learned encoder can discover
the useful row-conditioned geometry by end-to-end ICL training.

Typical commands
----------------

Smoke:
    python pure_icl_parametric_operator_richardson_attention.py \
      --mode smoke --device cuda

Hardcoded sweep:
    python pure_icl_parametric_operator_richardson_attention.py \
      --mode hardcoded \
      --d-grid 16,32,64 --K-grid 4,8,16 --m-grid 4,8,16,32 \
      --device cuda --outdir runs_pure_icl_rich_hardcoded

Train exact primal solver:
    python pure_icl_parametric_operator_richardson_attention.py \
      --mode train --solver exact \
      --d 32 --K 8 --m 16 --R 32 \
      --steps 30000 --batch-size 256 --device cuda \
      --outdir runs_pure_icl_exact

Train explicit no-MLP dual attention Richardson:
    python pure_icl_parametric_operator_richardson_attention.py \
      --mode train --solver dual_attention_richardson \
      --d 32 --K 8 --m 16 --R 32 --z-depth 16 \
      --heads 1 --d-head 64 \
      --steps 50000 --batch-size 256 --device cuda \
      --outdir runs_pure_icl_dual_attention_richardson

Train attention decoder from near-true dictionary for solver isolation:
    python pure_icl_parametric_operator_richardson_attention.py \
      --mode train --solver dual_attention_richardson \
      --init true_noisy --learn-dictionary 0 \
      --d 32 --K 8 --m 16 --R 32 --z-depth 16 \
      --steps 30000 --device cuda \
      --outdir runs_pure_icl_dual_solver_only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def parse_grid(s: str, typ=int) -> List:
    return [typ(x) for x in str(s).split(",") if str(x).strip()]


def batch_eye(B: int, n: int, device, dtype) -> Tensor:
    return torch.eye(n, device=device, dtype=dtype).expand(B, n, n)


def stable_solve(A: Tensor, b: Tensor, jitter: float = 1e-7) -> Tensor:
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A + jitter * I, b.unsqueeze(-1)).squeeze(-1)


def ridge_forward_solve(A: Tensor, f: Tensor, gamma: float, jitter: float = 1e-7) -> Tensor:
    """Solve argmin_u ||A u - f||^2 + gamma ||u||^2."""
    B, d, _ = A.shape
    AtA = torch.einsum("bij,bik->bjk", A, A)
    Atf = torch.einsum("bij,bi->bj", A, f)
    I = torch.eye(d, device=A.device, dtype=A.dtype).expand(B, d, d)
    return stable_solve(AtA + gamma * I, Atf, jitter=jitter)


def mse(a: Tensor, b: Tensor) -> float:
    return (a - b).pow(2).mean().detach().item()


def relerr(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(eps)).mean().detach().item()


def frob_rel(A: Tensor, B: Tensor, eps: float = 1e-12) -> float:
    return ((A - B).flatten(1).norm(dim=-1) / B.flatten(1).norm(dim=-1).clamp_min(eps)).mean().detach().item()


def richardson_safe_scale(DA: Tensor) -> Tensor:
    """Upper bound on spectral radius via Gershgorin (cheap vs full eigvals)."""
    row_sum = DA.abs().sum(dim=-1).max(dim=-1).values.clamp_min(1e-12)
    return 1.0 / row_sum


def reset_csv(path: Path) -> None:
    if path.exists():
        path.unlink()


def corr_flat(A: Tensor, B: Tensor, eps: float = 1e-12) -> float:
    A = A.flatten(1)
    B = B.flatten(1)
    A = A - A.mean(dim=1, keepdim=True)
    B = B - B.mean(dim=1, keepdim=True)
    return ((A * B).sum(dim=1) / (A.norm(dim=1) * B.norm(dim=1)).clamp_min(eps)).mean().detach().item()


def sample_log_uniform(lo: float, hi: float) -> float:
    u = random.random()
    return math.exp(math.log(lo) * (1 - u) + math.log(hi) * u)


# -----------------------------------------------------------------------------
# parametric operator task
# -----------------------------------------------------------------------------

@dataclass
class OperatorFamily:
    A0: Tensor
    Abasis: Tensor


@dataclass
class ICLBatch:
    z: Tensor
    A: Tensor
    f_prompt: Tensor
    u_prompt: Tensor
    f_star: Tensor
    u_star: Tensor


def make_true_family(d: int, K: int, basis_scale: float, A0_scale: float, device, dtype=torch.float32) -> OperatorFamily:
    I = torch.eye(d, device=device, dtype=dtype)
    A0 = A0_scale * I
    R = torch.randn(K, d, d, device=device, dtype=dtype)
    Abasis = 0.5 * (R + R.transpose(-1, -2))
    norms = Abasis.flatten(1).norm(dim=-1).clamp_min(1e-12)
    Abasis = Abasis / norms[:, None, None] * (basis_scale * math.sqrt(d))
    return OperatorFamily(A0=A0, Abasis=Abasis)


def assemble_A(A0: Tensor, Abasis: Tensor, z: Tensor, spd_shift: float = 0.25) -> Tensor:
    A = A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z, Abasis)
    eig_min = torch.linalg.eigvalsh(A)[:, 0]
    shift = (spd_shift - eig_min).clamp_min(0.0)
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return A + shift[:, None, None] * I


def sample_icl_batch(
    family: OperatorFamily,
    B: int,
    m: int,
    z_scale: float,
    f_std: float,
    noise_std: float,
    device,
) -> ICLBatch:
    K, d, _ = family.Abasis.shape
    z = torch.randn(B, K, device=device, dtype=family.A0.dtype) * z_scale
    A = assemble_A(family.A0, family.Abasis, z)
    f_prompt = torch.randn(B, m, d, device=device, dtype=family.A0.dtype) * f_std
    u_prompt = torch.linalg.solve(A, f_prompt.transpose(1, 2)).transpose(1, 2)
    if noise_std > 0:
        u_prompt = u_prompt + noise_std * torch.randn_like(u_prompt)
    f_star = torch.randn(B, d, device=device, dtype=family.A0.dtype) * f_std
    u_star = stable_solve(A, f_star)
    return ICLBatch(z=z, A=A, f_prompt=f_prompt, u_prompt=u_prompt, f_star=f_star, u_star=u_star)


# -----------------------------------------------------------------------------
# weak encoder
# -----------------------------------------------------------------------------

def coordinate_probes(d: int, R: int, device, dtype) -> Tensor:
    if R <= d:
        return torch.eye(d, device=device, dtype=dtype)[:R]
    reps = math.ceil(R / d)
    return torch.eye(d, device=device, dtype=dtype).repeat(reps, 1)[:R]


def build_weak_system(
    A0: Tensor,
    Abasis: Tensor,
    probes: Tensor,
    f_prompt: Tensor,
    u_prompt: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Build weak equation system G z ~= b.

    A(z) = A0 + sum_k z_k A_k
    A(z) u_i = f_i

    For probe v_r:
        g_{i,r,k} = <A_k u_i, v_r>
        b_{i,r}   = <f_i - A0 u_i, v_r>
    """
    B, m, d = u_prompt.shape
    K = Abasis.shape[0]
    R = probes.shape[0]

    Ak_u = torch.einsum("kde,bme->bmkd", Abasis, u_prompt)       # [B,m,K,d]
    G = torch.einsum("rd,bmkd->bmrk", probes, Ak_u).reshape(B, m * R, K)

    A0u = torch.einsum("de,bme->bmd", A0, u_prompt)
    rhs = f_prompt - A0u
    b = torch.einsum("rd,bmd->bmr", probes, rhs).reshape(B, m * R)
    return G, b


# -----------------------------------------------------------------------------
# z solvers
# -----------------------------------------------------------------------------

def solve_z_exact(G: Tensor, b: Tensor, lam: float) -> Tensor:
    B, M, K = G.shape
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(B, K, G.device, G.dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    return stable_solve(H, c)


def solve_z_primal_jacobi(G: Tensor, b: Tensor, lam: float, depth: int) -> Tensor:
    B, M, K = G.shape
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(B, K, G.device, G.dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    z = torch.zeros(B, K, device=G.device, dtype=G.dtype)

    diag = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-12)
    Dinv = 1.0 / diag
    DH = Dinv.unsqueeze(-1) * H
    rad = torch.linalg.eigvals(DH).abs().real.max(dim=-1).values.clamp_min(1e-12)
    eta = 1.0 / rad

    for _ in range(depth):
        grad = c - torch.einsum("bij,bj->bi", H, z)
        z = z + eta[:, None] * Dinv * grad
    return z


def solve_z_dual_diag(G: Tensor, b: Tensor, lam: float, depth: int) -> Tuple[Tensor, Tensor]:
    """Hardcoded dual Jacobi/Richardson baseline.

    alpha^{l+1}_i = alpha_i^l + eta_i [b_i - (K alpha)_i - lam alpha_i]
    with K=GG^T and eta from diagonal preconditioning plus safe global scale.
    """
    B, M, Kdim = G.shape
    Kdual = torch.einsum("bik,bjk->bij", G, G)
    alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)

    Dinv = 1.0 / (torch.diagonal(Kdual, dim1=-2, dim2=-1) + lam).clamp_min(1e-12)
    Adual = Kdual + lam * batch_eye(B, M, G.device, G.dtype)
    DA = Dinv.unsqueeze(-1) * Adual
    rad = torch.linalg.eigvals(DA).abs().real.max(dim=-1).values.clamp_min(1e-12)
    safe_scale = 1.0 / rad

    for _ in range(depth):
        Kalpha = torch.einsum("bij,bj->bi", Kdual, alpha)
        res = b - Kalpha - lam * alpha
        alpha = alpha + safe_scale[:, None] * Dinv * res

    z = torch.einsum("bmk,bm->bk", G, alpha)
    return z, alpha



class DualAttentionBase(nn.Module):
    """Shared Q/K machinery for the two narratives."""

    def __init__(
        self,
        K: int,
        depth: int,
        lam: float,
        n_heads: int = 1,
        d_head: int = 64,
        qk_from: str = "g",
        use_safe_scale: bool = True,
        init_linear_kernel: bool = True,
    ):
        super().__init__()
        self.K = K
        self.depth = depth
        self.lam = lam
        self.n_heads = n_heads
        self.d_head = d_head
        self.qk_from = qk_from
        self.use_safe_scale = use_safe_scale

        token_dim = K + 3  # g, b, alpha, ||g||
        qk_dim = K if qk_from == "g" else token_dim
        self.Wq = nn.Linear(qk_dim, n_heads * d_head, bias=False)
        self.Wk = nn.Linear(qk_dim, n_heads * d_head, bias=False)
        if init_linear_kernel:
            self._init_near_linear_kernel()

    def _init_near_linear_kernel(self) -> None:
        """Initialize scores near g_i^T g_j. Training may move away."""
        with torch.no_grad():
            self.Wq.weight.zero_()
            self.Wk.weight.zero_()
            in_dim = self.K if self.qk_from == "g" else self.K + 3
            dim = min(in_dim, self.d_head)
            for h in range(self.n_heads):
                off = h * self.d_head
                for i in range(dim):
                    self.Wq.weight[off + i, i] = 1.0
                    self.Wk.weight[off + i, i] = 1.0

    def make_token(self, G: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
        rownorm = G.norm(dim=-1, keepdim=True)
        return torch.cat([G, b.unsqueeze(-1), alpha.unsqueeze(-1), rownorm], dim=-1)

    def attention(self, G: Tensor, b: Tensor, alpha: Tensor) -> Tuple[Tensor, Tensor]:
        B, M, K = G.shape
        token = self.make_token(G, b, alpha)
        qk_src = G if self.qk_from == "g" else token
        Q = self.Wq(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        Kmat = self.Wk(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.einsum("bhid,bhjd->bhij", Q, Kmat) / math.sqrt(self.d_head)
        Att = torch.softmax(scores, dim=-1)
        return Att, scores

    @staticmethod
    def _corr_vec(a: Tensor, b: Tensor) -> Tensor:
        a = a - a.mean(dim=-1, keepdim=True)
        b = b - b.mean(dim=-1, keepdim=True)
        return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)

    def common_diag(self, G: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, M, K = G.shape
        Kdual = torch.einsum("bik,bjk->bij", G, G)
        Dinv = 1.0 / (torch.diagonal(Kdual, dim1=-2, dim2=-1) + self.lam).clamp_min(1e-12)
        if self.use_safe_scale:
            Adual = Kdual + self.lam * batch_eye(B, M, G.device, G.dtype)
            DA = Dinv.unsqueeze(-1) * Adual
            safe_scale = richardson_safe_scale(DA)
        else:
            safe_scale = torch.ones(B, device=G.device, dtype=G.dtype)
        return Kdual, Dinv, safe_scale, batch_eye(B, M, G.device, G.dtype)

    def score_diagnostics(self, scores: Tensor, Att: Tensor, Kdual: Tensor) -> Tuple[Tensor, Tensor]:
        ent = -(Att.clamp_min(1e-12) * Att.clamp_min(1e-12).log()).sum(-1).mean()
        smean = scores.mean(dim=1)
        score_corr = torch.tensor(corr_flat(smean.detach(), Kdual.detach()), device=Kdual.device)
        return ent.detach(), score_corr.detach()


class AttentionAsKernelRichardson(DualAttentionBase):
    """Narrative A: attention is the dual operator.

    This is the tested formula:
        o = P_theta alpha
        alpha <- alpha + s D^{-1}[b - o - lambda alpha]

    This only behaves like true dual Richardson if P_theta alpha approximates
    GG^T alpha. Therefore corr(QK^T, GG^T) and corr(o, GG^T alpha) are causal
    diagnostics for this narrative.
    """

    def forward(self, G: Tensor, b: Tensor, return_layers: bool = False) -> Tuple[Tensor, Tensor, Dict]:
        B, M, K = G.shape
        alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)
        Kdual, Dinv, safe_scale, _ = self.common_diag(G)

        alayers, zlayers = [], []
        entropies, score_corrs, o_corrs, resid_norms = [], [], [], []

        for _ in range(self.depth):
            Att, scores = self.attention(G, b, alpha)
            o_heads = torch.einsum("bhij,bj->bhi", Att, alpha)
            o = o_heads.mean(dim=1)

            r_kernel = b - o - self.lam * alpha
            alpha = alpha + safe_scale[:, None] * Dinv * r_kernel
            z = torch.einsum("bmk,bm->bk", G, alpha)

            if return_layers:
                alayers.append(alpha)
                zlayers.append(z)
                ent, sc = self.score_diagnostics(scores, Att, Kdual)
                entropies.append(ent); score_corrs.append(sc)
                Kalpha = torch.einsum("bij,bj->bi", Kdual, alpha.detach())
                o_corrs.append(torch.tensor(corr_flat(o.detach(), Kalpha.detach()), device=G.device))
                r_exact = b - torch.einsum("bmk,bk->bm", G, z) - self.lam * alpha
                resid_norms.append(r_exact.norm(dim=-1).mean().detach())

        z = torch.einsum("bmk,bm->bk", G, alpha)
        info = {}
        if return_layers:
            info = {
                "alpha_layers": torch.stack(alayers, dim=1),
                "z_layers": torch.stack(zlayers, dim=1),
                "attn_entropy": torch.stack(entropies),
                "score_corr_linear_kernel": torch.stack(score_corrs),
                "o_corr_Kalpha": torch.stack(o_corrs),
                "dual_resid_norm": torch.stack(resid_norms),
            }
        return z, alpha, info


class AttentionAsPreconditionerRichardson(DualAttentionBase):
    """Narrative B: attention preconditions the exact residual.

    Compute exact residual by primal readout:
        z = G^T alpha
        r = b - G z - lambda alpha
          = b - (GG^T + lambda I) alpha

    Jacobi step:
        e = D^{-1} r

    Attention preconditioner:
        delta = [(1-rho) I + rho P_theta] e
        alpha <- alpha + s delta

    Here P_theta does not need to approximate GG^T. It should improve the
    Jacobi correction. The causal diagnostics are delta_z alignment with
    z_star-z and residual decay, not corr(QK^T, GG^T).
    """

    def __init__(self, *args, rho: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = float(rho)

    def forward(self, G: Tensor, b: Tensor, return_layers: bool = False) -> Tuple[Tensor, Tensor, Dict]:
        B, M, K = G.shape
        alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)
        Kdual, Dinv, safe_scale, _ = self.common_diag(G)
        z_star = solve_z_exact(G, b, self.lam)  # diagnostic only

        alayers, zlayers = [], []
        entropies, score_corrs = [], []
        pe_corrs, resid_norms, dz_corrs, jac_corrs = [], [], [], []

        for _ in range(self.depth):
            z = torch.einsum("bmk,bm->bk", G, alpha)
            Gz = torch.einsum("bmk,bk->bm", G, z)
            r = b - Gz - self.lam * alpha
            e = Dinv * r

            Att, scores = self.attention(G, b, alpha)
            Pe_heads = torch.einsum("bhij,bj->bhi", Att, e)
            Pe = Pe_heads.mean(dim=1)
            delta = (1.0 - self.rho) * e + self.rho * Pe

            alpha = alpha + safe_scale[:, None] * delta
            z_new = torch.einsum("bmk,bm->bk", G, alpha)

            if return_layers:
                alayers.append(alpha); zlayers.append(z_new)
                ent, sc = self.score_diagnostics(scores, Att, Kdual)
                entropies.append(ent); score_corrs.append(sc)
                pe_corrs.append(torch.tensor(corr_flat(Pe.detach(), e.detach()), device=G.device))
                r_new = b - torch.einsum("bmk,bk->bm", G, z_new) - self.lam * alpha
                resid_norms.append(r_new.norm(dim=-1).mean().detach())
                dz = torch.einsum("bmk,bm->bk", G, delta)
                jac_dz = torch.einsum("bmk,bm->bk", G, e)
                target = z_star - z
                dz_corrs.append(torch.tensor(corr_flat(dz.detach(), target.detach()), device=G.device))
                jac_corrs.append(torch.tensor(corr_flat(jac_dz.detach(), target.detach()), device=G.device))

        z = torch.einsum("bmk,bm->bk", G, alpha)
        info = {}
        if return_layers:
            info = {
                "alpha_layers": torch.stack(alayers, dim=1),
                "z_layers": torch.stack(zlayers, dim=1),
                "attn_entropy": torch.stack(entropies),
                "score_corr_linear_kernel": torch.stack(score_corrs),
                "Pe_corr_jacobi_e": torch.stack(pe_corrs),
                "dual_resid_norm": torch.stack(resid_norms),
                "delta_z_corr_exact": torch.stack(dz_corrs),
                "jacobi_z_corr_exact": torch.stack(jac_corrs),
            }
        return z, alpha, info

# -----------------------------------------------------------------------------
# full model
# -----------------------------------------------------------------------------

class ParametricOperatorICL(nn.Module):
    """Full pure ICL model.

    Encoder:
        learned A0, A_k and probes produce weak equations G,b.

    Decoder options:
        exact                 : exact primal ridge solve
        primal_jacobi         : hardcoded primal Richardson/Jacobi
        dual_diag             : hardcoded dual Richardson/Jacobi
        dual_attention_richardson : learned Q/K attention + explicit Richardson alpha update

    Forward head:
        reconstruct A(z_hat) and solve for u_*.
    """

    def __init__(
        self,
        d: int,
        K: int,
        R: int,
        lam_z: float,
        gamma_u: float,
        solver: str,
        z_depth: int,
        learn_dictionary: bool,
        learn_probes: bool,
        true_family: Optional[OperatorFamily],
        init: str,
        init_noise: float,
        heads: int,
        d_head: int,
        qk_from: str,
        use_safe_scale: bool,
        attn_rho: float,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.R = R
        self.lam_z = lam_z
        self.gamma_u = gamma_u
        self.solver = solver
        self.z_depth = z_depth

        if true_family is not None and init in ["true", "true_noisy"]:
            A0_init = true_family.A0.detach().clone()
            Ab_init = true_family.Abasis.detach().clone()
            if init == "true_noisy":
                A0_init = A0_init + init_noise * torch.randn_like(A0_init)
                Ab_init = Ab_init + init_noise * torch.randn_like(Ab_init)
        else:
            A0_init = 2.0 * torch.eye(d)
            Rnd = torch.randn(K, d, d)
            Ab_init = 0.5 * (Rnd + Rnd.transpose(-1, -2)) * (0.1 / math.sqrt(d))

        self.A0 = nn.Parameter(A0_init, requires_grad=bool(learn_dictionary))
        self.Abasis = nn.Parameter(Ab_init, requires_grad=bool(learn_dictionary))

        probes_init = coordinate_probes(d, R, A0_init.device, A0_init.dtype)
        self.probes = nn.Parameter(probes_init, requires_grad=bool(learn_probes))

        if solver in ["dual_attention_richardson", "dual_attention_kernel"]:
            # alias: dual_attention_richardson = narrative A / attention-as-kernel
            self.dual_attention = AttentionAsKernelRichardson(
                K=K,
                depth=z_depth,
                lam=lam_z,
                n_heads=heads,
                d_head=d_head,
                qk_from=qk_from,
                use_safe_scale=bool(use_safe_scale),
            )
        elif solver == "dual_attention_precond":
            self.dual_attention = AttentionAsPreconditionerRichardson(
                K=K,
                depth=z_depth,
                lam=lam_z,
                n_heads=heads,
                d_head=d_head,
                qk_from=qk_from,
                use_safe_scale=bool(use_safe_scale),
                rho=attn_rho,
            )
        else:
            self.dual_attention = None

    def weak_system(self, f_prompt: Tensor, u_prompt: Tensor) -> Tuple[Tensor, Tensor]:
        return build_weak_system(self.A0, self.Abasis, self.probes, f_prompt, u_prompt)

    def solve_z(self, G: Tensor, b: Tensor, return_layers: bool = False) -> Tuple[Tensor, Dict]:
        info = {}
        if self.solver == "exact":
            z = solve_z_exact(G, b, self.lam_z)
        elif self.solver == "primal_jacobi":
            z = solve_z_primal_jacobi(G, b, self.lam_z, self.z_depth)
        elif self.solver == "dual_diag":
            z, alpha = solve_z_dual_diag(G, b, self.lam_z, self.z_depth)
            info["alpha"] = alpha
        elif self.solver in ["dual_attention_richardson", "dual_attention_kernel", "dual_attention_precond"]:
            z, alpha, dinfo = self.dual_attention(G, b, return_layers=return_layers)
            info.update(dinfo)
            info["alpha"] = alpha
        else:
            raise ValueError(self.solver)
        return z, info

    def forward(self, f_prompt: Tensor, u_prompt: Tensor, f_star: Tensor, return_info: bool = False):
        G, b = self.weak_system(f_prompt, u_prompt)
        z_hat, info = self.solve_z(G, b, return_layers=return_info)
        A_hat = self.A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z_hat, self.Abasis)
        u_hat = ridge_forward_solve(A_hat, f_star, self.gamma_u)
        if return_info:
            info.update({"G": G, "b": b, "z_hat": z_hat, "A_hat": A_hat})
            return u_hat, info
        return u_hat, {}


# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def oracle_hardcoded_eval(
    family: OperatorFamily,
    args,
    device,
    csv_path: Optional[Path] = None,
    tag: str = "hardcoded",
) -> Dict:
    batch = sample_icl_batch(family, args.eval_batch_size, args.m, args.z_scale, args.f_std, args.noise_std, device)
    probes = coordinate_probes(args.d, args.R, device, family.A0.dtype)
    G, b = build_weak_system(family.A0, family.Abasis, probes, batch.f_prompt, batch.u_prompt)

    z_exact = solve_z_exact(G, b, args.lam_z)
    z_primal = solve_z_primal_jacobi(G, b, args.lam_z, args.z_depth)
    z_dual, _ = solve_z_dual_diag(G, b, args.lam_z, args.z_depth)

    def pred_from_z(z):
        Ahat = family.A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z, family.Abasis)
        return ridge_forward_solve(Ahat, batch.f_star, args.gamma_u), Ahat

    u_exact, A_exact = pred_from_z(z_exact)
    u_primal, A_primal = pred_from_z(z_primal)
    u_dual, A_dual = pred_from_z(z_dual)

    row = {
        "tag": tag,
        "d": args.d, "K": args.K, "m": args.m, "R": args.R, "M": args.m * args.R,
        "z_mse_exact": mse(z_exact, batch.z),
        "z_mse_primal_jacobi": mse(z_primal, batch.z),
        "z_mse_dual_diag": mse(z_dual, batch.z),
        "u_mse_exact": mse(u_exact, batch.u_star),
        "u_rel_exact": relerr(u_exact, batch.u_star),
        "u_mse_primal_jacobi": mse(u_primal, batch.u_star),
        "u_rel_primal_jacobi": relerr(u_primal, batch.u_star),
        "u_mse_dual_diag": mse(u_dual, batch.u_star),
        "u_rel_dual_diag": relerr(u_dual, batch.u_star),
        "A_rel_exact": frob_rel(A_exact, batch.A),
        "A_rel_primal_jacobi": frob_rel(A_primal, batch.A),
        "A_rel_dual_diag": frob_rel(A_dual, batch.A),
    }
    if csv_path is not None:
        append_csv(csv_path, row)
    return row


@torch.no_grad()
def model_eval(
    model: ParametricOperatorICL,
    true_family: OperatorFamily,
    args,
    device,
    csv_path: Optional[Path],
    step: int,
    tag: str,
) -> Dict:
    batch = sample_icl_batch(true_family, args.eval_batch_size, args.m, args.z_scale, args.f_std, args.noise_std, device)
    uhat, info = model(batch.f_prompt, batch.u_prompt, batch.f_star, return_info=True)
    zhat = info["z_hat"]
    Ahat = info["A_hat"]

    # true weak system for diagnostics only
    true_probes = coordinate_probes(args.d, args.R, device, true_family.A0.dtype)
    G_true, b_true = build_weak_system(true_family.A0, true_family.Abasis, true_probes, batch.f_prompt, batch.u_prompt)
    G_model, b_model = info["G"], info["b"]

    row = {
        "tag": tag,
        "step": step,
        "solver": model.solver,
        "d": args.d, "K": args.K, "m": args.m, "R": args.R, "M": args.m * args.R,
        "u_mse": mse(uhat, batch.u_star),
        "u_rel": relerr(uhat, batch.u_star),
        "z_mse_true_basis": mse(zhat, batch.z),
        "A_rel": frob_rel(Ahat, batch.A),
        "A0_rel": (model.A0 - true_family.A0).norm().item() / true_family.A0.norm().clamp_min(1e-12).item(),
        "Abasis_rel_raw": (model.Abasis - true_family.Abasis).flatten().norm().item() / true_family.Abasis.flatten().norm().clamp_min(1e-12).item(),
        "G_rel_oracle_raw": relerr(G_model.flatten(1), G_true.flatten(1)),
        "b_rel_oracle_raw": relerr(b_model, b_true),
    }

    for key in [
        "attn_entropy",
        "score_corr_linear_kernel",
        "o_corr_Kalpha",
        "dual_resid_norm",
        "Pe_corr_jacobi_e",
        "delta_z_corr_exact",
        "jacobi_z_corr_exact",
    ]:
        if key in info:
            row[key + "_last"] = float(info[key][-1].detach().cpu())

    if csv_path is not None:
        append_csv(csv_path, row)
    return row


# -----------------------------------------------------------------------------
# runners
# -----------------------------------------------------------------------------

def train(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    train_csv = outdir / "train_metrics.csv"
    eval_csv = outdir / "eval_metrics.csv"
    reset_csv(train_csv)
    reset_csv(eval_csv)

    true_family = make_true_family(args.d, args.K, args.basis_scale, args.A0_scale, device)
    model = ParametricOperatorICL(
        d=args.d,
        K=args.K,
        R=args.R,
        lam_z=args.lam_z,
        gamma_u=args.gamma_u,
        solver=args.solver,
        z_depth=args.z_depth,
        learn_dictionary=bool(args.learn_dictionary),
        learn_probes=bool(args.learn_probes),
        true_family=true_family,
        init=args.init,
        init_noise=args.init_noise,
        heads=args.heads,
        d_head=args.d_head,
        qk_from=args.qk_from,
        use_safe_scale=bool(args.use_safe_scale),
        attn_rho=args.attn_rho,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hard = oracle_hardcoded_eval(true_family, args, device, outdir / "hardcoded_before_training.csv", tag="oracle_before")
    print("ORACLE BEFORE TRAINING", json.dumps(hard, indent=2))

    if hard["u_rel_exact"] > args.require_hardcoded_u_rel:
        print(f"WARNING: hardcoded u_rel_exact={hard['u_rel_exact']:.3e} exceeds threshold {args.require_hardcoded_u_rel:.3e}")

    for step in range(1, args.steps + 1):
        z_scale = sample_log_uniform(args.z_scale_min, args.z_scale_max) if args.sample_z_scale_loguniform else args.z_scale
        batch = sample_icl_batch(true_family, args.batch_size, args.m, z_scale, args.f_std, args.noise_std, device)

        uhat, info = model(batch.f_prompt, batch.u_prompt, batch.f_star, return_info=True)
        Ahat = info["A_hat"]
        zhat = info["z_hat"]

        loss_u = F.mse_loss(uhat, batch.u_star)

        # Prompt consistency is allowed: it says the reconstructed operator should explain the prompt.
        # This is not a layer loss and not an attention constraint.
        loss_prompt = torch.tensor(0.0, device=device)
        if args.loss_prompt_weight > 0:
            pred_f = torch.einsum("bij,bmj->bmi", Ahat, batch.u_prompt)
            loss_prompt = F.mse_loss(pred_f, batch.f_prompt)

        # Optional diagnostics/supervision, off by default.
        loss_A = torch.tensor(0.0, device=device)
        if args.loss_A_weight > 0:
            loss_A = F.mse_loss(Ahat, batch.A)

        loss_z = torch.tensor(0.0, device=device)
        if args.loss_z_weight > 0:
            loss_z = F.mse_loss(zhat, batch.z)

        loss = loss_u + args.loss_prompt_weight * loss_prompt + args.loss_A_weight * loss_A + args.loss_z_weight * loss_z

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            train_row = {
                "step": step,
                "solver": args.solver,
                "loss": loss.item(),
                "loss_u": loss_u.item(),
                "loss_prompt": loss_prompt.item(),
                "loss_A": loss_A.item(),
                "loss_z": loss_z.item(),
                "z_scale": z_scale,
            }
            append_csv(train_csv, train_row)
            eval_row = model_eval(model, true_family, args, device, eval_csv, step=step, tag="eval")
            print("TRAIN", json.dumps(train_row, indent=2))
            print("EVAL", json.dumps(eval_row, indent=2))

        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / f"model_step{step}.pt")

    torch.save({"model": model.state_dict(), "args": vars(args)}, outdir / "model_final.pt")
    final = model_eval(model, true_family, args, device, eval_csv, step=args.steps, tag="final")
    print("FINAL", json.dumps(final, indent=2))


def hardcoded_sweep(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    csv_path = outdir / "hardcoded_sweep.csv"
    for d in parse_grid(args.d_grid, int):
        for K in parse_grid(args.K_grid, int):
            for m in parse_grid(args.m_grid, int):
                old = (args.d, args.K, args.m, args.R)
                args.d, args.K, args.m = d, K, m
                args.R = min(args.R, d)
                family = make_true_family(d, K, args.basis_scale, args.A0_scale, device)
                row = oracle_hardcoded_eval(family, args, device, csv_path, tag="sweep")
                print(json.dumps(row, indent=2))
                args.d, args.K, args.m, args.R = old


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="smoke", choices=["smoke", "hardcoded", "train"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--outdir", type=str, default=str(resolve_outdir("runs_pure_icl_richardson_attention")))
    p.add_argument("--seed", type=int, default=0)

    # sizes
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--R", type=int, default=32)
    p.add_argument("--d-grid", type=str, default="16,32,64")
    p.add_argument("--K-grid", type=str, default="4,8,16")
    p.add_argument("--m-grid", type=str, default="4,8,16,32")

    # data
    p.add_argument("--basis-scale", type=float, default=0.25)
    p.add_argument("--A0-scale", type=float, default=2.0)
    p.add_argument("--z-scale", type=float, default=0.5)
    p.add_argument("--z-scale-min", type=float, default=0.1)
    p.add_argument("--z-scale-max", type=float, default=1.0)
    p.add_argument("--sample-z-scale-loguniform", type=int, default=0)
    p.add_argument("--f-std", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.0)

    # solver
    p.add_argument("--solver", type=str, default="exact",
                   choices=["exact", "primal_jacobi", "dual_diag", "dual_attention_richardson", "dual_attention_kernel", "dual_attention_precond"])
    p.add_argument("--lam-z", type=float, default=1e-3)
    p.add_argument("--gamma-u", type=float, default=1e-5)
    p.add_argument("--z-depth", type=int, default=16)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--qk-from", type=str, default="g", choices=["g", "token"])
    p.add_argument("--use-safe-scale", type=int, default=1)
    p.add_argument("--attn-rho", type=float, default=1.0)

    # dictionary/encoder
    p.add_argument("--learn-dictionary", type=int, default=1)
    p.add_argument("--learn-probes", type=int, default=0)
    p.add_argument("--init", type=str, default="identity_random", choices=["identity_random", "true", "true_noisy"])
    p.add_argument("--init-noise", type=float, default=0.05)

    # training
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--loss-prompt-weight", type=float, default=1.0)
    p.add_argument("--loss-A-weight", type=float, default=0.0)
    p.add_argument("--loss-z-weight", type=float, default=0.0)
    p.add_argument("--require-hardcoded-u-rel", type=float, default=1e-2)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.outdir = str(resolve_outdir(args.outdir))
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ensure_dir(args.outdir)
    print("device:", device)

    if args.mode == "smoke":
        family = make_true_family(args.d, args.K, args.basis_scale, args.A0_scale, device)
        row = oracle_hardcoded_eval(family, args, device, ensure_dir(args.outdir) / "smoke_hardcoded.csv")
        print("SMOKE HARDCODED", json.dumps(row, indent=2))

        args.steps = min(args.steps, 50)
        args.eval_batch_size = min(args.eval_batch_size, 64)
        args.batch_size = min(args.batch_size, 64)
        train(args, device)
        return

    if args.mode == "hardcoded":
        hardcoded_sweep(args, device)
    elif args.mode == "train":
        train(args, device)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
