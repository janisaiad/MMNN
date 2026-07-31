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
       either exact small primal ridge solve, a legacy dual attention baseline,
       or the retained primal loop decoder with one learned softmax subspace
       head and exact Richardson/HeavyBall/Chebyshev/PCG algebra.

   Forward head:
       reconstructs A(z_hat) and predicts u_* by a differentiable ridge solve.

The retained decoder (because mR > K in the target regime) is:

    H = G^T G + lambda I,       c = G^T b
    Q_theta = one_softmax_head(G,H)
    B_theta = exact_jacobi_plus_Ritz(Q_theta,H)
    z^{l+1} = exact_loop_cell(H,c,B_theta,state^l)

For adaptive Heavy--Ball, a width-16 MLP sees seven fixed spectral summaries,
predicts only (mu,L), and the analytic formulas construct (alpha,beta).  It
does not approximate the HVP, products, divisions, or routing.  The old dual
softmax recurrence below is retained only as a Richardson-era baseline:

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

No MLP is used in that legacy Richardson update. The only learned parts of the dual
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

Train the same tied attention decoder with learned HeavyBall momentum:
    python pure_icl_parametric_operator_richardson_attention.py \
      --mode train --solver dual_attention_heavy_ball \
      --d 32 --K 8 --m 16 --R 32 --z-depth 16 \
      --heads 1 --d-head 64 --hb-alpha-init 1.0 --hb-beta-init 0.05 \
      --steps 50000 --batch-size 256 --device cuda \
      --outdir runs_pure_icl_dual_attention_heavy_ball

Train the retained exact primal loop-HB decoder:
    python pure_icl_parametric_operator_richardson_attention.py \
      --mode train --solver primal_loop_heavy_ball \
      --d 32 --K 8 --m 16 --R 32 --z-depth 8 \
      --subspace-slots 6 --d-head 64 \
      --loop-lmax-bound 2 --loop-step-init 0.9 --hb-beta-init 0.05 \
      --steps 5000 --batch-size 256 --device cuda \
      --outdir runs_pure_icl_loop_heavy_ball

The adaptive spectral policy is then fitted with
``train_adaptive_heavy_ball_interval.py`` while the encoder and subspace head
remain frozen.

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

try:
    from .exact_loop_transformer_decoder import ExactLoopTransformerDecoder
except ImportError:
    from exact_loop_transformer_decoder import ExactLoopTransformerDecoder

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
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A + jitter * eye, b.unsqueeze(-1)).squeeze(-1)


def ridge_forward_solve(A: Tensor, f: Tensor, gamma: float, jitter: float = 1e-7) -> Tensor:
    """Solve argmin_u ||A u - f||^2 + gamma ||u||^2."""
    B, d, _ = A.shape
    AtA = torch.einsum("bij,bik->bjk", A, A)
    Atf = torch.einsum("bij,bi->bj", A, f)
    eye = torch.eye(d, device=A.device, dtype=A.dtype).expand(B, d, d)
    return stable_solve(AtA + gamma * eye, Atf, jitter=jitter)


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


def make_true_family(
    d: int,
    K: int,
    basis_scale: float,
    A0_scale: float,
    device,
    dtype=torch.float32,
    operator_family: str = "dense_spd",
) -> OperatorFamily:
    if operator_family == "elliptic_1d":
        # Dirichlet first-difference map on d interior nodes and d+1 edges.
        difference = torch.zeros(d + 1, d, device=device, dtype=dtype)
        difference[0, 0] = 1.0
        difference[-1, -1] = -1.0
        if d > 1:
            indices = torch.arange(1, d, device=device)
            difference[indices, indices] = 1.0
            difference[indices, indices - 1] = -1.0
        stiffness = difference.transpose(0, 1) @ difference
        eye = torch.eye(d, device=device, dtype=dtype)
        A0 = 0.5 * eye + (A0_scale / 4.0) * stiffness
        edge_coordinate = (
            torch.arange(d + 1, device=device, dtype=dtype) + 0.5
        ) / (d + 1)
        basis = []
        for mode in range(1, K + 1):
            coefficient_mode = torch.sin(math.pi * mode * edge_coordinate)
            coefficient_mode = coefficient_mode - coefficient_mode.mean()
            matrix = difference.transpose(0, 1) @ (
                coefficient_mode[:, None] * difference
            )
            matrix = matrix - torch.trace(matrix) / d * eye
            matrix = matrix / matrix.norm().clamp_min(1e-12)
            basis.append(matrix * (basis_scale * math.sqrt(d)))
        return OperatorFamily(A0=A0, Abasis=torch.stack(basis))
    if operator_family != "dense_spd":
        raise ValueError(f"unknown operator family {operator_family}")
    eye = torch.eye(d, device=device, dtype=dtype)
    A0 = A0_scale * eye
    R = torch.randn(K, d, d, device=device, dtype=dtype)
    Abasis = 0.5 * (R + R.transpose(-1, -2))
    norms = Abasis.flatten(1).norm(dim=-1).clamp_min(1e-12)
    Abasis = Abasis / norms[:, None, None] * (basis_scale * math.sqrt(d))
    return OperatorFamily(A0=A0, Abasis=Abasis)


def assemble_A(A0: Tensor, Abasis: Tensor, z: Tensor, spd_shift: float = 0.25) -> Tensor:
    A = A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z, Abasis)
    eig_min = torch.linalg.eigvalsh(A)[:, 0]
    shift = (spd_shift - eig_min).clamp_min(0.0)
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return A + shift[:, None, None] * eye


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

def solve_z_exact(
    G: Tensor,
    b: Tensor,
    lam: float,
    ridge_metric: Tensor | None = None,
) -> Tensor:
    B, M, K = G.shape
    metric = (
        batch_eye(B, K, G.device, G.dtype)
        if ridge_metric is None
        else ridge_metric.expand(B, -1, -1)
    )
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * metric
    c = torch.einsum("bmk,bm->bk", G, b)
    # A coordinate-identity jitter would destroy GL(K) covariance.  The
    # covariant metric already makes H SPD, so no additional jitter is needed.
    return stable_solve(H, c, jitter=0.0 if ridge_metric is not None else 1e-7)


def solve_z_primal_jacobi(
    G: Tensor,
    b: Tensor,
    lam: float,
    depth: int,
    ridge_metric: Tensor | None = None,
) -> Tensor:
    B, M, K = G.shape
    metric = (
        batch_eye(B, K, G.device, G.dtype)
        if ridge_metric is None
        else ridge_metric.expand(B, -1, -1)
    )
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * metric
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


class NoMLPDualAttentionRichardson(nn.Module):
    """Pure attention + explicit Richardson or HeavyBall update.

    Learned:
        W_Q, W_K only.

    Fixed:
        V is alpha.
        eta_i = 1/(||g_i||^2 + lambda).
        update is explicit Richardson:
            alpha_i <- alpha_i + eta_i [ b_i - o_i - lambda alpha_i ]

    No MLP, no learned eta, no layer loss, no forced score loss to GG^T.
    """

    def __init__(
        self,
        K: int,
        depth: int,
        lam: float,
        n_heads: int = 1,
        d_head: int = 64,
        qk_from: str = "g",       # "g" or "token"
        use_safe_scale: bool = True,
        iteration: str = "richardson",
        hb_alpha_init: float = 1.0,
        hb_beta_init: float = 0.05,
    ):
        super().__init__()
        self.K = K
        self.depth = depth
        self.lam = lam
        self.n_heads = n_heads
        self.d_head = d_head
        self.qk_from = qk_from
        self.use_safe_scale = use_safe_scale
        self.iteration = iteration

        if iteration == "heavy_ball":
            beta_fraction = min(max(hb_beta_init / 0.999, 1e-5), 1.0 - 1e-5)
            self.raw_beta = nn.Parameter(torch.tensor(math.log(beta_fraction / (1.0 - beta_fraction))))
            alpha_cap = 2.0 * (1.0 + hb_beta_init)
            alpha_fraction = min(max(hb_alpha_init / (0.999 * alpha_cap), 1e-5), 1.0 - 1e-5)
            self.raw_alpha = nn.Parameter(torch.tensor(math.log(alpha_fraction / (1.0 - alpha_fraction))))
        elif iteration != "richardson":
            raise ValueError(f"unknown iteration {iteration}")

        token_dim = K + 3  # g, b, alpha, ||g||
        qk_dim = K if qk_from == "g" else token_dim
        self.Wq = nn.Linear(qk_dim, n_heads * d_head, bias=False)
        self.Wk = nn.Linear(qk_dim, n_heads * d_head, bias=False)
        self._init_linear_kernel_qk()

    def _init_linear_kernel_qk(self) -> None:
        """Start near scores_ij = g_i·g_j so the loop is stable like dual_diag."""
        with torch.no_grad():
            self.Wq.weight.zero_()
            self.Wk.weight.zero_()
            in_dim = self.K if self.qk_from == "g" else self.K + 3
            scale = 1.0 / math.sqrt(self.d_head)
            for h in range(self.n_heads):
                off = h * self.d_head
                dim = min(in_dim, self.d_head)
                for i in range(dim):
                    self.Wq.weight[off + i, i] = scale
                    self.Wk.weight[off + i, i] = scale

    def make_token(self, G: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
        rownorm = G.norm(dim=-1, keepdim=True)
        return torch.cat([G, b.unsqueeze(-1), alpha.unsqueeze(-1), rownorm], dim=-1)

    def iteration_coefficients(self) -> Tuple[Tensor, Tensor]:
        if self.iteration == "richardson":
            one = self.Wq.weight.new_tensor(1.0)
            return one, self.Wq.weight.new_tensor(0.0)
        momentum = 0.999 * torch.sigmoid(self.raw_beta)
        stable_step_cap = 2.0 * (1.0 + momentum)
        step_scale = 0.999 * stable_step_cap * torch.sigmoid(self.raw_alpha)
        return step_scale, momentum

    def forward(self, G: Tensor, b: Tensor, return_layers: bool = False) -> Tuple[Tensor, Tensor, Dict]:
        B, M, K = G.shape
        alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)
        alpha_prev = torch.zeros_like(alpha)
        step_scale, momentum = self.iteration_coefficients()

        Kdual = torch.einsum("bik,bjk->bij", G, G)
        diag_eta = 1.0 / (torch.diagonal(Kdual, dim1=-2, dim2=-1) + self.lam).clamp_min(1e-12)

        if self.use_safe_scale:
            Adual = Kdual + self.lam * batch_eye(B, M, G.device, G.dtype)
            DA = diag_eta.unsqueeze(-1) * Adual
            safe_scale = richardson_safe_scale(DA)
        else:
            safe_scale = torch.ones(B, device=G.device, dtype=G.dtype)

        alayers, zlayers, entropies, score_corrs, o_corrs = [], [], [], [], []

        for _ in range(self.depth):
            token = self.make_token(G, b, alpha)
            qk_src = G if self.qk_from == "g" else token

            Q = self.Wq(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
            Kmat = self.Wk(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)

            scores = torch.einsum("bhid,bhjd->bhij", Q, Kmat) / math.sqrt(self.d_head)
            Att = torch.softmax(scores, dim=-1)

            o_heads = torch.einsum("bhij,bj->bhi", Att, alpha)
            o = o_heads.mean(dim=1)

            res = b - o - self.lam * alpha
            alpha_next = (
                alpha
                + step_scale * safe_scale[:, None] * diag_eta * res
                + momentum * (alpha - alpha_prev)
            )
            alpha_prev, alpha = alpha, alpha_next

            if return_layers:
                alayers.append(alpha)
                zlayers.append(torch.einsum("bmk,bm->bk", G, alpha))
                ent = -(Att.clamp_min(1e-12) * Att.clamp_min(1e-12).log()).sum(-1).mean()
                entropies.append(ent.detach())

                smean = scores.mean(dim=1)
                score_corrs.append(torch.tensor(corr_flat(smean.detach(), Kdual.detach()), device=G.device))

                Kalpha = torch.einsum("bij,bj->bi", Kdual, alpha.detach())
                if Kalpha.norm(dim=-1).mean() > 1e-8 and o.norm(dim=-1).mean() > 1e-8:
                    # correlation between attention output o and exact K alpha
                    oc = []
                    for bb in range(B):
                        aa = o[bb] - o[bb].mean()
                        kk = Kalpha[bb] - Kalpha[bb].mean()
                        oc.append((aa * kk).sum() / (aa.norm() * kk.norm()).clamp_min(1e-12))
                    o_corrs.append(torch.stack(oc).mean().detach())
                else:
                    o_corrs.append(torch.tensor(0.0, device=G.device))

        z = torch.einsum("bmk,bm->bk", G, alpha)
        info = {}
        if return_layers:
            info["alpha_layers"] = torch.stack(alayers, dim=1)
            info["z_layers"] = torch.stack(zlayers, dim=1)
            info["attn_entropy"] = torch.stack(entropies)
            info["score_corr_linear_kernel"] = torch.stack(score_corrs)
            info["o_corr_Kalpha"] = torch.stack(o_corrs)
            info["step_scale"] = step_scale
            info["momentum"] = momentum
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
        dual_attention_heavy_ball : same tied attention + learned stable momentum

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
        hb_alpha_init: float,
        hb_beta_init: float,
        subspace_slots: int = 4,
        loop_lmax_bound: float = 4.0,
        loop_step_init: float = 0.25,
        chebyshev_hidden_dimension: int = 16,
        adaptive_heavy_ball: bool = False,
        interval_lower_calibration: float = 1.0,
        interval_upper_calibration: float = 1.0,
        hybrid_residual_threshold: float = 1e-8,
        dictionary_projection: str = "none",
        freeze_A0: bool = False,
        covariant_ridge: bool = False,
        loop_preconditioner_head: str = "coordinate_ritz",
        prompt_subspace_refinement_steps: int = 2,
        chebyshev_interval_policy: str = "learned",
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.R = R
        self.lam_z = lam_z
        self.gamma_u = gamma_u
        self.solver = solver
        self.z_depth = z_depth
        self.dictionary_projection = dictionary_projection
        self.covariant_ridge = bool(covariant_ridge)
        if dictionary_projection not in {"none", "elliptic_1d"}:
            raise ValueError(f"unknown dictionary projection {dictionary_projection}")

        if true_family is not None and init in ["true", "true_noisy"]:
            A0_init = true_family.A0.detach().clone()
            Ab_init = true_family.Abasis.detach().clone()
            if init == "true_noisy":
                A0_init = A0_init + init_noise * torch.randn_like(A0_init)
                Ab_init = Ab_init + init_noise * torch.randn_like(Ab_init)
        elif true_family is not None and init == "true_A0_random_basis":
            A0_init = true_family.A0.detach().clone()
            random_basis = torch.randn(K, d, d, device=A0_init.device, dtype=A0_init.dtype)
            Ab_init = 0.5 * (random_basis + random_basis.transpose(-1, -2))
            Ab_init = Ab_init * (0.1 / math.sqrt(d))
        else:
            A0_init = 2.0 * torch.eye(d)
            Rnd = torch.randn(K, d, d)
            Ab_init = 0.5 * (Rnd + Rnd.transpose(-1, -2)) * (0.1 / math.sqrt(d))

        self.A0 = nn.Parameter(
            A0_init,
            requires_grad=bool(learn_dictionary) and not bool(freeze_A0),
        )
        self.Abasis = nn.Parameter(Ab_init, requires_grad=bool(learn_dictionary))
        if dictionary_projection == "elliptic_1d":
            difference = torch.zeros(d + 1, d, device=A0_init.device, dtype=A0_init.dtype)
            difference[0, 0] = 1.0
            difference[-1, -1] = -1.0
            if d > 1:
                indices = torch.arange(1, d, device=A0_init.device)
                difference[indices, indices] = 1.0
                difference[indices, indices - 1] = -1.0
            atoms = [
                torch.outer(difference[edge], difference[edge])
                for edge in range(d + 1)
            ]
            atoms.append(torch.eye(d, device=A0_init.device, dtype=A0_init.dtype))
            atom_matrix = torch.stack(atoms).flatten(1).transpose(0, 1)
            left, singular_values, _ = torch.linalg.svd(atom_matrix, full_matrices=False)
            numerical_rank = int(
                (singular_values > singular_values[0] * 1e-6).sum().item()
            )
            projection_basis = left[:, :numerical_rank]
        else:
            projection_basis = torch.empty(
                d * d,
                0,
                device=A0_init.device,
                dtype=A0_init.dtype,
            )
        self.register_buffer("dictionary_projection_basis", projection_basis)
        self.project_dictionary_()

        probes_init = coordinate_probes(d, R, A0_init.device, A0_init.dtype)
        self.probes = nn.Parameter(probes_init, requires_grad=bool(learn_probes))

        if solver in ["dual_attention_richardson", "dual_attention_heavy_ball"]:
            self.dual_attention = NoMLPDualAttentionRichardson(
                K=K,
                depth=z_depth,
                lam=lam_z,
                n_heads=heads,
                d_head=d_head,
                qk_from=qk_from,
                use_safe_scale=bool(use_safe_scale),
                iteration="heavy_ball" if solver == "dual_attention_heavy_ball" else "richardson",
                hb_alpha_init=hb_alpha_init,
                hb_beta_init=hb_beta_init,
            )
        else:
            self.dual_attention = None
        if solver.startswith("primal_loop_"):
            controller = solver.removeprefix("primal_loop_")
            self.loop_decoder = ExactLoopTransformerDecoder(
                dimension=K,
                depth=z_depth,
                head_dimension=d_head,
                slots=subspace_slots,
                controller=controller,
                spectral_lmax_bound=loop_lmax_bound,
                step_init=loop_step_init,
                momentum_init=hb_beta_init,
                chebyshev_hidden_dimension=chebyshev_hidden_dimension,
                adaptive_heavy_ball=adaptive_heavy_ball,
                interval_lower_calibration=interval_lower_calibration,
                interval_upper_calibration=interval_upper_calibration,
                hybrid_residual_threshold=hybrid_residual_threshold,
                preconditioner_head_type=loop_preconditioner_head,
                prompt_subspace_refinement_steps=prompt_subspace_refinement_steps,
                chebyshev_interval_policy=chebyshev_interval_policy,
            )
        else:
            self.loop_decoder = None

    def weak_system(self, f_prompt: Tensor, u_prompt: Tensor) -> Tuple[Tensor, Tensor]:
        return build_weak_system(self.A0, self.Abasis, self.probes, f_prompt, u_prompt)

    def coefficient_ridge_metric(self) -> Tensor | None:
        if not self.covariant_ridge:
            return None
        flattened = self.Abasis.flatten(1)
        metric = flattened @ flattened.transpose(0, 1)
        scale = torch.diagonal(metric).mean().clamp_min(1e-12)
        identity = torch.eye(self.K, device=metric.device, dtype=metric.dtype)
        return metric + 1e-8 * scale * identity

    @torch.no_grad()
    def project_dictionary_(self) -> None:
        """Project only known PDE structure; leave the low-rank span learned."""

        if self.dictionary_projection == "none":
            return
        basis = self.dictionary_projection_basis

        def project(matrices: Tensor) -> Tensor:
            original_shape = matrices.shape
            flattened = matrices.reshape(-1, self.d * self.d)
            coordinates = flattened @ basis
            return (coordinates @ basis.transpose(0, 1)).reshape(original_shape)

        if self.A0.requires_grad:
            self.A0.copy_(project(self.A0))
        # Re-project only trainable tensors.  Repeated finite-precision
        # projection of an already frozen basis is not exactly idempotent and
        # can otherwise create a slow, optimizer-independent encoder drift.
        if self.Abasis.requires_grad:
            self.Abasis.copy_(project(self.Abasis))

    def solve_z(self, G: Tensor, b: Tensor, return_layers: bool = False) -> Tuple[Tensor, Dict]:
        info = {}
        ridge_metric = self.coefficient_ridge_metric()
        if self.solver == "exact":
            z = solve_z_exact(G, b, self.lam_z, ridge_metric)
        elif self.solver == "primal_jacobi":
            z = solve_z_primal_jacobi(
                G, b, self.lam_z, self.z_depth, ridge_metric
            )
        elif self.solver == "dual_diag":
            if ridge_metric is not None:
                raise ValueError("covariant ridge is implemented on the primal side")
            z, alpha = solve_z_dual_diag(G, b, self.lam_z, self.z_depth)
            info["alpha"] = alpha
        elif self.solver in ["dual_attention_richardson", "dual_attention_heavy_ball"]:
            z, alpha, dinfo = self.dual_attention(G, b, return_layers=return_layers)
            info.update(dinfo)
            info["alpha"] = alpha
        elif self.solver.startswith("primal_loop_"):
            if self.loop_decoder is None:
                raise RuntimeError("loop decoder was not initialized")
            z, loop_info = self.loop_decoder(
                G, b, self.lam_z, ridge_metric=ridge_metric
            )
            info.update(loop_info)
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
    ridge_metric = None
    if bool(getattr(args, "covariant_ridge", 0)):
        flattened_basis = family.Abasis.flatten(1)
        ridge_metric = flattened_basis @ flattened_basis.transpose(0, 1)

    z_exact = solve_z_exact(G, b, args.lam_z, ridge_metric)
    z_primal = solve_z_primal_jacobi(
        G, b, args.lam_z, args.z_depth, ridge_metric
    )
    evaluate_dual = bool(getattr(args, "eval_dual_baseline", 1))
    z_dual = None
    if evaluate_dual:
        z_dual, _ = solve_z_dual_diag(G, b, args.lam_z, args.z_depth)

    def pred_from_z(z):
        Ahat = family.A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z, family.Abasis)
        return ridge_forward_solve(Ahat, batch.f_star, args.gamma_u), Ahat

    u_exact, A_exact = pred_from_z(z_exact)
    u_primal, A_primal = pred_from_z(z_primal)
    if z_dual is not None:
        u_dual, A_dual = pred_from_z(z_dual)
    else:
        u_dual = A_dual = None

    row = {
        "tag": tag,
        "d": args.d, "K": args.K, "m": args.m, "R": args.R, "M": args.m * args.R,
        "z_mse_exact": mse(z_exact, batch.z),
        "z_mse_primal_jacobi": mse(z_primal, batch.z),
        "z_mse_dual_diag": mse(z_dual, batch.z) if z_dual is not None else math.nan,
        "u_mse_exact": mse(u_exact, batch.u_star),
        "u_rel_exact": relerr(u_exact, batch.u_star),
        "u_mse_primal_jacobi": mse(u_primal, batch.u_star),
        "u_rel_primal_jacobi": relerr(u_primal, batch.u_star),
        "u_mse_dual_diag": mse(u_dual, batch.u_star) if u_dual is not None else math.nan,
        "u_rel_dual_diag": relerr(u_dual, batch.u_star) if u_dual is not None else math.nan,
        "A_rel_exact": frob_rel(A_exact, batch.A),
        "A_rel_primal_jacobi": frob_rel(A_primal, batch.A),
        "A_rel_dual_diag": frob_rel(A_dual, batch.A) if A_dual is not None else math.nan,
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
    learned_basis_q = torch.linalg.qr(model.Abasis.flatten(1).transpose(0, 1), mode="reduced").Q
    true_basis_q = torch.linalg.qr(true_family.Abasis.flatten(1).transpose(0, 1), mode="reduced").Q
    basis_overlap = torch.linalg.matrix_norm(
        true_basis_q.transpose(0, 1) @ learned_basis_q,
        ord="fro",
    ).square() / min(learned_basis_q.shape[1], true_basis_q.shape[1])

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
        "Abasis_subspace_overlap": basis_overlap.item(),
        "G_rel_oracle_raw": relerr(G_model.flatten(1), G_true.flatten(1)),
        "b_rel_oracle_raw": relerr(b_model, b_true),
    }

    if "attn_entropy" in info:
        row["attn_entropy_last"] = float(info["attn_entropy"][-1].detach().cpu())
    if "score_corr_linear_kernel" in info:
        row["score_corr_linear_kernel_last"] = float(info["score_corr_linear_kernel"][-1].detach().cpu())
    if "o_corr_Kalpha" in info:
        row["o_corr_Kalpha_last"] = float(info["o_corr_Kalpha"][-1].detach().cpu())
    if "step_scale" in info:
        row["decoder_step_scale"] = float(info["step_scale"].detach().cpu())
        row["decoder_momentum"] = float(info["momentum"].detach().cpu())
    elif "step" in info:
        row["decoder_step_scale"] = float(info["step"].mean().detach().cpu())
        row["decoder_momentum"] = float(info["momentum"].mean().detach().cpu())
    if "spectral_min" in info:
        row["predicted_spectral_min_mean"] = float(info["spectral_min"].mean().detach().cpu())
        row["predicted_spectral_max_mean"] = float(info["spectral_max"].mean().detach().cpu())
    if "pcg_fallback_rate" in info:
        row["pcg_fallback_rate"] = float(
            info["pcg_fallback_rate"].detach().cpu()
        )
        row["hb_final_preconditioned_residual_ratio_mean"] = float(
            info["hb_final_preconditioned_residual_ratio"].mean().detach().cpu()
        )

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

    true_family = make_true_family(
        args.d,
        args.K,
        args.basis_scale,
        args.A0_scale,
        device,
        operator_family=args.operator_family,
    )
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
        hb_alpha_init=args.hb_alpha_init,
        hb_beta_init=args.hb_beta_init,
        subspace_slots=args.subspace_slots,
        loop_lmax_bound=args.loop_lmax_bound,
        loop_step_init=args.loop_step_init,
        chebyshev_hidden_dimension=args.chebyshev_hidden_dimension,
        adaptive_heavy_ball=bool(args.adaptive_heavy_ball),
        interval_lower_calibration=args.interval_lower_calibration,
        interval_upper_calibration=args.interval_upper_calibration,
        hybrid_residual_threshold=args.hybrid_residual_threshold,
        dictionary_projection=args.dictionary_projection,
        freeze_A0=bool(args.freeze_A0),
        covariant_ridge=bool(args.covariant_ridge),
        loop_preconditioner_head=args.loop_preconditioner_head,
        prompt_subspace_refinement_steps=args.prompt_subspace_refinement_steps,
        chebyshev_interval_policy=args.chebyshev_interval_policy,
    ).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
    elif args.encoder_checkpoint:
        checkpoint = torch.load(
            args.encoder_checkpoint,
            map_location=device,
            weights_only=True,
        )
        source_state = checkpoint["model"]
        with torch.no_grad():
            model.A0.copy_(source_state["A0"])
            model.Abasis.copy_(source_state["Abasis"])
            model.probes.copy_(source_state["probes"])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hard = oracle_hardcoded_eval(true_family, args, device, outdir / "hardcoded_before_training.csv", tag="oracle_before")
    print("ORACLE BEFORE TRAINING", json.dumps(hard, indent=2))

    if hard["u_rel_exact"] > args.require_hardcoded_u_rel:
        print(f"WARNING: hardcoded u_rel_exact={hard['u_rel_exact']:.3e} exceeds threshold {args.require_hardcoded_u_rel:.3e}")

    best_eval_u_mse = math.inf
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
        model.project_dictionary_()

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
            if eval_row["u_mse"] < best_eval_u_mse:
                best_eval_u_mse = eval_row["u_mse"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "best_eval_u_mse": best_eval_u_mse,
                        "best_step": step,
                    },
                    outdir / "model_best.pt",
                )
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
                family = make_true_family(
                    d,
                    K,
                    args.basis_scale,
                    args.A0_scale,
                    device,
                    operator_family=args.operator_family,
                )
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
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--encoder-checkpoint", type=str, default="")

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
    p.add_argument(
        "--operator-family",
        choices=["dense_spd", "elliptic_1d"],
        default="dense_spd",
    )
    p.add_argument("--z-scale", type=float, default=0.5)
    p.add_argument("--z-scale-min", type=float, default=0.1)
    p.add_argument("--z-scale-max", type=float, default=1.0)
    p.add_argument("--sample-z-scale-loguniform", type=int, default=0)
    p.add_argument("--f-std", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.0)

    # solver
    p.add_argument(
        "--solver",
        type=str,
        default="exact",
        choices=[
            "exact",
            "primal_jacobi",
            "dual_diag",
            "dual_attention_richardson",
            "dual_attention_heavy_ball",
            "primal_loop_richardson",
            "primal_loop_heavy_ball",
            "primal_loop_chebyshev",
            "primal_loop_pcg",
            "primal_loop_certified_hb_pcg",
        ],
    )
    p.add_argument("--lam-z", type=float, default=1e-3)
    p.add_argument("--gamma-u", type=float, default=1e-5)
    p.add_argument("--z-depth", type=int, default=16)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--qk-from", type=str, default="g", choices=["g", "token"])
    p.add_argument("--use-safe-scale", type=int, default=1)
    p.add_argument("--hb-alpha-init", type=float, default=1.0)
    p.add_argument("--hb-beta-init", type=float, default=0.05)
    p.add_argument("--subspace-slots", type=int, default=6)
    p.add_argument("--loop-lmax-bound", type=float, default=4.0)
    p.add_argument("--loop-step-init", type=float, default=0.25)
    p.add_argument(
        "--loop-preconditioner-head",
        choices=[
            "coordinate_ritz",
            "equivariant_ritz_softmax",
            "equivariant_prompt_nystrom",
            "equivariant_matrix_free_nystrom",
        ],
        default="coordinate_ritz",
    )
    p.add_argument("--prompt-subspace-refinement-steps", type=int, default=2)
    p.add_argument(
        "--chebyshev-interval-policy",
        choices=["learned", "exact_head_spectrum"],
        default="learned",
    )
    p.add_argument("--chebyshev-hidden-dimension", type=int, default=16)
    p.add_argument("--adaptive-heavy-ball", type=int, default=0)
    p.add_argument("--interval-lower-calibration", type=float, default=1.0)
    p.add_argument("--interval-upper-calibration", type=float, default=1.0)
    p.add_argument("--hybrid-residual-threshold", type=float, default=1e-8)

    # dictionary/encoder
    p.add_argument("--learn-dictionary", type=int, default=1)
    p.add_argument("--learn-probes", type=int, default=0)
    p.add_argument(
        "--init",
        type=str,
        default="identity_random",
        choices=["identity_random", "true", "true_noisy", "true_A0_random_basis"],
    )
    p.add_argument("--init-noise", type=float, default=0.05)
    p.add_argument("--freeze-A0", type=int, default=0)
    p.add_argument("--covariant-ridge", type=int, default=0)
    p.add_argument(
        "--dictionary-projection",
        choices=["none", "elliptic_1d"],
        default="none",
    )

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
    p.add_argument(
        "--eval-dual-baseline",
        type=int,
        default=0,
        help="materialize the legacy M-by-M dual baseline during oracle checks",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.outdir = str(resolve_outdir(args.outdir))
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ensure_dir(args.outdir)
    print("device:", device)

    if args.mode == "smoke":
        family = make_true_family(
            args.d,
            args.K,
            args.basis_scale,
            args.A0_scale,
            device,
            operator_family=args.operator_family,
        )
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
