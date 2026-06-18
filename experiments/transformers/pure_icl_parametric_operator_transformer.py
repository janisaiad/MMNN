#!/usr/bin/env python3
"""
pure_icl_parametric_operator_transformer.py

Self-contained CUDA/PyTorch code for the actual pure ICL task from the proposal:

    pretraining tasks:
        sample z
        A(z) = A0 + sum_k z_k A_k
        prompt pairs: (f_i, u_i), with A(z) u_i = f_i
        query: f_*
        target: u_* = A(z)^(-1) f_*

    model:
        prompt (f_i, u_i), f_*  ->  u_hat_*

This file has both parts:

1. HARDCODED / ORACLE CHECK
   Uses the true A0,A_k and weak probes v_r to build the exact weak LS system

        G z ~= b

   then solves z, reconstructs A(z), and predicts u_*.
   This verifies the parametric weak formulation itself.

2. TRAINING
   Trains one structured encoder-decoder Transformer-like model end-to-end.
   The model learns a global operator dictionary A0,A_1,...,A_K.
   From prompt pairs, it builds weak equation tokens (G_hat,b_hat).
   A decoder either:
      - solves z by exact small ridge solve,
      - or runs a recurrent dual softmax-attention Richardson solver,
      - or runs primal Richardson.
   Then it reconstructs A(z_hat) and predicts u_*.

The architecture is deliberately not a generic black-box transformer.
It is the cleanest structured ICL version of our proposal:

    encoder heads learn A_k,
    weak tokens are produced from (f_i,u_i),
    decoder infers task coefficients z,
    final forward decoder solves with A(z_hat).

Key notation:
    d = Galerkin solution dimension
    K = number of operator coefficients / basis matrices A_k
    m = prompt length
    R = number of weak probes / test vectors
    M = m * R weak equations

Typical commands
----------------

Smoke hardcoded + tiny train:
    python pure_icl_parametric_operator_transformer.py --mode smoke --device cuda

Hardcoded sweep:
    python pure_icl_parametric_operator_transformer.py --mode hardcoded \
      --d-grid 16,32,64 --K-grid 4,8,16 --m-grid 4,8,16,32 \
      --device cuda --outdir runs/pure_icl_hardcoded

Train exact-z structured model:
    python pure_icl_parametric_operator_transformer.py --mode train \
      --solver exact --learn-dictionary 1 --learn-probes 0 \
      --d 32 --K 8 --m 16 --R 32 --steps 30000 \
      --batch-size 256 --device cuda --outdir runs/pure_icl_train_exact

Train recurrent dual-attention model:
    python pure_icl_parametric_operator_transformer.py --mode train \
      --solver dual_attention --learn-dictionary 1 --learn-probes 0 \
      --d 32 --K 8 --m 16 --R 32 --z-depth 12 --steps 50000 \
      --batch-size 256 --device cuda --outdir runs/pure_icl_train_dual_attention

Train on broader distribution:
    python pure_icl_parametric_operator_transformer.py --mode train \
      --solver exact --d 64 --K 16 --m 32 --R 64 \
      --z-scale-min 0.1 --z-scale-max 1.0 --sample-z-scale-loguniform 1 \
      --steps 50000 --device cuda --outdir runs/pure_icl_train_broad
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


def sample_log_uniform(lo: float, hi: float) -> float:
    u = random.random()
    return math.exp(math.log(lo) * (1 - u) + math.log(hi) * u)


# -----------------------------------------------------------------------------
# parametric operator task generation
# -----------------------------------------------------------------------------

@dataclass
class OperatorFamily:
    A0: Tensor       # [d,d]
    Abasis: Tensor   # [K,d,d]


@dataclass
class ICLBatch:
    z: Tensor        # [B,K]
    A: Tensor        # [B,d,d]
    f_prompt: Tensor # [B,m,d]
    u_prompt: Tensor # [B,m,d]
    f_star: Tensor   # [B,d]
    u_star: Tensor   # [B,d]


def make_true_family(d: int, K: int, basis_scale: float, A0_scale: float, device, dtype=torch.float32) -> OperatorFamily:
    I = torch.eye(d, device=device, dtype=dtype)
    A0 = A0_scale * I
    R = torch.randn(K, d, d, device=device, dtype=dtype)
    Abasis = 0.5 * (R + R.transpose(-1, -2))
    # Normalize each basis matrix to Frobenius norm about basis_scale.
    norms = Abasis.flatten(1).norm(dim=-1).clamp_min(1e-12)
    Abasis = Abasis / norms[:, None, None] * (basis_scale * math.sqrt(d))
    return OperatorFamily(A0=A0, Abasis=Abasis)


def assemble_A(A0: Tensor, Abasis: Tensor, z: Tensor, spd_shift: float = 0.25) -> Tensor:
    A = A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z, Abasis)
    # Make true A safe SPD by shifting if needed.
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
# weak encoder and solvers
# -----------------------------------------------------------------------------

def coordinate_probes(d: int, R: int, device, dtype) -> Tensor:
    if R <= d:
        return torch.eye(d, device=device, dtype=dtype)[:R]
    # repeat coordinates if R>d; normally use R<=d or R=d.
    reps = math.ceil(R / d)
    return torch.eye(d, device=device, dtype=dtype).repeat(reps, 1)[:R]


def build_weak_system(
    A0: Tensor,
    Abasis: Tensor,
    probes: Tensor,
    f_prompt: Tensor,
    u_prompt: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Build G,b from prompt.

    A(z)=A0+sum_k z_k A_k
    A(z)u_i=f_i

    For probe v_r:
        g_{i,r,k} = <A_k u_i, v_r>
        b_{i,r}   = <f_i - A0 u_i, v_r>

    Returns:
        G [B, m*R, K]
        b [B, m*R]
    """
    B, m, d = u_prompt.shape
    K = Abasis.shape[0]
    R = probes.shape[0]
    Ak_u = torch.einsum("kde,bme->bmkd", Abasis, u_prompt)  # [B,m,K,d]
    G = torch.einsum("rd,bmkd->bmrk", probes, Ak_u).reshape(B, m * R, K)
    A0u = torch.einsum("de,bme->bmd", A0, u_prompt)
    rhs = f_prompt - A0u
    b = torch.einsum("rd,bmd->bmr", probes, rhs).reshape(B, m * R)
    return G, b


def solve_z_exact(G: Tensor, b: Tensor, lam: float) -> Tensor:
    B, M, K = G.shape
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(B, K, G.device, G.dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    return stable_solve(H, c)


def solve_z_primal_richardson(G: Tensor, b: Tensor, lam: float, depth: int, mode: str = "jacobi") -> Tensor:
    B, M, K = G.shape
    H = torch.einsum("bmk,bml->bkl", G, G) + lam * batch_eye(B, K, G.device, G.dtype)
    c = torch.einsum("bmk,bm->bk", G, b)
    z = torch.zeros(B, K, device=G.device, dtype=G.dtype)
    if mode == "scalar":
        eig = torch.linalg.eigvalsh(H)
        eta = 2.0 / (eig[:, 0] + eig[:, -1]).clamp_min(1e-12)
    elif mode == "jacobi":
        diag = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-12)
        Dinv = 1.0 / diag
        DH = Dinv.unsqueeze(-1) * H
        rad = torch.linalg.eigvals(DH).abs().real.max(dim=-1).values.clamp_min(1e-12)
        eta = 1.0 / rad
    else:
        raise ValueError(mode)
    for _ in range(depth):
        grad = c - torch.einsum("bij,bj->bi", H, z)
        if mode == "scalar":
            z = z + eta[:, None] * grad
        else:
            z = z + eta[:, None] * Dinv * grad
    return z


def solve_z_dual_diag(G: Tensor, b: Tensor, lam: float, depth: int) -> Tuple[Tensor, Tensor]:
    B, M, K = G.shape
    Kdual = torch.einsum("bik,bjk->bij", G, G)
    alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)
    Dinv = 1.0 / (torch.diagonal(Kdual, dim1=-2, dim2=-1) + lam).clamp_min(1e-12)
    # safe local scaling
    Adual = Kdual + lam * batch_eye(B, M, G.device, G.dtype)
    DA = Dinv.unsqueeze(-1) * Adual
    rad = torch.linalg.eigvals(DA).abs().real.max(dim=-1).values.clamp_min(1e-12)
    eta = 1.0 / rad
    for _ in range(depth):
        res = b - torch.einsum("bij,bj->bi", Kdual, alpha) - lam * alpha
        alpha = alpha + eta[:, None] * Dinv * res
    z = torch.einsum("bmk,bm->bk", G, alpha)
    return z, alpha


class DualAttentionZSolver(nn.Module):
    """Trainable recurrent dual attention solver.

    It receives G,b produced by the encoder and tries to infer z.

    This is intentionally close to a Transformer block:
        token_i = [g_i, b_i, alpha_i, ||g_i||]
        q_i = W_Q token_i
        k_i = W_K token_i
        v_i = W_V token_i
        alpha_i <- alpha_i + local_update(token_i, attn_i)

    Final readout:
        z = G^T alpha
    """
    def __init__(
        self,
        K: int,
        d_model: int = 128,
        n_heads: int = 1,
        d_head: int = 64,
        value_dim: int = 32,
        depth: int = 8,
        hidden: int = 256,
        qk_from_g_only: bool = True,
    ):
        super().__init__()
        self.K = K
        self.depth = depth
        self.n_heads = n_heads
        self.d_head = d_head
        self.value_dim = value_dim
        self.qk_from_g_only = qk_from_g_only

        tok_dim = K + 3  # g, b, alpha, rownorm
        qk_dim = K if qk_from_g_only else tok_dim

        self.Wq = nn.Linear(qk_dim, n_heads * d_head, bias=False)
        self.Wk = nn.Linear(qk_dim, n_heads * d_head, bias=False)
        self.Wv = nn.Linear(tok_dim, n_heads * value_dim, bias=True)

        self.update = nn.Sequential(
            nn.Linear(tok_dim + n_heads * value_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.step = nn.Parameter(torch.tensor(0.1))

    def token(self, G: Tensor, b: Tensor, alpha: Tensor) -> Tensor:
        rownorm = G.norm(dim=-1, keepdim=True)
        return torch.cat([G, b.unsqueeze(-1), alpha.unsqueeze(-1), rownorm], dim=-1)

    def forward(self, G: Tensor, b: Tensor, return_layers: bool = False) -> Tuple[Tensor, Tensor, Dict]:
        B, M, K = G.shape
        alpha = torch.zeros(B, M, device=G.device, dtype=G.dtype)
        alpha_layers = []
        z_layers = []
        entropies = []
        score_corrs = []
        Kdual = torch.einsum("bik,bjk->bij", G, G)
        for _ in range(self.depth):
            tok = self.token(G, b, alpha)
            qk_src = G if self.qk_from_g_only else tok
            Q = self.Wq(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
            Kmat = self.Wk(qk_src).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
            V = self.Wv(tok).view(B, M, self.n_heads, self.value_dim).transpose(1, 2)
            scores = torch.einsum("bhid,bhjd->bhij", Q, Kmat) / math.sqrt(self.d_head)
            Att = torch.softmax(scores, dim=-1)
            O = torch.einsum("bhij,bhjv->bhiv", Att, V).transpose(1, 2).reshape(B, M, self.n_heads * self.value_dim)
            delta = self.update(torch.cat([tok, O], dim=-1)).squeeze(-1)
            alpha = alpha + self.step * delta
            if return_layers:
                alpha_layers.append(alpha)
                z_layers.append(torch.einsum("bmk,bm->bk", G, alpha))
                ent = -(Att.clamp_min(1e-12) * Att.clamp_min(1e-12).log()).sum(-1).mean()
                entropies.append(ent.detach())
                # average heads score correlation with linear dual kernel
                smean = scores.mean(dim=1)
                a = (smean - smean.mean(dim=(-1, -2), keepdim=True)).flatten(1)
                c = (Kdual - Kdual.mean(dim=(-1, -2), keepdim=True)).flatten(1)
                corr = (a * c).sum(-1) / (a.norm(dim=-1) * c.norm(dim=-1)).clamp_min(1e-12)
                score_corrs.append(corr.mean().detach())
        z = torch.einsum("bmk,bm->bk", G, alpha)
        info = {}
        if return_layers:
            info["alpha_layers"] = torch.stack(alpha_layers, dim=1)
            info["z_layers"] = torch.stack(z_layers, dim=1)
            info["attn_entropy"] = torch.stack(entropies)
            info["score_corr_linear_kernel"] = torch.stack(score_corrs)
        return z, alpha, info


class ParametricOperatorICL(nn.Module):
    """Full structured ICL model.

    Input:
        prompt (f_i,u_i), query f_*

    Learned/shared:
        A0_model, A_basis_model, probes
        optional recurrent dual attention z-solver

    Output:
        u_hat_*
    """
    def __init__(
        self,
        d: int,
        K: int,
        R: int,
        lam_z: float,
        gamma_u: float,
        solver: str = "exact",  # exact | primal_jacobi | dual_diag | dual_attention
        z_depth: int = 8,
        learn_dictionary: bool = True,
        learn_probes: bool = False,
        true_family: Optional[OperatorFamily] = None,
        init: str = "identity_random",  # identity_random | true | true_noisy
        init_noise: float = 0.05,
        dual_d_model: int = 128,
        dual_heads: int = 1,
        dual_d_head: int = 64,
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
            A0_init = true_family.A0.clone()
            Ab_init = true_family.Abasis.clone()
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

        if solver == "dual_attention":
            self.dual_solver = DualAttentionZSolver(
                K=K,
                d_model=dual_d_model,
                n_heads=dual_heads,
                d_head=dual_d_head,
                depth=z_depth,
            )
        else:
            self.dual_solver = None

    def weak_system(self, f_prompt: Tensor, u_prompt: Tensor) -> Tuple[Tensor, Tensor]:
        return build_weak_system(self.A0, self.Abasis, self.probes, f_prompt, u_prompt)

    def solve_z(self, G: Tensor, b: Tensor) -> Tuple[Tensor, Dict]:
        info = {}
        if self.solver == "exact":
            z = solve_z_exact(G, b, self.lam_z)
        elif self.solver == "primal_jacobi":
            z = solve_z_primal_richardson(G, b, self.lam_z, self.z_depth, mode="jacobi")
        elif self.solver == "primal_scalar":
            z = solve_z_primal_richardson(G, b, self.lam_z, self.z_depth, mode="scalar")
        elif self.solver == "dual_diag":
            z, alpha = solve_z_dual_diag(G, b, self.lam_z, self.z_depth)
            info["alpha"] = alpha
        elif self.solver == "dual_attention":
            z, alpha, dinfo = self.dual_solver(G, b, return_layers=True)
            info.update(dinfo)
            info["alpha"] = alpha
        else:
            raise ValueError(self.solver)
        return z, info

    def forward(self, f_prompt: Tensor, u_prompt: Tensor, f_star: Tensor, return_info: bool = False):
        G, b = self.weak_system(f_prompt, u_prompt)
        z_hat, info = self.solve_z(G, b)
        A_hat = self.A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z_hat, self.Abasis)
        u_hat = ridge_forward_solve(A_hat, f_star, self.gamma_u)
        if return_info:
            info.update({"G": G, "b": b, "z_hat": z_hat, "A_hat": A_hat})
            return u_hat, info
        return u_hat, {}


# -----------------------------------------------------------------------------
# metrics and experiment runners
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
    z_primal = solve_z_primal_richardson(G, b, args.lam_z, args.z_depth, mode="jacobi")
    z_dual, _ = solve_z_dual_diag(G, b, args.lam_z, args.z_depth)

    def pred_from_z(z):
        Ahat = family.A0.unsqueeze(0) + torch.einsum("bk,kij->bij", z, family.Abasis)
        return ridge_forward_solve(Ahat, batch.f_star, args.gamma_u), Ahat

    u_exact, A_exact = pred_from_z(z_exact)
    u_primal, A_primal = pred_from_z(z_primal)
    u_dual, A_dual = pred_from_z(z_dual)
    row = {
        "tag": tag,
        "d": args.d,
        "K": args.K,
        "m": args.m,
        "R": args.R,
        "M": args.m * args.R,
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
    csv_path: Optional[Path] = None,
    step: int = 0,
    tag: str = "eval",
) -> Dict:
    batch = sample_icl_batch(true_family, args.eval_batch_size, args.m, args.z_scale, args.f_std, args.noise_std, device)
    uhat, info = model(batch.f_prompt, batch.u_prompt, batch.f_star, return_info=True)
    zhat = info["z_hat"]
    Ahat = info["A_hat"]
    row = {
        "tag": tag,
        "step": step,
        "solver": model.solver,
        "d": args.d,
        "K": args.K,
        "m": args.m,
        "R": args.R,
        "M": args.m * args.R,
        "u_mse": mse(uhat, batch.u_star),
        "u_rel": relerr(uhat, batch.u_star),
        "z_mse_true_basis": mse(zhat, batch.z),
        "A_rel": frob_rel(Ahat, batch.A),
        "A0_rel": (model.A0 - true_family.A0).norm().item() / true_family.A0.norm().clamp_min(1e-12).item(),
        "Abasis_rel_raw": (model.Abasis - true_family.Abasis).flatten().norm().item() / true_family.Abasis.flatten().norm().clamp_min(1e-12).item(),
    }
    if "attn_entropy" in info:
        row["attn_entropy_last"] = float(info["attn_entropy"][-1].detach().cpu())
    if "score_corr_linear_kernel" in info:
        row["score_corr_linear_kernel_last"] = float(info["score_corr_linear_kernel"][-1].detach().cpu())
    if csv_path is not None:
        append_csv(csv_path, row)
    return row


def train(args, device) -> None:
    outdir = ensure_dir(args.outdir)
    train_csv = outdir / "train_metrics.csv"
    eval_csv = outdir / "eval_metrics.csv"

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
        dual_d_model=args.d_model,
        dual_heads=args.heads,
        dual_d_head=args.d_head,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hard = oracle_hardcoded_eval(true_family, args, device, outdir / "hardcoded_before_training.csv", tag="oracle_before")
    print("ORACLE BEFORE TRAINING", json.dumps(hard, indent=2))

    for step in range(1, args.steps + 1):
        if args.sample_z_scale_loguniform:
            z_scale = sample_log_uniform(args.z_scale_min, args.z_scale_max)
        else:
            z_scale = args.z_scale
        batch = sample_icl_batch(true_family, args.batch_size, args.m, z_scale, args.f_std, args.noise_std, device)
        uhat, info = model(batch.f_prompt, batch.u_prompt, batch.f_star, return_info=True)
        zhat = info["z_hat"]
        Ahat = info["A_hat"]

        loss_u = F.mse_loss(uhat, batch.u_star)
        loss_prompt = torch.tensor(0.0, device=device)
        if args.loss_prompt_weight > 0:
            # Make reconstructed operator explain the prompt as well.
            pred_f = torch.einsum("bij,bmj->bmi", Ahat, batch.u_prompt)
            loss_prompt = F.mse_loss(pred_f, batch.f_prompt)
        loss_z = torch.tensor(0.0, device=device)
        if args.loss_z_weight > 0:
            # This is only valid if the learned basis is initialized/fixed near true basis.
            loss_z = F.mse_loss(zhat, batch.z)
        loss_A = torch.tensor(0.0, device=device)
        if args.loss_A_weight > 0:
            loss_A = F.mse_loss(Ahat, batch.A)
        loss = loss_u + args.loss_prompt_weight * loss_prompt + args.loss_z_weight * loss_z + args.loss_A_weight * loss_A

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": loss.item(),
                "loss_u": loss_u.item(),
                "loss_prompt": loss_prompt.item(),
                "loss_z": loss_z.item(),
                "loss_A": loss_A.item(),
                "z_scale": z_scale,
            }
            append_csv(train_csv, row)
            erow = model_eval(model, true_family, args, device, eval_csv, step=step, tag="eval")
            print("TRAIN", json.dumps(row, indent=2))
            print("EVAL", json.dumps(erow, indent=2))

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
    p.add_argument("--outdir", type=str, default=str(resolve_outdir("runs_pure_icl_parametric_operator")))
    p.add_argument("--seed", type=int, default=0)

    # problem sizes
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--R", type=int, default=32)
    p.add_argument("--d-grid", type=str, default="16,32,64")
    p.add_argument("--K-grid", type=str, default="4,8,16")
    p.add_argument("--m-grid", type=str, default="4,8,16,32")

    # data generation
    p.add_argument("--basis-scale", type=float, default=0.25)
    p.add_argument("--A0-scale", type=float, default=2.0)
    p.add_argument("--z-scale", type=float, default=0.5)
    p.add_argument("--z-scale-min", type=float, default=0.1)
    p.add_argument("--z-scale-max", type=float, default=1.0)
    p.add_argument("--sample-z-scale-loguniform", type=int, default=0)
    p.add_argument("--f-std", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.0)

    # solvers
    p.add_argument("--solver", type=str, default="exact",
                   choices=["exact", "primal_jacobi", "primal_scalar", "dual_diag", "dual_attention"])
    p.add_argument("--lam-z", type=float, default=1e-3)
    p.add_argument("--gamma-u", type=float, default=1e-5)
    p.add_argument("--z-depth", type=int, default=12)

    # model architecture
    p.add_argument("--learn-dictionary", type=int, default=1)
    p.add_argument("--learn-probes", type=int, default=0)
    p.add_argument("--init", type=str, default="identity_random", choices=["identity_random", "true", "true_noisy"])
    p.add_argument("--init-noise", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--d-head", type=int, default=64)

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
    p.add_argument("--loss-z-weight", type=float, default=0.0)
    p.add_argument("--loss-A-weight", type=float, default=0.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.outdir = str(resolve_outdir(args.outdir))
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    ensure_dir(args.outdir)
    print("device:", device)

    if args.mode == "smoke":
        # hardcoded check
        family = make_true_family(args.d, args.K, args.basis_scale, args.A0_scale, device)
        row = oracle_hardcoded_eval(family, args, device, ensure_dir(args.outdir) / "smoke_hardcoded.csv")
        print("SMOKE HARDCODED", json.dumps(row, indent=2))

        # tiny train with true initialization so we know gradients run
        args.steps = min(args.steps, 50)
        args.eval_batch_size = min(args.eval_batch_size, 64)
        args.batch_size = min(args.batch_size, 64)
        args.init = "true_noisy"
        args.solver = "exact"
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
